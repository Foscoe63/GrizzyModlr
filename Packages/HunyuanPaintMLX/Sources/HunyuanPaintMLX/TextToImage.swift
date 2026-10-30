import Foundation
import MLX
import MLXNN

// Text-to-image with Stable Diffusion 2.1 (base, 512²): CLIP BPE tokenizer, OpenCLIP-H text tower,
// SD2.1 UNet, DDIM sampling with classifier-free guidance, VAE decode. Reuses the paint package's
// UNet / VAE primitives and the Zero123++ text tower (same architecture).

// MARK: - tokenizer

/// OpenAI CLIP byte-pair-encoding tokenizer (what `CLIPTokenizer` does), from `vocab.json` +
/// `merges.txt`.
public final class CLIPTokenizerBPE {
    private let vocab: [String: Int32]
    private let ranks: [String: Int]
    private let byteEncoder: [UInt8: Character]
    private var cache: [String: [String]] = [:]
    private let pattern: NSRegularExpression
    public static let bos: Int32 = 49406, eos: Int32 = 49407, pad: Int32 = 0

    public init(directory: URL) throws {
        let vocabData = try Data(contentsOf: directory.appendingPathComponent("vocab.json"))
        guard let v = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] else {
            throw NSError(domain: "CLIPTokenizer", code: 1, userInfo: [NSLocalizedDescriptionKey: "bad vocab.json"])
        }
        vocab = v.mapValues { Int32($0) }
        let merges = try String(contentsOf: directory.appendingPathComponent("merges.txt"), encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: false)
        // HF CLIPTokenizer keeps merges[1 : 49152 - 256 - 2 + 1]
        var r: [String: Int] = [:]
        for (i, line) in merges[1 ..< min(merges.count, 49152 - 256 - 2 + 1)].enumerated() { r[String(line)] = i }
        ranks = r
        // GPT-2 byte → printable unicode map
        var bs = Array(UInt8(ascii: "!") ... UInt8(ascii: "~")) + Array(UInt8(0xA1) ... UInt8(0xAC)) + Array(UInt8(0xAE) ... UInt8(0xFF))
        var cs = bs.map { UInt32($0) }
        var n: UInt32 = 0
        for b in 0 ... 255 where !bs.contains(UInt8(b)) { bs.append(UInt8(b)); cs.append(256 + n); n += 1 }
        var enc: [UInt8: Character] = [:]
        for (b, c) in zip(bs, cs) { enc[b] = Character(UnicodeScalar(c)!) }
        byteEncoder = enc
        pattern = try NSRegularExpression(
            pattern: #"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#,
            options: [.caseInsensitive])
    }

    private func bpe(_ token: String) -> [String] {
        if let c = cache[token] { return c }
        var word = token.map { String($0) }
        guard !word.isEmpty else { return [] }
        word[word.count - 1] += "</w>"
        while word.count > 1 {
            var best: (rank: Int, index: Int)?
            for i in 0 ..< word.count - 1 {
                if let r = ranks["\(word[i]) \(word[i + 1])"], best == nil || r < best!.rank { best = (r, i) }
            }
            guard let b = best else { break }
            let (first, second) = (word[b.index], word[b.index + 1])
            var merged: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1, word[i] == first, word[i + 1] == second { merged.append(first + second); i += 2 }
                else { merged.append(word[i]); i += 1 }
            }
            word = merged
        }
        cache[token] = word
        return word
    }

    /// `<|startoftext|> tokens… <|endoftext|>` then pad (`!`) up to `length`, truncating if needed.
    ///
    /// SD2's tokenizer config declares `!` as its pad token, and HF splits registered special
    /// tokens off before BPE, so a literal `!` in a prompt becomes the pad id (0) rather than
    /// the `!</w>` merge. That quirk is reproduced so prompts tokenize exactly like diffusers.
    public func encode(_ text: String, length: Int = 77) -> [Int32] {
        var ids: [Int32] = [Self.bos]
        let chunks = text.components(separatedBy: "!")
        for (n, chunk) in chunks.enumerated() {
            ids.append(contentsOf: encodeChunk(chunk))
            if n < chunks.count - 1 { ids.append(Self.pad) }
        }
        if ids.count > length - 1 { ids = Array(ids.prefix(length - 1)) }
        ids.append(Self.eos)
        while ids.count < length { ids.append(Self.pad) }
        return ids
    }

    private func encodeChunk(_ text: String) -> [Int32] {
        var clean = text.replacingOccurrences(of: "&amp;", with: "&")
        clean = clean.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ").lowercased()
        var ids: [Int32] = []
        let ns = clean as NSString
        for m in pattern.matches(in: clean, range: NSRange(location: 0, length: ns.length)) {
            let tok = ns.substring(with: m.range)
            if tok == "<|startoftext|>" { ids.append(Self.bos); continue }
            if tok == "<|endoftext|>" { ids.append(Self.eos); continue }
            let mapped = String(tok.utf8.map { byteEncoder[$0]! })
            for piece in bpe(mapped) { if let id = vocab[piece] { ids.append(id) } }
        }
        return ids
    }
}

// MARK: - scheduler

/// diffusers DDIMScheduler for epsilon-prediction SD checkpoints (scaled-linear betas, leading
/// spacing with steps_offset 1, `set_alpha_to_one = false`, eta 0).
public final class DDIMEpsScheduler {
    public let timesteps: [Int]
    let acp: [Double]
    let ratio: Int
    let finalAlpha: Double

    public init(steps: Int, numTrain: Int = 1000, betaStart: Double = 0.00085, betaEnd: Double = 0.012, stepsOffset: Int = 1) {
        let a = betaStart.squareRoot(), b = betaEnd.squareRoot()
        var prod = 1.0
        var acp = [Double]()
        for i in 0 ..< numTrain {
            let beta = pow(a + (b - a) * Double(i) / Double(numTrain - 1), 2)
            prod *= 1 - beta; acp.append(prod)
        }
        self.acp = acp
        ratio = numTrain / steps
        finalAlpha = acp[0]
        timesteps = (0 ..< steps).map { $0 * (numTrain / steps) + stepsOffset }.reversed()
    }

    /// Forward-diffuses a clean latent to timestep `t` with the given noise.
    public func noised(_ z0: MLXArray, noise: MLXArray, at t: Int) -> MLXArray {
        let a = acp[t]
        return Float(a.squareRoot()) * z0 + Float((1 - a).squareRoot()) * noise
    }

    /// Cumulative alpha the sample has after stepping from `t` (what `step` lands on).
    func previousTimestep(_ t: Int) -> Int { t - ratio }

    public func step(_ eps: MLXArray, _ t: Int, _ sample: MLXArray) -> MLXArray {
        let aT = acp[t]
        let prev = t - ratio
        let aPrev = prev >= 0 ? acp[prev] : finalAlpha
        let x0 = (sample - Float((1 - aT).squareRoot()) * eps) / Float(aT.squareRoot())
        return Float(aPrev.squareRoot()) * x0 + Float((1 - aPrev).squareRoot()) * eps
    }
}

// MARK: - pipeline

public final class TextToImagePipeline {
    let unetW: W, vaeW: W
    let text: Z123Text
    let tokenizer: CLIPTokenizerBPE
    let dtype: DType

    /// `dir` holds `tokenizer/{vocab.json,merges.txt}` and the fp16 `unet/`, `vae/`, `text_encoder/` files.
    public init(dir: URL, dtype: DType = .float16) throws {
        self.dtype = dtype
        tokenizer = try CLIPTokenizerBPE(directory: dir.appendingPathComponent("tokenizer"))
        unetW = try Z123Weights.load(dir.appendingPathComponent("unet/diffusion_pytorch_model.fp16.safetensors"), dtype: dtype)
        vaeW = try Z123Weights.load(dir.appendingPathComponent("vae/diffusion_pytorch_model.fp16.safetensors"), dtype: dtype, vae: true)
        text = Z123Text(try Z123Weights.load(dir.appendingPathComponent("text_encoder/model.fp16.safetensors"), dtype: dtype))
    }

    public func embed(_ prompt: String) -> MLXArray { text.hidden(tokenizer.encode(prompt)) }

    /// Returns scaled latents `[1,H/8,W/8,4]` (before VAE decode).
    public func sample(prompt: String, negative: String = "", steps: Int = 30, guidance: Float = 7.5,
                       width: Int = 512, height: Int = 512, initialNoise: MLXArray,
                       isCancelled: () -> Bool = { false }, onProgress: ((Int, Int) -> Void)? = nil) -> MLXArray? {
        let cond = embed(prompt), uncond = embed(negative)
        let ctx = concatenated([uncond, cond], axis: 0)                    // [uncond, cond]
        eval(ctx)
        let sched = DDIMEpsScheduler(steps: steps)
        let unet = PaintUNet(unetW)
        var lat = initialNoise
        for (i, t) in sched.timesteps.enumerated() {
            if isCancelled() { return nil }
            let input = concatenated([lat, lat], axis: 0).asType(dtype)
            let out = unet(input, MLXArray([Float(t), Float(t)]), ctx)
            let parts = split(out.asType(.float32), parts: 2, axis: 0)
            let eps = parts[0] + guidance * (parts[1] - parts[0])
            lat = sched.step(eps, t, lat)
            eval(lat)
            onProgress?(i + 1, steps)
        }
        return lat
    }

    /// Masked inpainting with the base (non-inpainting) checkpoint, by blended latent diffusion:
    /// every step the unmasked latents are replaced by the source image's latents diffused to the
    /// same noise level, so only the masked region is invented. `image` is `[1,H,W,3]` in -1…1 and
    /// `mask` is `[1,H/8,W/8,1]` (1 = regenerate). `strength` 0…1 sets how much of the schedule runs.
    /// Returns latents; the caller composites the decoded result into the masked area only.
    public func inpaint(image: MLXArray, mask: MLXArray, prompt: String, negative: String = "",
                        steps: Int = 25, strength: Float = 0.85, guidance: Float = 7.5, seed: UInt64,
                        isCancelled: () -> Bool = { false },
                        onProgress: ((Int, Int) -> Void)? = nil) -> MLXArray? {
        let z0 = PaintVAE(vaeW).encodeMean(image.asType(dtype)).asType(.float32) * 0.18215
        let noise = MLXRandom.normal(z0.shape, key: MLXRandom.key(seed))
        let cond = embed(prompt), uncond = embed(negative)
        let ctx = concatenated([uncond, cond], axis: 0)
        eval(z0, noise, ctx)
        let sched = DDIMEpsScheduler(steps: steps)
        let skip = min(steps - 1, max(0, Int((Float(steps) * (1 - strength)).rounded())))
        let timesteps = Array(sched.timesteps.dropFirst(skip))
        guard let first = timesteps.first else { return z0 }
        let unet = PaintUNet(unetW)
        var lat = sched.noised(z0, noise: noise, at: first)
        for (i, t) in timesteps.enumerated() {
            if isCancelled() { return nil }
            let input = concatenated([lat, lat], axis: 0).asType(dtype)
            let out = unet(input, MLXArray([Float(t), Float(t)]), ctx)
            let parts = split(out.asType(.float32), parts: 2, axis: 0)
            let eps = parts[0] + guidance * (parts[1] - parts[0])
            var next = sched.step(eps, t, lat)
            let prev = sched.previousTimestep(t)
            let known = prev >= 0 ? sched.noised(z0, noise: noise, at: prev) : z0
            next = mask * next + (1 - mask) * known
            lat = next
            eval(lat)
            onProgress?(i + 1, timesteps.count)
        }
        return lat
    }

    /// Latents → RGB `[H,W,3]` in 0…1.
    public func decode(_ latents: MLXArray) -> MLXArray {
        let img = PaintVAE(vaeW).decode((latents / 0.18215).asType(dtype))
        return clip((img.asType(.float32) * 0.5 + 0.5), min: 0, max: 1)[0]
    }

    public static func randomNoise(seed: UInt64, width: Int = 512, height: Int = 512) -> MLXArray {
        MLXRandom.normal([1, height / 8, width / 8, 4], key: MLXRandom.key(seed))
    }
}
