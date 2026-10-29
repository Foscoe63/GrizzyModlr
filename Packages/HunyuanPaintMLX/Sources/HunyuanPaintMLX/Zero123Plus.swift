import Foundation
import MLX
import MLXFast
import MLXNN

// Zero123++ v1.2 (sudo-ai/zero123plus-v1.2): single image → a 3×2 grid of six novel views.
// SD2.1 UNet with reference-only self-attention, v-prediction, Euler-ancestral sampling, and a
// CLIP image embedding blended into the (empty) text conditioning. Functional port that reuses
// the paint package's SD2.1 UNet / VAE primitives.

// MARK: - weights

public enum Z123Weights {
    /// Loads a diffusers safetensors file into a `W`: 4-D conv kernels become MLX layout
    /// `[O,kH,kW,I]`, and floating tensors are cast to `dtype`.
    public static func load(_ url: URL, dtype: DType, vae: Bool = false) throws -> W {
        var out: [String: MLXArray] = [:]
        for (key, value) in try loadArrays(url: url) {
            var v = value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value
            if v.dtype == .float16 || v.dtype == .float32 || v.dtype == .bfloat16 { v = v.asType(dtype) }
            // diffusers VAE attention nests the output projection as `to_out.0`; the shared
            // `L2D.vaeAttn` expects `to_out`.
            out[vae ? key.replacingOccurrences(of: ".to_out.0.", with: ".to_out.") : key] = v
        }
        return W(out)
    }
}

// MARK: - CLIP

/// Pre-LN CLIP transformer block pieces shared by the vision and text towers.
enum CLIPBlocks {
    static func layer(_ x: MLXArray, _ w: W, _ p: String, heads: Int, causal: Bool) -> MLXArray {
        let B = x.dim(0), N = x.dim(1), C = x.dim(2), dh = C / heads
        let a = MLXFast.layerNorm(x, weight: w.a("\(p).layer_norm1.weight"), bias: w.a("\(p).layer_norm1.bias"), eps: 1e-5)
        func proj(_ n: String) -> MLXArray {
            L2D.linear(a, w, "\(p).self_attn.\(n)").reshaped([B, N, heads, dh]).transposed(0, 2, 1, 3)
        }
        let o = MLXFast.scaledDotProductAttention(queries: proj("q_proj"), keys: proj("k_proj"), values: proj("v_proj"),
                                                  scale: Float(pow(Double(dh), -0.5)),
                                                  mask: causal ? .causal : .none)
        var h = x + L2D.linear(o.transposed(0, 2, 1, 3).reshaped([B, N, C]), w, "\(p).self_attn.out_proj")
        let m = MLXFast.layerNorm(h, weight: w.a("\(p).layer_norm2.weight"), bias: w.a("\(p).layer_norm2.bias"), eps: 1e-5)
        h = h + L2D.linear(gelu(L2D.linear(m, w, "\(p).mlp.fc1")), w, "\(p).mlp.fc2")
        return h
    }
}

/// CLIPVisionModelWithProjection (OpenCLIP ViT-H/14): NHWC `[B,224,224,3]` → `[B,1024]` image embeds.
public struct Z123Vision {
    let w: W
    let layers = 32, heads = 16
    public init(_ w: W) { self.w = w }

    public func imageEmbeds(_ pixels: MLXArray) -> MLXArray {
        let B = pixels.dim(0)
        var x = L2D.conv(pixels, w, "vision_model.embeddings.patch_embedding", stride: 14, padding: 0)
        x = x.reshaped([B, -1, x.dim(3)])
        let cls = broadcast(w.a("vision_model.embeddings.class_embedding").reshaped([1, 1, -1]), to: [B, 1, x.dim(2)])
        x = concatenated([cls, x], axis: 1) + w.a("vision_model.embeddings.position_embedding.weight")
        x = MLXFast.layerNorm(x, weight: w.a("vision_model.pre_layrnorm.weight"), bias: w.a("vision_model.pre_layrnorm.bias"), eps: 1e-5)
        for i in 0 ..< layers { x = CLIPBlocks.layer(x, w, "vision_model.encoder.layers.\(i)", heads: heads, causal: false) }
        let pooled = MLXFast.layerNorm(x[0..., 0], weight: w.a("vision_model.post_layernorm.weight"),
                                       bias: w.a("vision_model.post_layernorm.bias"), eps: 1e-5)
        return matmul(pooled, w.a("visual_projection.weight").transposed(1, 0))
    }
}

/// CLIPTextModel (OpenCLIP ViT-H text tower, 23 layers) → last hidden state `[B,77,1024]`.
public struct Z123Text {
    let w: W
    let layers = 23, heads = 16
    public init(_ w: W) { self.w = w }

    /// Token ids of the empty prompt: `<|startoftext|> <|endoftext|>` then the pad token `!` (id 0).
    public static let emptyPromptIDs: [Int32] = [49406, 49407] + [Int32](repeating: 0, count: 75)

    public func hidden(_ ids: [Int32]) -> MLXArray {
        var x = take(w.a("text_model.embeddings.token_embedding.weight"), MLXArray(ids).reshaped([1, -1]), axis: 0)
        x = x + w.a("text_model.embeddings.position_embedding.weight")[0 ..< ids.count]
        for i in 0 ..< layers { x = CLIPBlocks.layer(x, w, "text_model.encoder.layers.\(i)", heads: heads, causal: true) }
        return MLXFast.layerNorm(x, weight: w.a("text_model.final_layer_norm.weight"),
                                 bias: w.a("text_model.final_layer_norm.bias"), eps: 1e-5)
    }
}

// MARK: - scheduler

/// diffusers EulerAncestralDiscreteScheduler (linear betas, v-prediction, linspace spacing).
public final class EulerAncestralScheduler {
    public let timesteps: [Float]
    public let sigmas: [Float]            // count = steps + 1 (last is 0)
    public var initNoiseSigma: Float { sigmas.max() ?? 1 }

    /// `trailing` = diffusers `timestep_spacing: trailing` (what the Zero123++ authors recommend);
    /// the checkpoint's own config says `linspace`.
    public init(steps: Int, trailing: Bool = false, numTrain: Int = 1000,
                betaStart: Double = 0.00085, betaEnd: Double = 0.012) {
        var acp = [Double](), prod = 1.0
        for i in 0 ..< numTrain {
            let beta = betaStart + (betaEnd - betaStart) * Double(i) / Double(numTrain - 1)
            prod *= 1 - beta; acp.append(prod)
        }
        let full = acp.map { (((1 - $0) / $0)).squareRoot() }
        // linspace(0, numTrain-1, steps) reversed, sigmas linearly interpolated at those timesteps
        let ts: [Double]
        if trailing {
            let ratio = Double(numTrain) / Double(steps)
            ts = (0 ..< steps).map { (Double(numTrain) - Double($0) * ratio).rounded() - 1 }
        } else {
            ts = (0 ..< steps).map { Double($0) * Double(numTrain - 1) / Double(max(steps - 1, 1)) }.reversed()
        }
        timesteps = ts.map { Float($0) }
        var sg = ts.map { t -> Float in
            let lo = Int(t.rounded(.down)), hi = min(lo + 1, numTrain - 1), f = t - Double(lo)
            return Float(full[lo] * (1 - f) + full[hi] * f)
        }
        sg.append(0)
        sigmas = sg
    }

    public func scaleModelInput(_ x: MLXArray, _ i: Int) -> MLXArray {
        x / Float((Double(sigmas[i]) * Double(sigmas[i]) + 1).squareRoot())
    }

    public func addNoise(_ original: MLXArray, _ noise: MLXArray, _ i: Int) -> MLXArray {
        original + noise * sigmas[i]
    }

    /// One ancestral step. `noise` is a standard normal tensor shaped like `sample`.
    public func step(_ modelOutput: MLXArray, _ i: Int, _ sample: MLXArray, noise: MLXArray) -> MLXArray {
        let sFrom = Double(sigmas[i]), sTo = Double(sigmas[i + 1])
        // v-prediction → predicted x0
        let x0 = modelOutput * Float(-sFrom / (sFrom * sFrom + 1).squareRoot()) + sample / Float(sFrom * sFrom + 1)
        let up = (sTo * sTo * (sFrom * sFrom - sTo * sTo) / (sFrom * sFrom)).squareRoot()
        let down = (sTo * sTo - up * up).squareRoot()
        let derivative = (sample - x0) / Float(sFrom)
        return sample + derivative * Float(down - sFrom) + noise * Float(up)
    }
}

// MARK: - pipeline

public final class Zero123PlusPipeline {
    public let unetW: W, vaeW: W
    let vision: Z123Vision
    /// Empty-prompt text embedding `[1,77,1024]` (the pipeline always uses prompt "").
    let textEmbed: MLXArray
    let dtype: DType

    /// Ramping coefficients (model_index.json): one weight per text token for the image embed.
    static let ramping: [Float] = [
        0.00301829120144248, 0.2204633206129074, 0.21527841687202454, 0.23498539626598358, 0.1914631873369217,
        0.20188239216804504, 0.19352824985980988, 0.17249998450279236, 0.15826298296451569, 0.15236389636993408,
        0.13444548845291138, 0.12044154852628708, 0.12808501720428467, 0.1271015852689743, 0.13629068434238434,
        0.14516159892082214, 0.15645112097263336, 0.16885493695735931, 0.18022602796554565, 0.1958882212638855,
        0.21415705978870392, 0.23056700825691223, 0.2505834102630615, 0.2574525773525238, 0.275470107793808,
        0.2808215022087097, 0.29953837394714355, 0.2967497408390045, 0.2883710563182831, 0.3023308515548706,
        0.3054688572883606, 0.32596179842948914, 0.3225354254245758, 0.3140765428543091, 0.3288663625717163,
        0.3435625731945038, 0.3342442810535431, 0.32937031984329224, 0.35734811425209045, 0.3601177930831909,
        0.3517529368400574, 0.3810708224773407, 0.40007662773132324, 0.4264647364616394, 0.3977527916431427,
        0.4314143657684326, 0.49558719992637634, 0.4665665030479431, 0.48960328102111816, 0.5141982436180115,
        0.5230164527893066, 0.5266074538230896, 0.5456079840660095, 0.5737904906272888, 0.5882097482681274,
        0.6210350394248962, 0.6530380845069885, 0.6383244395256042, 0.6792004704475403, 0.6567418575286865,
        0.7517656683921814, 0.736494243144989, 0.7586457133293152, 0.8130561709403992, 0.9578766226768494,
        1.001284122467041, 0.9404520988464355, 1.004292368888855, 0.9145274758338928, 0.9771682620048523,
        1.0350638628005981, 1.0265849828720093, 1.0594775676727295, 0.980824887752533, 1.0715670585632324,
        1.0140161514282227, 1.1983819007873535,
    ]

    /// `dir` holds the diffusers layout (`unet/`, `vae/`, `vision_encoder/`, `text_encoder/`).
    public init(dir: URL, dtype: DType = .float16) throws {
        self.dtype = dtype
        unetW = try Z123Weights.load(dir.appendingPathComponent("unet/diffusion_pytorch_model.safetensors"), dtype: dtype)
        vaeW = try Z123Weights.load(dir.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"), dtype: dtype, vae: true)
        vision = Z123Vision(try Z123Weights.load(dir.appendingPathComponent("vision_encoder/model.safetensors"), dtype: dtype))
        let text = Z123Text(try Z123Weights.load(dir.appendingPathComponent("text_encoder/model.safetensors"), dtype: dtype))
        textEmbed = text.hidden(Z123Text.emptyPromptIDs)
        eval(textEmbed)
    }

    /// CLIP image embedding → text conditioning for the cond branch (`[1,77,1024]`).
    public func conditioning(clipPixels: MLXArray) -> (cond: MLXArray, uncond: MLXArray, imageEmbeds: MLXArray) {
        let g = vision.imageEmbeds(clipPixels.asType(dtype)).reshaped([1, 1, -1])      // [1,1,1024]
        let ramp = MLXArray(Self.ramping).reshaped([1, 77, 1]).asType(dtype)
        return (textEmbed + g * ramp, textEmbed, g.reshaped([1, -1]))
    }

    /// One CFG denoise pass: reference write (noised condition latent) then the main read pass.
    public func noisePrediction(latents: MLXArray, t: Float, i: Int, scheduler: EulerAncestralScheduler,
                         condLat: MLXArray, ehs: MLXArray, refNoise: MLXArray, guidance: Float) -> MLXArray {
        let tt = MLXArray([t, t])
        let unet = PaintUNet(unetW)
        // Reference pass: noise the (cond, uncond) condition latents like the sample, run mode "w".
        let noisy = scheduler.scaleModelInput(scheduler.addNoise(condLat, refNoise, i), i)
        let xa = XAttn(mode: "w", numInBatch: 1)
        xa.selfRef = true
        _ = unet(noisy, tt, ehs, xattn: xa)
        xa.mode = "r"
        let input = scheduler.scaleModelInput(concatenated([latents, latents], axis: 0), i)
        let out = unet(input, tt, ehs, xattn: xa)
        let parts = split(out, parts: 2, axis: 0)               // [uncond, cond]
        return parts[0] + guidance * (parts[1] - parts[0])
    }

    public struct Noise {
        /// Initial latent noise `[1,H/8,W/8,4]`, per-step reference noise `[2,64,64,4]` and step noise.
        public var initial: () -> MLXArray
        public var reference: (Int) -> MLXArray
        public var step: (Int) -> MLXArray
        public init(initial: @escaping () -> MLXArray, reference: @escaping (Int) -> MLXArray, step: @escaping (Int) -> MLXArray) {
            self.initial = initial; self.reference = reference; self.step = step
        }
        public static func random(seed: UInt64, height: Int, width: Int, condSize: Int) -> Noise {
            var key = MLXRandom.key(seed)
            func next() -> MLXArray { let (a, b) = MLXRandom.split(key: key); key = a; return b }
            return Noise(initial: { MLXRandom.normal([1, height / 8, width / 8, 4], key: next()) },
                         reference: { _ in MLXRandom.normal([2, condSize / 8, condSize / 8, 4], key: next()) },
                         step: { _ in MLXRandom.normal([1, height / 8, width / 8, 4], key: next()) })
        }
    }

    /// Runs the full sampler and returns scaled latents `[1,H/8,W/8,4]` (before unscaling/decoding).
    public func sample(vaePixels: MLXArray, clipPixels: MLXArray, steps: Int = 28, guidance: Float = 4,
                       height: Int = 960, width: Int = 640, trailingSpacing: Bool = true, noise: Noise,
                       isCancelled: () -> Bool = { false },
                       onProgress: ((Int, Int) -> Void)? = nil) -> MLXArray? {
        let vae = PaintVAE(vaeW)
        let cond = vae.encodeMean(vaePixels.asType(dtype))
        let uncondLat = vae.encodeMean(zeros(like: vaePixels).asType(dtype))
        let condLat = concatenated([uncondLat, cond], axis: 0)       // [negative, positive]
        let (c, u, _) = conditioning(clipPixels: clipPixels)
        let ehs = concatenated([u, c], axis: 0)
        let sched = EulerAncestralScheduler(steps: steps, trailing: trailingSpacing)
        var lat = noise.initial() * sched.initNoiseSigma
        for i in 0 ..< steps {
            if isCancelled() { return nil }
            let pred = noisePrediction(latents: lat.asType(dtype), t: sched.timesteps[i], i: i, scheduler: sched,
                                       condLat: condLat, ehs: ehs, refNoise: noise.reference(i), guidance: guidance)
            lat = sched.step(pred.asType(.float32), i, lat, noise: noise.step(i))
            eval(lat)
            onProgress?(i + 1, steps)
        }
        return lat
    }

    /// Scaled latents → RGB image `[H,W,3]` in 0…1.
    public func decode(_ latents: MLXArray) -> MLXArray {
        let unscaled = latents / 0.75 + 0.22                                   // unscale_latents
        let img = PaintVAE(vaeW).decode((unscaled / 0.18215).asType(dtype))    // [1,H,W,3] in [-1,1]
        let out = (img / 0.5 * 0.8) * 0.5 + 0.5                                // unscale_image, then → 0…1
        return clip(out.asType(.float32), min: 0, max: 1)[0]
    }
}
