import Foundation
import MLX
import MLXFast
import MLXNN

/// TripoSR (stabilityai/TripoSR, MIT): single image → triplane NeRF → density field → mesh.
/// DINO ViT-B/16 image tokens, a 16-layer cross-attending transformer over 3×32×32 triplane
/// tokens, a transposed-conv upsampler (→ 3×64×64×40), and a 10-layer MLP decoder.
public final class TripoSR {
    let w: W
    static let radius: Float = 0.87
    public static let imageSize = 512

    /// Position embedding of the 32×32 token grid (bicubic-resized from the checkpoint's 14×14).
    private let positionEmbedding: MLXArray

    public init(checkpoint url: URL) throws {
        var weights: [String: MLXArray] = [:]
        for (name, t) in try TorchCheckpoint.load(url) {
            guard t.dtype == "float32" else { continue }
            let data = t.bytes()
            var a = data.withUnsafeBytes { MLXArray($0.bindMemory(to: Float.self), t.shape) }
            if a.ndim == 4, name != "post_processor.upsample.weight", name != "tokenizer.embeddings" {
                a = a.transposed(0, 2, 3, 1)                              // OIHW → OHWI
            }
            weights[name] = a
        }
        w = W(weights)
        positionEmbedding = Self.resizedPositions(weights["image_tokenizer.model.embeddings.position_embeddings"]!)
        eval(positionEmbedding)
    }

    // MARK: preprocessing

    /// TripoSR's input prep from straight RGBA8: crop to the foreground box, pad it square and out
    /// to `1 / foregroundRatio` of its size, composite over mid-gray, resize to 512² (antialiased
    /// bilinear). Returns `[1,512,512,3]` in 0…1, or nil when the image has no foreground.
    public static func preprocess(rgba: [UInt8], width: Int, height: Int, foregroundRatio: Float = 0.85) -> MLXArray? {
        guard rgba.count == width * height * 4 else { return nil }
        var y1 = Int.max, y2 = -1, x1 = Int.max, x2 = -1
        for y in 0 ..< height { for x in 0 ..< width where rgba[(y * width + x) * 4 + 3] > 0 {
            y1 = min(y1, y); y2 = max(y2, y); x1 = min(x1, x); x2 = max(x2, x)
        } }
        guard y2 >= 0 else { return nil }
        // reference crop is image[y1:y2, x1:x2] (exclusive of the last row/column)
        let fgH = max(y2 - y1, 1), fgW = max(x2 - x1, 1)
        let size = max(fgH, fgW)
        let newSize = Int(Float(size) / foregroundRatio)
        let top = (newSize - size) / 2 + (size - fgH) / 2, left = (newSize - size) / 2 + (size - fgW) / 2
        var out = [Float](repeating: 0.5, count: newSize * newSize * 3)
        for y in 0 ..< fgH { for x in 0 ..< fgW {
            let s = ((y1 + y) * width + x1 + x) * 4
            let a = Float(rgba[s + 3]) / 255
            let o = ((top + y) * newSize + left + x) * 3
            for c in 0 ..< 3 { out[o + c] = Float(rgba[s + c]) / 255 * a + (1 - a) * 0.5 }
        } }
        let resized = Z123Preprocess.resize(.init(pixels: out, width: newSize, height: newSize),
                                            to: (imageSize, imageSize), bicubic: false)
        return MLXArray(resized.pixels, [1, imageSize, imageSize, 3])
    }

    // MARK: image tokens (DINO ViT-B/16 with interpolated position embeddings)

    /// HF `ViTEmbeddings.interpolate_pos_encoding` for a 512² input: CLS embedding kept, the 14×14
    /// patch grid resized to 32×32 with bicubic (A = −0.75) using the `(32 + 0.1) / 14` scale factor.
    static func resizedPositions(_ pos: MLXArray) -> MLXArray {
        let dim = pos.dim(2), inN = 14, outN = imageSize / 16
        let cls = pos[0..., 0 ..< 1]
        let host = pos[0, 1...].asArray(Float.self)                                   // [196 * dim]
        let scale = 1.0 / (Double(outN) + 0.1) * Double(inN)                          // input px per output px
        func weights(_ o: Int) -> (idx: [Int], w: [Float]) {
            let src = (Double(o) + 0.5) * scale - 0.5
            let f = Int(src.rounded(.down)), t = src - Double(f)
            func cubic(_ x: Double) -> Double {
                let a = -0.75, ax = abs(x)
                if ax <= 1 { return ((a + 2) * ax - (a + 3)) * ax * ax + 1 }
                if ax < 2 { return ((a * ax - 5 * a) * ax + 8 * a) * ax - 4 * a }
                return 0
            }
            let offs = [-1, 0, 1, 2]
            return (offs.map { min(max(f + $0, 0), inN - 1) }, offs.map { Float(cubic(t - Double($0))) })
        }
        let axis = (0 ..< outN).map(weights)
        var out = [Float](repeating: 0, count: outN * outN * dim)
        for oy in 0 ..< outN { for ox in 0 ..< outN {
            for c in 0 ..< dim {
                var acc: Float = 0
                for (jy, wy) in zip(axis[oy].idx, axis[oy].w) { for (jx, wx) in zip(axis[ox].idx, axis[ox].w) {
                    acc += host[(jy * inN + jx) * dim + c] * wy * wx
                } }
                out[(oy * outN + ox) * dim + c] = acc
            }
        } }
        return concatenated([cls, MLXArray(out, [1, outN * outN, dim])], axis: 1)
    }

    /// `rgb`: `[1,512,512,3]` in 0…1 → `[1,1025,768]`.
    public func imageTokens(_ rgb: MLXArray) -> MLXArray {
        let mean = MLXArray([0.485, 0.456, 0.406] as [Float]).reshaped([1, 1, 1, 3])
        let std = MLXArray([0.229, 0.224, 0.225] as [Float]).reshaped([1, 1, 1, 3])
        let p = "image_tokenizer.model"
        var x = L2D.conv((rgb - mean) / std, w, "\(p).embeddings.patch_embeddings.projection", stride: 16, padding: 0)
        x = x.reshaped([rgb.dim(0), -1, 768])
        x = concatenated([broadcast(w.a("\(p).embeddings.cls_token"), to: [rgb.dim(0), 1, 768]), x], axis: 1) + positionEmbedding
        for i in 0 ..< 12 { x = vitLayer(x, "\(p).encoder.layer.\(i)") }
        return MLXFast.layerNorm(x, weight: w.a("\(p).layernorm.weight"), bias: w.a("\(p).layernorm.bias"), eps: 1e-12)
    }

    private func vitLayer(_ x: MLXArray, _ p: String) -> MLXArray {
        let B = x.dim(0), N = x.dim(1), heads = 12, dh = 64
        let a = MLXFast.layerNorm(x, weight: w.a("\(p).layernorm_before.weight"), bias: w.a("\(p).layernorm_before.bias"), eps: 1e-12)
        func proj(_ n: String) -> MLXArray {
            L2D.linear(a, w, "\(p).attention.attention.\(n)").reshaped([B, N, heads, dh]).transposed(0, 2, 1, 3)
        }
        let o = MLXFast.scaledDotProductAttention(queries: proj("query"), keys: proj("key"), values: proj("value"),
                                                  scale: Float(pow(Double(dh), -0.5)), mask: .none)
        var h = x + L2D.linear(o.transposed(0, 2, 1, 3).reshaped([B, N, 768]), w, "\(p).attention.output.dense")
        let m = MLXFast.layerNorm(h, weight: w.a("\(p).layernorm_after.weight"), bias: w.a("\(p).layernorm_after.bias"), eps: 1e-12)
        h = h + L2D.linear(gelu(L2D.linear(m, w, "\(p).intermediate.dense")), w, "\(p).output.dense")
        return h
    }

    // MARK: triplane transformer

    private func block(_ x: MLXArray, _ ctx: MLXArray, _ p: String) -> MLXArray {
        func attn(_ q: MLXArray, _ kv: MLXArray, _ a: String) -> MLXArray {
            let B = q.dim(0), heads = 16, dh = 64
            func split(_ t: MLXArray) -> MLXArray { t.reshaped([B, -1, heads, dh]).transposed(0, 2, 1, 3) }
            let o = MLXFast.scaledDotProductAttention(queries: split(L2D.linear(q, w, "\(a).to_q")),
                                                      keys: split(L2D.linear(kv, w, "\(a).to_k")),
                                                      values: split(L2D.linear(kv, w, "\(a).to_v")),
                                                      scale: 0.125, mask: .none)
            return L2D.linear(o.transposed(0, 2, 1, 3).reshaped([B, -1, heads * dh]), w, "\(a).to_out.0")
        }
        func ln(_ t: MLXArray, _ n: String) -> MLXArray {
            MLXFast.layerNorm(t, weight: w.a("\(p).\(n).weight"), bias: w.a("\(p).\(n).bias"), eps: 1e-5)
        }
        var h = x
        let n1 = ln(h, "norm1"); h = h + attn(n1, n1, "\(p).attn1")
        h = h + attn(ln(h, "norm2"), ctx, "\(p).attn2")
        let n3 = ln(h, "norm3")
        let proj = L2D.linear(n3, w, "\(p).ff.net.0.proj")
        let parts = split(proj, parts: 2, axis: -1)
        return h + L2D.linear(parts[0] * gelu(parts[1]), w, "\(p).ff.net.2")
    }

    /// Image tokens → triplane features `[3,64,64,40]` (planes first, NHWC).
    public func triplanes(_ imageTokens: MLXArray) -> MLXArray {
        let emb = w.a("tokenizer.embeddings")                                       // [3,1024,32,32]
        var h = emb.transposed(0, 2, 3, 1).reshaped([1, 3 * 32 * 32, 1024])          // channels last
        let residual = h
        h = L2D.groupNorm(h.reshaped([1, 1, 3072, 1024]), w, "backbone.norm", groups: 32, eps: 1e-6).reshaped([1, 3072, 1024])
        h = L2D.linear(h, w, "backbone.proj_in")
        for i in 0 ..< 16 { h = block(h, imageTokens, "backbone.transformer_blocks.\(i)") }
        h = L2D.linear(h, w, "backbone.proj_out") + residual
        let planes = h.reshaped([3, 32, 32, 1024])                                   // B Np·Hp·Wp C → Np Hp Wp C
        // ConvTranspose2d(1024→40, k=2, s=2): per-pixel matmul + pixel shuffle
        let kernel = w.a("post_processor.upsample.weight")                           // [1024, 40, 2, 2]
        let mat = kernel.transposed(0, 2, 3, 1).reshaped([1024, 4 * 40])             // (i, j, out)
        var up = matmul(planes.reshaped([3 * 32 * 32, 1024]), mat)
        up = up.reshaped([3, 32, 32, 2, 2, 40]).transposed(0, 1, 3, 2, 4, 5).reshaped([3, 64, 64, 40])
        return up + w.a("post_processor.upsample.bias")
    }

    // MARK: NeRF field

    /// Bilinear (zero-padded, align_corners = false) lookup of each triplane at `uv` in −1…1, `[N,40]`.
    private func sample(_ planes: MLXArray, plane p: Int, _ u: MLXArray, _ v: MLXArray) -> MLXArray {
        let H = 64, W = 64
        let x = ((u + 1) * Float(W) - 1) / 2, y = ((v + 1) * Float(H) - 1) / 2
        let x0 = floor(x), y0 = floor(y)
        let wx1 = x - x0, wy1 = y - y0, wx0 = 1 - wx1, wy0 = 1 - wy1
        let flat = planes[p].reshaped([H * W, 40])
        var out = MLXArray.zeros([u.dim(0), 40])
        for (dx, wx) in [(Float(0), wx0), (1, wx1)] {
            for (dy, wy) in [(Float(0), wy0), (1, wy1)] {
                let xi = x0 + dx, yi = y0 + dy
                let valid = (xi .>= 0) .&& (xi .<= Float(W - 1)) .&& (yi .>= 0) .&& (yi .<= Float(H - 1))
                let idx = (clip(yi, min: 0, max: Float(H - 1)) * Float(W) + clip(xi, min: 0, max: Float(W - 1))).asType(.int32)
                let weight = wx * wy * valid.asType(.float32)
                out = out + take(flat, idx, axis: 0) * weight.reshaped([-1, 1])
            }
        }
        return out
    }

    /// Density (`exp(σ − 1)`) and RGB at world positions `[N,3]` (TripoSR frame: z up).
    public func query(_ planes: MLXArray, _ positions: MLXArray) -> (density: MLXArray, color: MLXArray) {
        let p = positions / Self.radius
        let (px, py, pz) = (p[0..., 0], p[0..., 1], p[0..., 2])
        let f = concatenated([sample(planes, plane: 0, px, py), sample(planes, plane: 1, px, pz),
                              sample(planes, plane: 2, py, pz)], axis: 1)             // [N,120]
        var h = f
        for i in 0 ..< 9 { h = L2D.silu(L2D.linear(h, w, "decoder.layers.\(i * 2)")) }
        let o = L2D.linear(h, w, "decoder.layers.18")
        return (exp(o[0..., 0] - 1), sigmoid(o[0..., 1 ... 3]))
    }

    /// Densities on an `n³` grid over (−radius, radius)³, flattened with x slowest (i, j, k).
    public func densityGrid(_ planes: MLXArray, resolution n: Int, chunk: Int = 1 << 20,
                            isCancelled: () -> Bool = { false }) -> [Float]? {
        let total = n * n * n
        var out = [Float](); out.reserveCapacity(total)
        let step = 2 * Self.radius / Float(n - 1)
        var start = 0
        while start < total {
            if isCancelled() { return nil }
            let count = min(chunk, total - start)
            let idx = MLXArray(Int32(start) ..< Int32(start + count))
            // integer index math (MLX's `/` is a true division, so use floorDivide)
            let i = floorDivide(idx, Int32(n * n)), j = floorDivide(idx, Int32(n)) % Int32(n), k = idx % Int32(n)
            let pos = stacked([i, j, k], axis: 1).asType(.float32) * step - Self.radius
            let d = query(planes, pos).density
            eval(d)
            out.append(contentsOf: d.asArray(Float.self))
            start += count
        }
        return out
    }
}
