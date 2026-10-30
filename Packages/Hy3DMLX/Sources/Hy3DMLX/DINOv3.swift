import Foundation
import MLX
import MLXFast

/// DINOv3 ViT feature extractor as TRELLIS.2 conditions on it: patch conv → [cls, registers, patches] → pre-norm
/// blocks with LayerScale and axial RoPE on the patch tokens only → plain (non-affine) layer norm over every token.
/// Mirrors HF `DINOv3ViTModel`. Depth, widths, register count and which biases exist are read from the checkpoint
/// tensors, so any size of the family loads; only `heads` must be given (ViT-L/16 has 16).
/// Not yet checked against the real, gated `facebook/dinov3-vitl16-pretrain-lvd1689m` weights.
final class DINOv3FeatureExtractor {
    private let w: [String: MLXArray]
    let heads: Int
    let layers: Int
    let hidden: Int
    let patch: Int
    private let ropeTheta: Float

    init(weights: [String: MLXArray], heads: Int, ropeTheta: Float = 100) {
        var m: [String: MLXArray] = [:]
        for (k, v) in weights {
            let key = k.hasPrefix("model.") ? String(k.dropFirst(6)) : k
            m[key] = v.asType(.float32)
        }
        // conv weight [O, I, kh, kw] -> MLX [O, kh, kw, I]
        m["embeddings.patch_embeddings.weight"] = m["embeddings.patch_embeddings.weight"]!.transposed(0, 2, 3, 1)
        w = m
        self.heads = heads
        self.ropeTheta = ropeTheta
        hidden = m["embeddings.cls_token"]!.dim(2)
        patch = m["embeddings.patch_embeddings.weight"]!.dim(1)
        var n = 0
        while m["layer.\(n).norm1.weight"] != nil { n += 1 }
        layers = n
    }

    private func lin(_ x: MLXArray, _ p: String) -> MLXArray {
        Layers.linear(x, w[p + ".weight"]!, w[p + ".bias"])
    }

    private func gelu(_ x: MLXArray) -> MLXArray {
        0.5 * x * (1 + erf(x / Float(2).squareRoot()))
    }

    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        let d = x.dim(-1)
        return concatenated([-x[.ellipsis, (d / 2)...], x[.ellipsis, ..<(d / 2)]], axis: -1)
    }

    /// - Parameter image: [1, H, W, 3] already resized and ImageNet-normalised, H and W multiples of the patch size.
    /// - Returns: [1, 1 + registers + (H/p)(W/p), hidden].
    func callAsFunction(_ image: MLXArray) -> MLXArray {
        let ph = image.dim(1) / patch, pw = image.dim(2) / patch
        let numPatches = ph * pw
        let hd = hidden / heads

        var pe = conv2d(image.asType(.float32), w["embeddings.patch_embeddings.weight"]!, stride: .init(patch))
        pe = pe + w["embeddings.patch_embeddings.bias"]!
        var h = concatenated([w["embeddings.cls_token"]!, w["embeddings.register_tokens"]!,
                              pe.reshaped(1, numPatches, hidden)], axis: 1)
        let numTokens = h.dim(1), npre = numTokens - numPatches

        // axial RoPE angles for patch centres in [-1, 1]
        let cy = (MLXArray(0 ..< Int32(ph)).asType(.float32) + 0.5) / Float(ph)
        let cx = (MLXArray(0 ..< Int32(pw)).asType(.float32) + 0.5) / Float(pw)
        let yy = broadcast(cy.reshaped(ph, 1), to: [ph, pw]).reshaped(-1)
        let xx = broadcast(cx.reshaped(1, pw), to: [ph, pw]).reshaped(-1)
        let coords = stacked([2 * yy - 1, 2 * xx - 1], axis: 1)                       // [N, 2]
        let nf = hd / 4
        let inv = 1 / pow(MLXArray(ropeTheta), MLXArray(0 ..< Int32(nf)).asType(.float32) * (4 / Float(hd)))
        let a = (2 * Float.pi * coords.reshaped(numPatches, 2, 1) * inv.reshaped(1, 1, nf)).reshaped(numPatches, 2 * nf)
        let ang = concatenated([a, a], axis: 1)                                        // tile(2) -> [N, hd]
        let cs = cos(ang), sn = sin(ang)

        func rope(_ t: MLXArray) -> MLXArray {                                         // t [1, H, T, hd]
            let pre = t[0..., 0..., ..<npre, 0...], pat = t[0..., 0..., npre..., 0...]
            return concatenated([pre, pat * cs + rotateHalf(pat) * sn], axis: 2)
        }

        for i in 0 ..< layers {
            let p = "layer.\(i)"
            var y = Layers.layerNorm(h, w[p + ".norm1.weight"]!, w[p + ".norm1.bias"]!, eps: 1e-5)
            func split(_ t: MLXArray) -> MLXArray { t.reshaped(1, numTokens, heads, hd).transposed(0, 2, 1, 3) }
            let q = rope(split(lin(y, p + ".attention.q_proj")))
            let k = rope(split(lin(y, p + ".attention.k_proj")))
            let v = split(lin(y, p + ".attention.v_proj"))
            let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / Float(hd).squareRoot(), mask: nil)
                .transposed(0, 2, 1, 3).reshaped(1, numTokens, hidden)
            h = h + lin(o, p + ".attention.o_proj") * w[p + ".layer_scale1.lambda1"]!
            y = Layers.layerNorm(h, w[p + ".norm2.weight"]!, w[p + ".norm2.bias"]!, eps: 1e-5)
            y = lin(gelu(lin(y, p + ".mlp.up_proj")), p + ".mlp.down_proj")
            h = h + y * w[p + ".layer_scale2.lambda1"]!
            if i % 6 == 5 { eval(h) }
        }
        return Layers.layerNorm(h, eps: 1e-5)
    }
}
