import Foundation
import MLX
import MLXFast
import MLXNN

/// Depth Anything V2 (Small): DINOv2-S backbone + DPT neck/head → relative inverse depth.
/// Port of the transformers `DepthAnythingForDepthEstimation` for a fixed square input
/// (518², a 37×37 patch grid, so no positional-embedding interpolation is needed).
public struct DepthAnythingSmall {
    let w: W
    static let hidden = 384, heads = 6, layers = 12, patch = 14
    static let stages = [3, 6, 9, 12]
    static let neck = [48, 96, 192, 384]
    static let factors: [Float] = [4, 2, 1, 0.5]
    public static let inputSize = 518

    public init(weights: [String: MLXArray]) {
        var out: [String: MLXArray] = [:]
        for (k, v) in weights { out[k] = v.ndim == 4 ? v.transposed(0, 2, 3, 1) : v }   // OIHW → OHWI
        w = W(out)
    }

    public static func load(_ url: URL, dtype: DType = .float32) throws -> DepthAnythingSmall {
        DepthAnythingSmall(weights: try loadArrays(url: url).mapValues {
            ($0.dtype == .float32 || $0.dtype == .float16) ? $0.asType(dtype) : $0
        })
    }

    /// `pixels`: NHWC `[1,518,518,3]`, ImageNet-normalized. Returns depth `[518,518]` (higher = nearer).
    public func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let feats = backbone(pixels)
        let gridH = pixels.dim(1) / Self.patch, gridW = pixels.dim(2) / Self.patch

        // neck: reassemble → 3×3 conv to the fusion width → top-down fusion
        var maps: [MLXArray] = []
        for i in 0 ..< 4 {
            let tokens = feats[i][0..., 1...]                                       // drop CLS
            var x = tokens.reshaped([tokens.dim(0), gridH, gridW, Self.hidden])
            x = L2D.conv(x, w, "neck.reassemble_stage.layers.\(i).projection", padding: 0)
            let f = Self.factors[i]
            if f > 1 { x = convTransposeBlock(x, "neck.reassemble_stage.layers.\(i).resize", Int(f)) }
            else if f < 1 { x = L2D.conv(x, w, "neck.reassemble_stage.layers.\(i).resize", stride: Int(1 / f), padding: 1) }
            maps.append(L2D.conv(x, w, "neck.convs.\(i)", padding: 1))
        }
        let rev = Array(maps.reversed())
        var fused = fusionLayer(0, rev[0], residual: nil, size: (rev[1].dim(1), rev[1].dim(2)))
        for idx in 0 ..< 3 {
            let size: (Int, Int)? = idx != 2 ? (rev[idx + 2].dim(1), rev[idx + 2].dim(2)) : nil
            fused = fusionLayer(idx + 1, fused, residual: rev[idx + 1], size: size)
        }

        // head
        var d = L2D.conv(fused, w, "head.conv1", padding: 1)
        d = Self.bilinear(d, gridH * Self.patch, gridW * Self.patch, alignCorners: true)
        d = relu(L2D.conv(d, w, "head.conv2", padding: 1))
        d = relu(L2D.conv(d, w, "head.conv3", padding: 0))
        return d[0, 0..., 0..., 0]
    }

    // MARK: backbone

    private func backbone(_ pixels: MLXArray) -> [MLXArray] {
        let B = pixels.dim(0)
        var x = L2D.conv(pixels, w, "backbone.embeddings.patch_embeddings.projection", stride: Self.patch, padding: 0)
        x = x.reshaped([B, -1, Self.hidden])
        let cls = broadcast(w.a("backbone.embeddings.cls_token"), to: [B, 1, Self.hidden])
        x = concatenated([cls, x], axis: 1) + w.a("backbone.embeddings.position_embeddings")
        var out: [MLXArray] = []
        for i in 0 ..< Self.layers {
            x = layer(x, i)
            if Self.stages.contains(i + 1) {
                out.append(MLXFast.layerNorm(x, weight: w.a("backbone.layernorm.weight"),
                                             bias: w.a("backbone.layernorm.bias"), eps: 1e-6))
            }
        }
        return out
    }

    private func layer(_ x: MLXArray, _ i: Int) -> MLXArray {
        let p = "backbone.encoder.layer.\(i)"
        let B = x.dim(0), N = x.dim(1), dh = Self.hidden / Self.heads
        let a = MLXFast.layerNorm(x, weight: w.a("\(p).norm1.weight"), bias: w.a("\(p).norm1.bias"), eps: 1e-6)
        func proj(_ n: String) -> MLXArray {
            L2D.linear(a, w, "\(p).attention.attention.\(n)").reshaped([B, N, Self.heads, dh]).transposed(0, 2, 1, 3)
        }
        let o = MLXFast.scaledDotProductAttention(queries: proj("query"), keys: proj("key"), values: proj("value"),
                                                  scale: Float(pow(Double(dh), -0.5)), mask: .none)
        var h = x + L2D.linear(o.transposed(0, 2, 1, 3).reshaped([B, N, Self.hidden]), w, "\(p).attention.output.dense")
            * w.a("\(p).layer_scale1.lambda1")
        let m = MLXFast.layerNorm(h, weight: w.a("\(p).norm2.weight"), bias: w.a("\(p).norm2.bias"), eps: 1e-6)
        h = h + L2D.linear(gelu(L2D.linear(m, w, "\(p).mlp.fc1")), w, "\(p).mlp.fc2") * w.a("\(p).layer_scale2.lambda1")
        return h
    }

    // MARK: neck / head pieces

    /// ConvTranspose2d with kernel == stride (no overlap): a per-pixel matmul plus a pixel shuffle.
    private func convTransposeBlock(_ x: MLXArray, _ p: String, _ f: Int) -> MLXArray {
        let weight = w.a("\(p).weight")                                              // [in, f, f, out]
        let cin = weight.dim(0), cout = weight.dim(3)
        let B = x.dim(0), H = x.dim(1), Wd = x.dim(2)
        var y = matmul(x.reshaped([B * H * Wd, cin]), weight.reshaped([cin, f * f * cout]))
        y = y.reshaped([B, H, Wd, f, f, cout]).transposed(0, 1, 3, 2, 4, 5).reshaped([B, H * f, Wd * f, cout])
        return y + w.a("\(p).bias")
    }

    private func residualUnit(_ x: MLXArray, _ p: String) -> MLXArray {
        var h = L2D.conv(relu(x), w, "\(p).convolution1", padding: 1)
        h = L2D.conv(relu(h), w, "\(p).convolution2", padding: 1)
        return h + x
    }

    private func fusionLayer(_ i: Int, _ hidden: MLXArray, residual: MLXArray?, size: (Int, Int)?) -> MLXArray {
        let p = "neck.fusion_stage.layers.\(i)"
        var h = hidden
        if var r = residual {
            if r.dim(1) != h.dim(1) || r.dim(2) != h.dim(2) { r = Self.bilinear(r, h.dim(1), h.dim(2), alignCorners: false) }
            h = h + residualUnit(r, "\(p).residual_layer1")
        }
        h = residualUnit(h, "\(p).residual_layer2")
        let (th, tw) = size ?? (h.dim(1) * 2, h.dim(2) * 2)
        h = Self.bilinear(h, th, tw, alignCorners: true)
        return L2D.conv(h, w, "\(p).projection", padding: 0)
    }

    /// Bilinear resize of an NHWC tensor matching `torch.nn.functional.interpolate(mode="bilinear")`.
    static func bilinear(_ x: MLXArray, _ outH: Int, _ outW: Int, alignCorners: Bool) -> MLXArray {
        func axis(_ inN: Int, _ outN: Int) -> (MLXArray, MLXArray, MLXArray) {
            var i0 = [Int32](), i1 = [Int32](), wt = [Float]()
            for o in 0 ..< outN {
                let src: Double = alignCorners
                    ? (outN > 1 ? Double(o) * Double(inN - 1) / Double(outN - 1) : 0)
                    : max((Double(o) + 0.5) * Double(inN) / Double(outN) - 0.5, 0)
                let lo = min(Int(src.rounded(.down)), inN - 1)
                i0.append(Int32(lo)); i1.append(Int32(min(lo + 1, inN - 1))); wt.append(Float(src - Double(lo)))
            }
            return (MLXArray(i0), MLXArray(i1), MLXArray(wt))
        }
        let (y0, y1, wy) = axis(x.dim(1), outH)
        let rows = take(x, y0, axis: 1) * (1 - wy).reshaped([1, outH, 1, 1]) + take(x, y1, axis: 1) * wy.reshaped([1, outH, 1, 1])
        let (x0, x1, wx) = axis(x.dim(2), outW)
        return take(rows, x0, axis: 2) * (1 - wx).reshaped([1, 1, outW, 1]) + take(rows, x1, axis: 2) * wx.reshaped([1, 1, outW, 1])
    }
}
