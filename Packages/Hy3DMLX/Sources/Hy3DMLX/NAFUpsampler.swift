import Foundation
import MLX
import MLXNN

/// NAF feature upsampler (valeoai/NAF, Apache-2.0) as used by Pixal3D's shape stages: RGB image → two conv
/// encoders → pooled to the output size → axial RoPE → 4-head 9×9 dilated neighbourhood cross-attention that
/// filters the low-resolution DINOv3 features up to the output size. NATTEN's window rule (9 positions on the
/// stride-d sub-grid, centred on the query's cell, shifted to stay inside) is reproduced with explicit gathers.
/// Weights are the converted `naf_release.pth` (torch key names, conv weights OIHW).
final class NAFUpsampler {
    private let w: [String: MLXArray]
    private let kernel = 9, heads = 4, dim = 256

    init(weights: [String: MLXArray]) {
        w = weights.mapValues { $0.ndim == 4 ? $0.transposed(0, 2, 3, 1) : $0 }   // OIHW -> OHWI
    }

    private func reflectPad(_ x: MLXArray, _ p: Int) -> MLXArray {              // x [1,H,W,C]
        if p == 0 { return x }
        let h = x.dim(1), wd = x.dim(2)
        let ri = MLXArray((0 ..< (h + 2 * p)).map { i -> Int32 in
            var j = i - p; if j < 0 { j = -j }; if j > h - 1 { j = 2 * (h - 1) - j }; return Int32(j) })
        let ci = MLXArray((0 ..< (wd + 2 * p)).map { i -> Int32 in
            var j = i - p; if j < 0 { j = -j }; if j > wd - 1 { j = 2 * (wd - 1) - j }; return Int32(j) })
        return x.take(ri, axis: 1).take(ci, axis: 2)
    }

    private func conv(_ x: MLXArray, _ p: String, _ k: Int) -> MLXArray {
        conv2d(reflectPad(x, k / 2), w[p + ".weight"]!) + w[p + ".bias"]!
    }

    private func groupNorm(_ x: MLXArray, _ p: String) -> MLXArray {
        let (h, wd, c) = (x.dim(1), x.dim(2), x.dim(3))
        let g = x.reshaped(1, h * wd, 8, c / 8)
        let mean = g.mean(axes: [1, 3], keepDims: true)
        let v = ((g - mean) * (g - mean)).mean(axes: [1, 3], keepDims: true)
        return ((g - mean) / sqrt(v + 1e-5)).reshaped(1, h, wd, c) * w[p + ".weight"]! + w[p + ".bias"]!
    }

    private func encoder(_ x: MLXArray, _ p: String, _ k: Int) -> MLXArray {
        var h = conv(x, p + ".0", k)
        for i in 1 ... 2 {
            let b = "\(p).\(i)"
            var y = conv(silu(groupNorm(h, b + ".norm1")), b + ".conv1", k)
            y = conv(silu(groupNorm(y, b + ".norm2")), b + ".conv2", k)
            h = y
        }
        return h
    }

    private func avgPool(_ x: MLXArray, to size: (Int, Int)) -> MLXArray {      // exact-divisor case, x [1,H,W,C]
        let (h, wd, c) = (x.dim(1), x.dim(2), x.dim(3))
        if (h, wd) == size { return x }
        precondition(h % size.0 == 0 && wd % size.1 == 0, "NAF pooling needs divisible sizes")
        return x.reshaped(1, size.0, h / size.0, size.1, wd / size.1, c).mean(axes: [2, 4])
    }

    private func rope(_ x: MLXArray) -> MLXArray {                              // [1,H,W,256]
        let (h, wd) = (x.dim(1), x.dim(2)), dh = dim / heads
        let periods = w["image_encoder.rope.periods"]!
        let ch = (MLXArray(0 ..< Int32(h)).asType(.float32) + 0.5) / Float(h)
        let cw = (MLXArray(0 ..< Int32(wd)).asType(.float32) + 0.5) / Float(wd)
        let cy = 2 * broadcast(ch.reshaped(h, 1), to: [h, wd]).reshaped(-1) - 1
        let cx = 2 * broadcast(cw.reshaped(1, wd), to: [h, wd]).reshaped(-1) - 1
        let coords = stacked([cy, cx], axis: 1)                                  // [HW,2]
        let ang = (2 * Float.pi * coords.reshaped(h * wd, 2, 1) / periods.reshaped(1, 1, -1)).reshaped(h * wd, dh / 2)
        let a = concatenated([ang, ang], axis: 1)                                // tile(2) -> [HW, dh]
        let cs = cos(a).reshaped(1, h * wd, 1, dh), sn = sin(a).reshaped(1, h * wd, 1, dh)
        let xh = x.reshaped(1, h * wd, heads, dh)
        let x1 = xh[0..., 0..., 0..., ..<(dh / 2)], x2 = xh[0..., 0..., 0..., (dh / 2)...]
        let r = xh * cs + concatenated([-x2, x1], axis: -1) * sn
        return r.reshaped(1, h, wd, dim)
    }

    /// - image: [1,H,W,3] in [0,1]. features: [1,hk,wk,C] (C divisible by 4). Returns [1,hq,wq,C].
    func callAsFunction(image: MLXArray, features: MLXArray, outputSize: (Int, Int)) -> MLXArray {
        let (hq, wq) = outputSize, hk = features.dim(1), wk = features.dim(2), c = features.dim(3)
        let dy = hq / hk, dx = wq / wk, dh = dim / heads, k = kernel
        var x = concatenated([encoder(image, "image_encoder.encoder", 1),
                              encoder(image, "image_encoder.sem_encoder", 3)], axis: -1)
        x = rope(avgPool(x, to: outputSize))
        let keys = avgPool(x, to: (hk, wk))

        // window start per low-res cell, shifted to stay inside the map
        func starts(_ n: Int) -> MLXArray {
            MLXArray((0 ..< n).map { Int32(min(max($0 - k / 2, 0), n - k)) })
        }
        let iy = starts(hk).reshaped(hk, 1) + MLXArray(0 ..< Int32(k)).reshaped(1, k)
        let ix = starts(wk).reshaped(wk, 1) + MLXArray(0 ..< Int32(k)).reshaped(1, k)
        func windows(_ m: MLXArray, _ ch: Int) -> MLXArray {                     // m [hk,wk,ch] -> [cells,81,ch]
            m.take(iy, axis: 0).take(ix, axis: 2)                                // [hk,k,wk,k,ch]
                .transposed(0, 2, 1, 3, 4).reshaped(hk * wk, k * k, ch)
        }
        let kn = windows(keys[0], dim).reshaped(hk * wk, k * k, heads, dh)
        let vn = windows(features[0], c).reshaped(hk * wk, k * k, heads, c / heads)
        let q = x[0].reshaped(hk, dy, wk, dx, heads, dh).transposed(0, 2, 1, 3, 4, 5)
            .reshaped(hk * wk, dy * dx, heads, dh)
        let scores = einsum("cqhd,cshd->chqs", q, kn) * (1 / Float(dh).squareRoot())
        let attn = softmax(scores, axis: -1)
        let o = einsum("chqs,csht->cqht", attn, vn)                              // [cells, dy*dx, heads, C/heads]
        return o.reshaped(hk, wk, dy, dx, c).transposed(0, 2, 1, 3, 4).reshaped(1, hq, wq, c)
    }
}
