import Foundation
import MLX
import MLXNN

/// TRELLIS.2 sparse-structure decoder: dense 3D conv U-Net, 16³×8 latent → 64³ occupancy logits.
/// Mirrors `SparseStructureDecoder` (norm_type "layer", channels [512, 128, 32], 2 res blocks per level).
/// Activations are channels-last (NDHWC); weights are converted from torch's [Co, Ci, kd, kh, kw] on load.
final class TRELLIS2SparseStructureDecoder {
    private let w: [String: MLXArray]

    init(weights: [String: MLXArray]) {
        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            let f = v.asType(.float32)
            out[k] = (k.hasSuffix(".weight") && f.ndim == 5) ? f.transposed(0, 2, 3, 4, 1) : f
        }
        w = out
    }

    private func conv(_ x: MLXArray, _ p: String) -> MLXArray {
        let k = w[p + ".weight"]!
        let pad = k.dim(1) / 2
        return conv3d(x, k, stride: 1, padding: .init(pad)) + w[p + ".bias"]!
    }

    private func norm(_ x: MLXArray, _ p: String) -> MLXArray {
        Layers.layerNorm(x, w[p + ".weight"]!, w[p + ".bias"]!, eps: 1e-5)
    }

    private func res(_ x: MLXArray, _ p: String) -> MLXArray {
        var h = conv(silu(norm(x, p + ".norm1")), p + ".conv1")
        h = conv(silu(norm(h, p + ".norm2")), p + ".conv2")
        return h + x
    }

    /// 3D pixel shuffle matching `pixel_shuffle_3d`: channel c*8 + (a*4 + b*2 + d) → voxel offset (a, b, d).
    private func shuffle(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), ww = x.dim(2), d = x.dim(3), c = x.dim(4) / 8
        return x.reshaped(b, h, ww, d, c, 2, 2, 2)
            .transposed(0, 1, 5, 2, 6, 3, 7, 4)
            .reshaped(b, h * 2, ww * 2, d * 2, c)
    }

    /// - Parameter latent: [B, 16, 16, 16, 8] (channels last). Returns logits [B, 64, 64, 64, 1].
    func decode(_ latent: MLXArray) -> MLXArray {
        var h = conv(latent.asType(.float32), "input_layer")
        for i in 0 ..< 2 { h = res(h, "middle_block.\(i)") }
        for i in 0 ..< 8 {
            if i == 2 || i == 5 { h = shuffle(conv(h, "blocks.\(i).conv")) } else { h = res(h, "blocks.\(i)") }
            if i % 3 == 2 { eval(h) }
        }
        h = silu(Layers.layerNorm(h, w["out_layer.0.weight"]!, w["out_layer.0.bias"]!, eps: 1e-5))
        return conv(h, "out_layer.2")
    }

    /// Occupied voxel coordinates (batch, x, y, z) of a logits volume, optionally max-pooled down to `resolution`.
    static func occupiedCoords(_ logits: MLXArray, resolution: Int? = nil) -> MLXArray {
        let side = logits.dim(1)
        var occ = (logits[0..., 0..., 0..., 0..., 0] .> 0)
        if let r = resolution, r != side {
            let f = side / r
            occ = occ.reshaped(occ.dim(0), r, f, r, f, r, f).max(axes: [2, 4, 6])
        }
        let flat = occ.asArray(Bool.self)
        let n = occ.dim(1)
        var out: [Int32] = []
        for (i, on) in flat.enumerated() where on {
            let z = i % n, y = (i / n) % n, x = (i / (n * n)) % n, b = i / (n * n * n)
            out += [Int32(b), Int32(x), Int32(y), Int32(z)]
        }
        return MLXArray(out, [out.count / 4, 4])
    }
}
