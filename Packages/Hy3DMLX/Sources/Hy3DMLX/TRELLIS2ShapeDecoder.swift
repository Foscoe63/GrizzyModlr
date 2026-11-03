import Foundation
import MLX
import MLXNN

/// TRELLIS.2 shape decoder trunk (FlexiDualGridVaeDecoder without the mesher): from_latent → 4 stages of
/// ConvNeXt blocks, each followed by a channel→spatial block that predicts which of a voxel's 8 children exist
/// → layer norm → 7-channel output (3 dual-vertex offsets, 3 edge-intersection logits, 1 quad weight).
/// Mirrors microsoft/TRELLIS.2 `sparse_unet_vae.py`; checkpoint keys are used unchanged.
final class TRELLIS2ShapeDecoder {
    struct Config {
        var channels = [1024, 512, 256, 128, 64]
        var blocks = [4, 16, 8, 4, 0]
        var latentChannels = 32
    }

    struct Output {
        var coords: MLXArray      // [M, 4] int32 at the final lattice
        var feats: MLXArray       // [M, 7]
        var stageCounts: [Int]
        var subdivLogits: [MLXArray]
    }

    let cfg: Config
    private let w: [String: MLXArray]

    /// `weights` are the raw safetensors tensors (fp16 ok); they are held as float32.
    init(weights: [String: MLXArray], config: Config = Config()) {
        cfg = config
        w = weights.mapValues { $0.asType(.float32) }
    }

    private func linear(_ x: MLXArray, _ p: String) -> MLXArray {
        Layers.linear(x, w[p + ".weight"]!, w[p + ".bias"]!)
    }

    private func conv(_ x: SparseVoxels, _ p: String, _ nbr: MLXArray) -> MLXArray {
        submanifoldConv3d(x, weight: w[p + ".weight"]!, bias: w[p + ".bias"]!, neighbors: nbr)
    }

    private func convNeXt(_ x: SparseVoxels, _ p: String, _ nbr: MLXArray) -> SparseVoxels {
        var h = conv(x, p + ".conv", nbr)
        h = Layers.layerNorm(h, w[p + ".norm.weight"]!, w[p + ".norm.bias"]!, eps: 1e-6)
        h = silu(linear(h, p + ".mlp.0"))
        h = linear(h, p + ".mlp.2")
        var out = x
        out.feats = h + x.feats
        return out
    }

    /// SparseResBlockC2S3d with a predicted subdivision. Returns the finer voxels and the subdivision logits.
    private func channelToSpatialBlock(_ x: SparseVoxels, _ p: String, cin: Int, cout: Int,
                                       nbr: MLXArray) -> (SparseVoxels, MLXArray) {
        let sub = linear(x.feats, p + ".to_subdiv")                  // [N, 8]
        var h = silu(Layers.layerNorm(x.feats, w[p + ".norm1.weight"]!, w[p + ".norm1.bias"]!, eps: 1e-6))
        var hv = x; hv.feats = h
        h = conv(hv, p + ".conv1", nbr)                               // [N, cout*8]

        // children kept where the logit is positive; row-major (parent, slot) order like torch.nonzero
        let keep = (sub .> 0).asArray(Bool.self)
        var ridx: [Int32] = [], sidx: [Int32] = []
        for (i, k) in keep.enumerated() where k { ridx.append(Int32(i / 8)); sidx.append(Int32(i % 8)) }
        let r = MLXArray(ridx), s = MLXArray(sidx)
        let c = x.coords.asType(Int32.self)[r]
        let newCoords = stacked([c[0..., 0],
                                 c[0..., 1] * 2 + (s % 2),
                                 c[0..., 2] * 2 + ((s / 2) % 2),
                                 c[0..., 3] * 2 + ((s / 4) % 2)], axis: 1)
        let flatIdx = r * 8 + s
        var hn = h.reshaped(h.dim(0) * 8, -1)[flatIdx]                // [M, cout]
        let xn = x.feats.reshaped(x.feats.dim(0) * 8, -1)[flatIdx]    // [M, cin/8]
        hn = silu(Layers.layerNorm(hn, eps: 1e-6))
        let fine = SparseVoxels(feats: hn, coords: newCoords, size: x.size * 2, batch: x.batch)
        let nbr2 = submanifoldNeighbors(coords: newCoords, size: fine.size, batch: x.batch)
        hn = conv(fine, p + ".conv2", nbr2)
        // skip: repeat_interleave of the gathered input features along channels
        let rep = cout / (cin / 8)
        let skip = repeated(xn, count: rep, axis: 1)
        var out = fine
        out.feats = hn + skip
        return (out, sub)
    }

    func decode(coords: MLXArray, latent: MLXArray, size: Int = 16, batch: Int = 1) -> Output {
        var x = SparseVoxels(feats: linear(latent.asType(.float32), "from_latent"),
                             coords: coords.asType(Int32.self), size: size, batch: batch)
        var counts: [Int] = [], subs: [MLXArray] = []
        for i in 0 ..< cfg.blocks.count {
            var nbr = submanifoldNeighbors(coords: x.coords, size: x.size, batch: batch)
            for j in 0 ..< cfg.blocks[i] { x = convNeXt(x, "blocks.\(i).\(j)", nbr) }
            if i < cfg.blocks.count - 1 {
                let (y, sub) = channelToSpatialBlock(x, "blocks.\(i).\(cfg.blocks[i])",
                                                      cin: cfg.channels[i], cout: cfg.channels[i + 1], nbr: nbr)
                x = y; subs.append(sub)
                nbr = MLXArray.zeros([0])
            }
            eval(x.feats)
            counts.append(x.count)
        }
        let h = Layers.layerNorm(x.feats, eps: 1e-5)
        return Output(coords: x.coords, feats: linear(h, "output_layer"), stageCounts: counts, subdivLogits: subs)
    }
}
