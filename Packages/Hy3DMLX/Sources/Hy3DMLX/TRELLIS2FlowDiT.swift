import Foundation
import MLX
import MLXFast
import MLXNN

/// TRELLIS.2 flow transformer, used for both the sparse-structure model (dense 16³ grid, 8 channels) and the
/// shape SLat model (sparse voxels, 32 channels): adaLN-modulated blocks with shared modulation, RoPE + per-head
/// RMS-normalised self-attention over the token set, then cross-attention to the DINOv3 tokens and a GELU MLP.
/// With batch 1 the sparse model is exactly this dense computation over N tokens. Mirrors microsoft/TRELLIS.2
/// `sparse_structure_flow.py` / `structured_latent_flow.py`; checkpoint keys are used unchanged.
final class TRELLIS2FlowDiT {
    struct Config {
        var channels = 1536
        var blocks = 30
        var heads = 12
        var cond = 1024
    }

    let cfg: Config
    private let w: [String: MLXArray]
    private let dtype: DType
    private var headDim: Int { cfg.channels / cfg.heads }

    /// `dtype` is the compute/weight dtype (float32 for parity tests; bf16 for speed).
    init(weights: [String: MLXArray], config: Config = Config(), dtype: DType = .float32) {
        cfg = config
        self.dtype = dtype
        w = weights.mapValues { $0.asType(dtype) }
    }

    private func lin(_ x: MLXArray, _ p: String) -> MLXArray {
        Layers.linear(x, w[p + ".weight"]!, w[p + ".bias"]!)
    }

    private func ln(_ x: MLXArray, _ p: String? = nil) -> MLXArray {
        Layers.layerNorm(x, p.map { w[$0 + ".weight"]! }, p.map { w[$0 + ".bias"]! }, eps: 1e-6)
    }

    /// Rotary angles per token [N, headDim/2]: three axes × freqs, padded with zeros (identity) to the head size.
    static func ropeAngles(coords: MLXArray, headDim: Int) -> MLXArray {
        let freqDim = headDim / 2 / 3
        let f = MLXArray(0 ..< Int32(freqDim)).asType(.float32) / Float(freqDim)
        let freqs = 1 / pow(MLXArray(Float(10000)), f)
        let n = coords.dim(0)
        let ang = (coords.asType(.float32).reshaped(n, 3, 1) * freqs.reshaped(1, 1, freqDim)).reshaped(n, 3 * freqDim)
        let pad = headDim / 2 - 3 * freqDim
        return pad > 0 ? concatenated([ang, MLXArray.zeros([n, pad])], axis: 1) : ang
    }

    private func rope(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray) -> MLXArray {   // x [N,H,D] float32
        let p = x.reshaped(x.dim(0), x.dim(1), headDim / 2, 2)
        let re = p[0..., 0..., 0..., 0], im = p[0..., 0..., 0..., 1]
        return stacked([re * c - im * s, re * s + im * c], axis: -1).reshaped(x.shape)
    }

    private func rms(_ x: MLXArray, _ gamma: MLXArray) -> MLXArray {      // x [N,H,D], gamma [H,D]
        let xf = x.asType(.float32)
        let norm = sqrt((xf * xf).sum(axis: -1, keepDims: true))
        return xf / maximum(norm, MLXArray(Float(1e-12))) * gamma.asType(.float32) * Float(headDim).squareRoot()
    }

    private func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {   // [N,H,D] x [M,H,D] -> [N,H*D]
        let o = MLXFast.scaledDotProductAttention(
            queries: q.transposed(1, 0, 2).expandedDimensions(axis: 0).asType(dtype),
            keys: k.transposed(1, 0, 2).expandedDimensions(axis: 0).asType(dtype),
            values: v.transposed(1, 0, 2).expandedDimensions(axis: 0).asType(dtype),
            scale: 1 / Float(headDim).squareRoot(), mask: nil)
        return o[0].transposed(1, 0, 2).reshaped(q.dim(0), cfg.channels)
    }

    private func gelu(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let inner = Float(2 / Double.pi).squareRoot() * (xf + 0.044715 * xf * xf * xf)
        return (0.5 * xf * (1 + tanh(inner))).asType(x.dtype)
    }

    /// Sinusoidal timestep embedding → MLP → shared adaLN projection ([1, 6C]).
    func modulation(t: Float) -> MLXArray {
        let half = 128
        let fr = exp(-Float(log(10000.0)) * MLXArray(0 ..< Int32(half)).asType(.float32) / Float(half))
        let a = t * fr
        let emb = concatenated([cos(a), sin(a)], axis: 0).reshaped(1, 2 * half).asType(dtype)
        let te = lin(silu(lin(emb, "t_embedder.mlp.0")), "t_embedder.mlp.2")
        return lin(silu(te), "adaLN_modulation.1")
    }

    /// Context K/V depend only on the conditioning, so callers sampling many steps can reuse them; kept simple here.
    /// `proj` ([N, projIn], Pixal3D only): per-token view-aligned features; each block adds `proj_linear(proj)` to
    /// its cross-attention output. Pixal3D checkpoints nest the attention under `cross_attn.cross_attn_block`.
    func callAsFunction(x: MLXArray, coords: MLXArray, t: Float, cond: MLXArray, proj: MLXArray? = nil) -> MLXArray {
        let n = x.dim(0), c = cfg.channels, hs = cfg.heads, d = headDim
        let ang = TRELLIS2FlowDiT.ropeAngles(coords: coords, headDim: d)
        let cs = cos(ang).reshaped(n, 1, d / 2), sn = sin(ang).reshaped(n, 1, d / 2)
        let mod = modulation(t: t)
        let ctx = cond.asType(dtype)
        let projIn = proj?.asType(dtype)
        var h = lin(x.asType(dtype), "input_layer")
        for i in 0 ..< cfg.blocks {
            let p = "blocks.\(i)"
            let m = (w[p + ".modulation"]! + mod).reshaped(6, c)
            let (shA, scA, gA, shM, scM, gM) = (m[0], m[1], m[2], m[3], m[4], m[5])

            var y = ln(h) * (1 + scA) + shA
            let qkv = lin(y, p + ".self_attn.to_qkv").reshaped(n, 3, hs, d)
            var q = rms(qkv[0..., 0], w[p + ".self_attn.q_rms_norm.gamma"]!)
            var k = rms(qkv[0..., 1], w[p + ".self_attn.k_rms_norm.gamma"]!)
            q = rope(q, cos: cs, sin: sn); k = rope(k, cos: cs, sin: sn)
            var o = attend(q, k, qkv[0..., 2])
            h = h + lin(o, p + ".self_attn.to_out") * gA

            y = ln(h, p + ".norm2")
            let ca = projIn != nil ? p + ".cross_attn.cross_attn_block" : p + ".cross_attn"
            q = rms(lin(y, ca + ".to_q").reshaped(n, hs, d), w[ca + ".q_rms_norm.gamma"]!)
            let kv = lin(ctx, ca + ".to_kv").reshaped(ctx.dim(0), 2, hs, d)
            k = rms(kv[0..., 0], w[ca + ".k_rms_norm.gamma"]!)
            o = attend(q, k, kv[0..., 1])
            o = lin(o, ca + ".to_out")
            if let pr = projIn { o = o + lin(pr, p + ".cross_attn.proj_linear") }
            h = h + o

            y = ln(h) * (1 + scM) + shM
            y = lin(gelu(lin(y, p + ".mlp.mlp.0")), p + ".mlp.mlp.2")
            h = h + y * gM
            if i % 6 == 5 { eval(h) }
        }
        return lin(Layers.layerNorm(h, eps: 1e-5), "out_layer")
    }
}
