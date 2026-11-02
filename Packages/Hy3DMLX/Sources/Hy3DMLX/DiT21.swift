import Foundation
import MLX

/// Hunyuan3D 2.1 shape denoiser (`HunYuanDiTPlain`): U-ViT style Hunyuan-DiT with a prepended timestep
/// token, cross-attention to DINOv2-large tokens, long skip connections in the second half and
/// mixture-of-experts MLPs in the last `moeLayers` blocks. Functional port of
/// hy3dshape/models/denoisers/hunyuandit.py + moe_layers.py.
public struct DiT21: ShapeDenoiser {
    let w: [String: MLXArray]
    let heads: Int
    let hidden: Int
    let depth: Int
    let moeLayers: Int
    let experts: Int
    let topK: Int
    let rmsEps: Float
    let bits: Int
    let groupSize: Int
    public let guidanceEmbed = false

    /// `rmsEps`: torch `nn.RMSNorm(eps=None)` uses the input dtype's machine epsilon (fp16 ≈ 9.8e-4,
    /// fp32 ≈ 1.2e-7); default matches the fp32 reference (cosine 0.99999 vs 0.9967 with the fp16 value).
    public init(weights: [String: MLXArray], heads: Int = 16, hidden: Int = 2048, depth: Int = 21,
                moeLayers: Int = 6, experts: Int = 8, topK: Int = 2, rmsEps: Float = 1.1920929e-7,
                bits: Int = 4, groupSize: Int = 64) {
        self.w = weights; self.heads = heads; self.hidden = hidden; self.depth = depth
        self.moeLayers = moeLayers; self.experts = experts; self.topK = topK; self.rmsEps = rmsEps
        self.bits = bits; self.groupSize = groupSize
    }

    private func g(_ k: String) -> MLXArray { w[k]! }
    private func lin(_ x: MLXArray, _ p: String) -> MLXArray {
        let wk = "\(p).weight"
        if let s = w["\(wk).scales"], let b = w["\(wk).biases"] {
            return Layers.qlinear(x, g(wk), scales: s, biases: b, bias: w["\(p).bias"],
                                  bits: bits, groupSize: groupSize)
        }
        return Layers.linear(x, g(wk), w["\(p).bias"])
    }
    private func ln(_ x: MLXArray, _ p: String) -> MLXArray {
        Layers.layerNorm(x, g("\(p).weight"), g("\(p).bias"), eps: 1e-6)
    }
    private func qkNorm(_ x: MLXArray, _ p: String) -> MLXArray { Layers.rmsNorm(x, g("\(p).weight"), eps: rmsEps) }

    /// Timesteps(hidden): sin|cos with max_period 10000, then Linear-GELU-Linear. t: [B] -> [B,1,hidden].
    private func timeToken(_ t: MLXArray) -> MLXArray {
        let half = hidden / 2
        let freqs = exp(-Float(log(10000.0)) * MLXArray(0 ..< half).asType(.float32) / Float(half))
        let args = t.asType(.float32).reshaped([t.dim(0), 1]) * freqs.reshaped([1, half])
        let emb = concatenated([sin(args), cos(args)], axis: -1).asType(g("t_embedder.mlp.0.weight").dtype)
        let h = lin(Layers.geluErf(lin(emb, "t_embedder.mlp.0")), "t_embedder.mlp.2")
        return h.expandedDimensions(axis: 1)
    }

    /// The reference concatenates q|k|v along the channel axis and then views the result as
    /// [heads, 3*head_dim]: the split is a literal reinterpretation of that buffer, reproduced here.
    private func selfAttn(_ x: MLXArray, _ p: String) -> MLXArray {
        let B = x.dim(0), N = x.dim(1)
        let qkv = concatenated([lin(x, "\(p).to_q"), lin(x, "\(p).to_k"), lin(x, "\(p).to_v")], axis: -1)
        let parts = split(qkv.reshaped([B, N, heads, -1]), parts: 3, axis: -1)   // [B,N,H,hd] each
        let q = qkNorm(parts[0], "\(p).q_norm").transposed(0, 2, 1, 3)
        let k = qkNorm(parts[1], "\(p).k_norm").transposed(0, 2, 1, 3)
        let v = parts[2].transposed(0, 2, 1, 3)
        let o = Layers.sdpa(q, k, v).transposed(0, 2, 1, 3).reshaped([B, N, -1])
        return lin(o, "\(p).out_proj")
    }

    private func crossAttn(_ x: MLXArray, _ y: MLXArray, _ p: String) -> MLXArray {
        let B = x.dim(0), N = x.dim(1), S = y.dim(1)
        let kv = concatenated([lin(y, "\(p).to_k"), lin(y, "\(p).to_v")], axis: -1)
        let parts = split(kv.reshaped([B, S, heads, -1]), parts: 2, axis: -1)
        let q = qkNorm(lin(x, "\(p).to_q").reshaped([B, N, heads, -1]), "\(p).q_norm").transposed(0, 2, 1, 3)
        let k = qkNorm(parts[0], "\(p).k_norm").transposed(0, 2, 1, 3)
        let v = parts[1].transposed(0, 2, 1, 3)
        let o = Layers.sdpa(q, k, v).transposed(0, 2, 1, 3).reshaped([B, N, -1])
        return lin(o, "\(p).out_proj")
    }

    private func ffn(_ x: MLXArray, _ p: String) -> MLXArray {   // diffusers FeedForward(gelu)
        lin(Layers.geluErf(lin(x, "\(p).net.0.proj")), "\(p).net.2")
    }

    /// Top-k softmax routing (weights not renormalised) + shared expert. Tokens are grouped by expert
    /// so each expert runs once over its own tokens.
    private func moe(_ x: MLXArray, _ p: String) -> MLXArray {
        let shape = x.shape, D = x.dim(-1)
        let xf = x.reshaped([-1, D])
        let T = xf.dim(0)
        let logits = matmul(xf.asType(.float32), g("\(p).gate.weight").asType(.float32).transposed(1, 0))
        let scores = softmax(logits, axis: -1)                                    // [T,E]
        let idx = argPartition(scores, kth: experts - topK, axis: -1)[0..., (experts - topK)...]   // [T,k]
        let topW = takeAlong(scores, idx, axis: -1).reshaped([-1])                // [T*k]
        let host = idx.asType(.int32).asArray(Int32.self)
        var buckets = [[Int32]](repeating: [], count: experts)
        for (a, e) in host.enumerated() { buckets[Int(e)].append(Int32(a)) }
        var order = [Int32](); order.reserveCapacity(host.count)
        var ranges = [(Int, Int, Int)]()
        for (e, b) in buckets.enumerated() where !b.isEmpty {
            ranges.append((e, order.count, order.count + b.count)); order += b
        }
        var inverse = [Int32](repeating: 0, count: order.count)
        for (j, a) in order.enumerated() { inverse[Int(a)] = Int32(j) }
        let orderA = MLXArray(order), tokens = floorDivide(orderA, MLXArray(Int32(topK)))
        let xs = xf[tokens]
        var outs = [MLXArray]()
        for (e, s, t) in ranges { outs.append(ffn(xs[s ..< t], "\(p).experts.\(e)")) }
        var ys = concatenated(outs, axis: 0)
        ys = ys * topW[orderA].asType(ys.dtype).reshaped([-1, 1])
        let routed = ys[MLXArray(inverse)].reshaped([T, topK, D]).sum(axis: 1)
        return routed.reshaped(shape) + ffn(x, "\(p).shared_experts")
    }

    private func block(_ x0: MLXArray, _ cond: MLXArray, _ skip: MLXArray?, _ i: Int) -> MLXArray {
        let p = "blocks.\(i)"
        var x = x0
        if let skip {
            x = lin(concatenated([skip, x], axis: -1), "\(p).skip_linear")
            x = ln(x, "\(p).skip_norm")
        }
        x = x + selfAttn(ln(x, "\(p).norm1"), "\(p).attn1")
        x = x + crossAttn(ln(x, "\(p).norm2"), cond, "\(p).attn2")
        let h = ln(x, "\(p).norm3")
        if depth - i <= moeLayers { return x + moe(h, "\(p).moe") }
        return x + lin(Layers.geluErf(lin(h, "\(p).mlp.fc1")), "\(p).mlp.fc2")
    }

    /// x [B,N,64], t [B] in [0,1], cond [B,T,1024] -> velocity [B,N,64].
    public func callAsFunction(_ x: MLXArray, _ t: MLXArray, _ cond: MLXArray, guidance: MLXArray? = nil) -> MLXArray {
        var h = concatenated([timeToken(t).asType(x.dtype), lin(x, "x_embedder")], axis: 1)
        var skips: [MLXArray] = []
        for i in 0 ..< depth {
            h = block(h, cond, i > depth / 2 ? skips.removeLast() : nil, i)
            if i < depth / 2 { skips.append(h) }
        }
        let out = lin(ln(h, "final_layer.norm_final"), "final_layer.linear")
        return out[0..., 1...]
    }
}
