import XCTest
import MLX
import MLXRandom
@testable import Hy3DMLX

/// Structural checks for the 2.1 denoiser on a tiny random network (the real-weights parity numbers
/// come from `h3d21 parity`, which needs the 7 GB checkpoint).
final class DiT21Tests: XCTestCase {
    private func tinyWeights(hidden: Int = 32, heads: Int = 2, depth: Int = 4, moe: Int = 2, experts: Int = 4,
                             ctx: Int = 16) -> [String: MLXArray] {
        var w: [String: MLXArray] = [:]
        func lin(_ n: String, _ o: Int, _ i: Int, bias: Bool = true) {
            w["\(n).weight"] = MLXRandom.normal([o, i]) * 0.05
            if bias { w["\(n).bias"] = MLXRandom.normal([o]) * 0.01 }
        }
        func norm(_ n: String, _ d: Int) { w["\(n).weight"] = MLXArray.ones([d]); w["\(n).bias"] = MLXArray.zeros([d]) }
        func ffn(_ n: String) { lin("\(n).net.0.proj", hidden * 4, hidden); lin("\(n).net.2", hidden, hidden * 4) }
        lin("x_embedder", hidden, 8)
        lin("t_embedder.mlp.0", hidden * 4, hidden); lin("t_embedder.mlp.2", hidden, hidden * 4)
        for i in 0 ..< depth {
            let p = "blocks.\(i)"
            norm("\(p).norm1", hidden); norm("\(p).norm2", hidden); norm("\(p).norm3", hidden)
            for a in ["attn1", "attn2"] {
                let kd = a == "attn1" ? hidden : ctx
                lin("\(p).\(a).to_q", hidden, hidden, bias: false)
                lin("\(p).\(a).to_k", hidden, kd, bias: false); lin("\(p).\(a).to_v", hidden, kd, bias: false)
                lin("\(p).\(a).out_proj", hidden, hidden)
                w["\(p).\(a).q_norm.weight"] = MLXArray.ones([hidden / heads])
                w["\(p).\(a).k_norm.weight"] = MLXArray.ones([hidden / heads])
            }
            if i > depth / 2 { norm("\(p).skip_norm", hidden); lin("\(p).skip_linear", hidden, hidden * 2) }
            if depth - i <= moe {
                for e in 0 ..< experts { ffn("\(p).moe.experts.\(e)") }
                ffn("\(p).moe.shared_experts"); lin("\(p).moe.gate", experts, hidden, bias: false)
            } else { lin("\(p).mlp.fc1", hidden * 4, hidden); lin("\(p).mlp.fc2", hidden, hidden * 4) }
        }
        norm("final_layer.norm_final", hidden); lin("final_layer.linear", 8, hidden)
        return w
    }

    func testForwardShapeAndPerTokenRouting() {
        MLXRandom.seed(7)
        let dit = DiT21(weights: tinyWeights(), heads: 2, hidden: 32, depth: 4, moeLayers: 2, experts: 4, topK: 2)
        let x = MLXRandom.normal([1, 24, 8]), c = MLXRandom.normal([1, 16, 16])
        let y = dit(x, MLXArray([Float(0.3)]), c)
        XCTAssertEqual(y.shape, [1, 24, 8])
        // A duplicated batch (as in classifier-free guidance) must give identical halves: MoE
        // routing is per token and independent of batch position.
        let y2 = dit(concatenated([x, x], axis: 0), MLXArray([Float(0.3), Float(0.3)]), concatenated([c, c], axis: 0))
        XCTAssertEqual(y2.shape, [2, 24, 8])
        let d = abs(y2[0] - y2[1]).max().item(Float.self)
        XCTAssertLessThan(d, 1e-5)
        XCTAssertLessThan(abs(y2[0] - y[0]).max().item(Float.self), 1e-4)
        XCTAssertTrue(y.sum().item(Float.self).isFinite)
    }
}
