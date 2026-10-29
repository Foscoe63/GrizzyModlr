import Foundation
import MLX

/// Camera slot of a reference view for multi-view (2mv) generation. The raw value is the
/// index into the checkpoint's view-embedding table (upstream order: front, left, back, right).
public enum ViewAngle: Int, CaseIterable, Sendable {
    case front = 0, left, back, right
}

public enum MultiView {
    /// Upstream `get_1d_sincos_pos_embed_from_grid(hidden, pos)` for a single position:
    /// `[sin(pos·ω_0 … pos·ω_{D/2-1}), cos(…)]`, ω_k = 1 / 10000^(k / (D/2)). Computed in
    /// Double like numpy, then returned as float32 `[hidden]`.
    public static func viewEmbedding(index: Int, hidden: Int) -> MLXArray {
        MLXArray(values(index: index, hidden: hidden))
    }

    /// The raw embedding values (exposed for tests that don't need the GPU).
    public static func values(index: Int, hidden: Int) -> [Float] {
        precondition(hidden % 2 == 0, "hidden size must be even")
        let half = hidden / 2
        var sins = [Float](), coss = [Float]()
        sins.reserveCapacity(half); coss.reserveCapacity(half)
        for k in 0..<half {
            let omega = 1.0 / pow(10000.0, Double(k) / Double(half))
            let angle = Double(index) * omega
            sins.append(Float(sin(angle)))
            coss.append(Float(cos(angle)))
        }
        return sins + coss
    }
}
