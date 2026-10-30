import Foundation
import MLX

/// Flow-matching Euler sampler with classifier-free guidance, a guidance interval and CFG rescale, matching
/// TRELLIS.2 `FlowEulerGuidanceIntervalSampler` (`pipeline.json` sampler params).
struct FlowEulerSampler {
    struct Params {
        var steps = 12
        var rescaleT: Float = 5
        var guidanceStrength: Float = 7.5
        var guidanceRescale: Float = 0.7
        var guidanceInterval: (Float, Float) = (0.6, 1.0)
        var sigmaMin: Float = 1e-5
    }

    var params: Params

    /// Timesteps from 1 to 0 with the `rescale_t` warp, as (t, tPrev) pairs.
    func schedule() -> [(Double, Double)] {
        let n = params.steps
        let r = Double(params.rescaleT)
        let seq = (0 ... n).map { i -> Double in
            let t = 1 - Double(i) / Double(n)
            return r * t / (1 + (r - 1) * t)
        }
        return (0 ..< n).map { (seq[$0], seq[$0 + 1]) }
    }

    /// `predict(x, t1000, conditional)` returns the model's velocity; t1000 is t×1000 as the model expects.
    func sample(noise: MLXArray, predict: (MLXArray, Float, Bool) -> MLXArray) -> MLXArray {
        let sig = params.sigmaMin
        var x = noise
        for (t, tPrev) in schedule() {
            let tf = Float(t)
            let s = (Double(params.guidanceInterval.0) <= t && t <= Double(params.guidanceInterval.1)) ? params.guidanceStrength : 1
            var pred: MLXArray
            if s == 1 { pred = predict(x, tf * 1000, true) }
            else if s == 0 { pred = predict(x, tf * 1000, false) }
            else {
                let pos = predict(x, tf * 1000, true), neg = predict(x, tf * 1000, false)
                pred = s * pos + (1 - s) * neg
                if params.guidanceRescale > 0 {
                    let denom = sig + (1 - sig) * tf
                    let x0p = (1 - sig) * x - denom * pos
                    let x0c = (1 - sig) * x - denom * pred
                    let x0r = x0c * (unbiasedStd(x0p) / unbiasedStd(x0c))
                    let x0 = params.guidanceRescale * x0r + (1 - params.guidanceRescale) * x0c
                    pred = ((1 - sig) * x - x0) / denom
                }
            }
            x = x - Float(t - tPrev) * pred
            eval(x)
        }
        return x
    }

    /// torch `Tensor.std()` default (Bessel-corrected) over all elements.
    private func unbiasedStd(_ a: MLXArray) -> MLXArray {
        let n = Float(a.size)
        let m = a.mean()
        return sqrt(((a - m) * (a - m)).sum() / (n - 1))
    }
}
