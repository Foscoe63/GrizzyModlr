import Foundation
import MLX
import HunyuanPaintMLX

/// Stable Diffusion 2.1 masked fill for texture patches (512²), on its own serial queue. Weights
/// are loaded per run and dropped afterwards, like the other engines.
final class InpaintEngine {
    private let queue = DispatchQueue(label: "com.zimeng.Modelr.inpaint", qos: .userInitiated)

    final class Run: CancellableRun {
        private let lock = NSLock()
        private var _cancelled = false
        var cancelled: Bool { lock.lock(); defer { lock.unlock() }; return _cancelled }
        func cancel() { lock.lock(); _cancelled = true; lock.unlock() }
    }

    struct Request {
        var rgb: [Float]            // 512·512·3, 0…1
        var mask: [Float]           // 512·512, 0…1 (1 = regenerate)
        var prompt: String
        var negative: String
        var steps: Int
        var strength: Float
        var guidance: Float
        var seed: UInt64
        var weightsDir: URL
    }

    /// Calls `onFinish` with the generated 512²·3 RGB floats (0…1) or a failure message.
    @discardableResult
    func fill(_ request: Request, onProgress: @escaping (Double) -> Void,
              onFinish: @escaping (Result<[Float], FillError>) -> Void) -> Run {
        let run = Run()
        queue.async {
            defer { MLX.GPU.clearCache() }
            if run.cancelled { onFinish(.failure(.cancelled)); return }
            let n = InpaintPatch.modelSize
            let pipe: TextToImagePipeline
            do { pipe = try TextToImagePipeline(dir: request.weightsDir, dtype: .float16) }
            catch { onFinish(.failure(.message("Couldn't load the model: \(error.localizedDescription)"))); return }

            let image = MLXArray(request.rgb.map { $0 * 2 - 1 }, [1, n, n, 3])
            // Pool the mask to latent resolution; any touched latent cell is regenerated.
            let m = MLXArray(request.mask, [1, n / 8, 8, n / 8, 8, 1])
            let latentMask = clip(m.max(axes: [2, 4]) * 1.0, min: 0, max: 1)
            guard let latents = pipe.inpaint(image: image, mask: latentMask, prompt: request.prompt,
                                             negative: request.negative, steps: request.steps,
                                             strength: request.strength, guidance: request.guidance,
                                             seed: request.seed, isCancelled: { run.cancelled },
                                             onProgress: { onProgress(Double($0) / Double($1) * 0.95) }) else {
                onFinish(.failure(.cancelled)); return
            }
            let img = pipe.decode(latents)                       // [H,W,3] 0…1
            eval(img)
            onProgress(1)
            onFinish(.success(img.asArray(Float.self)))
        }
        return run
    }

    enum FillError: Error, Equatable {
        case cancelled
        case message(String)
    }
}
