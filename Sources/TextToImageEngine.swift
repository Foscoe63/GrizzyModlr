import Foundation
import CoreGraphics
import ImageIO
import MLX
import HunyuanPaintMLX

/// Stable Diffusion 2.1 text → image (512²) on its own serial queue; weights are dropped when
/// the run ends so the 3D engines keep their memory.
final class TextToImageEngine {
    private let queue = DispatchQueue(label: "com.zimeng.Modelr.texttoimage", qos: .userInitiated)

    final class Run: CancellableRun {
        private let lock = NSLock()
        private var _cancelled = false
        var cancelled: Bool { lock.lock(); defer { lock.unlock() }; return _cancelled }
        func cancel() { lock.lock(); _cancelled = true; lock.unlock() }
    }

    @discardableResult
    func generate(prompt: String, negative: String, steps: Int, guidance: Float, seed: UInt64,
                  weightsDir: URL, output: URL, onProgress: @escaping (Double) -> Void,
                  onFinish: @escaping (GenerationOutcome) -> Void) -> Run {
        let run = Run()
        queue.async {
            defer { MLX.GPU.clearCache() }
            if run.cancelled { onFinish(.failure("Cancelled")); return }
            let pipe: TextToImagePipeline
            do { pipe = try TextToImagePipeline(dir: weightsDir, dtype: .float16) }
            catch { onFinish(.failure("Couldn't load the model: \(error.localizedDescription)")); return }
            guard let latents = pipe.sample(prompt: prompt, negative: negative, steps: steps, guidance: guidance,
                                            initialNoise: TextToImagePipeline.randomNoise(seed: seed),
                                            isCancelled: { run.cancelled },
                                            onProgress: { onProgress(Double($0) / Double($1) * 0.95) }) else {
                onFinish(.failure("Cancelled")); return
            }
            let img = pipe.decode(latents)
            let u8 = round(img * 255).asType(.uint8)
            eval(u8)
            let (h, w) = (u8.dim(0), u8.dim(1))
            let bytes = u8.asArray(UInt8.self)
            var rgba = [UInt8](repeating: 255, count: h * w * 4)
            for i in 0 ..< h * w {
                rgba[i * 4] = bytes[i * 3]; rgba[i * 4 + 1] = bytes[i * 3 + 1]; rgba[i * 4 + 2] = bytes[i * 3 + 2]
            }
            let cg = rgba.withUnsafeMutableBytes { raw in
                CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                          space: CGColorSpaceCreateDeviceRGB(),
                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)?.makeImage()
            }
            guard let cg, let dest = CGImageDestinationCreateWithURL(output as CFURL, "public.png" as CFString, 1, nil) else {
                onFinish(.failure("Couldn't save the image.")); return
            }
            CGImageDestinationAddImage(dest, cg, nil)
            guard CGImageDestinationFinalize(dest) else { onFinish(.failure("Couldn't save the image.")); return }
            onProgress(1)
            onFinish(.success)
        }
        return run
    }
}
