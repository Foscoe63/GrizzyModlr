import Foundation
import CoreGraphics
import ImageIO
import MLX
import HunyuanPaintMLX

/// Zero123++ view synthesis: one image → six candidate side views (a 3×2 grid, split into
/// tiles). Runs on its own serial queue and drops the weights when it finishes so the shape
/// and paint engines keep their memory.
final class ViewSynthEngine {
    private let queue = DispatchQueue(label: "com.zimeng.Modelr.viewsynth", qos: .userInitiated)

    final class Run: CancellableRun {
        private let lock = NSLock()
        private var _cancelled = false
        var cancelled: Bool { lock.lock(); defer { lock.unlock() }; return _cancelled }
        func cancel() { lock.lock(); _cancelled = true; lock.unlock() }
    }

    /// Camera of each output tile, row-major in the 3×2 grid (azimuth is relative to the
    /// input view; elevation is absolute): 30°/20°, 90°/-10°, 150°/20°, 210°/-10°, ...
    static let cameras: [(azimuth: Int, elevation: Int)] =
        (0 ..< 6).map { (azimuth: 30 + 60 * $0, elevation: $0 % 2 == 0 ? 20 : -10) }

    /// Writes the six tiles to `outputs` (`outputs.count == 6`).
    @discardableResult
    func synthesize(imageURL: URL, weightsDir: URL, seed: UInt64, steps: Int = 28,
                    outputs: [URL], onProgress: @escaping (Double) -> Void,
                    onFinish: @escaping (GenerationOutcome) -> Void) -> Run {
        let run = Run()
        queue.async {
            defer { MLX.GPU.clearCache() }
            if run.cancelled { onFinish(.failure("Cancelled")); return }
            guard let src = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
                  let cg = CGImageSourceCreateImageAtIndex(src, 0, nil),
                  let px = Z123Preprocess.prepare(cg) else {
                onFinish(.failure("Couldn't read the input image.")); return
            }
            let pipe: Zero123PlusPipeline
            do { pipe = try Zero123PlusPipeline(dir: weightsDir, dtype: .float16) }
            catch { onFinish(.failure("Couldn't load the model: \(error.localizedDescription)")); return }
            if run.cancelled { onFinish(.failure("Cancelled")); return }

            let noise = Zero123PlusPipeline.Noise.random(seed: seed, height: 960, width: 640, condSize: 512)
            guard let latents = pipe.sample(vaePixels: px.vae, clipPixels: px.clip, steps: steps, guidance: 4,
                                            height: 960, width: 640, noise: noise,
                                            isCancelled: { run.cancelled },
                                            onProgress: { onProgress(Double($0) / Double($1) * 0.95) }) else {
                onFinish(.failure("Cancelled")); return
            }
            let grid = pipe.decode(latents)
            guard let image = Self.cgImage(grid) else { onFinish(.failure("Couldn't decode the views.")); return }
            for (i, url) in outputs.enumerated() {
                let tile = image.cropping(to: CGRect(x: (i % 2) * 320, y: (i / 2) * 320, width: 320, height: 320))
                guard let tile, let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
                    onFinish(.failure("Couldn't save the views.")); return
                }
                CGImageDestinationAddImage(dest, tile, nil)
                guard CGImageDestinationFinalize(dest) else { onFinish(.failure("Couldn't save the views.")); return }
            }
            onProgress(1)
            onFinish(.success)
        }
        return run
    }

    /// `[H,W,3]` floats in 0…1 → opaque CGImage.
    private static func cgImage(_ a: MLXArray) -> CGImage? {
        let u8 = round(a * 255).asType(.uint8)
        eval(u8)
        let (h, w) = (u8.dim(0), u8.dim(1))
        let bytes = u8.asArray(UInt8.self)
        var rgba = [UInt8](repeating: 255, count: h * w * 4)
        for i in 0 ..< h * w {
            rgba[i * 4] = bytes[i * 3]; rgba[i * 4 + 1] = bytes[i * 3 + 1]; rgba[i * 4 + 2] = bytes[i * 3 + 2]
        }
        return rgba.withUnsafeMutableBytes { raw in
            CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                      space: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)?.makeImage()
        }
    }
}
