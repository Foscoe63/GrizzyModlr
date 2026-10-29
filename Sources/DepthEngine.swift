import Foundation
import CoreGraphics
import ImageIO
import MLX
import HunyuanPaintMLX

/// Depth Anything → relief mesh. Runs on its own serial queue and frees its weights afterwards.
final class DepthEngine {
    private let queue = DispatchQueue(label: "com.zimeng.Modelr.depth", qos: .userInitiated)

    final class Run: CancellableRun {
        private let lock = NSLock()
        private var _cancelled = false
        var cancelled: Bool { lock.lock(); defer { lock.unlock() }; return _cancelled }
        func cancel() { lock.lock(); _cancelled = true; lock.unlock() }
    }

    /// `colorURL` supplies the picture the depth is estimated from; `maskURL` (an image with
    /// transparency, e.g. the background-removed cutout) trims the silhouette. Writes a `.mesh`.
    @discardableResult
    func relief(colorURL: URL, maskURL: URL?, weights: URL, options: ReliefMesh.Options, resolution: Int,
                output: URL, onFinish: @escaping (GenerationOutcome) -> Void) -> Run {
        let run = Run()
        queue.async {
            defer { MLX.GPU.clearCache() }
            if run.cancelled { onFinish(.failure("Cancelled")); return }
            guard let color = Self.load(colorURL) else { onFinish(.failure("Couldn't read the image.")); return }
            let model: DepthAnythingSmall
            do { model = try DepthAnythingSmall.load(weights) }
            catch { onFinish(.failure("Couldn't load the depth model: \(error.localizedDescription)")); return }

            let size = DepthAnythingSmall.inputSize
            guard let px = Self.pixels(color, size: size) else { onFinish(.failure("Couldn't prepare the image.")); return }
            let depth = model(px)
            eval(depth)
            if run.cancelled { onFinish(.failure("Cancelled")); return }
            let raw = depth.asArray(Float.self)                                   // size × size

            // Grid in the image's aspect, mask from the cutout's alpha.
            let (gw, gh) = ReliefMesh.gridSize(imageWidth: color.width, imageHeight: color.height, resolution: resolution)
            var samples = [Float](repeating: 0, count: gw * gh)
            for y in 0 ..< gh { for x in 0 ..< gw {
                let fx = (Float(x) + 0.5) / Float(gw) * Float(size) - 0.5, fy = (Float(y) + 0.5) / Float(gh) * Float(size) - 0.5
                samples[y * gw + x] = Self.sample(raw, size, fx, fy)
            } }
            var mask: [Bool]?
            if let maskURL, let m = Self.load(maskURL), let alpha = Self.alpha(m, gw, gh) { mask = alpha }
            let normalized = ReliefMesh.normalize(samples, mask: mask)
            guard let mesh = ReliefMesh.build(depth: normalized, mask: mask, width: gw, height: gh, options: options) else {
                onFinish(.failure("The image has no object to build a relief from.")); return
            }
            do {
                try MeshDecimator.serialize(positions: mesh.positions, faces: mesh.faces).write(to: output)
                onFinish(.success)
            } catch { onFinish(.failure("Couldn't write the relief mesh.")) }
        }
        return run
    }

    private static func load(_ url: URL) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }

    /// ImageNet-normalized NHWC `[1,size,size,3]` (squashed to a square, like the reference).
    private static func pixels(_ image: CGImage, size: Int) -> MLXArray? {
        var buf = [UInt8](repeating: 0, count: size * size * 4)
        let ok = buf.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(data: raw.baseAddress, width: size, height: size, bitsPerComponent: 8, bytesPerRow: size * 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            ctx.setFillColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1)                 // transparent areas → gray
            ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
            return true
        }
        guard ok else { return nil }
        let mean: [Float] = [0.485, 0.456, 0.406], std: [Float] = [0.229, 0.224, 0.225]
        var f = [Float](repeating: 0, count: size * size * 3)
        for i in 0 ..< size * size { for c in 0 ..< 3 { f[i * 3 + c] = (Float(buf[i * 4 + c]) / 255 - mean[c]) / std[c] } }
        return MLXArray(f, [1, size, size, 3])
    }

    private static func sample(_ d: [Float], _ n: Int, _ fx: Float, _ fy: Float) -> Float {
        let x0 = min(max(Int(fx.rounded(.down)), 0), n - 1), y0 = min(max(Int(fy.rounded(.down)), 0), n - 1)
        let x1 = min(x0 + 1, n - 1), y1 = min(y0 + 1, n - 1)
        let tx = min(max(fx - Float(x0), 0), 1), ty = min(max(fy - Float(y0), 0), 1)
        let top = d[y0 * n + x0] * (1 - tx) + d[y0 * n + x1] * tx
        let bot = d[y1 * n + x0] * (1 - tx) + d[y1 * n + x1] * tx
        return top * (1 - ty) + bot * ty
    }

    /// Opaque-pixel mask on a `gw × gh` grid, or nil when the image has no transparency.
    private static func alpha(_ image: CGImage, _ gw: Int, _ gh: Int) -> [Bool]? {
        var buf = [UInt8](repeating: 0, count: gw * gh * 4)
        let ok = buf.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(data: raw.baseAddress, width: gw, height: gh, bitsPerComponent: 8, bytesPerRow: gw * 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            ctx.interpolationQuality = .medium
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: gw, height: gh))
            return true
        }
        guard ok else { return nil }
        let mask = (0 ..< gw * gh).map { buf[$0 * 4 + 3] > 127 }
        return mask.allSatisfy { $0 } ? nil : mask
    }
}
