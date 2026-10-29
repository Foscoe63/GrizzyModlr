import Foundation
import CoreGraphics
import MLX

/// Host-side image preparation for Zero123++ (what the diffusers feature extractors do):
/// composite onto constant gray, resize the shortest edge with PIL-style antialiased filters,
/// center-crop, normalize. Returns NHWC.
public enum Z123Preprocess {
    /// `vae`: 512×512 bilinear, `(x − 0.5) / 0.8`. `clip`: 224×224 bicubic with CLIP mean/std.
    public static func prepare(_ image: CGImage) -> (vae: MLXArray, clip: MLXArray)? {
        guard let rgb = compositeOnGray(image) else { return nil }
        let v = resizeCrop(rgb, size: 512, bicubic: false)
        let c = resizeCrop(rgb, size: 224, bicubic: true)
        let mean: [Float] = [0.48145466, 0.4578275, 0.40821073], std: [Float] = [0.26862954, 0.26130258, 0.27577711]
        var vf = v.pixels, cf = c.pixels
        for i in 0 ..< vf.count { vf[i] = (vf[i] - 0.5) / 0.8 }
        for i in 0 ..< cf.count { cf[i] = (cf[i] - mean[i % 3]) / std[i % 3] }
        return (MLXArray(vf, [1, 512, 512, 3]), MLXArray(cf, [1, 224, 224, 3]))
    }

    struct RGB { var pixels: [Float]; var width: Int; var height: Int }

    /// Straight RGB floats in 0…1 with transparency composited over gray 127.
    static func compositeOnGray(_ image: CGImage) -> RGB? {
        let w = image.width, h = image.height
        guard w > 0, h > 0 else { return nil }
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        let ok = buf.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8,
                                      bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { return nil }
        var out = [Float](repeating: 0, count: w * h * 3)
        let gray: Float = 127
        for i in 0 ..< w * h {
            let a = Float(buf[i * 4 + 3]) / 255
            for c in 0 ..< 3 {                                   // premultiplied: rgb + bg·(1−a)
                out[i * 3 + c] = min(255, Float(buf[i * 4 + c]) + gray * (1 - a)) / 255
            }
        }
        return RGB(pixels: out, width: w, height: h)
    }

    /// Shortest edge → `size` (long edge `int(size·long/short)`), then center crop `size²`.
    static func resizeCrop(_ img: RGB, size: Int, bicubic: Bool) -> RGB {
        let (w, h) = (img.width, img.height)
        let (nw, nh) = w <= h ? (size, Int(Double(size) * Double(h) / Double(w))) : (Int(Double(size) * Double(w) / Double(h)), size)
        let resized = resize(img, to: (nw, nh), bicubic: bicubic)
        let left = (nw - size) / 2, top = (nh - size) / 2
        var out = [Float](repeating: 0, count: size * size * 3)
        for y in 0 ..< size {
            let s = ((y + top) * nw + left) * 3
            out.replaceSubrange(y * size * 3 ..< (y + 1) * size * 3, with: resized.pixels[s ..< s + size * 3])
        }
        return RGB(pixels: out, width: size, height: size)
    }

    // MARK: PIL-compatible separable resampling (antialiased: filter support scales with the shrink)

    private static func coefficients(inSize: Int, outSize: Int, bicubic: Bool) -> [(start: Int, weights: [Float])] {
        let scale = Double(inSize) / Double(outSize)
        let fscale = max(scale, 1)
        let support = (bicubic ? 2.0 : 1.0) * fscale
        func filter(_ x: Double) -> Double {
            if bicubic {
                let a = -0.5, ax = abs(x)
                if ax < 1 { return ((a + 2) * ax - (a + 3)) * ax * ax + 1 }
                if ax < 2 { return (((ax - 5) * ax + 8) * ax - 4) * a }
                return 0
            }
            return max(0, 1 - abs(x))
        }
        return (0 ..< outSize).map { i in
            let center = (Double(i) + 0.5) * scale
            let lo = max(Int((center - support + 0.5).rounded(.down)), 0)
            let hi = min(Int((center + support + 0.5).rounded(.down)), inSize)
            var ws = (lo ..< hi).map { filter((Double($0) - center + 0.5) / fscale) }
            let sum = ws.reduce(0, +)
            if sum != 0 { ws = ws.map { $0 / sum } }
            return (lo, ws.map { Float($0) })
        }
    }

    static func resize(_ img: RGB, to size: (Int, Int), bicubic: Bool) -> RGB {
        let (w, h) = (img.width, img.height), (nw, nh) = size
        let cx = coefficients(inSize: w, outSize: nw, bicubic: bicubic)
        var tmp = [Float](repeating: 0, count: nw * h * 3)            // horizontal pass
        for y in 0 ..< h { for x in 0 ..< nw {
            let (start, ws) = cx[x]
            for c in 0 ..< 3 {
                var acc: Float = 0
                for (k, wt) in ws.enumerated() { acc += img.pixels[(y * w + start + k) * 3 + c] * wt }
                tmp[(y * nw + x) * 3 + c] = acc
            }
        } }
        let cy = coefficients(inSize: h, outSize: nh, bicubic: bicubic)
        var out = [Float](repeating: 0, count: nw * nh * 3)           // vertical pass
        for y in 0 ..< nh { for x in 0 ..< nw {
            let (start, ws) = cy[y]
            for c in 0 ..< 3 {
                var acc: Float = 0
                for (k, wt) in ws.enumerated() { acc += tmp[((start + k) * nw + x) * 3 + c] * wt }
                out[(y * nw + x) * 3 + c] = acc
            }
        } }
        return RGB(pixels: out, width: nw, height: nh)
    }
}
