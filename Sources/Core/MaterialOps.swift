import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// Global material adjustments for a painted version. Albedo gets brightness /
/// saturation / hue; the PBR metallic-roughness map (glTF packing: G = roughness,
/// B = metallic) gets per-channel multipliers.
struct MaterialAdjustments: Equatable {
    /// Additive, -1…1 (in 0…1 color units).
    var brightness: Float = 0
    /// Multiplier on chroma, 0…2 (1 = unchanged).
    var saturation: Float = 1
    /// Degrees, -180…180.
    var hue: Float = 0
    var roughness: Float = 1
    var metalness: Float = 1
    /// When set, the edit applies only to texels near this color (soft edge); nil = everywhere.
    var region: ColorRegion?
    /// UV island ids (`UVLayout.islands` order) the edit is limited to; empty = all islands.
    var islands: Set<Int> = []

    var altersAlbedo: Bool { brightness != 0 || saturation != 1 || hue != 0 }
    var altersMR: Bool { roughness != 1 || metalness != 1 }
    var isIdentity: Bool { !altersAlbedo && !altersMR }

    var summary: String {
        var parts: [String] = []
        if brightness != 0 { parts.append(String(format: "brightness %+.2f", brightness)) }
        if saturation != 1 { parts.append(String(format: "saturation ×%.2f", saturation)) }
        if hue != 0 { parts.append(String(format: "hue %+.0f°", hue)) }
        if roughness != 1 { parts.append(String(format: "roughness ×%.2f", roughness)) }
        if metalness != 1 { parts.append(String(format: "metalness ×%.2f", metalness)) }
        if region != nil { parts.append("selected color only") }
        if !islands.isEmpty { parts.append("\(islands.count) UV island\(islands.count == 1 ? "" : "s")") }
        return parts.joined(separator: ", ")
    }
}

enum MaterialOps {
    /// In-place albedo adjustment on straight (non-premultiplied) RGBA8. Alpha untouched.
    static func adjustAlbedo(_ rgba: inout [UInt8], _ a: MaterialAdjustments) {
        guard a.altersAlbedo else { return }
        let hueRad = a.hue * .pi / 180
        let cosH = cos(hueRad), sinH = sin(hueRad)
        // Rotation about the gray axis in RGB space (standard hue-rotate matrix).
        let m: [Float] = [
            0.213 + cosH * 0.787 - sinH * 0.213, 0.715 - cosH * 0.715 - sinH * 0.715, 0.072 - cosH * 0.072 + sinH * 0.928,
            0.213 - cosH * 0.213 + sinH * 0.143, 0.715 + cosH * 0.285 + sinH * 0.140, 0.072 - cosH * 0.072 - sinH * 0.283,
            0.213 - cosH * 0.213 - sinH * 0.787, 0.715 - cosH * 0.715 + sinH * 0.715, 0.072 + cosH * 0.928 + sinH * 0.072,
        ]
        var i = 0
        while i + 3 < rgba.count {
            var r = Float(rgba[i]) / 255, g = Float(rgba[i + 1]) / 255, b = Float(rgba[i + 2]) / 255
            let luma = 0.299 * r + 0.587 * g + 0.114 * b
            r = luma + (r - luma) * a.saturation
            g = luma + (g - luma) * a.saturation
            b = luma + (b - luma) * a.saturation
            if a.hue != 0 {
                let nr = m[0] * r + m[1] * g + m[2] * b
                let ng = m[3] * r + m[4] * g + m[5] * b
                let nb = m[6] * r + m[7] * g + m[8] * b
                r = nr; g = ng; b = nb
            }
            rgba[i] = byte(r + a.brightness)
            rgba[i + 1] = byte(g + a.brightness)
            rgba[i + 2] = byte(b + a.brightness)
            i += 4
        }
    }

    /// In-place metallic-roughness scaling (G = roughness, B = metallic).
    static func adjustMR(_ rgba: inout [UInt8], _ a: MaterialAdjustments) {
        guard a.altersMR else { return }
        var i = 0
        while i + 3 < rgba.count {
            rgba[i + 1] = byte(Float(rgba[i + 1]) / 255 * a.roughness)
            rgba[i + 2] = byte(Float(rgba[i + 2]) / 255 * a.metalness)
            i += 4
        }
    }

    private static func byte(_ v: Float) -> UInt8 { UInt8(max(0, min(1, v)) * 255 + 0.5) }

    // MARK: - image files

    /// Decode an image to straight RGBA8.
    static func decode(_ url: URL) -> (rgba: [UInt8], width: Int, height: Int)? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        let w = img.width, h = img.height
        guard w > 0, h > 0 else { return nil }
        var px = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        // Un-premultiply so color math isn't skewed on translucent texels.
        var i = 0
        while i < px.count {
            let al = Int(px[i + 3])
            if al > 0 && al < 255 {
                for c in 0..<3 { px[i + c] = UInt8(min(255, Int(px[i + c]) * 255 / al)) }
            }
            i += 4
        }
        return (px, w, h)
    }

    /// Encode straight RGBA8 as PNG.
    static func writePNG(_ rgba: [UInt8], width: Int, height: Int, to url: URL) -> Bool {
        var px = rgba
        var i = 0
        while i < px.count {                       // re-premultiply for CoreGraphics
            let al = Int(px[i + 3])
            if al < 255 { for c in 0..<3 { px[i + c] = UInt8(Int(px[i + c]) * al / 255) } }
            i += 4
        }
        guard let ctx = CGContext(data: &px, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let img = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { return false }
        CGImageDestinationAddImage(dest, img, nil)
        return CGImageDestinationFinalize(dest)
    }

    /// Adjust an albedo or MR PNG file into `dst`.
    /// With `mask` (the region weights over the albedo, sized `maskSize`), only the selected
    /// texels change; the rest keep their original values.
    static func processFile(_ src: URL, to dst: URL, isMR: Bool, _ a: MaterialAdjustments,
                            mask: [Float]? = nil, maskSize: (w: Int, h: Int) = (0, 0)) -> Bool {
        guard var d = decode(src) else { return false }
        let original = d.rgba
        if isMR { adjustMR(&d.rgba, a) } else { adjustAlbedo(&d.rgba, a) }
        if let mask {
            MaterialRegion.blend(original: original, edited: &d.rgba, mask: mask,
                                 maskSize: maskSize, size: (d.width, d.height))
        }
        return writePNG(d.rgba, width: d.width, height: d.height, to: dst)
    }
}
