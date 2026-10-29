import Foundation
import CoreGraphics
import simd

enum PaintTool: String, CaseIterable, Identifiable {
    case paint, soften, heal
    var id: String { rawValue }
    var label: String {
        switch self {
        case .paint: return "Paint"
        case .soften: return "Soften"
        case .heal: return "Heal"
        }
    }
    var icon: String {
        switch self {
        case .paint: return "paintbrush.pointed"
        case .soften: return "drop"
        case .heal: return "bandage"
        }
    }
}

/// Editable RGBA8 (straight alpha) texture for brush painting. Pure value logic: UV
/// coordinates follow SceneKit's mapping for image contents (column = u·w, row = v·h).
struct TextureCanvas {
    let width: Int
    let height: Int
    private(set) var pixels: [UInt8]

    // Undo: per stroke, the original contents of every 64×64 tile it touched.
    private static let tile = 64
    private var strokeTiles: [Int: [UInt8]]?
    private var undoStack: [[Int: [UInt8]]] = []
    var canUndo: Bool { !undoStack.isEmpty }
    private(set) var isEdited = false

    init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(pixels.count == width * height * 4)
        self.width = width; self.height = height; self.pixels = pixels
    }

    // MARK: strokes / undo

    mutating func beginStroke() { strokeTiles = [:] }
    mutating func endStroke() {
        if let t = strokeTiles, !t.isEmpty { undoStack.append(t) }
        strokeTiles = nil
    }

    mutating func undo() {
        guard let last = undoStack.popLast() else { return }
        let tilesX = (width + Self.tile - 1) / Self.tile
        for (index, data) in last {
            let (x0, y0) = ((index % tilesX) * Self.tile, (index / tilesX) * Self.tile)
            let (w, h) = (min(Self.tile, width - x0), min(Self.tile, height - y0))
            for row in 0..<h {
                let dst = ((y0 + row) * width + x0) * 4
                let src = row * w * 4
                pixels.replaceSubrange(dst..<dst + w * 4, with: data[src..<src + w * 4])
            }
        }
        isEdited = !undoStack.isEmpty
    }

    private mutating func remember(x0: Int, y0: Int, x1: Int, y1: Int) {
        guard strokeTiles != nil else { return }
        let tilesX = (width + Self.tile - 1) / Self.tile
        for ty in (y0 / Self.tile)...(y1 / Self.tile) {
            for tx in (x0 / Self.tile)...(x1 / Self.tile) {
                let index = ty * tilesX + tx
                guard strokeTiles![index] == nil else { continue }
                let (tx0, ty0) = (tx * Self.tile, ty * Self.tile)
                let (w, h) = (min(Self.tile, width - tx0), min(Self.tile, height - ty0))
                var data = [UInt8](); data.reserveCapacity(w * h * 4)
                for row in 0..<h {
                    let s = ((ty0 + row) * width + tx0) * 4
                    data.append(contentsOf: pixels[s..<s + w * 4])
                }
                strokeTiles![index] = data
            }
        }
    }

    // MARK: dabs

    /// Pixel rectangle covered by a dab at (u, v) with `radius` pixels, clamped.
    private func bounds(_ u: Float, _ v: Float, _ radius: Float) -> (x0: Int, y0: Int, x1: Int, y1: Int)? {
        let cx = u * Float(width), cy = v * Float(height)
        let x0 = max(0, Int((cx - radius).rounded(.down))), x1 = min(width - 1, Int((cx + radius).rounded(.up)))
        let y0 = max(0, Int((cy - radius).rounded(.down))), y1 = min(height - 1, Int((cy + radius).rounded(.up)))
        return x0 <= x1 && y0 <= y1 ? (x0, y0, x1, y1) : nil
    }

    /// Paints `color` (straight RGBA 0…255) with the given opacity (0…1).
    mutating func paint(u: Float, v: Float, radius: Float, color: SIMD4<Float>,
                        hardness: Float, opacity: Float) {
        guard let b = bounds(u, v, radius) else { return }
        remember(x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1)
        let cx = u * Float(width), cy = v * Float(height)
        for y in b.y0...b.y1 {
            for x in b.x0...b.x1 {
                let d = ((Float(x) + 0.5 - cx) * (Float(x) + 0.5 - cx)
                         + (Float(y) + 0.5 - cy) * (Float(y) + 0.5 - cy)).squareRoot()
                guard d < radius else { continue }
                let t = d / radius
                let a = opacity * (t <= hardness ? 1 : 0.5 * (1 + cos(.pi * (t - hardness) / max(1 - hardness, 1e-4))))
                let i = (y * width + x) * 4
                for c in 0..<3 {
                    pixels[i + c] = UInt8(max(0, min(255, Float(pixels[i + c]) * (1 - a) + color[c] * a + 0.5)))
                }
            }
        }
        isEdited = true
    }

    /// Blurs the area under the brush (softens harsh paint/seams).
    mutating func soften(u: Float, v: Float, radius: Float, strength: Float) {
        guard let b = bounds(u, v, radius) else { return }
        remember(x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1)
        let cx = u * Float(width), cy = v * Float(height)
        let src = pixels
        for y in b.y0...b.y1 {
            for x in b.x0...b.x1 {
                let d = ((Float(x) + 0.5 - cx) * (Float(x) + 0.5 - cx)
                         + (Float(y) + 0.5 - cy) * (Float(y) + 0.5 - cy)).squareRoot()
                guard d < radius else { continue }
                let w = strength * 0.5 * (1 + cos(.pi * d / radius))
                var acc = SIMD3<Float>.zero, n: Float = 0
                for dy in -1...1 { for dx in -1...1 {
                    let xx = min(max(x + dx, 0), width - 1), yy = min(max(y + dy, 0), height - 1)
                    let j = (yy * width + xx) * 4
                    acc += SIMD3(Float(src[j]), Float(src[j + 1]), Float(src[j + 2])); n += 1
                } }
                let avg = acc / n
                let i = (y * width + x) * 4
                for c in 0..<3 { pixels[i + c] = UInt8(max(0, min(255, Float(src[i + c]) * (1 - w) + avg[c] * w + 0.5))) }
            }
        }
        isEdited = true
    }

    /// Removes blemishes by diffusion: the disk under the brush is relaxed toward the
    /// harmonic interpolation of the colors around it (like a Laplace inpaint), so
    /// gradients continue smoothly across the filled area.
    mutating func heal(u: Float, v: Float, radius: Float, iterations: Int = 60) {
        guard let b = bounds(u, v, radius) else { return }
        remember(x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1)
        let cx = u * Float(width), cy = v * Float(height)
        let w = b.x1 - b.x0 + 1, h = b.y1 - b.y0 + 1
        var inside = [Bool](repeating: false, count: w * h)
        var cur = [SIMD3<Float>](repeating: .zero, count: w * h)
        for y in 0..<h { for x in 0..<w {
            let px = b.x0 + x, py = b.y0 + y
            let d = ((Float(px) + 0.5 - cx) * (Float(px) + 0.5 - cx) + (Float(py) + 0.5 - cy) * (Float(py) + 0.5 - cy)).squareRoot()
            inside[y * w + x] = d < radius
            let i = (py * width + px) * 4
            cur[y * w + x] = SIMD3(Float(pixels[i]), Float(pixels[i + 1]), Float(pixels[i + 2]))
        } }
        func sample(_ arr: [SIMD3<Float>], _ x: Int, _ y: Int) -> SIMD3<Float> {
            // Outside the window use the live canvas pixel (the fixed boundary).
            if x >= 0, x < w, y >= 0, y < h { return arr[y * w + x] }
            let px = min(max(b.x0 + x, 0), width - 1), py = min(max(b.y0 + y, 0), height - 1)
            let i = (py * width + px) * 4
            return SIMD3(Float(pixels[i]), Float(pixels[i + 1]), Float(pixels[i + 2]))
        }
        // Start from the mean of the boundary so Jacobi converges quickly.
        var seed = SIMD3<Float>.zero, count: Float = 0
        for y in 0..<h { for x in 0..<w where inside[y * w + x] {
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let nx = x + dx, ny = y + dy
                if nx < 0 || ny < 0 || nx >= w || ny >= h || !inside[ny * w + nx] { seed += sample(cur, nx, ny); count += 1 }
            }
        } }
        guard count > 0 else { return }
        seed /= count
        for i in cur.indices where inside[i] { cur[i] = seed }
        for _ in 0..<iterations {
            var next = cur
            for y in 0..<h { for x in 0..<w where inside[y * w + x] {
                next[y * w + x] = (sample(cur, x + 1, y) + sample(cur, x - 1, y)
                                   + sample(cur, x, y + 1) + sample(cur, x, y - 1)) * 0.25
            } }
            cur = next
        }
        for y in 0..<h { for x in 0..<w where inside[y * w + x] {
            let i = ((b.y0 + y) * width + b.x0 + x) * 4
            for c in 0..<3 { pixels[i + c] = UInt8(max(0, min(255, cur[y * w + x][c] + 0.5))) }
        } }
        isEdited = true
    }

    // MARK: image I/O

    static func load(_ url: URL) -> TextureCanvas? {
        guard let d = MaterialOps.decode(url) else { return nil }
        return TextureCanvas(width: d.width, height: d.height, pixels: d.rgba)
    }

    func write(to url: URL) -> Bool {
        MaterialOps.writePNG(pixels, width: width, height: height, to: url)
    }

    func cgImage() -> CGImage? {
        var px = pixels
        return px.withUnsafeMutableBytes { raw -> CGImage? in
            CGContext(data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                      bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)?.makeImage()
        }
    }
}
