import Foundation
import simd

/// Builds a watertight relief solid from a depth map: the front surface is displaced by depth,
/// the back is flat, and walls close the silhouette. A fast 2.5D alternative to a full shape
/// generation, and a valid input for painting.
enum ReliefMesh {
    struct Options {
        /// Relief height as a fraction of the larger image side.
        var reliefDepth: Double = 0.25
        /// Minimum solid thickness (same units as the larger side = 2).
        var baseThickness: Double = 0.06
        /// Box-blur passes applied to the depth map before meshing.
        var smoothing: Int = 2
    }

    /// Rescales a raw (relative inverse) depth map to 0…1 using robust percentiles of the
    /// masked pixels (2nd…98th), so a few outliers don't flatten the relief.
    static func normalize(_ depth: [Float], mask: [Bool]?) -> [Float] {
        let samples = zip(depth, mask ?? [Bool](repeating: true, count: depth.count)).filter { $0.1 }.map { $0.0 }.sorted()
        guard samples.count > 8 else { return depth.map { _ in 0.5 } }
        let lo = samples[Int(Double(samples.count - 1) * 0.02)], hi = samples[Int(Double(samples.count - 1) * 0.98)]
        let span = max(hi - lo, 1e-6)
        return depth.map { min(max(($0 - lo) / span, 0), 1) }
    }

    static func blur(_ d: [Float], width: Int, height: Int, passes: Int) -> [Float] {
        var cur = d
        for _ in 0 ..< max(passes, 0) {
            var next = cur
            for y in 0 ..< height { for x in 0 ..< width {
                var acc: Float = 0, n: Float = 0
                for dy in -1 ... 1 { for dx in -1 ... 1 {
                    let xx = x + dx, yy = y + dy
                    if xx >= 0, yy >= 0, xx < width, yy < height { acc += cur[yy * width + xx]; n += 1 }
                } }
                next[y * width + x] = acc / n
            } }
            cur = next
        }
        return cur
    }

    /// `depth` is normalized 0…1 (1 = nearest), `mask` marks the object (nil = everything).
    /// X spans the grid width, Y is up (image rows run downward), Z is toward the viewer; the
    /// larger image side spans 2 units.
    static func build(depth: [Float], mask: [Bool]?, width: Int, height: Int, options: Options = Options())
    -> (positions: [SIMD3<Float>], faces: [SIMD3<UInt32>])? {
        guard width >= 2, height >= 2, depth.count == width * height else { return nil }
        let d = blur(depth, width: width, height: height, passes: options.smoothing)
        let inside = mask ?? [Bool](repeating: true, count: width * height)
        let unit = 2 / Float(max(width, height))
        let relief = Float(options.reliefDepth) * 2
        let base = Float(options.baseThickness)

        // front vertices exist only where a kept cell touches them
        var cellKept = [Bool](repeating: false, count: (width - 1) * (height - 1))
        for y in 0 ..< height - 1 { for x in 0 ..< width - 1 {
            cellKept[y * (width - 1) + x] = inside[y * width + x] && inside[y * width + x + 1]
                && inside[(y + 1) * width + x] && inside[(y + 1) * width + x + 1]
        } }
        var frontIndex = [Int32](repeating: -1, count: width * height)
        var positions: [SIMD3<Float>] = []
        func front(_ x: Int, _ y: Int) -> UInt32 {
            let i = y * width + x
            if frontIndex[i] < 0 {
                frontIndex[i] = Int32(positions.count)
                positions.append(SIMD3((Float(x) - Float(width - 1) / 2) * unit, (Float(height - 1) / 2 - Float(y)) * unit,
                                       base + d[i] * relief))
            }
            return UInt32(frontIndex[i])
        }
        var frontFaces: [SIMD3<UInt32>] = []
        for y in 0 ..< height - 1 { for x in 0 ..< width - 1 where cellKept[y * (width - 1) + x] {
            let a = front(x, y), b = front(x + 1, y), c = front(x, y + 1), e = front(x + 1, y + 1)
            // image rows run downward, so (a, c, b) is counter-clockwise seen from +Z
            frontFaces.append([a, c, b]); frontFaces.append([b, c, e])
        } }
        guard !frontFaces.isEmpty else { return nil }

        // flat back: same vertices at z = 0, reversed winding
        let frontCount = UInt32(positions.count)
        for i in 0 ..< Int(frontCount) { positions.append(SIMD3(positions[i].x, positions[i].y, 0)) }
        var faces = frontFaces
        for f in frontFaces { faces.append([f.x + frontCount, f.z + frontCount, f.y + frontCount]) }

        // walls on boundary edges (used by exactly one front triangle)
        var uses: [UInt64: Int] = [:]
        func key(_ a: UInt32, _ b: UInt32) -> UInt64 { UInt64(a) << 32 | UInt64(b) }
        for f in frontFaces { for (a, b) in [(f.x, f.y), (f.y, f.z), (f.z, f.x)] { uses[key(a, b), default: 0] += 1 } }
        for (k, _) in uses {
            let a = UInt32(k >> 32), b = UInt32(k & 0xFFFF_FFFF)
            guard uses[key(b, a)] == nil else { continue }               // interior edge (seen both ways)
            faces.append([a, a + frontCount, b]); faces.append([b, a + frontCount, b + frontCount])
        }
        return (positions, faces)
    }

    /// Output grid size for an image of `w × h` pixels with `resolution` samples on the long side.
    static func gridSize(imageWidth w: Int, imageHeight h: Int, resolution: Int) -> (Int, Int) {
        let longest = max(w, h)
        let s = Double(resolution) / Double(max(longest, 1))
        return (max(Int((Double(w) * s).rounded()), 2), max(Int((Double(h) * s).rounded()), 2))
    }
}
