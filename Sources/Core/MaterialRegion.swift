import Foundation
import simd

/// A region of the texture picked by color: texels near `color` (straight sRGB 0…1), with a
/// soft edge. Used to limit a material edit to, say, "just the leather" or "just the metal".
struct ColorRegion: Equatable {
    var color: SIMD3<Float>
    /// Distance (0…1, RGB space) inside which a texel is fully selected.
    var tolerance: Float = 0.12
    /// Extra distance over which the selection fades out.
    var feather: Float = 0.10
}

enum MaterialRegion {
    /// Up to `k` dominant colors, largest first. Deterministic k-means on a strided sample;
    /// transparent texels are ignored and clusters closer than `minSeparation` are merged.
    static func dominantColors(_ rgba: [UInt8], k: Int = 6, minSeparation: Float = 0.12) -> [SIMD3<Float>] {
        let count = rgba.count / 4
        guard count > 0 else { return [] }
        let stride = max(1, count / 4096)
        var samples: [SIMD3<Float>] = []
        var i = 0
        while i < count {
            if rgba[i * 4 + 3] > 127 {
                samples.append(SIMD3(Float(rgba[i * 4]), Float(rgba[i * 4 + 1]), Float(rgba[i * 4 + 2])) / 255)
            }
            i += stride
        }
        guard !samples.isEmpty else { return [] }

        // Seed with farthest-point picks from the first sample (deterministic).
        var centers = [samples[0]]
        while centers.count < min(k, samples.count) {
            var best = samples[0], bestDist: Float = -1
            for s in samples {
                let d = centers.map { simd_distance(s, $0) }.min() ?? 0
                if d > bestDist { bestDist = d; best = s }
            }
            if bestDist < minSeparation { break }
            centers.append(best)
        }
        var sizes = [Int](repeating: 0, count: centers.count)
        for _ in 0..<12 {
            var sums = [SIMD3<Float>](repeating: .zero, count: centers.count)
            sizes = [Int](repeating: 0, count: centers.count)
            for s in samples {
                var bi = 0, bd = Float.greatestFiniteMagnitude
                for (ci, c) in centers.enumerated() {
                    let d = simd_distance_squared(s, c)
                    if d < bd { bd = d; bi = ci }
                }
                sums[bi] += s; sizes[bi] += 1
            }
            for ci in centers.indices where sizes[ci] > 0 { centers[ci] = sums[ci] / Float(sizes[ci]) }
        }
        var ranked = Array(zip(centers, sizes)).filter { $0.1 > 0 }.sorted { $0.1 > $1.1 }.map(\.0)
        var merged: [SIMD3<Float>] = []
        for c in ranked where !merged.contains(where: { simd_distance($0, c) < minSeparation }) { merged.append(c) }
        ranked = merged
        return ranked
    }

    /// Per-texel selection weight 0…1 for `region` over straight RGBA8 pixels.
    static func mask(_ rgba: [UInt8], region: ColorRegion) -> [Float] {
        let count = rgba.count / 4
        var out = [Float](repeating: 0, count: count)
        let inner = max(0, region.tolerance), outer = inner + max(0.0001, region.feather)
        for i in 0..<count {
            if rgba[i * 4 + 3] == 0 { continue }
            let c = SIMD3(Float(rgba[i * 4]), Float(rgba[i * 4 + 1]), Float(rgba[i * 4 + 2])) / 255
            let d = simd_distance(c, region.color)
            if d <= inner { out[i] = 1 } else if d < outer {
                let t = 1 - (d - inner) / (outer - inner)
                out[i] = t * t * (3 - 2 * t)
            }
        }
        return out
    }

    /// Fraction of opaque texels (0…1) with weight > 0.5.
    static func coverage(of mask: [Float], rgba: [UInt8]) -> Float {
        var on = 0, total = 0
        for i in 0..<mask.count where rgba[i * 4 + 3] > 0 {
            total += 1
            if mask[i] > 0.5 { on += 1 }
        }
        return total == 0 ? 0 : Float(on) / Float(total)
    }

    /// `original` where weight is 0, `edited` where it is 1, blended between. Alpha stays original.
    /// `mask` may come from an image of different size; it is sampled nearest-neighbour.
    static func blend(original: [UInt8], edited: inout [UInt8], mask: [Float],
                      maskSize: (w: Int, h: Int), size: (w: Int, h: Int)) {
        guard maskSize.w > 0, maskSize.h > 0, mask.count == maskSize.w * maskSize.h else { return }
        for y in 0..<size.h {
            let my = min(maskSize.h - 1, y * maskSize.h / size.h)
            for x in 0..<size.w {
                let mx = min(maskSize.w - 1, x * maskSize.w / size.w)
                let w = mask[my * maskSize.w + mx]
                if w >= 1 { continue }
                let p = (y * size.w + x) * 4
                for c in 0..<3 {
                    edited[p + c] = UInt8(Float(original[p + c]) * (1 - w) + Float(edited[p + c]) * w + 0.5)
                }
                edited[p + 3] = original[p + 3]
            }
        }
    }
}
