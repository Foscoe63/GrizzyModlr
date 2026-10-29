import Foundation
import simd

/// A rigid 2D transform of one UV island: rotate and scale about its center, then translate.
struct IslandTransform: Equatable {
    var du: Float = 0
    var dv: Float = 0
    var scale: Float = 1
    /// Radians, counter-clockwise in UV space.
    var rotation: Float = 0
    var isIdentity: Bool { du == 0 && dv == 0 && scale == 1 && rotation == 0 }
}

/// UV island inspection and editing for a textured mesh. An island is a set of faces
/// connected through shared vertex indices (the unwrapper splits vertices along seams,
/// so charts are exactly the connected components).
enum UVLayout {
    struct Island: Identifiable {
        let id: Int
        var faces: [Int]
        var vertices: [Int]
        var lo: SIMD2<Float>
        var hi: SIMD2<Float>
        var center: SIMD2<Float> { (lo + hi) / 2 }
    }

    static func islands(faces: [SIMD3<UInt32>], uvs: [SIMD2<Float>]) -> [Island] {
        var parent = Array(0..<uvs.count)
        func find(_ x: Int) -> Int {
            var r = x
            while parent[r] != r { r = parent[r] }
            var c = x
            while parent[c] != r { let n = parent[c]; parent[c] = r; c = n }
            return r
        }
        for f in faces {
            let a = find(Int(f.x)), b = find(Int(f.y)), c = find(Int(f.z))
            parent[b] = a; parent[find(c)] = a
        }
        var byRoot: [Int: Int] = [:]
        var result: [Island] = []
        for (fi, f) in faces.enumerated() {
            let root = find(Int(f.x))
            let idx: Int
            if let i = byRoot[root] { idx = i } else {
                idx = result.count; byRoot[root] = idx
                result.append(Island(id: idx, faces: [], vertices: [],
                                     lo: SIMD2(repeating: .infinity), hi: SIMD2(repeating: -.infinity)))
            }
            result[idx].faces.append(fi)
        }
        for i in result.indices {
            var seen = Set<Int>()
            for fi in result[i].faces {
                let f = faces[fi]
                for v in [Int(f.x), Int(f.y), Int(f.z)] where seen.insert(v).inserted {
                    result[i].vertices.append(v)
                    result[i].lo = simd_min(result[i].lo, uvs[v]); result[i].hi = simd_max(result[i].hi, uvs[v])
                }
            }
        }
        return result
    }

    /// Edges used by exactly one face of the island — its outline.
    static func outline(of island: Island, faces: [SIMD3<UInt32>]) -> [(Int, Int)] {
        var count: [UInt64: Int] = [:]
        var order: [(UInt32, UInt32)] = []
        func key(_ a: UInt32, _ b: UInt32) -> UInt64 { UInt64(min(a, b)) << 32 | UInt64(max(a, b)) }
        for fi in island.faces {
            let f = faces[fi]
            for (a, b) in [(f.x, f.y), (f.y, f.z), (f.z, f.x)] {
                let k = key(a, b)
                if count[k] == nil { order.append((a, b)) }
                count[k, default: 0] += 1
            }
        }
        return order.filter { count[key($0.0, $0.1)] == 1 }.map { (Int($0.0), Int($0.1)) }
    }

    /// UVs after applying the per-island transforms.
    static func apply(_ transforms: [Int: IslandTransform], islands: [Island],
                      to uvs: [SIMD2<Float>]) -> [SIMD2<Float>] {
        var out = uvs
        for island in islands {
            guard let t = transforms[island.id], !t.isIdentity else { continue }
            let c = island.center
            let cs = cos(t.rotation), sn = sin(t.rotation)
            for v in island.vertices {
                let p = (uvs[v] - c) * t.scale
                out[v] = c + SIMD2(p.x * cs - p.y * sn, p.x * sn + p.y * cs) + SIMD2(t.du, t.dv)
            }
        }
        return out
    }

    /// Whether any transformed island's bounds leave the 0…1 square.
    static func leavesUnitSquare(_ uvs: [SIMD2<Float>], islands: [Island], changed: Set<Int>) -> Bool {
        for island in islands where changed.contains(island.id) {
            for v in island.vertices where uvs[v].x < 0 || uvs[v].y < 0 || uvs[v].x > 1 || uvs[v].y > 1 { return true }
        }
        return false
    }

    /// Whether any changed island now covers texels used by a different island
    /// (checked on a coarse `size`² occupancy grid).
    static func overlapsOthers(faces: [SIMD3<UInt32>], uvs: [SIMD2<Float>], islands: [Island],
                               changed: Set<Int>, size: Int = 256) -> Bool {
        var owner = [Int32](repeating: -1, count: size * size)
        var clash = false
        // Others first, then changed islands, so a clash is any changed texel that is taken.
        for pass in 0..<2 {
            for island in islands where changed.contains(island.id) == (pass == 1) {
                for fi in island.faces {
                    let f = faces[fi]
                    raster(uvs[Int(f.x)], uvs[Int(f.y)], uvs[Int(f.z)], w: size, h: size, tolerance: 0) { x, y, _ in
                        let i = y * size + x
                        if pass == 1, owner[i] >= 0, owner[i] != Int32(island.id) { clash = true }
                        if owner[i] < 0 { owner[i] = Int32(island.id) }
                    }
                }
            }
        }
        return clash
    }

    // MARK: texture remap

    /// Rewrites `source` so every face in `faceIDs` shows the same image at its new UVs.
    /// Texels outside the moved faces are untouched; the moved faces' old footprint is kept
    /// (it may be reused by another island or simply left as unused texture).
    static func remap(_ source: TextureCanvas, faces: [SIMD3<UInt32>], faceIDs: [Int],
                      oldUVs: [SIMD2<Float>], newUVs: [SIMD2<Float>]) -> TextureCanvas {
        var out = source.pixels
        let w = source.width, h = source.height
        for fi in faceIDs {
            let f = faces[fi]
            let (a, b, c) = (Int(f.x), Int(f.y), Int(f.z))
            raster(newUVs[a], newUVs[b], newUVs[c], w: w, h: h, tolerance: 0.03) { x, y, bary in
                let old = oldUVs[a] * bary.x + oldUVs[b] * bary.y + oldUVs[c] * bary.z
                let px = bilinear(source, old.x * Float(w) - 0.5, old.y * Float(h) - 0.5)
                let i = (y * w + x) * 4
                for k in 0..<4 { out[i + k] = px[k] }
            }
        }
        return TextureCanvas(width: w, height: h, pixels: out)
    }

    /// Calls `body(x, y, barycentric)` for each pixel center inside the UV triangle
    /// (`tolerance` extends coverage slightly past the edges to avoid hairline gaps).
    private static func raster(_ a: SIMD2<Float>, _ b: SIMD2<Float>, _ c: SIMD2<Float>, w: Int, h: Int,
                               tolerance: Float, _ body: (Int, Int, SIMD3<Float>) -> Void) {
        let pa = SIMD2(a.x * Float(w), a.y * Float(h)), pb = SIMD2(b.x * Float(w), b.y * Float(h)),
            pc = SIMD2(c.x * Float(w), c.y * Float(h))
        let lo = simd_min(pa, simd_min(pb, pc)), hi = simd_max(pa, simd_max(pb, pc))
        let x0 = max(0, Int(lo.x.rounded(.down)) - 1), x1 = min(w - 1, Int(hi.x.rounded(.up)) + 1)
        let y0 = max(0, Int(lo.y.rounded(.down)) - 1), y1 = min(h - 1, Int(hi.y.rounded(.up)) + 1)
        guard x0 <= x1, y0 <= y1 else { return }
        let det = (pb.y - pc.y) * (pa.x - pc.x) + (pc.x - pb.x) * (pa.y - pc.y)
        guard abs(det) > 1e-9 else { return }
        for y in y0...y1 { for x in x0...x1 {
            let p = SIMD2(Float(x) + 0.5, Float(y) + 0.5)
            let l0 = ((pb.y - pc.y) * (p.x - pc.x) + (pc.x - pb.x) * (p.y - pc.y)) / det
            let l1 = ((pc.y - pa.y) * (p.x - pc.x) + (pa.x - pc.x) * (p.y - pc.y)) / det
            let l2 = 1 - l0 - l1
            if l0 >= -tolerance, l1 >= -tolerance, l2 >= -tolerance { body(x, y, SIMD3(l0, l1, l2)) }
        } }
    }

    private static func bilinear(_ t: TextureCanvas, _ fx: Float, _ fy: Float) -> [UInt8] {
        let x0 = Int(fx.rounded(.down)), y0 = Int(fy.rounded(.down))
        let tx = fx - Float(x0), ty = fy - Float(y0)
        func at(_ x: Int, _ y: Int) -> SIMD4<Float> {
            let i = (min(max(y, 0), t.height - 1) * t.width + min(max(x, 0), t.width - 1)) * 4
            return SIMD4(Float(t.pixels[i]), Float(t.pixels[i + 1]), Float(t.pixels[i + 2]), Float(t.pixels[i + 3]))
        }
        let top = at(x0, y0) * (1 - tx) + at(x0 + 1, y0) * tx
        let bot = at(x0, y0 + 1) * (1 - tx) + at(x0 + 1, y0 + 1) * tx
        let v = top * (1 - ty) + bot * ty
        return (0..<4).map { UInt8(max(0, min(255, v[$0] + 0.5))) }
    }
}
