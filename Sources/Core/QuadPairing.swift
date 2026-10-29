import Foundation
import simd

/// Greedily merges adjacent triangle pairs into quads for quad-based export. Pairs are chosen by
/// shape: nearly coplanar, convex, with corners close to 90°, best first — so a triangulated
/// quad grid (such as a remesh) comes back as its quads, while irregular areas stay triangles.
enum QuadPairing {
    struct Result {
        var quads: [SIMD4<UInt32>]
        var triangles: [SIMD3<UInt32>]
    }

    /// `maxNormalAngle` (degrees) is the largest bend allowed across the merged diagonal.
    static func pair(positions: [SIMD3<Float>], faces: [SIMD3<UInt32>], maxNormalAngle: Float = 25) -> Result {
        // edge (low, high) → faces using it, only keeping manifold edges (exactly two faces)
        var edgeFaces: [UInt64: [Int]] = [:]
        func key(_ a: UInt32, _ b: UInt32) -> UInt64 { UInt64(min(a, b)) << 32 | UInt64(max(a, b)) }
        for (i, f) in faces.enumerated() {
            for (a, b) in [(f.x, f.y), (f.y, f.z), (f.z, f.x)] { edgeFaces[key(a, b), default: []].append(i) }
        }
        func normal(_ f: SIMD3<UInt32>) -> SIMD3<Float> {
            simd_normalize(simd_cross(positions[Int(f.y)] - positions[Int(f.x)], positions[Int(f.z)] - positions[Int(f.x)]))
        }
        let minDot = cos(maxNormalAngle * .pi / 180)

        struct Candidate { var f1: Int; var f2: Int; var quad: SIMD4<UInt32>; var score: Float }
        var candidates: [Candidate] = []
        for (k, fs) in edgeFaces where fs.count == 2 {
            let (i, j) = (fs[0], fs[1])
            let a = UInt32(k >> 32), b = UInt32(k & 0xFFFF_FFFF)
            guard let q = mergedQuad(faces[i], faces[j], a, b) else { continue }
            if simd_dot(normal(faces[i]), normal(faces[j])) < minDot { continue }
            guard let score = quadScore(q, positions) else { continue }
            candidates.append(Candidate(f1: i, f2: j, quad: q, score: score))
        }
        candidates.sort { $0.score != $1.score ? $0.score < $1.score : ($0.f1, $0.f2) < ($1.f1, $1.f2) }

        var used = [Bool](repeating: false, count: faces.count)
        var quads: [SIMD4<UInt32>] = []
        for c in candidates where !used[c.f1] && !used[c.f2] {
            used[c.f1] = true; used[c.f2] = true; quads.append(c.quad)
        }
        let tris = faces.indices.filter { !used[$0] }.map { faces[$0] }
        return Result(quads: quads, triangles: tris)
    }

    /// The quad (consistent winding) formed by two triangles sharing edge `a`–`b`.
    private static func mergedQuad(_ f1: SIMD3<UInt32>, _ f2: SIMD3<UInt32>, _ a: UInt32, _ b: UInt32) -> SIMD4<UInt32>? {
        func opposite(_ f: SIMD3<UInt32>) -> UInt32? { [f.x, f.y, f.z].first { $0 != a && $0 != b } }
        guard let c = opposite(f1), let d = opposite(f2), c != d else { return nil }
        // Orient from f1's winding: if f1 runs a→b the quad is (a, d, b, c); if it runs b→a, swap.
        let v1 = [f1.x, f1.y, f1.z]
        let ia = v1.firstIndex(of: a)!, ib = v1.firstIndex(of: b)!
        let forward = (ia + 1) % 3 == ib                                  // f1 contains a→b
        return forward ? SIMD4(a, d, b, c) : SIMD4(b, d, a, c)
    }

    /// Lower is better: the worst deviation of a corner from 90°. nil for a non-convex or
    /// degenerate quad.
    private static func quadScore(_ q: SIMD4<UInt32>, _ p: [SIMD3<Float>]) -> Float? {
        let v = [p[Int(q.x)], p[Int(q.y)], p[Int(q.z)], p[Int(q.w)]]
        var worst: Float = 0
        var turn: [SIMD3<Float>] = []
        for i in 0..<4 {
            let prev = v[(i + 3) % 4], cur = v[i], next = v[(i + 1) % 4]
            let e1 = prev - cur, e2 = next - cur
            let l1 = simd_length(e1), l2 = simd_length(e2)
            guard l1 > 1e-9, l2 > 1e-9 else { return nil }
            worst = max(worst, abs(simd_dot(e1, e2) / (l1 * l2)))          // |cos| of the corner angle
            turn.append(simd_cross(next - cur, cur - prev))
        }
        // Convex: every corner turns the same way.
        let reference = turn[0]
        for t in turn where simd_dot(t, reference) <= 0 { return nil }
        return worst
    }
}
