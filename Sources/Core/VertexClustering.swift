import Foundation
import simd

/// Fast display-only simplification: snap vertices to a uniform grid, average each cell, drop the
/// triangles that collapse. Linear time, so it handles multi-million-triangle shapes in seconds where
/// edge-collapse decimation would take far too long. Not watertight-safe — for viewing, never export.
enum VertexClustering {
    private static func keys(_ positions: [SIMD3<Float>], lo: SIMD3<Float>, scale: Float, res: Int) -> [UInt64] {
        let top = Float(res - 1)
        return positions.map { p in
            let c = simd_min(simd_max((p - lo) * scale, SIMD3(repeating: 0)), SIMD3(repeating: top))
            return UInt64(c.x) << 42 | UInt64(c.y) << 21 | UInt64(c.z)
        }
    }

    @_optimize(speed)
    static func simplify(positions: [SIMD3<Float>], faces: [SIMD3<UInt32>],
                         targetFaces: Int) -> (positions: [SIMD3<Float>], faces: [SIMD3<UInt32>]) {
        guard faces.count > targetFaces, targetFaces > 0, let first = positions.first else { return (positions, faces) }
        var lo = first, hi = first
        for p in positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let extent = max(hi.x - lo.x, max(hi.y - lo.y, hi.z - lo.z))
        guard extent > 0 else { return (positions, faces) }

        // A surface's occupied cells grow with res², and faces ≈ 2 × vertices: steer res to the target.
        let wantVerts = Double(max(targetFaces / 2, 1))
        var res = 128
        for _ in 0 ..< 5 {
            let n = Set(keys(positions, lo: lo, scale: Float(res) / extent, res: res)).count
            if abs(Double(n) / wantVerts - 1) < 0.2 { break }
            res = max(8, min(2000, Int(Double(res) * (wantVerts / Double(n)).squareRoot())))
        }

        let cellKeys = keys(positions, lo: lo, scale: Float(res) / extent, res: res)
        var cellOf: [UInt64: UInt32] = [:]
        var sum: [SIMD3<Float>] = [], count: [Float] = []
        var cluster = [UInt32](repeating: 0, count: positions.count)
        for (i, k) in cellKeys.enumerated() {
            let id: UInt32
            if let e = cellOf[k] { id = e } else { id = UInt32(sum.count); cellOf[k] = id; sum.append(.zero); count.append(0) }
            sum[Int(id)] += positions[i]; count[Int(id)] += 1
            cluster[i] = id
        }
        var outFaces: [SIMD3<UInt32>] = []
        outFaces.reserveCapacity(targetFaces * 2)
        for f in faces {
            let a = cluster[Int(f.x)], b = cluster[Int(f.y)], c = cluster[Int(f.z)]
            if a != b && b != c && a != c { outFaces.append([a, b, c]) }
        }
        return (zip(sum, count).map { $0 / $1 }, outFaces)
    }
}
