import Foundation
import simd

/// Turns a TripoSR density grid into a mesh in Modelr's frame.
///
/// TripoSR works in a z-up frame whose camera looks down −x at the object's front (image-right is
/// +y). Modelr is y-up with the front toward +z and image-right toward +x, so a TripoSR point
/// (x, y, z) becomes (y, z, x) — a cyclic permutation, which keeps handedness (and winding).
enum TripoMesh {
    /// `density` is `n³` values over (−radius, radius)³ flattened with x slowest (i, j, k) — the
    /// order `TripoSR.densityGrid` produces. The surface is where density crosses `threshold`.
    static func extract(density: [Float], resolution n: Int, radius: Float, threshold: Float,
                        keepLargestPiece: Bool = true) -> (positions: [SIMD3<Float>], faces: [SIMD3<UInt32>])? {
        guard n >= 4, density.count == n * n * n else { return nil }
        // surface nets expects x fastest and negative inside; density is x slowest and high inside
        var field = [Float](repeating: 0, count: density.count)
        for i in 0 ..< n { for j in 0 ..< n { for k in 0 ..< n {
            field[(k * n + j) * n + i] = threshold - density[(i * n + j) * n + k]
        } } }
        let cell = 2 * radius / Float(n - 1)
        let grid = MeshBoolean.Grid(origin: SIMD3(repeating: -radius), cell: cell, nx: n, ny: n, nz: n)
        let m = MeshBoolean.surfaceNets(field, grid: grid)
        guard !m.faces.isEmpty else { return nil }
        var positions = m.positions.map { SIMD3<Float>($0.y, $0.z, $0.x) }          // TripoSR → Modelr axes
        var faces = m.faces
        if keepLargestPiece { (positions, faces) = largestComponent(positions, faces) }
        return (positions, faces)
    }

    /// Keeps only the connected piece (by shared vertices) with the most triangles, dropping the
    /// floating specks a density threshold can leave behind.
    static func largestComponent(_ positions: [SIMD3<Float>], _ faces: [SIMD3<UInt32>])
    -> ([SIMD3<Float>], [SIMD3<UInt32>]) {
        var parent = Array(0 ..< positions.count)
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
        var counts: [Int: Int] = [:]
        for f in faces { counts[find(Int(f.x)), default: 0] += 1 }
        guard let best = counts.max(by: { $0.value < $1.value })?.key, counts.count > 1 else { return (positions, faces) }
        var remap = [Int32](repeating: -1, count: positions.count)
        var outP: [SIMD3<Float>] = [], outF: [SIMD3<UInt32>] = []
        func map(_ v: UInt32) -> UInt32 {
            if remap[Int(v)] < 0 { remap[Int(v)] = Int32(outP.count); outP.append(positions[Int(v)]) }
            return UInt32(remap[Int(v)])
        }
        for f in faces where find(Int(f.x)) == best { outF.append([map(f.x), map(f.y), map(f.z)]) }
        return (outP, outF)
    }
}
