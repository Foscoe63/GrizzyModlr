import Foundation

extension MarchingCubes {
    /// Marching cubes over a block-sparse SDF grid. Same cells, tables and per-edge vertex sharing as
    /// `extract(grid:)`, but walks only allocated blocks and keeps vertex ids in block-sparse pools
    /// (no dictionary, no n³ keys), so it works at lattices far beyond what a dense array can hold.
    /// Cells with any NaN corner (band edge) emit nothing, like the dense version.
    @_optimize(speed)
    static func extract(sparse g: SparseSDFGrid, level: Float = 0.0,
                        bboxMin: Float = -1.01, bboxMax: Float = 1.01,
                        isCancelled: () -> Bool = { false }) -> Mesh? {
        let n = g.n
        let scale = (bboxMax - bboxMin) / Float(n - 1)
        let edgeCorners = MCTables.edgeCorners
        let triTable = MCTables.triTable
        // Vertex ids per edge direction, indexed like the grid data (lower corner of the edge).
        let none = UInt32.max
        var vid: [[UInt32]] = [
            [UInt32](repeating: none, count: g.data.count),
            [UInt32](repeating: none, count: g.data.count),
            [UInt32](repeating: none, count: g.data.count),
        ]
        var verts: [SIMD3<Float>] = []
        var faces: [(UInt32, UInt32, UInt32)] = []
        var halo = [Float](repeating: .nan, count: 9 * 9 * 9)
        var cv = [Float](repeating: 0, count: 8)

        for s in 0 ..< g.slots {
            if s & 0x3FF == 0, isCancelled() { return nil }
            let (bi, bj, bk) = g.origin(of: s)
            // Block plus a one-cell halo on the +x/+y/+z sides, so every cell reads its 8 corners locally.
            for di in 0 ... 1 { for dj in 0 ... 1 { for dk in 0 ... 1 {
                let ni = bi + di * sparseBlockEdge, nj = bj + dj * sparseBlockEdge, nk = bk + dk * sparseBlockEdge
                let ns = (ni < n && nj < n && nk < n) ? g.slot(ni, nj, nk) : -1
                let ra = di == 0 ? 0 ..< 8 : 8 ..< 9, rb = dj == 0 ? 0 ..< 8 : 8 ..< 9, rc = dk == 0 ? 0 ..< 8 : 8 ..< 9
                for a in ra { for b in rb { for c in rc {
                    halo[(a * 9 + b) * 9 + c] = ns < 0 ? .nan
                        : g.data[ns * sparseBlockCells + (((a & 7) << 6) | ((b & 7) << 3) | (c & 7))]
                } } }
            } } }
            for a in 0 ..< sparseBlockEdge {
                let i = bi + a
                if i >= n - 1 { break }
                for b in 0 ..< sparseBlockEdge {
                    let j = bj + b
                    if j >= n - 1 { break }
                    for c in 0 ..< sparseBlockEdge {
                        let k = bk + c
                        if k >= n - 1 { break }
                        var config = 0
                        var hasNaN = false
                        for corner in 0 ..< 8 {
                            let v = halo[((a + (corner & 1)) * 9 + (b + ((corner >> 1) & 1))) * 9 + (c + ((corner >> 2) & 1))]
                            if v.isNaN { hasNaN = true; break }
                            cv[corner] = v
                            if v < level { config |= 1 << corner }
                        }
                        if hasNaN || config == 0 || config == 255 { continue }
                        let tris = triTable[config]
                        if tris.isEmpty { continue }
                        var m = 0
                        while m < tris.count {
                            var ids: (UInt32, UInt32, UInt32) = (0, 0, 0)
                            for t in 0 ..< 3 {
                                let (lo, hi) = edgeCorners[tris[m + t]]
                                let axis = (hi - lo) == 1 ? 0 : ((hi - lo) == 2 ? 1 : 2)
                                let li = i + (lo & 1), lj = j + ((lo >> 1) & 1), lk = k + ((lo >> 2) & 1)
                                let at = g.slot(li, lj, lk) * sparseBlockCells + SparseBlocks<Float>.offset(li, lj, lk)
                                var id = vid[axis][at]
                                if id == none {
                                    let tt = (level - cv[lo]) / (cv[hi] - cv[lo])
                                    var p = SIMD3<Float>(Float(li), Float(lj), Float(lk))
                                    p[axis] += tt
                                    id = UInt32(verts.count)
                                    verts.append(p * scale + SIMD3(repeating: bboxMin))
                                    vid[axis][at] = id
                                }
                                if t == 0 { ids.0 = id } else if t == 1 { ids.1 = id } else { ids.2 = id }
                            }
                            faces.append(ids)
                            m += 3
                        }
                    }
                }
            }
        }
        return Mesh(vertices: verts, faces: faces)
    }
}
