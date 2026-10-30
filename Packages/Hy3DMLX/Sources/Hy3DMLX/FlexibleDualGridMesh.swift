import Foundation

/// TRELLIS.2 flexible dual grid → triangle mesh (CPU), after o-voxel `flexible_dual_grid_to_mesh` (inference path).
/// Each active voxel owns one dual vertex; every voxel edge flagged as intersected joins the four voxels around it
/// into a quad (dropped unless all four are active), split into two triangles by the predicted quad weights.
enum FlexibleDualGridMesh {
    struct Mesh {
        var vertices: [Float]   // xyz, in [-0.5, 0.5]
        var faces: [Int32]      // triangle index triples
    }

    /// Edge-neighbour voxel offsets per axis (x, y, z), four voxels around the edge.
    private static let offsets: [[(Int32, Int32, Int32)]] = [
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    ]

    /// - Parameters:
    ///   - coords: N×4 int32 (batch, x, y, z) of active voxels, batch 0 only.
    ///   - feats: N×7 decoder output (3 vertex-offset logits, 3 intersection logits, 1 quad-weight logit).
    ///   - grid: lattice edge length of the final voxels (aabb is the unit cube centred at the origin).
    static func extract(coords: [Int32], feats: [Float], grid: Int) -> Mesh {
        let n = coords.count / 4
        let g = Int32(grid)
        let voxel = 1 / Float(grid)
        var verts = [Float](repeating: 0, count: n * 3)
        var sw = [Float](repeating: 0, count: n)
        var table = [Int64: Int32](minimumCapacity: n * 2)
        let s = Int64(grid + 2)
        @inline(__always) func key(_ x: Int32, _ y: Int32, _ z: Int32) -> Int64 { (Int64(x) * s + Int64(y)) * s + Int64(z) }
        for i in 0 ..< n {
            let x = coords[i * 4 + 1], y = coords[i * 4 + 2], z = coords[i * 4 + 3]
            table[key(x, y, z)] = Int32(i)
            let c = [x, y, z]
            for a in 0 ..< 3 {
                let dual = 2 / (1 + exp(-feats[i * 7 + a])) - 0.5
                verts[i * 3 + a] = (Float(c[a]) + dual) * voxel - 0.5
            }
            let t = feats[i * 7 + 6]
            sw[i] = t > 20 ? t : log1p(exp(t))     // softplus
        }
        var faces: [Int32] = []
        faces.reserveCapacity(n * 6)
        var quad = [Int32](repeating: 0, count: 4)
        for i in 0 ..< n {
            let x = coords[i * 4 + 1], y = coords[i * 4 + 2], z = coords[i * 4 + 3]
            for a in 0 ..< 3 where feats[i * 7 + 3 + a] > 0 {
                var ok = true
                for (k, o) in offsets[a].enumerated() {
                    let nx = x + o.0, ny = y + o.1, nz = z + o.2
                    if nx >= g || ny >= g || nz >= g { ok = false; break }
                    guard let idx = table[key(nx, ny, nz)] else { ok = false; break }
                    quad[k] = idx
                }
                if !ok { continue }
                let w02 = sw[Int(quad[0])] * sw[Int(quad[2])], w13 = sw[Int(quad[1])] * sw[Int(quad[3])]
                if w02 > w13 { faces += [quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]] }
                else { faces += [quad[0], quad[1], quad[3], quad[3], quad[1], quad[2]] }
            }
        }
        return Mesh(vertices: verts, faces: faces)
    }
}
