import Foundation
import simd

enum BooleanOp: String, CaseIterable, Identifiable {
    case union, subtract, intersect
    var id: String { rawValue }
    var label: String {
        switch self {
        case .union: return "Union"
        case .subtract: return "Subtract"
        case .intersect: return "Intersect"
        }
    }
}

enum BooleanPrimitive: String, CaseIterable, Identifiable {
    case sphere, box, cylinder
    var id: String { rawValue }
    var label: String { rawValue.capitalized }
}

/// Mesh booleans through signed distance fields: each operand is sampled on a shared
/// voxel grid (negative = inside), combined with min / max, and re-meshed with surface
/// nets. Resolution-limited (features finer than a voxel are lost) but robust — it
/// needs no exact triangle intersection and tolerates slightly open meshes.
enum MeshBoolean {
    struct Grid {
        var origin: SIMD3<Float>
        var cell: Float
        var nx: Int, ny: Int, nz: Int
        var count: Int { nx * ny * nz }
        @inline(__always) func index(_ x: Int, _ y: Int, _ z: Int) -> Int { (z * ny + y) * nx + x }
        func position(_ x: Int, _ y: Int, _ z: Int) -> SIMD3<Float> {
            origin + SIMD3(Float(x), Float(y), Float(z)) * cell
        }
    }

    /// A grid covering `lo…hi` with `resolution` cells along the longest axis plus padding.
    static func grid(lo: SIMD3<Float>, hi: SIMD3<Float>, resolution: Int, pad: Int = 4) -> Grid {
        let size = hi - lo
        let cell = max(max(size.x, size.y), size.z) / Float(max(resolution, 8))
        let origin = lo - SIMD3(repeating: cell * Float(pad))
        func n(_ s: Float) -> Int { Int((s / cell).rounded(.up)) + 2 * pad + 1 }
        return Grid(origin: origin, cell: cell, nx: n(size.x), ny: n(size.y), nz: n(size.z))
    }

    // MARK: operand fields

    /// Signed distance of a closed triangle mesh, exact within `band` voxels of the surface
    /// and clamped beyond it (the sign everywhere comes from ray parity along +x).
    static func meshField(positions: [SIMD3<Float>], faces: [SIMD3<UInt32>], grid g: Grid,
                          band: Int = 3) -> [Float] {
        let clamp = Float(band) * g.cell
        var dist = [Float](repeating: clamp, count: g.count)
        // 1) unsigned distance, splatted per triangle into its dilated bounding box
        for f in faces {
            let a = positions[Int(f.x)], b = positions[Int(f.y)], c = positions[Int(f.z)]
            let lo = simd_min(a, simd_min(b, c)) - SIMD3(repeating: clamp)
            let hi = simd_max(a, simd_max(b, c)) + SIMD3(repeating: clamp)
            let i0 = max(0, Int(((lo.x - g.origin.x) / g.cell).rounded(.down))), i1 = min(g.nx - 1, Int(((hi.x - g.origin.x) / g.cell).rounded(.up)))
            let j0 = max(0, Int(((lo.y - g.origin.y) / g.cell).rounded(.down))), j1 = min(g.ny - 1, Int(((hi.y - g.origin.y) / g.cell).rounded(.up)))
            let k0 = max(0, Int(((lo.z - g.origin.z) / g.cell).rounded(.down))), k1 = min(g.nz - 1, Int(((hi.z - g.origin.z) / g.cell).rounded(.up)))
            guard i0 <= i1, j0 <= j1, k0 <= k1 else { continue }
            for k in k0...k1 { for j in j0...j1 { for i in i0...i1 {
                let d = pointTriangleDistance(g.position(i, j, k), a, b, c)
                let idx = g.index(i, j, k)
                if d < dist[idx] { dist[idx] = d }
            } } }
        }
        // 2) inside/outside by parity of +x ray crossings per (y, z) column
        var crossings = [[Float]](repeating: [], count: g.ny * g.nz)
        let eps = SIMD2<Float>(1.0e-3, 1.7e-3) * g.cell              // dodge exact edge hits
        for f in faces {
            let a = positions[Int(f.x)], b = positions[Int(f.y)], c = positions[Int(f.z)]
            let lo = simd_min(a, simd_min(b, c)), hi = simd_max(a, simd_max(b, c))
            // Slack of 0.01 cell keeps triangles whose edge lies on a grid line (float error
            // would otherwise drop them); the barycentric test still decides the hit.
            let j0 = max(0, Int((((lo.y - g.origin.y) / g.cell) - 0.01).rounded(.up))), j1 = min(g.ny - 1, Int((((hi.y - g.origin.y) / g.cell) + 0.01).rounded(.down)))
            let k0 = max(0, Int((((lo.z - g.origin.z) / g.cell) - 0.01).rounded(.up))), k1 = min(g.nz - 1, Int((((hi.z - g.origin.z) / g.cell) + 0.01).rounded(.down)))
            guard j0 <= j1, k0 <= k1 else { continue }
            let det = (b.y - a.y) * (c.z - a.z) - (c.y - a.y) * (b.z - a.z)
            guard abs(det) > 1e-20 else { continue }                   // edge-on to the ray
            for k in k0...k1 { for j in j0...j1 {
                let py = g.origin.y + Float(j) * g.cell + eps.x, pz = g.origin.z + Float(k) * g.cell + eps.y
                let u = ((py - a.y) * (c.z - a.z) - (c.y - a.y) * (pz - a.z)) / det
                let v = ((b.y - a.y) * (pz - a.z) - (py - a.y) * (b.z - a.z)) / det
                guard u >= 0, v >= 0, u + v <= 1 else { continue }
                crossings[k * g.ny + j].append(a.x + u * (b.x - a.x) + v * (c.x - a.x))
            } }
        }
        for k in 0..<g.nz { for j in 0..<g.ny {
            let xs = crossings[k * g.ny + j].sorted()
            var p = 0, inside = false
            for i in 0..<g.nx {
                let x = g.origin.x + Float(i) * g.cell
                while p < xs.count, xs[p] <= x { inside.toggle(); p += 1 }
                if inside { dist[g.index(i, j, k)] = -dist[g.index(i, j, k)] }
            }
        } }
        return dist
    }

    /// Analytic SDF of a primitive centered at `center`; `size` is the radius / half-extent.
    static func primitiveField(_ shape: BooleanPrimitive, center: SIMD3<Float>, size: Float, grid g: Grid) -> [Float] {
        var out = [Float](repeating: 0, count: g.count)
        for k in 0..<g.nz { for j in 0..<g.ny { for i in 0..<g.nx {
            let p = g.position(i, j, k) - center
            let d: Float
            switch shape {
            case .sphere:
                d = simd_length(p) - size
            case .box:
                let q = simd_abs(p) - SIMD3(repeating: size)
                d = simd_length(simd_max(q, .zero)) + min(max(q.x, max(q.y, q.z)), 0)
            case .cylinder:                                   // axis = Y, half-height = size
                let q = SIMD2(simd_length(SIMD2(p.x, p.z)) - size, abs(p.y) - size)
                d = min(max(q.x, q.y), 0) + simd_length(simd_max(q, .zero))
            }
            out[g.index(i, j, k)] = d
        } } }
        return out
    }

    static func combine(_ a: [Float], _ b: [Float], _ op: BooleanOp) -> [Float] {
        var out = a
        for i in a.indices {
            switch op {
            case .union: out[i] = min(a[i], b[i])
            case .intersect: out[i] = max(a[i], b[i])
            case .subtract: out[i] = max(a[i], -b[i])
            }
        }
        return out
    }

    // MARK: surface nets

    /// Extracts the zero level set (negative = inside) as an outward-wound triangle mesh.
    static func surfaceNets(_ f: [Float], grid g: Grid) -> (positions: [SIMD3<Float>], faces: [SIMD3<UInt32>]) {
        let cx = g.nx - 1, cy = g.ny - 1, cz = g.nz - 1
        var cellVertex = [Int32](repeating: -1, count: cx * cy * cz)
        var positions: [SIMD3<Float>] = []
        func cid(_ x: Int, _ y: Int, _ z: Int) -> Int { (z * cy + y) * cx + x }
        let corners: [(Int, Int, Int)] = [(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)]
        let edges: [(Int, Int)] = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
        var v = [Float](repeating: 0, count: 8)                       // reused: no per-cell allocation
        for z in 0..<cz { for y in 0..<cy { for x in 0..<cx {
            var neg = 0
            for (n, o) in corners.enumerated() {
                v[n] = f[g.index(x + o.0, y + o.1, z + o.2)]
                if v[n] < 0 { neg += 1 }
            }
            guard neg != 0, neg != 8 else { continue }
            var sum = SIMD3<Float>.zero, n = 0
            for (a, b) in edges where (v[a] < 0) != (v[b] < 0) {
                let t = v[a] / (v[a] - v[b])
                let pa = SIMD3(Float(corners[a].0), Float(corners[a].1), Float(corners[a].2))
                let pb = SIMD3(Float(corners[b].0), Float(corners[b].1), Float(corners[b].2))
                sum += pa + (pb - pa) * t; n += 1
            }
            cellVertex[cid(x, y, z)] = Int32(positions.count)
            positions.append(g.position(x, y, z) + sum / Float(n) * g.cell)
        } } }

        var faces: [SIMD3<UInt32>] = []
        func quad(_ a: Int32, _ b: Int32, _ c: Int32, _ d: Int32, flip: Bool) {
            guard a >= 0, b >= 0, c >= 0, d >= 0 else { return }
            let (A, B, C, D) = (UInt32(a), UInt32(b), UInt32(c), UInt32(d))
            if flip { faces.append([A, C, B]); faces.append([A, D, C]) }
            else { faces.append([A, B, C]); faces.append([A, C, D]) }
        }
        // One quad per grid edge crossing the surface, joining the 4 cells around it.
        for z in 1..<max(g.nz - 1, 1) { for y in 1..<max(g.ny - 1, 1) { for x in 1..<max(g.nx - 1, 1) {
            let here = f[g.index(x, y, z)] < 0
            // x-edge: (x,y,z)→(x+1,y,z); cells around: (x, y-1..y, z-1..z)
            if x < g.nx - 1, here != (f[g.index(x + 1, y, z)] < 0) {
                quad(cellVertex[cid(x, y - 1, z - 1)], cellVertex[cid(x, y, z - 1)],
                     cellVertex[cid(x, y, z)], cellVertex[cid(x, y - 1, z)], flip: !here)
            }
            if y < g.ny - 1, here != (f[g.index(x, y + 1, z)] < 0) {
                quad(cellVertex[cid(x - 1, y, z - 1)], cellVertex[cid(x - 1, y, z)],
                     cellVertex[cid(x, y, z)], cellVertex[cid(x, y, z - 1)], flip: !here)
            }
            if z < g.nz - 1, here != (f[g.index(x, y, z + 1)] < 0) {
                quad(cellVertex[cid(x - 1, y - 1, z)], cellVertex[cid(x, y - 1, z)],
                     cellVertex[cid(x, y, z)], cellVertex[cid(x - 1, y, z)], flip: !here)
            }
        } } }
        return (positions, faces)
    }

    // MARK: remesh

    /// Re-meshes a closed shape through its signed distance field: evenly sized, grid-regular
    /// triangles (feature-preserving only down to a voxel), lightly relaxed to remove the
    /// staircase. Returns nil when the sampled field has no surface (e.g. an open mesh).
    static func remesh(positions: [SIMD3<Float>], faces: [SIMD3<UInt32>], resolution: Int,
                       relaxIterations: Int = 3) -> (positions: [SIMD3<Float>], faces: [SIMD3<UInt32>])? {
        guard !positions.isEmpty, !faces.isEmpty else { return nil }
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for q in positions { lo = simd_min(lo, q); hi = simd_max(hi, q) }
        let g = grid(lo: lo, hi: hi, resolution: resolution)
        let field = meshField(positions: positions, faces: faces, grid: g)
        let m = surfaceNets(field, grid: g)
        guard m.faces.count > 8 else { return nil }
        let relaxed = MeshOps.smooth(positions: m.positions, faces: m.faces, iterations: relaxIterations)
        return (relaxed, m.faces)
    }

    // MARK: entry points

    struct Operand {
        enum Kind {
            case primitive(BooleanPrimitive, center: SIMD3<Float>, size: Float)
            case mesh(positions: [SIMD3<Float>], faces: [SIMD3<UInt32>])
        }
        var kind: Kind
        func bounds() -> (SIMD3<Float>, SIMD3<Float>) {
            switch kind {
            case .primitive(_, let c, let s): return (c - SIMD3(repeating: s), c + SIMD3(repeating: s))
            case .mesh(let p, _):
                var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
                for q in p { lo = simd_min(lo, q); hi = simd_max(hi, q) }
                return (lo, hi)
            }
        }
    }

    /// `a op b` on a shared grid. Returns nil if either operand is empty or the result is empty.
    static func apply(_ op: BooleanOp, a: ([SIMD3<Float>], [SIMD3<UInt32>]), b: Operand,
                      resolution: Int = 160) -> (positions: [SIMD3<Float>], faces: [SIMD3<UInt32>])? {
        guard !a.0.isEmpty, !a.1.isEmpty else { return nil }
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for q in a.0 { lo = simd_min(lo, q); hi = simd_max(hi, q) }
        let (blo, bhi) = b.bounds()
        // Subtract/intersect only ever keep material inside A, so A's box is enough.
        if op == .union { lo = simd_min(lo, blo); hi = simd_max(hi, bhi) }
        let g = grid(lo: lo, hi: hi, resolution: resolution)
        let fa = meshField(positions: a.0, faces: a.1, grid: g)
        let fb: [Float]
        switch b.kind {
        case .primitive(let shape, let c, let s): fb = primitiveField(shape, center: c, size: s, grid: g)
        case .mesh(let p, let f): fb = meshField(positions: p, faces: f, grid: g)
        }
        let out = surfaceNets(combine(fa, fb, op), grid: g)
        return out.faces.isEmpty ? nil : out
    }

    // MARK: geometry

    /// Distance from `p` to triangle `abc` (Ericson, Real-Time Collision Detection).
    static func pointTriangleDistance(_ p: SIMD3<Float>, _ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) -> Float {
        let ab = b - a, ac = c - a, ap = p - a
        let d1 = simd_dot(ab, ap), d2 = simd_dot(ac, ap)
        if d1 <= 0, d2 <= 0 { return simd_length(ap) }
        let bp = p - b
        let d3 = simd_dot(ab, bp), d4 = simd_dot(ac, bp)
        if d3 >= 0, d4 <= d3 { return simd_length(bp) }
        let vc = d1 * d4 - d3 * d2
        if vc <= 0, d1 >= 0, d3 <= 0 { return simd_length(p - (a + ab * (d1 / (d1 - d3)))) }
        let cp = p - c
        let d5 = simd_dot(ab, cp), d6 = simd_dot(ac, cp)
        if d6 >= 0, d5 <= d6 { return simd_length(cp) }
        let vb = d5 * d2 - d1 * d6
        if vb <= 0, d2 >= 0, d6 <= 0 { return simd_length(p - (a + ac * (d2 / (d2 - d6)))) }
        let va = d3 * d6 - d5 * d4
        if va <= 0, d4 - d3 >= 0, d5 - d6 >= 0 {
            return simd_length(p - (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))))
        }
        let denom = 1 / (va + vb + vc)
        return simd_length(p - (a + ab * (vb * denom) + ac * (vc * denom)))
    }
}
