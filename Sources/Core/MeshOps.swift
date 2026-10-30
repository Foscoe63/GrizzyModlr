import Foundation
import simd

/// A user-requested edit to a finished shape mesh. Applied off the reducer: it is a
/// short, pure, local transform whose result becomes a new version in the history.
enum MeshEdit: Equatable {
    case decimate(faceBudget: Int)
    case smooth(iterations: Int)
    case fillHoles(maxEdges: Int)
    /// Mirrors one half of the model across the plane through its bounding-box center.
    case symmetrize(axis: SymmetryAxis, keepPositive: Bool)
    /// Rebuilds the surface as a clean, evenly sized mesh at `resolution` voxels.
    case remesh(resolution: Int)
    /// Remesh in the model's own axis frame with sharp edges kept, giving quad rows along its flats.
    case quadRemesh(resolution: Int)
    /// Fits a humanoid skeleton, skins the mesh, and bakes the named pose (see `PoseLibrary`).
    case pose(name: String, rig: RigConfig = RigConfig())
    case flowRemesh(targetQuads: Int, featureAngle: Float, fallbackResolution: Int, guides: [FlowGuide] = [], guideStrength: Float = 40)

    var label: String {
        switch self {
        case .decimate(let n): return "decimated to \(n / 1000)k tris"
        case .smooth(let n): return "smoothed ×\(n)"
        case .fillHoles: return "holes filled"
        case .symmetrize(let axis, let positive): return "mirrored \(positive ? "+" : "−")\(axis.rawValue.uppercased()) half"
        case .remesh(let n): return "remeshed at \(n)"
        case .quadRemesh(let n): return "quad remeshed at \(n)"
        case .pose(let name, _): return "posed: \(name)"
        case .flowRemesh(let q, _, _, _, _): return "flow remeshed to ~\(q) quads"
        }
    }
}

enum SymmetryAxis: String, CaseIterable, Identifiable {
    case x, y, z
    var id: String { rawValue }
    var index: Int { self == .x ? 0 : (self == .y ? 1 : 2) }
}

/// Pure mesh-edit operations on indexed triangle meshes (no I/O, no UI).
enum MeshOps {
    typealias Positions = [SIMD3<Float>]
    typealias Faces = [SIMD3<UInt32>]

    struct Result: Equatable {
        var positions: Positions
        var faces: Faces
        /// Human-readable change summary, e.g. "filled 3 holes".
        var note: String
        /// False when the edit left the mesh untouched (nothing to do).
        var changed: Bool
    }

    // MARK: - smoothing

    /// Taubin (λ|μ) smoothing: a shrinking Laplacian pass followed by an inflating
    /// one, so surface noise drops without the model collapsing in size. Vertices on
    /// an open boundary are pinned so the silhouette of a cut stays put.
    static func smooth(positions: Positions, faces: Faces, iterations: Int,
                       lambda: Float = 0.5, mu: Float = -0.53) -> Positions {
        guard iterations > 0, !faces.isEmpty else { return positions }
        let n = positions.count
        var neighbors = [Set<Int>](repeating: [], count: n)
        var edgeCount: [UInt64: Int] = [:]
        func key(_ a: UInt32, _ b: UInt32) -> UInt64 {
            UInt64(min(a, b)) << 32 | UInt64(max(a, b))
        }
        for f in faces {
            let v = [f.x, f.y, f.z]
            for k in 0..<3 {
                let a = v[k], b = v[(k + 1) % 3]
                neighbors[Int(a)].insert(Int(b))
                neighbors[Int(b)].insert(Int(a))
                edgeCount[key(a, b), default: 0] += 1
            }
        }
        var pinned = [Bool](repeating: false, count: n)
        for (k, c) in edgeCount where c == 1 {
            pinned[Int(k >> 32)] = true
            pinned[Int(k & 0xFFFF_FFFF)] = true
        }
        let adjacency = neighbors.map { Array($0) }

        var cur = positions
        var next = positions
        func pass(_ weight: Float) {
            for i in 0..<n {
                let adj = adjacency[i]
                guard !pinned[i], !adj.isEmpty else { next[i] = cur[i]; continue }
                var avg = SIMD3<Float>.zero
                for j in adj { avg += cur[j] }
                avg /= Float(adj.count)
                next[i] = cur[i] + weight * (avg - cur[i])
            }
            swap(&cur, &next)
        }
        for _ in 0..<iterations {
            pass(lambda)
            pass(mu)
        }
        return cur
    }

    // MARK: - hole filling

    /// Closes boundary loops of at most `maxEdges` edges with a centroid fan. Larger
    /// loops (e.g. an intentionally open base) and loops that branch non-manifoldly
    /// are left alone. Returns the number of holes filled.
    static func fillHoles(positions: Positions, faces: Faces, maxEdges: Int)
    -> (positions: Positions, faces: Faces, filled: Int) {
        var directed: [UInt64: Int] = [:]   // a→b half-edge present
        func key(_ a: UInt32, _ b: UInt32) -> UInt64 { UInt64(a) << 32 | UInt64(b) }
        for f in faces {
            let v = [f.x, f.y, f.z]
            for k in 0..<3 { directed[key(v[k], v[(k + 1) % 3]), default: 0] += 1 }
        }
        // Boundary half-edges: no opposite half-edge.
        var outgoing: [UInt32: [UInt32]] = [:]
        for (k, _) in directed {
            let a = UInt32(k >> 32), b = UInt32(k & 0xFFFF_FFFF)
            if directed[key(b, a)] == nil { outgoing[a, default: []].append(b) }
        }
        var outP = positions
        var outF = faces
        var filled = 0
        var used = Set<UInt64>()
        for start in outgoing.keys.sorted() {
            for first in outgoing[start]!.sorted() where !used.contains(key(start, first)) {
                var loop: [UInt32] = [start]
                var cur = first
                used.insert(key(start, first))
                var closed = false
                while loop.count <= maxEdges {
                    if cur == start { closed = true; break }
                    loop.append(cur)
                    guard let nxt = outgoing[cur]?.sorted().first(where: { !used.contains(key(cur, $0)) })
                    else { break }
                    used.insert(key(cur, nxt))
                    cur = nxt
                }
                guard closed, loop.count >= 3, loop.count <= maxEdges else { continue }
                var centroid = SIMD3<Float>.zero
                for i in loop { centroid += positions[Int(i)] }
                centroid /= Float(loop.count)
                let c = UInt32(outP.count)
                outP.append(centroid)
                for k in 0..<loop.count {
                    // Existing faces run loop[k]→loop[k+1]; the patch runs the other way.
                    outF.append(SIMD3(c, loop[(k + 1) % loop.count], loop[k]))
                }
                filled += 1
            }
        }
        return (outP, outF, filled)
    }

    // MARK: - symmetry

    /// Keeps the half of the mesh on one side of the bounding-box center plane (clipping the
    /// triangles that cross it), then mirrors it back, welding along the plane. A closed input
    /// stays closed; the result is exactly symmetric.
    static func symmetrize(positions: Positions, faces: Faces, axis: Int, keepPositive: Bool)
    -> (positions: Positions, faces: Faces)? {
        guard !positions.isEmpty, !faces.isEmpty else { return nil }
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for p in positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let plane = (lo[axis] + hi[axis]) / 2
        let eps = max(hi[axis] - lo[axis], 1e-9) * 1e-6
        let sign: Float = keepPositive ? 1 : -1
        func dist(_ p: SIMD3<Float>) -> Float {
            let d = (p[axis] - plane) * sign
            return abs(d) < eps ? 0 : d
        }

        var outP = Positions(), onPlane = [Bool]()
        var remap = [Int: UInt32]()                       // old vertex → new index
        var edgeVertex = [UInt64: UInt32]()                // clipped edge → new vertex on the plane
        func vertex(_ old: Int) -> UInt32 {
            if let i = remap[old] { return i }
            var p = positions[old]
            let onP = dist(p) == 0
            if onP { p[axis] = plane }
            let i = UInt32(outP.count); outP.append(p); onPlane.append(onP); remap[old] = i
            return i
        }
        func crossing(_ a: Int, _ b: Int) -> UInt32 {
            let key = UInt64(min(a, b)) << 32 | UInt64(max(a, b))
            if let i = edgeVertex[key] { return i }
            let (da, db) = (dist(positions[a]), dist(positions[b]))
            let t = da / (da - db)
            var p = positions[a] + (positions[b] - positions[a]) * t
            p[axis] = plane
            let i = UInt32(outP.count); outP.append(p); onPlane.append(true); edgeVertex[key] = i
            return i
        }

        var kept = Faces()
        for f in faces {
            let idx = [Int(f.x), Int(f.y), Int(f.z)]
            let d = idx.map { dist(positions[$0]) }
            let inside = d.filter { $0 >= 0 }.count
            if inside == 0 || d.allSatisfy({ $0 <= 0 }) && d.contains(where: { $0 < 0 }) && !d.contains(where: { $0 > 0 }) { continue }
            if inside == 3 { kept.append([vertex(idx[0]), vertex(idx[1]), vertex(idx[2])]); continue }
            // Sutherland–Hodgman against the plane (keep d >= 0), then fan-triangulate.
            var poly = [UInt32]()
            for k in 0..<3 {
                let a = idx[k], b = idx[(k + 1) % 3], da = d[k], db = d[(k + 1) % 3]
                if da >= 0 { poly.append(vertex(a)) }
                if (da > 0 && db < 0) || (da < 0 && db > 0) { poly.append(crossing(a, b)) }
            }
            if poly.count >= 3 {
                for k in 1..<(poly.count - 1) where poly[0] != poly[k] && poly[k] != poly[k + 1] && poly[0] != poly[k + 1] {
                    kept.append([poly[0], poly[k], poly[k + 1]])
                }
            }
        }
        guard !kept.isEmpty else { return nil }

        // Mirror: vertices on the plane are shared, everything else is reflected.
        var mirror = [UInt32](repeating: 0, count: outP.count)
        var all = outP
        for i in 0..<outP.count {
            if onPlane[i] { mirror[i] = UInt32(i) } else {
                var p = outP[i]; p[axis] = 2 * plane - p[axis]
                mirror[i] = UInt32(all.count); all.append(p)
            }
        }
        var allFaces = kept
        for f in kept { allFaces.append([mirror[Int(f.x)], mirror[Int(f.z)], mirror[Int(f.y)]]) }
        // Drop slivers collapsed by welding.
        allFaces = allFaces.filter { $0.x != $0.y && $0.y != $0.z && $0.x != $0.z }
        return (all, allFaces)
    }

    // MARK: - dispatch

    static func apply(_ edit: MeshEdit, positions: Positions, faces: Faces) -> Result {
        switch edit {
        case .decimate(let budget):
            guard budget > 0, faces.count > budget else {
                return Result(positions: positions, faces: faces,
                              note: "already under \(budget / 1000)k triangles", changed: false)
            }
            let r = MeshDecimator.decimate(positions: positions, faces: faces, faceBudget: budget)
            let ok = !r.faces.isEmpty && r.positions.count >= 3
                && r.positions.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }
            guard ok else {
                return Result(positions: positions, faces: faces,
                              note: "decimation produced an empty mesh", changed: false)
            }
            return Result(positions: r.positions, faces: r.faces,
                          note: "\(faces.count) → \(r.faces.count) triangles", changed: true)
        case .smooth(let iterations):
            let p = smooth(positions: positions, faces: faces, iterations: iterations)
            return Result(positions: p, faces: faces, note: "\(iterations) iterations", changed: true)
        case .symmetrize(let axis, let keepPositive):
            guard let r = symmetrize(positions: positions, faces: faces, axis: axis.index, keepPositive: keepPositive) else {
                return Result(positions: positions, faces: faces, note: "nothing on that side of the plane", changed: false)
            }
            return Result(positions: r.positions, faces: r.faces, note: "\(faces.count) → \(r.faces.count) triangles", changed: true)
        case .remesh(let resolution):
            guard let r = MeshBoolean.remesh(positions: positions, faces: faces, resolution: resolution) else {
                return Result(positions: positions, faces: faces, note: "the mesh isn't closed enough to remesh", changed: false)
            }
            return Result(positions: r.positions, faces: r.faces, note: "\(faces.count) → \(r.faces.count) triangles", changed: true)
        case .quadRemesh(let resolution):
            guard let r = MeshBoolean.quadRemesh(positions: positions, faces: faces, resolution: resolution) else {
                return Result(positions: positions, faces: faces, note: "the mesh isn't closed enough to remesh", changed: false)
            }
            let pairs = QuadPairing.pair(positions: r.positions, faces: r.faces)
            let share = r.faces.isEmpty ? 0 : Int((Float(pairs.quads.count * 2) / Float(r.faces.count) * 100).rounded())
            return Result(positions: r.positions, faces: r.faces,
                          note: "\(faces.count) → \(r.faces.count) triangles, \(share)% pair into quads", changed: true)
        case .flowRemesh(let target, let angle, let fallback, let guides, let strength):
            var o = FlowRemesh.Options()
            o.targetQuads = target; o.featureAngle = angle; o.guides = guides; o.guideStrength = strength
            if let r = FlowRemesh.remesh(positions: positions, faces: faces, options: o) {
                let pct = Int((Float(r.quads.count * 2) / Float(max(r.faces.count, 1)) * 100).rounded())
                return Result(positions: r.positions, faces: r.faces,
                              note: "flow remesh: \(r.quads.count) quads, \(pct)% of faces", changed: true)
            }
            // The flow could not be turned into a closed surface; keep the user's result useful.
            let q = apply(.quadRemesh(resolution: fallback), positions: positions, faces: faces)
            return Result(positions: q.positions, faces: q.faces,
                          note: "flow remesh didn't close; used the aligned remesh instead (\(q.note))", changed: q.changed)
        case .pose(let name, let rig):
            guard let pose = PoseLibrary.named(name), let skeleton = Skeleton.fit(to: positions, rig: rig) else {
                return Result(positions: positions, faces: faces, note: "couldn't fit a skeleton to this mesh", changed: false)
            }
            let weights = SkinWeights.compute(positions: positions, skeleton: skeleton, faces: faces, smoothing: rig.smoothing, strokes: rig.strokes)
            let height = (positions.map(\.y).max() ?? 0) - (positions.map(\.y).min() ?? 0)
            let posed = pose.apply(to: positions, skeleton: skeleton, weights: weights, height: height)
            guard posed.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else {
                return Result(positions: positions, faces: faces, note: "posing produced invalid points", changed: false)
            }
            return Result(positions: posed, faces: faces, note: "\(name) pose on a fitted \(skeleton.joints.count)-joint skeleton", changed: name != "Rest")
        case .fillHoles(let maxEdges):
            let r = fillHoles(positions: positions, faces: faces, maxEdges: maxEdges)
            return Result(positions: r.positions, faces: r.faces,
                          note: r.filled == 0 ? "no holes found" : "filled \(r.filled) hole\(r.filled == 1 ? "" : "s")",
                          changed: r.filled > 0)
        }
    }

    /// Reads a `.mesh` file, applies the edit and writes the result (with fresh
    /// normals) to `dst`. Returns nil on failure, else the result summary; when the
    /// edit changed nothing no file is written.
    static func applyToFile(_ edit: MeshEdit, from src: URL, to dst: URL) -> Result? {
        guard let raw = try? Data(contentsOf: src), let mesh = MeshDecimator.parse(raw) else { return nil }
        let r = apply(edit, positions: mesh.positions, faces: mesh.faces)
        guard r.changed else { return r }
        let data = MeshDecimator.serialize(positions: r.positions, faces: r.faces)
        do { try data.write(to: dst, options: .atomic) } catch { return nil }
        return r
    }
}
