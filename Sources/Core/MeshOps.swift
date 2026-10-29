import Foundation
import simd

/// A user-requested edit to a finished shape mesh. Applied off the reducer: it is a
/// short, pure, local transform whose result becomes a new version in the history.
enum MeshEdit: Equatable {
    case decimate(faceBudget: Int)
    case smooth(iterations: Int)
    case fillHoles(maxEdges: Int)

    var label: String {
        switch self {
        case .decimate(let n): return "decimated to \(n / 1000)k tris"
        case .smooth(let n): return "smoothed ×\(n)"
        case .fillHoles: return "holes filled"
        }
    }
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
