import Foundation
import simd

enum SculptTool: String, CaseIterable, Identifiable {
    case push, pull, smooth, flatten
    var id: String { rawValue }
    var label: String {
        switch self {
        case .push: return "Push"
        case .pull: return "Pull"
        case .smooth: return "Smooth"
        case .flatten: return "Flatten"
        }
    }
    var icon: String {
        switch self {
        case .push: return "arrow.down.to.line"
        case .pull: return "arrow.up.to.line"
        case .smooth: return "scribble.variable"
        case .flatten: return "equal"
        }
    }
}

/// Editable in-memory mesh for brush sculpting. Pure value logic (no SceneKit): the
/// viewer feeds it brush hits and reads back positions and normals.
struct SculptMesh {
    private(set) var positions: [SIMD3<Float>]
    private(set) var normals: [SIMD3<Float>]
    let faces: [SIMD3<UInt32>]
    private let neighbors: [[Int]]
    private let vertexFaces: [[Int]]

    /// Largest distance from the bounding-box center — scales brush radius.
    let boundsRadius: Float

    // Undo: each stroke records the original position of every vertex it touched.
    private var strokeBefore: [Int: SIMD3<Float>]?
    private var undoStack: [[Int: SIMD3<Float>]] = []
    var canUndo: Bool { !undoStack.isEmpty }
    private(set) var isEdited = false

    init(positions: [SIMD3<Float>], faces: [SIMD3<UInt32>]) {
        self.positions = positions
        self.faces = faces
        var nb = [Set<Int>](repeating: [], count: positions.count)
        var vf = [[Int]](repeating: [], count: positions.count)
        for (fi, f) in faces.enumerated() {
            let v = [Int(f.x), Int(f.y), Int(f.z)]
            for k in 0..<3 {
                nb[v[k]].insert(v[(k + 1) % 3]); nb[v[k]].insert(v[(k + 2) % 3])
                vf[v[k]].append(fi)
            }
        }
        neighbors = nb.map { Array($0) }
        vertexFaces = vf
        normals = [SIMD3<Float>](repeating: .zero, count: positions.count)
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for p in positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        boundsRadius = positions.isEmpty ? 1 : max(simd_length(hi - lo) / 2, 1e-4)
        recomputeNormals(for: Array(0..<positions.count))
    }

    var center: SIMD3<Float> {
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for p in positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        return (lo + hi) / 2
    }

    // MARK: strokes

    mutating func beginStroke() { strokeBefore = [:] }

    mutating func endStroke() {
        if let s = strokeBefore, !s.isEmpty { undoStack.append(s) }
        strokeBefore = nil
    }

    /// Reverts the last finished stroke. Returns the vertices that changed.
    @discardableResult
    mutating func undo() -> [Int] {
        guard let last = undoStack.popLast() else { return [] }
        for (i, p) in last { positions[i] = p }
        isEdited = !undoStack.isEmpty
        let touched = Array(last.keys)
        recomputeNormals(for: expanded(touched))
        return touched
    }

    /// Applies one dab of the brush at `point` (mesh space). `strength` is 0…1 of the
    /// brush radius per dab at the center. Returns the vertices whose position or
    /// normal changed, so the viewer can update only what it must.
    @discardableResult
    mutating func dab(tool: SculptTool, at point: SIMD3<Float>, radius: Float, strength: Float) -> [Int] {
        guard radius > 0, strength > 0 else { return [] }
        let r2 = radius * radius
        var hit: [(i: Int, w: Float)] = []
        for i in positions.indices {
            let d2 = simd_length_squared(positions[i] - point)
            if d2 < r2 {
                let t = (d2 / r2).squareRoot()                 // 0 center … 1 edge
                hit.append((i, 0.5 * (1 + cos(.pi * t))))      // smooth cosine falloff
            }
        }
        guard !hit.isEmpty else { return [] }

        // Brush normal: area-ish weighted average of normals under the inner brush.
        var bn = SIMD3<Float>.zero
        var bc = SIMD3<Float>.zero, wsum: Float = 0
        for h in hit { bn += normals[h.i] * h.w; bc += positions[h.i] * h.w; wsum += h.w }
        let brushNormal = simd_length(bn) > 1e-9 ? simd_normalize(bn) : SIMD3<Float>(0, 0, 1)
        let plane = bc / max(wsum, 1e-9)

        var moves: [(Int, SIMD3<Float>)] = []
        moves.reserveCapacity(hit.count)
        for h in hit {
            let p = positions[h.i]
            let amount = strength * radius * 0.25 * h.w
            switch tool {
            case .push:
                moves.append((h.i, p - brushNormal * amount))
            case .pull:
                moves.append((h.i, p + brushNormal * amount))
            case .smooth:
                let nb = neighbors[h.i]
                guard !nb.isEmpty else { continue }
                var avg = SIMD3<Float>.zero
                for j in nb { avg += positions[j] }
                avg /= Float(nb.count)
                moves.append((h.i, p + (avg - p) * min(1, strength * 2 * h.w)))
            case .flatten:
                let dist = simd_dot(p - plane, brushNormal)
                moves.append((h.i, p - brushNormal * dist * min(1, strength * 2 * h.w)))
            }
        }
        for (i, np) in moves {
            if strokeBefore != nil, strokeBefore![i] == nil { strokeBefore![i] = positions[i] }
            positions[i] = np
        }
        isEdited = true
        let changed = moves.map { $0.0 }
        recomputeNormals(for: expanded(changed))
        return expanded(changed)
    }

    // MARK: normals

    /// Vertices plus their 1-ring: a moved vertex changes its neighbors' normals too.
    private func expanded(_ verts: [Int]) -> [Int] {
        var s = Set(verts)
        for v in verts { for n in neighbors[v] { s.insert(n) } }
        return Array(s)
    }

    private mutating func recomputeNormals(for verts: [Int]) {
        for v in verts {
            var n = SIMD3<Float>.zero
            for fi in vertexFaces[v] {
                let f = faces[fi]
                let a = positions[Int(f.x)], b = positions[Int(f.y)], c = positions[Int(f.z)]
                n += simd_cross(b - a, c - a)               // area-weighted
            }
            let len = simd_length(n)
            normals[v] = len > 1e-12 ? n / len : SIMD3<Float>(0, 0, 1)
        }
    }
}
