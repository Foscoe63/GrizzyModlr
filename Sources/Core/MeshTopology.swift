import Foundation
import simd

/// Adjacency and local geometry of a triangle mesh, built once for the flow-field tools: face
/// frames and normals, the face-to-face (dual) graph, vertex fans and a per-face curvature tensor.
struct MeshTopology {
    struct Neighbor {
        var face: Int
        /// The shared edge's vertices, in the order this face winds them.
        var a: UInt32
        var b: UInt32
    }

    let positions: [SIMD3<Float>]
    let faces: [SIMD3<UInt32>]
    private(set) var faceNormal: [SIMD3<Float>] = []
    private(set) var faceArea: [Float] = []
    private(set) var frameX: [SIMD3<Float>] = []
    private(set) var frameY: [SIMD3<Float>] = []
    private(set) var vertexNormal: [SIMD3<Float>] = []
    /// Vertex normals smoothed over a few rings: curvature from raw normals is mostly noise on scans.
    private(set) var curvatureNormal: [SIMD3<Float>] = []
    /// Up to three neighbors per face (fewer on open or non-manifold edges).
    private(set) var neighbors: [[Neighbor]] = []
    /// Faces around each vertex, in counter-clockwise cyclic order, when the fan is a closed manifold loop; else empty.
    private(set) var vertexFan: [[Int]] = []

    init(positions: [SIMD3<Float>], faces: [SIMD3<UInt32>]) {
        self.positions = positions
        self.faces = faces
        let nF = faces.count
        faceNormal = [SIMD3<Float>](repeating: [0, 0, 1], count: nF)
        faceArea = [Float](repeating: 0, count: nF)
        frameX = [SIMD3<Float>](repeating: [1, 0, 0], count: nF)
        frameY = [SIMD3<Float>](repeating: [0, 1, 0], count: nF)
        vertexNormal = [SIMD3<Float>](repeating: .zero, count: positions.count)
        for (i, f) in faces.enumerated() {
            let p0 = positions[Int(f.x)], p1 = positions[Int(f.y)], p2 = positions[Int(f.z)]
            let c = simd_cross(p1 - p0, p2 - p0)
            let len = simd_length(c)
            faceArea[i] = len / 2
            guard len > 1e-20 else { continue }
            let n = c / len
            faceNormal[i] = n
            let x = simd_normalize(p1 - p0)
            frameX[i] = x
            frameY[i] = simd_cross(n, x)
            for v in [f.x, f.y, f.z] { vertexNormal[Int(v)] += c }       // area-weighted
        }
        for i in vertexNormal.indices {
            let l = simd_length(vertexNormal[i])
            vertexNormal[i] = l > 1e-20 ? vertexNormal[i] / l : [0, 0, 1]
        }
        buildAdjacency()
        smoothCurvatureNormals(rounds: 4)
    }

    private mutating func smoothCurvatureNormals(rounds: Int) {
        var cur = vertexNormal
        var ring = [[Int]](repeating: [], count: positions.count)
        for f in faces {
            let ids = [Int(f.x), Int(f.y), Int(f.z)]
            for k in 0..<3 { ring[ids[k]].append(ids[(k + 1) % 3]); ring[ids[k]].append(ids[(k + 2) % 3]) }
        }
        for _ in 0..<rounds {
            var next = cur
            for v in cur.indices where !ring[v].isEmpty {
                var sum = cur[v] * 2
                for w in ring[v] { sum += cur[w] }
                let l = simd_length(sum)
                if l > 1e-12 { next[v] = sum / l }
            }
            cur = next
        }
        curvatureNormal = cur
    }

    private mutating func buildAdjacency() {
        func key(_ a: UInt32, _ b: UInt32) -> UInt64 { UInt64(min(a, b)) << 32 | UInt64(max(a, b)) }
        var edgeFaces: [UInt64: [Int]] = [:]
        for (i, f) in faces.enumerated() {
            for (a, b) in [(f.x, f.y), (f.y, f.z), (f.z, f.x)] { edgeFaces[key(a, b), default: []].append(i) }
        }
        neighbors = [[Neighbor]](repeating: [], count: faces.count)
        for (i, f) in faces.enumerated() {
            for (a, b) in [(f.x, f.y), (f.y, f.z), (f.z, f.x)] {
                let fs = edgeFaces[key(a, b)] ?? []
                guard fs.count == 2, let other = fs.first(where: { $0 != i }) else { continue }
                neighbors[i].append(Neighbor(face: other, a: a, b: b))
            }
        }
        // Cyclic fans: walk from a face across edges that contain the vertex.
        var incident = [[Int]](repeating: [], count: positions.count)
        for (i, f) in faces.enumerated() { for v in [f.x, f.y, f.z] { incident[Int(v)].append(i) } }
        vertexFan = [[Int]](repeating: [], count: positions.count)
        for v in positions.indices {
            let inc = incident[v]
            guard inc.count >= 3 else { continue }
            var fan = [inc[0]]
            var cur = inc[0]
            var closed = false
            while fan.count <= inc.count {
                // Counter-clockwise about the vertex: the next face lies across the edge this face winds into v.
                let around = neighbors[cur].filter { $0.b == UInt32(v) }
                guard let next = around.first?.face else { break }
                if next == fan[0] { closed = fan.count == inc.count; break }
                fan.append(next); cur = next
            }
            if closed { vertexFan[v] = fan }
        }
    }

    /// Angle of the 3D direction `t` in the frame of `face` (radians, counter-clockwise about the normal).
    @inline(__always)
    func angle(of t: SIMD3<Float>, in face: Int) -> Float {
        atan2(simd_dot(t, frameY[face]), simd_dot(t, frameX[face]))
    }

    func centroid(_ face: Int) -> SIMD3<Float> {
        let f = faces[face]
        return (positions[Int(f.x)] + positions[Int(f.y)] + positions[Int(f.z)]) / 3
    }

    /// Area around each vertex (a third of its incident faces), used to weight nodes when coarsening.
    func vertexAreaWeights() -> [Float] {
        var w = [Float](repeating: 0, count: positions.count)
        for (i, f) in faces.enumerated() {
            let share = faceArea[i] / 3
            w[Int(f.x)] += share; w[Int(f.y)] += share; w[Int(f.z)] += share
        }
        return w.map { max($0, 1e-12) }
    }

    /// Average edge length, used to size overlays and lattices.
    var averageEdgeLength: Float {
        var total: Float = 0
        for f in faces {
            total += simd_distance(positions[Int(f.x)], positions[Int(f.y)])
                + simd_distance(positions[Int(f.y)], positions[Int(f.z)])
                + simd_distance(positions[Int(f.z)], positions[Int(f.x)])
        }
        return faces.isEmpty ? 0 : total / Float(faces.count * 3)
    }

    /// Principal curvature direction of a face as an angle in its frame, plus the anisotropy
    /// |κ₁ − κ₂| that says how trustworthy the direction is. The tensor is the symmetric 2×2
    /// shape operator that best maps the face's edge vectors to its vertex-normal differences.
    func curvatureDirection(face i: Int) -> (angle: Float, anisotropy: Float) {
        let f = faces[i]
        let ids = [Int(f.x), Int(f.y), Int(f.z)]
        // Normal equations for S = [a b; b c]: rows (ex, ey, 0 | dnx) and (0, ex, ey | dny).
        var ata = simd_float3x3(0), atb = SIMD3<Float>.zero
        for k in 0..<3 {
            let p = positions[ids[k]], q = positions[ids[(k + 1) % 3]]
            let e = q - p, dn = curvatureNormal[ids[(k + 1) % 3]] - curvatureNormal[ids[k]]
            let ex = simd_dot(e, frameX[i]), ey = simd_dot(e, frameY[i])
            let dx = simd_dot(dn, frameX[i]), dy = simd_dot(dn, frameY[i])
            let r1 = SIMD3<Float>(ex, ey, 0), r2 = SIMD3<Float>(0, ex, ey)
            ata += simd_float3x3(columns: (r1 * r1.x + r2 * r2.x, r1 * r1.y + r2 * r2.y, r1 * r1.z + r2 * r2.z))
            atb += r1 * dx + r2 * dy
        }
        ata += simd_float3x3(diagonal: SIMD3(repeating: 1e-9))
        let s = ata.inverse * atb
        let (a, b, c) = (s.x, s.y, s.z)
        guard a.isFinite, b.isFinite, c.isFinite else { return (0, 0) }
        // Eigen-decomposition of [a b; b c]: the eigenvector angle is half the angle of (a−c, 2b).
        let angle = 0.5 * atan2(2 * b, a - c)
        let spread = ((a - c) * (a - c) + 4 * b * b).squareRoot()      // = |κ₁ − κ₂|
        return (angle, spread)
    }
}
