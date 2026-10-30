import Foundation
import simd

/// A smooth 4-direction ("cross") field over a triangle mesh: every face carries one direction
/// defined modulo a quarter turn, so it stands for a pair of perpendicular axes. It is seeded from
/// the principal curvature directions and from sharp creases, then smoothed across the surface.
/// Quad edge rows follow it.
struct CrossField {
    let topology: MeshTopology
    /// Angle of one axis in each face's frame, in [0, π/2).
    private(set) var angle: [Float]

    struct Singularity: Equatable {
        var vertex: Int
        /// Multiples of 1/4: +1 is a 4-valent-excess point (a vertex with 3 edges), −1 has 5.
        var index: Float
    }

    struct Options {
        /// Dihedral angle (degrees) above which an edge is a crease the field must follow.
        var featureAngle: Float = 35
        var iterations = 600
        /// The data pull is annealed down over this many passes so smoothness wins at large scale.
        var passes = 6
        /// How strongly curvature pulls on the field relative to neighbor smoothing (0…∞).
        var curvatureWeight: Float = 1.5
        var featureWeight: Float = 12
    }

    // MARK: - construction

    static func compute(topology t: MeshTopology, options o: Options = Options()) -> CrossField {
        let n = t.faces.count
        var z = [SIMD2<Float>](repeating: SIMD2(1, 0), count: n)         // e^{i4θ}
        var target = [SIMD2<Float>](repeating: .zero, count: n)       // curvature seeds
        var hard = [SIMD2<Float>](repeating: .zero, count: n)         // crease constraints

        // Curvature: weight by anisotropy relative to the strongly anisotropic faces.
        var curv: [(Float, Float)] = []
        curv.reserveCapacity(n)
        for i in 0..<n { curv.append(t.curvatureDirection(face: i)) }
        let spreads = curv.map { $0.1 }.sorted()
        let scale = max(spreads.isEmpty ? 0 : spreads[min(spreads.count - 1, Int(Float(spreads.count) * 0.9))], 1e-9)
        for i in 0..<n {
            let w = min(1, curv[i].1 / scale) * o.curvatureWeight
            target[i] += w * unit(4 * curv[i].0)
        }

        // Creases: align an axis with the crease direction on both adjacent faces.
        let cosFeature = cos(o.featureAngle * .pi / 180)
        for i in 0..<n {
            for nb in t.neighbors[i] where nb.face > i {
                guard simd_dot(t.faceNormal[i], t.faceNormal[nb.face]) < cosFeature else { continue }
                let edge = t.positions[Int(nb.b)] - t.positions[Int(nb.a)]
                let len = simd_length(edge)
                guard len > 1e-12 else { continue }
                let d = edge / len
                hard[i] += o.featureWeight * unit(4 * t.angle(of: d, in: i))
                hard[nb.face] += o.featureWeight * unit(4 * t.angle(of: d, in: nb.face))
            }
        }
        for i in 0..<n {
            let seed = target[i] + hard[i]
            if simd_length(seed) > 0 { z[i] = simd_normalize(seed) }
        }

        // Transport rotation of 4θ from face g into face f across their shared edge.
        struct Link { var face: Int; var rot: SIMD2<Float>; var weight: Float }
        var links = [[Link]](repeating: [], count: n)
        for f in 0..<n {
            for nb in t.neighbors[f] {
                let e = t.positions[Int(nb.b)] - t.positions[Int(nb.a)]
                let len = simd_length(e)
                guard len > 1e-12 else { continue }
                let d = e / len
                let delta = t.angle(of: d, in: f) - t.angle(of: d, in: nb.face)    // α_f − α_g
                // Dual-graph Laplacian weight: shared edge length over the distance between face centers,
                // so long thin faces don't dominate and irregular (decimated) meshes still smooth well.
                let weight = len / max(simd_distance(t.centroid(f), t.centroid(nb.face)), 1e-12)
                links[f].append(Link(face: nb.face, rot: unit(4 * delta), weight: weight))
            }
        }

        // Gauss–Seidel in breadth-first order so information spreads from the seeds.
        let order = breadthFirst(t, seeds: (0..<n).filter { simd_length(target[$0] + hard[$0]) > 0 })
        let passes = max(1, o.passes), perPass = max(1, o.iterations / passes)
        for pass in 0..<passes {
            let pull = powf(0.5, Float(pass))                               // 1, ½, ¼, …
            for _ in 0..<perPass {
                for f in order {
                    var sum = target[f] * pull + hard[f]
                    for l in links[f] { sum += mul(z[l.face], l.rot) * l.weight }
                    let len = simd_length(sum)
                    if len > 1e-9 { z[f] = sum / len }
                }
            }
        }
        var angle = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var a = atan2(z[i].y, z[i].x) / 4
            if a < 0 { a += .pi / 2 }
            angle[i] = a
        }
        return CrossField(topology: t, angle: angle)
    }

    /// Field on a mesh simplified to about `maxFaces`: smoothing only reaches a fixed number of faces,
    /// so a dense scan needs a coarser mesh for the flow to be smooth at the scale of the whole shape.
    /// The field's own `topology` is the simplified mesh.
    static func computeAdaptive(positions: [SIMD3<Float>], faces: [SIMD3<UInt32>], maxFaces: Int = 12_000,
                                options: Options = Options()) -> CrossField {
        if faces.count > maxFaces {
            let r = MeshDecimator.decimate(positions: positions, faces: faces, faceBudget: maxFaces)
            if !r.faces.isEmpty, r.positions.count >= 3 {
                return compute(topology: MeshTopology(positions: r.positions, faces: r.faces), options: options)
            }
        }
        return compute(topology: MeshTopology(positions: positions, faces: faces), options: options)
    }

    private static func unit(_ a: Float) -> SIMD2<Float> { SIMD2(cos(a), sin(a)) }
    private static func mul(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> SIMD2<Float> {
        SIMD2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)
    }

    private static func breadthFirst(_ t: MeshTopology, seeds: [Int]) -> [Int] {
        var seen = [Bool](repeating: false, count: t.faces.count)
        var order: [Int] = []
        var queue: [Int] = []
        func visit(_ f: Int) { if !seen[f] { seen[f] = true; queue.append(f) } }
        seeds.forEach(visit)
        for start in 0..<t.faces.count {
            visit(start)                                      // covers disconnected pieces and the no-seed case
            var head = 0
            while head < queue.count {
                let f = queue[head]; head += 1
                order.append(f)
                for nb in t.neighbors[f] { visit(nb.face) }
            }
            queue.removeAll(keepingCapacity: true)
        }
        return order
    }

    // MARK: - queries

    /// The two perpendicular axes of a face as unit 3D vectors.
    func axes(face i: Int) -> (SIMD3<Float>, SIMD3<Float>) {
        let a = angle[i]
        let u = cos(a) * topology.frameX[i] + sin(a) * topology.frameY[i]
        return (u, simd_cross(topology.faceNormal[i], u))
    }

    /// Smallest angle (radians, 0…π/4) between direction `d` and the nearest axis of `face`.
    func misalignment(of d: SIMD3<Float>, face i: Int) -> Float {
        let delta = topology.angle(of: d, in: i) - angle[i]
        let r = delta.truncatingRemainder(dividingBy: .pi / 2)
        let m = abs(r < 0 ? r + .pi / 2 : r)
        return min(m, .pi / 2 - m)
    }

    /// Vertices whose face fan does not close up on itself: the field has an index there. Computed
    /// from the mismatch the smooth field leaves across each fan edge after parallel transport, plus
    /// the surface's own angle defect at the vertex.
    func singularities(threshold: Float = 0.125) -> [Singularity] {
        singularityDetails(threshold: threshold).map { Singularity(vertex: $0.vertex, index: $0.quantized) }
    }

    /// Per-vertex raw terms, for tests and debugging.
    func singularityDetails(threshold: Float = 0.125) -> [(vertex: Int, defect: Float, mismatch: Float, quantized: Float)] {
        let t = topology
        var out: [(vertex: Int, defect: Float, mismatch: Float, quantized: Float)] = []
        for v in t.positions.indices {
            let fan = t.vertexFan[v]
            guard fan.count >= 3 else { continue }
            var mismatch: Float = 0
            var cornerSum: Float = 0
            for j in 0..<fan.count {
                let g = fan[j], f = fan[(j + 1) % fan.count]
                guard let nb = t.neighbors[g].first(where: { $0.face == f }) else { mismatch = .nan; break }
                let e = t.positions[Int(nb.b)] - t.positions[Int(nb.a)]
                let len = simd_length(e)
                guard len > 1e-12 else { mismatch = .nan; break }
                let d = e / len
                let delta = t.angle(of: d, in: f) - t.angle(of: d, in: g)         // α_f − α_g
                var eps = angle[f] - (angle[g] + delta)                           // leftover rotation
                eps = wrapQuarter(eps)
                mismatch += eps
                cornerSum += corner(of: g, at: UInt32(v))
            }
            guard mismatch.isFinite else { continue }
            let defect = 2 * .pi - cornerSum
            let index = (defect + mismatch) / (2 * .pi)
            let quantized = (index * 4).rounded() / 4
            if abs(quantized) >= threshold { out.append((v, defect, mismatch, quantized)) }
        }
        return out
    }

    /// Wraps an angle into (−π/4, π/4], the representative of its class modulo a quarter turn.
    private func wrapQuarter(_ a: Float) -> Float {
        var r = a.truncatingRemainder(dividingBy: .pi / 2)
        if r > .pi / 4 { r -= .pi / 2 } else if r <= -.pi / 4 { r += .pi / 2 }
        return r
    }

    private func corner(of face: Int, at vertex: UInt32) -> Float {
        let f = topology.faces[face]
        let ids = [f.x, f.y, f.z]
        guard let k = ids.firstIndex(of: vertex) else { return 0 }
        let p = topology.positions[Int(ids[k])]
        let a = topology.positions[Int(ids[(k + 1) % 3])] - p, b = topology.positions[Int(ids[(k + 2) % 3])] - p
        let c = simd_dot(a, b) / max(simd_length(a) * simd_length(b), 1e-20)
        return acos(max(-1, min(1, c)))
    }
}
