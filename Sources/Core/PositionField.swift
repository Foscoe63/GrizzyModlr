import Foundation
import simd

/// Per-vertex orientation and lattice position for field-aligned quad meshing (after Instant Meshes).
/// Every vertex owns a square lattice of spacing `rho` in its tangent plane, oriented by the cross
/// field and anchored at `position`. Optimization nudges each vertex's anchor toward the lattice
/// points its neighbors imply, until neighboring lattices agree: then lattice points that coincide
/// across vertices are the vertices of the output mesh.
struct PositionField {
    let topology: MeshTopology
    let rho: Float
    /// Unit tangent direction of the lattice's first axis at each vertex (second axis = n × o).
    private(set) var orientation: [SIMD3<Float>]
    /// Lattice anchor (a lattice point of the vertex's own lattice), near the vertex.
    private(set) var position: [SIMD3<Float>]
    /// Neighboring vertices.
    let adjacency: [[Int]]

    var normal: [SIMD3<Float>] { topology.vertexNormal }

    init(field: CrossField, rho: Float, iterations: Int = 120, guides: [FlowGuide] = []) {
        let t = field.topology
        topology = t
        self.rho = rho
        let n = t.positions.count

        var adj = [Set<Int>](repeating: [], count: n)
        for f in t.faces {
            let ids = [Int(f.x), Int(f.y), Int(f.z)]
            for k in 0..<3 { adj[ids[k]].insert(ids[(k + 1) % 3]); adj[ids[k]].insert(ids[(k + 2) % 3]) }
        }
        adjacency = adj.map { Array($0).sorted() }

        // Vertex orientation: average the incident faces' crosses in the vertex tangent plane (4-RoSy).
        var incident = [[Int]](repeating: [], count: n)
        for (i, f) in t.faces.enumerated() { for v in [f.x, f.y, f.z] { incident[Int(v)].append(i) } }
        var orient = [SIMD3<Float>](repeating: [1, 0, 0], count: n)
        for v in 0..<n {
            let nv = t.vertexNormal[v]
            let e1 = Self.anyTangent(nv), e2 = simd_cross(nv, e1)
            var z = SIMD2<Float>.zero
            for f in incident[v] {
                let (u, _) = field.axes(face: f)
                // Rotate the face's axis into the vertex tangent plane (project and renormalize).
                let p = u - nv * simd_dot(u, nv)
                let len = simd_length(p)
                guard len > 1e-6 else { continue }
                let q = p / len
                let phi = atan2(simd_dot(q, e2), simd_dot(q, e1))
                z += t.faceArea[f] * SIMD2(cos(4 * phi), sin(4 * phi))
            }
            let phi = simd_length(z) > 1e-12 ? atan2(z.y, z.x) / 4 : 0
            orient[v] = cos(phi) * e1 + sin(phi) * e2
        }
        // Guides: vertices within reach of a guide curve take its direction as their lattice axis and are
        // pinned so a lattice row runs through the curve itself.
        var pins = [SIMD3<Float>?](repeating: nil, count: n)
        if !guides.isEmpty {
            var samples: [(point: SIMD3<Float>, dir: SIMD3<Float>)] = []
            for g in guides {
                for (a, b) in g.segments(maxLength: rho * 0.25) {
                    let d = b - a
                    if simd_length(d) > 1e-12 { samples.append(((a + b) / 2, simd_normalize(d))) }
                }
            }
            for v in 0..<n {
                var best: Int?, bestD = rho * 0.6
                for (k, s) in samples.enumerated() {
                    let d = simd_distance(s.point, t.positions[v])
                    if d < bestD { bestD = d; best = k }
                }
                guard let k = best else { continue }
                let nv = t.vertexNormal[v]
                let tangent = samples[k].dir - nv * simd_dot(samples[k].dir, nv)
                guard simd_length(tangent) > 0.3 else { continue }
                orient[v] = simd_normalize(tangent)
                pins[v] = samples[k].point
            }
        }
        orientation = orient
        // Random (but deterministic) lattice offsets break the symmetry that would otherwise leave
        // every vertex's lattice centered on itself; the optimization then pulls them into agreement.
        var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
        func random() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float((seed >> 40) & 0xFFFF) / 65536 - 0.5
        }
        position = (0..<n).map { v in
            let b = simd_cross(t.vertexNormal[v], orient[v])
            return t.positions[v] + orient[v] * random() * rho + b * random() * rho
        }
        optimize(iterations: iterations, pins: pins)
    }

    private static func anyTangent(_ n: SIMD3<Float>) -> SIMD3<Float> {
        let helper: SIMD3<Float> = abs(n.x) < 0.9 ? [1, 0, 0] : [0, 1, 0]
        return simd_normalize(simd_cross(n, helper))
    }

    // MARK: lattice compatibility

    /// `snap` for callers outside the optimization.
    @inline(__always)
    func snap(_ p: SIMD3<Float>, origin: SIMD3<Float>, a: SIMD3<Float>, b: SIMD3<Float>) -> SIMD3<Float> {
        Level.snap(p, origin: origin, a: a, b: b, rho: rho)
    }

    /// One level of the optimization hierarchy: nodes with a position (`x`), a normal, a lattice
    /// orientation, a lattice anchor and neighbors.
    struct Level {
        var x: [SIMD3<Float>]
        var normal: [SIMD3<Float>]
        var orientation: [SIMD3<Float>]
        var weight: [Float]
        var adjacency: [[Int]]
        var anchor: [SIMD3<Float>] = []
        /// Point a pinned node's lattice row must pass through (guide curves), or nil.
        var pin: [SIMD3<Float>?] = []

        @inline(__always)
        static func snap(_ p: SIMD3<Float>, origin: SIMD3<Float>, a: SIMD3<Float>, b: SIMD3<Float>, rho: Float) -> SIMD3<Float> {
            let d = p - origin
            return origin + a * (simd_dot(d, a) / rho).rounded() * rho + b * (simd_dot(d, b) / rho).rounded() * rho
        }

        /// `j`'s lattice axes turned (by quarter turns) to line up with `i`'s first axis.
        func alignedAxes(of j: Int, to i: Int) -> (SIMD3<Float>, SIMD3<Float>) {
            let a = orientation[j], b = simd_cross(normal[j], a)
            let target = orientation[i]
            let best = [a, b, -a, -b].max { simd_dot($0, target) < simd_dot($1, target) }!
            return (best, simd_cross(normal[j], best))
        }

        /// The point of `j`'s lattice nearest `i`'s anchor; averaging these pulls lattices into agreement.
        func neighborLatticePoint(_ j: Int, seenFrom i: Int, rho: Float) -> SIMD3<Float> {
            let (a, b) = alignedAxes(of: j, to: i)
            let p = anchor[i] - normal[j] * simd_dot(normal[j], anchor[i] - x[j])
            return Level.snap(p, origin: anchor[j], a: a, b: b, rho: rho)
        }

        mutating func optimize(iterations: Int, rho: Float) {
            for _ in 0..<iterations {
                for i in 0..<x.count {
                    let nbrs = adjacency[i]
                    guard !nbrs.isEmpty else { continue }
                    var sum = SIMD3<Float>.zero
                    for j in nbrs { sum += neighborLatticePoint(j, seenFrom: i, rho: rho) }
                    var p = sum / Float(nbrs.count)
                    p -= normal[i] * simd_dot(normal[i], p - x[i])               // back onto the tangent plane
                    let a = orientation[i], b = simd_cross(normal[i], a)
                    if let g = pin[i] { p += b * simd_dot(g - p, b) }            // the row across the guide stays on it
                    anchor[i] = Level.snap(x[i], origin: p, a: a, b: b, rho: rho)   // same lattice, kept near the node
                }
            }
        }

        /// Pairs each node with its nearest unmatched neighbor; returns the coarser level and the fine→coarse map.
        func coarsened() -> (Level, [Int]) {
            let n = x.count
            var match = [Int](repeating: -1, count: n)
            let order = (0..<n).sorted { weight[$0] < weight[$1] }               // light nodes pair first
            for i in order where match[i] < 0 {
                var best = -1, bestD = Float.infinity
                for j in adjacency[i] where match[j] < 0 && j != i {
                    let facing = simd_dot(normal[i], normal[j])
                    guard facing > 0.5 else { continue }
                    let d = simd_distance(x[i], x[j]) / max(facing, 0.1)
                    if d < bestD { bestD = d; best = j }
                }
                if best >= 0 { match[i] = best; match[best] = i }
            }
            var parent = [Int](repeating: -1, count: n)
            var cx: [SIMD3<Float>] = [], cn: [SIMD3<Float>] = [], co: [SIMD3<Float>] = [], cw: [Float] = []
            var cpin: [SIMD3<Float>?] = []
            for i in 0..<n where parent[i] < 0 {
                let id = cx.count
                parent[i] = id
                if match[i] >= 0 { parent[match[i]] = id }
                let group = match[i] >= 0 ? [i, match[i]] : [i]
                let w = group.reduce(Float(0)) { $0 + weight[$1] }
                var px = SIMD3<Float>.zero, pn = SIMD3<Float>.zero
                for g in group { px += x[g] * weight[g]; pn += normal[g] * weight[g] }
                px /= max(w, 1e-12)
                let nl = simd_length(pn)
                let pnorm = nl > 1e-9 ? pn / nl : normal[i]
                var o = orientation[i]
                if group.count == 2 {
                    let (a, _) = alignedAxes(of: group[1], to: group[0])
                    o += a
                    o -= pnorm * simd_dot(pnorm, o)
                    o = simd_length(o) > 1e-6 ? simd_normalize(o) : orientation[i] - pnorm * simd_dot(pnorm, orientation[i])
                } else {
                    o -= pnorm * simd_dot(pnorm, o)
                }
                var chosenPin: SIMD3<Float>?
                if let g = group.first(where: { pin[$0] != nil }) {
                    chosenPin = pin[g]
                    let po = orientation[g] - pnorm * simd_dot(pnorm, orientation[g])
                    if simd_length(po) > 1e-6 { o = simd_normalize(po) }
                }
                cx.append(px); cn.append(pnorm); co.append(simd_length(o) > 1e-9 ? simd_normalize(o) : o); cw.append(w)
                cpin.append(chosenPin)
            }
            var adj = [Set<Int>](repeating: [], count: cx.count)
            for i in 0..<n { for j in adjacency[i] where parent[i] != parent[j] { adj[parent[i]].insert(parent[j]) } }
            var coarse = Level(x: cx, normal: cn, orientation: co, weight: cw, adjacency: adj.map { Array($0).sorted() })
            coarse.pin = cpin
            return (coarse, parent)
        }
    }

    // MARK: optimization

    /// Coarse-to-fine: nodes are merged in pairs until few remain, the lattice is agreed on there
    /// first, and each finer level starts from its parent's lattice. A single fine level alone
    /// converges far too slowly to one lattice across the whole surface.
    private mutating func optimize(iterations: Int, pins: [SIMD3<Float>?]) {
        let n = topology.positions.count
        var levels = [Level(x: topology.positions, normal: topology.vertexNormal, orientation: orientation,
                            weight: topology.vertexAreaWeights(), adjacency: adjacency)]
        levels[0].pin = pins
        var parents: [[Int]] = []
        while levels.last!.x.count > 150 {
            let (coarse, parent) = levels.last!.coarsened()
            guard coarse.x.count < levels.last!.x.count * 9 / 10 else { break }
            levels.append(coarse); parents.append(parent)
        }
        // Random (deterministic) lattice offsets on the coarsest level break the symmetry.
        var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
        func random() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float((seed >> 40) & 0xFFFF) / 65536 - 0.5
        }
        var top = levels.count - 1
        levels[top].anchor = (0..<levels[top].x.count).map { v in
            let l = levels[top]
            return l.x[v] + l.orientation[v] * random() * rho + simd_cross(l.normal[v], l.orientation[v]) * random() * rho
        }
        levels[top].optimize(iterations: iterations, rho: rho)
        while top > 0 {
            top -= 1
            let parent = parents[top]
            let coarse = levels[top + 1]
            levels[top].anchor = (0..<levels[top].x.count).map { v in
                // Inherit the parent's lattice, anchored near this node.
                let pa = coarse.anchor[parent[v]]
                let l = levels[top]
                let a = l.orientation[v], b = simd_cross(l.normal[v], a)
                let (pa1, pb1) = coarse.alignedAxesToward(parent[v], direction: a)
                _ = (b, pb1)
                let p = pa - l.normal[v] * simd_dot(l.normal[v], pa - l.x[v])
                return Level.snap(l.x[v], origin: p, a: pa1, b: simd_cross(l.normal[v], pa1), rho: rho)
            }
            levels[top].optimize(iterations: iterations, rho: rho)
        }
        _ = n
        position = levels[0].anchor
    }
}

extension PositionField.Level {
    /// The node's lattice axes turned to line up best with `direction`.
    func alignedAxesToward(_ i: Int, direction: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) {
        let a = orientation[i], b = simd_cross(normal[i], a)
        let best = [a, b, -a, -b].max { simd_dot($0, direction) < simd_dot($1, direction) }!
        return (best, simd_cross(normal[i], best))
    }
}
