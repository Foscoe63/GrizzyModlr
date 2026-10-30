import Foundation
import simd

/// Field-aligned quad remeshing: cross field → lattice position field → lattice points shared by
/// neighboring vertices become output vertices → 4-cycles of lattice edges become quads → small
/// gaps are filled and the vertices are projected back onto the input surface. Returns nil when the
/// result isn't a closed manifold, so callers can fall back to the axis-aligned remesh.
enum FlowRemesh {
    /// Diagnostics hook for tests and smoke runs.
    /// Debug: cluster centres and lattice edges of the last extraction.
    nonisolated(unsafe) static var debugGraph: (points: [SIMD3<Float>], edges: [(Int, Int)])?
    nonisolated(unsafe) static var trace: ((String) -> Void)?

    struct Options {
        var targetQuads = 2000
        var featureAngle: Float = 35
        var fieldIterations = 400
        var positionIterations = 120
        /// Gaps bounded by at most this many edges are filled with triangles.
        var maxHoleEdges = 8
        var guides: [FlowGuide] = []
        var guideStrength: Float = 40
    }

    struct Output {
        var positions: [SIMD3<Float>]
        var faces: [SIMD3<UInt32>]          // each quad is two triangles
        var quads: [SIMD4<UInt32>]
        var filledTriangles: Int
        /// Share of vertices with exactly four edges.
        var regularVertexShare: Float
    }

    static func remesh(positions: [SIMD3<Float>], faces: [SIMD3<UInt32>], options o: Options = Options()) -> Output? {
        guard !faces.isEmpty, o.targetQuads >= 8 else { return nil }
        let area = faces.reduce(Float(0)) { acc, f in
            acc + simd_length(simd_cross(positions[Int(f.y)] - positions[Int(f.x)],
                                         positions[Int(f.z)] - positions[Int(f.x)])) / 2
        }
        guard area > 1e-12 else { return nil }
        let rho = (area / Float(o.targetQuads)).squareRoot()

        // Working mesh: fine enough that several vertices fall in every lattice cell.
        var wp = positions, wf = faces
        let wanted = Int(area / (0.433 * pow(rho / 2.8, 2)))
        if faces.count > wanted * 3 / 2 {
            let r = MeshDecimator.decimate(positions: positions, faces: faces, faceBudget: min(max(wanted, 2000), 80_000))
            if !r.faces.isEmpty { wp = r.positions; wf = r.faces }
        }
        // Uniform sampling: flat areas keep very long triangles after decimation, and there the
        // lattice neighbours are too far apart to ever become edges. Split only the long edges.
        var guardRounds = 0
        while guardRounds < 8, wf.count < 300_000 {
            let (np, nf, split) = refineLongEdges(wp, wf, maxLength: rho / 1.6)
            guard split else { break }
            (wp, wf) = (np, nf); guardRounds += 1
        }

        let topology = MeshTopology(positions: wp, faces: wf)
        var fieldOptions = CrossField.Options()
        fieldOptions.featureAngle = o.featureAngle
        fieldOptions.iterations = o.fieldIterations
        fieldOptions.guides = o.guides
        fieldOptions.guideWeight = o.guideStrength
        let field = CrossField.compute(topology: topology, options: fieldOptions)
        let pf = PositionField(field: field, rho: rho, iterations: o.positionIterations, guides: o.guides)

        guard var mesh = extract(pf, maxHoleEdges: o.maxHoleEdges) else { return nil }
        // Back onto the input surface.
        let closest = ClosestPoint(positions: positions, faces: faces)
        mesh.positions = mesh.positions.map { closest.nearest(to: $0) }
        relaxFolds(&mesh, closest: closest)
        let closed = isClosedManifold(mesh.positions.count, mesh.quads, mesh.fillTriangles)
        if trace != nil {
            var c: [UInt64: Int] = [:]
            for q in mesh.quads { for (a, b) in [(q.x, q.y), (q.y, q.z), (q.z, q.w), (q.w, q.x)] { c[UInt64(a) << 32 | UInt64(b), default: 0] += 1 } }
            for t in mesh.fillTriangles { for (a, b) in [(t.x, t.y), (t.y, t.z), (t.z, t.x)] { c[UInt64(a) << 32 | UInt64(b), default: 0] += 1 } }
            let dup = c.values.filter { $0 > 1 }.count
            let open = c.keys.filter { c[UInt64($0 & 0xFFFF_FFFF) << 32 | $0 >> 32] == nil }.count
            trace?("duplicate directed edges \(dup), open \(open), tris \(mesh.fillTriangles.count)")
        }
        trace?("closed manifold: \(closed)")
        guard closed else { return nil }

        var tris: [SIMD3<UInt32>] = []
        for q in mesh.quads {
            let p = mesh.positions
            let d1 = simd_distance(p[Int(q.x)], p[Int(q.z)]), d2 = simd_distance(p[Int(q.y)], p[Int(q.w)])
            if d1 <= d2 { tris += [[q.x, q.y, q.z], [q.x, q.z, q.w]] } else { tris += [[q.x, q.y, q.w], [q.y, q.z, q.w]] }
        }
        tris += mesh.fillTriangles
        var valence = [Int](repeating: 0, count: mesh.positions.count)
        var seen = Set<UInt64>()
        func edge(_ a: UInt32, _ b: UInt32) {
            let k = UInt64(min(a, b)) << 32 | UInt64(max(a, b))
            if seen.insert(k).inserted { valence[Int(a)] += 1; valence[Int(b)] += 1 }
        }
        for q in mesh.quads { edge(q.x, q.y); edge(q.y, q.z); edge(q.z, q.w); edge(q.w, q.x) }
        for t in mesh.fillTriangles { edge(t.x, t.y); edge(t.y, t.z); edge(t.z, t.x) }
        let regular = valence.filter { $0 == 4 }.count
        return Output(positions: mesh.positions, faces: tris, quads: mesh.quads,
                      filledTriangles: mesh.fillTriangles.count,
                      regularVertexShare: Float(regular) / Float(max(valence.count, 1)))
    }

    // MARK: metrics

    struct Metrics {
        /// Share of faces that belong to quads (the rest are triangles).
        var quadShare: Float
        /// Share of vertices with exactly four edges.
        var regularShare: Float
        /// Mean distance of a quad corner's angle from 90°, in degrees.
        var cornerDeviation: Float
        /// Mean distance from face centres to the input surface, in mean edge lengths.
        var surfaceSag: Float
        /// Largest such distance, in mean edge lengths.
        var maxSurfaceSag: Float
        /// Share of faces whose normal opposes the input surface.
        var foldedShare: Float
        var meanEdge: Float
    }

    static func metrics(of out: Output, surface positions: [SIMD3<Float>], faces: [SIMD3<UInt32>]) -> Metrics {
        let closest = ClosestPoint(positions: positions, faces: faces)
        let p = out.positions
        var edgeSum: Float = 0, edgeCount = 0
        var seen = Set<UInt64>()
        func edge(_ a: UInt32, _ b: UInt32) {
            if seen.insert(UInt64(min(a, b)) << 32 | UInt64(max(a, b))).inserted {
                edgeSum += simd_distance(p[Int(a)], p[Int(b)]); edgeCount += 1
            }
        }
        for f in out.faces { edge(f.x, f.y); edge(f.y, f.z); edge(f.z, f.x) }
        let meanEdge = edgeSum / Float(max(edgeCount, 1))
        var devSum: Float = 0, devCount = 0
        for q in out.quads {
            let c = [q.x, q.y, q.z, q.w].map { p[Int($0)] }
            for i in 0..<4 {
                let a = c[(i + 3) % 4] - c[i], b = c[(i + 1) % 4] - c[i]
                let cosA = simd_dot(a, b) / max(simd_length(a) * simd_length(b), 1e-12)
                devSum += abs(acos(max(-1, min(1, cosA))) * 180 / .pi - 90); devCount += 1
            }
        }
        var sagSum: Float = 0, sagMax: Float = 0, folded = 0
        for f in out.faces {
            let (a, b, c) = (p[Int(f.x)], p[Int(f.y)], p[Int(f.z)])
            let centre = (a + b + c) / 3
            let sag = simd_distance(centre, closest.nearest(to: centre)) / max(meanEdge, 1e-12)
            sagSum += sag; sagMax = max(sagMax, sag)
            if simd_dot(simd_cross(b - a, c - a), closest.normal(at: centre)) < 0 { folded += 1 }
        }
        return Metrics(quadShare: Float(out.quads.count * 2) / Float(max(out.faces.count, 1)),
                       regularShare: out.regularVertexShare,
                       cornerDeviation: devSum / Float(max(devCount, 1)),
                       surfaceSag: sagSum / Float(max(out.faces.count, 1)), maxSurfaceSag: sagMax,
                       foldedShare: Float(folded) / Float(max(out.faces.count, 1)), meanEdge: meanEdge)
    }

    // MARK: fold repair

    /// Faces whose normal points against the input surface (folds from projecting hole caps onto
    /// curved or concave areas) are relaxed: their vertices move toward their neighbours' average and
    /// are projected back, repeatedly, until no fold is left or the rounds run out.
    static func relaxFolds(_ mesh: inout Extracted, closest: ClosestPoint) {
        var polys: [[UInt32]] = mesh.quads.map { [$0.x, $0.y, $0.z, $0.w] } + mesh.fillTriangles.map { [$0.x, $0.y, $0.z] }
        var nbr = [Set<UInt32>](repeating: [], count: mesh.positions.count)
        for f in polys { for i in 0..<f.count { nbr[Int(f[i])].insert(f[(i + 1) % f.count]); nbr[Int(f[(i + 1) % f.count])].insert(f[i]) } }
        func flipped(_ f: [UInt32]) -> Bool {
            let p = f.map { mesh.positions[Int($0)] }
            var n = SIMD3<Float>.zero
            for i in 1..<(p.count - 1) { n += simd_cross(p[i] - p[0], p[i + 1] - p[0]) }
            let centre = p.reduce(.zero, +) / Float(p.count)
            return simd_dot(n, closest.normal(at: centre)) < 0
        }
        var bad = polys.filter(flipped)
        trace?("folded faces: \(bad.count) of \(polys.count)")
        if bad.count * 2 > polys.count {                       // the whole mesh winds inward
            mesh.quads = mesh.quads.map { [$0.x, $0.w, $0.z, $0.y] }
            mesh.fillTriangles = mesh.fillTriangles.map { [$0.x, $0.z, $0.y] }
            polys = mesh.quads.map { [$0.x, $0.y, $0.z, $0.w] } + mesh.fillTriangles.map { [$0.x, $0.y, $0.z] }
            bad = polys.filter(flipped)
            trace?("after reversing winding: \(bad.count)")
        }
        for round in 0..<80 where !bad.isEmpty {
            let step: Float = round < 20 ? 0.5 : 0.85
            var touched = Set<UInt32>()
            for f in bad { for v in f { touched.insert(v); nbr[Int(v)].forEach { touched.insert($0) } } }
            var next = mesh.positions
            for v in touched where !nbr[Int(v)].isEmpty {
                let avg = nbr[Int(v)].reduce(SIMD3<Float>.zero) { $0 + mesh.positions[Int($1)] } / Float(nbr[Int(v)].count)
                next[Int(v)] = closest.nearest(to: mesh.positions[Int(v)] * (1 - step) + avg * step)
            }
            mesh.positions = next
            bad = polys.filter(flipped)
            if round % 10 == 9 { trace?("fold repair round \(round): \(bad.count) left") }
        }
        trace?("folded faces after repair: \(bad.count)")
    }

    // MARK: extraction

    struct Extracted {
        var positions: [SIMD3<Float>]
        var quads: [SIMD4<UInt32>]
        var fillTriangles: [SIMD3<UInt32>]
    }

    static func extract(_ pf: PositionField, maxHoleEdges: Int) -> Extracted? {
        let n = pf.topology.positions.count
        let rho = pf.rho
        // 1. Every vertex votes for the lattice point of its own lattice nearest to the vertex itself;
        // adjacent vertices voting for (nearly) the same point become one output vertex.
        var rep = [SIMD3<Float>](repeating: .zero, count: n)
        for i in 0..<n {
            let a = pf.orientation[i], b = simd_cross(pf.normal[i], a)
            rep[i] = pf.snap(pf.topology.positions[i], origin: pf.position[i], a: a, b: b)
        }
        if let trace {
            var offP: Float = 0, offR: Float = 0, close = 0, edges = 0
            for i in 0..<n {
                offP += simd_distance(pf.position[i], pf.topology.positions[i]) / rho
                offR += simd_distance(rep[i], pf.topology.positions[i]) / rho
                for j in pf.adjacency[i] where j > i { edges += 1; if simd_distance(rep[i], rep[j]) < 0.3 * rho { close += 1 } }
            }
            trace("rho \(rho) avg|p-x|/rho \(offP / Float(n)) avg|rep-x|/rho \(offR / Float(n)) edges merging \(close)/\(edges)")
        }
        var parent = Array(0..<n)
        func find(_ x: Int) -> Int {
            var r = x
            while parent[r] != r { r = parent[r] }
            var c = x
            while parent[c] != r { let nx = parent[c]; parent[c] = r; c = nx }
            return r
        }
        for i in 0..<n { for j in pf.adjacency[i] where j > i {
            if simd_distance(rep[i], rep[j]) < 0.3 * rho { parent[find(i)] = find(j) }
        } }
        var clusterOf = [Int](repeating: -1, count: n)
        var members: [[Int]] = []
        var cpos: [SIMD3<Float>] = [], cnormal: [SIMD3<Float>] = [], caxis: [SIMD3<Float>] = []
        // Rounds of: group by union-find, place each group at the mean of its votes, then merge groups
        // whose centers are closer than half a lattice step (true lattice points are a full step apart).
        for round in 0..<4 {
            clusterOf = [Int](repeating: -1, count: n)
            members = []
            for i in 0..<n {
                let r = find(i)
                if clusterOf[r] < 0 { clusterOf[r] = members.count; members.append([]) }
                clusterOf[i] = clusterOf[r]
                members[clusterOf[i]].append(i)
            }
            let m = members.count
            cpos = [SIMD3<Float>](repeating: .zero, count: m)
            cnormal = [SIMD3<Float>](repeating: .zero, count: m)
            caxis = [SIMD3<Float>](repeating: [1, 0, 0], count: m)
            for c in 0..<m {
                for v in members[c] { cpos[c] += rep[v]; cnormal[c] += pf.normal[v] }
                cpos[c] /= Float(members[c].count)
                let l = simd_length(cnormal[c]); cnormal[c] = l > 1e-9 ? cnormal[c] / l : [0, 0, 1]
                let a = pf.orientation[members[c][0]]
                let p = a - cnormal[c] * simd_dot(a, cnormal[c])
                caxis[c] = simd_length(p) > 1e-6 ? simd_normalize(p) : a
            }
            guard round < 3 else { break }
            var merged = false
            for i in 0..<n { for j in pf.adjacency[i] where j > i {
                let (ci, cj) = (clusterOf[i], clusterOf[j])
                if ci != cj, simd_distance(cpos[ci], cpos[cj]) < 0.5 * rho { parent[find(i)] = find(j); merged = true }
            } }
            if !merged { break }
        }
        let m = members.count
        trace?("clusters \(m) of \(n) vertices")
        // 2. Candidate lattice edges between clusters of adjacent vertices, then drop diagonals: in a
        // square grid no three clusters are mutually adjacent, so the longest edge of any triangle is
        // a diagonal.
        var nbrs = [Set<Int>](repeating: [], count: m)
        var crossing = 0, lengthOK = 0
        for i in 0..<n { for j in pf.adjacency[i] where j > i {
            let (ci, cj) = (clusterOf[i], clusterOf[j])
            guard ci != cj else { continue }
            crossing += 1
            let d = cpos[cj] - cpos[ci]
            let len = simd_length(d)
            guard len >= 0.4 * rho, len <= 1.9 * rho else { continue }
            lengthOK += 1
            func aligned(_ c: Int) -> Bool {
                let a = caxis[c], b = simd_cross(cnormal[c], a)
                return max(abs(simd_dot(d, a)), abs(simd_dot(d, b))) >= 0.8 * len
            }
            guard aligned(ci) || aligned(cj) else { continue }
            nbrs[ci].insert(cj); nbrs[cj].insert(ci)
        } }
        trace?("crossing mesh edges \(crossing), length ok \(lengthOK)")
        for c in 0..<m {
            for a in nbrs[c] where a > c {
                for b in nbrs[c] where b > a && nbrs[a].contains(b) {
                    let edges = [(c, a), (c, b), (a, b)]
                    let longest = edges.max { simd_distance(cpos[$0.0], cpos[$0.1]) < simd_distance(cpos[$1.0], cpos[$1.1]) }!
                    nbrs[longest.0].remove(longest.1); nbrs[longest.1].remove(longest.0)
                }
            }
        }
        if trace != nil { debugGraph = (cpos, (0..<m).flatMap { c in nbrs[c].filter { $0 > c }.map { (c, $0) } }) }
        trace?("lattice edges \(nbrs.reduce(0) { $0 + $1.count } / 2), nodes with >=3 nbrs \(nbrs.filter { $0.count >= 3 }.count)")
        // 3. Quads from 4-cycles around each node, best-fitting first.
        struct Candidate { var quad: SIMD4<UInt32>; var score: Float }
        var candidates: [Candidate] = []
        var seen = Set<[Int]>()
        var pairsTried = 0, angleOK = 0, hasW = 0
        for c in 0..<m {
            let nrm = cnormal[c], e1 = caxis[c], e2 = simd_cross(nrm, e1)
            let around = nbrs[c].sorted { atan2(simd_dot(cpos[$0] - cpos[c], e2), simd_dot(cpos[$0] - cpos[c], e1))
                                        < atan2(simd_dot(cpos[$1] - cpos[c], e2), simd_dot(cpos[$1] - cpos[c], e1)) }
            guard around.count >= 3 else { continue }
            for k in 0..<around.count {
                let a = around[k], b = around[(k + 1) % around.count]
                pairsTried += 1
                let da = cpos[a] - cpos[c], db = cpos[b] - cpos[c]
                let cosAngle = simd_dot(da, db) / max(simd_length(da) * simd_length(db), 1e-12)
                guard cosAngle > -0.55, cosAngle < 0.55 else { continue }                // roughly 57°…123°
                angleOK += 1
                let ideal = cpos[a] + cpos[b] - cpos[c]
                var best: (w: Int, d: Float)?
                for w in nbrs[a].union(nbrs[b]) where w != c && w != a && w != b {
                    let d = simd_distance(cpos[w], ideal)
                    if best == nil || d < best!.d { best = (w, d) }
                }
                if best != nil { hasW += 1 }
                guard let (w, d) = best, d < 0.5 * rho else { continue }
                let key = [c, a, w, b].sorted()
                guard seen.insert(key).inserted else { continue }
                // Counter-clockwise about the surface normal: a then b follows the sorted angle order.
                candidates.append(Candidate(quad: [UInt32(c), UInt32(a), UInt32(w), UInt32(b)], score: d + abs(cosAngle) * 0.2 * rho))
            }
        }
        trace?("pairs \(pairsTried) angleOK \(angleOK) hasW \(hasW)")
        candidates.sort { $0.score < $1.score }
        var directed = Set<UInt64>()
        func dkey(_ a: UInt32, _ b: UInt32) -> UInt64 { UInt64(a) << 32 | UInt64(b) }
        var quads: [SIMD4<UInt32>] = []
        for cand in candidates {
            let q = cand.quad
            let es = [dkey(q.x, q.y), dkey(q.y, q.z), dkey(q.z, q.w), dkey(q.w, q.x)]
            if es.contains(where: directed.contains) { continue }
            es.forEach { directed.insert($0) }
            quads.append(q)
        }

        trace?("candidates \(candidates.count) accepted \(quads.count)")
        // 4. Fill small gaps; give up on large ones.
        var fill: [SIMD3<UInt32>] = []
        var next: [UInt32: [UInt32]] = [:]
        let basePositions = cpos.count
        var ring: [SIMD4<UInt32>] = []
        for round in 0..<6 {
        fill = []; next = [:]; ring = []
        cpos.removeSubrange(basePositions...)
        directed = Set(quads.flatMap { q in [dkey(q.x, q.y), dkey(q.y, q.z), dkey(q.z, q.w), dkey(q.w, q.x)] })
        for q in quads {
            for (a, b) in [(q.x, q.y), (q.y, q.z), (q.z, q.w), (q.w, q.x)] where !directed.contains(dkey(b, a)) {
                next[a, default: []].append(b)
            }
        }
        // The loop runs along existing faces' edges, so the cap winds the other way.
        func cap(_ loop: [UInt32]) { fillLoop(Array(loop.reversed()), depth: 0) }
        func fillLoop(_ r: [UInt32], depth: Int) {
            if r.count == 4 { ring.append([r[0], r[1], r[2], r[3]]); return }
            if r.count > 12, depth < 30 {
                // Large gap: peel off a ring of quads one lattice step wide and fill what is left.
                let centre = r.reduce(SIMD3<Float>.zero) { $0 + cpos[Int($1)] } / Float(r.count)
                let radius = r.reduce(Float(0)) { $0 + simd_distance(cpos[Int($1)], centre) } / Float(r.count)
                let perimeter = zip(r, r.dropFirst() + [r[0]]).reduce(Float(0)) { $0 + simd_distance(cpos[Int($1.0)], cpos[Int($1.1)]) }
                let shrink = 1 - rho / max(radius, 1e-9)
                // Keep peeling only while the inner ring's vertices stay a sensible distance apart;
                // otherwise the rings pinch into a pole of squeezed quads.
                if radius > 1.5 * rho, perimeter * shrink / Float(r.count) >= 0.6 * rho {
                    var inner: [UInt32] = []
                    for v in r {
                        inner.append(UInt32(cpos.count))
                        cpos.append(centre + (cpos[Int(v)] - centre) * shrink)
                    }
                    for i in 0..<r.count {
                        let j = (i + 1) % r.count
                        ring.append([r[i], r[j], inner[j], inner[i]])
                    }
                    fillLoop(inner, depth: depth + 1)
                    return
                }
            }
            if r.count <= 80 {
                // Minimum total diagonal length: short, even triangles and no high-valence centre.
                let n = r.count
                func dist(_ i: Int, _ j: Int) -> Float { simd_distance(cpos[Int(r[i])], cpos[Int(r[j])]) }
                var cost = [[Float]](repeating: [Float](repeating: 0, count: n), count: n)
                var split = [[Int]](repeating: [Int](repeating: 0, count: n), count: n)
                if n >= 3 {
                    for len in 2..<n {
                        for i in 0..<(n - len) {
                            let j = i + len
                            var best = Float.infinity, bk = i + 1
                            for k in (i + 1)..<j {
                                let c = cost[i][k] + cost[k][j] + (k - i > 1 ? dist(i, k) : 0) + (j - k > 1 ? dist(k, j) : 0)
                                if c < best { best = c; bk = k }
                            }
                            cost[i][j] = best; split[i][j] = bk
                        }
                    }
                }
                var stack = [(0, n - 1)]
                while let (i, j) = stack.popLast() {
                    guard j - i >= 2 else { continue }
                    let k = split[i][j]
                    fill.append([r[i], r[k], r[j]])
                    stack.append((i, k)); stack.append((k, j))
                }
            } else {
                // Larger gaps: a fan around a new vertex at the loop's centre (projected onto the surface later).
                let centre = UInt32(cpos.count)
                cpos.append(r.reduce(SIMD3<Float>.zero) { $0 + cpos[Int($1)] } / Float(r.count))
                for k in 0..<r.count { fill.append([centre, r[k], r[(k + 1) % r.count]]) }
            }
        }
        // Follow boundary edges; whenever the walk returns to a vertex already on the path, that
        // stretch is a closed loop (touching loops pinch at shared vertices) and is capped on its own.
        var used = Set<UInt64>()
        for start in next.keys.sorted() {
            for first in next[start] ?? [] where !used.contains(dkey(start, first)) {
                used.insert(dkey(start, first))
                var path = [start, first]
                var steps = 0
                while path.count > 1, steps < 100_000 {
                    steps += 1
                    let cur = path[path.count - 1]
                    if let idx = path.dropLast().firstIndex(of: cur) {
                        let loop = Array(path[idx..<(path.count - 1)])
                        if loop.count >= 3 { cap(loop) }
                        path.removeLast(path.count - 1 - idx)
                        if path.count == 1 { break }
                        continue
                    }
                    guard let nxt = next[cur]?.first(where: { !used.contains(dkey(cur, $0)) }) else { break }
                    used.insert(dkey(cur, nxt)); path.append(nxt)
                }
            }
        }
        // Capping can reuse a directed edge where loops touch; drop the quads on such edges and cap again.
        var count: [UInt64: Int] = [:]
        for q in quads { for (x, y) in [(q.x, q.y), (q.y, q.z), (q.z, q.w), (q.w, q.x)] { count[dkey(x, y), default: 0] += 1 } }
        for t in fill { for (x, y) in [(t.x, t.y), (t.y, t.z), (t.z, t.x)] { count[dkey(x, y), default: 0] += 1 } }
        for q in ring { for (x, y) in [(q.x, q.y), (q.y, q.z), (q.z, q.w), (q.w, q.x)] { count[dkey(x, y), default: 0] += 1 } }
        let bad = Set(count.filter { $0.value > 1 }.keys)
        trace?("round \(round): duplicate edges \(bad.count)")
        if bad.isEmpty || round == 5 { break }
        // Fill-only duplicates can't be cured by removing quads on the edge itself, so widen to its endpoints
        // on later rounds.
        var badVerts = Set<UInt32>()
        if round >= 1 { for k in bad { badVerts.insert(UInt32(k >> 32)); badVerts.insert(UInt32(k & 0xFFFF_FFFF)) } }
        quads.removeAll { q in
            [(q.x, q.y), (q.y, q.z), (q.z, q.w), (q.w, q.x)].contains { bad.contains(dkey($0.0, $0.1)) }
                || badVerts.contains(q.x) || badVerts.contains(q.y) || badVerts.contains(q.z) || badVerts.contains(q.w)
        }
        }
        trace?("boundary half-edges \(next.values.reduce(0) { $0 + $1.count }), fill tris \(fill.count)")
        // Drop nodes no face uses.
        var remap = [Int32](repeating: -1, count: cpos.count)
        var outPos: [SIMD3<Float>] = []
        func map(_ v: UInt32) -> UInt32 {
            if remap[Int(v)] < 0 { remap[Int(v)] = Int32(outPos.count); outPos.append(cpos[Int(v)]) }
            return UInt32(remap[Int(v)])
        }
        let outQuads = (quads + ring).map { SIMD4(map($0.x), map($0.y), map($0.z), map($0.w)) }
        let outFill = fill.map { SIMD3(map($0.x), map($0.y), map($0.z)) }
        guard !outQuads.isEmpty else { return nil }
        return Extracted(positions: outPos, quads: outQuads, fillTriangles: outFill)
    }

    // MARK: helpers

    /// True when every directed edge is used exactly once and has an opposite partner.
    static func isClosedManifold(_ vertexCount: Int, _ quads: [SIMD4<UInt32>], _ tris: [SIMD3<UInt32>]) -> Bool {
        var count: [UInt64: Int] = [:]
        func add(_ a: UInt32, _ b: UInt32) { count[UInt64(a) << 32 | UInt64(b), default: 0] += 1 }
        for q in quads { add(q.x, q.y); add(q.y, q.z); add(q.z, q.w); add(q.w, q.x) }
        for t in tris { add(t.x, t.y); add(t.y, t.z); add(t.z, t.x) }
        for (k, c) in count {
            guard c == 1 else { return false }
            let (a, b) = (UInt32(k >> 32), UInt32(k & 0xFFFF_FFFF))
            guard count[UInt64(b) << 32 | UInt64(a)] == 1 else { return false }
        }
        return !count.isEmpty
    }

    /// Splits every edge longer than `maxLength` at its midpoint (triangles are re-cut consistently).
    static func refineLongEdges(_ p: [SIMD3<Float>], _ f: [SIMD3<UInt32>], maxLength: Float)
        -> ([SIMD3<Float>], [SIMD3<UInt32>], Bool) {
        var pos = p, mid: [UInt64: UInt32] = [:]
        func key(_ a: UInt32, _ b: UInt32) -> UInt64 { UInt64(min(a, b)) << 32 | UInt64(max(a, b)) }
        func long(_ a: UInt32, _ b: UInt32) -> Bool { simd_distance(p[Int(a)], p[Int(b)]) > maxLength }
        func midpoint(_ a: UInt32, _ b: UInt32) -> UInt32 {
            let k = key(a, b)
            if let m = mid[k] { return m }
            pos.append((pos[Int(a)] + pos[Int(b)]) / 2)
            mid[k] = UInt32(pos.count - 1)
            return mid[k]!
        }
        var out: [SIMD3<UInt32>] = []
        var any = false
        for t in f {
            var (x, y, z) = (t.x, t.y, t.z)
            var s0 = long(x, y), s1 = long(y, z), s2 = long(z, x)
            let n = (s0 ? 1 : 0) + (s1 ? 1 : 0) + (s2 ? 1 : 0)
            if n == 0 { out.append(t); continue }
            any = true
            if n == 1 {
                while !s0 { (x, y, z) = (y, z, x); (s0, s1, s2) = (s1, s2, s0) }
                let m = midpoint(x, y)
                out += [[x, m, z], [m, y, z]]
            } else if n == 2 {
                while s2 { (x, y, z) = (y, z, x); (s0, s1, s2) = (s1, s2, s0) }   // unsplit edge is z→x
                let m1 = midpoint(x, y), m2 = midpoint(y, z)
                out.append([m1, y, m2])
                if simd_distance(pos[Int(x)], pos[Int(m2)]) <= simd_distance(pos[Int(m1)], pos[Int(z)]) {
                    out += [[x, m1, m2], [x, m2, z]]
                } else {
                    out += [[x, m1, z], [m1, m2, z]]
                }
            } else {
                let a = midpoint(x, y), b = midpoint(y, z), c = midpoint(z, x)
                out += [[x, a, c], [y, b, a], [z, c, b], [a, b, c]]
            }
        }
        return (pos, out, any)
    }

    static func subdivide(_ p: [SIMD3<Float>], _ f: [SIMD3<UInt32>]) -> ([SIMD3<Float>], [SIMD3<UInt32>]) {
        var pos = p, mid: [UInt64: UInt32] = [:]
        func midpoint(_ a: UInt32, _ b: UInt32) -> UInt32 {
            let k = UInt64(min(a, b)) << 32 | UInt64(max(a, b))
            if let m = mid[k] { return m }
            pos.append((pos[Int(a)] + pos[Int(b)]) / 2)
            mid[k] = UInt32(pos.count - 1)
            return mid[k]!
        }
        var out: [SIMD3<UInt32>] = []
        out.reserveCapacity(f.count * 4)
        for t in f {
            let a = midpoint(t.x, t.y), b = midpoint(t.y, t.z), c = midpoint(t.z, t.x)
            out += [[t.x, a, c], [t.y, b, a], [t.z, c, b], [a, b, c]]
        }
        return (pos, out)
    }
}

/// Nearest point on a triangle mesh, through a uniform grid over the triangles.
struct ClosestPoint {
    private let positions: [SIMD3<Float>]
    private let faces: [SIMD3<UInt32>]
    private var cells: [Int: [Int]] = [:]
    private let origin: SIMD3<Float>
    private let cell: Float
    private let dims: SIMD3<Int>

    init(positions: [SIMD3<Float>], faces: [SIMD3<UInt32>]) {
        self.positions = positions
        self.faces = faces
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for p in positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let size = hi - lo
        let target = max(8, Int(Double(faces.count).squareRoot() / 2))
        cell = max(max(size.x, max(size.y, size.z)) / Float(target), 1e-6)
        origin = lo
        dims = SIMD3(Int(size.x / cell) + 1, Int(size.y / cell) + 1, Int(size.z / cell) + 1)
        for (i, f) in faces.enumerated() {
            let a = positions[Int(f.x)], b = positions[Int(f.y)], c = positions[Int(f.z)]
            let tlo = simd_min(a, simd_min(b, c)), thi = simd_max(a, simd_max(b, c))
            let l = cellIndex(tlo), h = cellIndex(thi)
            for z in l.z...h.z { for y in l.y...h.y { for x in l.x...h.x { cells[key(x, y, z), default: []].append(i) } } }
        }
    }

    private func cellIndex(_ p: SIMD3<Float>) -> SIMD3<Int> {
        let q = (p - origin) / cell
        return SIMD3(max(0, min(dims.x - 1, Int(q.x))), max(0, min(dims.y - 1, Int(q.y))), max(0, min(dims.z - 1, Int(q.z))))
    }
    private func key(_ x: Int, _ y: Int, _ z: Int) -> Int { (z * dims.y + y) * dims.x + x }

    func nearest(to p: SIMD3<Float>) -> SIMD3<Float> { nearestFace(to: p).point }

    /// Unit normal of the surface triangle closest to `p`.
    func normal(at p: SIMD3<Float>) -> SIMD3<Float> {
        let f = faces[nearestFace(to: p).face]
        let n = simd_cross(positions[Int(f.y)] - positions[Int(f.x)], positions[Int(f.z)] - positions[Int(f.x)])
        let l = simd_length(n)
        return l > 1e-20 ? n / l : [0, 0, 1]
    }

    func nearestFace(to p: SIMD3<Float>) -> (point: SIMD3<Float>, face: Int) {
        let c = cellIndex(p)
        var best = p, bestD = Float.infinity, bestFace = 0
        var ring = 0
        let maxRing = max(dims.x, max(dims.y, dims.z))
        while ring <= maxRing {
            for z in max(0, c.z - ring)...min(dims.z - 1, c.z + ring) {
                for y in max(0, c.y - ring)...min(dims.y - 1, c.y + ring) {
                    for x in max(0, c.x - ring)...min(dims.x - 1, c.x + ring) {
                        guard max(abs(x - c.x), max(abs(y - c.y), abs(z - c.z))) == ring else { continue }
                        for i in cells[key(x, y, z)] ?? [] {
                            let f = faces[i]
                            let q = Self.closest(p, positions[Int(f.x)], positions[Int(f.y)], positions[Int(f.z)])
                            let d = simd_distance_squared(p, q)
                            if d < bestD { bestD = d; best = q; bestFace = i }
                        }
                    }
                }
            }
            if bestD.isFinite, bestD.squareRoot() <= Float(ring) * cell { break }
            ring += 1
        }
        return (best, bestFace)
    }

    /// Closest point on triangle `abc` to `p` (Ericson, Real-Time Collision Detection).
    static func closest(_ p: SIMD3<Float>, _ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) -> SIMD3<Float> {
        let ab = b - a, ac = c - a, ap = p - a
        let d1 = simd_dot(ab, ap), d2 = simd_dot(ac, ap)
        if d1 <= 0, d2 <= 0 { return a }
        let bp = p - b
        let d3 = simd_dot(ab, bp), d4 = simd_dot(ac, bp)
        if d3 >= 0, d4 <= d3 { return b }
        let vc = d1 * d4 - d3 * d2
        if vc <= 0, d1 >= 0, d3 <= 0 { return a + ab * (d1 / (d1 - d3)) }
        let cp = p - c
        let d5 = simd_dot(ab, cp), d6 = simd_dot(ac, cp)
        if d6 >= 0, d5 <= d6 { return c }
        let vb = d5 * d2 - d1 * d6
        if vb <= 0, d2 >= 0, d6 <= 0 { return a + ac * (d2 / (d2 - d6)) }
        let va = d3 * d6 - d5 * d4
        if va <= 0, d4 - d3 >= 0, d5 - d6 >= 0 { return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))) }
        let denom = 1 / (va + vb + vc)
        return a + ab * (vb * denom) + ac * (vc * denom)
    }
}
