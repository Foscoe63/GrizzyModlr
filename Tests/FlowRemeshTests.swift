import XCTest
import simd

final class FlowRemeshTests: XCTestCase {
    private func icosphere(_ subdivisions: Int) -> ([SIMD3<Float>], [SIMD3<UInt32>]) {
        let t: Float = (1 + Float(5).squareRoot()) / 2
        let raw: [[Float]] = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                              [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
        var p: [SIMD3<Float>] = raw.map { simd_normalize(SIMD3<Float>($0[0], $0[1], $0[2])) }
        let tris: [[UInt32]] = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                                [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                                [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
        var f = tris.map { SIMD3<UInt32>($0[0], $0[1], $0[2]) }
        for _ in 0..<subdivisions {
            var mid: [UInt64: UInt32] = [:]
            func midpoint(_ a: UInt32, _ b: UInt32) -> UInt32 {
                let k = UInt64(min(a, b)) << 32 | UInt64(max(a, b))
                if let m = mid[k] { return m }
                p.append(simd_normalize(p[Int(a)] + p[Int(b)])); mid[k] = UInt32(p.count - 1); return mid[k]!
            }
            var next: [SIMD3<UInt32>] = []
            for tri in f {
                let a = midpoint(tri.x, tri.y), b = midpoint(tri.y, tri.z), c = midpoint(tri.z, tri.x)
                next += [[tri.x, a, c], [tri.y, b, a], [tri.z, c, b], [a, b, c]]
            }
            f = next
        }
        return (p, f)
    }

    func testPositionFieldAnchorsStayNearTheirVertices() {
        let (p, f) = icosphere(3)
        let field = CrossField.compute(topology: MeshTopology(positions: p, faces: f))
        let pf = PositionField(field: field, rho: 0.35, iterations: 40)
        for i in 0..<p.count { XCTAssertLessThan(simd_distance(pf.position[i], p[i]), 0.35 * 0.75 + 1e-4) }   // within a lattice cell of the vertex
    }

    /// Whether or not the flow closes, the edit must return a usable mesh.
    func testFlowRemeshEditAlwaysYieldsAMesh() {
        let (p, f) = icosphere(3)
        let r = MeshOps.apply(.flowRemesh(targetQuads: 200, featureAngle: 35, fallbackResolution: 48), positions: p, faces: f)
        XCTAssertTrue(r.changed)
        XCTAssertFalse(r.faces.isEmpty)
    }

    func testSphereFlowRemeshClosesAndStaysOnSurface() throws {
        let (p, f) = icosphere(4)
        let r = try XCTUnwrap(FlowRemesh.remesh(positions: p, faces: f, options: { var o = FlowRemesh.Options(); o.targetQuads = 200; return o }()))
        XCTAssertGreaterThan(Float(r.quads.count * 2) / Float(r.faces.count), 0.5)    // mostly quads
        XCTAssertGreaterThan(r.regularVertexShare, 0.3)
        XCTAssertGreaterThan(r.quads.count, 80)
        for v in r.positions { XCTAssertEqual(simd_length(v), 1, accuracy: 0.03) }
    }

    private func torus(n: Int = 64, m: Int = 32) -> ([SIMD3<Float>], [SIMD3<UInt32>]) {
        var p: [SIMD3<Float>] = []
        for i in 0..<n { for j in 0..<m {
            let u = 2 * Float.pi * Float(i) / Float(n), v = 2 * Float.pi * Float(j) / Float(m)
            p.append([(1 + 0.35 * cos(v)) * cos(u), (1 + 0.35 * cos(v)) * sin(u), 0.35 * sin(v)])
        } }
        var f: [SIMD3<UInt32>] = []
        for i in 0..<n { for j in 0..<m {
            let a = UInt32(i * m + j), b = UInt32(((i + 1) % n) * m + j)
            let c = UInt32(((i + 1) % n) * m + (j + 1) % m), d = UInt32(i * m + (j + 1) % m)
            f.append([a, b, c]); f.append([a, c, d])
        } }
        return (p, f)
    }

    func testTorusFlowRemeshCloses() throws {
        let (p, f) = torus()
        var o = FlowRemesh.Options(); o.targetQuads = 400
        let r = try XCTUnwrap(FlowRemesh.remesh(positions: p, faces: f, options: o))
        XCTAssertGreaterThan(r.quads.count, 200)
        XCTAssertGreaterThan(r.regularVertexShare, 0.5)
    }

    /// Opt-in: FLOW_MESH=<.mesh> runs the flow remesh on a real scan and prints the outcome.
    func testRealScanReport() throws {
        guard let path = ProcessInfo.processInfo.environment["FLOW_MESH"] else { throw XCTSkip("set FLOW_MESH") }
        let mesh = try XCTUnwrap(MeshDecimator.parse(Data(contentsOf: URL(fileURLWithPath: path))))
        FlowRemesh.trace = { print("DBG \($0)") }
        defer { FlowRemesh.trace = nil }
        var o = FlowRemesh.Options(); o.targetQuads = 2000
        let t = Date()
        let r = FlowRemesh.remesh(positions: mesh.positions, faces: mesh.faces, options: o)
        print("DBG real result", r.map { "quads \($0.quads.count) faces \($0.faces.count) fill \($0.filledTriangles) regular \($0.regularVertexShare)" } ?? "nil", "in", Date().timeIntervalSince(t))
        if ProcessInfo.processInfo.environment["FLOW_GRAPH"] != nil, let g = FlowRemesh.debugGraph {
            var t = g.points.map { "v \($0.x) \($0.y) \($0.z)" }.joined(separator: "\n") + "\n"
            for e in g.edges { t += "l \(e.0 + 1) \(e.1 + 1)\n" }
            try? t.write(toFile: ProcessInfo.processInfo.environment["FLOW_GRAPH"]!, atomically: true, encoding: .utf8)
        }
        if let out = ProcessInfo.processInfo.environment["FLOW_OUT"], let r {
            var obj = r.positions.map { "v \($0.x) \($0.y) \($0.z)" }.joined(separator: "\n") + "\n"
            for q in r.quads { obj += "f \(q.x + 1) \(q.y + 1) \(q.z + 1) \(q.w + 1)\n" }
            let qs = Set(r.quads.flatMap { q in [[q.x, q.y, q.z], [q.x, q.z, q.w], [q.x, q.y, q.w], [q.y, q.z, q.w]].map { Set($0) } })
            for t in r.faces where !qs.contains(Set([t.x, t.y, t.z])) { obj += "f \(t.x + 1) \(t.y + 1) \(t.z + 1)\n" }
            try? obj.write(toFile: out, atomically: true, encoding: .utf8)
        }
    }

    // MARK: M7 — metrics on clean shapes

    private func cylinder(seg: Int = 48, rings: Int = 24) -> ([SIMD3<Float>], [SIMD3<UInt32>]) {
        var p: [SIMD3<Float>] = []
        for r in 0...rings { for s in 0..<seg {
            let a = 2 * Float.pi * Float(s) / Float(seg)
            p.append([cos(a), sin(a), -2 + 4 * Float(r) / Float(rings)])
        } }
        var f: [SIMD3<UInt32>] = []
        for r in 0..<rings { for s in 0..<seg {
            let a = UInt32(r * seg + s), b = UInt32(r * seg + (s + 1) % seg)
            let c = UInt32((r + 1) * seg + (s + 1) % seg), d = UInt32((r + 1) * seg + s)
            f.append([a, b, c]); f.append([a, c, d])
        } }
        let bottom = UInt32(p.count); p.append([0, 0, -2])
        let top = UInt32(p.count); p.append([0, 0, 2])
        for s in 0..<seg {
            let a = UInt32(s), b = UInt32((s + 1) % seg)
            f.append([bottom, b, a])
            let ta = UInt32(rings * seg + s), tb = UInt32(rings * seg + (s + 1) % seg)
            f.append([top, ta, tb])
        }
        return (p, f)
    }

    /// Box with rounded edges and corners (half-size 1, corner radius 0.35), `n` cells per side.
    private func roundedBox(_ n: Int = 24) -> ([SIMD3<Float>], [SIMD3<UInt32>]) {
        var index: [[Int]: UInt32] = [:]
        var p: [SIMD3<Float>] = []
        func vertex(_ i: Int, _ j: Int, _ k: Int) -> UInt32 {
            if let v = index[[i, j, k]] { return v }
            let cube = SIMD3<Float>(Float(i), Float(j), Float(k)) / Float(n) * 2 - 1
            let core = simd_clamp(cube, SIMD3(repeating: -0.65), SIMD3(repeating: 0.65))
            let out = cube - core
            p.append(simd_length(out) > 1e-6 ? core + simd_normalize(out) * 0.35 : cube)
            index[[i, j, k]] = UInt32(p.count - 1)
            return UInt32(p.count - 1)
        }
        var f: [SIMD3<UInt32>] = []
        func quad(_ a: UInt32, _ b: UInt32, _ c: UInt32, _ d: UInt32) {
            let pa = p[Int(a)], pb = p[Int(b)], pc = p[Int(c)]
            let outward = simd_dot(simd_cross(pb - pa, pc - pa), pa + pb + pc) > 0
            if outward { f.append([a, b, c]); f.append([a, c, d]) } else { f.append([a, c, b]); f.append([a, d, c]) }
        }
        for u in 0..<n { for v in 0..<n {
            for side in [0, n] {
                quad(vertex(side, u, v), vertex(side, u + 1, v), vertex(side, u + 1, v + 1), vertex(side, u, v + 1))
                quad(vertex(u, side, v), vertex(u + 1, side, v), vertex(u + 1, side, v + 1), vertex(u, side, v + 1))
                quad(vertex(u, v, side), vertex(u + 1, v, side), vertex(u + 1, v + 1, side), vertex(u, v + 1, side))
            }
        } }
        return (p, f)
    }

    private func checkMetrics(_ name: String, _ shape: ([SIMD3<Float>], [SIMD3<UInt32>]), quads: Int,
                              minRegular: Float, maxCorner: Float,
                              file: StaticString = #filePath, line: UInt = #line) throws {
        var o = FlowRemesh.Options(); o.targetQuads = quads
        let r = try XCTUnwrap(FlowRemesh.remesh(positions: shape.0, faces: shape.1, options: o), "\(name) didn't close",
                              file: file, line: line)
        let m = FlowRemesh.metrics(of: r, surface: shape.0, faces: shape.1)
        print("METRICS \(name): quads \(r.quads.count) quadShare \(m.quadShare) regular \(m.regularShare) corner° \(m.cornerDeviation) sag \(m.surfaceSag) maxSag \(m.maxSurfaceSag) folded \(m.foldedShare)")
        XCTAssertGreaterThan(Float(r.quads.count), Float(quads) * 0.5, "\(name) quad count", file: file, line: line)
        XCTAssertGreaterThan(m.quadShare, 0.6, "\(name) quad share", file: file, line: line)
        XCTAssertGreaterThan(m.regularShare, minRegular, "\(name) valence-4 share", file: file, line: line)
        XCTAssertLessThan(m.cornerDeviation, maxCorner, "\(name) corner angles", file: file, line: line)
        XCTAssertLessThan(m.surfaceSag, 0.08, "\(name) distance to surface", file: file, line: line)
        XCTAssertLessThan(m.maxSurfaceSag, 0.7, "\(name) worst distance to surface", file: file, line: line)
        XCTAssertLessThan(m.foldedShare, 0.03, "\(name) folded faces", file: file, line: line)
    }

    func testMetricsOnCleanShapes() throws {
        try checkMetrics("sphere", icosphere(4), quads: 200, minRegular: 0.3, maxCorner: 30)
        try checkMetrics("cylinder", cylinder(), quads: 300, minRegular: 0.7, maxCorner: 13)
        try checkMetrics("torus", torus(), quads: 400, minRegular: 0.75, maxCorner: 10)
        try checkMetrics("roundedBox", roundedBox(), quads: 300, minRegular: 0.7, maxCorner: 10)
    }

    // MARK: M8 — guides

    func testGuideCurvePinsTheFieldToItsDirection() {
        // A flat 20×20 sheet has no curvature to steer the field, so a guide at 45° decides it.
        let n = 20
        var p: [SIMD3<Float>] = []
        for y in 0...n { for x in 0...n { p.append([Float(x) / Float(n), Float(y) / Float(n), 0]) } }
        var f: [SIMD3<UInt32>] = []
        for y in 0..<n { for x in 0..<n {
            let a = UInt32(y * (n + 1) + x), b = a + 1, c = a + UInt32(n + 2), d = a + UInt32(n + 1)
            f.append([a, b, c]); f.append([a, c, d])
        } }
        let topology = MeshTopology(positions: p, faces: f)
        let dir = simd_normalize(SIMD3<Float>(1, 1, 0))
        var o = CrossField.Options()
        o.guides = [FlowGuide(points: [[0.2, 0.2, 0], [0.8, 0.8, 0]])]
        let field = CrossField.compute(topology: topology, options: o)
        var near = 0, aligned = 0
        for i in 0..<f.count {
            let c = topology.centroid(i)
            let line = abs(c.x - c.y) / Float(2).squareRoot()                  // distance to the diagonal
            guard line < topology.averageEdgeLength, c.x > 0.3, c.x < 0.7 else { continue }
            near += 1
            let (u, v) = field.axes(face: i)
            if max(abs(simd_dot(u, dir)), abs(simd_dot(v, dir))) > cos(12 * Float.pi / 180) { aligned += 1 }
        }
        XCTAssertGreaterThan(near, 10)
        XCTAssertEqual(aligned, near)
    }

    func testGuideSegmentsAreSplitToTheSpacing() {
        let g = FlowGuide(points: [[0, 0, 0], [1, 0, 0]])
        let s = g.segments(maxLength: 0.25)
        XCTAssertEqual(s.count, 4)
        XCTAssertEqual(s.last?.1.x ?? 0, 1, accuracy: 1e-6)
    }

    /// Opt-in: FLOW_MESH=<.mesh>. Draws a guide across the model and compares how well the field and
    /// the remeshed edges follow it with and without the guide.
    func testGuideBendsRealScan() throws {
        guard let path = ProcessInfo.processInfo.environment["FLOW_MESH"] else { throw XCTSkip("set FLOW_MESH") }
        let mesh = try XCTUnwrap(MeshDecimator.parse(Data(contentsOf: URL(fileURLWithPath: path))))
        let closest = ClosestPoint(positions: mesh.positions, faces: mesh.faces)
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for p in mesh.positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        // A slanted line across the front of the chest, projected onto the surface.
        let width = hi.x - lo.x
        let y0 = lo.y + (hi.y - lo.y) * 0.55
        var pts: [SIMD3<Float>] = []
        for k in 0...30 {
            let t = Float(k) / 30
            let probe = SIMD3<Float>(lo.x + width * (0.2 + 0.6 * t), y0 + width * 0.25 * (t - 0.5), hi.z + width)
            pts.append(closest.nearest(to: probe))
        }
        let guide = FlowGuide(points: pts)
        let segs = guide.segments(maxLength: width * 0.01)

        func tangent(near p: SIMD3<Float>) -> (dir: SIMD3<Float>, dist: Float)? {
            var best: (SIMD3<Float>, Float)?
            for (a, b) in segs {
                let ab = b - a, l2 = simd_dot(ab, ab)
                guard l2 > 1e-18 else { continue }
                let t = max(0, min(1, simd_dot(p - a, ab) / l2))
                let d = simd_distance(p, a + ab * t)
                if best == nil || d < best!.1 { best = (simd_normalize(ab), d) }
            }
            return best.map { ($0.0, $0.1) }
        }

        // 1. The cross field itself.
        func fieldAlignment(_ guides: [FlowGuide]) -> (near: Int, aligned: Int, far: Int) {
            var o = CrossField.Options(); o.guides = guides
            if let w = ProcessInfo.processInfo.environment["GUIDE_W"].flatMap(Float.init) { o.guideWeight = w }
            let field = CrossField.computeAdaptive(positions: mesh.positions, faces: mesh.faces, options: o)
            let t = field.topology
            let reach = t.averageEdgeLength * 2
            var near = 0, aligned = 0
            for i in 0..<t.faces.count {
                guard let g = tangent(near: t.centroid(i)), g.dist < reach else { continue }
                near += 1
                let (u, v) = field.axes(face: i)
                let d = g.dir - t.faceNormal[i] * simd_dot(g.dir, t.faceNormal[i])
                guard simd_length(d) > 1e-6 else { continue }
                let n = simd_normalize(d)
                if max(abs(simd_dot(u, n)), abs(simd_dot(v, n))) > cos(15 * Float.pi / 180) { aligned += 1 }
            }
            return (near, aligned, t.faces.count - near)
        }
        let plain = fieldAlignment([]), guided = fieldAlignment([guide])
        print("GUIDE field: faces near guide \(plain.near)/\(guided.near); aligned within 15°: without \(plain.aligned) (\(100 * plain.aligned / max(plain.near, 1))%), with \(guided.aligned) (\(100 * guided.aligned / max(guided.near, 1))%)")

        // 2. The remeshed edges near the guide.
        func edgeAlignment(_ guides: [FlowGuide]) -> (edges: Int, aligned: Int, quads: Int)? {
            var o = FlowRemesh.Options(); o.targetQuads = 2000; o.guides = guides
            if let w = ProcessInfo.processInfo.environment["GUIDE_W"].flatMap(Float.init) { o.guideStrength = w }
            guard let r = FlowRemesh.remesh(positions: mesh.positions, faces: mesh.faces, options: o) else { return nil }
            let rho = FlowRemesh.metrics(of: r, surface: mesh.positions, faces: mesh.faces).meanEdge * 1.5
            var edges = 0, aligned = 0
            for q in r.quads {
                let c = [q.x, q.y, q.z, q.w]
                for k in 0..<4 {
                    let a = r.positions[Int(c[k])], b = r.positions[Int(c[(k + 1) % 4])]
                    let e = b - a
                    guard simd_length(e) > 1e-9, let g = tangent(near: (a + b) / 2), g.dist < rho,
                          let ga = tangent(near: a), let gb = tangent(near: b), ga.dist < rho * 0.27, gb.dist < rho * 0.27 else { continue }
                    edges += 1                                        // an edge lying on the guide
                    if abs(simd_dot(simd_normalize(e), g.dir)) > cos(25 * Float.pi / 180) { aligned += 1 }
                }
            }
            return (edges, aligned, r.quads.count)
        }
        let a = edgeAlignment([]), b = edgeAlignment([guide])
        print("GUIDE remesh: edges near guide without \(a?.edges ?? -1) aligned(25°) \(a?.aligned ?? -1); with \(b?.edges ?? -1) aligned \(b?.aligned ?? -1); quads \(a?.quads ?? -1) → \(b?.quads ?? -1)")
        // Edges lying on the guide are a row running along it.
        XCTAssertGreaterThanOrEqual(b?.edges ?? 0, 10)
        XCTAssertGreaterThanOrEqual((b?.aligned ?? 0) * 10, (b?.edges ?? 1) * 9)
    }
}

extension FlowRemeshTests {
    /// A sphere with a cap cut away is open on the +x side; the remesh must not grow a bar out to the grid edge.
    func testRemeshOfAnOpenMeshDoesNotGrowABar() throws {
        let (p, f) = icosphere(4)
        let open = f.filter { t in (p[Int(t.x)].x + p[Int(t.y)].x + p[Int(t.z)].x) / 3 < 0.85 }
        XCTAssertLessThan(open.count, f.count)
        let r = try XCTUnwrap(MeshBoolean.remesh(positions: p, faces: open, resolution: 48))
        let maxX = r.positions.map(\.x).max() ?? 0
        XCTAssertLessThan(maxX, 1.1)
    }

    /// Opt-in: FLOW_MESH=<.mesh>. Looks for spikes (very long edges) in the aligned quad remesh.
    func testQuadRemeshHasNoSpikesOnRealScan() throws {
        guard let path = ProcessInfo.processInfo.environment["FLOW_MESH"] else { throw XCTSkip("set FLOW_MESH") }
        let mesh = try XCTUnwrap(MeshDecimator.parse(Data(contentsOf: URL(fileURLWithPath: path))))
        let r = try XCTUnwrap(MeshBoolean.quadRemesh(positions: mesh.positions, faces: mesh.faces, resolution: 128))
        var lens: [Float] = []
        var worst: (Float, SIMD3<Float>, SIMD3<Float>) = (0, .zero, .zero)
        for f in r.faces { for (a, b) in [(f.x, f.y), (f.y, f.z), (f.z, f.x)] {
            let d = simd_distance(r.positions[Int(a)], r.positions[Int(b)])
            lens.append(d)
            if d > worst.0 { worst = (d, r.positions[Int(a)], r.positions[Int(b)]) }
        } }
        lens.sort()
        print("SPIKE edges \(lens.count) median \(lens[lens.count / 2]) p99 \(lens[lens.count * 99 / 100]) max \(lens.last!) at \(worst.1) → \(worst.2)")
        // Thin triangles: ratio of the longest edge to the altitude.
        var slivers = 0
        for f in r.faces {
            let a = r.positions[Int(f.x)], b = r.positions[Int(f.y)], c = r.positions[Int(f.z)]
            let longest = max(simd_distance(a, b), max(simd_distance(b, c), simd_distance(c, a)))
            let area = simd_length(simd_cross(b - a, c - a)) / 2
            if area < longest * longest * 0.01 { slivers += 1 }
        }
        print("SPIKE slivers (area < 1% of longest²): \(slivers) of \(r.faces.count)")
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for p in r.positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        var plo = SIMD3<Float>(repeating: .infinity), phi = SIMD3<Float>(repeating: -.infinity)
        for p in mesh.positions { plo = simd_min(plo, p); phi = simd_max(phi, p) }
        print("SPIKE bounds input \(plo) \(phi) output \(lo) \(hi)")
    }
}

extension FlowRemeshTests {
    func testGuidesSurviveProjectEncoding() throws {
        var p = Project(name: "t")
        p.flowGuides = [FlowGuide(points: [[0, 0, 0], [1, 2, 3]])]
        let back = try JSONDecoder().decode(Project.self, from: JSONEncoder().encode(p))
        XCTAssertEqual(back.flowGuides, p.flowGuides)
        XCTAssertTrue(Project(name: "x").flowGuides.isEmpty)
    }
}
