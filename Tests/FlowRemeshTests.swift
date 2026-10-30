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
}
