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
        XCTAssertGreaterThan(r.quads.count, 100)
        for v in r.positions { XCTAssertEqual(simd_length(v), 1, accuracy: 0.03) }
    }
}
