import XCTest
import simd

final class QuadRemeshTests: XCTestCase {
    /// An axis-aligned cuboid (half extents h) rotated by `q`, as an outward-wound triangle mesh.
    private func cuboid(_ h: SIMD3<Float>, rotation q: simd_quatf) -> ([SIMD3<Float>], [SIMD3<UInt32>]) {
        var p: [SIMD3<Float>] = []
        for z in [-h.z, h.z] { for y in [-h.y, h.y] { for x in [-h.x, h.x] { p.append(q.act([x, y, z])) } } }
        let quads: [[UInt32]] = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
        var f: [SIMD3<UInt32>] = []
        for c in quads { f.append([c[0], c[1], c[2]]); f.append([c[0], c[2], c[3]]) }
        return (p, f)
    }

    func testBestFrameRecoversRotatedBoxAxes() {
        let q = simd_quatf(angle: 0.52, axis: simd_normalize([0.3, 1, 0.2]))
        let (p, f) = cuboid([1, 0.6, 0.4], rotation: q)
        let frame = MeshFrame.bestFrame(positions: p, faces: f)
        // Each recovered axis lines up with one of the box's true axes (sign / order free).
        let truth = [q.act([1, 0, 0]), q.act([0, 1, 0]), q.act([0, 0, 1])]
        for axis in [frame.columns.0, frame.columns.1, frame.columns.2] {
            XCTAssertGreaterThan(truth.map { abs(simd_dot(axis, $0)) }.max()!, 0.995)
        }
    }

    func testBestFrameOfAxisAlignedBoxIsStable() {
        let (p, f) = cuboid([1, 1, 1], rotation: simd_quatf(angle: 0, axis: [0, 0, 1]))
        let frame = MeshFrame.bestFrame(positions: p, faces: f)
        for axis in [frame.columns.0, frame.columns.1, frame.columns.2] {
            XCTAssertGreaterThan(max(abs(axis.x), abs(axis.y), abs(axis.z)), 0.999)
        }
    }

    /// Share of vertices that sit on one of the box's six faces (within `tol`).
    private func onFlats(_ v: [SIMD3<Float>], h: SIMD3<Float>, q: simd_quatf, tol: Float) -> Float {
        let inv = q.inverse
        let on = v.filter { w in
            let p = inv.act(w)
            return abs(abs(p.x) - h.x) < tol || abs(abs(p.y) - h.y) < tol || abs(abs(p.z) - h.z) < tol
        }
        return Float(on.count) / Float(v.count)
    }

    func testQuadRemeshKeepsRotatedBoxFlatAndPairsIntoQuads() throws {
        let h = SIMD3<Float>(1, 0.6, 0.4)
        let q = simd_quatf(angle: 0.52, axis: simd_normalize([0.3, 1, 0.2]))
        let (p, f) = cuboid(h, rotation: q)
        let plain = try XCTUnwrap(MeshBoolean.remesh(positions: p, faces: f, resolution: 40))
        let aligned = try XCTUnwrap(MeshBoolean.quadRemesh(positions: p, faces: f, resolution: 40))

        let tol: Float = 0.02
        XCTAssertGreaterThan(onFlats(aligned.positions, h: h, q: q, tol: tol), 0.97)
        XCTAssertGreaterThan(onFlats(aligned.positions, h: h, q: q, tol: tol),
                             onFlats(plain.positions, h: h, q: q, tol: tol))

        let pairs = QuadPairing.pair(positions: aligned.positions, faces: aligned.faces)
        XCTAssertGreaterThan(Float(pairs.quads.count * 2) / Float(aligned.faces.count), 0.9)
    }

    func testQuadRemeshEditReportsQuadShare() {
        let (p, f) = cuboid([1, 1, 1], rotation: simd_quatf(angle: 0.3, axis: [0, 0, 1]))
        let r = MeshOps.apply(.quadRemesh(resolution: 32), positions: p, faces: f)
        XCTAssertTrue(r.changed)
        XCTAssertTrue(r.note.contains("quads"))
        XCTAssertEqual(MeshEdit.quadRemesh(resolution: 32).label, "quad remeshed at 32")
    }
}
