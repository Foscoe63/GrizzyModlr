import XCTest
import simd

final class SculptMeshTests: XCTestCase {
    /// Flat 21×21 grid in the XY plane, normals +Z.
    private func grid() -> SculptMesh {
        var p: [SIMD3<Float>] = []
        for y in 0..<21 { for x in 0..<21 { p.append([Float(x) * 0.1, Float(y) * 0.1, 0]) } }
        var f: [SIMD3<UInt32>] = []
        for y in 0..<20 { for x in 0..<20 {
            let a = UInt32(y * 21 + x), b = a + 1, c = a + 21, d = c + 1
            f.append([a, b, d]); f.append([a, d, c])
        } }
        return SculptMesh(positions: p, faces: f)
    }
    private let mid = SIMD3<Float>(1, 1, 0)
    private var midIndex: Int { 10 * 21 + 10 }

    func testPullRaisesAlongNormalAndFalloffIsLocal() {
        var m = grid()
        XCTAssertEqual(m.normals[midIndex].z, 1, accuracy: 1e-4)
        m.dab(tool: .pull, at: mid, radius: 0.4, strength: 1)
        XCTAssertGreaterThan(m.positions[midIndex].z, 0.01)
        XCTAssertEqual(m.positions[0].z, 0)                  // far corner untouched
    }

    func testPushIsOppositeOfPull() {
        var a = grid(), b = grid()
        a.dab(tool: .pull, at: mid, radius: 0.4, strength: 0.5)
        b.dab(tool: .push, at: mid, radius: 0.4, strength: 0.5)
        XCTAssertEqual(a.positions[midIndex].z, -b.positions[midIndex].z, accuracy: 1e-5)
    }

    func testSmoothAndFlattenReduceABump() {
        var m = grid()
        m.dab(tool: .pull, at: mid, radius: 0.3, strength: 1)
        let bump = m.positions[midIndex].z
        var s = m
        for _ in 0..<5 { s.dab(tool: .smooth, at: mid, radius: 0.4, strength: 0.5) }
        XCTAssertLessThan(s.positions[midIndex].z, bump)
        var f = m
        for _ in 0..<5 { f.dab(tool: .flatten, at: mid, radius: 0.4, strength: 0.5) }
        XCTAssertLessThan(abs(f.positions[midIndex].z), bump)
    }

    func testUndoRestoresStroke() {
        var m = grid()
        let before = m.positions
        m.beginStroke()
        m.dab(tool: .pull, at: mid, radius: 0.4, strength: 1)
        m.dab(tool: .pull, at: mid, radius: 0.4, strength: 1)
        m.endStroke()
        XCTAssertTrue(m.canUndo)
        XCTAssertTrue(m.isEdited)
        m.undo()
        XCTAssertEqual(m.positions, before)
        XCTAssertFalse(m.canUndo)
        XCTAssertFalse(m.isEdited)
        XCTAssertEqual(m.normals[midIndex].z, 1, accuracy: 1e-4)
    }

    func testMissOutsideRadiusChangesNothing() {
        var m = grid()
        XCTAssertEqual(m.dab(tool: .pull, at: [10, 10, 0], radius: 0.2, strength: 1), [])
        XCTAssertFalse(m.isEdited)
    }
}
