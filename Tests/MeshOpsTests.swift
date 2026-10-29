import XCTest
import simd

final class MeshOpsTests: XCTestCase {

    /// Unit cube, 12 outward-wound triangles.
    private func cube(skipFaces: Set<Int> = []) -> (MeshOps.Positions, MeshOps.Faces) {
        let p: MeshOps.Positions = [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ]
        let f: [SIMD3<UInt32>] = [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7],
        ]
        return (p, f.enumerated().filter { !skipFaces.contains($0.offset) }.map(\.element))
    }

    private func boundaryEdgeCount(_ faces: MeshOps.Faces) -> Int {
        var dir = Set<UInt64>()
        for f in faces {
            let v = [f.x, f.y, f.z]
            for k in 0..<3 { dir.insert(UInt64(v[k]) << 32 | UInt64(v[(k + 1) % 3])) }
        }
        return dir.filter { e in
            let a = e >> 32, b = e & 0xFFFF_FFFF
            return !dir.contains(b << 32 | a)
        }.count
    }

    func testFillHolesClosesMissingFacePair() {
        let (p, f) = cube(skipFaces: [2, 3])          // remove the top quad → one 4-edge hole
        XCTAssertEqual(boundaryEdgeCount(f), 4)
        let r = MeshOps.fillHoles(positions: p, faces: f, maxEdges: 16)
        XCTAssertEqual(r.filled, 1)
        XCTAssertEqual(boundaryEdgeCount(r.faces), 0)
        XCTAssertEqual(r.positions.count, p.count + 1)
    }

    func testFillHolesLeavesClosedMeshAndLargeLoops() {
        let (p, f) = cube()
        XCTAssertEqual(MeshOps.fillHoles(positions: p, faces: f, maxEdges: 16).filled, 0)
        let (p2, f2) = cube(skipFaces: [2, 3])
        let r = MeshOps.fillHoles(positions: p2, faces: f2, maxEdges: 3)   // loop has 4 edges
        XCTAssertEqual(r.filled, 0)
        XCTAssertEqual(r.faces.count, f2.count)
    }

    func testSmoothReducesNoiseAndPinsBoundary() {
        // Flat open grid with one bumped interior vertex.
        var p: MeshOps.Positions = []
        for y in 0..<4 { for x in 0..<4 { p.append([Float(x), Float(y), 0]) } }
        var f: MeshOps.Faces = []
        for y in 0..<3 { for x in 0..<3 {
            let a = UInt32(y * 4 + x), b = a + 1, c = a + 4, d = c + 1
            f.append([a, b, d]); f.append([a, d, c])
        } }
        p[5].z = 1
        let s = MeshOps.smooth(positions: p, faces: f, iterations: 5)
        XCTAssertLessThan(abs(s[5].z), 1)
        for i in [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15] { XCTAssertEqual(s[i], p[i]) }
    }

    func testSmoothKeepsSizeRoughly() {
        let (p, f) = cube()
        let s = MeshOps.smooth(positions: p, faces: f, iterations: 3)
        let ext = s.map { $0.x }.max()! - s.map { $0.x }.min()!
        XCTAssertGreaterThan(ext, 0.5)
        XCTAssertTrue(s.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite })
    }

    func testApplyReportsUnchangedAndFileRoundTrip() throws {
        let (p, f) = cube(skipFaces: [2, 3])
        XCTAssertFalse(MeshOps.apply(.decimate(faceBudget: 100), positions: p, faces: f).changed)
        XCTAssertFalse(MeshOps.apply(.fillHoles(maxEdges: 3), positions: p, faces: f).changed)

        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mo-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let src = dir.appendingPathComponent("a.mesh"), dst = dir.appendingPathComponent("b.mesh")
        try MeshDecimator.serialize(positions: p, faces: f).write(to: src)
        let r = MeshOps.applyToFile(.fillHoles(maxEdges: 16), from: src, to: dst)
        XCTAssertEqual(r?.changed, true)
        let out = MeshDecimator.parse(try Data(contentsOf: dst))
        XCTAssertEqual(out?.faces.count, f.count + 4)
    }
}
