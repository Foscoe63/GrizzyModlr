import XCTest
@testable import Hy3DMLX

/// Parity with scripts/trellis2/make_mesher_fixture.py, fed the decoder fixture's final features.
final class FlexibleDualGridMeshTests: XCTestCase {
    private struct Decoder: Decodable { let final_coords: [Int32]; let final: [Float]; let n_final: Int }
    private struct Mesh: Decodable { let grid, n_verts, n_faces: Int; let verts: [Float]; let faces: [Int32] }

    func testMeshMatchesTorch() throws {
        let root = ProcessInfo.processInfo.environment["TRELLIS2_DIR"] ?? "/Volumes/Models/Modelr/trellis2"
        let a = URL(fileURLWithPath: root + "/fixtures/decoder_fixture.json")
        let b = URL(fileURLWithPath: root + "/fixtures/mesher_fixture.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: a.path) && FileManager.default.fileExists(atPath: b.path),
                          "TRELLIS.2 mesher fixtures not installed")
        let d = try JSONDecoder().decode(Decoder.self, from: Data(contentsOf: a))
        let ref = try JSONDecoder().decode(Mesh.self, from: Data(contentsOf: b))
        let m = FlexibleDualGridMesh.extract(coords: d.final_coords, feats: d.final, grid: ref.grid)
        XCTAssertEqual(m.faces.count, ref.n_faces * 3)
        XCTAssertEqual(m.faces, ref.faces)
        let maxd = zip(m.vertices, ref.verts).map { abs($0 - $1) }.max() ?? 0
        print("mesher faces \(m.faces.count / 3) vertex maxAbsDiff \(maxd)")
        XCTAssertLessThan(maxd, 1e-5)
    }
}
