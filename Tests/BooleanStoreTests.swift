import XCTest
import simd

/// Booleans through the store: the result is a new shape version derived from the base.
@MainActor
final class BooleanStoreTests: XCTestCase {
    private var root: URL!
    private var store: ProjectStore!

    private static let tinyPNG = Data(base64Encoded:
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("bs-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        store = ProjectStore(rootDir: root)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    /// Project whose only shape is a unit cube (closed, outward winding).
    private func projectWithCube() throws -> Project.ID {
        let p = store.newProject()
        let img = root.appendingPathComponent("in.png")
        try Self.tinyPNG.write(to: img)
        store.setImage(fromURL: img, for: p.id)
        let staged = try store.stageShapeRun(for: p.id)
        let v: [SIMD3<Float>] = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]
        let f: [SIMD3<UInt32>] = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[2,3,7],[2,7,6],[1,2,6],[1,6,5],[3,0,4],[3,4,7]]
        try MeshDecimator.serialize(positions: v, faces: f).write(to: staged.mesh)
        XCTAssertNil(store.commitShapeRun(staged))
        return p.id
    }

    func testSubtractSphereMakesNewVersion() async throws {
        let id = try projectWithCube()
        let baseID = try XCTUnwrap(store.project(id)?.currentGeneration?.id)
        let err = await store.applyBoolean(.subtract, with: .primitive(.sphere, offset: [0, 0, 0], size: 0.5),
                                           resolution: 48, for: id)
        XCTAssertNil(err)
        let gens = try XCTUnwrap(store.project(id)?.generations)
        XCTAssertEqual(gens.count, 2)
        let new = try XCTUnwrap(store.project(id)?.currentGeneration)
        XCTAssertNotEqual(new.id, baseID)
        XCTAssertEqual(new.editedFromID, baseID)
        XCTAssertEqual(new.editSummary, "subtract sphere")
        let mesh = try XCTUnwrap(store.meshURL(for: store.project(id)!))
        let parsed = try XCTUnwrap(MeshDecimator.parse(Data(contentsOf: mesh)))
        XCTAssertGreaterThan(parsed.faces.count, 100)
    }

    func testDisjointIntersectReportsAndAddsNothing() async throws {
        let id = try projectWithCube()
        let err = await store.applyBoolean(.intersect, with: .primitive(.sphere, offset: [9, 0, 0], size: 0.1),
                                           resolution: 48, for: id)
        XCTAssertEqual(err, "The shapes don't overlap.")
        XCTAssertEqual(store.project(id)?.generations.count, 1)
    }
}
