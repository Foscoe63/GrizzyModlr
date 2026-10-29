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

/// UV edit through the store: UVs move, texture re-maps, and a new paint version results.
@MainActor
final class UVEditStoreTests: XCTestCase {
    private var root: URL!
    private var store: ProjectStore!
    private static let tinyPNG = Data(base64Encoded:
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("uv-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        store = ProjectStore(rootDir: root)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    func testApplyUVEditCreatesNewPaintVersionWithMovedUVs() async throws {
        // Project with a committed shape, then a hand-built paint version (mesh + albedo).
        let p = store.newProject()
        let img = root.appendingPathComponent("in.png")
        try Self.tinyPNG.write(to: img)
        store.setImage(fromURL: img, for: p.id)
        let shape = try store.stageShapeRun(for: p.id)
        try Data(repeating: 7, count: 64).write(to: shape.mesh)
        XCTAssertNil(store.commitShapeRun(shape))
        store.setPaintModel(.small, for: p.id)
        let paint = try store.stagePaintRun(for: p.id)

        let mesh = TexturedMesh(
            positions: [[0,0,0],[1,0,0],[1,1,0],[0,1,0]], normals: Array(repeating: [0,0,1], count: 4),
            uvs: [[0.1,0.1],[0.4,0.1],[0.4,0.4],[0.1,0.4]], faces: [[0,1,2],[0,2,3]])
        try mesh.serialize().write(to: paint.outMesh)
        let canvas = TextureCanvas(width: 64, height: 64, pixels: [UInt8](repeating: 200, count: 64 * 64 * 4))
        XCTAssertTrue(canvas.write(to: paint.outTexture))
        XCTAssertNil(store.commitPaintRun(paint))

        let before = try XCTUnwrap(store.project(p.id)).generations.count
        let err = await store.applyUVEdit([0: IslandTransform(du: 0.5, dv: 0.5)], for: p.id)
        XCTAssertNil(err)
        let proj = try XCTUnwrap(store.project(p.id))
        XCTAssertEqual(proj.generations.count, before + 1)
        let gen = try XCTUnwrap(proj.currentGeneration)
        XCTAssertEqual(gen.editSummary, "UV layout edited")
        guard case .texturedMesh(let meshURL, _)? = store.currentViewerContent(for: proj) else {
            return XCTFail("expected a textured version")
        }
        let newMesh = try XCTUnwrap(TexturedMesh.parse(Data(contentsOf: meshURL)))
        XCTAssertEqual(newMesh.uvs[0].x, 0.6, accuracy: 1e-5)

        let leave = await store.applyUVEdit([0: IslandTransform(du: 5)], for: p.id)
        XCTAssertEqual(leave, "An island would leave the texture.")
    }
}

/// Topology tools through the store: LOD sets and symmetry become new shape versions.
@MainActor
final class TopologyStoreTests: XCTestCase {
    private var root: URL!
    private var store: ProjectStore!
    private static let tinyPNG = Data(base64Encoded:
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("tp-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        store = ProjectStore(rootDir: root)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    /// Project whose shape is a ~2k-triangle sphere.
    private func projectWithSphere() throws -> Project.ID {
        let p = store.newProject()
        let img = root.appendingPathComponent("in.png")
        try Self.tinyPNG.write(to: img)
        store.setImage(fromURL: img, for: p.id)
        let staged = try store.stageShapeRun(for: p.id)
        let g = MeshBoolean.grid(lo: [-1, -1, -1], hi: [1, 1, 1], resolution: 24)
        let m = MeshBoolean.surfaceNets(MeshBoolean.primitiveField(.sphere, center: .zero, size: 1, grid: g), grid: g)
        try MeshDecimator.serialize(positions: m.positions, faces: m.faces).write(to: staged.mesh)
        XCTAssertNil(store.commitShapeRun(staged))
        return p.id
    }

    func testLODsSaveOneVersionPerLevelAndSelectTheLargest() async throws {
        let id = try projectWithSphere()
        let err = await store.applyLODs(percents: [50, 25], for: id)
        XCTAssertNil(err)
        let gens = try XCTUnwrap(store.project(id)?.generations)
        XCTAssertEqual(gens.count, 3)
        XCTAssertEqual(gens.compactMap(\.editSummary).map { String($0.prefix(7)) }, ["LOD 25%", "LOD 50%"])
        XCTAssertEqual(store.project(id)?.currentGeneration?.id, gens.last?.id)
        let counts = try gens.map { g -> Int in
            let url = root.appendingPathComponent("projects").appendingPathComponent(id.uuidString).appendingPathComponent(g.meshFileName)
            return try XCTUnwrap(MeshDecimator.parse(Data(contentsOf: url))).faces.count
        }
        // history order: original, LOD 25%, LOD 50%
        XCTAssertLessThan(counts[1], counts[2])
        XCTAssertLessThan(counts[2], counts[0])
        let none = await store.applyLODs(percents: [], for: id)
        XCTAssertEqual(none, "Add at least one level in Settings → Tools.")
    }

    func testSymmetrizeThroughTheStore() async throws {
        let id = try projectWithSphere()
        let err = await store.applyMeshEdit(.symmetrize(axis: .y, keepPositive: true), for: id)
        XCTAssertNil(err)
        XCTAssertEqual(store.project(id)?.currentGeneration?.editSummary?.hasPrefix("mirrored +Y half"), true)
    }
}

/// Composed scenes through the store: items persist per project and merge into a new version.
@MainActor
final class SceneStoreTests: XCTestCase {
    private var root: URL!
    private var store: ProjectStore!
    private static let tinyPNG = Data(base64Encoded:
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("sc-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        store = ProjectStore(rootDir: root)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

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

    func testAddUpdateRemoveAndPersist() throws {
        let id = try projectWithCube()
        let gen = try XCTUnwrap(store.project(id)?.generations.first)
        store.addSceneItem(generation: gen, from: id, to: id, spacing: 2.5)
        store.addSceneItem(generation: gen, from: id, to: id, spacing: 2.5)
        var items = try XCTUnwrap(store.project(id)?.sceneItems)
        XCTAssertEqual(items.map(\.x), [0, 2.5])
        items[1].scale = 2
        store.updateSceneItem(items[1], for: id)
        XCTAssertEqual(store.project(id)?.sceneItems[1].scale, 2)
        let reloaded = ProjectStore(rootDir: root)                              // survives a relaunch
        XCTAssertEqual(reloaded.project(id)?.sceneItems.count, 2)
        store.removeSceneItem(items[0].id, for: id)
        XCTAssertEqual(store.project(id)?.sceneItems.count, 1)
    }

    func testSaveSceneMergesObjectsIntoANewShapeVersion() async throws {
        let id = try projectWithCube()
        let gen = try XCTUnwrap(store.project(id)?.generations.first)
        store.addSceneItem(generation: gen, from: id, to: id, spacing: 3)
        store.addSceneItem(generation: gen, from: id, to: id, spacing: 3)
        let err = await store.saveSceneAsVersion(for: id)
        XCTAssertNil(err)
        let proj = try XCTUnwrap(store.project(id))
        XCTAssertEqual(proj.generations.count, 2)
        XCTAssertEqual(proj.currentGeneration?.editSummary, "scene of 2 objects")
        let url = try XCTUnwrap(store.meshURL(for: proj))
        let mesh = try XCTUnwrap(MeshDecimator.parse(Data(contentsOf: url)))
        XCTAssertEqual(mesh.faces.count, 24)
        let xs = mesh.positions.map(\.x)
        XCTAssertGreaterThan(xs.max()! - xs.min()!, 3)                         // two objects 3 apart
        let empty = await ProjectStore(rootDir: root).saveSceneAsVersion(for: UUID())
        XCTAssertEqual(empty, "Generate a shape first.")
    }

    func testCommitShapeRunRecordsASummary() throws {
        let p = store.newProject()
        let img = root.appendingPathComponent("in.png")
        try Self.tinyPNG.write(to: img)
        store.setImage(fromURL: img, for: p.id)
        let staged = try store.stageShapeRun(for: p.id)
        try Data(repeating: 7, count: 64).write(to: staged.mesh)
        XCTAssertNil(store.commitShapeRun(staged, summary: "depth relief"))
        XCTAssertEqual(store.project(p.id)?.currentGeneration?.editSummary, "depth relief")
    }
}
