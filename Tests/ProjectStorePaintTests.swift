import XCTest
import AppKit
// ProjectStore + Models are compiled into this bundle (see project.yml) — same module.

/// Storage round-trip for PBR (Large) paint versions: the extra metallic-roughness
/// map is staged, committed, restored, and cleaned up alongside the albedo — and a
/// PBR commit that never produced an MR map is rejected rather than degraded.
@MainActor
final class ProjectStorePaintTests: XCTestCase {

    private var root: URL!
    private var store: ProjectStore!

    /// Tiny valid 1×1 PNG for the input image (applyImage re-encodes it to input.png).
    private static let tinyPNG = Data(base64Encoded:
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("psp-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        store = ProjectStore(rootDir: root)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: - helpers

    /// A committed shape generation the paint run can texture, plus the project id.
    private func makeShape() throws -> Project.ID {
        let p = store.newProject()
        let img = root.appendingPathComponent("in.png")
        try Self.tinyPNG.write(to: img)
        store.setImage(fromURL: img, for: p.id)
        XCTAssertNotNil(store.imageURL(for: store.project(p.id)!), "input image should process")

        let staged = try store.stageShapeRun(for: p.id)
        try Data(repeating: 7, count: 64).write(to: staged.mesh)   // stand-in mesh (>8 bytes)
        XCTAssertNil(store.commitShapeRun(staged))
        return p.id
    }

    /// Stage a paint run for `id` with the given model + seed.
    private func stagePaint(_ id: Project.ID, model: PaintModel, seed: UInt64?)
    throws -> ProjectStore.StagedPaintRun {
        store.setPaintModel(model, for: id)
        store.setPaintAdvanced(true, for: id)          // advanced surfaces the seed field
        store.setPaintSeed(seed, for: id)
        return try store.stagePaintRun(for: id)
    }

    private func writeMaps(_ staged: ProjectStore.StagedPaintRun, mr: Bool) throws {
        try Data(repeating: 1, count: 64).write(to: staged.outMesh)
        try Data(repeating: 2, count: 64).write(to: staged.outTexture)
        if mr { try Data(repeating: 3, count: 64).write(to: staged.outMR) }
    }

    // MARK: - commit

    func testBakedNormalMapIsRecordedAndSelectsDetailContent() throws {
        let id = try makeShape()
        let staged = try stagePaint(id, model: .small, seed: nil)
        try writeMaps(staged, mr: false)
        try Data(repeating: 4, count: 64).write(to: staged.outNormal)
        XCTAssertNil(store.commitPaintRun(staged))

        let gen = try XCTUnwrap(store.project(id)?.generations.last)
        XCTAssertEqual(gen.paintedNormalFileName, staged.outNormal.lastPathComponent)
        if case .detailMesh(_, _, let mr, let normal)? = store.currentViewerContent(for: store.project(id)!) {
            XCTAssertNil(mr)
            XCTAssertEqual(normal.lastPathComponent, staged.outNormal.lastPathComponent)
        } else {
            XCTFail("expected .detailMesh viewer content")
        }
    }

    func testNoNormalMapWhenBakeWroteNothing() throws {
        let id = try makeShape()
        let staged = try stagePaint(id, model: .small, seed: nil)
        try writeMaps(staged, mr: false)
        XCTAssertNil(store.commitPaintRun(staged))
        XCTAssertNil(store.project(id)?.generations.last?.paintedNormalFileName)
    }

    func testPBRCommitRecordsMRAndSeed() throws {
        let id = try makeShape()
        let staged = try stagePaint(id, model: .large, seed: 4242)
        XCTAssertTrue(staged.isPBR)
        XCTAssertEqual(staged.seed, 4242)
        XCTAssertEqual(staged.outMR.lastPathComponent, "painted_\(staged.paintID.uuidString)_mr.png")

        try writeMaps(staged, mr: true)
        XCTAssertNil(store.commitPaintRun(staged))

        let gen = try XCTUnwrap(store.project(id)?.generations.last)
        XCTAssertTrue(gen.isPBR)
        XCTAssertEqual(gen.paintedMRFileName, staged.outMR.lastPathComponent)
        XCTAssertEqual(gen.paintedTextureFileName, staged.outTexture.lastPathComponent)
        XCTAssertEqual(gen.paintSeedRaw, 4242)
        XCTAssertEqual(gen.paintModel, .large)
        XCTAssertTrue(FileManager.default.fileExists(atPath: staged.outMR.path))

        // The viewer picks the PBR content variant when the MR map is present.
        if case .pbrMesh(_, _, let mr)? = store.currentViewerContent(for: store.project(id)!) {
            XCTAssertEqual(mr.lastPathComponent, staged.outMR.lastPathComponent)
        } else {
            XCTFail("expected .pbrMesh viewer content")
        }
    }

    func testColorCommitHasNoMRAndStagesButLeavesItUnwritten() throws {
        let id = try makeShape()
        let staged = try stagePaint(id, model: .small, seed: nil)
        XCTAssertFalse(staged.isPBR)
        try writeMaps(staged, mr: false)               // Color path never writes the MR map
        XCTAssertNil(store.commitPaintRun(staged))

        let gen = try XCTUnwrap(store.project(id)?.generations.last)
        XCTAssertFalse(gen.isPBR)
        XCTAssertNil(gen.paintedMRFileName)
        XCTAssertFalse(FileManager.default.fileExists(atPath: staged.outMR.path))
        if case .texturedMesh? = store.currentViewerContent(for: store.project(id)!) {} else {
            XCTFail("expected .texturedMesh viewer content for a Color version")
        }
    }

    /// A PBR run that finished without an MR map is rejected (not silently downgraded).
    func testPBRCommitWithoutMRFails() throws {
        let id = try makeShape()
        let staged = try stagePaint(id, model: .large, seed: 1)
        try writeMaps(staged, mr: false)               // outMR absent
        XCTAssertNotNil(store.commitPaintRun(staged))
        XCTAssertFalse(store.project(id)!.generations.contains { $0.isPainted })
        XCTAssertFalse(FileManager.default.fileExists(atPath: staged.outMesh.path))   // discarded
    }

    // MARK: - discard

    func testDiscardRemovesMR() throws {
        let id = try makeShape()
        let staged = try stagePaint(id, model: .large, seed: 9)
        try writeMaps(staged, mr: true)
        store.discardPaintRun(staged)
        for u in [staged.outMesh, staged.outTexture, staged.outMR] {
            XCTAssertFalse(FileManager.default.fileExists(atPath: u.path), "\(u.lastPathComponent) survived discard")
        }
    }

    // MARK: - delete cascade + orphan sweep

    func testDeletingShapeCascadesAndRemovesMR() throws {
        let id = try makeShape()
        let staged = try stagePaint(id, model: .large, seed: 5)
        try writeMaps(staged, mr: true)
        XCTAssertNil(store.commitPaintRun(staged))
        let shapeID = store.project(id)!.generations.first { $0.kind == .shape }!.id

        store.deleteGeneration(shapeID, for: id)       // cascades to the paint version
        XCTAssertTrue(store.project(id)!.generations.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: staged.outMR.path),
                       "the MR map should be swept with the cascaded paint version")
    }

    /// A reopened store sweeps a stray painted_*_mr.png no generation references.
    func testOrphanSweepRemovesStrayMR() throws {
        let id = try makeShape()
        let dir = root.appendingPathComponent("projects/\(id.uuidString)", isDirectory: true)
        let orphan = dir.appendingPathComponent("painted_\(UUID().uuidString)_mr.png")
        try Data(repeating: 4, count: 32).write(to: orphan)

        _ = ProjectStore(rootDir: root)               // load() runs sweepOrphanGenFiles()
        XCTAssertFalse(FileManager.default.fileExists(atPath: orphan.path))
    }
}
