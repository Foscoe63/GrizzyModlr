import XCTest
import AppKit

/// Multi-view reference images: stored per project, snapshotted into each run,
/// recorded on the saved version and cleaned up with it.
@MainActor
final class ReferenceViewsTests: XCTestCase {
    private var root: URL!
    private var store: ProjectStore!

    private static let tinyPNG = Data(base64Encoded:
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("rv-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        store = ProjectStore(rootDir: root)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    private func projectWithImage() throws -> Project.ID {
        let p = store.newProject()
        let img = root.appendingPathComponent("in.png")
        try Self.tinyPNG.write(to: img)
        store.setImage(fromURL: img, for: p.id)
        return p.id
    }

    func testMultiViewModelIsOptionalAndNotInOnboardingBundles() {
        XCTAssertEqual(ModelID.shapeMultiView.kind, .shape)
        XCTAssertFalse(ModelCatalog.everything.contains(.shapeMultiView))
        let files = ModelCatalog.model(.shapeMultiView).files
        XCTAssertEqual(files.map(\.path), ["config.yaml", "model.fp16.safetensors"])
        XCTAssertTrue(files.allSatisfy { $0.remotePath != nil && $0.sha256.count == 64 })
        XCTAssertTrue(ModelCatalog.model(.shapeMultiView)
            .remoteURL(for: files[1]).absoluteString.contains("hunyuan3d-dit-v2-mv/model.fp16.safetensors"))
    }

    func testSetAndRemoveExtraView() throws {
        let id = try projectWithImage()
        let src = root.appendingPathComponent("left.png")
        try Self.tinyPNG.write(to: src)
        store.setExtraView(.left, fromURL: src, for: id)
        let project = store.project(id)!
        XCTAssertEqual(project.extraViews.keys.sorted { $0.rawValue < $1.rawValue }, [.left])
        XCTAssertNotNil(store.extraViewURL(.left, for: project))
        store.removeExtraView(.left, for: id)
        XCTAssertTrue(store.project(id)!.extraViews.isEmpty)
        XCTAssertNil(store.project(id)!.extraViewsRaw)
    }

    func testStageCommitDeleteLifecycle() throws {
        let id = try projectWithImage()
        let src = root.appendingPathComponent("back.png")
        try Self.tinyPNG.write(to: src)
        store.setExtraView(.back, fromURL: src, for: id)

        let staged = try store.stageShapeRun(for: id)
        let snap = try XCTUnwrap(staged.extraViews[.back])
        XCTAssertTrue(FileManager.default.fileExists(atPath: snap.path))
        XCTAssertEqual(staged.settings.model, .small)               // 2mv is not a turbo schedule

        try Data(repeating: 7, count: 64).write(to: staged.mesh)
        XCTAssertNil(store.commitShapeRun(staged))
        let gen = try XCTUnwrap(store.project(id)?.currentGeneration)
        XCTAssertEqual(gen.extraViewFiles?["back"], snap.lastPathComponent)

        store.deleteGeneration(gen.id, for: id)
        XCTAssertFalse(FileManager.default.fileExists(atPath: snap.path))
    }

    func testDiscardRemovesSnapshots() throws {
        let id = try projectWithImage()
        let src = root.appendingPathComponent("right.png")
        try Self.tinyPNG.write(to: src)
        store.setExtraView(.right, fromURL: src, for: id)
        let staged = try store.stageShapeRun(for: id)
        let snap = try XCTUnwrap(staged.extraViews[.right])
        store.discardShapeRun(staged)
        XCTAssertFalse(FileManager.default.fileExists(atPath: snap.path))
    }
}
