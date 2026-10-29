import XCTest

final class ModelRelocationTests: XCTestCase {
    private var tmp: URL!
    override func setUpWithError() throws {
        tmp = FileManager.default.temporaryDirectory.appendingPathComponent("mr-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: tmp) }

    private func makeLibrary(_ name: String = "src") throws -> URL {
        let root = tmp.appendingPathComponent(name, isDirectory: true)
        let fm = FileManager.default
        for (slot, file, n) in [("shape-small", "model.fp16.safetensors", 3000), ("shape-small", "config.yaml", 50),
                                ("paint-small", "unet/diffusion_pytorch_model.safetensors", 5000),
                                ("paint-small", "vae/x.safetensors.partial", 700)] {
            let url = root.appendingPathComponent(slot).appendingPathComponent(file)
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data((0..<n).map { UInt8($0 % 251) }).write(to: url)
        }
        try fm.createDirectory(at: root.appendingPathComponent(".compat/paint-small"), withIntermediateDirectories: true)
        try Data([1]).write(to: root.appendingPathComponent("notes.txt"))      // unrelated file
        return root
    }

    func testItemsListSlotsOnly() throws {
        let src = try makeLibrary()
        let items = ModelRelocation.items(in: src)
        XCTAssertEqual(items.map(\.relativePath), [
            "paint-small/unet/diffusion_pytorch_model.safetensors", "paint-small/vae/x.safetensors.partial",
            "shape-small/config.yaml", "shape-small/model.fp16.safetensors"])
        XCTAssertEqual(items.map(\.bytes).reduce(0, +), 8750)
    }

    func testCopyThenRemoveSourcesMovesLibraryAndKeepsUnrelatedFiles() throws {
        let src = try makeLibrary(), dst = tmp.appendingPathComponent("dst/Modelr Models", isDirectory: true)
        let items = ModelRelocation.items(in: src)
        var lastDone: Int64 = 0
        try ModelRelocation.copy(items, from: src, to: dst, progress: { done, _ in lastDone = done }, isCancelled: { false })
        XCTAssertEqual(lastDone, 8750)
        XCTAssertEqual(ModelRelocation.items(in: dst), items)
        let a = try Data(contentsOf: src.appendingPathComponent(items[0].relativePath))
        XCTAssertEqual(a, try Data(contentsOf: dst.appendingPathComponent(items[0].relativePath)))

        ModelRelocation.removeSources(items, from: src)
        XCTAssertTrue(ModelRelocation.items(in: src).isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: src.appendingPathComponent("shape-small").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: src.appendingPathComponent(".compat").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: src.appendingPathComponent("notes.txt").path))
    }

    func testCancelLeavesSourceIntactAndNoPartialAtDestination() throws {
        let src = try makeLibrary(), dst = tmp.appendingPathComponent("dst", isDirectory: true)
        let items = ModelRelocation.items(in: src)
        var calls = 0
        XCTAssertThrowsError(try ModelRelocation.copy(items, from: src, to: dst, progress: { _, _ in },
                                                      isCancelled: { calls += 1; return calls > 2 })) {
            XCTAssertEqual($0 as? ModelRelocation.Failure, .cancelled)
        }
        XCTAssertEqual(ModelRelocation.items(in: src), items)                 // source untouched
        let leftovers = (FileManager.default.enumerator(atPath: dst.path)?.allObjects as? [String]) ?? []
        XCTAssertFalse(leftovers.contains { $0.hasSuffix(".copying") })
    }

    func testResumeSkipsFilesAlreadyAtDestination() throws {
        let src = try makeLibrary(), dst = tmp.appendingPathComponent("dst", isDirectory: true)
        let items = ModelRelocation.items(in: src)
        var calls = 0
        _ = try? ModelRelocation.copy(items, from: src, to: dst, progress: { _, _ in },
                                      isCancelled: { calls += 1; return calls > 3 })
        var fresh: [String] = []
        try ModelRelocation.copy(items, from: src, to: dst, progress: { _, name in fresh.append(name) }, isCancelled: { false })
        XCTAssertEqual(ModelRelocation.items(in: dst), items)
        XCTAssertEqual(fresh.count, items.count)
    }

    func testResolveRootAndVolumeHelpers() throws {
        let plain = tmp.appendingPathComponent("Drive", isDirectory: true)
        try FileManager.default.createDirectory(at: plain, withIntermediateDirectories: true)
        XCTAssertEqual(ModelRelocation.resolveRoot(picked: plain).lastPathComponent, "Modelr Models")
        let lib = try makeLibrary()
        XCTAssertEqual(ModelRelocation.resolveRoot(picked: lib), lib)           // already a library
        XCTAssertTrue(ModelRelocation.sameVolume(tmp, tmp.appendingPathComponent("not/yet/created")))
        XCTAssertGreaterThan(ModelRelocation.freeBytes(at: tmp.appendingPathComponent("x/y")), 0)
    }
}
