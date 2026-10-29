import XCTest

final class TorchCheckpointTests: XCTestCase {
    private func fixture() throws -> URL {
        try XCTUnwrap(Bundle(for: Self.self).url(forResource: "tiny_state", withExtension: "ckpt"))
    }

    private func floats(_ t: TorchCheckpoint.TensorRef) -> [Float] {
        t.bytes().withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    func testLoadsEveryTensorWithShapesAndDtypes() throws {
        let sd = try TorchCheckpoint.load(fixture())
        XCTAssertEqual(Set(sd.keys), ["w.float", "w.half", "w.int64", "w.base", "w.transposed", "w.slice", "w.scalarish"])
        XCTAssertEqual(sd["w.float"]?.shape, [2, 3])
        XCTAssertEqual(sd["w.float"]?.dtype, "float32")
        XCTAssertEqual(sd["w.half"]?.dtype, "float16")
        XCTAssertEqual(sd["w.int64"]?.dtype, "int64")
        XCTAssertEqual(sd["w.scalarish"]?.shape, [1])
    }

    func testValuesRoundTrip() throws {
        let sd = try TorchCheckpoint.load(fixture())
        XCTAssertEqual(floats(sd["w.float"]!), [1.5, -2.0, 3.25, 0.0, 4.5, -0.5])
        XCTAssertEqual(floats(sd["w.scalarish"]!), [42])
        let halves = sd["w.half"]!.bytes().withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
        XCTAssertEqual(halves, [1.0, 2.0, 0.5, -3.0])
        let ints = sd["w.int64"]!.bytes().withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) }
        XCTAssertEqual(ints, [7, -8, 9_000_000_000])
    }

    func testStridedViewsAreMaterializedInLogicalOrder() throws {
        let sd = try TorchCheckpoint.load(fixture())
        let base = Array(0..<12).map(Float.init)
        XCTAssertEqual(floats(sd["w.base"]!), base)
        XCTAssertTrue(sd["w.base"]!.isContiguous)
        // base is [3,4]; its transpose is [4,3] with strides (1,4)
        XCTAssertEqual(sd["w.transposed"]?.shape, [4, 3])
        XCTAssertFalse(sd["w.transposed"]!.isContiguous)
        XCTAssertEqual(floats(sd["w.transposed"]!), [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])
        // base[1:, 1:3] = [[5, 6], [9, 10]] (offset 5, strides (4, 1))
        XCTAssertEqual(sd["w.slice"]?.shape, [2, 2])
        XCTAssertEqual(floats(sd["w.slice"]!), [5, 6, 9, 10])
    }

    func testRejectsNonCheckpointsAndNeverRunsCode() throws {
        let junk = FileManager.default.temporaryDirectory.appendingPathComponent("junk-\(UUID()).ckpt")
        try Data("not a zip".utf8).write(to: junk)
        defer { try? FileManager.default.removeItem(at: junk) }
        XCTAssertThrowsError(try TorchCheckpoint.load(junk))
        XCTAssertThrowsError(try TorchCheckpoint.load(junk.appendingPathExtension("missing")))
    }
}
