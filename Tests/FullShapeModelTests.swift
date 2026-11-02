import XCTest

final class FullShapeModelTests: XCTestCase {
    func testFullMapsToItsOwnModelAndKeepsSavedRawValues() {
        XCTAssertEqual(ShapeModel.full.modelID, .shapeFull)
        XCTAssertEqual(ShapeModel(legacyRaw: "full"), .full)
        XCTAssertEqual(ShapeModel(legacyRaw: "large"), .large)           // existing projects unchanged
        XCTAssertEqual(ShapeModel.full.defaultSteps, 50)
        XCTAssertEqual(ModelID.shapeFull.kind, .shape)
        XCTAssertEqual(ShapeModel.allCases.map(\.label), ["Small", "Large", "Full", "2.1", "Fast", "TRELLIS.2", "Pixal3D"])
    }

    func testFullStepsLadderIsMonotonicAndBeyondTurbo() {
        let steps = QualityPreset.ordered.map { $0.steps(for: .full) }
        XCTAssertEqual(steps, steps.sorted())
        XCTAssertGreaterThan(steps.first!, QualityPreset.max.steps(for: .large))
    }

    func testCatalogEntryPointsAtTencentSubfolderWithPinnedChecksums() {
        let m = ModelCatalog.model(.shapeFull)
        XCTAssertEqual(m.repo, "tencent/Hunyuan3D-2")
        XCTAssertEqual(m.files.map(\.path).sorted(), ["config.yaml", "model.fp16.safetensors"])   // local names the engine loads
        for f in m.files {
            XCTAssertEqual(f.sha256.count, 64)
            XCTAssertTrue(m.remoteURL(for: f).absoluteString.contains("/hunyuan3d-dit-v2-0/"))
        }
        let weights = m.files.first { $0.path == "model.fp16.safetensors" }!
        XCTAssertEqual(weights.bytes, 4_928_151_562)
        XCTAssertNotNil(m.revision)
    }
}
