import XCTest

final class ToolDefaultsTests: XCTestCase {
    private func suite() -> UserDefaults {
        let name = "td-\(UUID().uuidString)"
        let d = UserDefaults(suiteName: name)!
        addTeardownBlock { d.removePersistentDomain(forName: name) }
        return d
    }

    func testFallbacksWhenNothingIsSet() {
        let d = suite()
        XCTAssertEqual(ToolDefaults.decimateK(d), 50)
        XCTAssertEqual(ToolDefaults.booleanResolution(d), 160)
        XCTAssertEqual(ToolDefaults.symmetryAxis(d), .x)
        XCTAssertTrue(ToolDefaults.symmetryKeepPositive(d))
        XCTAssertEqual(ToolDefaults.lodLevels(d), [50, 25, 10])
    }

    func testStoredValuesWin() {
        let d = suite()
        d.set(120.0, forKey: ToolDefaults.Key.decimateK)
        d.set("z", forKey: ToolDefaults.Key.symmetryAxis)
        d.set(false, forKey: ToolDefaults.Key.symmetryKeepPositive)
        XCTAssertEqual(ToolDefaults.decimateK(d), 120)
        XCTAssertEqual(ToolDefaults.symmetryAxis(d), .z)
        XCTAssertFalse(ToolDefaults.symmetryKeepPositive(d))
    }

    func testLODParsingIsForgiving() {
        XCTAssertEqual(ToolDefaults.parseLODLevels("50, 25, 10"), [50, 25, 10])
        XCTAssertEqual(ToolDefaults.parseLODLevels("10% 75;25 / 25, abc, 0, 99, 200"), [75, 25, 10])
        XCTAssertEqual(ToolDefaults.parseLODLevels("nonsense"), [])
        let d = suite()
        d.set("nonsense", forKey: ToolDefaults.Key.lodLevels)
        XCTAssertEqual(ToolDefaults.lodLevels(d), [50, 25, 10])             // falls back, never empty
    }

    func testTextToImageDefaultsAndPromptComposition() {
        let d = suite()
        XCTAssertEqual(ToolDefaults.ttiSteps(d), 25)
        XCTAssertEqual(ToolDefaults.ttiGuidance(d), 7.5)
        XCTAssertTrue(ToolDefaults.ttiRemoveBackground(d))
        XCTAssertFalse(ToolDefaults.ttiAutoGenerate(d))
        d.set(40.0, forKey: ToolDefaults.Key.ttiSteps)
        d.set(false, forKey: ToolDefaults.Key.ttiRemoveBackground)
        XCTAssertEqual(ToolDefaults.ttiSteps(d), 40)
        XCTAssertFalse(ToolDefaults.ttiRemoveBackground(d))
        XCTAssertEqual(ToolDefaults.composePrompt("  a red mug \n", suffix: ", studio"), "a red mug, studio")
        XCTAssertEqual(ToolDefaults.composePrompt("   ", suffix: ", studio"), "")
    }

    func testTextToImageModelIsOptionalAndPinned() {
        XCTAssertFalse(ModelCatalog.everything.contains(.textToImage))
        let m = ModelCatalog.model(.textToImage)
        XCTAssertEqual(m.files.count, 5)
        XCTAssertTrue(m.files.allSatisfy { $0.sha256.count == 64 })
        XCTAssertEqual(m.files.map(\.path).sorted(), [
            "text_encoder/model.fp16.safetensors", "tokenizer/merges.txt", "tokenizer/vocab.json",
            "unet/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors"])
        XCTAssertGreaterThan(m.totalBytes, 2_500_000_000)
    }

    func testDepthModelAndReliefDefaults() {
        XCTAssertFalse(ModelCatalog.everything.contains(.depthEstimation))
        let m = ModelCatalog.model(.depthEstimation)
        XCTAssertEqual(m.files.map(\.path), ["model.safetensors"])
        XCTAssertEqual(m.files[0].sha256.count, 64)
        let d = suite()
        XCTAssertEqual(ToolDefaults.reliefDepth(d), 0.25)
        XCTAssertEqual(ToolDefaults.reliefResolution(d), 256)
        XCTAssertEqual(ToolDefaults.reliefSmoothing(d), 2)
        d.set(384.0, forKey: ToolDefaults.Key.reliefResolution)
        XCTAssertEqual(ToolDefaults.reliefResolution(d), 384)
    }
}
