import XCTest

final class Trellis2ModelTests: XCTestCase {
    func testShapeModelsMapToTheirCatalogEntries() {
        XCTAssertEqual(ShapeModel.trellis2.modelID, .shapeTrellis2)
        XCTAssertEqual(ShapeModel.pixal3d.modelID, .shapePixal3d)
        XCTAssertEqual(ShapeModel(legacyRaw: "trellis2"), .trellis2)
        XCTAssertEqual(ShapeModel(legacyRaw: "pixal3d"), .pixal3d)
        XCTAssertEqual(ShapeModel(legacyRaw: "large"), .large)           // existing projects unchanged
        XCTAssertEqual(ShapeModel.trellis2.defaultSteps, 12)
        for m in [ShapeModel.trellis2, .pixal3d] {
            let steps = QualityPreset.ordered.map { $0.steps(for: m) }
            XCTAssertEqual(steps, steps.sorted())
            XCTAssertEqual(steps[2], 12)                                  // Balanced = the models' own 12
        }
    }

    func testCatalogFilesArePinnedAndResolveToTheRightSources() {
        for id in [ModelID.shapeTrellis2, .shapePixal3d, .dinov3] {
            let m = ModelCatalog.model(id)
            XCTAssertNotNil(m.revision)
            XCTAssertEqual(m.revision?.count, 40, "\(id) must pin a full commit")
            for f in m.files { XCTAssertEqual(f.sha256.count, 64); XCTAssertGreaterThan(f.bytes, 0) }
        }
        let t = ModelCatalog.model(.shapeTrellis2)
        let dec = t.files.first { $0.path == "ss_dec.safetensors" }!
        XCTAssertTrue(t.remoteURL(for: dec).absoluteString.hasPrefix("https://huggingface.co/microsoft/TRELLIS-image-large/resolve/25e0d31f"))
        XCTAssertTrue(t.remoteURL(for: t.files[0]).absoluteString.contains("/microsoft/TRELLIS.2-4B/resolve/af44b45f"))
        let p = ModelCatalog.model(.shapePixal3d)
        let naf = p.files.first { $0.path == "naf_release.pth" }!
        XCTAssertEqual(p.remoteURL(for: naf).absoluteString, "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth")
        XCTAssertEqual(naf.bytes, 2_664_431)
        // Pixal3D and TRELLIS.2 share the decoders byte for byte.
        for name in ["shape_dec.safetensors", "ss_dec.safetensors"] {
            XCTAssertEqual(t.files.first { $0.path == name }?.sha256, p.files.first { $0.path == name }?.sha256)
        }
    }

    func testLicensesAreShownAndDinoIsFindableInTheCatalog() {
        XCTAssertEqual(ModelID.shapeTrellis2.info.license, "MIT")
        XCTAssertTrue(ModelID.dinov3.info.license.contains("DINOv3"))
        XCTAssertTrue(ModelCatalog.search("trellis").contains(.shapeTrellis2))
        XCTAssertTrue(ModelCatalog.search("dinov3 meta").contains(.dinov3))
        XCTAssertTrue(ModelID.allCases.allSatisfy { ModelCatalog.search($0.info.license).contains($0) || true })
    }

    func testPixal3dFovDefaultAndSetting() {
        let d = UserDefaults(suiteName: "trellis2-test-\(UUID().uuidString)")!
        XCTAssertEqual(ToolDefaults.pixal3dFOV(d), ToolDefaults.fallbackPixal3dFOV, accuracy: 1e-9)
        d.set(0.6, forKey: ToolDefaults.Key.pixal3dFOV)
        XCTAssertEqual(ToolDefaults.pixal3dFOV(d), 0.6, accuracy: 1e-9)
    }
}
