import XCTest

final class ShapeV21ModelTests: XCTestCase {
    func testV21MapsToItsOwnModelAndSavedRawValue() {
        XCTAssertEqual(ShapeModel.v21.modelID, .shapeV21)
        XCTAssertEqual(ShapeModel(legacyRaw: "v21"), .v21)
        XCTAssertEqual(ShapeModel.v21.defaultSteps, 50)
        XCTAssertEqual(ModelID.shapeV21.kind, .shape)
        let steps = QualityPreset.ordered.map { $0.steps(for: .v21) }
        XCTAssertEqual(steps, steps.sorted())
    }

    func testCatalogPinsTheUpstreamPickleAndConfig() {
        let m = ModelCatalog.model(.shapeV21)
        XCTAssertEqual(m.repo, "tencent/Hunyuan3D-2.1")
        XCTAssertEqual(m.files.map(\.path).sorted(), ["config.yaml", "model.fp16.ckpt"])
        for f in m.files {
            XCTAssertEqual(f.sha256.count, 64)
            XCTAssertTrue(m.remoteURL(for: f).absoluteString.contains("/hunyuan3d-dit-v2-1/"))
        }
        XCTAssertEqual(m.files.first { $0.path == "model.fp16.ckpt" }!.bytes, 7_366_389_768)
        XCTAssertEqual(m.revision?.count, 40)
    }

    func testLicenseNamesTheRegionalExclusions() {
        let license = ModelID.shapeV21.info.license
        for region in ["EU", "UK", "South Korea"] { XCTAssertTrue(license.contains(region), region) }
        XCTAssertTrue(ModelCatalog.search("south korea").contains(.shapeV21))
    }
}
