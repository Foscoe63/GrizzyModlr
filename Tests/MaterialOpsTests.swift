import XCTest

final class MaterialOpsTests: XCTestCase {
    func testIdentityLeavesPixels() {
        var px: [UInt8] = [10, 120, 250, 255, 0, 0, 0, 255]
        let before = px
        MaterialOps.adjustAlbedo(&px, MaterialAdjustments())
        MaterialOps.adjustMR(&px, MaterialAdjustments())
        XCTAssertEqual(px, before)
        XCTAssertTrue(MaterialAdjustments().isIdentity)
    }

    func testBrightnessClampsAndKeepsAlpha() {
        var px: [UInt8] = [200, 100, 0, 77]
        MaterialOps.adjustAlbedo(&px, MaterialAdjustments(brightness: 0.5))
        XCTAssertEqual(px[0], 255)
        XCTAssertEqual(px[3], 77)
        XCTAssertGreaterThan(px[1], 100)
    }

    func testZeroSaturationIsGray() {
        var px: [UInt8] = [255, 0, 0, 255]
        MaterialOps.adjustAlbedo(&px, MaterialAdjustments(saturation: 0))
        XCTAssertEqual(px[0], px[1])
        XCTAssertEqual(px[1], px[2])
    }

    func testHueShift120RotatesRedTowardGreen() {
        var px: [UInt8] = [255, 0, 0, 255]
        MaterialOps.adjustAlbedo(&px, MaterialAdjustments(hue: 120))
        XCTAssertTrue(px[1] > px[0] && px[1] > px[2])
    }

    func testMRScalesOnlyGAndB() {
        var px: [UInt8] = [9, 100, 200, 255]
        MaterialOps.adjustMR(&px, MaterialAdjustments(roughness: 0.5, metalness: 0))
        XCTAssertEqual(px[0], 9)
        XCTAssertEqual(px[1], 50)
        XCTAssertEqual(px[2], 0)
    }

    func testPNGRoundTrip() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mat-\(UUID()).png")
        defer { try? FileManager.default.removeItem(at: url) }
        let px: [UInt8] = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 9, 9, 9, 255]
        XCTAssertTrue(MaterialOps.writePNG(px, width: 2, height: 2, to: url))
        let d = MaterialOps.decode(url)
        XCTAssertEqual(d?.rgba, px)
    }
}
