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

final class MaterialRegionTests: XCTestCase {
    private func twoColorImage() -> [UInt8] {
        var px: [UInt8] = []
        for i in 0..<100 { px += i < 50 ? [200, 30, 30, 255] : [30, 30, 200, 255] }
        return px
    }

    func testDominantColorsFindsBothColors() {
        let colors = MaterialRegion.dominantColors(twoColorImage())
        XCTAssertEqual(colors.count, 2)
        XCTAssertTrue(colors.contains { $0.x > 0.7 && $0.z < 0.2 })
        XCTAssertTrue(colors.contains { $0.z > 0.7 && $0.x < 0.2 })
    }

    func testMaskSelectsOnlyNearColor() {
        let px = twoColorImage()
        let red = SIMD3<Float>(200, 30, 30) / 255
        let mask = MaterialRegion.mask(px, region: ColorRegion(color: red))
        XCTAssertEqual(mask[0], 1); XCTAssertEqual(mask[99], 0)
        XCTAssertEqual(MaterialRegion.coverage(of: mask, rgba: px), 0.5, accuracy: 0.01)
    }

    func testBlendEditsOnlyMaskedTexels() {
        let original = twoColorImage()
        var edited = original
        MaterialOps.adjustAlbedo(&edited, MaterialAdjustments(brightness: 0.3))
        let red = SIMD3<Float>(200, 30, 30) / 255
        let mask = MaterialRegion.mask(original, region: ColorRegion(color: red))
        MaterialRegion.blend(original: original, edited: &edited, mask: mask, maskSize: (10, 10), size: (10, 10))
        XCTAssertGreaterThan(edited[0], original[0] - 1)   // red texel brightened (clamped at 255 channel-wise)
        XCTAssertGreaterThan(edited[1], original[1])
        XCTAssertEqual(Array(edited[(99 * 4)..<(99 * 4 + 4)]), Array(original[(99 * 4)..<(99 * 4 + 4)]))
    }

    func testBlendResamplesMaskOfDifferentSize() {
        let original = [UInt8](repeating: 100, count: 4 * 4 * 4)
        var edited = [UInt8](repeating: 200, count: 4 * 4 * 4)
        let mask: [Float] = [1, 0, 0, 0]                    // 2x2: only top-left quadrant selected
        MaterialRegion.blend(original: original, edited: &edited, mask: mask, maskSize: (2, 2), size: (4, 4))
        XCTAssertEqual(edited[0], 200)                       // pixel (0,0) edited
        XCTAssertEqual(edited[(3 * 4 + 3) * 4], 100)         // pixel (3,3) untouched
    }
}
