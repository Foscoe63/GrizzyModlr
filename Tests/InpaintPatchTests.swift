import XCTest

final class InpaintPatchTests: XCTestCase {
    func testMaskBoundsAndWindowStayInsideTexture() {
        var mask = [Float](repeating: 0, count: 1000 * 800)
        for y in 700..<720 { for x in 950..<990 { mask[y * 1000 + x] = 1 } }
        let b = try! XCTUnwrap(InpaintPatch.maskBounds(mask, width: 1000, height: 800))
        XCTAssertEqual(b.x0, 950); XCTAssertEqual(b.y1, 719)
        let w = InpaintPatch.window(bounds: b, width: 1000, height: 800)
        XCTAssertGreaterThanOrEqual(w.x, 0); XCTAssertGreaterThanOrEqual(w.y, 0)
        XCTAssertLessThanOrEqual(w.x + w.side, 1000); XCTAssertLessThanOrEqual(w.y + w.side, 800)
        XCTAssertGreaterThanOrEqual(w.side, 192)
        XCTAssertTrue(w.x <= b.x0 && w.x + w.side > b.x1 && w.y <= b.y0 && w.y + w.side > b.y1)
    }

    func testEmptyMaskHasNoBounds() {
        XCTAssertNil(InpaintPatch.maskBounds([Float](repeating: 0, count: 16), width: 4, height: 4))
    }

    func testCompositeLeavesOutsideMaskUntouchedAndKeepsAlpha() {
        let (w, h) = (256, 256)
        var px = [UInt8](repeating: 0, count: w * h * 4)
        for i in 0..<(w * h) { px[i * 4] = 50; px[i * 4 + 1] = 60; px[i * 4 + 2] = 70; px[i * 4 + 3] = 200 }
        let before = px
        let win = InpaintPatch.Window(x: 0, y: 0, side: 256)
        var weight = [Float](repeating: 0, count: 256 * 256)
        for y in 100..<140 { for x in 100..<140 { weight[y * 256 + x] = 1 } }
        let result = [Float](repeating: 1, count: 512 * 512 * 3)          // white
        InpaintPatch.composite(&px, width: w, result: result, window: win, weight: weight)
        XCTAssertEqual(Array(px[0..<4]), Array(before[0..<4]))
        let inside = (120 * w + 120) * 4
        XCTAssertEqual(px[inside], 255); XCTAssertEqual(px[inside + 3], 200)
    }

    func testExtractResamplesConstantImageExactly() {
        let (w, h) = (300, 300)
        var px = [UInt8](repeating: 0, count: w * h * 4)
        for i in 0..<(w * h) { px[i * 4] = 255; px[i * 4 + 1] = 128; px[i * 4 + 3] = 255 }
        let rgb = InpaintPatch.extractRGB(px, width: w, window: .init(x: 10, y: 20, side: 200))
        XCTAssertEqual(rgb.count, 512 * 512 * 3)
        XCTAssertEqual(rgb[0], 1, accuracy: 1e-5)
        XCTAssertEqual(rgb[1], 128 / 255, accuracy: 1e-5)
    }

    func testFeatheredMaskGrowsAndSoftens() {
        var m = [Float](repeating: 0, count: 64 * 64)
        for y in 30..<34 { for x in 30..<34 { m[y * 64 + x] = 1 } }
        let f = InpaintPatch.feathered(m, width: 64, height: 64, grow: 3, blur: 3)
        XCTAssertGreaterThan(f[29 * 64 + 29], 0)                  // grew past the original edge
        XCTAssertLessThan(f[28 * 64 + 28], f[32 * 64 + 32])       // soft falloff
        XCTAssertEqual(f[0], 0)
    }

    func testReplaceUndoableRestoresOriginal() {
        var c = TextureCanvas(width: 130, height: 70, pixels: [UInt8](repeating: 10, count: 130 * 70 * 4))
        var next = c.pixels
        next[(40 * 130 + 100) * 4] = 99
        c.replaceUndoable(next)
        XCTAssertEqual(c.pixels[(40 * 130 + 100) * 4], 99)
        XCTAssertTrue(c.canUndo)
        c.undo()
        XCTAssertEqual(c.pixels[(40 * 130 + 100) * 4], 10)
    }
}
