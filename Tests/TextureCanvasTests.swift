import XCTest
import simd

final class TextureCanvasTests: XCTestCase {
    private func gradient(_ w: Int = 64, _ h: Int = 64) -> TextureCanvas {
        var px = [UInt8](repeating: 255, count: w * h * 4)
        for y in 0..<h { for x in 0..<w {
            let i = (y * w + x) * 4
            px[i] = UInt8(x * 255 / (w - 1)); px[i + 1] = 100; px[i + 2] = UInt8(y * 255 / (h - 1))
        } }
        return TextureCanvas(width: w, height: h, pixels: px)
    }
    private func rgb(_ c: TextureCanvas, _ x: Int, _ y: Int) -> SIMD3<Int> {
        let i = (y * c.width + x) * 4
        return SIMD3(Int(c.pixels[i]), Int(c.pixels[i + 1]), Int(c.pixels[i + 2]))
    }

    func testPaintCenterAndLocality() {
        var c = gradient()
        c.paint(u: 0.5, v: 0.5, radius: 6, color: [255, 0, 0, 255], hardness: 1, opacity: 1)
        XCTAssertEqual(rgb(c, 32, 32), SIMD3(255, 0, 0))
        XCTAssertEqual(rgb(c, 2, 2), SIMD3(2 * 255 / 63, 100, 2 * 255 / 63))   // untouched
        XCTAssertTrue(c.isEdited)
    }

    func testUVMappingIsColumnUTimesWidthRowVTimesHeight() {
        var c = gradient()
        c.paint(u: 0.25, v: 0.75, radius: 2, color: [0, 255, 0, 255], hardness: 1, opacity: 1)
        XCTAssertEqual(rgb(c, 16, 48), SIMD3(0, 255, 0))
    }

    func testOpacityBlends() {
        var c = gradient()
        let before = rgb(c, 32, 32)
        c.paint(u: 0.5, v: 0.5, radius: 4, color: [0, 0, 0, 255], hardness: 1, opacity: 0.5)
        let after = rgb(c, 32, 32)
        XCTAssertLessThan(after.x, before.x)
        XCTAssertGreaterThan(after.x, 0)
    }

    func testHealFillsBlemishWithSurroundingGradient() {
        var c = gradient()
        let expected = rgb(c, 32, 32)
        c.paint(u: 0.5, v: 0.5, radius: 6, color: [255, 0, 255, 255], hardness: 1, opacity: 1)
        XCTAssertGreaterThan(abs(rgb(c, 32, 32).y - expected.y), 50)
        c.heal(u: 0.5, v: 0.5, radius: 8)
        let healed = rgb(c, 32, 32)
        XCTAssertLessThan(abs(healed.x - expected.x), 12)
        XCTAssertLessThan(abs(healed.y - expected.y), 12)
        XCTAssertLessThan(abs(healed.z - expected.z), 12)
    }

    func testSoftenReducesContrast() {
        var px = [UInt8](repeating: 0, count: 32 * 32 * 4)
        for i in stride(from: 3, to: px.count, by: 4) { px[i] = 255 }
        for y in 0..<32 { for x in 16..<32 { let i = (y * 32 + x) * 4; px[i] = 255; px[i + 1] = 255; px[i + 2] = 255 } }
        var c = TextureCanvas(width: 32, height: 32, pixels: px)
        let edgeBefore = Int(c.pixels[(16 * 32 + 15) * 4])
        for _ in 0..<4 { c.soften(u: 0.5, v: 0.5, radius: 8, strength: 1) }
        XCTAssertGreaterThan(Int(c.pixels[(16 * 32 + 15) * 4]), edgeBefore)
    }

    func testUndoRestoresAcrossTiles() {
        var c = gradient(200, 200)
        let before = c.pixels
        c.beginStroke()
        c.paint(u: 0.5, v: 0.5, radius: 40, color: [1, 2, 3, 255], hardness: 1, opacity: 1)
        c.soften(u: 0.2, v: 0.2, radius: 10, strength: 1)
        c.endStroke()
        XCTAssertNotEqual(c.pixels, before)
        c.undo()
        XCTAssertEqual(c.pixels, before)
        XCTAssertFalse(c.isEdited)
    }

    func testTexturedMeshRoundTripAndTexelDensity() throws {
        let m = TexturedMesh(
            positions: [[0, 0, 0], [2, 0, 0], [0, 2, 0]], normals: [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            uvs: [[0, 0], [0.5, 0], [0, 0.5]], faces: [[0, 1, 2]])
        let back = try XCTUnwrap(TexturedMesh.parse(m.serialize()))
        XCTAssertEqual(back.positions, m.positions)
        XCTAssertEqual(back.uvs, m.uvs)
        XCTAssertEqual(back.faces, m.faces)
        // Edges of world length 2 span 0.5·1000 texels on the axes (0.707·1000 on the hypotenuse).
        let d = m.texelsPerUnit(face: 0, textureWidth: 1000)
        XCTAssertGreaterThan(d, 240)
        XCTAssertLessThan(d, 300)
        XCTAssertNil(TexturedMesh.parse(Data(count: 4)))
    }
}
