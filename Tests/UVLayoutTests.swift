import XCTest
import simd

final class UVLayoutTests: XCTestCase {
    /// Two separate square charts (two triangles each): A on the left half, B on the right.
    private func twoIslands() -> ([SIMD3<UInt32>], [SIMD2<Float>]) {
        let uvs: [SIMD2<Float>] = [
            [0.05, 0.05], [0.45, 0.05], [0.45, 0.45], [0.05, 0.45],      // A: 0…3
            [0.55, 0.55], [0.95, 0.55], [0.95, 0.95], [0.55, 0.95],      // B: 4…7
        ]
        let faces: [SIMD3<UInt32>] = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]
        return (faces, uvs)
    }

    /// Texture that is red inside chart A and blue inside chart B, gray elsewhere.
    private func texture() -> TextureCanvas {
        let n = 128
        var px = [UInt8](repeating: 128, count: n * n * 4)
        for y in 0..<n { for x in 0..<n {
            let u = Float(x) / Float(n), v = Float(y) / Float(n)
            let i = (y * n + x) * 4
            px[i + 3] = 255
            if u > 0.05, u < 0.45, v > 0.05, v < 0.45 { px[i] = 255; px[i + 1] = 0; px[i + 2] = 0 }
            if u > 0.55, u < 0.95, v > 0.55, v < 0.95 { px[i] = 0; px[i + 1] = 0; px[i + 2] = 255 }
        } }
        return TextureCanvas(width: n, height: n, pixels: px)
    }

    private func rgb(_ t: TextureCanvas, u: Float, v: Float) -> SIMD3<Int> {
        let i = (Int(v * Float(t.height)) * t.width + Int(u * Float(t.width))) * 4
        return SIMD3(Int(t.pixels[i]), Int(t.pixels[i + 1]), Int(t.pixels[i + 2]))
    }

    func testFindsTwoIslandsWithBoundsAndOutline() {
        let (f, uv) = twoIslands()
        let isl = UVLayout.islands(faces: f, uvs: uv)
        XCTAssertEqual(isl.count, 2)
        XCTAssertEqual(isl[0].faces, [0, 1])
        XCTAssertEqual(isl[0].lo, SIMD2(0.05, 0.05))
        XCTAssertEqual(isl[1].hi, SIMD2(0.95, 0.95))
        XCTAssertEqual(UVLayout.outline(of: isl[0], faces: f).count, 4)   // the quad's 4 sides, not the diagonal
    }

    func testTranslateMovesColorAndLeavesOthers() {
        let (f, uv) = twoIslands()
        let isl = UVLayout.islands(faces: f, uvs: uv)
        let moved = UVLayout.apply([0: IslandTransform(du: 0.0, dv: 0.5)], islands: isl, to: uv)
        XCTAssertEqual(moved[0].x, 0.05, accuracy: 1e-5)
        XCTAssertEqual(moved[0].y, 0.55, accuracy: 1e-5)
        XCTAssertEqual(moved[4], uv[4])
        let out = UVLayout.remap(texture(), faces: f, faceIDs: isl[0].faces, oldUVs: uv, newUVs: moved)
        // The red chart now also appears 0.5 lower in v (but that area overlaps nothing here: v 0.55…0.95, u 0.05…0.45).
        XCTAssertEqual(rgb(out, u: 0.25, v: 0.75), SIMD3(255, 0, 0))
        XCTAssertEqual(rgb(out, u: 0.75, v: 0.75), SIMD3(0, 0, 255))      // chart B untouched
    }

    func testScaleHalfShrinksChartAboutItsCenter() {
        let (f, uv) = twoIslands()
        let isl = UVLayout.islands(faces: f, uvs: uv)
        let scaled = UVLayout.apply([1: IslandTransform(scale: 0.5)], islands: isl, to: uv)
        XCTAssertEqual(scaled[4].x, 0.75 - 0.1, accuracy: 1e-5)
        let out = UVLayout.remap(texture(), faces: f, faceIDs: isl[1].faces, oldUVs: uv, newUVs: scaled)
        XCTAssertEqual(rgb(out, u: 0.75, v: 0.75), SIMD3(0, 0, 255))      // center still blue
    }

    func testRotate90KeepsFootprintOfSquare() {
        let (f, uv) = twoIslands()
        let isl = UVLayout.islands(faces: f, uvs: uv)
        let r = UVLayout.apply([0: IslandTransform(rotation: .pi / 2)], islands: isl, to: uv)
        XCTAssertEqual(r[0].x, 0.45, accuracy: 1e-5)                       // (0.05,0.05) → (0.45,0.05)
        XCTAssertEqual(r[0].y, 0.05, accuracy: 1e-5)
    }

    func testOverlapAndUnitSquareChecks() {
        let (f, uv) = twoIslands()
        let isl = UVLayout.islands(faces: f, uvs: uv)
        let onto = UVLayout.apply([0: IslandTransform(du: 0.5, dv: 0.5)], islands: isl, to: uv)
        XCTAssertTrue(UVLayout.overlapsOthers(faces: f, uvs: onto, islands: isl, changed: [0]))
        XCTAssertFalse(UVLayout.overlapsOthers(faces: f, uvs: uv, islands: isl, changed: [0]))
        let off = UVLayout.apply([0: IslandTransform(du: 2)], islands: isl, to: uv)
        XCTAssertTrue(UVLayout.leavesUnitSquare(off, islands: isl, changed: [0]))
        XCTAssertFalse(UVLayout.leavesUnitSquare(uv, islands: isl, changed: [0]))
    }
}
