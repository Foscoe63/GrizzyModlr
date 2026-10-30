import XCTest
import simd

final class NormalBakerTests: XCTestCase {
    /// A unit quad in the XY plane facing +Z: u → +x, image-down (v) → −y, so the image top is +y.
    private func quad() -> TexturedMesh {
        TexturedMesh(
            positions: [SIMD3(0, 1, 0), SIMD3(1, 1, 0), SIMD3(1, 0, 0), SIMD3(0, 0, 0)],
            normals: Array(repeating: SIMD3(0, 0, 1), count: 4),
            uvs: [SIMD2(0, 0), SIMD2(1, 0), SIMD2(1, 1), SIMD2(0, 1)],
            faces: [SIMD3(0, 3, 2), SIMD3(0, 2, 1)])
    }

    /// The same quad rotated by `angle` about `axis` around its center, as a high-res stand-in.
    private func tilted(_ angle: Float, axis: SIMD3<Float>) -> NormalBaker.HighMesh {
        let q = simd_quatf(angle: angle, axis: axis)
        let c = SIMD3<Float>(0.5, 0.5, 0)
        let pts = quad().positions.map { q.act($0 - c) * 1.6 + c }      // bigger so it covers the cage
        return .init(positions: pts, faces: quad().faces)
    }

    private func pixel(_ rgb: [UInt8], _ size: Int, _ x: Int, _ y: Int) -> SIMD3<Float> {
        let o = (y * size + x) * 3
        return SIMD3(Float(rgb[o]), Float(rgb[o + 1]), Float(rgb[o + 2])) / 255 * 2 - 1
    }

    func testFlatSurfaceBakesFlatNormal() throws {
        let rgb = try XCTUnwrap(NormalBaker.bake(
            low: quad(), high: .init(positions: quad().positions, faces: quad().faces),
            options: .init(size: 32)))
        let p = pixel(rgb, 32, 16, 16)
        XCTAssertEqual(p.x, 0, accuracy: 0.02)
        XCTAssertEqual(p.y, 0, accuracy: 0.02)
        XCTAssertEqual(p.z, 1, accuracy: 0.02)
    }

    func testTiltTowardTangentRaisesRed() throws {
        // Rotating about +y swings the normal toward +x, the U tangent.
        let angle: Float = 0.5
        let rgb = try XCTUnwrap(NormalBaker.bake(low: quad(), high: tilted(angle, axis: SIMD3(0, 1, 0)),
                                                 options: .init(size: 32, cage: 0.2)))
        let p = pixel(rgb, 32, 16, 16)
        XCTAssertEqual(p.x, sin(angle), accuracy: 0.05)
        XCTAssertEqual(p.y, 0, accuracy: 0.05)
    }

    func testTiltTowardImageUpRaisesGreen() throws {
        // Rotating about −x swings the normal toward +y, which is image-up: green goes positive.
        let angle: Float = 0.5
        let rgb = try XCTUnwrap(NormalBaker.bake(low: quad(), high: tilted(angle, axis: SIMD3(-1, 0, 0)),
                                                 options: .init(size: 32, cage: 0.2)))
        let p = pixel(rgb, 32, 16, 16)
        XCTAssertEqual(p.y, sin(angle), accuracy: 0.05)
        XCTAssertEqual(p.x, 0, accuracy: 0.05)
    }

    func testMirroredUVsKeepWorldNormalMeaning() throws {
        // Flip u: the tangent frame mirrors, but the stored tangent-space normal must still
        // reconstruct to the same world normal. T = dP/du now points −x, so a +x world tilt reads −red.
        var mirrored = quad()
        mirrored.uvs = mirrored.uvs.map { SIMD2(1 - $0.x, $0.y) }
        let angle: Float = 0.5
        let rgb = try XCTUnwrap(NormalBaker.bake(low: mirrored, high: tilted(angle, axis: SIMD3(0, 1, 0)),
                                                 options: .init(size: 32, cage: 0.2)))
        let p = pixel(rgb, 32, 16, 16)
        XCTAssertEqual(p.x, -sin(angle), accuracy: 0.05)
    }

    func testUncoveredTexelsBecomeFlat() throws {
        var half = quad()
        half.faces = [half.faces[0]]                       // only half the square is covered
        let rgb = try XCTUnwrap(NormalBaker.bake(low: half, high: tilted(0.5, axis: SIMD3(0, 1, 0)),
                                                 options: .init(size: 32, cage: 0.2, padding: 0)))
        let empty = pixel(rgb, 32, 30, 2)                  // outside the triangle, no padding
        XCTAssertEqual(empty, SIMD3(0.004, 0.004, 1), accuracy: 0.01)
    }

    func testBVHFindsNearestHit() {
        let tris = NormalBaker.BVH(positions: [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(0, 1, 0),
                                               SIMD3(0, 0, 1), SIMD3(1, 0, 1), SIMD3(0, 1, 1)],
                                   faces: [SIMD3(0, 1, 2), SIMD3(3, 4, 5)])
        let hit = tris.closestHit(origin: SIMD3(0.2, 0.2, 3), direction: SIMD3(0, 0, -1), maxDistance: 10)
        XCTAssertEqual(hit?.face, 1)
        XCTAssertEqual(hit?.t ?? 0, 2, accuracy: 1e-4)
        XCTAssertNil(tris.closestHit(origin: SIMD3(0.2, 0.2, 3), direction: SIMD3(0, 0, -1), maxDistance: 1))
    }

    func testPNGRoundTrip() throws {
        let rgb = [UInt8](repeating: 200, count: 16 * 16 * 3)
        let data = try XCTUnwrap(NormalBaker.pngData(rgb: rgb, size: 16))
        XCTAssertEqual(Array(data.prefix(4)), [0x89, 0x50, 0x4E, 0x47])
    }
}

private func XCTAssertEqual(_ a: SIMD3<Float>, _ b: SIMD3<Float>, accuracy: Float,
                            file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(a.x, b.x, accuracy: accuracy, file: file, line: line)
    XCTAssertEqual(a.y, b.y, accuracy: accuracy, file: file, line: line)
    XCTAssertEqual(a.z, b.z, accuracy: accuracy, file: file, line: line)
}
