import XCTest
import simd

final class TripoMeshTests: XCTestCase {
    /// Density grid (x slowest) with value 100 inside the given blobs, 0 outside.
    private func density(n: Int, radius: Float, blobs: [(c: SIMD3<Float>, r: SIMD3<Float>)]) -> [Float] {
        var d = [Float](repeating: 0, count: n * n * n)
        let step = 2 * radius / Float(n - 1)
        for i in 0..<n { for j in 0..<n { for k in 0..<n {
            let p = SIMD3<Float>(Float(i), Float(j), Float(k)) * step - SIMD3(repeating: radius)
            for b in blobs {
                let q = (p - b.c) / b.r
                if simd_length(q) < 1 { d[(i * n + j) * n + k] = 100 }
            }
        } } }
        return d
    }

    private func boundaryEdges(_ faces: [SIMD3<UInt32>]) -> Int {
        var dir = Set<UInt64>()
        for f in faces { let v = [f.x, f.y, f.z]; for k in 0..<3 { dir.insert(UInt64(v[k]) << 32 | UInt64(v[(k + 1) % 3])) } }
        return dir.filter { !dir.contains(($0 & 0xFFFF_FFFF) << 32 | ($0 >> 32)) }.count
    }
    private func volume(_ p: [SIMD3<Float>], _ f: [SIMD3<UInt32>]) -> Float {
        f.reduce(0) { $0 + simd_dot(p[Int($1.x)], simd_cross(p[Int($1.y)], p[Int($1.z)])) / 6 }
    }

    func testAxesMapToModelrFrameAndMeshIsClosedOutward() throws {
        // an ellipsoid long along TripoSR's z (up) and offset toward +x (front) and +y (image-right)
        let d = density(n: 56, radius: 0.87, blobs: [(c: [0.1, 0.2, 0], r: [0.2, 0.25, 0.55])])
        let m = try XCTUnwrap(TripoMesh.extract(density: d, resolution: 56, radius: 0.87, threshold: 25))
        XCTAssertEqual(boundaryEdges(m.faces), 0)
        XCTAssertGreaterThan(volume(m.positions, m.faces), 0)                         // outward winding survives the remap
        func extent(_ a: Int) -> (Float, Float) { (m.positions.map { $0[a] }.min()!, m.positions.map { $0[a] }.max()!) }
        let (xl, xh) = extent(0), (yl, yh) = extent(1), (zl, zh) = extent(2)
        XCTAssertEqual(yh - yl, 1.1, accuracy: 0.1)                                   // TripoSR z → Modelr y (height)
        XCTAssertEqual((xl + xh) / 2, 0.2, accuracy: 0.03)                            // TripoSR y → Modelr x
        XCTAssertEqual((zl + zh) / 2, 0.1, accuracy: 0.03)                            // TripoSR x → Modelr z
    }

    func testKeepsOnlyTheLargestPiece() throws {
        let blobs: [(c: SIMD3<Float>, r: SIMD3<Float>)] = [(c: [0, 0, 0], r: [0.4, 0.4, 0.4]), (c: [0.6, 0.6, 0.6], r: [0.1, 0.1, 0.1])]
        let d = density(n: 56, radius: 0.87, blobs: blobs)
        let all = try XCTUnwrap(TripoMesh.extract(density: d, resolution: 56, radius: 0.87, threshold: 25, keepLargestPiece: false))
        let kept = try XCTUnwrap(TripoMesh.extract(density: d, resolution: 56, radius: 0.87, threshold: 25))
        XCTAssertLessThan(kept.faces.count, all.faces.count)
        XCTAssertLessThan(kept.positions.map { $0.x }.max()!, 0.5)                    // the speck at 0.6 is gone
        XCTAssertEqual(boundaryEdges(kept.faces), 0)
    }

    func testRejectsBadInput() {
        XCTAssertNil(TripoMesh.extract(density: [1, 2, 3], resolution: 8, radius: 1, threshold: 25))
        XCTAssertNil(TripoMesh.extract(density: [Float](repeating: 0, count: 512), resolution: 8, radius: 1, threshold: 25))   // empty
    }
}
