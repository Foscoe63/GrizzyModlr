import XCTest
import simd

final class ReliefMeshTests: XCTestCase {
    private func boundaryEdges(_ faces: [SIMD3<UInt32>]) -> Int {
        var dir = Set<UInt64>()
        for f in faces { let v = [f.x, f.y, f.z]; for k in 0..<3 { dir.insert(UInt64(v[k]) << 32 | UInt64(v[(k + 1) % 3])) } }
        return dir.filter { !dir.contains(($0 & 0xFFFF_FFFF) << 32 | ($0 >> 32)) }.count
    }
    private func volume(_ p: [SIMD3<Float>], _ f: [SIMD3<UInt32>]) -> Float {
        f.reduce(0) { $0 + simd_dot(p[Int($1.x)], simd_cross(p[Int($1.y)], p[Int($1.z)])) / 6 }
    }

    func testRampIsWatertightAndOutwardWound() throws {
        let (w, h) = (20, 12)
        let depth = (0..<(w * h)).map { Float($0 % w) / Float(w - 1) }                // nearer to the right
        let m = try XCTUnwrap(ReliefMesh.build(depth: depth, mask: nil, width: w, height: h,
                                               options: .init(reliefDepth: 0.2, baseThickness: 0.1, smoothing: 0)))
        XCTAssertEqual(boundaryEdges(m.faces), 0)
        XCTAssertGreaterThan(volume(m.positions, m.faces), 0)                          // outward winding
        let zs = m.positions.map(\.z)
        XCTAssertEqual(zs.min()!, 0, accuracy: 1e-6)                                   // flat back
        XCTAssertEqual(zs.max()!, 0.1 + 0.4, accuracy: 1e-4)                           // base + depth·relief(0.2·2)
        let xs = m.positions.map(\.x)
        XCTAssertEqual(xs.max()! - xs.min()!, 2 * Float(w - 1) / Float(w), accuracy: 1e-4)   // long side ≈ 2 units
    }

    func testMaskTrimsTheSilhouetteAndStaysClosed() throws {
        let (w, h) = (30, 30)
        var mask = [Bool](repeating: false, count: w * h)
        for y in 0..<h { for x in 0..<w { mask[y * w + x] = Float((x - 15) * (x - 15) + (y - 15) * (y - 15)) < 100 } }   // disk
        let depth = [Float](repeating: 0.5, count: w * h)
        let full = try XCTUnwrap(ReliefMesh.build(depth: depth, mask: nil, width: w, height: h, options: .init(smoothing: 0)))
        let disk = try XCTUnwrap(ReliefMesh.build(depth: depth, mask: mask, width: w, height: h, options: .init(smoothing: 0)))
        XCTAssertEqual(boundaryEdges(disk.faces), 0)
        XCTAssertLessThan(volume(disk.positions, disk.faces), volume(full.positions, full.faces) * 0.9)
        XCTAssertGreaterThan(volume(disk.positions, disk.faces), 0)
        XCTAssertNil(ReliefMesh.build(depth: depth, mask: [Bool](repeating: false, count: w * h), width: w, height: h))
    }

    func testNormalizeIsRobustToOutliersAndHonorsTheMask() {
        var d = (0..<1000).map { Float($0) / 999 }
        d[0] = -50; d[999] = 90                                                         // extreme outliers
        let n = ReliefMesh.normalize(d, mask: nil)
        XCTAssertEqual(n.min()!, 0); XCTAssertEqual(n.max()!, 1)
        XCTAssertGreaterThan(n[500], 0.4); XCTAssertLessThan(n[500], 0.6)               // middle stays in the middle
        let masked = ReliefMesh.normalize([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 7], mask: [false, false, false, false, false, false, false, false, false, false, true, true, true])
        XCTAssertEqual(masked.count, 13)
    }

    func testGridSizeKeepsAspect() {
        XCTAssertEqual(ReliefMesh.gridSize(imageWidth: 1000, imageHeight: 500, resolution: 200).0, 200)
        XCTAssertEqual(ReliefMesh.gridSize(imageWidth: 1000, imageHeight: 500, resolution: 200).1, 100)
        XCTAssertEqual(ReliefMesh.gridSize(imageWidth: 300, imageHeight: 900, resolution: 90).0, 30)
    }
}
