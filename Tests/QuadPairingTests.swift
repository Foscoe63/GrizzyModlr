import XCTest
import simd

final class QuadPairingTests: XCTestCase {
    /// An n×n quad grid in the XY plane, each quad split into two triangles (alternating diagonals).
    private func grid(_ n: Int) -> ([SIMD3<Float>], [SIMD3<UInt32>]) {
        var p: [SIMD3<Float>] = []
        for y in 0...n { for x in 0...n { p.append([Float(x), Float(y), 0]) } }
        var f: [SIMD3<UInt32>] = []
        for y in 0..<n { for x in 0..<n {
            let a = UInt32(y * (n + 1) + x), b = a + 1, c = a + UInt32(n + 1), d = c + 1
            if (x + y) % 2 == 0 { f.append([a, b, d]); f.append([a, d, c]) } else { f.append([a, b, c]); f.append([b, d, c]) }
        } }
        return (p, f)
    }

    private func area(_ p: [SIMD3<Float>], _ tri: SIMD3<UInt32>) -> Float {
        simd_length(simd_cross(p[Int(tri.y)] - p[Int(tri.x)], p[Int(tri.z)] - p[Int(tri.x)])) / 2
    }

    func testTriangulatedGridPairsBackIntoItsQuads() {
        let (p, f) = grid(6)
        let r = QuadPairing.pair(positions: p, faces: f)
        XCTAssertEqual(r.quads.count, 36)
        XCTAssertTrue(r.triangles.isEmpty)
        // every quad is a unit square: 4 distinct vertices, area 1 (two triangles of 0.5)
        for q in r.quads {
            XCTAssertEqual(Set([q.x, q.y, q.z, q.w]).count, 4)
            let a = area(p, SIMD3(q.x, q.y, q.z)) + area(p, SIMD3(q.x, q.z, q.w))
            XCTAssertEqual(a, 1, accuracy: 1e-4)
        }
    }

    func testWindingIsPreserved() {
        let (p, f) = grid(2)
        let r = QuadPairing.pair(positions: p, faces: f)
        for q in r.quads {                                                     // grid triangles are CCW (+Z)
            let n = simd_cross(p[Int(q.y)] - p[Int(q.x)], p[Int(q.z)] - p[Int(q.x)])
            XCTAssertGreaterThan(n.z, 0)
        }
    }

    func testBentEdgeIsNotMergedAcrossASharpFold() {
        // Two triangles folded 90° along their shared edge must stay triangles.
        let p: [SIMD3<Float>] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        let f: [SIMD3<UInt32>] = [[0, 1, 2], [0, 3, 1]]
        let r = QuadPairing.pair(positions: p, faces: f)
        XCTAssertTrue(r.quads.isEmpty)
        XCTAssertEqual(r.triangles.count, 2)
    }

    func testNonManifoldEdgesAndOddTriangleCountsStayTriangles() {
        let (p, f) = grid(1)                                                   // one quad = 2 triangles
        let extra = f + [SIMD3<UInt32>(0, 1, 3)]                               // a third face on edge 0-1 (non-manifold)
        let r = QuadPairing.pair(positions: p, faces: extra)
        XCTAssertEqual(r.quads.count * 2 + r.triangles.count, 3)
        XCTAssertLessThanOrEqual(r.quads.count, 1)
    }

    func testExporterWritesQuadFacesWhenAsked() throws {
        let (p, f) = grid(2)
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("qp-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let src = dir.appendingPathComponent("g.mesh")
        try MeshDecimator.serialize(positions: p, faces: f).write(to: src)
        let tri = dir.appendingPathComponent("tri.obj"), quad = dir.appendingPathComponent("quad.obj")
        try MeshExporter.export(meshURL: src, texture: nil, format: .obj, to: tri)
        try MeshExporter.export(meshURL: src, texture: nil, format: .obj, quads: true, to: quad)
        func faceLines(_ u: URL) throws -> [[Substring]] {
            try String(contentsOf: u).split(separator: "\n").filter { $0.hasPrefix("f ") }.map { $0.split(separator: " ") }
        }
        XCTAssertTrue(try faceLines(tri).allSatisfy { $0.count == 4 })           // "f" + 3 corners
        let q = try faceLines(quad)
        XCTAssertEqual(q.count, 4)                                               // 2×2 quads
        XCTAssertTrue(q.allSatisfy { $0.count == 5 })                            // "f" + 4 corners
    }
}
