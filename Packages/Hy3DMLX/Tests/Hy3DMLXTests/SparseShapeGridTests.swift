import XCTest
import simd
import MLX
@testable import Hy3DMLX

/// The block-sparse shape decoder must give exactly what the dense reference gives, and keep working
/// at lattices (1024³–2048³) where a dense array cannot be held in memory.
final class SparseShapeGridTests: XCTestCase {
    /// Sphere unioned with a torus: concave, two kinds of curvature, value ≈ 12× distance so the
    /// ±0.95 band is a few voxels wide at these resolutions.
    private func sdf(_ p: SIMD3<Float>) -> Float {
        let sphere = simd_length(p - SIMD3(0.25, 0, 0)) - 0.35
        let q = SIMD2<Float>(simd_length(SIMD2(p.x + 0.3, p.z)) - 0.45, p.y)
        let torus = simd_length(q) - 0.14
        return 12 * Swift.min(sphere, torus)
    }
    private func decode(_ pts: [Float], _ count: Int) -> [Float] {
        (0 ..< count).map { sdf(SIMD3(pts[3 * $0], pts[3 * $0 + 1], pts[3 * $0 + 2])) }
    }

    private func sphereDecode(_ pts: [Float], _ count: Int) -> [Float] {
        (0 ..< count).map { 40 * (simd_length(SIMD3(pts[3 * $0], pts[3 * $0 + 1], pts[3 * $0 + 2])) - 0.6) }
    }

    func testSparseDecodeMatchesDenseReference() throws {
        for R in [128, 256] {
            let dense = OctreeSDF.dense(resolution: R, bound: 1.01, minResolution: 63, mcLevel: 0, decode: decode)
            let sparse = try XCTUnwrap(OctreeSDF.sparse(resolution: R, bound: 1.01, minResolution: 63, mcLevel: 0,
                                                       batch: 4096, decode: decode))
            XCTAssertEqual(sparse.n, dense.n)
            let s = sparse.toDense()
            XCTAssertEqual(s.count, dense.grid.count)
            var mismatches = 0, active = 0
            for t in 0 ..< s.count {
                let a = s[t], b = dense.grid[t]
                if a.isNaN != b.isNaN || (!a.isNaN && a != b) { mismatches += 1 }
                if !b.isNaN { active += 1 }
            }
            XCTAssertEqual(mismatches, 0, "R=\(R)")
            XCTAssertGreaterThan(active, 1000)
            XCTAssertLessThan(active, dense.n * dense.n * dense.n / 4, "band should be sparse")
        }
    }

    func testSparseMarchingCubesMatchesDense() throws {
        let R = 192
        let dense = OctreeSDF.dense(resolution: R, bound: 1.01, minResolution: 63, mcLevel: 0, decode: decode)
        let sparse = try XCTUnwrap(OctreeSDF.sparse(resolution: R, bound: 1.01, minResolution: 63, mcLevel: 0, decode: decode))
        let a = MarchingCubes.extract(grid: MLXArray(dense.grid, [dense.n, dense.n, dense.n]), level: 0)
        let b = try XCTUnwrap(MarchingCubes.extract(sparse: sparse, level: 0))
        XCTAssertEqual(a.vertices.count, b.vertices.count)
        XCTAssertEqual(a.faces.count, b.faces.count)
        func key(_ v: SIMD3<Float>) -> [Float] { [v.x, v.y, v.z] }
        let sa = a.vertices.sorted { key($0).lexicographicallyPrecedes(key($1)) }
        let sb = b.vertices.sorted { key($0).lexicographicallyPrecedes(key($1)) }
        var worst: Float = 0
        for t in 0 ..< Swift.min(sa.count, sb.count) { worst = Swift.max(worst, simd_distance(sa[t], sb[t])) }
        XCTAssertLessThan(worst, 1e-4)
        // same enclosed volume (winding and connectivity agree)
        func volume(_ m: Mesh) -> Float {
            m.faces.reduce(0) { $0 + simd_dot(m.vertices[Int($1.0)], simd_cross(m.vertices[Int($1.1)], m.vertices[Int($1.2)])) / 6 }
        }
        XCTAssertEqual(volume(a), volume(b), accuracy: 0.002)
    }

    func testCancelStopsDecode() {
        var calls = 0
        let grid = OctreeSDF.sparse(resolution: 256, bound: 1.01, minResolution: 63, mcLevel: 0, batch: 1024,
                                    isCancelled: { calls += 1; return calls > 3 }, decode: decode)
        XCTAssertNil(grid)
    }

    /// A sphere is a closed genus-0 triangle mesh, so F = 2V − 4 exactly when the extraction is watertight.
    private func checkSphere(R: Int) throws {
        let t0 = Date()
        let grid = try XCTUnwrap(OctreeSDF.sparse(resolution: R, bound: 1.01, minResolution: 63, mcLevel: 0, decode: sphereDecode))
        let t1 = Date()
        let mesh = try XCTUnwrap(MarchingCubes.extract(sparse: grid, level: 0))
        let t2 = Date()
        print("sparse R=\(R): decode \(t1.timeIntervalSince(t0))s, mesh \(t2.timeIntervalSince(t1))s, "
              + "\(grid.slots) blocks, \(mesh.vertices.count) verts, \(mesh.faces.count) faces")
        XCTAssertEqual(mesh.faces.count, 2 * mesh.vertices.count - 4)
        let r = mesh.vertices.prefix(2000).map { simd_length($0) }
        XCTAssertEqual(r.reduce(0, +) / Float(r.count), 0.6, accuracy: 0.002)
        // far below what a dense (R+1)³ float array would take
        XCTAssertLessThan(grid.data.count * 4, (R + 1) * (R + 1) * (R + 1) * 4 / 8)
    }

    func testSphereAt512IsWatertight() throws { try checkSphere(R: 512) }

    func testSphereAt2048IsWatertight() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MODELR_BIG_GRID"] != nil,
                          "set TEST_RUNNER_MODELR_BIG_GRID=1 (needs several GB, slow in Debug)")
        try checkSphere(R: 2048)
    }
}
