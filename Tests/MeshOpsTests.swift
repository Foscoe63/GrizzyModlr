import XCTest
import simd

final class MeshOpsTests: XCTestCase {

    /// Unit cube, 12 outward-wound triangles.
    private func cube(skipFaces: Set<Int> = []) -> (MeshOps.Positions, MeshOps.Faces) {
        let p: MeshOps.Positions = [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ]
        let f: [SIMD3<UInt32>] = [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7],
        ]
        return (p, f.enumerated().filter { !skipFaces.contains($0.offset) }.map(\.element))
    }

    private func boundaryEdgeCount(_ faces: MeshOps.Faces) -> Int {
        var dir = Set<UInt64>()
        for f in faces {
            let v = [f.x, f.y, f.z]
            for k in 0..<3 { dir.insert(UInt64(v[k]) << 32 | UInt64(v[(k + 1) % 3])) }
        }
        return dir.filter { e in
            let a = e >> 32, b = e & 0xFFFF_FFFF
            return !dir.contains(b << 32 | a)
        }.count
    }

    func testFillHolesClosesMissingFacePair() {
        let (p, f) = cube(skipFaces: [2, 3])          // remove the top quad → one 4-edge hole
        XCTAssertEqual(boundaryEdgeCount(f), 4)
        let r = MeshOps.fillHoles(positions: p, faces: f, maxEdges: 16)
        XCTAssertEqual(r.filled, 1)
        XCTAssertEqual(boundaryEdgeCount(r.faces), 0)
        XCTAssertEqual(r.positions.count, p.count + 1)
    }

    func testFillHolesLeavesClosedMeshAndLargeLoops() {
        let (p, f) = cube()
        XCTAssertEqual(MeshOps.fillHoles(positions: p, faces: f, maxEdges: 16).filled, 0)
        let (p2, f2) = cube(skipFaces: [2, 3])
        let r = MeshOps.fillHoles(positions: p2, faces: f2, maxEdges: 3)   // loop has 4 edges
        XCTAssertEqual(r.filled, 0)
        XCTAssertEqual(r.faces.count, f2.count)
    }

    func testSmoothReducesNoiseAndPinsBoundary() {
        // Flat open grid with one bumped interior vertex.
        var p: MeshOps.Positions = []
        for y in 0..<4 { for x in 0..<4 { p.append([Float(x), Float(y), 0]) } }
        var f: MeshOps.Faces = []
        for y in 0..<3 { for x in 0..<3 {
            let a = UInt32(y * 4 + x), b = a + 1, c = a + 4, d = c + 1
            f.append([a, b, d]); f.append([a, d, c])
        } }
        p[5].z = 1
        let s = MeshOps.smooth(positions: p, faces: f, iterations: 5)
        XCTAssertLessThan(abs(s[5].z), 1)
        for i in [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15] { XCTAssertEqual(s[i], p[i]) }
    }

    func testSmoothKeepsSizeRoughly() {
        let (p, f) = cube()
        let s = MeshOps.smooth(positions: p, faces: f, iterations: 3)
        let ext = s.map { $0.x }.max()! - s.map { $0.x }.min()!
        XCTAssertGreaterThan(ext, 0.5)
        XCTAssertTrue(s.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite })
    }

    func testApplyReportsUnchangedAndFileRoundTrip() throws {
        let (p, f) = cube(skipFaces: [2, 3])
        XCTAssertFalse(MeshOps.apply(.decimate(faceBudget: 100), positions: p, faces: f).changed)
        XCTAssertFalse(MeshOps.apply(.fillHoles(maxEdges: 3), positions: p, faces: f).changed)

        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mo-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let src = dir.appendingPathComponent("a.mesh"), dst = dir.appendingPathComponent("b.mesh")
        try MeshDecimator.serialize(positions: p, faces: f).write(to: src)
        let r = MeshOps.applyToFile(.fillHoles(maxEdges: 16), from: src, to: dst)
        XCTAssertEqual(r?.changed, true)
        let out = MeshDecimator.parse(try Data(contentsOf: dst))
        XCTAssertEqual(out?.faces.count, f.count + 4)
    }
}

final class TopologyOpsTests: XCTestCase {
    /// UV sphere (outward winding).
    private func sphere(center: SIMD3<Float> = .zero, radius: Float = 1, seg: Int = 24) -> ([SIMD3<Float>], [SIMD3<UInt32>]) {
        var p: [SIMD3<Float>] = [center + [0, radius, 0]]
        for i in 1..<seg {
            let phi = Float.pi * Float(i) / Float(seg)
            for j in 0..<(seg * 2) {
                let th = 2 * Float.pi * Float(j) / Float(seg * 2)
                p.append(center + radius * SIMD3(sin(phi) * cos(th), cos(phi), sin(phi) * sin(th)))
            }
        }
        p.append(center + [0, -radius, 0])
        let ring = seg * 2
        var f: [SIMD3<UInt32>] = []
        for j in 0..<ring { f.append([0, UInt32(1 + (j + 1) % ring), UInt32(1 + j)]) }
        for i in 0..<(seg - 2) { for j in 0..<ring {
            let a = UInt32(1 + i * ring + j), b = UInt32(1 + i * ring + (j + 1) % ring)
            let c = UInt32(1 + (i + 1) * ring + j), d = UInt32(1 + (i + 1) * ring + (j + 1) % ring)
            f.append([a, b, c]); f.append([b, d, c])
        } }
        let last = UInt32(p.count - 1), base = UInt32(1 + (seg - 2) * ring)
        for j in 0..<ring { f.append([last, base + UInt32(j), base + UInt32((j + 1) % ring)]) }
        return (p, f)
    }

    private func volume(_ p: [SIMD3<Float>], _ f: [SIMD3<UInt32>]) -> Float {
        f.reduce(0) { $0 + simd_dot(p[Int($1.x)], simd_cross(p[Int($1.y)], p[Int($1.z)])) / 6 }
    }
    private func boundaryEdges(_ faces: [SIMD3<UInt32>]) -> Int {
        var dir = Set<UInt64>()
        for f in faces { let v = [f.x, f.y, f.z]; for k in 0..<3 { dir.insert(UInt64(v[k]) << 32 | UInt64(v[(k + 1) % 3])) } }
        return dir.filter { !dir.contains(($0 & 0xFFFF_FFFF) << 32 | ($0 >> 32)) }.count
    }
    private func extent(_ p: [SIMD3<Float>], _ a: Int) -> (Float, Float) { (p.map { $0[a] }.min()!, p.map { $0[a] }.max()!) }

    /// A sphere whose +x half is shoved sideways in z — asymmetric, but the x extent (and so
    /// the mirror plane at the bounding-box center, x = 0) is unchanged.
    private func lopsided() -> ([SIMD3<Float>], [SIMD3<UInt32>]) {
        var (p, f) = sphere()
        for i in p.indices where p[i].x > 0.3 { p[i].z += 0.25 }
        return (p, f)
    }

    func testSymmetrizeKeepNegativeDropsTheLopsidedHalfAndStaysClosed() throws {
        let (p, f) = lopsided()
        let r = try XCTUnwrap(MeshOps.symmetrize(positions: p, faces: f, axis: 0, keepPositive: false))   // keep −x half
        let (lo, hi) = extent(r.positions, 0)
        XCTAssertLessThan(abs(hi + lo), 0.02)                      // symmetric about x = 0
        XCTAssertLessThan(extent(r.positions, 2).1, 1.02)          // the shoved +x half is gone
        XCTAssertEqual(boundaryEdges(r.faces), 0)
        XCTAssertGreaterThan(volume(r.positions, r.faces), 3.5)    // ~ a sphere's 4.19, not a shell
    }

    func testSymmetrizeKeepPositiveMirrorsTheLopsidedHalf() throws {
        let (p, f) = lopsided()
        let r = try XCTUnwrap(MeshOps.symmetrize(positions: p, faces: f, axis: 0, keepPositive: true))
        let (lo, hi) = extent(r.positions, 0)
        XCTAssertLessThan(abs(hi + lo), 0.02)
        XCTAssertGreaterThan(extent(r.positions, 2).1, 1.1)        // the z shove now appears on both sides
        XCTAssertEqual(boundaryEdges(r.faces), 0)
    }

    func testSymmetrizeIsIdempotentOnASymmetricMesh() throws {
        let (p, f) = sphere()
        let r = try XCTUnwrap(MeshOps.symmetrize(positions: p, faces: f, axis: 1, keepPositive: true))
        XCTAssertLessThan(abs(volume(r.positions, r.faces) - volume(p, f)) / volume(p, f), 0.03)
        XCTAssertEqual(boundaryEdges(r.faces), 0)
    }

    func testRemeshKeepsVolumeAndIsClosed() throws {
        let (p, f) = sphere()
        let r = try XCTUnwrap(MeshBoolean.remesh(positions: p, faces: f, resolution: 48))
        XCTAssertEqual(boundaryEdges(r.faces), 0)
        let v = volume(r.positions, r.faces), ref = Float(4.0 / 3.0 * .pi)
        XCTAssertLessThan(abs(v - ref) / ref, 0.08)
        XCTAssertTrue(r.positions.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite })
    }

    func testApplyDispatchesNewEdits() {
        let (p, f) = sphere()
        XCTAssertTrue(MeshOps.apply(.symmetrize(axis: .z, keepPositive: true), positions: p, faces: f).changed)
        XCTAssertTrue(MeshOps.apply(.remesh(resolution: 32), positions: p, faces: f).changed)
        XCTAssertEqual(MeshEdit.symmetrize(axis: .x, keepPositive: false).label, "mirrored −X half")
    }
}
