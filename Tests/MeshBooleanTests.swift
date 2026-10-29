import XCTest
import simd

final class MeshBooleanTests: XCTestCase {
    /// UV sphere with outward winding.
    private func sphere(center: SIMD3<Float> = .zero, radius: Float = 1, seg: Int = 32) -> ([SIMD3<Float>], [SIMD3<UInt32>]) {
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

    private func extent(_ p: [SIMD3<Float>]) -> (SIMD3<Float>, SIMD3<Float>) {
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for q in p { lo = simd_min(lo, q); hi = simd_max(hi, q) }
        return (lo, hi)
    }

    private func boundaryEdges(_ faces: [SIMD3<UInt32>]) -> Int {
        var dir = Set<UInt64>()
        for f in faces { let v = [f.x, f.y, f.z]; for k in 0..<3 { dir.insert(UInt64(v[k]) << 32 | UInt64(v[(k + 1) % 3])) } }
        return dir.filter { !dir.contains(($0 & 0xFFFF_FFFF) << 32 | ($0 >> 32)) }.count
    }

    /// Signed volume via the divergence theorem (positive for outward winding).
    private func volume(_ p: [SIMD3<Float>], _ f: [SIMD3<UInt32>]) -> Float {
        f.reduce(0) { $0 + simd_dot(p[Int($1.x)], simd_cross(p[Int($1.y)], p[Int($1.z)])) / 6 }
    }

    func testMeshFieldSignsInsideAndOutside() {
        let (p, f) = sphere()
        let g = MeshBoolean.grid(lo: [-1, -1, -1], hi: [1, 1, 1], resolution: 40)
        let field = MeshBoolean.meshField(positions: p, faces: f, grid: g)
        XCTAssertLessThan(field[g.index(g.nx / 2, g.ny / 2, g.nz / 2)], 0)
        XCTAssertGreaterThan(field[g.index(0, 0, 0)], 0)
    }

    func testRoundTripThroughSurfaceNetsKeepsSphereVolume() throws {
        let (p, f) = sphere()
        let g = MeshBoolean.grid(lo: [-1, -1, -1], hi: [1, 1, 1], resolution: 64)
        let field = MeshBoolean.meshField(positions: p, faces: f, grid: g)
        let m = MeshBoolean.surfaceNets(field, grid: g)
        XCTAssertEqual(boundaryEdges(m.faces), 0)
        let v = volume(m.positions, m.faces)
        XCTAssertGreaterThan(v, 0)                                   // outward winding
        XCTAssertLessThan(abs(v - 4.0 / 3.0 * .pi) / (4.0 / 3.0 * .pi), 0.06)
    }

    func testUnionOfTwoSpheres() throws {
        let a = sphere(center: [-0.5, 0, 0]), b = sphere(center: [0.5, 0, 0])
        let r = try XCTUnwrap(MeshBoolean.apply(.union, a: a, b: .init(kind: .mesh(positions: b.0, faces: b.1)), resolution: 72))
        let (lo, hi) = extent(r.positions)
        XCTAssertLessThan(abs(lo.x + 1.5), 0.06)
        XCTAssertLessThan(abs(hi.x - 1.5), 0.06)
        XCTAssertEqual(boundaryEdges(r.faces), 0)
        let single = Float(4.0 / 3.0 * .pi)
        let v = volume(r.positions, r.faces)
        XCTAssertGreaterThan(v, single * 1.3)
        XCTAssertLessThan(v, single * 2)
    }

    func testSubtractCarvesBoxOutOfSphere() throws {
        let a = sphere()
        let r = try XCTUnwrap(MeshBoolean.apply(.subtract, a: a,
                                                b: .init(kind: .primitive(.box, center: [0, 1, 0], size: 0.6)), resolution: 72))
        let (_, hi) = extent(r.positions)
        XCTAssertLessThan(hi.y, 0.85)                                 // pole removed; rim stays at √(1−0.6²)=0.8
        XCTAssertLessThan(volume(r.positions, r.faces), Float(4.0 / 3.0 * .pi) * 0.9)
    }

    func testIntersectWithSphereIsSmallerThanBoth() throws {
        let a = sphere(radius: 1)
        let r = try XCTUnwrap(MeshBoolean.apply(.intersect, a: a,
                                                b: .init(kind: .primitive(.sphere, center: [1, 0, 0], size: 1)), resolution: 72))
        let (lo, hi) = extent(r.positions)
        XCTAssertGreaterThan(lo.x, -0.08)                             // lens between x = 0 and x = 1
        XCTAssertLessThan(hi.x, 1.08)
        XCTAssertLessThan(volume(r.positions, r.faces), Float(4.0 / 3.0 * .pi) * 0.4)
    }

    func testDisjointIntersectIsNil() {
        let a = sphere()
        XCTAssertNil(MeshBoolean.apply(.intersect, a: a,
                                       b: .init(kind: .primitive(.sphere, center: [5, 0, 0], size: 0.5)), resolution: 48))
    }
}
