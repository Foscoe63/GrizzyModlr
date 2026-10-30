import XCTest
import simd

final class CrossFieldTests: XCTestCase {
    // MARK: shapes

    private func icosphere(subdivisions: Int) -> ([SIMD3<Float>], [SIMD3<UInt32>]) {
        let t: Float = (1 + Float(5).squareRoot()) / 2
        let raw: [[Float]] = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                              [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
        var p: [SIMD3<Float>] = raw.map { simd_normalize(SIMD3<Float>($0[0], $0[1], $0[2])) }
        let tris: [[UInt32]] = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                                [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                                [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
        var f: [SIMD3<UInt32>] = tris.map { SIMD3<UInt32>($0[0], $0[1], $0[2]) }
        for _ in 0..<subdivisions {
            var mid: [UInt64: UInt32] = [:]
            func midpoint(_ a: UInt32, _ b: UInt32) -> UInt32 {
                let k = UInt64(min(a, b)) << 32 | UInt64(max(a, b))
                if let m = mid[k] { return m }
                p.append(simd_normalize(p[Int(a)] + p[Int(b)]))
                mid[k] = UInt32(p.count - 1)
                return mid[k]!
            }
            var next: [SIMD3<UInt32>] = []
            for tri in f {
                let a = midpoint(tri.x, tri.y), b = midpoint(tri.y, tri.z), c = midpoint(tri.z, tri.x)
                next += [[tri.x, a, c], [tri.y, b, a], [tri.z, c, b], [a, b, c]]
            }
            f = next
        }
        return (p, f)
    }

    private func torus(major: Float = 1, minor: Float = 0.35, n: Int = 48, m: Int = 24) -> ([SIMD3<Float>], [SIMD3<UInt32>]) {
        var p: [SIMD3<Float>] = []
        for i in 0..<n { for j in 0..<m {
            let u = 2 * Float.pi * Float(i) / Float(n), v = 2 * Float.pi * Float(j) / Float(m)
            p.append([(major + minor * cos(v)) * cos(u), (major + minor * cos(v)) * sin(u), minor * sin(v)])
        } }
        var f: [SIMD3<UInt32>] = []
        for i in 0..<n { for j in 0..<m {
            let a = UInt32(i * m + j), b = UInt32(((i + 1) % n) * m + j)
            let c = UInt32(((i + 1) % n) * m + (j + 1) % m), d = UInt32(i * m + (j + 1) % m)
            f.append([a, b, c]); f.append([a, c, d])
        } }
        return (p, f)
    }

    /// Capped cylinder along Z (radius 1, height 4) with `seg` sides and `rings` rows.
    private func cylinder(seg: Int = 48, rings: Int = 24) -> ([SIMD3<Float>], [SIMD3<UInt32>]) {
        var p: [SIMD3<Float>] = []
        for r in 0...rings { for s in 0..<seg {
            let a = 2 * Float.pi * Float(s) / Float(seg)
            p.append([cos(a), sin(a), -2 + 4 * Float(r) / Float(rings)])
        } }
        var f: [SIMD3<UInt32>] = []
        for r in 0..<rings { for s in 0..<seg {
            let a = UInt32(r * seg + s), b = UInt32(r * seg + (s + 1) % seg)
            let c = UInt32((r + 1) * seg + (s + 1) % seg), d = UInt32((r + 1) * seg + s)
            f.append([a, b, c]); f.append([a, c, d])
        } }
        let bottom = UInt32(p.count); p.append([0, 0, -2])
        let top = UInt32(p.count); p.append([0, 0, 2])
        for s in 0..<seg {
            let a = UInt32(s), b = UInt32((s + 1) % seg)
            f.append([bottom, b, a])
            let ta = UInt32(rings * seg + s), tb = UInt32(rings * seg + (s + 1) % seg)
            f.append([top, ta, tb])
        }
        return (p, f)
    }

    // MARK: topology

    func testTopologyBuildsClosedFansAndNeighbors() {
        let (p, f) = icosphere(subdivisions: 1)
        let t = MeshTopology(positions: p, faces: f)
        XCTAssertTrue(t.neighbors.allSatisfy { $0.count == 3 })
        XCTAssertTrue(t.vertexFan.allSatisfy { $0.count == 5 || $0.count == 6 })
    }

    func testSphereCurvatureIsIsotropic() {
        let (p, f) = icosphere(subdivisions: 2)
        let t = MeshTopology(positions: p, faces: f)
        let spread = (0..<f.count).map { t.curvatureDirection(face: $0).anisotropy }
        XCTAssertLessThan(spread.sorted()[spread.count / 2], 0.25)        // κ₁ ≈ κ₂ ≈ 1
    }

    func testCylinderCurvatureDirectionIsAxisOrCircumference() {
        let (p, f) = cylinder()
        let t = MeshTopology(positions: p, faces: f)
        var good = 0, total = 0
        for i in 0..<(48 * 24 * 2) {                                       // side faces only
            let (a, spread) = t.curvatureDirection(face: i)
            guard spread > 0.5 else { continue }
            total += 1
            let d = cos(a) * t.frameX[i] + sin(a) * t.frameY[i]
            if abs(d.z) > 0.97 || abs(d.z) < 0.25 { good += 1 }
        }
        XCTAssertGreaterThan(total, 1000)
        XCTAssertGreaterThan(Float(good) / Float(total), 0.95)
    }

    // MARK: field

    func testCylinderFieldFollowsAxisAndCircumference() {
        let (p, f) = cylinder()
        let field = CrossField.compute(topology: MeshTopology(positions: p, faces: f))
        var good = 0, total = 0
        for i in 0..<(48 * 24 * 2) {
            let c = field.topology.centroid(i)
            guard abs(c.z) < 1.5 else { continue }                          // away from the caps
            total += 1
            if field.misalignment(of: [0, 0, 1], face: i) < 0.17 { good += 1 }     // within ~10°
        }
        XCTAssertGreaterThan(Float(good) / Float(total), 0.9)
    }

    func testBoxFieldFollowsCreases() {
        // A box with each face split into a 6×6 grid so creases have interior faces beside them.
        var p: [SIMD3<Float>] = [], f: [SIMD3<UInt32>] = []
        let n = 6
        let axes: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = [
            ([1, 0, 0], [0, 1, 0], [0, 0, 1]), ([-1, 0, 0], [0, 0, 1], [0, 1, 0]),
            ([0, 1, 0], [0, 0, 1], [1, 0, 0]), ([0, -1, 0], [1, 0, 0], [0, 0, 1]),
            ([0, 0, 1], [1, 0, 0], [0, 1, 0]), ([0, 0, -1], [0, 1, 0], [1, 0, 0])]
        for (nrm, u, v) in axes {
            let base = UInt32(p.count)
            for j in 0...n { for i in 0...n {
                p.append(nrm + u * (2 * Float(i) / Float(n) - 1) + v * (2 * Float(j) / Float(n) - 1))
            } }
            for j in 0..<n { for i in 0..<n {
                let a = base + UInt32(j * (n + 1) + i), b = a + 1, c = a + UInt32(n + 1), d = c + 1
                f.append([a, b, d]); f.append([a, d, c])
            } }
        }
        // Weld the duplicated edge/corner vertices so faces share edges.
        var index: [SIMD3<Int32>: UInt32] = [:]
        var wp: [SIMD3<Float>] = []
        func weld(_ q: SIMD3<Float>) -> UInt32 {
            let k = SIMD3<Int32>(Int32((q.x * 100).rounded()), Int32((q.y * 100).rounded()), Int32((q.z * 100).rounded()))
            if let i = index[k] { return i }
            wp.append(q); index[k] = UInt32(wp.count - 1); return index[k]!
        }
        var wf: [SIMD3<UInt32>] = []
        for tri in f {
            let w = SIMD3(weld(p[Int(tri.x)]), weld(p[Int(tri.y)]), weld(p[Int(tri.z)]))
            // Orient outward.
            let c = simd_cross(wp[Int(w.y)] - wp[Int(w.x)], wp[Int(w.z)] - wp[Int(w.x)])
            let centroid = (wp[Int(w.x)] + wp[Int(w.y)] + wp[Int(w.z)]) / 3
            wf.append(simd_dot(c, centroid) >= 0 ? w : SIMD3(w.x, w.z, w.y))
        }
        let t = MeshTopology(positions: wp, faces: wf)
        let field = CrossField.compute(topology: t)
        // Faces touching a crease must have an axis along it.
        var checked = 0
        for i in 0..<wf.count {
            for nb in t.neighbors[i] where simd_dot(t.faceNormal[i], t.faceNormal[nb.face]) < 0.5 {
                let e = simd_normalize(wp[Int(nb.b)] - wp[Int(nb.a)])
                XCTAssertLessThan(field.misalignment(of: e, face: i), 0.1)
                checked += 1
            }
        }
        XCTAssertGreaterThan(checked, 100)
    }

    func testSingularityIndicesSumToEulerCharacteristic() {
        let (sp, sf) = icosphere(subdivisions: 3)
        let sphere = CrossField.compute(topology: MeshTopology(positions: sp, faces: sf))
        let sphereSum = sphere.singularities().reduce(0) { $0 + $1.index }
        XCTAssertEqual(sphereSum, 2, accuracy: 0.6)
        XCTAssertTrue(sphere.singularities().allSatisfy { abs($0.index) <= 1.25 })

        let (tp, tf) = torus()
        let tor = CrossField.compute(topology: MeshTopology(positions: tp, faces: tf))
        XCTAssertEqual(tor.singularities().reduce(0) { $0 + $1.index }, 0, accuracy: 0.6)
    }
}
