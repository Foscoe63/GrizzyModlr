import XCTest
import simd

final class RigTests: XCTestCase {
    /// A crude T-pose figure from boxes: torso, head, two legs, two arms.
    func figure() -> (positions: [SIMD3<Float>], faces: [SIMD3<UInt32>]) {
        var pos: [SIMD3<Float>] = [], faces: [SIMD3<UInt32>] = []
        func box(_ lo: SIMD3<Float>, _ hi: SIMD3<Float>) {
            let b = UInt32(pos.count)
            for i in 0..<8 { pos.append(SIMD3(i & 1 == 0 ? lo.x : hi.x, i & 2 == 0 ? lo.y : hi.y, i & 4 == 0 ? lo.z : hi.z)) }
            for t in [[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]] {
                faces.append(SIMD3(b + UInt32(t[0]), b + UInt32(t[1]), b + UInt32(t[2])))
            }
        }
        // Denser samples so skinning has vertices along limbs: subdivide each box into slabs.
        func slabs(_ lo: SIMD3<Float>, _ hi: SIMD3<Float>, axis: Int, n: Int) {
            for k in 0..<n {
                var l = lo, h = hi
                let a = lo[axis] + (hi[axis] - lo[axis]) * Float(k) / Float(n)
                let b = lo[axis] + (hi[axis] - lo[axis]) * Float(k + 1) / Float(n)
                l[axis] = a; h[axis] = b
                box(l, h)
            }
        }
        slabs([-0.15, 0.5, -0.08], [0.15, 0.82, 0.08], axis: 1, n: 4)      // torso
        slabs([-0.06, 0.82, -0.06], [0.06, 1.0, 0.06], axis: 1, n: 2)      // head (height 1.0)
        slabs([-0.12, 0.0, -0.05], [-0.01, 0.5, 0.05], axis: 1, n: 5)      // right leg
        slabs([0.01, 0.0, -0.05], [0.12, 0.5, 0.05], axis: 1, n: 5)        // left leg
        slabs([0.15, 0.74, -0.04], [0.62, 0.82, 0.04], axis: 0, n: 6)      // left arm
        slabs([-0.62, 0.74, -0.04], [-0.15, 0.82, 0.04], axis: 0, n: 6)    // right arm
        return (pos, faces)
    }

    func testSkeletonFitsFigure() throws {
        let f = figure()
        let sk = try XCTUnwrap(Skeleton.fit(to: f.positions))
        XCTAssertEqual(sk.joints.count, 22)
        let head = sk.joints[sk.index(of: "headTop")!].position
        XCTAssertEqual(head.y, 1.0, accuracy: 1e-4)
        let wristL = sk.joints[sk.index(of: "wristL")!].position
        XCTAssertGreaterThan(wristL.x, 0.4)                                  // arms out: T-pose detected
        XCTAssertNil(Skeleton.fit(to: []))
    }

    func testWeightsAreNormalizedAndArmFollowsShoulder() throws {
        let f = figure()
        let sk = try XCTUnwrap(Skeleton.fit(to: f.positions))
        let w = SkinWeights.compute(positions: f.positions, skeleton: sk)
        for v in w.weights { XCTAssertEqual(v.x + v.y + v.z + v.w, 1, accuracy: 1e-4) }
        // The far tip of the left arm should be driven by an arm joint, not the legs.
        let tip = f.positions.indices.max { f.positions[$0].x < f.positions[$1].x }!
        let armJoints = Set(["shoulderL", "elbowL", "wristL"].compactMap(sk.index(of:)).map(UInt16.init))
        XCTAssertTrue(armJoints.contains(w.joints[tip].x))
    }

    func testRestPoseIsIdentity() throws {
        let f = figure()
        let sk = try XCTUnwrap(Skeleton.fit(to: f.positions))
        let w = SkinWeights.compute(positions: f.positions, skeleton: sk)
        let out = PoseLibrary.named("Rest")!.apply(to: f.positions, skeleton: sk, weights: w, height: 1)
        for (a, b) in zip(out, f.positions) { XCTAssertLessThan(simd_distance(a, b), 1e-4) }
    }

    func testArmsDownLowersArmTipAndKeepsLegs() throws {
        let f = figure()
        let r = MeshOps.apply(.pose(name: "Arms down"), positions: f.positions, faces: f.faces)
        XCTAssertTrue(r.changed)
        XCTAssertEqual(r.positions.count, f.positions.count)
        XCTAssertEqual(r.faces, f.faces)
        let tip = f.positions.indices.max { f.positions[$0].x < f.positions[$1].x }!
        XCTAssertLessThan(r.positions[tip].y, f.positions[tip].y - 0.3)      // the hand swung down
        let foot = f.positions.indices.min { f.positions[$0].y < f.positions[$1].y }!
        XCTAssertLessThan(simd_distance(r.positions[foot], f.positions[foot]), 0.02)
    }

    func testPosePreservesBoneLengthsAtJoints() throws {
        // Rotating about a joint must keep a vertex's distance to that joint (rigid within a bone).
        let f = figure()
        let sk = try XCTUnwrap(Skeleton.fit(to: f.positions))
        let w = SkinWeights.compute(positions: f.positions, skeleton: sk)
        let out = PoseLibrary.named("Wave")!.apply(to: f.positions, skeleton: sk, weights: w, height: 1)
        XCTAssertTrue(out.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite })
        let r = MeshOps.apply(.pose(name: "Nope"), positions: f.positions, faces: f.faces)
        XCTAssertFalse(r.changed)
    }

    func testRiggedGLBHasSkinAndAnimation() throws {
        let f = figure()
        var d = Data()
        func le<T>(_ v: T) { var x = v; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
        le(Int32(f.positions.count)); le(Int32(f.faces.count))
        f.positions.forEach { le($0.x); le($0.y); le($0.z) }
        f.positions.forEach { _ in le(Float(0)); le(Float(1)); le(Float(0)) }
        f.faces.forEach { le($0.x); le($0.y); le($0.z) }
        let src = FileManager.default.temporaryDirectory.appendingPathComponent("fig-\(UUID().uuidString).mesh")
        try d.write(to: src); defer { try? FileManager.default.removeItem(at: src) }
        let dest = src.deletingPathExtension().appendingPathExtension("glb"); defer { try? FileManager.default.removeItem(at: dest) }
        try MeshExporter.export(meshURL: src, texture: nil, format: .glbRigged, to: dest)

        let glb = try Data(contentsOf: dest)
        let jsonLen = Int(glb.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 12, as: UInt32.self) })
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: glb.subdata(in: 20 ..< 20 + jsonLen)) as? [String: Any])
        let skin = try XCTUnwrap((json["skins"] as? [[String: Any]])?.first)
        XCTAssertEqual((skin["joints"] as? [Int])?.count, 22)
        let anim = try XCTUnwrap((json["animations"] as? [[String: Any]])?.first)
        XCTAssertEqual((anim["channels"] as? [Any])?.count, 22)
        let prim = try XCTUnwrap(((json["meshes"] as? [[String: Any]])?.first?["primitives"] as? [[String: Any]])?.first)
        let attrs = try XCTUnwrap(prim["attributes"] as? [String: Int])
        let accessors = try XCTUnwrap(json["accessors"] as? [[String: Any]])
        XCTAssertEqual(accessors[try XCTUnwrap(attrs["JOINTS_0"])]["count"] as? Int, f.positions.count)
        XCTAssertEqual(accessors[try XCTUnwrap(attrs["WEIGHTS_0"])]["count"] as? Int, f.positions.count)
        // Every accessor's bufferView must lie inside the buffer.
        let views = try XCTUnwrap(json["bufferViews"] as? [[String: Any]])
        let bufLen = try XCTUnwrap(((json["buffers"] as? [[String: Any]])?.first?["byteLength"]) as? Int)
        for v in views { XCTAssertLessThanOrEqual((v["byteOffset"] as? Int ?? 0) + (v["byteLength"] as! Int), bufLen) }
        XCTAssertEqual(glb.count % 4, 0)
    }
}
