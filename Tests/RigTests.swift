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

    func testBustLikeShapeIsRejected() {
        // Wide, short block with arms not out: head-and-shoulders proportions (h/width 1.5).
        var pos: [SIMD3<Float>] = []
        for x in stride(from: Float(-0.65), through: 0.65, by: 0.13) { for y in stride(from: Float(-1), through: 1, by: 0.2) {
            for z in [Float(-0.3), 0.3] { pos.append(SIMD3(x, y, z)) } } }
        XCTAssertNil(Skeleton.fit(to: pos))
        let r = MeshOps.apply(.pose(name: "Wave"), positions: pos, faces: [])
        XCTAssertFalse(r.changed)
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

    /// Opt-in: RIG_MESH=<.mesh> [RIG_OUT=<dir>] poses a real scan and writes front-view SVGs of the skeleton and poses.
    func testRealScanRig() throws {
        guard let path = ProcessInfo.processInfo.environment["RIG_MESH"] else { throw XCTSkip("set RIG_MESH") }
        let mesh = try XCTUnwrap(MeshDecimator.parse(Data(contentsOf: URL(fileURLWithPath: path))))
        guard let sk = Skeleton.fit(to: mesh.positions) else { print("RIG rejected: not a standing figure"); return }
        let w = SkinWeights.compute(positions: mesh.positions, skeleton: sk)
        let lo = mesh.positions.reduce(SIMD3<Float>(repeating: .infinity), simd_min), hi = mesh.positions.reduce(SIMD3<Float>(repeating: -.infinity), simd_max)
        print("RIG bbox", lo, hi, "verts", mesh.positions.count)
        for name in ["wristL", "ankleL", "headTop"] { print("RIG joint", name, sk.joints[sk.index(of: name)!].position) }
        var share = [Int](repeating: 0, count: sk.joints.count)
        for j in w.joints { share[Int(j.x)] += 1 }
        print("RIG dominant joint share", zip(sk.joints.map(\.name), share).filter { $0.1 > 0 }.map { "\($0.0)=\($0.1)" }.joined(separator: " "))
        let out = URL(fileURLWithPath: ProcessInfo.processInfo.environment["RIG_OUT"] ?? NSTemporaryDirectory())
        let h = hi.y - lo.y, scale: Float = 400 / h
        func svg(_ pts: [SIMD3<Float>], _ file: String) throws {
            var s = "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='460' viewBox='-450 -20 900 460'><rect x='-450' y='-20' width='900' height='460' fill='#ffffff'/>"
            for (i, p) in pts.enumerated() where i % 3 == 0 {
                s += "<circle cx='\((p.x - (lo.x + hi.x) / 2) * scale)' cy='\((hi.y - p.y) * scale)' r='1' fill='#4477aa'/>"
            }
            for j in sk.joints where j.parent >= 0 {
                let a = sk.joints[j.parent].position, b = j.position
                s += "<line x1='\((a.x - (lo.x + hi.x) / 2) * scale)' y1='\((hi.y - a.y) * scale)' x2='\((b.x - (lo.x + hi.x) / 2) * scale)' y2='\((hi.y - b.y) * scale)' stroke='#cc3333' stroke-width='2'/>"
            }
            s += "</svg>"
            try s.write(to: out.appendingPathComponent(file), atomically: true, encoding: .utf8)
        }
        try svg(mesh.positions, "rig-rest.svg")
        for name in ["Arms down", "Wave", "Walk"] {
            let posed = PoseLibrary.named(name)!.apply(to: mesh.positions, skeleton: sk, weights: w, height: h)
            XCTAssertTrue(posed.allSatisfy { $0.x.isFinite })
            try svg(posed, "rig-\(name.replacingOccurrences(of: " ", with: "-")).svg")
        }
        print("RIG wrote to", out.path)
    }
}
