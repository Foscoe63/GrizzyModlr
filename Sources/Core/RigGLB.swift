import Foundation
import simd

/// A short animation: pose keyframes at times (seconds).
struct PoseClip {
    var name: String
    var keys: [(time: Float, pose: RigPose)]
}

extension PoseLibrary {
    private static func mirrored(_ pose: RigPose) -> RigPose {
        var d: [String: SIMD3<Float>] = [:]
        for (k, v) in pose.directions {
            let swapped = k.hasSuffix("L") ? String(k.dropLast()) + "R" : (k.hasSuffix("R") ? String(k.dropLast()) + "L" : k)
            d[swapped] = SIMD3(-v.x, v.y, v.z)
        }
        return RigPose(name: pose.name + " (mirrored)", directions: d)
    }

    static var clips: [PoseClip] {
        let rest = named("Rest")!, walk = named("Walk")!, wave = named("Wave")!
        return [
            PoseClip(name: "Walk", keys: [(0, walk), (0.5, mirrored(walk)), (1, walk)]),
            PoseClip(name: "Wave", keys: [(0, rest), (0.4, wave), (0.8, wave), (1.2, rest)]),
            PoseClip(name: "Arms up", keys: [(0, rest), (0.8, named("Arms up")!), (1.6, rest)]),
        ]
    }
    static func clip(named name: String) -> PoseClip? { clips.first { $0.name == name } }
}

/// Adds a skeleton, skin weights and an animation clip to a static GLB.
enum RigGLB {
    enum RigError: Error { case notGLB, noSkeleton }

    /// `speed` scales clip time (2 = twice as fast).
    static func encode(_ mesh: MeshExporter.MeshData, clip: PoseClip, speed: Float = 1) throws -> Data {
        let positions = (0..<mesh.vertCount).map { SIMD3(mesh.verts[$0 * 3], mesh.verts[$0 * 3 + 1], mesh.verts[$0 * 3 + 2]) }
        guard let skeleton = Skeleton.fit(to: positions) else { throw RigError.noSkeleton }
        let weights = SkinWeights.compute(positions: positions, skeleton: skeleton)
        return try addRig(to: MeshExporter.encodeGLB(mesh), skeleton: skeleton, weights: weights, clip: clip, speed: speed)
    }

    static func addRig(to glb: Data, skeleton: Skeleton, weights: SkinWeights, clip: PoseClip, speed: Float) throws -> Data {
        // --- split the GLB
        func u32(_ o: Int) -> Int { Int(glb.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt32.self) }) }
        guard glb.count > 28, u32(0) == 0x4654_6C67 else { throw RigError.notGLB }
        let jsonLen = u32(12)
        let jsonData = glb.subdata(in: 20 ..< 20 + jsonLen)
        let binStart = 20 + jsonLen + 8
        guard binStart <= glb.count, var json = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else { throw RigError.notGLB }
        var bin = glb.subdata(in: binStart ..< min(glb.count, binStart + u32(20 + jsonLen)))
        var buffers = (json["buffers"] as? [[String: Any]]) ?? []
        var bufferViews = (json["bufferViews"] as? [[String: Any]]) ?? []
        var accessors = (json["accessors"] as? [[String: Any]]) ?? []
        var nodes = (json["nodes"] as? [[String: Any]]) ?? []
        var meshes = (json["meshes"] as? [[String: Any]]) ?? []

        func align() { while bin.count % 4 != 0 { bin.append(0) } }
        func append<T>(_ values: [T]) -> (view: Int, count: Int) {
            align()
            let offset = bin.count
            values.withUnsafeBytes { bin.append(contentsOf: $0) }
            bufferViews.append(["buffer": 0, "byteOffset": offset, "byteLength": values.count * MemoryLayout<T>.stride])
            return (bufferViews.count - 1, values.count)
        }
        func accessor(view: Int, component: Int, count: Int, type: String, extra: [String: Any] = [:]) -> Int {
            var a: [String: Any] = ["bufferView": view, "componentType": component, "count": count, "type": type]
            a.merge(extra) { $1 }
            accessors.append(a)
            return accessors.count - 1
        }

        // --- vertex attributes
        let n = skeleton.joints.count
        var jointsRaw: [UInt16] = [], weightsRaw: [Float] = []
        for v in 0..<weights.joints.count {
            for k in 0..<4 { jointsRaw.append(weights.joints[v][k]); weightsRaw.append(weights.weights[v][k]) }
        }
        let jView = append(jointsRaw), wView = append(weightsRaw)
        let jAcc = accessor(view: jView.view, component: 5123, count: weights.joints.count, type: "VEC4")
        let wAcc = accessor(view: wView.view, component: 5126, count: weights.joints.count, type: "VEC4")

        // --- joint nodes (local translations; bind rotation identity)
        let nodeBase = nodes.count
        for (i, j) in skeleton.joints.enumerated() {
            let local = j.parent < 0 ? j.position : j.position - skeleton.joints[j.parent].position
            var node: [String: Any] = ["name": j.name, "translation": [local.x, local.y, local.z]]
            let kids = skeleton.children(of: i).map { nodeBase + $0 }
            if !kids.isEmpty { node["children"] = kids }
            nodes.append(node)
        }
        // inverse bind matrices: translate(-bindPosition), column-major
        var ibm: [Float] = []
        for j in skeleton.joints {
            ibm += [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -j.position.x, -j.position.y, -j.position.z, 1]
        }
        let ibmView = append(ibm)
        let ibmAcc = accessor(view: ibmView.view, component: 5126, count: n, type: "MAT4")
        nodes[0]["skin"] = 0
        json["skins"] = [["joints": (0..<n).map { nodeBase + $0 }, "inverseBindMatrices": ibmAcc, "skeleton": nodeBase]]
        if var scenes = json["scenes"] as? [[String: Any]], !scenes.isEmpty {
            scenes[0]["nodes"] = ((scenes[0]["nodes"] as? [Int]) ?? [0]) + [nodeBase]
            json["scenes"] = scenes
        }
        if var prim = (meshes.first?["primitives"] as? [[String: Any]])?.first {
            var attrs = (prim["attributes"] as? [String: Any]) ?? [:]
            attrs["JOINTS_0"] = jAcc; attrs["WEIGHTS_0"] = wAcc
            prim["attributes"] = attrs
            meshes[0]["primitives"] = [prim]
            json["meshes"] = meshes
        }

        // --- animation: one rotation channel per joint
        let times = clip.keys.map { $0.time / max(speed, 0.01) }
        let tView = append(times)
        let tAcc = accessor(view: tView.view, component: 5126, count: times.count, type: "SCALAR",
                            extra: ["min": [times.min() ?? 0], "max": [times.max() ?? 0]])
        let globals = clip.keys.map { $0.pose.rotations(for: skeleton) }
        var samplers: [[String: Any]] = [], channels: [[String: Any]] = []
        for j in 0..<n {
            var prev: simd_quatf?
            var values: [Float] = []
            for g in globals {
                let parent = skeleton.joints[j].parent >= 0 ? g[skeleton.joints[j].parent] : simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
                var q = simd_normalize(parent.inverse * g[j])
                if let p = prev, simd_dot(p.vector, q.vector) < 0 { q = simd_quatf(vector: -q.vector) }  // shortest path
                prev = q
                values += [q.imag.x, q.imag.y, q.imag.z, q.real]
            }
            let view = append(values)
            let acc = accessor(view: view.view, component: 5126, count: globals.count, type: "VEC4")
            samplers.append(["input": tAcc, "output": acc, "interpolation": "LINEAR"])
            channels.append(["sampler": samplers.count - 1, "target": ["node": nodeBase + j, "path": "rotation"]])
        }
        json["animations"] = [["name": clip.name, "samplers": samplers, "channels": channels]]

        // --- repack
        align()
        buffers = [["byteLength": bin.count]]
        json["buffers"] = buffers; json["bufferViews"] = bufferViews; json["accessors"] = accessors; json["nodes"] = nodes
        var jsonOut = try JSONSerialization.data(withJSONObject: json)
        while jsonOut.count % 4 != 0 { jsonOut.append(0x20) }
        var out = Data()
        func le(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { out.append(contentsOf: $0) } }
        le(0x4654_6C67); le(2); le(UInt32(12 + 8 + jsonOut.count + 8 + bin.count))
        le(UInt32(jsonOut.count)); le(0x4E4F_534A); out.append(jsonOut)
        le(UInt32(bin.count)); le(0x004E_4942); out.append(bin)
        return out
    }
}
