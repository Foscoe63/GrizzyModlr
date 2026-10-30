import Foundation
import simd

/// Per-project hand adjustments to the fitted rig.
struct RigConfig: Codable, Hashable {
    /// Joint name → [x, y, z] offset as a fraction of model height.
    var offsets: [String: [Float]] = [:]
    /// Joint name → multiplier on how far that joint's bones reach (default 1).
    var influence: [String: Float] = [:]
    /// Passes that average each vertex's weights with its neighbours' (softer joints).
    var smoothing = 0

    var isDefault: Bool { offsets.isEmpty && influence.isEmpty && smoothing == 0 }
    func offset(_ joint: String) -> SIMD3<Float> {
        guard let o = offsets[joint], o.count == 3 else { return .zero }
        return SIMD3(o[0], o[1], o[2])
    }
    mutating func setOffset(_ joint: String, _ v: SIMD3<Float>) {
        if simd_length(v) < 1e-5 { offsets[joint] = nil } else { offsets[joint] = [v.x, v.y, v.z] }
    }
    mutating func setInfluence(_ joint: String, _ k: Float) {
        if abs(k - 1) < 1e-3 { influence[joint] = nil } else { influence[joint] = k }
    }
}

/// Joint names written into exported rigs, so animation made for another rig can be
/// retargeted by name. Our skeleton has no clavicles or fingers; importers skip what's missing.
enum JointNaming: String, CaseIterable, Identifiable {
    case generic, mixamo, unity
    var id: String { rawValue }
    var label: String {
        switch self {
        case .generic: return "Generic (hips, shoulderL…)"
        case .mixamo: return "Mixamo (mixamorig:…)"
        case .unity: return "Unity humanoid"
        }
    }

    func name(_ joint: String) -> String {
        guard self != .generic else { return joint }
        let side = joint.hasSuffix("L") ? "Left" : (joint.hasSuffix("R") ? "Right" : "")
        let base = side.isEmpty ? joint : String(joint.dropLast())
        let map: [String: (mixamo: String, unity: String)] = [
            "hips": ("Hips", "Hips"), "spine": ("Spine", "Spine"), "chest": ("Spine1", "Chest"),
            "neck": ("Neck", "Neck"), "head": ("Head", "Head"), "headTop": ("HeadTop_End", "HeadTop_End"),
            "shoulder": ("Arm", "UpperArm"), "elbow": ("ForeArm", "LowerArm"), "wrist": ("Hand", "Hand"),
            "hand": ("Hand_End", "Hand_End"), "hip": ("UpLeg", "UpperLeg"), "knee": ("Leg", "LowerLeg"),
            "ankle": ("Foot", "Foot"), "toe": ("ToeBase", "Toes"),
        ]
        guard let m = map[base] else { return joint }
        let n = self == .mixamo ? m.mixamo : m.unity
        return (self == .mixamo ? "mixamorig:" : "") + side + n
    }
}

/// A humanoid skeleton fitted to a mesh, per-vertex skin weights, and poses.
///
/// This is a geometric heuristic, not a learned rigger: it assumes an upright (+Y) humanoid
/// with arms out to the sides (T-pose) or hanging (A-pose), facing ±Z, and places a standard
/// 22-joint skeleton from the bounding box and the model's width at shoulder height.
/// Weights come from distance to the bones, so arms touching the torso or legs together
/// will share weights across the gap. Anything that isn't a standing biped will fit badly.
struct Skeleton: Equatable {
    struct Joint: Equatable {
        var name: String
        var parent: Int          // -1 for the root
        var position: SIMD3<Float>
        /// Skin radius scale: vertices farther than this from the bone lose influence quickly.
        var radius: Float
    }
    var joints: [Joint]

    func index(of name: String) -> Int? { joints.firstIndex { $0.name == name } }
    /// First child, which defines the bone's direction.
    func primaryChild(of j: Int) -> Int? { joints.indices.first { joints[$0].parent == j } }
    func children(of j: Int) -> [Int] { joints.indices.filter { joints[$0].parent == j } }

    static let names = ["hips", "spine", "chest", "neck", "head", "headTop",
                        "shoulderL", "elbowL", "wristL", "handL",
                        "shoulderR", "elbowR", "wristR", "handR",
                        "hipL", "kneeL", "ankleL", "toeL",
                        "hipR", "kneeR", "ankleR", "toeR"]
    static let parents = [-1, 0, 1, 2, 3, 4,
                          2, 6, 7, 8,
                          2, 10, 11, 12,
                          0, 14, 15, 16,
                          0, 18, 19, 20]

    /// Fits the standard skeleton to `positions`. Returns nil for degenerate input.
    static func fit(to positions: [SIMD3<Float>], rig: RigConfig = RigConfig()) -> Skeleton? {
        guard positions.count > 8 else { return nil }
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for p in positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let h = hi.y - lo.y
        guard h > 1e-6, h.isFinite else { return nil }
        func y(_ f: Float) -> Float { lo.y + h * f }

        // Torso centerline from the vertices around mid-body height.
        let band = positions.filter { abs($0.y - y(0.58)) < h * 0.06 }
        let center = band.isEmpty ? (lo + hi) / 2
            : band.reduce(SIMD3<Float>.zero, +) / Float(band.count)
        let cx = center.x, cz = center.z

        // Arm reach: how far the model extends sideways at shoulder height.
        let shoulderBand = positions.filter { abs($0.y - y(0.80)) < h * 0.05 }
        let reach = shoulderBand.map { abs($0.x - cx) }.max() ?? h * 0.2
        let tPose = reach > h * 0.45
        // A standing figure with its arms down is tall and narrow; a bust, a prop or a creature
        // is not, and a humanoid skeleton squashed into it only produces nonsense poses.
        if !tPose, h / max(hi.x - lo.x, 1e-6) < 2.0 { return nil }
        let shoulderX = h * 0.11
        let wristX = tPose ? max(shoulderX * 2.5, reach * 0.9) : shoulderX + h * 0.04
        let elbowX = shoulderX + (wristX - shoulderX) * 0.5
        let shoulderY = y(0.80)
        let elbowY = tPose ? shoulderY : y(0.62)
        let wristY = tPose ? shoulderY : y(0.46)
        let handX = tPose ? wristX + h * 0.08 : wristX
        let handY = tPose ? shoulderY : y(0.40)
        let legX = h * 0.06

        var pos = [SIMD3<Float>](repeating: .zero, count: names.count)
        func set(_ name: String, _ p: SIMD3<Float>) { pos[names.firstIndex(of: name)!] = p }
        set("hips", SIMD3(cx, y(0.52), cz)); set("spine", SIMD3(cx, y(0.62), cz))
        set("chest", SIMD3(cx, y(0.74), cz)); set("neck", SIMD3(cx, y(0.84), cz))
        set("head", SIMD3(cx, y(0.88), cz)); set("headTop", SIMD3(cx, hi.y, cz))
        for (side, sign) in [("L", Float(1)), ("R", Float(-1))] {
            set("shoulder" + side, SIMD3(cx + sign * shoulderX, shoulderY, cz))
            set("elbow" + side, SIMD3(cx + sign * elbowX, elbowY, cz))
            set("wrist" + side, SIMD3(cx + sign * wristX, wristY, cz))
            set("hand" + side, SIMD3(cx + sign * handX, handY, cz))
            set("hip" + side, SIMD3(cx + sign * legX, y(0.50), cz))
            set("knee" + side, SIMD3(cx + sign * legX, y(0.27), cz))
            set("ankle" + side, SIMD3(cx + sign * legX, y(0.05), cz))
            set("toe" + side, SIMD3(cx + sign * legX, lo.y, cz + h * 0.07))
        }
        let radii: [String: Float] = ["hips": 0.13, "spine": 0.13, "chest": 0.14, "neck": 0.05, "head": 0.09]
        let joints = names.indices.map { i in
            Joint(name: names[i], parent: parents[i], position: pos[i],
                  radius: h * (radii[names[i]] ?? (names[i].hasPrefix("hip") || names[i].hasPrefix("knee") || names[i].hasPrefix("ankle") ? 0.07 : 0.05)))
        }
        var result = Skeleton(joints: joints)
        for i in result.joints.indices {
            let name = result.joints[i].name
            result.joints[i].position += rig.offset(name) * h
            result.joints[i].radius *= rig.influence[name] ?? 1
        }
        return result
    }
}

/// Up to four joint influences per vertex.
struct SkinWeights: Equatable {
    var joints: [SIMD4<UInt16>]
    var weights: [SIMD4<Float>]

    static func compute(positions: [SIMD3<Float>], skeleton: Skeleton,
                        faces: [SIMD3<UInt32>] = [], smoothing: Int = 0) -> SkinWeights {
        // Bone segments: joint j owns segments to each of its children.
        var segments: [(joint: Int, a: SIMD3<Float>, b: SIMD3<Float>, radius: Float)] = []
        for j in skeleton.joints.indices {
            for c in skeleton.children(of: j) {
                segments.append((j, skeleton.joints[j].position, skeleton.joints[c].position, skeleton.joints[j].radius))
            }
        }
        var js: [SIMD4<UInt16>] = [], ws: [SIMD4<Float>] = []
        js.reserveCapacity(positions.count); ws.reserveCapacity(positions.count)
        for p in positions {
            var best = [Int: Float]()          // joint → strongest weight among its segments
            for s in segments {
                let ab = s.b - s.a
                let t = max(0, min(1, simd_dot(p - s.a, ab) / max(simd_length_squared(ab), 1e-12)))
                let d = simd_distance(p, s.a + ab * t) / s.radius
                let w = 1 / (d * d + 0.05)
                let w4 = w * w
                if w4 > (best[s.joint] ?? 0) { best[s.joint] = w4 }
            }
            let top = best.sorted { $0.value > $1.value }.prefix(4)
            let sum = top.reduce(Float(0)) { $0 + $1.value }
            var jv = SIMD4<UInt16>(repeating: 0), wv = SIMD4<Float>(repeating: 0)
            for (k, e) in top.enumerated() { jv[k] = UInt16(e.key); wv[k] = e.value / max(sum, 1e-20) }
            js.append(jv); ws.append(wv)
        }
        var result = SkinWeights(joints: js, weights: ws)
        if smoothing > 0, !faces.isEmpty { result.smooth(faces: faces, passes: smoothing) }
        return result
    }

    /// Averages each vertex's weights with its neighbours' (keeps the strongest four).
    mutating func smooth(faces: [SIMD3<UInt32>], passes: Int) {
        var adjacency = [[Int]](repeating: [], count: joints.count)
        for f in faces {
            let v = [Int(f.x), Int(f.y), Int(f.z)]
            for k in 0..<3 { adjacency[v[k]].append(v[(k + 1) % 3]); adjacency[v[k]].append(v[(k + 2) % 3]) }
        }
        for _ in 0..<passes {
            var nj = joints, nw = weights
            for v in joints.indices {
                var acc = [UInt16: Float]()
                func add(_ i: Int, _ scale: Float) {
                    for k in 0..<4 where weights[i][k] > 0 { acc[joints[i][k], default: 0] += weights[i][k] * scale }
                }
                add(v, 2)
                for n in adjacency[v] { add(n, 1) }
                let top = acc.sorted { $0.value > $1.value }.prefix(4)
                let sum = top.reduce(Float(0)) { $0 + $1.value }
                var jv = SIMD4<UInt16>(repeating: 0), wv = SIMD4<Float>(repeating: 0)
                for (k, e) in top.enumerated() { jv[k] = e.key; wv[k] = e.value / max(sum, 1e-20) }
                nj[v] = jv; nw[v] = wv
            }
            joints = nj; weights = nw
        }
    }
}

/// A pose: for each named joint, the world direction its bone should point
/// (from the joint toward its first child). Joints not named follow their parent.
struct RigPose: Equatable {
    var name: String
    var directions: [String: SIMD3<Float>]
    /// Hip translation as a fraction of model height.
    var hipsOffset: SIMD3<Float> = .zero

    /// Global (world-space) rotation of every joint for this pose on `skeleton`.
    func rotations(for skeleton: Skeleton) -> [simd_quatf] {
        var g = [simd_quatf](repeating: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), count: skeleton.joints.count)
        for (i, joint) in skeleton.joints.enumerated() {   // parents precede children
            let parent = joint.parent >= 0 ? g[joint.parent] : simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
            g[i] = parent
            guard let target = directions[joint.name], simd_length(target) > 0,
                  let c = skeleton.primaryChild(of: i) else { continue }
            let bind = simd_normalize(skeleton.joints[c].position - joint.position)
            let current = parent.act(bind)
            g[i] = simd_quatf(from: current, to: simd_normalize(target)) * parent
        }
        return g
    }

    /// Linear-blend skinning of `positions` into this pose.
    func apply(to positions: [SIMD3<Float>], skeleton: Skeleton, weights: SkinWeights,
               height: Float) -> [SIMD3<Float>] {
        let g = rotations(for: skeleton)
        var pivot = [SIMD3<Float>](repeating: .zero, count: g.count)
        for (i, joint) in skeleton.joints.enumerated() {
            if joint.parent < 0 { pivot[i] = joint.position + hipsOffset * height } else {
                let offset = joint.position - skeleton.joints[joint.parent].position
                pivot[i] = pivot[joint.parent] + g[joint.parent].act(offset)
            }
        }
        return positions.indices.map { v in
            var out = SIMD3<Float>.zero
            let jv = weights.joints[v], wv = weights.weights[v]
            for k in 0..<4 where wv[k] > 0 {
                let j = Int(jv[k])
                out += wv[k] * (pivot[j] + g[j].act(positions[v] - skeleton.joints[j].position))
            }
            return out
        }
    }
}

enum PoseLibrary {
    private static func dir(_ x: Float, _ y: Float, _ z: Float) -> SIMD3<Float> { simd_normalize(SIMD3(x, y, z)) }

    /// Directions are in the model frame: +X = the model's left (the "L" joints), +Y up, +Z forward.
    static let all: [RigPose] = [
        RigPose(name: "Rest", directions: [:]),
        RigPose(name: "Arms down", directions: arms(L: dir(0.15, -1, 0), R: dir(-0.15, -1, 0), forearmL: dir(0.1, -1, 0.05), forearmR: dir(-0.1, -1, 0.05))),
        RigPose(name: "Arms up", directions: arms(L: dir(0.3, 1, 0), R: dir(-0.3, 1, 0), forearmL: dir(0.2, 1, 0), forearmR: dir(-0.2, 1, 0))),
        RigPose(name: "Wave", directions: arms(L: dir(0.1, -1, 0), R: dir(-1, 0.4, 0), forearmL: dir(0.1, -1, 0.05), forearmR: dir(-0.3, 1, 0))),
        RigPose(name: "Walk", directions: legs(swing: 0.45)),
        RigPose(name: "Stride", directions: legs(swing: 0.8)),
    ]

    static func named(_ name: String) -> RigPose? { all.first { $0.name == name } }

    private static func arms(L: SIMD3<Float>, R: SIMD3<Float>, forearmL: SIMD3<Float>, forearmR: SIMD3<Float>) -> [String: SIMD3<Float>] {
        ["shoulderL": L, "elbowL": forearmL, "shoulderR": R, "elbowR": forearmR]
    }

    /// Opposite-leg swing with arms counter-swinging; the knee trails the thigh.
    private static func legs(swing s: Float) -> [String: SIMD3<Float>] {
        [
            "hipL": dir(0.05, -1, s), "kneeL": dir(0.05, -1, s * 0.2),
            "hipR": dir(-0.05, -1, -s), "kneeR": dir(-0.05, -1, -s - 0.5),
            "shoulderL": dir(0.15, -1, -s * 0.6), "elbowL": dir(0.1, -1, -s * 0.6 + 0.3),
            "shoulderR": dir(-0.15, -1, s * 0.6), "elbowR": dir(-0.1, -1, s * 0.6 + 0.3),
        ]
    }
}
