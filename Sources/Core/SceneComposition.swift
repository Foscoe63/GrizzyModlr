import Foundation
import simd

/// One object placed in a composed scene: a saved shape/paint version (possibly from another
/// project) with a transform. Objects are normalized to a unit bounding sphere centered at the
/// origin first, so `scale` and `position` mean the same thing for every model.
struct SceneItem: Identifiable, Codable, Hashable {
    var id = UUID()
    var projectID: UUID
    var generationID: UUID
    var x: Double = 0
    var y: Double = 0
    var z: Double = 0
    /// Degrees about the vertical axis.
    var rotationY: Double = 0
    var scale: Double = 1

    var position: SIMD3<Float> { SIMD3(Float(x), Float(y), Float(z)) }
}

enum SceneComposition {
    /// Matrix taking a mesh into scene space: recenter on its bounding-sphere center, scale to a
    /// unit radius, then apply the item's scale, Y rotation and position.
    static func matrix(for item: SceneItem, center: SIMD3<Float>, radius: Float) -> simd_float4x4 {
        let r = max(radius, 1e-6)
        let normalize = scaling(1 / r) * translation(-center)
        let angle = Float(item.rotationY) * .pi / 180
        let rotate = simd_float4x4(simd_quatf(angle: angle, axis: [0, 1, 0]))
        return translation(item.position) * rotate * scaling(Float(item.scale)) * normalize
    }

    static func bounds(_ positions: [SIMD3<Float>]) -> (center: SIMD3<Float>, radius: Float) {
        guard !positions.isEmpty else { return (.zero, 1) }
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for p in positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        return ((lo + hi) / 2, max(simd_length(hi - lo) / 2, 1e-6))
    }

    /// Merges transformed meshes into one (positions in scene space, faces re-indexed).
    static func merge(_ parts: [(positions: [SIMD3<Float>], faces: [SIMD3<UInt32>], item: SceneItem)])
    -> (positions: [SIMD3<Float>], faces: [SIMD3<UInt32>]) {
        var outP: [SIMD3<Float>] = [], outF: [SIMD3<UInt32>] = []
        for part in parts {
            let (c, r) = bounds(part.positions)
            let m = matrix(for: part.item, center: c, radius: r)
            let base = UInt32(outP.count)
            for p in part.positions {
                let v = m * SIMD4(p, 1)
                outP.append(SIMD3(v.x, v.y, v.z))
            }
            // A negative scale would flip winding; scales here are positive, but stay safe.
            outF.append(contentsOf: part.faces.map { SIMD3($0.x + base, $0.y + base, $0.z + base) })
        }
        return (outP, outF)
    }

    /// X position for the next object so it lands beside the existing ones.
    static func nextSlotX(existing: [SceneItem], spacing: Double) -> Double {
        guard let maxX = existing.map(\.x).max() else { return 0 }
        return maxX + spacing
    }

    private static func translation(_ t: SIMD3<Float>) -> simd_float4x4 {
        var m = matrix_identity_float4x4; m.columns.3 = SIMD4(t, 1); return m
    }
    private static func scaling(_ s: Float) -> simd_float4x4 {
        simd_float4x4(diagonal: SIMD4(s, s, s, 1))
    }
}
