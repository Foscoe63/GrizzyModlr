import Foundation
import simd

/// Finds the orthonormal frame a model's surface is most aligned to (the "cube frame" that its
/// face normals cluster around), so a grid-based remesh puts its quad rows along the model's
/// own flats and edges rather than along the world axes.
enum MeshFrame {
    /// Rotation whose columns are the frame axes (model space). Maximizes Σ area · Σₖ (n·rₖ)⁴ over
    /// the faces: that sum peaks when many normals point along the axes. Deterministic.
    static func bestFrame(positions: [SIMD3<Float>], faces: [SIMD3<UInt32>], maxSamples: Int = 6000) -> simd_float3x3 {
        var samples: [SIMD4<Float>] = []                      // xyz = unit normal, w = area
        let step = max(1, faces.count / maxSamples)
        var i = 0
        while i < faces.count {
            let f = faces[i]
            let c = simd_cross(positions[Int(f.y)] - positions[Int(f.x)], positions[Int(f.z)] - positions[Int(f.x)])
            let len = simd_length(c)
            if len > 1e-12 { samples.append(SIMD4(c / len, len / 2)) }
            i += step
        }
        guard !samples.isEmpty else { return matrix_identity_float3x3 }

        func score(_ r: simd_float3x3) -> Float {
            var total: Float = 0
            let c0 = r.columns.0, c1 = r.columns.1, c2 = r.columns.2
            for s in samples {
                let n = SIMD3(s.x, s.y, s.z)
                let a = simd_dot(n, c0), b = simd_dot(n, c1), c = simd_dot(n, c2)
                total += s.w * (a * a * a * a + b * b * b * b + c * c * c * c)
            }
            return total
        }
        func rotation(_ e: SIMD3<Float>) -> simd_float3x3 {
            simd_float3x3(simd_quatf(angle: e.z, axis: [0, 0, 1]) * simd_quatf(angle: e.y, axis: [0, 1, 0])
                          * simd_quatf(angle: e.x, axis: [1, 0, 0]))
        }

        // Coarse search over Euler angles (a cube frame repeats every quarter turn).
        var best = matrix_identity_float3x3, bestScore = score(best)
        let n = 6, span = Float.pi / 2
        for a in 0..<n { for b in 0..<n { for c in 0..<n {
            let r = rotation(SIMD3(Float(a), Float(b), Float(c)) * span / Float(n))
            let s = score(r)
            if s > bestScore + 1e-6 { bestScore = s; best = r }
        } } }
        // Refine by trying small rotations about the three frame axes, shrinking the step.
        var delta: Float = span / Float(n) / 2
        while delta > 1e-3 {
            var improved = false
            for axis in 0..<3 { for sign: Float in [-1, 1] {
                var v = SIMD3<Float>.zero; v[axis] = 1
                let r = simd_float3x3(simd_quatf(angle: sign * delta, axis: v)) * best
                let s = score(r)
                if s > bestScore + 1e-6 { bestScore = s; best = r; improved = true }
            } }
            if !improved { delta /= 2 }
        }
        return best
    }
}
