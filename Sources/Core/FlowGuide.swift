import Foundation
import SceneKit
import simd

/// A curve drawn on the model that the flow must follow (around an eye, along a limb, down a seam).
/// Points are in mesh coordinates; the cross field is pinned to the curve's direction near it.
struct FlowGuide: Equatable {
    var points: [SIMD3<Float>]

    /// Straight pieces `(start, end)` of the curve, split so none is longer than `spacing`.
    func segments(maxLength spacing: Float) -> [(SIMD3<Float>, SIMD3<Float>)] {
        guard points.count >= 2, spacing > 0 else { return [] }
        var out: [(SIMD3<Float>, SIMD3<Float>)] = []
        for i in 0..<(points.count - 1) {
            let a = points[i], b = points[i + 1]
            let parts = max(1, Int((simd_distance(a, b) / spacing).rounded(.up)))
            for k in 0..<parts {
                out.append((a + (b - a) * (Float(k) / Float(parts)), a + (b - a) * (Float(k + 1) / Float(parts))))
            }
        }
        return out
    }

    /// Drawing of all guides as bright lines, for the viewer.
    static func node(for guides: [FlowGuide]) -> SCNNode {
        let node = SCNNode()
        node.name = "FlowGuides"
        var vertices: [SCNVector3] = [], indices: [Int32] = []
        for g in guides where g.points.count >= 2 {
            for i in 0..<(g.points.count - 1) {
                let n = Int32(vertices.count)
                for p in [g.points[i], g.points[i + 1]] { vertices.append(SCNVector3(p.x, p.y, p.z)) }
                indices += [n, n + 1]
            }
        }
        guard !vertices.isEmpty else { return node }
        let geometry = SCNGeometry(sources: [SCNGeometrySource(vertices: vertices)],
                                   elements: [SCNGeometryElement(indices: indices, primitiveType: .line)])
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.diffuse.contents = NSColor.systemYellow
        m.readsFromDepthBuffer = false
        geometry.materials = [m]
        let lines = SCNNode(geometry: geometry)
        lines.renderingOrder = 100
        node.addChildNode(lines)
        return node
    }
}
