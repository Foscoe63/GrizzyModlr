import Foundation
import SceneKit
import simd

/// The cross field drawn on the model: a short pair of crossing lines on (a sample of) the faces and
/// a dot at every singular vertex (red +¼, blue −¼, yellow otherwise). Pure data, so it can be built
/// off the main thread and handed to the viewer.
struct FlowOverlay: Equatable {
    let id = UUID()
    /// Line endpoints, two per segment.
    var segments: [SIMD3<Float>]
    var singularities: [(position: SIMD3<Float>, index: Float)]
    var radius: Float

    static func == (a: FlowOverlay, b: FlowOverlay) -> Bool { a.id == b.id }

    static func make(from field: CrossField, maxFaces: Int = 8000) -> FlowOverlay {
        let t = field.topology
        let step = max(1, t.faces.count / maxFaces)
        let length = t.averageEdgeLength * 0.42 * Float(step).squareRoot()
        var segments: [SIMD3<Float>] = []
        var i = 0
        while i < t.faces.count {
            let c = t.centroid(i) + t.faceNormal[i] * length * 0.15           // lift off the surface
            let (u, v) = field.axes(face: i)
            segments += [c - u * length, c + u * length, c - v * length, c + v * length]
            i += step
        }
        let singular = field.singularities().prefix(600).map { (position: t.positions[$0.vertex], index: $0.index) }
        return FlowOverlay(segments: segments, singularities: Array(singular), radius: t.averageEdgeLength * 0.5)
    }

    /// SceneKit node for the overlay (identity transform, matching the mesh's own coordinates).
    func node() -> SCNNode {
        let root = SCNNode()
        root.name = "FlowOverlay"
        if !segments.isEmpty {
            let data = segments.withUnsafeBufferPointer { Data(buffer: $0) }
            let source = SCNGeometrySource(data: data, semantic: .vertex, vectorCount: segments.count,
                                           usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4,
                                           dataOffset: 0, dataStride: MemoryLayout<SIMD3<Float>>.stride)
            let indices = (0..<UInt32(segments.count)).map { $0 }
            let element = SCNGeometryElement(data: indices.withUnsafeBufferPointer { Data(buffer: $0) },
                                             primitiveType: .line, primitiveCount: segments.count / 2, bytesPerIndex: 4)
            let geometry = SCNGeometry(sources: [source], elements: [element])
            let m = SCNMaterial()
            m.lightingModel = .constant
            m.diffuse.contents = NSColor(calibratedRed: 0.1, green: 0.85, blue: 0.95, alpha: 1)
            m.readsFromDepthBuffer = true
            geometry.materials = [m]
            root.addChildNode(SCNNode(geometry: geometry))
        }
        for s in singularities {
            let sphere = SCNSphere(radius: CGFloat(radius))
            let m = SCNMaterial()
            m.lightingModel = .constant
            m.diffuse.contents = s.index > 0 ? NSColor.systemRed : (s.index < 0 ? NSColor.systemBlue : NSColor.systemYellow)
            m.readsFromDepthBuffer = false
            sphere.materials = [m]
            let n = SCNNode(geometry: sphere)
            n.position = SCNVector3(s.position.x, s.position.y, s.position.z)
            n.renderingOrder = 10
            root.addChildNode(n)
        }
        return root
    }
}
