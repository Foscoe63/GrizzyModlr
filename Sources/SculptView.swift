import SwiftUI
import SceneKit
import simd

/// Drives a sculpting session: the editable mesh plus brush settings. The viewer
/// reads it on every stroke; SwiftUI observes `revision` to refresh undo state.
@MainActor @Observable
final class SculptController {
    var mesh: SculptMesh
    var tool: SculptTool = .pull
    /// Brush radius as a fraction of the model's bounding radius.
    var radiusFraction: Double = 0.12
    /// 0…1 strength of each dab.
    var strength: Double = 0.4
    /// Bumped after every edit so views depending on `canUndo` / `isEdited` refresh.
    private(set) var revision = 0

    init(mesh: SculptMesh) { self.mesh = mesh }

    var canUndo: Bool { _ = revision; return mesh.canUndo }
    var isEdited: Bool { _ = revision; return mesh.isEdited }
    var brushRadius: Float { Float(radiusFraction) * mesh.boundsRadius }

    func touch() { revision += 1 }
    func undo() { mesh.undo(); touch() }
}

/// SceneKit view with a brush layer: when a `SculptController` is attached, plain
/// left-drag sculpts (ray-picked on the surface, with a ring cursor); holding ⌥
/// falls through to the normal orbit camera. Scroll/pinch zoom is unaffected.
final class SculptSCNView: SCNView {
    private(set) var controller: SculptController?
    private var sculptNode: SCNNode?
    private var element: SCNGeometryElement?
    private var material: SCNMaterial?
    private let cursorRoot = SCNNode()
    private var cursor: SCNNode?
    private var tracking: NSTrackingArea?
    private var stroking = false

    // MARK: attach / detach

    /// Replaces the container's content with the editable mesh.
    @MainActor
    func attach(_ controller: SculptController, container: SCNNode) {
        self.controller = controller
        container.childNodes.forEach { $0.removeFromParentNode() }
        let m = controller.mesh
        let el = SCNGeometryElement(data: Data(bytes: m.faces, count: m.faces.count * MemoryLayout<SIMD3<UInt32>>.stride),
                                    primitiveType: .triangles, primitiveCount: m.faces.count, bytesPerIndex: 4)
        element = el
        let mat = SCNMaterial()
        mat.lightingModel = .blinn
        mat.diffuse.contents = NSColor(white: 0.82, alpha: 1)
        mat.specular.contents = NSColor(white: 0.25, alpha: 1)
        mat.isDoubleSided = true
        material = mat
        let node = SCNNode()
        sculptNode = node
        rebuildGeometry()
        container.addChildNode(node)

        let torus = SCNTorus(ringRadius: 1, pipeRadius: 0.03)
        let cm = SCNMaterial()
        cm.diffuse.contents = NSColor.systemYellow
        cm.emission.contents = NSColor.systemYellow
        cm.lightingModel = .constant
        torus.materials = [cm]
        let c = SCNNode(geometry: torus)
        c.isHidden = true
        cursor = c
        cursorRoot.name = "BrushCursor"
        cursorRoot.addChildNode(c)
        scene?.rootNode.addChildNode(cursorRoot)
        updateTrackingAreas()
    }

    @MainActor
    func detach() {
        controller = nil
        sculptNode = nil; element = nil; material = nil
        cursorRoot.removeFromParentNode()
        cursorRoot.childNodes.forEach { $0.removeFromParentNode() }
        cursor = nil
        stroking = false
    }

    @MainActor
    private func rebuildGeometry() {
        guard let controller, let el = element, let node = sculptNode else { return }
        let m = controller.mesh
        let n = m.positions.count
        let stride = MemoryLayout<SIMD3<Float>>.stride          // 16: SIMD3<Float> is padded
        let vSource = SCNGeometrySource(data: Data(bytes: m.positions, count: n * stride),
                                        semantic: .vertex, vectorCount: n, usesFloatComponents: true,
                                        componentsPerVector: 3, bytesPerComponent: 4,
                                        dataOffset: 0, dataStride: stride)
        let nSource = SCNGeometrySource(data: Data(bytes: m.normals, count: n * stride),
                                        semantic: .normal, vectorCount: n, usesFloatComponents: true,
                                        componentsPerVector: 3, bytesPerComponent: 4,
                                        dataOffset: 0, dataStride: stride)
        let g = SCNGeometry(sources: [vSource, nSource], elements: [el])
        if let material { g.materials = [material] }
        node.geometry = g
    }

    // MARK: picking

    private func pick(_ event: NSEvent) -> (point: SIMD3<Float>, normal: SIMD3<Float>)? {
        guard let node = sculptNode else { return nil }
        let p = convert(event.locationInWindow, from: nil)
        let hits = hitTest(p, options: [.rootNode: node,
                                        .searchMode: SCNHitTestSearchMode.closest.rawValue,
                                        .ignoreHiddenNodes: true])
        guard let h = hits.first else { return nil }
        let lc = h.localCoordinates, ln = h.localNormal
        return (SIMD3(Float(lc.x), Float(lc.y), Float(lc.z)), SIMD3(Float(ln.x), Float(ln.y), Float(ln.z)))
    }

    private func sculpting(_ event: NSEvent) -> Bool {
        controller != nil && !event.modifierFlags.contains(.option)
    }

    private func moveCursor(_ hit: (point: SIMD3<Float>, normal: SIMD3<Float>)?) {
        guard let controller, let cursor, let torus = cursor.geometry as? SCNTorus else { return }
        guard let hit else { cursor.isHidden = true; return }
        let r = CGFloat(controller.brushRadius)
        torus.ringRadius = r
        torus.pipeRadius = max(r * 0.02, 1e-5)
        cursor.simdPosition = hit.point
        let n = simd_length(hit.normal) > 1e-9 ? simd_normalize(hit.normal) : SIMD3<Float>(0, 1, 0)
        cursor.simdOrientation = simd_quatf(from: SIMD3<Float>(0, 1, 0), to: n)
        cursor.isHidden = false
    }

    @MainActor
    private func dab(_ event: NSEvent) {
        guard let controller else { return }
        let hit = pick(event)
        moveCursor(hit)
        guard let hit else { return }
        controller.mesh.dab(tool: controller.tool, at: hit.point,
                            radius: controller.brushRadius, strength: Float(controller.strength))
        rebuildGeometry()
        controller.touch()
    }

    // MARK: events

    override func mouseDown(with event: NSEvent) {
        guard sculpting(event) else { cursor?.isHidden = true; super.mouseDown(with: event); return }
        MainActor.assumeIsolated {
            controller?.mesh.beginStroke()
            stroking = true
            dab(event)
        }
    }

    override func mouseDragged(with event: NSEvent) {
        guard stroking else { super.mouseDragged(with: event); return }
        MainActor.assumeIsolated { dab(event) }
    }

    override func mouseUp(with event: NSEvent) {
        guard stroking else { super.mouseUp(with: event); return }
        stroking = false
        MainActor.assumeIsolated {
            controller?.mesh.endStroke()
            controller?.touch()
        }
    }

    override func mouseMoved(with event: NSEvent) {
        super.mouseMoved(with: event)
        guard controller != nil else { return }
        moveCursor(sculpting(event) ? pick(event) : nil)
    }

    override func mouseExited(with event: NSEvent) {
        super.mouseExited(with: event)
        cursor?.isHidden = true
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let tracking { removeTrackingArea(tracking) }
        let area = NSTrackingArea(rect: bounds, options: [.mouseMoved, .mouseEnteredAndExited,
                                                          .activeInKeyWindow, .inVisibleRect],
                                  owner: self, userInfo: nil)
        addTrackingArea(area)
        tracking = area
    }
}
