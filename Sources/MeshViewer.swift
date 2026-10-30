import SwiftUI
import SceneKit
import ImageIO

/// What the viewer is showing right now.
enum ViewerContent: Equatable {
    case mesh(URL)                    // binary .mesh (preview or final shape)
    case texturedMesh(URL, URL)       // .tmesh (with UVs) + texture image
    case pbrMesh(URL, albedo: URL, metallicRoughness: URL)  // .tmesh + albedo + MR (G=rough, B=metal)
    case points(URL)                  // a streaming near-surface point cloud (.bin float32 xyz)
}

/// Renders a mesh or point cloud in an orbitable SceneKit view. The scene, lights
/// and camera persist; only the content node is swapped in place, and the camera
/// is re-fit to the content so even a large/messy early preview fills the viewport.
struct MeshViewer: NSViewRepresentable {
    let content: ViewerContent?
    var onLoading: ((Bool) -> Void)? = nil    // true while async-loading content
    /// When set, the viewer shows this editable mesh and left-drag sculpts it.
    var sculpt: SculptController? = nil
    /// When set, left-drag paints the albedo texture of the (textured) content.
    var paint: TexturePaintController? = nil
    /// One-shot camera / snapshot request from the viewport menu.
    var command: ViewerCommand? = nil
    /// Optional cross-field drawing shown over the mesh.
    var overlay: FlowOverlay? = nil
    /// Flow guide curves drawn on the model; with a recorder set, left-drag draws a new one.
    var guides: [FlowGuide] = []
    var guideRecorder: GuideRecorder? = nil

    @AppStorage(ViewerDefaults.Key.environment) private var envName = ViewerDefaults.fallbackEnvironment
    @AppStorage(ViewerDefaults.Key.environmentIntensity) private var envIntensity = ViewerDefaults.fallbackIntensity
    @AppStorage(ViewerDefaults.Key.environmentBackground) private var envBackground = false
    @AppStorage(ViewerDefaults.Key.showGround) private var showGround = false
    @AppStorage(ViewerDefaults.Key.shadows) private var shadows = false
    @AppStorage(ViewerDefaults.Key.autoRotate) private var autoRotate = false
    @AppStorage(ViewerDefaults.Key.customEnvironmentPath) private var customEnvPath = ""

    private var stageOptions: ViewerDefaults.Options {
        ViewerDefaults.Options(environment: ViewerEnvironment(rawValue: envName) ?? .none, intensity: envIntensity,
                               environmentBackground: envBackground, showGround: showGround, shadows: shadows,
                               autoRotate: autoRotate, customPath: customEnvPath)
    }

    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator {
        var loaded: ViewerContent?
        var lastFitCenter: SCNVector3?
        var lastFitRadius: CGFloat?
        var lastCommand: UUID?
        var lastOverlay: UUID?
        var lastGuides: [FlowGuide] = []
    }

    func makeNSView(context: Context) -> SculptSCNView {
        let view = SculptSCNView()
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true
        view.backgroundColor = .clear
        view.antialiasingMode = .multisampling4X

        let scene = SCNScene()

        let container = SCNNode()
        container.name = "MeshContainer"
        scene.rootNode.addChildNode(container)

        let ambient = SCNNode()
        ambient.name = "BaseAmbient"
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 280
        scene.rootNode.addChildNode(ambient)

        let camera = SCNCamera()
        camera.fieldOfView = 40
        camera.projectionDirection = .horizontal
        camera.zNear = 0.001
        camera.zFar = 1000
        let cameraNode = SCNNode()
        cameraNode.name = "GrizzyModlrCamera"
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(0, 0, 3.3)
        cameraNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(cameraNode)

        view.scene = scene
        view.pointOfView = cameraNode
        return view
    }

    func updateNSView(_ view: SculptSCNView, context: Context) {
        ViewerStage.apply(view, options: stageOptions, allowSpin: sculpt == nil && paint == nil && guideRecorder == nil)
        view.guideRecorder = guideRecorder
        if context.coordinator.lastGuides != guides {
            context.coordinator.lastGuides = guides
            view.scene?.rootNode.childNode(withName: "FlowGuides", recursively: false)?.removeFromParentNode()
            if !guides.isEmpty {
                let node = FlowGuide.node(for: guides)
                if let container = container(in: view) { node.transform = container.transform }
                view.scene?.rootNode.addChildNode(node)
            }
        }
        if context.coordinator.lastOverlay != overlay?.id {
            context.coordinator.lastOverlay = overlay?.id
            view.scene?.rootNode.childNode(withName: "FlowOverlay", recursively: false)?.removeFromParentNode()
            if let overlay { view.scene?.rootNode.addChildNode(overlay.node()) }
        }
        if let command, context.coordinator.lastCommand != command.id {
            context.coordinator.lastCommand = command.id
            run(command, in: view)
        }
        if let sculpt {
            if view.controller !== sculpt, let container = container(in: view) {
                view.attach(sculpt, container: container)
                context.coordinator.loaded = nil        // reload the real content when sculpting ends
                let c = sculpt.mesh.center
                Self.fitCamera(view, center: SCNVector3(c.x, c.y, c.z), radius: CGFloat(sculpt.mesh.boundsRadius))
                context.coordinator.lastFitCenter = nil
                context.coordinator.lastFitRadius = nil
            }
            return
        }
        if view.controller != nil { view.detach() }
        if let paint {
            if view.paintController !== paint, let container = container(in: view) {
                view.attachPaint(paint, container: container)
            }
        } else if view.paintController != nil {
            view.detachPaint()
            context.coordinator.loaded = nil        // discard unsaved strokes: reload the saved texture
        }
        if context.coordinator.loaded == content { return }
        context.coordinator.loaded = content

        guard let content else {
            container(in: view)?.childNodes.forEach { $0.removeFromParentNode() }
            context.coordinator.lastFitCenter = nil
            context.coordinator.lastFitRadius = nil
            if let onLoading { DispatchQueue.main.async { onLoading(false) } }
            return
        }

        if let onLoading { DispatchQueue.main.async { onLoading(true) } }
        DispatchQueue.global(qos: .userInitiated).async {
            let result = Self.load(content)
            DispatchQueue.main.async {
                // Ignore a load whose content was already superseded — both applying it
                // and clearing the loading flag would be wrong for the current content.
                guard context.coordinator.loaded == content else { return }
                if let container = container(in: view), let result {
                    container.childNodes.forEach { $0.removeFromParentNode() }
                    container.addChildNode(result.node)
                    ViewerStage.apply(view, options: stageOptions, allowSpin: sculpt == nil && paint == nil)
                    // Re-fit only when the content's bounds change meaningfully (>~3%),
                    // so streaming frames don't yank the camera while the user orbits.
                    if Self.boundsChanged(center: result.center, radius: result.radius,
                                          lastCenter: context.coordinator.lastFitCenter,
                                          lastRadius: context.coordinator.lastFitRadius) {
                        Self.fitCamera(view, center: result.center, radius: result.radius)
                        context.coordinator.lastFitCenter = result.center
                        context.coordinator.lastFitRadius = result.radius
                    }
                }
                onLoading?(false)
            }
        }
    }

    private static func boundsChanged(center: SCNVector3, radius: CGFloat,
                                      lastCenter: SCNVector3?, lastRadius: CGFloat?) -> Bool {
        guard let lc = lastCenter, let lr = lastRadius, lr > 0 else { return true }
        if abs(radius - lr) / lr > 0.03 { return true }
        let dx = center.x - lc.x
        let dy = center.y - lc.y
        let dz = center.z - lc.z
        return (dx * dx + dy * dy + dz * dz).squareRoot() / lr > 0.03
    }

    private func container(in view: SCNView) -> SCNNode? {
        view.scene?.rootNode.childNode(withName: "MeshContainer", recursively: false)
    }

    // MARK: - loading

    static func load(_ content: ViewerContent) -> (node: SCNNode, center: SCNVector3, radius: CGFloat)? {
        switch content {
        case .mesh(let url):                      return loadMesh(url)
        case .texturedMesh(let mesh, let tex):    return loadTexturedMesh(mesh, texture: tex)
        case .pbrMesh(let mesh, let albedo, let mr):
            return loadTexturedMesh(mesh, texture: albedo, metallicRoughness: mr)
        case .points(let url):                    return loadPoints(url)
        }
    }

    /// Loads the textured .tmesh (verts f32, normals f32, UVs f32, faces i32) and
    /// applies the baked texture. With `metallicRoughness` set (PBR/Large), switches
    /// to physically-based lighting: albedo → diffuse, and the MR map is split into
    /// its glTF channels — G → roughness, B → metalness — as grayscale images.
    private static func loadTexturedMesh(_ url: URL, texture: URL,
                                         metallicRoughness: URL? = nil) -> (SCNNode, SCNVector3, CGFloat)? {
        guard let data = try? Data(contentsOf: url), data.count >= 8 else { return nil }
        let n = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: Int32.self) })
        let m = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: Int32.self) })
        guard n > 0, m > 0 else { return nil }
        let vBytes = n * 12, nBytes = n * 12, uvBytes = n * 8, fBytes = m * 12
        guard data.count >= 8 + vBytes + nBytes + uvBytes + fBytes else { return nil }

        var off = 8
        let vData = data.subdata(in: off ..< off + vBytes); off += vBytes
        let nData = data.subdata(in: off ..< off + nBytes); off += nBytes
        let uvData = data.subdata(in: off ..< off + uvBytes); off += uvBytes
        let fData = data.subdata(in: off ..< off + fBytes)

        let vSource = SCNGeometrySource(data: vData, semantic: .vertex, vectorCount: n,
                                        usesFloatComponents: true, componentsPerVector: 3,
                                        bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
        let nSource = SCNGeometrySource(data: nData, semantic: .normal, vectorCount: n,
                                        usesFloatComponents: true, componentsPerVector: 3,
                                        bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
        let uvSource = SCNGeometrySource(data: uvData, semantic: .texcoord, vectorCount: n,
                                         usesFloatComponents: true, componentsPerVector: 2,
                                         bytesPerComponent: 4, dataOffset: 0, dataStride: 8)
        let element = SCNGeometryElement(data: fData, primitiveType: .triangles,
                                         primitiveCount: m, bytesPerIndex: 4)
        let geometry = SCNGeometry(sources: [vSource, nSource, uvSource], elements: [element])

        let material = SCNMaterial()
        material.diffuse.contents = NSImage(contentsOf: texture) ?? NSColor(white: 0.82, alpha: 1)
        material.diffuse.wrapS = .repeat
        material.diffuse.wrapT = .repeat
        material.isDoubleSided = true
        if let mrURL = metallicRoughness, let mr = loadCGImage(mrURL) {
            // glTF metallic-roughness packing: G = roughness, B = metallic. SceneKit's
            // physicallyBased model wants separate single-channel maps, so pull each out.
            material.lightingModel = .physicallyBased
            if let rough = channelImage(mr, channel: 1) { material.roughness.contents = rough }
            if let metal = channelImage(mr, channel: 2) { material.metalness.contents = metal }
            for prop in [material.roughness, material.metalness] {
                prop.wrapS = .repeat; prop.wrapT = .repeat
            }
        } else {
            material.lightingModel = .blinn
        }
        geometry.materials = [material]

        let node = SCNNode(geometry: geometry)
        let (lo, hi) = node.boundingBox
        let center = SCNVector3((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)
        let extent = max(hi.x - lo.x, max(hi.y - lo.y, hi.z - lo.z))
        return (node, center, max(extent / 2, 0.05))
    }

    /// Loads the worker's compact binary mesh (int32 nVerts, int32 nFaces, then
    /// verts f32, normals f32, faces i32) straight into SceneKit geometry — no text
    /// parsing or normal computation, so even a huge mesh loads in milliseconds.
    private static func loadMesh(_ url: URL) -> (SCNNode, SCNVector3, CGFloat)? {
        guard let data = try? Data(contentsOf: url), data.count >= 8 else { return nil }
        let n = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: Int32.self) })
        let m = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: Int32.self) })
        guard n > 0, m > 0 else { return nil }
        let vBytes = n * 12, nBytes = n * 12, fBytes = m * 12
        guard data.count >= 8 + vBytes + nBytes + fBytes else { return nil }

        let vData = data.subdata(in: 8 ..< 8 + vBytes)
        let nData = data.subdata(in: 8 + vBytes ..< 8 + vBytes + nBytes)
        let fData = data.subdata(in: 8 + vBytes + nBytes ..< 8 + vBytes + nBytes + fBytes)

        let vSource = SCNGeometrySource(data: vData, semantic: .vertex,
                                        vectorCount: n, usesFloatComponents: true,
                                        componentsPerVector: 3, bytesPerComponent: 4,
                                        dataOffset: 0, dataStride: 12)
        let nSource = SCNGeometrySource(data: nData, semantic: .normal,
                                        vectorCount: n, usesFloatComponents: true,
                                        componentsPerVector: 3, bytesPerComponent: 4,
                                        dataOffset: 0, dataStride: 12)
        let element = SCNGeometryElement(data: fData, primitiveType: .triangles,
                                         primitiveCount: m, bytesPerIndex: 4)
        let geometry = SCNGeometry(sources: [vSource, nSource], elements: [element])

        let material = SCNMaterial()
        material.lightingModel = .blinn
        material.diffuse.contents = NSColor(white: 0.82, alpha: 1)
        material.specular.contents = NSColor(white: 0.25, alpha: 1)
        material.isDoubleSided = true
        geometry.materials = [material]

        let node = SCNNode(geometry: geometry)
        let (lo, hi) = node.boundingBox
        let center = SCNVector3((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)
        let extent = max(hi.x - lo.x, max(hi.y - lo.y, hi.z - lo.z))
        return (node, center, max(extent / 2, 0.05))
    }

    /// Decode a PNG to a CGImage for channel extraction.
    private static func loadCGImage(_ url: URL) -> CGImage? {
        guard let data = try? Data(contentsOf: url),
              let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }

    /// Extract one 8-bit channel (0=R, 1=G, 2=B) of an RGBA image into a grayscale
    /// image — SceneKit reads a single-channel map's luminance for roughness/metalness.
    private static func channelImage(_ image: CGImage, channel: Int) -> NSImage? {
        let w = image.width, h = image.height
        guard w > 0, h > 0 else { return nil }
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        var gray = [UInt8](repeating: 0, count: w * h)
        for i in 0..<(w * h) { gray[i] = rgba[i * 4 + channel] }
        guard let grayCtx = CGContext(data: &gray, width: w, height: h, bitsPerComponent: 8,
                                      bytesPerRow: w,
                                      space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue),
              let out = grayCtx.makeImage() else { return nil }
        return NSImage(cgImage: out, size: NSSize(width: w, height: h))
    }

    private static func loadPoints(_ url: URL) -> (SCNNode, SCNVector3, CGFloat)? {
        guard let data = try? Data(contentsOf: url), data.count >= 12 else { return nil }
        let count = data.count / 12     // 3 × float32 per point
        var verts = [SCNVector3]()
        verts.reserveCapacity(count)
        let big = CGFloat.greatestFiniteMagnitude
        var lo = SCNVector3(big, big, big)
        var hi = SCNVector3(-big, -big, -big)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            for i in 0..<count {
                let x = CGFloat(raw.loadUnaligned(fromByteOffset: i * 12, as: Float32.self))
                let y = CGFloat(raw.loadUnaligned(fromByteOffset: i * 12 + 4, as: Float32.self))
                let z = CGFloat(raw.loadUnaligned(fromByteOffset: i * 12 + 8, as: Float32.self))
                verts.append(SCNVector3(x, y, z))
                if x < lo.x { lo.x = x }; if y < lo.y { lo.y = y }; if z < lo.z { lo.z = z }
                if x > hi.x { hi.x = x }; if y > hi.y { hi.y = y }; if z > hi.z { hi.z = z }
            }
        }
        guard !verts.isEmpty else { return nil }

        let source = SCNGeometrySource(vertices: verts)
        let element = SCNGeometryElement(indices: Array(0..<count), primitiveType: .point)
        element.pointSize = 4
        element.minimumPointScreenSpaceRadius = 1.0
        element.maximumPointScreenSpaceRadius = 3
        let geometry = SCNGeometry(sources: [source], elements: [element])
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = NSColor.controlAccentColor
        geometry.materials = [material]

        let cx: CGFloat = (lo.x + hi.x) / 2
        let cy: CGFloat = (lo.y + hi.y) / 2
        let cz: CGFloat = (lo.z + hi.z) / 2
        let extent: CGFloat = max(hi.x - lo.x, max(hi.y - lo.y, hi.z - lo.z))
        return (SCNNode(geometry: geometry), SCNVector3(cx, cy, cz), max(extent / 2, 0.05))
    }

    // MARK: - framing

    private func run(_ command: ViewerCommand, in view: SCNView) {
        guard let container = container(in: view) else { return }
        switch command.kind {
        case .camera(let preset):
            let (c, r) = ViewerStage.bounds(of: container)
            container.removeAction(forKey: "spin")
            container.eulerAngles = SCNVector3(0, 0, 0)               // undo any turntable rotation
            Self.fitCamera(view, center: c, radius: r, direction: preset.direction, up: preset.up)
        case .snapshot:
            let image = view.snapshot()
            DispatchQueue.main.async {
                let panel = NSSavePanel()
                panel.allowedContentTypes = [.png]
                panel.nameFieldStringValue = "view.png"
                guard panel.runModal() == .OK, let url = panel.url,
                      let tiff = image.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
                      let png = rep.representation(using: .png, properties: [:]) else { return }
                try? png.write(to: url)
            }
        }
    }

    private static func fitCamera(_ view: SCNView, center: SCNVector3, radius: CGFloat,
                                  direction: SIMD3<Float> = [0, 0, 1], up: SIMD3<Float> = [0, 1, 0]) {
        guard let cameraNode = view.pointOfView, let camera = cameraNode.camera else { return }
        // Fit the object to the *limiting* field of view so it never overflows a
        // non-square pane; generous margin so it sits comfortably inside.
        let size = view.bounds.size
        let aspect = (size.width > 1 && size.height > 1) ? Double(size.width / size.height) : 1.0
        let hFov = camera.fieldOfView * .pi / 180          // horizontal (projectionDirection = .horizontal)
        let vFov = 2 * atan(tan(hFov / 2) / max(aspect, 0.001))
        let limitFov = min(hFov, vFov)
        let distance = Double(radius) / tan(limitFov / 2) * 1.5
        cameraNode.position = SCNVector3(center.x + CGFloat(direction.x) * CGFloat(distance),
                                         center.y + CGFloat(direction.y) * CGFloat(distance),
                                         center.z + CGFloat(direction.z) * CGFloat(distance))
        cameraNode.look(at: center, up: SCNVector3(CGFloat(up.x), CGFloat(up.y), CGFloat(up.z)),
                        localFront: SCNVector3(0, 0, -1))
        camera.zNear = max(distance * 0.01, 0.001)
        camera.zFar = distance * 10 + 100
    }
}
