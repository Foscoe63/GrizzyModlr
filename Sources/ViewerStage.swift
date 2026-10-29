import SceneKit
import AppKit

/// Procedural equirectangular environments, so image-based lighting works with no downloads.
/// A custom HDR/EXR/PNG chosen in Settings → Viewer replaces them.
enum EnvironmentImages {
    nonisolated(unsafe) private static var cache: [String: NSImage] = [:]

    static func image(for env: ViewerEnvironment, customPath: String) -> NSImage? {
        switch env {
        case .none: return nil
        case .custom:
            guard !customPath.isEmpty else { return nil }
            if let hit = cache["custom:" + customPath] { return hit }
            guard let img = NSImage(contentsOf: URL(fileURLWithPath: customPath)) else { return nil }
            cache["custom:" + customPath] = img
            return img
        default:
            if let hit = cache[env.rawValue] { return hit }
            let img = render(env)
            if let img { cache[env.rawValue] = img }
            return img
        }
    }

    private static func render(_ env: ViewerEnvironment) -> NSImage? {
        let w = 1024, h = 512
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        let rect = CGRect(x: 0, y: 0, width: w, height: h)
        func gradient(_ stops: [(CGFloat, CGColor)]) {
            let g = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: stops.map(\.1) as CFArray,
                               locations: stops.map(\.0))!
            ctx.drawLinearGradient(g, start: CGPoint(x: 0, y: CGFloat(h)), end: .zero, options: [])   // top → bottom
        }
        func color(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat = 1) -> CGColor { CGColor(red: r, green: g, blue: b, alpha: a) }
        func glow(_ x: CGFloat, _ y: CGFloat, _ radius: CGFloat, _ c: CGColor) {
            let g = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(),
                               colors: [c, c.copy(alpha: 0)!] as CFArray, locations: [0, 1])!
            ctx.drawRadialGradient(g, startCenter: CGPoint(x: x, y: y), startRadius: 0,
                                   endCenter: CGPoint(x: x, y: y), endRadius: radius, options: [])
        }
        switch env {
        case .studio:
            gradient([(0, color(0.30, 0.31, 0.33)), (0.5, color(0.16, 0.16, 0.17)), (1, color(0.07, 0.07, 0.08))])
            ctx.setFillColor(color(1, 1, 1))                                                   // key softbox
            ctx.fill(CGRect(x: 120, y: 250, width: 220, height: 130))
            ctx.setFillColor(color(0.85, 0.9, 1))                                              // fill softbox
            ctx.fill(CGRect(x: 640, y: 230, width: 160, height: 110))
            ctx.setFillColor(color(1, 0.95, 0.9))                                              // rim strip
            ctx.fill(CGRect(x: 440, y: 330, width: 40, height: 150))
        case .outdoor:
            gradient([(0, color(0.22, 0.46, 0.88)), (0.48, color(0.78, 0.88, 1.0)), (0.5, color(0.45, 0.43, 0.40)), (1, color(0.20, 0.19, 0.18))])
            glow(720, 400, 90, color(1, 0.97, 0.88))
        case .sunset:
            gradient([(0, color(0.25, 0.22, 0.45)), (0.35, color(0.85, 0.45, 0.35)), (0.5, color(1.0, 0.7, 0.4)), (0.52, color(0.22, 0.14, 0.14)), (1, color(0.08, 0.06, 0.07))])
            glow(256, 262, 110, color(1, 0.85, 0.55))
        default:
            ctx.setFillColor(color(0.5, 0.5, 0.5)); ctx.fill(rect)
        }
        guard let cg = ctx.makeImage() else { return nil }
        return NSImage(cgImage: cg, size: NSSize(width: w, height: h))
    }
}

/// Lights, ground and shadows around the viewer's content, driven by `ViewerDefaults.Options`.
@MainActor
enum ViewerStage {
    private static let upgradedName = "modelr-ibl"

    /// (Re)applies the stage. Safe to call repeatedly; `spin` is suppressed while editing.
    static func apply(_ view: SCNView, options: ViewerDefaults.Options, allowSpin: Bool = true) {
        guard let scene = view.scene, let container = scene.rootNode.childNode(withName: "MeshContainer", recursively: false) else { return }
        let staged = options.isStaged
        let image = EnvironmentImages.image(for: options.environment, customPath: options.customPath)

        // Image-based lighting (affects physically based materials) and optional backdrop.
        scene.lightingEnvironment.contents = image
        scene.lightingEnvironment.intensity = CGFloat(options.intensity)
        scene.background.contents = (options.environmentBackground && image != nil) ? image : nil

        view.autoenablesDefaultLighting = !staged
        if let base = scene.rootNode.childNode(withName: "BaseAmbient", recursively: false)?.light {
            base.intensity = staged ? (image == nil ? 450 : 120) : 280
        }

        let (center, radius) = bounds(of: container)
        configureKeyLight(scene, center: center, radius: radius, staged: staged, shadows: options.shadows, hasImage: image != nil)
        configureGround(scene, container: container, center: center, radius: radius, options: options)
        restyleMaterials(container, usePBR: image != nil)

        // Turntable
        let spinning = container.action(forKey: "spin") != nil
        if options.autoRotate && allowSpin {
            if !spinning {
                container.runAction(.repeatForever(.rotateBy(x: 0, y: .pi * 2, z: 0, duration: 24)), forKey: "spin")
            }
        } else if spinning {
            container.removeAction(forKey: "spin")
        }
    }

    /// Bounding sphere of the content in scene space (center, radius), with a safe fallback.
    static func bounds(of container: SCNNode) -> (SCNVector3, CGFloat) {
        let (lo, hi) = container.boundingBox
        guard hi.x > lo.x || hi.y > lo.y || hi.z > lo.z else { return (SCNVector3(0, 0, 0), 1) }
        let c = SCNVector3((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)
        let r = max(max(hi.x - lo.x, hi.y - lo.y), hi.z - lo.z) / 2
        return (c, max(r, 0.05))
    }

    private static func configureKeyLight(_ scene: SCNScene, center: SCNVector3, radius: CGFloat,
                                          staged: Bool, shadows: Bool, hasImage: Bool) {
        var node = scene.rootNode.childNode(withName: "KeyLight", recursively: false)
        guard staged else { node?.removeFromParentNode(); return }
        if node == nil {
            let n = SCNNode(); n.name = "KeyLight"; n.light = SCNLight(); n.light?.type = .directional
            scene.rootNode.addChildNode(n); node = n
        }
        guard let n = node, let light = n.light else { return }
        light.intensity = hasImage ? 650 : 1100
        light.castsShadow = shadows
        light.shadowMode = .deferred
        light.shadowRadius = 10
        light.shadowSampleCount = 12
        light.shadowColor = NSColor.black.withAlphaComponent(0.6)
        light.automaticallyAdjustsShadowProjection = true
        n.position = SCNVector3(center.x + radius * 1.5, center.y + radius * 4, center.z + radius * 2.5)
        n.look(at: center)
    }

    private static func configureGround(_ scene: SCNScene, container: SCNNode, center: SCNVector3,
                                        radius: CGFloat, options: ViewerDefaults.Options) {
        var node = scene.rootNode.childNode(withName: "Ground", recursively: false)
        guard options.showGround || options.shadows else { node?.removeFromParentNode(); return }
        if node == nil {
            let n = SCNNode(geometry: SCNPlane(width: 1, height: 1)); n.name = "Ground"
            n.eulerAngles = SCNVector3(-CGFloat.pi / 2, 0, 0)
            scene.rootNode.addChildNode(n); node = n
        }
        guard let n = node, let plane = n.geometry as? SCNPlane else { return }
        let (lo, _) = container.boundingBox
        plane.width = radius * 14; plane.height = radius * 14
        n.position = SCNVector3(center.x, lo.y - radius * 0.002, center.z)
        let m = SCNMaterial()
        if options.showGround {
            m.lightingModel = .lambert
            m.diffuse.contents = NSColor(white: 0.17, alpha: 1)
        } else {
            m.lightingModel = .shadowOnly                      // invisible floor that only catches the shadow
        }
        m.isDoubleSided = true
        plane.materials = [m]
    }

    /// With an environment, upgrade plain materials to physically based so reflections show;
    /// without one, put them back.
    private static func restyleMaterials(_ node: SCNNode, usePBR: Bool) {
        if let g = node.geometry {
            for m in g.materials {
                if usePBR, m.lightingModel == .blinn, m.name != upgradedName {
                    m.lightingModel = .physicallyBased
                    m.roughness.contents = 0.55
                    m.metalness.contents = 0.0
                    m.name = upgradedName
                } else if !usePBR, m.name == upgradedName {
                    m.lightingModel = .blinn
                    m.name = nil
                }
            }
        }
        node.childNodes.forEach { restyleMaterials($0, usePBR: usePBR) }
    }
}
