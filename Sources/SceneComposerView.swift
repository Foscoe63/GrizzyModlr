import SwiftUI
import SceneKit

/// Arrange several saved models (from this or other projects) in one scene, light and frame it,
/// and merge the result into a new shape version.
struct SceneComposerView: View {
    @Environment(ProjectStore.self) private var store
    let projectID: Project.ID
    var onClose: () -> Void

    @State private var selection: UUID?
    @State private var command: ViewerCommand?
    @State private var busy = false
    @State private var message: String?

    private var project: Project? { store.project(projectID) }
    private var items: [SceneItem] { project?.sceneItems ?? [] }

    var body: some View {
        HStack(spacing: 0) {
            sidebar.frame(width: 300)
            Divider()
            ZStack(alignment: .topLeading) {
                SceneViewer(items: items, selected: selection, command: command) { item in
                    guard let owner = store.project(item.projectID),
                          let gen = owner.generations.first(where: { $0.id == item.generationID }) else { return nil }
                    return store.viewerContent(for: gen, in: item.projectID)
                }
                HStack(spacing: 6) {
                    ViewerMenuButton { command = $0 }
                    Button { command = ViewerCommand(kind: .camera(.threeQuarter)) } label: {
                        Image(systemName: "arrow.up.left.and.down.right.magnifyingglass")
                            .font(.system(size: 12, weight: .semibold)).frame(width: 28, height: 28)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                    }
                    .buttonStyle(.plain).help("Frame all objects")
                }
                .padding(10)
                if items.isEmpty {
                    Text("Add objects to start a scene").foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(minWidth: 520)
        }
        .frame(width: 900, height: 620)
        .disabled(busy)
    }

    // MARK: sidebar

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Scene").font(.headline)
                Spacer()
                addMenu
            }
            List(selection: $selection) {
                ForEach(items) { item in
                    HStack {
                        Image(systemName: "cube").foregroundStyle(.secondary)
                        Text(name(of: item)).lineLimit(1)
                        Spacer()
                        Button { store.removeSceneItem(item.id, for: projectID); if selection == item.id { selection = nil } } label: {
                            Image(systemName: "minus.circle")
                        }
                        .buttonStyle(.plain).foregroundStyle(.secondary).help("Remove from scene")
                    }
                    .tag(item.id)
                }
            }
            .frame(height: 170)
            .listStyle(.bordered)

            if let item = items.first(where: { $0.id == selection }) {
                controls(for: item)
            } else {
                Text("Select an object to move, turn or scale it.").font(.caption).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            if let message { Text(message).font(.caption).foregroundStyle(.secondary) }
            HStack {
                Button("Done", action: onClose)
                Spacer()
                Button {
                    busy = true; message = nil
                    Task {
                        let result = await store.saveSceneAsVersion(for: projectID)
                        busy = false
                        message = result ?? "Merged into a new shape version."
                    }
                } label: { Label("Save as version", systemImage: "square.stack.3d.up") }
                .buttonStyle(.borderedProminent).disabled(items.isEmpty)
            }
            Text("Merging combines the objects' geometry (untextured). Paint the new version to texture the whole scene.")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(14)
    }

    private var addMenu: some View {
        Menu {
            ForEach(store.projects) { p in
                let versions = p.generations.filter { store.viewerContent(for: $0, in: p.id) != nil }
                if !versions.isEmpty {
                    Section(p.name) {
                        ForEach(Array(p.generations.enumerated()), id: \.element.id) { n, gen in
                            if store.viewerContent(for: gen, in: p.id) != nil {
                                Button("Version \(n + 1)\(gen.kind == .paint ? " · painted" : "")") {
                                    store.addSceneItem(generation: gen, from: p.id, to: projectID,
                                                       spacing: ToolDefaults.sceneSpacing())
                                    selection = store.project(projectID)?.sceneItems.last?.id
                                }
                            }
                        }
                    }
                }
            }
        } label: { Label("Add object", systemImage: "plus") }
        .menuStyle(.borderlessButton).fixedSize()
    }

    private func name(of item: SceneItem) -> String {
        guard let owner = store.project(item.projectID),
              let idx = owner.generations.firstIndex(where: { $0.id == item.generationID }) else { return "Missing object" }
        let gen = owner.generations[idx]
        return "\(owner.name) · v\(idx + 1)\(gen.kind == .paint ? " painted" : "")"
    }

    @ViewBuilder
    private func controls(for item: SceneItem) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            slider("X", item, \.x, -8...8)
            slider("Y", item, \.y, -4...4)
            slider("Z", item, \.z, -8...8)
            slider("Turn", item, \.rotationY, -180...180, format: "%.0f°")
            slider("Scale", item, \.scale, 0.2...3, format: "×%.2f")
            Button("Reset") {
                var r = item; r.x = item.x; r.y = 0; r.z = 0; r.rotationY = 0; r.scale = 1
                store.updateSceneItem(r, for: projectID)
            }
            .controlSize(.small)
        }
    }

    private func slider(_ title: String, _ item: SceneItem, _ key: WritableKeyPath<SceneItem, Double>,
                        _ range: ClosedRange<Double>, format: String = "%.2f") -> some View {
        HStack(spacing: 8) {
            Text(title).font(.caption).frame(width: 36, alignment: .leading)
            Slider(value: Binding(get: { item[keyPath: key] }, set: { v in
                var u = item; u[keyPath: key] = v; store.updateSceneItem(u, for: projectID)
            }), in: range)
            Text(String(format: format, item[keyPath: key])).font(.caption).monospacedDigit()
                .frame(width: 46, alignment: .trailing).foregroundStyle(.secondary)
        }
    }
}

/// SceneKit view for the composer: one wrapper node per item (user transform) around its
/// normalized model, lit and staged by `ViewerStage` like every other viewport.
struct SceneViewer: NSViewRepresentable {
    let items: [SceneItem]
    let selected: UUID?
    var command: ViewerCommand?
    let content: (SceneItem) -> ViewerContent?

    @AppStorage(ViewerDefaults.Key.environment) private var envName = ViewerDefaults.fallbackEnvironment
    @AppStorage(ViewerDefaults.Key.environmentIntensity) private var envIntensity = ViewerDefaults.fallbackIntensity
    @AppStorage(ViewerDefaults.Key.environmentBackground) private var envBackground = false
    @AppStorage(ViewerDefaults.Key.showGround) private var showGround = false
    @AppStorage(ViewerDefaults.Key.shadows) private var shadows = false
    @AppStorage(ViewerDefaults.Key.autoRotate) private var autoRotate = false
    @AppStorage(ViewerDefaults.Key.customEnvironmentPath) private var customEnvPath = ""

    final class Coordinator {
        var wrappers: [UUID: SCNNode] = [:]
        var framed = false
        var lastCommand: UUID?
    }
    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> SCNView {
        let view = SCNView()
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true
        view.backgroundColor = .clear
        view.antialiasingMode = .multisampling4X
        let scene = SCNScene()
        let container = SCNNode(); container.name = "MeshContainer"
        scene.rootNode.addChildNode(container)
        let ambient = SCNNode(); ambient.name = "BaseAmbient"
        ambient.light = SCNLight(); ambient.light?.type = .ambient; ambient.light?.intensity = 280
        scene.rootNode.addChildNode(ambient)
        let camera = SCNCamera(); camera.fieldOfView = 40; camera.projectionDirection = .horizontal
        camera.zNear = 0.01; camera.zFar = 1000
        let cam = SCNNode(); cam.camera = camera; cam.position = SCNVector3(0, 2, 8)
        cam.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(cam)
        view.scene = scene; view.pointOfView = cam
        return view
    }

    func updateNSView(_ view: SCNView, context: Context) {
        guard let container = view.scene?.rootNode.childNode(withName: "MeshContainer", recursively: false) else { return }
        let co = context.coordinator
        let live = Set(items.map(\.id))
        for (id, node) in co.wrappers where !live.contains(id) { node.removeFromParentNode(); co.wrappers[id] = nil }

        for item in items {
            let wrapper: SCNNode
            if let w = co.wrappers[item.id] { wrapper = w } else {
                wrapper = SCNNode(); co.wrappers[item.id] = wrapper
                container.addChildNode(wrapper)
                if let c = content(item) {
                    DispatchQueue.global(qos: .userInitiated).async {
                        let loaded = MeshViewer.load(c)
                        DispatchQueue.main.async {
                            guard let loaded, co.wrappers[item.id] === wrapper else { return }
                            let s = 1 / Float(loaded.radius)
                            loaded.node.scale = SCNVector3(CGFloat(s), CGFloat(s), CGFloat(s))
                            loaded.node.position = SCNVector3(-loaded.center.x * CGFloat(s), -loaded.center.y * CGFloat(s),
                                                              -loaded.center.z * CGFloat(s))
                            wrapper.addChildNode(loaded.node)
                            self.stage(view)
                            // Frame once every placed object has finished loading.
                            if !co.framed, co.wrappers.values.allSatisfy({ $0.childNodes.contains { $0.name != "SelectionRing" } }) {
                                co.framed = true; self.fit(view, preset: .threeQuarter)
                            }
                        }
                    }
                }
            }
            wrapper.position = SCNVector3(item.x, item.y, item.z)
            wrapper.eulerAngles = SCNVector3(0, CGFloat(item.rotationY) * .pi / 180, 0)
            wrapper.scale = SCNVector3(item.scale, item.scale, item.scale)
            // Selection ring at the object's base.
            let ring = wrapper.childNode(withName: "SelectionRing", recursively: false)
            if item.id == selected, ring == nil {
                let torus = SCNTorus(ringRadius: 1.1, pipeRadius: 0.015)
                let m = SCNMaterial(); m.lightingModel = .constant; m.diffuse.contents = NSColor.systemYellow
                torus.materials = [m]
                let n = SCNNode(geometry: torus); n.name = "SelectionRing"; n.position = SCNVector3(0, -1, 0)
                wrapper.addChildNode(n)
            } else if item.id != selected { ring?.removeFromParentNode() }
        }
        stage(view)
        if let command, co.lastCommand != command.id {
            co.lastCommand = command.id
            switch command.kind {
            case .camera(let p): container.removeAction(forKey: "spin"); container.eulerAngles = SCNVector3(0, 0, 0); fit(view, preset: p)
            case .snapshot:
                let image = view.snapshot()
                DispatchQueue.main.async {
                    let panel = NSSavePanel(); panel.allowedContentTypes = [.png]; panel.nameFieldStringValue = "scene.png"
                    guard panel.runModal() == .OK, let url = panel.url, let tiff = image.tiffRepresentation,
                          let rep = NSBitmapImageRep(data: tiff), let png = rep.representation(using: .png, properties: [:]) else { return }
                    try? png.write(to: url)
                }
            }
        }
    }

    private func stage(_ view: SCNView) {
        ViewerStage.apply(view, options: ViewerDefaults.Options(
            environment: ViewerEnvironment(rawValue: envName) ?? .none, intensity: envIntensity,
            environmentBackground: envBackground, showGround: showGround, shadows: shadows,
            autoRotate: autoRotate, customPath: customEnvPath))
    }

    private func fit(_ view: SCNView, preset: CameraPreset) {
        guard let container = view.scene?.rootNode.childNode(withName: "MeshContainer", recursively: false),
              let node = view.pointOfView, let camera = node.camera else { return }
        let (c, r) = ViewerStage.bounds(of: container)
        let size = view.bounds.size
        let aspect = size.height > 1 ? Double(size.width / size.height) : 1.5
        let h = camera.fieldOfView * .pi / 180
        let limit = min(h, 2 * atan(tan(h / 2) / max(aspect, 0.001)))
        let d = CGFloat(Double(r) / tan(limit / 2) * 1.35)
        let dir = preset.direction, up = preset.up
        node.position = SCNVector3(c.x + CGFloat(dir.x) * d, c.y + CGFloat(dir.y) * d, c.z + CGFloat(dir.z) * d)
        node.look(at: c, up: SCNVector3(CGFloat(up.x), CGFloat(up.y), CGFloat(up.z)), localFront: SCNVector3(0, 0, -1))
        camera.zNear = max(Double(d) * 0.01, 0.01); camera.zFar = Double(d) * 10 + 100
    }
}
