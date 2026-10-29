import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Image input and 3D output. Side-by-side when there's room, stacked when narrow,
/// so nothing ever gets clipped. Status pills mirror the §4.4/§4.5 state machines
/// 1:1 — the UI shows the state machine, no bespoke strings.
struct ProjectDetailView: View {
    @Environment(AppRuntime.self) private var runtime
    @Environment(ProjectStore.self) private var store
    let project: Project
    @State private var objectLoading = false
    @State private var showSettings = false
    @State private var showHistory = false
    @State private var showMeshEdit = false
    @State private var sculpt: SculptController?
    @State private var sculptBusy = false
    @State private var texturePaint: TexturePaintController?
    @State private var paintBusy = false
    @State private var showMaterialEdit = false
    @State private var showPaintSettings = false
    @State private var editorInputs: MaskEditorInputs?

    // Touch-up editor state (controls live in the top toolbar while editing).
    @State private var tool: MaskTool = .brush
    @State private var keepMode = false
    @State private var brushRadius: CGFloat = 22
    @StateObject private var maskHolder = MaskCanvasHolder()
    private var isEditingMask: Bool { editorInputs != nil }

    private var shapeJob: ShapeJobState { runtime.state.shapeState(project.id) }
    private var paintJob: PaintJobState { runtime.state.paintState(project.id) }

    var body: some View {
        GeometryReader { geo in
            // Prefer side-by-side; stack top/bottom only when the detail area gets
            // narrower than a minimum. The active section (paint) keeps 50%, and the
            // image+model pair stays stacked together in their half.
            let narrow = geo.size.width < 720
            let mainAxis = narrow
                ? AnyLayout(VStackLayout(spacing: 0))
                : AnyLayout(HStackLayout(spacing: 0))

            ZStack {
                if paintMode {
                    mainAxis {
                        VStack(spacing: 0) {
                            imagePane.frame(maxWidth: .infinity, maxHeight: .infinity)
                            Divider()
                            outputPane.frame(maxWidth: .infinity, maxHeight: .infinity)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        Divider()
                        paintPane.frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                } else {
                    mainAxis {
                        imagePane.frame(maxWidth: .infinity, maxHeight: .infinity)
                        Divider()
                        outputPane.frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                }

                if let inputs = editorInputs {
                    MaskEditorOverlay(inputs: inputs, holder: maskHolder,
                                      tool: tool, keepMode: keepMode, brushRadius: brushRadius)
                        .transition(.opacity)
                }
            }
            .animation(.easeInOut(duration: 0.2), value: editorInputs?.id)
            .animation(.easeInOut(duration: 0.25), value: paintMode)
        }
        .navigationTitle(project.name)
        .toolbar {
            if isEditingMask { editToolbar } else { normalToolbar }
        }
    }

    /// Show the paint strip while painting, on a paint error, or when viewing a paint
    /// version (so its texture is visible without replacing the shape model).
    private var paintMode: Bool {
        if paintJob.isRunning { return true }
        if case .failed = paintJob { return true }
        return project.currentGeneration?.kind == .paint
    }

    private var imagePane: some View {
        ZStack(alignment: .topTrailing) {
            ImageDropView(imageURL: store.imageURL(for: project),
                          reloadKey: store.inputVersion(project.id)) { dropped in
                switch dropped {
                case .url(let url):   store.setImage(fromURL: url, for: project.id)
                case .image(let img): store.setImage(img, for: project.id)
                }
            }
            if store.imageURL(for: project) != nil {
                if !project.removeBackground {
                    badge("Remove background?", system: "person.crop.rectangle.badge.xmark", prominent: true) {
                        store.setRemoveBackground(true, for: project.id)
                    }
                    .padding(14)
                    .transition(.move(edge: .top).combined(with: .opacity))
                } else {
                    badge("Touch up removal?", system: "paintbrush.pointed", prominent: false) {
                        if let (original, mask) = store.maskEditingInputs(for: project.id) {
                            editorInputs = MaskEditorInputs(original: original, mask: mask)
                        }
                    }
                    .padding(14)
                    .transition(.move(edge: .top).combined(with: .opacity))
                }
            }
            if store.imageURL(for: project) != nil, !shapeJob.isRunning {
                VStack {
                    Spacer()
                    ReferenceViewsStrip(projectID: project.id, reloadKey: store.inputVersion(project.id))
                        .environment(store)
                }
                .padding(.bottom, store.importErrors[project.id] == nil ? 14 : 70)
                .frame(maxWidth: .infinity, alignment: .center)
            }
            if let importError = store.importErrors[project.id] {
                VStack {
                    Spacer()
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange)
                        Text(importError).font(.caption)
                        Button { store.clearImportError(project.id) } label: {
                            Image(systemName: "xmark").font(.caption2)
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(10)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
                    .padding(12)
                }
                .frame(maxWidth: .infinity, alignment: .center)
            }
        }
        .animation(.snappy(duration: 0.2), value: project.removeBackground)
    }

    /// The paint process: the streaming 6-view grid while generating, then the
    /// interactive textured mesh once baked.
    private var paintPane: some View {
        ZStack {
            if paintJob.isRunning {
                if let grid = runtime.paintViewPreviews[project.id] {
                    StreamImage(url: grid)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .padding(16)
                } else {
                    PointCloud().frame(width: 130, height: 130)
                }
            } else if let textured = texturedContent {
                MeshViewer(content: textured, paint: texturePaint)
            }
            paintStatusOverlay
            if let texturePaint {
                VStack {
                    Spacer()
                    TexturePaintToolbar(paint: texturePaint, busy: paintBusy,
                                        onSave: {
                                            paintBusy = true
                                            Task {
                                                let err = await store.commitTexturePaint(texturePaint, for: project.id)
                                                paintBusy = false
                                                if err == nil { self.texturePaint = nil }
                                            }
                                        },
                                        onCancel: { self.texturePaint = nil })
                }
                .padding(12)
            }
            exportOverlay(meshURL: paintExportSource?.mesh, texture: paintExportSource?.texture,
                          metallicRoughness: paintExportSource?.mr)
            if paintExportSource != nil {
                VStack {
                    HStack {
                        Spacer()
                        Button { showMaterialEdit = true } label: {
                            Image(systemName: "paintpalette")
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(.primary)
                                .frame(width: 28, height: 28)
                                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                                .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous)
                                    .strokeBorder(.white.opacity(0.12)))
                        }
                        .buttonStyle(.plain)
                        .help("Edit material: brightness, saturation, hue, roughness, metalness")
                        .popover(isPresented: $showMaterialEdit, arrowEdge: .bottom) {
                            MaterialEditPopover(projectID: project.id).environment(store)
                        }
                        Button { Task { texturePaint = await store.beginTexturePaint(for: project.id) } } label: {
                            Image(systemName: "paintbrush.pointed")
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(.primary)
                                .frame(width: 28, height: 28)
                                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                                .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous)
                                    .strokeBorder(.white.opacity(0.12)))
                        }
                        .buttonStyle(.plain)
                        .disabled(texturePaint != nil)
                        .help("Paint on the texture")
                        Color.clear.frame(width: 28 + 8, height: 28)   // clear the export button
                    }
                    Spacer()
                }
                .padding(10)
            }
        }
    }

    /// The interactive textured mesh — only when a paint version is selected.
    private var texturedContent: ViewerContent? {
        guard project.currentGeneration?.kind == .paint else { return nil }
        return store.currentViewerContent(for: project)
    }

    /// The textured mesh + maps to export from the paint viewport (nil while painting).
    /// `mr` is set only for a PBR version → the GLB carries both textures.
    private var paintExportSource: (mesh: URL, texture: URL, mr: URL?)? {
        guard !paintJob.isRunning else { return nil }
        switch texturedContent {
        case .pbrMesh(let mesh, let albedo, let mr)?: return (mesh, albedo, mr)
        case .texturedMesh(let mesh, let tex)?:       return (mesh, tex, nil)
        default:                                      return nil
        }
    }

    @ViewBuilder
    private var paintStatusOverlay: some View {
        switch paintJob {
        case .idle:
            EmptyView()
        case .failed(let stage, let message):
            FailurePill(title: "Paint failed — \(stage)", message: message, stage: stage,
                        model: runtime.lastPaintRunDetails[project.id]?.model,
                        seed: runtime.lastPaintRunDetails[project.id]?.seed,
                        retry: { runtime.requestPaint(project.id) },
                        dismiss: { runtime.dispatch(.paintFailureDismissed(project: project.id)) })
        case .denoising(let k, let n):
            statusPill("Denoising", detail: n > 0 ? "\(k)/\(n)" : nil,
                       fraction: runtime.state.paint[project.id]?.fraction)
        default:
            statusPill(AppReducer.paintStageLabel(paintJob), detail: nil,
                       fraction: runtime.state.paint[project.id]?.fraction)
        }
    }

    // The touch-up tools take over the toolbar while editing — run/quality are
    // irrelevant mid-edit, so they're swapped out rather than shown alongside.
    @ToolbarContentBuilder
    private var editToolbar: some ToolbarContent {
        ToolbarItemGroup {
            Picker("", selection: $tool) {
                Image(systemName: "paintbrush.pointed.fill").tag(MaskTool.brush)
                Image(systemName: "lasso").tag(MaskTool.lasso)
            }
            .pickerStyle(.segmented).help("Brush or lasso")

            Picker("", selection: $keepMode) {
                Image(systemName: "minus").tag(false)
                Image(systemName: "plus").tag(true)
            }
            .pickerStyle(.segmented).help("Remove (−) or keep (+)")

            if tool == .brush {
                HStack(spacing: 6) {
                    Image(systemName: "circle.dotted").foregroundStyle(.secondary).imageScale(.small)
                    Slider(value: $brushRadius, in: 3...90).frame(width: 100)
                }
            }

            Button { maskHolder.canvas?.fit() } label: {
                Image(systemName: "arrow.up.left.and.down.right.magnifyingglass")
            }
            .help("Fit")
        }
        ToolbarItemGroup(placement: .primaryAction) {
            Button { editorInputs = nil } label: { Image(systemName: "xmark") }
                .help("Cancel")
            Button {
                if let mask = maskHolder.canvas?.currentMask() {
                    store.applyEditedMask(mask, for: project.id)
                }
                editorInputs = nil
            } label: { Image(systemName: "checkmark") }
                .buttonStyle(.borderedProminent)
                .help("Apply")
        }
    }

    @ToolbarContentBuilder
    private var normalToolbar: some ToolbarContent {
        // Island 1 — version history
        if !project.generations.isEmpty {
            ToolbarItem(placement: .primaryAction) {
                Button { showHistory = true } label: {
                    Image(systemName: "clock.arrow.circlepath")
                }
                .help("Version history")
                .popover(isPresented: $showHistory, arrowEdge: .bottom) {
                    VersionHistoryView(projectID: project.id, onClose: { showHistory = false })
                        .environment(store)
                }
            }
            if #available(macOS 26.0, *) {
                ToolbarSpacer(.fixed, placement: .primaryAction)
            }
        }

        // Island 2 — SHAPE: section icon + setting · | · action
        ToolbarItem(placement: .primaryAction) {
            HStack(spacing: 12) {
                configButton(settingsLabel, icon: "cube", disabled: shapeJob.isRunning) {
                    showSettings = true
                }
                .help("Shape model & quality")
                .popover(isPresented: $showSettings, arrowEdge: .bottom) {
                    ModelSettingsPopover(projectID: project.id).environment(store)
                }
                Divider().frame(height: 18)
                shapeActionButton
            }
            .padding(.horizontal, 10)
        }

        // Island 3 — PAINT: same template, paint-colored action. Visible as soon as
        // the project has an image so the pipeline is discoverable; the action is
        // gated (with a reason tooltip) until a shape mesh exists (§5, edge states).
        if store.imageURL(for: project) != nil || project.currentGeneration != nil {
            if #available(macOS 26.0, *) {
                ToolbarSpacer(.fixed, placement: .primaryAction)
            }
            ToolbarItem(placement: .primaryAction) {
                HStack(spacing: 12) {
                    configButton(paintLabel, icon: "paintpalette", disabled: paintJob.isRunning) {
                        showPaintSettings = true
                    }
                    .help("Paint settings")
                    .popover(isPresented: $showPaintSettings, arrowEdge: .bottom) {
                        PaintSettingsPopover(projectID: project.id).environment(store)
                    }
                    Divider().frame(height: 18)
                    paintActionButton
                }
                .padding(.horizontal, 10)
                // .help on the island (not the disabled pill): macOS suppresses help
                // tags on disabled controls, and the reason must stay discoverable.
                .help(hasShapeMesh || paintJob.isRunning ? "" : "Generate a shape first")
            }
        }
    }

    // MARK: - install-state gates (§4.9 weightsMissing / §5 edge states)

    private var anyShapeModelInstalled: Bool {
        runtime.state.installState(.shapeSmall).isInstalled
            || runtime.state.installState(.shapeLarge).isInstalled
    }
    private var anyPaintModelInstalled: Bool {
        runtime.state.installState(.paintSmall).isInstalled
            || runtime.state.installState(.paintLarge).isInstalled
    }
    private var hasShapeMesh: Bool { store.shapeMeshURL(for: project) != nil }

    /// A section's config control: section icon + the current setting, opens a popover.
    /// Identical shape for Shape and Paint so the two islands read consistently.
    private func configButton(_ text: String, icon: String, disabled: Bool,
                              action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: icon).font(.callout).foregroundStyle(.secondary)
                Text(text).font(.callout.weight(.medium)).foregroundStyle(.primary)
            }
            .padding(.vertical, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.borderless)
        .disabled(disabled)
        .opacity(disabled ? 0.5 : 1)
    }

    /// A section's primary action: a filled, color-coded capsule. The fill stays on the
    /// capsule, never flooding the toolbar island. Same shape for both sections.
    private func actionPill(_ title: String, _ icon: String, fill: Color,
                            action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: icon)
                .labelStyle(.titleAndIcon)
                .font(.callout.weight(.semibold))
                .foregroundStyle(.white)
                .lineLimit(1)
                .fixedSize()
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(fill, in: Capsule())
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }

    /// Paint config summary for the toolbar (model + chosen texture quality).
    private var paintLabel: String {
        let quality = project.paintAdvanced ? "Custom" : project.paintQuality.label
        return "\(project.paintModel.label) · \(quality)"
    }

    private var paintPink: Color { Color(red: 0.90, green: 0.25, blue: 0.85) }
    private var shapeBlue: Color { Color(red: 0.20, green: 0.50, blue: 0.95) }

    @ViewBuilder
    private var shapeActionButton: some View {
        if store.imageURL(for: project) != nil {
            if shapeJob.isRunning {
                // Cancel edges exist for Preparing…Decoding only (§4.4): the
                // button stays visible but inert during Meshing/Committing.
                actionPill("Stop", "stop.fill", fill: .red) { runtime.cancelShape(project.id) }
                    .disabled(!shapeJob.isCancellable)
                    .opacity(shapeJob.isCancellable ? 1 : 0.45)
            } else if !anyShapeModelInstalled {
                // No shape weights at all: an explicit CTA that routes to the model
                // manager (§4.9 weightsMissing — never a bare error).
                actionPill("Get Models", "arrow.down.circle", fill: .gray) { runtime.showModelManager() }
                    .help("Install a shape model to generate")
            } else if project.generations.isEmpty {
                actionPill("Generate", "play.fill", fill: shapeBlue) { runtime.requestGenerate(project.id) }
            } else {
                actionPill("Regenerate", "arrow.clockwise", fill: shapeBlue) { runtime.requestGenerate(project.id) }
            }
        }
    }

    @ViewBuilder
    private var paintActionButton: some View {
        if paintJob.isRunning {
            // §4.5 allows cancel only from Rendering/Denoising.
            actionPill("Stop", "stop.fill", fill: .red) { runtime.cancelPaint(project.id) }
                .disabled(!paintJob.isCancellable)
                .opacity(paintJob.isCancellable ? 1 : 0.45)
        } else if !anyPaintModelInstalled {
            actionPill("Get Models", "arrow.down.circle", fill: .gray) { runtime.showModelManager() }
                .help("Install a paint model to texture")
        } else {
            // Gated until a shape mesh exists; the island carries the reason tooltip
            // (macOS hides help tags on disabled controls).
            actionPill("Paint", "paintbrush.fill", fill: paintPink) { runtime.requestPaint(project.id) }
                .disabled(!hasShapeMesh)
                .opacity(hasShapeMesh ? 1 : 0.45)
                .help(hasShapeMesh ? "Generate a texture" : "")
        }
    }

    /// App-styled corner badge (rounded, material/accent, padded) for the image pane.
    private func badge(_ title: String, system: String, prominent: Bool,
                       action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: system)
                .font(.callout.weight(.medium))
                .padding(.horizontal, 15)
                .padding(.vertical, 10)
                .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .buttonStyle(.plain)
        .foregroundStyle(prominent ? Color.white : Color.primary)
        .background(
            prominent ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.ultraThinMaterial),
            in: RoundedRectangle(cornerRadius: 14, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(.white.opacity(prominent ? 0.0 : 0.10))
        )
        .shadow(color: .black.opacity(0.22), radius: 9, y: 3)
    }

    private var outputPane: some View {
        ZStack {
            if let content = displayedContent {
                MeshViewer(content: content, onLoading: { objectLoading = $0 }, sculpt: sculpt)
            } else if shapeJob.isRunning {
                PointCloud()                     // animation while loading (no content yet)
                    .frame(width: 150, height: 150)
            } else if store.imageURL(for: project) != nil, case .idle = shapeJob {
                // Image imported but nothing generated yet: a clear CTA in the pane
                // itself (the toolbar pill alone is easy to miss on first use).
                generateCTA
            }
            statusOverlay
            exportOverlay(meshURL: shapeJob.isRunning ? nil : store.shapeMeshURL(for: project),
                          texture: nil, editable: !shapeJob.isRunning && sculpt == nil)
            if let sculpt {
                VStack {
                    Spacer()
                    SculptToolbar(sculpt: sculpt, busy: sculptBusy,
                                  onSave: {
                                      sculptBusy = true
                                      Task {
                                          let err = await store.commitSculpt(sculpt, for: project.id)
                                          sculptBusy = false
                                          if err == nil { self.sculpt = nil }
                                      }
                                  },
                                  onCancel: { self.sculpt = nil })
                }
                .padding(12)
            }
        }
    }

    /// Empty-state call to action for the shape pane. Routes to the model manager
    /// when no shape model is installed (§4.9 weightsMissing), otherwise generates.
    private var generateCTA: some View {
        VStack(spacing: 10) {
            Image(systemName: "cube.transparent")
                .font(.system(size: 34, weight: .light))
                .foregroundStyle(.tertiary)
            Text("No 3D model yet")
                .font(.callout)
                .foregroundStyle(.secondary)
            if anyShapeModelInstalled {
                actionPill("Generate", "play.fill", fill: shapeBlue) { runtime.requestGenerate(project.id) }
            } else {
                actionPill("Get Models", "arrow.down.circle", fill: .gray) { runtime.showModelManager() }
                Text("Install a shape model to generate")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: - Export

    /// A floating export menu in the top-right of a viewport. Renders only when there's
    /// a finished mesh to export. `texture` non-nil ⇒ GLB/OBJ embed the painted texture.
    @ViewBuilder
    private func exportOverlay(meshURL: URL?, texture: URL?, metallicRoughness: URL? = nil,
                               editable: Bool = false) -> some View {
        if let meshURL {
            VStack {
                HStack {
                    Spacer()
                    if editable {
                        Button { showMeshEdit = true } label: {
                            Image(systemName: "wand.and.stars")
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(.primary)
                                .frame(width: 28, height: 28)
                                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                                .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous)
                                    .strokeBorder(.white.opacity(0.12)))
                        }
                        .buttonStyle(.plain)
                        .help("Edit mesh: decimate, smooth, fill holes")
                        .popover(isPresented: $showMeshEdit, arrowEdge: .bottom) {
                            MeshEditPopover(projectID: project.id, onSculpt: {
                                showMeshEdit = false
                                Task { sculpt = await store.beginSculpt(for: project.id) }
                            }).environment(store)
                        }
                    }
                    Menu {
                        ForEach(MeshExportFormat.allCases) { fmt in
                            Button {
                                startExport(meshURL: meshURL, texture: texture,
                                            metallicRoughness: metallicRoughness, format: fmt)
                            } label: {
                                Label(fmt.menuTitle, systemImage: fmt.icon)
                            }
                        }
                    } label: {
                        Image(systemName: "square.and.arrow.up")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(.primary)
                            .frame(width: 28, height: 28)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                            .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous)
                                .strokeBorder(.white.opacity(0.12)))
                    }
                    .menuStyle(.borderlessButton)
                    .menuIndicator(.hidden)
                    .fixedSize()
                    .help("Export model (STL, GLB, OBJ, PLY)")
                }
                Spacer()
            }
            .padding(10)
        }
    }

    private func startExport(meshURL: URL, texture: URL?, metallicRoughness: URL? = nil,
                             format: MeshExportFormat) {
        let panel = NSSavePanel()
        let safe = project.name
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: ":", with: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        panel.nameFieldStringValue = "\(safe.isEmpty ? "model" : safe).\(format.ext)"
        if let ut = UTType(filenameExtension: format.ext) { panel.allowedContentTypes = [ut] }
        panel.canCreateDirectories = true
        panel.isExtensionHidden = false
        panel.title = "Export \(format.ext.uppercased())"
        panel.begin { response in
            guard response == .OK, let dest = panel.url else { return }
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try MeshExporter.export(meshURL: meshURL, texture: texture,
                                            metallicRoughness: metallicRoughness,
                                            format: format, to: dest)
                    DispatchQueue.main.async { NSWorkspace.shared.activateFileViewerSelecting([dest]) }
                } catch {
                    DispatchQueue.main.async {
                        let alert = NSAlert()
                        alert.messageText = "Export failed"
                        alert.informativeText = "\(error)"
                        alert.runModal()
                    }
                }
            }
        }
    }

    /// The currently-selected shape config (what the next Start/Regenerate will use),
    /// so it updates immediately when you change quality. Full quant is omitted.
    private var settingsLabel: String {
        let s = project.resolvedSettings
        return s.quant == .full ? s.model.label : "\(s.model.label) · \(s.quant.label)"
    }

    /// What to show in the viewer: the latest streamed preview mesh while running,
    /// else the final mesh.
    private var displayedContent: ViewerContent? {
        if let preview = runtime.shapePreviews[project.id],
           FileManager.default.fileExists(atPath: preview.path) {
            return .mesh(preview)
        }
        // The model pane is always the (untextured) shape geometry.
        if let shape = store.shapeMeshURL(for: project) { return .mesh(shape) }
        return nil
    }

    @ViewBuilder
    private var statusOverlay: some View {
        switch shapeJob {
        case .idle:
            // Not generating, but the (large) mesh may still be loading into the view —
            // right after a run finishes, or when switching to another project.
            if objectLoading {
                statusPill("Loading object…", detail: nil, fraction: nil)
            }
        case .failed(let stage, let message):
            FailurePill(title: "Couldn't generate — \(stage)", message: message, stage: stage,
                        model: runtime.lastShapeRunDetails[project.id]?.model,
                        seed: runtime.lastShapeRunDetails[project.id]?.seed,
                        retry: { runtime.requestGenerate(project.id) },
                        dismiss: { runtime.dispatch(.shapeFailureDismissed(project: project.id)) })
        case .denoising(let k, let n):
            statusPill("Denoising", detail: n > 0 ? "\(k)/\(n)" : nil,
                       fraction: runtime.state.shape[project.id]?.fraction)
        default:
            statusPill(AppReducer.shapeStageLabel(shapeJob), detail: nil,
                       fraction: runtime.state.shape[project.id]?.fraction)
        }
    }

    private func statusPill(_ label: String, detail: String?, fraction: Double?) -> some View {
        VStack {
            Spacer()
            VStack(spacing: 8) {
                Text(detail.map { "\(label) \($0)" } ?? label)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                SmoothProgressBar(fraction: fraction)
                    .frame(width: 200, height: 4)
            }
            .padding(.horizontal, 22)
            .padding(.vertical, 14)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
            .padding(.bottom, 22)
        }
    }
}

/// §4.9 engineFailed: the sticky failure pill. Retry re-dispatches the original
/// request event (the reducer accepts a new run from Failed); Details opens a
/// popover with the full message, stage, model, and seed, plus Copy details.
private struct FailurePill: View {
    let title: String
    let message: String
    let stage: String
    let model: String?
    let seed: UInt64?
    let retry: () -> Void
    let dismiss: () -> Void
    @State private var showDetails = false

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle")
                .foregroundStyle(.orange)
            Text(title)
                .font(.callout.weight(.medium))
                .multilineTextAlignment(.center)
            Text(message)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .lineLimit(3)
            HStack(spacing: 12) {
                Button("Retry", action: retry)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                Button("Details…") { showDetails = true }
                    .buttonStyle(.borderless)
                    .font(.caption)
                    .popover(isPresented: $showDetails, arrowEdge: .bottom) { detailsView }
                Button("Dismiss", action: dismiss)
                    .buttonStyle(.borderless)
                    .font(.caption)
            }
        }
        .padding(16)
        .frame(maxWidth: 280)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private var detailsView: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Failure Details").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 5) {
                GridRow {
                    Text("Stage").foregroundStyle(.secondary)
                    Text(stage)
                }
                if let model {
                    GridRow {
                        Text("Model").foregroundStyle(.secondary)
                        Text(model)
                    }
                }
                if let seed {
                    GridRow {
                        Text("Seed").foregroundStyle(.secondary)
                        Text(String(seed)).monospacedDigit()
                    }
                }
            }
            .font(.caption)
            Divider()
            ScrollView {
                Text(message)
                    .font(.caption)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 140)
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(detailsText, forType: .string)
            } label: {
                Label("Copy Details", systemImage: "doc.on.doc")
            }
            .controlSize(.small)
        }
        .padding(14)
        .frame(width: 320)
    }

    private var detailsText: String {
        var lines = ["Stage: \(stage)"]
        if let model { lines.append("Model: \(model)") }
        if let seed { lines.append("Seed: \(seed)") }
        lines.append("Message: \(message)")
        return lines.joined(separator: "\n")
    }
}

/// Loads an image from a (changing) URL, decoding only when the path changes — used
/// for the streaming paint view-grid so frequent status re-renders don't re-decode.
private struct StreamImage: View {
    let url: URL
    @State private var image: NSImage?

    var body: some View {
        ZStack {
            if let image {
                Image(nsImage: image).resizable().scaledToFit()
            } else {
                ProgressView()
            }
        }
        .task(id: url.path) {
            image = (try? Data(contentsOf: url)).flatMap(NSImage.init(data:))
        }
    }
}
