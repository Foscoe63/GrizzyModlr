import SwiftUI
import UniformTypeIdentifiers

/// Settings window: currently the single Models tab (§5 model manager).
struct SettingsView: View {
    var body: some View {
        TabView {
            ModelManagerView()
                .tabItem { Label("Models", systemImage: "square.stack.3d.up") }
            ToolSettingsView()
                .tabItem { Label("Tools", systemImage: "wand.and.stars") }
            ViewerSettingsView()
                .tabItem { Label("Viewer", systemImage: "camera.viewfinder") }
        }
        .frame(width: 560)
    }
}

/// Settings → Models: one row per model rendering the §4.3 state machine —
/// install / pause / resume / remove / retry / import, live progress, size on
/// disk, and total library usage.
struct ModelManagerView: View {
    @Environment(AppRuntime.self) private var runtime
    @State private var importTarget: ModelID?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(ModelID.allCases) { model in
                ModelRow(model: model, importTarget: $importTarget)
                if model != ModelID.allCases.last {
                    Divider().padding(.leading, 16)
                }
            }
            Divider()
            ModelLocationSection()
            Divider()
            footer
        }
        .padding(.vertical, 8)
        .fileImporter(isPresented: importPresented, allowedContentTypes: [.folder]) { result in
            if let target = importTarget, case .success(let url) = result {
                runtime.importWeights(target, from: url)
            }
            importTarget = nil
        }
    }

    private var importPresented: Binding<Bool> {
        Binding(get: { importTarget != nil },
                set: { if !$0 { importTarget = nil } })
    }

    private var footer: some View {
        TimelineView(.periodic(from: .now, by: 3)) { _ in
            HStack(spacing: 6) {
                Image(systemName: "internaldrive").foregroundStyle(.secondary)
                Text("\(OnboardingView.gb(ModelStore.totalBytesOnDisk())) on disk · \(OnboardingView.gb(DiskSpace.free())) free")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
        }
    }
}

/// One §4.3 machine, rendered. Every state maps to exactly one control set.
private struct ModelRow: View {
    @Environment(AppRuntime.self) private var runtime
    let model: ModelID
    @Binding var importTarget: ModelID?

    private var catalog: CatalogModel { ModelCatalog.model(model) }
    private var state: ModelInstallState { runtime.state.installState(model) }

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text(catalog.displayName).font(.body.weight(.semibold))
                Text(catalog.detail).font(.caption).foregroundStyle(.secondary)
                statusLine
                    .padding(.top, 2)
            }
            Spacer(minLength: 12)
            controls
                .frame(minWidth: 108, alignment: .trailing)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 16)
    }

    // MARK: controls per state

    @ViewBuilder
    private var controls: some View {
        switch state {
        case .notInstalled:
            Menu {
                Button("Download (\(OnboardingView.gb(catalog.totalBytes)))") {
                    runtime.install(model)
                }
                Button("Import weights folder…") { importTarget = model }
            } label: {
                Text("Install").frame(maxWidth: .infinity)
            }
            .frame(width: 108)

        case .queued:
            Text("Waiting…").font(.caption).foregroundStyle(.secondary)
                .frame(width: 108, alignment: .trailing)

        case .downloading:
            Button { runtime.pauseInstall(model) } label: {
                Text("Pause").frame(maxWidth: .infinity)
            }
            .frame(width: 108)

        case .paused:
            HStack(spacing: 8) {
                Button("Cancel", role: .destructive) { runtime.removeInstall(model) }
                Button { runtime.resumeInstall(model) } label: {
                    Text("Resume")
                }
                .buttonStyle(.borderedProminent)
            }

        case .verifying:
            ProgressView().controlSize(.small)
                .frame(width: 108, alignment: .trailing)

        case .installed:
            Button(role: .destructive) { runtime.removeInstall(model) } label: {
                Text("Remove").frame(maxWidth: .infinity)
            }
            .frame(width: 108)

        case .failed:
            HStack(spacing: 8) {
                Button("Import…") { importTarget = model }
                Button("Retry") { runtime.retryInstall(model) }
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    // MARK: status per state

    @ViewBuilder
    private var statusLine: some View {
        switch state {
        case .notInstalled:
            Label("\(OnboardingView.gb(catalog.totalBytes)) download",
                  systemImage: "arrow.down.circle.dotted")
                .font(.caption).foregroundStyle(.tertiary)

        case .queued:
            Label("Queued", systemImage: "clock")
                .font(.caption).foregroundStyle(.secondary)

        case .downloading(let p):
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: p.fraction)
                    .progressViewStyle(.linear)
                Text("File \(p.fileIndex) of \(p.fileCount) — \(p.currentFileName) · \(OnboardingView.gb(p.totalBytes)) of \(OnboardingView.gb(p.totalExpected))")
                    .font(.caption2).foregroundStyle(.tertiary).monospacedDigit()
                    .lineLimit(1)
            }

        case .paused(let bytes):
            Label("Paused · \(OnboardingView.gb(bytes)) of \(OnboardingView.gb(catalog.totalBytes))",
                  systemImage: "pause.circle")
                .font(.caption).foregroundStyle(.secondary)

        case .verifying:
            Label("Verifying checksums…", systemImage: "checkmark.shield")
                .font(.caption).foregroundStyle(.secondary)

        case .installed:
            Label("Installed · \(OnboardingView.gb(ModelStore.bytesOnDisk(model))) on disk",
                  systemImage: "checkmark.circle.fill")
                .font(.caption).foregroundStyle(.green)

        case .failed(let message):
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .font(.caption).foregroundStyle(.orange)
                .lineLimit(2)
        }
    }
}

/// Settings → Models → location: keep the library on another (e.g. fast external) drive.
private struct ModelLocationSection: View {
    @Environment(AppRuntime.self) private var runtime

    private var root: URL { ModelStore.modelsRoot }
    private var isCustom: Bool { ModelStore.customRoot != nil }

    var body: some View {
        TimelineView(.periodic(from: .now, by: 3)) { _ in
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Image(systemName: isCustom ? "externaldrive.fill" : "internaldrive").foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Models location").font(.callout.weight(.semibold))
                        Text(root.path(percentEncoded: false))
                            .font(.caption).foregroundStyle(.secondary)
                            .lineLimit(1).truncationMode(.middle)
                            .help(root.path(percentEncoded: false))
                        if !ModelStore.rootIsAvailable {
                            Label("This drive isn't connected. Reconnect it, or switch back to the default location.",
                                  systemImage: "exclamationmark.triangle.fill")
                                .font(.caption).foregroundStyle(.orange)
                        }
                    }
                    Spacer()
                    controls
                }
                progress
            }
            .padding(.horizontal, 16).padding(.vertical, 10)
        }
    }

    @ViewBuilder
    private var controls: some View {
        let enabled = runtime.canChangeModelsLocation
        HStack(spacing: 6) {
            Button("Move…") { if let url = pickFolder("Move Models Here", "Choose the folder or drive to move the models to.") { runtime.moveModels(toFolder: url) } }
                .disabled(!enabled || !ModelStore.rootIsAvailable)
            Button("Use Existing…") { if let url = pickFolder("Use Models Here", "Choose a folder that already contains the models.") { runtime.useExistingModels(at: url) } }
                .disabled(!enabled)
            if isCustom {
                Menu("Default") {
                    Button("Use default location") { runtime.resetModelsLocation(moveBack: false) }
                    Button("Move back to default location") { runtime.resetModelsLocation(moveBack: true) }
                        .disabled(!ModelStore.rootIsAvailable)
                }
                .menuStyle(.borderlessButton).fixedSize().disabled(!enabled)
            }
        }
        .controlSize(.small)
        .help(enabled ? "" : "Finish downloads and generations first.")
    }

    @ViewBuilder
    private var progress: some View {
        switch runtime.relocation {
        case .running(let copied, let total, let file)?:
            VStack(alignment: .leading, spacing: 3) {
                ProgressView(value: total > 0 ? Double(copied) / Double(total) : 0)
                HStack {
                    Text("Copying \(OnboardingView.gb(copied)) of \(OnboardingView.gb(total)) · \((file as NSString).lastPathComponent)")
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    Spacer()
                    Button("Cancel") { runtime.cancelRelocation() }.controlSize(.small)
                }
                Text("The old copy is deleted only after everything is verified.")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        case .failed(let message)?:
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                Text(message).font(.caption)
                Spacer()
                Button { runtime.dismissRelocationFailure() } label: { Image(systemName: "xmark") }.buttonStyle(.plain)
            }
        case nil:
            Text(isCustom ? "Models are stored on a custom drive. Generation reads them from there."
                          : "Models are stored in the app's folder. Move them to a faster or larger drive any time.")
                .font(.caption2).foregroundStyle(.tertiary)
        }
    }

    private func pickFolder(_ title: String, _ message: String) -> URL? {
        let panel = NSOpenPanel()
        panel.title = title; panel.message = message; panel.prompt = "Choose"
        panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.canCreateDirectories = true; panel.allowsMultipleSelection = false
        return panel.runModal() == .OK ? panel.url : nil
    }
}

/// Settings → Tools: starting values for the editing tools. Each tool opens with these and
/// you can still adjust them per use in the tool's own panel.
struct ToolSettingsView: View {
    @AppStorage(ToolDefaults.Key.decimateK) private var decimateK = ToolDefaults.fallbackDecimateK
    @AppStorage(ToolDefaults.Key.smoothIterations) private var smoothIterations = ToolDefaults.fallbackSmoothIterations
    @AppStorage(ToolDefaults.Key.holeEdges) private var holeEdges = ToolDefaults.fallbackHoleEdges
    @AppStorage(ToolDefaults.Key.booleanResolution) private var booleanResolution = ToolDefaults.fallbackBooleanResolution
    @AppStorage(ToolDefaults.Key.remeshResolution) private var remeshResolution = ToolDefaults.fallbackRemeshResolution
    @AppStorage(ToolDefaults.Key.symmetryAxis) private var symmetryAxis = ToolDefaults.fallbackSymmetryAxis
    @AppStorage(ToolDefaults.Key.symmetryKeepPositive) private var keepPositive = true
    @AppStorage(ToolDefaults.Key.lodLevels) private var lodLevels = ToolDefaults.fallbackLODLevels
    @AppStorage(ToolDefaults.Key.sculptSize) private var sculptSize = ToolDefaults.fallbackSculptSize
    @AppStorage(ToolDefaults.Key.sculptStrength) private var sculptStrength = ToolDefaults.fallbackSculptStrength
    @AppStorage(ToolDefaults.Key.paintSize) private var paintSize = ToolDefaults.fallbackPaintSize
    @AppStorage(ToolDefaults.Key.paintHardness) private var paintHardness = ToolDefaults.fallbackPaintHardness
    @AppStorage(ToolDefaults.Key.paintOpacity) private var paintOpacity = ToolDefaults.fallbackPaintOpacity
    @AppStorage(ToolDefaults.Key.sceneSpacing) private var sceneSpacing = ToolDefaults.fallbackSceneSpacing
    @AppStorage(ToolDefaults.Key.exportQuads) private var exportQuads = false
    @AppStorage(ToolDefaults.Key.ttiSteps) private var ttiSteps = ToolDefaults.fallbackTTISteps
    @AppStorage(ToolDefaults.Key.ttiGuidance) private var ttiGuidance = ToolDefaults.fallbackTTIGuidance
    @AppStorage(ToolDefaults.Key.ttiNegative) private var ttiNegative = ToolDefaults.fallbackTTINegative
    @AppStorage(ToolDefaults.Key.ttiSuffix) private var ttiSuffix = ToolDefaults.fallbackTTISuffix
    @AppStorage(ToolDefaults.Key.ttiRemoveBackground) private var ttiRemoveBackground = true
    @AppStorage(ToolDefaults.Key.ttiAutoGenerate) private var ttiAutoGenerate = false

    var body: some View {
        Form {
            Section("Mesh editing") {
                LabeledContent("Decimate target") {
                    slider($decimateK, 5...400, step: 5, label: "\(Int(decimateK))k triangles")
                }
                LabeledContent("Smoothing") {
                    slider($smoothIterations, 1...20, step: 1, label: "\(Int(smoothIterations)) iterations")
                }
                LabeledContent("Fill holes up to") {
                    slider($holeEdges, 4...256, step: 4, label: "\(Int(holeEdges)) edges")
                }
            }
            Section("Topology") {
                Picker("Symmetry axis", selection: $symmetryAxis) {
                    ForEach(SymmetryAxis.allCases) { Text($0.rawValue.uppercased()).tag($0.rawValue) }
                }
                .pickerStyle(.segmented)
                Picker("Keep", selection: $keepPositive) {
                    Text("Positive half").tag(true); Text("Negative half").tag(false)
                }
                .pickerStyle(.segmented)
                LabeledContent("Remesh detail") {
                    slider($remeshResolution, 48...320, step: 16, label: "\(Int(remeshResolution)) voxels")
                }
                LabeledContent("LOD levels") {
                    HStack {
                        TextField("LOD levels", text: $lodLevels, prompt: Text("50, 25, 10")).labelsHidden().textFieldStyle(.roundedBorder).frame(width: 120)
                        Text("% of triangles").font(.caption).foregroundStyle(.secondary)
                    }
                }
                if ToolDefaults.parseLODLevels(lodLevels).isEmpty {
                    Text("No valid levels — the default (50, 25, 10) is used.").font(.caption).foregroundStyle(.orange)
                }
            }
            Section("Combine") {
                LabeledContent("Boolean detail") {
                    slider($booleanResolution, 64...256, step: 16, label: "\(Int(booleanResolution)) voxels")
                }
            }
            Section("Scenes") {
                LabeledContent("Spacing for new objects") {
                    slider($sceneSpacing, 1.5...6, step: 0.25, label: String(format: "%.2f units", sceneSpacing))
                }
            }
            Section("Text to 3D") {
                LabeledContent("Image steps") { slider($ttiSteps, 10...50, step: 1, label: "\(Int(ttiSteps)) steps") }
                LabeledContent("Prompt strength") { slider($ttiGuidance, 3...14, step: 0.5, label: String(format: "%.1f", ttiGuidance)) }
                TextField("Style added to every prompt", text: $ttiSuffix, axis: .vertical).lineLimit(1...3)
                TextField("Negative prompt", text: $ttiNegative, axis: .vertical).lineLimit(1...3)
                Toggle("Remove the background after generating", isOn: $ttiRemoveBackground)
                Toggle("Start the 3D generation automatically", isOn: $ttiAutoGenerate)
                Text("Describe an object in the image panel; Stable Diffusion 2.1 makes the starting image, then the shape model builds it.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Export") {
                Toggle("OBJ as quads where triangles pair cleanly", isOn: $exportQuads)
                Text("Merges adjacent triangles into quads (best on remeshed models). GLB, STL and PLY stay triangles.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Brushes") {
                LabeledContent("Sculpt size") { slider($sculptSize, 0.02...0.4, step: 0.01, label: String(format: "%.0f%%", sculptSize * 100)) }
                LabeledContent("Sculpt strength") { slider($sculptStrength, 0.05...1, step: 0.05, label: String(format: "%.0f%%", sculptStrength * 100)) }
                LabeledContent("Paint size") { slider($paintSize, 0.01...0.2, step: 0.01, label: String(format: "%.0f%%", paintSize * 100)) }
                LabeledContent("Paint hardness") { slider($paintHardness, 0...1, step: 0.05, label: String(format: "%.0f%%", paintHardness * 100)) }
                LabeledContent("Paint opacity") { slider($paintOpacity, 0.05...1, step: 0.05, label: String(format: "%.0f%%", paintOpacity * 100)) }
            }
            Section {
                Button("Reset to defaults") { reset() }
            }
        }
        .formStyle(.grouped)
        .frame(height: 640)
    }

    private func slider(_ value: Binding<Double>, _ range: ClosedRange<Double>, step: Double, label: String) -> some View {
        HStack {
            Slider(value: value, in: range, step: step).frame(width: 170)
            Text(label).font(.caption).foregroundStyle(.secondary).monospacedDigit().frame(width: 100, alignment: .trailing)
        }
    }

    private func reset() {
        decimateK = ToolDefaults.fallbackDecimateK; smoothIterations = ToolDefaults.fallbackSmoothIterations
        holeEdges = ToolDefaults.fallbackHoleEdges; booleanResolution = ToolDefaults.fallbackBooleanResolution
        remeshResolution = ToolDefaults.fallbackRemeshResolution; symmetryAxis = ToolDefaults.fallbackSymmetryAxis
        keepPositive = true; lodLevels = ToolDefaults.fallbackLODLevels
        sculptSize = ToolDefaults.fallbackSculptSize; sculptStrength = ToolDefaults.fallbackSculptStrength
        paintSize = ToolDefaults.fallbackPaintSize; paintHardness = ToolDefaults.fallbackPaintHardness
        paintOpacity = ToolDefaults.fallbackPaintOpacity; sceneSpacing = ToolDefaults.fallbackSceneSpacing
        exportQuads = false
        ttiSteps = ToolDefaults.fallbackTTISteps; ttiGuidance = ToolDefaults.fallbackTTIGuidance
        ttiNegative = ToolDefaults.fallbackTTINegative; ttiSuffix = ToolDefaults.fallbackTTISuffix
        ttiRemoveBackground = true; ttiAutoGenerate = false
    }
}

/// Settings → Viewer: lighting, ground and turntable defaults for every 3D viewport.
struct ViewerSettingsView: View {
    @AppStorage(ViewerDefaults.Key.environment) private var envName = ViewerDefaults.fallbackEnvironment
    @AppStorage(ViewerDefaults.Key.environmentIntensity) private var intensity = ViewerDefaults.fallbackIntensity
    @AppStorage(ViewerDefaults.Key.environmentBackground) private var envBackground = false
    @AppStorage(ViewerDefaults.Key.showGround) private var showGround = false
    @AppStorage(ViewerDefaults.Key.shadows) private var shadows = false
    @AppStorage(ViewerDefaults.Key.autoRotate) private var autoRotate = false
    @AppStorage(ViewerDefaults.Key.customEnvironmentPath) private var customPath = ""

    var body: some View {
        Form {
            Section("Lighting") {
                Picker("Environment", selection: $envName) {
                    ForEach(ViewerEnvironment.allCases) { Text($0.label).tag($0.rawValue) }
                }
                if envName != ViewerEnvironment.none.rawValue {
                    LabeledContent("Intensity") {
                        HStack {
                            Slider(value: $intensity, in: 0.2...3, step: 0.1).frame(width: 170)
                            Text(String(format: "×%.1f", intensity)).font(.caption).foregroundStyle(.secondary).frame(width: 40)
                        }
                    }
                    Toggle("Show environment as background", isOn: $envBackground)
                }
                if envName == ViewerEnvironment.custom.rawValue {
                    LabeledContent("HDR image") {
                        HStack {
                            Text(customPath.isEmpty ? "None chosen" : (customPath as NSString).lastPathComponent)
                                .font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle).frame(maxWidth: 160)
                            Button("Choose…") { chooseHDR() }
                        }
                    }
                    if customPath.isEmpty {
                        Text("Pick an equirectangular .hdr, .exr, .png or .jpg.").font(.caption).foregroundStyle(.orange)
                    }
                }
                Text("Environments light physically based surfaces and add reflections. Plain meshes switch to a matte PBR look while one is active.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Stage") {
                Toggle("Ground plane", isOn: $showGround)
                Toggle("Shadows", isOn: $shadows)
                Toggle("Turntable (slow auto-rotate)", isOn: $autoRotate)
            }
            Section {
                Button("Reset to defaults") {
                    envName = ViewerDefaults.fallbackEnvironment; intensity = ViewerDefaults.fallbackIntensity
                    envBackground = false; showGround = false; shadows = false; autoRotate = false; customPath = ""
                }
            }
        }
        .formStyle(.grouped)
        .frame(height: 440)
    }

    private func chooseHDR() {
        let panel = NSOpenPanel()
        panel.title = "Choose an environment image"
        panel.allowedContentTypes = [.image]
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url { customPath = url.path }
    }
}
