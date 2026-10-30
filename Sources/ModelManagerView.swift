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
    @AppStorage(ToolDefaults.Key.booleanSharp) private var booleanSharp = true
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
    @AppStorage(ToolDefaults.Key.aiFillPrompt) private var aiFillPrompt = ToolDefaults.fallbackAIFillPrompt
    @AppStorage(ToolDefaults.Key.aiFillSteps) private var aiFillSteps = ToolDefaults.fallbackAIFillSteps
    @AppStorage(ToolDefaults.Key.aiFillStrength) private var aiFillStrength = ToolDefaults.fallbackAIFillStrength
    @AppStorage(ToolDefaults.Key.aiFillGuidance) private var aiFillGuidance = ToolDefaults.fallbackAIFillGuidance
    @AppStorage(ToolDefaults.Key.flowFeatureAngle) private var flowFeatureAngle = ToolDefaults.fallbackFlowFeatureAngle
    @AppStorage(ToolDefaults.Key.flowTargetQuads) private var flowTargetQuads = ToolDefaults.fallbackFlowTargetQuads
    @AppStorage(ToolDefaults.Key.flowGuideStrength) private var flowGuideStrength = ToolDefaults.fallbackFlowGuideStrength
    @AppStorage(ToolDefaults.Key.regionTolerance) private var regionTolerance = ToolDefaults.fallbackRegionTolerance
    @AppStorage(ToolDefaults.Key.regionFeather) private var regionFeather = ToolDefaults.fallbackRegionFeather
    @AppStorage(ToolDefaults.Key.islandPadding) private var islandPadding = ToolDefaults.fallbackIslandPadding
    @AppStorage(ToolDefaults.Key.printSizeMM) private var printSizeMM = ToolDefaults.fallbackPrintSizeMM
    @AppStorage(ToolDefaults.Key.slicer) private var slicerChoice = ToolDefaults.fallbackSlicer
    @AppStorage(ToolDefaults.Key.batchPaint) private var batchPaint = false
    @AppStorage(ToolDefaults.Key.tsrThreshold) private var tsrThreshold = ToolDefaults.fallbackTSRThreshold
    @AppStorage(ToolDefaults.Key.tsrForegroundRatio) private var tsrForegroundRatio = ToolDefaults.fallbackTSRForegroundRatio
    @AppStorage(ToolDefaults.Key.tsrKeepLargest) private var tsrKeepLargest = true
    @AppStorage(ToolDefaults.Key.reliefDepth) private var reliefDepth = ToolDefaults.fallbackReliefDepth
    @AppStorage(ToolDefaults.Key.reliefResolution) private var reliefResolution = ToolDefaults.fallbackReliefResolution
    @AppStorage(ToolDefaults.Key.reliefSmoothing) private var reliefSmoothing = ToolDefaults.fallbackReliefSmoothing
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
                    slider($booleanResolution, 64...384, step: 16, label: "\(Int(booleanResolution)) voxels")
                }
                Toggle("Keep sharp edges", isOn: $booleanSharp)
                Text("Places result vertices on the creases where shapes meet, so cuts stay crisp at lower detail. Above 256 voxels uses a lot of memory.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Scenes") {
                LabeledContent("Spacing for new objects") {
                    slider($sceneSpacing, 1.5...6, step: 0.25, label: String(format: "%.2f units", sceneSpacing))
                }
            }
            Section("Fast shape (TripoSR)") {
                LabeledContent("Surface threshold") { slider($tsrThreshold, 5...80, step: 1, label: String(format: "%.0f", tsrThreshold)) }
                LabeledContent("Subject size in frame") { slider($tsrForegroundRatio, 0.5...1, step: 0.05, label: String(format: "%.0f%%", tsrForegroundRatio * 100)) }
                Toggle("Drop floating specks (keep the largest piece)", isOn: $tsrKeepLargest)
                Text("Pick Fast in the shape model menu. Higher threshold = thinner surface; mesh detail follows the quality slider. Fast makes the shape only; paint it afterwards.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Depth relief") {
                LabeledContent("Relief height") { slider($reliefDepth, 0.05...0.6, step: 0.05, label: String(format: "%.0f%% of width", reliefDepth * 100)) }
                LabeledContent("Detail") { slider($reliefResolution, 96...512, step: 32, label: "\(Int(reliefResolution)) samples") }
                LabeledContent("Smoothing") { slider($reliefSmoothing, 0...8, step: 1, label: "\(Int(reliefSmoothing)) passes") }
                Text("Estimates depth from the image (Depth Anything V2) and extrudes a solid with a flat back — a quick 2.5D shape.")
                    .font(.caption).foregroundStyle(.secondary)
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
            Section("Flow field") {
                LabeledContent("Crease angle") { slider($flowFeatureAngle, 10...80, step: 5, label: "\(Int(flowFeatureAngle))°") }
                LabeledContent("Guide strength") { slider($flowGuideStrength, 5...100, step: 5, label: "\(Int(flowGuideStrength))") }
                LabeledContent("Flow remesh quads") { slider($flowTargetQuads, 500...8000, step: 500, label: "\(Int(flowTargetQuads))") }
                Text("Edges sharper than this are treated as creases the flow lines follow. Used by the flow field view and flow remesh. Guide strength is how hard drawn guide curves pull the flow compared with creases.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("3D printing") {
                LabeledContent("3MF longest side") { slider($printSizeMM, 0...400, step: 5, label: printSizeMM == 0 ? "as is" : "\(Int(printSizeMM)) mm") }
                Picker("Slicer", selection: $slicerChoice) {
                    Text("First installed").tag("auto")
                    ForEach(Slicer.allCases) { Text($0.title + ($0.appURL == nil ? " (not installed)" : "")).tag($0.rawValue) }
                }
                Text("3MF export scales the model so its longest side is this many millimetres and sets it on the bed (0 keeps the model's own units). \"Open in slicer\" in the export menu sends that 3MF to the chosen slicer.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Batch and plugins") {
                Toggle("Batch also paints textures", isOn: $batchPaint)
                LabeledContent("Plugins") {
                    Button("Open plugins folder (\(ToolPlugin.load().count) installed)") {
                        try? FileManager.default.createDirectory(at: ToolPlugin.folder, withIntermediateDirectories: true)
                        NSWorkspace.shared.open(ToolPlugin.folder)
                    }
                }
                Text("Generate ▸ Batch Generate from Images makes one project per image and runs them one at a time. A plugin is a .json file in the plugins folder naming a command to run on the exported model (Model ▸ Tools); see docs/PLUGINS.md.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Material regions") {
                LabeledContent("Color tolerance") { slider($regionTolerance, 0.02...0.5, step: 0.01, label: String(format: "%.0f%%", regionTolerance * 100)) }
                LabeledContent("Edge softness") { slider($regionFeather, 0.01...0.4, step: 0.01, label: String(format: "%.0f%%", regionFeather * 100)) }
                LabeledContent("UV island padding") { slider($islandPadding, 0...8, step: 1, label: "\(Int(islandPadding)) px") }
                Text("Starting values when limiting a material edit to one color of the texture. Island padding grows a selected UV island past its edge so seams don't show a halo.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("AI Fill") {
                TextField("Default prompt", text: $aiFillPrompt, axis: .vertical).lineLimit(2...3)
                LabeledContent("Steps") { slider($aiFillSteps, 10...50, step: 1, label: "\(Int(aiFillSteps))") }
                LabeledContent("Strength") { slider($aiFillStrength, 0.3...1, step: 0.05, label: String(format: "%.0f%%", aiFillStrength * 100)) }
                LabeledContent("Prompt guidance") { slider($aiFillGuidance, 1...12, step: 0.5, label: String(format: "%.1f", aiFillGuidance)) }
                Text("The texture's AI Fill tool regenerates the marked area with Stable Diffusion 2.1 (the Text to 3D model). Lower strength stays closer to what is there.")
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
        .frame(height: 960)
    }

    private func slider(_ value: Binding<Double>, _ range: ClosedRange<Double>, step: Double, label: String) -> some View {
        HStack {
            Slider(value: value, in: range, step: step).frame(width: 170)
            Text(label).font(.caption).foregroundStyle(.secondary).monospacedDigit().frame(width: 100, alignment: .trailing)
        }
    }

    private func reset() {
        decimateK = ToolDefaults.fallbackDecimateK; smoothIterations = ToolDefaults.fallbackSmoothIterations
        booleanSharp = true; holeEdges = ToolDefaults.fallbackHoleEdges; booleanResolution = ToolDefaults.fallbackBooleanResolution
        remeshResolution = ToolDefaults.fallbackRemeshResolution; symmetryAxis = ToolDefaults.fallbackSymmetryAxis
        keepPositive = true; lodLevels = ToolDefaults.fallbackLODLevels
        sculptSize = ToolDefaults.fallbackSculptSize; sculptStrength = ToolDefaults.fallbackSculptStrength
        paintSize = ToolDefaults.fallbackPaintSize; paintHardness = ToolDefaults.fallbackPaintHardness
        paintOpacity = ToolDefaults.fallbackPaintOpacity; sceneSpacing = ToolDefaults.fallbackSceneSpacing
        exportQuads = false
        aiFillPrompt = ToolDefaults.fallbackAIFillPrompt; aiFillSteps = ToolDefaults.fallbackAIFillSteps
        aiFillStrength = ToolDefaults.fallbackAIFillStrength; aiFillGuidance = ToolDefaults.fallbackAIFillGuidance
        flowFeatureAngle = ToolDefaults.fallbackFlowFeatureAngle
        flowTargetQuads = ToolDefaults.fallbackFlowTargetQuads
        flowGuideStrength = ToolDefaults.fallbackFlowGuideStrength
        regionTolerance = ToolDefaults.fallbackRegionTolerance; regionFeather = ToolDefaults.fallbackRegionFeather
        islandPadding = ToolDefaults.fallbackIslandPadding
        printSizeMM = ToolDefaults.fallbackPrintSizeMM; slicerChoice = ToolDefaults.fallbackSlicer
        batchPaint = false
        tsrThreshold = ToolDefaults.fallbackTSRThreshold; tsrForegroundRatio = ToolDefaults.fallbackTSRForegroundRatio
        tsrKeepLargest = true
        reliefDepth = ToolDefaults.fallbackReliefDepth; reliefResolution = ToolDefaults.fallbackReliefResolution
        reliefSmoothing = ToolDefaults.fallbackReliefSmoothing
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
