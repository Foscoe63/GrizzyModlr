import SwiftUI
import UniformTypeIdentifiers

/// Settings window: currently the single Models tab (§5 model manager).
struct SettingsView: View {
    var body: some View {
        TabView {
            ModelManagerView()
                .tabItem { Label("Models", systemImage: "square.stack.3d.up") }
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
