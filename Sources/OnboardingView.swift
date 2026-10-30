import SwiftUI

/// First-run sheet (DESIGN.md §4.2/§5): welcome → three choice cards with a live
/// free-disk check → download progress with per-file detail → done. "Later" is
/// available at every step; skipping never blocks the app.
struct OnboardingView: View {
    @Environment(AppRuntime.self) private var runtime
    @State private var choice: Choice = .fastStart

    enum Choice: CaseIterable, Identifiable {
        case fastStart, bestQuality, everything
        var id: Self { self }

        var title: String {
            switch self {
            case .fastStart: return "Fast start"
            case .bestQuality: return "Best quality"
            case .everything: return "Everything"
            }
        }
        var subtitle: String {
            switch self {
            case .fastStart: return "Small shape + color texture"
            case .bestQuality: return "Large shape + PBR texture"
            case .everything: return "All four models"
            }
        }
        var icon: String {
            switch self {
            case .fastStart: return "hare"
            case .bestQuality: return "sparkles"
            case .everything: return "square.stack.3d.up"
            }
        }
        var models: Set<ModelID> {
            switch self {
            case .fastStart: return ModelCatalog.fastStart
            case .bestQuality: return ModelCatalog.bestQuality
            case .everything: return ModelCatalog.everything
            }
        }
        var bytes: Int64 { ModelCatalog.totalBytes(of: models) }
    }

    var body: some View {
        VStack(spacing: 0) {
            switch runtime.state.onboarding.step {
            case .welcome: welcome
            case .chooseModels: chooseModels
            case .downloading: downloading
            case .failed(let message): failed(message)
            case .done: done
            }
        }
        .frame(width: 540)
        .animation(.easeInOut(duration: 0.2), value: runtime.state.onboarding.step)
    }

    // MARK: welcome

    private var welcome: some View {
        VStack(spacing: 14) {
            Image(systemName: "cube.transparent")
                .font(.system(size: 52, weight: .light))
                .foregroundStyle(.tint)
                .padding(.top, 40)
            Text("Welcome to GrizzyModlr")
                .font(.title.weight(.semibold))
            Text("Image to 3D, on this Mac.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Spacer().frame(height: 16)
            Button("Continue") { runtime.dispatch(.onboardingAdvanced) }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .keyboardShortcut(.defaultAction)
            laterButton
                .padding(.bottom, 24)
        }
        .padding(.horizontal, 40)
    }

    // MARK: choose models

    private var chooseModels: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Choose your models")
                .font(.title2.weight(.semibold))
                .padding(.top, 28)
            Text("Add or remove anytime in Settings → Models.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            ForEach(Choice.allCases) { c in
                card(c)
            }

            TimelineView(.periodic(from: .now, by: 2)) { _ in
                diskFooter
            }

            HStack {
                laterButton
                Spacer()
                Button("Download") { runtime.dispatch(.onboardingStartDownload(choice.models)) }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .keyboardShortcut(.defaultAction)
                    .disabled(!fits(choice))
            }
            .padding(.bottom, 20)
        }
        .padding(.horizontal, 28)
    }

    private func card(_ c: Choice) -> some View {
        let selected = choice == c
        return Button {
            choice = c
        } label: {
            HStack(spacing: 14) {
                Image(systemName: c.icon)
                    .font(.title2)
                    .frame(width: 34)
                    .foregroundStyle(selected ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
                VStack(alignment: .leading, spacing: 2) {
                    Text(c.title).font(.headline)
                    Text(c.subtitle).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text(Self.gb(c.bytes)).font(.callout.weight(.medium)).monospacedDigit()
                    if !fits(c) {
                        Text("Not enough space").font(.caption2).foregroundStyle(.orange)
                    }
                }
            }
            .padding(14)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(selected ? Color.accentColor.opacity(0.12) : Color.primary.opacity(0.04))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(selected ? Color.accentColor : Color.primary.opacity(0.08),
                                  lineWidth: selected ? 1.5 : 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
        .buttonStyle(.plain)
    }

    private var diskFooter: some View {
        HStack(spacing: 6) {
            Image(systemName: "internaldrive")
                .foregroundStyle(.secondary)
            Text("\(Self.gb(DiskSpace.free())) free on this Mac")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func fits(_ c: Choice) -> Bool {
        // Leave 2 GB of headroom beyond the download itself.
        DiskSpace.free() > c.bytes + (2 << 30)
    }

    // MARK: downloading

    private var downloading: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Downloading models")
                .font(.title2.weight(.semibold))
                .padding(.top, 28)

            ForEach(selectedModels, id: \.self) { model in
                downloadRow(model)
            }

            HStack {
                Button("Later") { runtime.dispatch(.onboardingSkipped) }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .help("Keeps downloading in the background")
                Spacer()
            }
            .padding(.bottom, 20)
        }
        .padding(.horizontal, 28)
    }

    private var selectedModels: [ModelID] {
        ModelID.allCases.filter { runtime.state.onboarding.selection.contains($0) }
    }

    @ViewBuilder
    private func downloadRow(_ model: ModelID) -> some View {
        let cat = ModelCatalog.model(model)
        let state = runtime.state.installState(model)
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(cat.displayName).font(.callout.weight(.medium))
                Spacer()
                Text(rowStatus(state)).font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            if case .downloading(let p) = state {
                ProgressView(value: p.fraction)
                    .progressViewStyle(.linear)
                Text("File \(p.fileIndex) of \(p.fileCount) — \(p.currentFileName) · \(Self.gb(p.fileBytes)) of \(Self.gb(p.fileTotal))")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            } else {
                ProgressView(value: state == .installed ? 1 : 0)
                    .progressViewStyle(.linear)
                    .tint(state == .installed ? .green : nil)
            }
        }
        .padding(12)
        .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private func rowStatus(_ state: ModelInstallState) -> String {
        switch state {
        case .notInstalled: return "Waiting"
        case .queued: return "Queued"
        case .downloading(let p): return "\(Int(p.fraction * 100))%"
        case .paused: return "Paused"
        case .verifying: return "Verifying…"
        case .installed: return "Ready"
        case .failed: return "Failed"
        }
    }

    // MARK: failed

    private func failed(_ message: String) -> some View {
        VStack(spacing: 14) {
            Image(systemName: "wifi.exclamationmark")
                .font(.system(size: 44, weight: .light))
                .foregroundStyle(.orange)
                .padding(.top, 40)
            Text("Download interrupted")
                .font(.title2.weight(.semibold))
            Text(message)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Text("Retrying resumes where it stopped.")
                .font(.caption)
                .foregroundStyle(.tertiary)
            Spacer().frame(height: 10)
            Button("Retry") { runtime.dispatch(.onboardingRetried) }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .keyboardShortcut(.defaultAction)
            Button("Skip for now") { runtime.dispatch(.onboardingSkipped) }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .padding(.bottom, 24)
        }
        .padding(.horizontal, 40)
    }

    // MARK: done

    private var done: some View {
        VStack(spacing: 14) {
            Image(systemName: "checkmark.circle")
                .font(.system(size: 52, weight: .light))
                .foregroundStyle(.green)
                .padding(.top, 40)
            Text("You're ready")
                .font(.title.weight(.semibold))
            Text("Drop an image into a project and press Generate.")
                .font(.callout)
                .foregroundStyle(.secondary)
            Spacer().frame(height: 16)
            Button("Start using GrizzyModlr") { runtime.dispatch(.onboardingFinished) }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .keyboardShortcut(.defaultAction)
                .padding(.bottom, 32)
        }
        .padding(.horizontal, 40)
    }

    private var laterButton: some View {
        Button("Later") { runtime.dispatch(.onboardingSkipped) }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
    }

    static func gb(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}

/// Free space on the volume that holds the model library.
enum DiskSpace {
    static func free() -> Int64 {
        let values = try? ModelStore.modelsRoot.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage ?? 0
    }
}
