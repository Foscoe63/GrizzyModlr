import SwiftUI
import UniformTypeIdentifiers

/// Optional extra reference views (left / back / right) beside the main front image.
/// With any extra view, generation uses the multi-view model.
struct ReferenceViewsStrip: View {
    @Environment(ProjectStore.self) private var store
    @Environment(AppRuntime.self) private var runtime
    let projectID: Project.ID
    var reloadKey: Int

    @State private var importing: ReferenceView?
    @State private var targeted: ReferenceView?
    @State private var showSynth = false
    @State private var showTextPanel = false

    private var project: Project? { store.project(projectID) }

    var body: some View {
        VStack(spacing: 6) {
            HStack(spacing: 10) {
                ForEach(ReferenceView.allCases) { view in slot(view) }
            }
            synthesisRow
            Text(project?.extraViews.isEmpty == false
                 ? "Multi-view model will be used"
                 : "Optional: add more angles for a more accurate back side")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .sheet(isPresented: $showSynth) {
            SynthesizedViewsSheet(projectID: projectID, onClose: { showSynth = false })
                .environment(store)
        }
        .fileImporter(isPresented: Binding(get: { importing != nil }, set: { if !$0 { importing = nil } }),
                      allowedContentTypes: [.image]) { result in
            guard let view = importing, case .success(let url) = result else { return }
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            store.setExtraView(view, fromURL: url, for: projectID)
        }
    }

    private func slot(_ view: ReferenceView) -> some View {
        let url = project.flatMap { store.extraViewURL(view, for: $0) }
        return VStack(spacing: 3) {
            ZStack(alignment: .topTrailing) {
                Button { importing = view } label: {
                    ZStack {
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(Color.black.opacity(0.18))
                        if let url, let img = NSImage(contentsOf: url) {
                            Image(nsImage: img).resizable().scaledToFit().padding(3)
                        } else {
                            Image(systemName: "plus").foregroundStyle(.secondary)
                        }
                    }
                    .frame(width: 54, height: 54)
                    .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .strokeBorder(targeted == view ? Color.accentColor : .white.opacity(0.1),
                                      lineWidth: targeted == view ? 2 : 1))
                }
                .buttonStyle(.plain)
                .help(url == nil ? "Add the \(view.label.lowercased()) view" : "Replace the \(view.label.lowercased()) view")
                if url != nil {
                    Button { store.removeExtraView(view, for: projectID) } label: {
                        Image(systemName: "xmark.circle.fill").font(.system(size: 14))
                            .symbolRenderingMode(.palette).foregroundStyle(.white, .black.opacity(0.6))
                    }
                    .buttonStyle(.plain)
                    .offset(x: 5, y: -5)
                    .help("Remove")
                }
            }
            Text(view.label).font(.caption2).foregroundStyle(.secondary)
        }
        .id("\(view.rawValue)#\(reloadKey)")
        .onDrop(of: [.fileURL], isTargeted: Binding(get: { targeted == view }, set: { targeted = $0 ? view : nil })) { providers in
            guard let p = providers.first else { return false }
            p.loadItem(forTypeIdentifier: UTType.fileURL.identifier) { item, _ in
                var u: URL?
                if let d = item as? Data { u = URL(dataRepresentation: d, relativeTo: nil) } else if let x = item as? URL { u = x }
                if let u, UTType(filenameExtension: u.pathExtension)?.conforms(to: .image) == true {
                    DispatchQueue.main.async { store.setExtraView(view, fromURL: u, for: projectID) }
                }
            }
            return true
        }
    }
}

extension ReferenceViewsStrip {
    private var hasCandidates: Bool { !store.synthesizedViews(for: projectID).isEmpty }

    @ViewBuilder
    var synthesisRow: some View {
        HStack(spacing: 8) {
            if case .running? = runtime.relief[projectID] {
                ProgressView().controlSize(.small)
                Text("Building relief…").font(.caption).foregroundStyle(.secondary)
                Button("Cancel") { runtime.cancelRelief(projectID) }.controlSize(.small)
            } else if case .failed(let message)? = runtime.relief[projectID] {
                Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange)
                Text(message).font(.caption).lineLimit(2)
                Button { runtime.dismissReliefFailure(projectID) } label: { Image(systemName: "xmark") }.buttonStyle(.plain)
            }
            switch runtime.viewSynth[projectID] {
            case .running(let p):
                ProgressView(value: p).frame(width: 90)
                Text("Synthesizing…").font(.caption).foregroundStyle(.secondary)
                Button("Cancel") { runtime.cancelViewSynthesis(projectID) }.controlSize(.small)
            case .failed(let message):
                Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange)
                Text(message).font(.caption).lineLimit(2)
                Button { runtime.dismissViewSynthFailure(projectID) } label: { Image(systemName: "xmark") }
                    .buttonStyle(.plain)
            case nil:
                Button { runtime.synthesizeViews(projectID) } label: {
                    Label("Synthesize views", systemImage: "sparkles")
                }
                .controlSize(.small)
                .help(ModelStore.isInstalled(.viewSynthesis)
                      ? "Generate six candidate side views from the main image"
                      : "Installs the Zero123++ view-synthesis model (5.6 GB, non-commercial license)")
                if hasCandidates {
                    Button { showSynth = true } label: { Label("Review", systemImage: "square.grid.3x2") }
                        .controlSize(.small)
                }
                Button { runtime.createRelief(for: projectID) } label: { Label("Depth relief", systemImage: "mountain.2") }
                    .controlSize(.small)
                    .disabled(runtime.relief[projectID] != nil)
                    .help(ModelStore.isInstalled(.depthEstimation)
                          ? "Build a relief solid from the image's estimated depth"
                          : "Installs the Depth Anything model (99 MB), then builds a relief solid from the image")
                Button { showTextPanel = true } label: { Label("From text", systemImage: "text.cursor") }
                    .controlSize(.small)
                    .help("Replace the main image with one generated from a description")
                    .popover(isPresented: $showTextPanel) {
                        TextToImagePanel(projectID: projectID, compact: true)
                            .environment(runtime).padding(14).frame(width: 400)
                    }
            }
        }
        .onChange(of: runtime.viewSynthRevision[projectID] ?? 0) { _, _ in showSynth = true }
    }
}

/// The six synthesized candidates; each can be assigned to the left, back or right slot.
struct SynthesizedViewsSheet: View {
    @Environment(ProjectStore.self) private var store
    let projectID: Project.ID
    var onClose: () -> Void

    var body: some View {
        let views = store.synthesizedViews(for: projectID)
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Synthesized views").font(.headline)
                Spacer()
                Button("Done", action: onClose)
            }
            Text("Pick the ones that best match the left, back and right of your object. Angles are relative to the main image.")
                .font(.caption).foregroundStyle(.secondary)
            LazyVGrid(columns: [GridItem(.fixed(180)), GridItem(.fixed(180))], spacing: 12) {
                ForEach(0 ..< 6, id: \.self) { i in
                    VStack(spacing: 4) {
                        ZStack {
                            RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.2))
                            if let url = views[i], let img = NSImage(contentsOf: url) {
                                Image(nsImage: img).resizable().scaledToFit()
                            }
                        }
                        .frame(width: 180, height: 180)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        HStack {
                            let cam = ViewSynthEngine.cameras[i]
                            Text("\(cam.azimuth)° · \(cam.elevation > 0 ? "+" : "")\(cam.elevation)°")
                                .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                            Spacer()
                            if let url = views[i] {
                                Menu("Use as") {
                                    ForEach(ReferenceView.allCases) { v in
                                        Button(v.label) { store.setExtraView(v, fromURL: url, for: projectID) }
                                    }
                                }
                                .menuStyle(.borderlessButton).fixedSize()
                            }
                        }
                        .frame(width: 180)
                    }
                }
            }
            Text("Synthesized with Zero123++ (weights CC-BY-NC 4.0, non-commercial use).")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(18)
    }
}
