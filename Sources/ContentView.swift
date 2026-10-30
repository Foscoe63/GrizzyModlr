import SwiftUI
import AppKit
import ImageIO

struct ContentView: View {
    @Environment(AppRuntime.self) private var runtime
    @Environment(ProjectStore.self) private var store
    @Environment(\.openSettings) private var openSettings
    @State private var templateNameFor: Project.ID?
    @State private var templateName = ""

    var body: some View {
        @Bindable var store = store
        NavigationSplitView {
            List(selection: $store.selection) {
                ForEach(store.projects) { project in
                    ProjectRow(project: project)
                        .tag(project.id)
                        .contextMenu {
                            Button("Delete", role: .destructive) {
                                runtime.deleteProject(project.id)
                            }
                        }
                }
            }
            .alert("Save as Template", isPresented: Binding(get: { templateNameFor != nil }, set: { if !$0 { templateNameFor = nil } })) {
                TextField("Template name", text: $templateName)
                Button("Save") {
                    if let id = templateNameFor { store.saveTemplate(from: id, named: templateName) }
                    templateName = ""; templateNameFor = nil
                }
                Button("Cancel", role: .cancel) { templateName = ""; templateNameFor = nil }
            } message: { Text("Keeps this project's quality, paint quality and background setting for new projects.") }
            .navigationTitle("Projects")
            .navigationSplitViewColumnWidth(min: 180, ideal: 220)
            .toolbar {
                ToolbarItem {
                    Menu {
                        ForEach(ProjectTemplate.all()) { t in
                            Button { store.newProject(template: t) } label: { Label(t.name, systemImage: t.icon) }
                                .help(t.detail)
                        }
                        if let id = store.selection {
                            Divider()
                            Button("Save “\(store.project(id)?.name ?? "")” Settings as Template…") { templateNameFor = id }
                        }
                    } label: {
                        Label("New Project", systemImage: "plus")
                    } primaryAction: {
                        store.newProject()
                    }
                    .help("New Project (click the arrow for templates)")
                }
            }
            .overlay {
                if store.projects.isEmpty {
                    ContentUnavailableView {
                        Label("No Projects", systemImage: "cube")
                    } description: {
                        Text("Click + to start.")
                    }
                }
            }
        } detail: {
            if let id = store.selection,
               let project = store.projects.first(where: { $0.id == id }) {
                ProjectDetailView(project: project)
                    .id(project.id)
            } else {
                ContentUnavailableView {
                    Label("No Project Selected", systemImage: "cube.transparent")
                } description: {
                    Text("Create a project, then drag in an image.")
                }
            }
        }
        .sheet(isPresented: onboardingPresented) {
            OnboardingView()
                .environment(runtime)
                .interactiveDismissDisabled()
        }
        .onChange(of: runtime.modelManagerSignal) { _, _ in
            openSettings()                       // generation CTA without weights (§4.9)
        }
    }

    /// First-run sheet visibility mirrors §4.1: shown while phase == onboarding.
    private var onboardingPresented: Binding<Bool> {
        Binding(get: { runtime.state.phase == .onboarding },
                set: { shown in
                    if !shown && runtime.state.phase == .onboarding {
                        runtime.dispatch(.onboardingSkipped)   // Esc/close = Later
                    }
                })
    }
}

/// A row in the sidebar: a squircle thumbnail of the project's image + name/status.
struct ProjectRow: View {
    @Environment(AppRuntime.self) private var runtime
    @Environment(ProjectStore.self) private var store
    let project: Project
    @State private var thumb: NSImage?

    private var thumbKey: String {
        "\(store.imageURL(for: project)?.path ?? "none")#\(store.inputVersion(project.id))"
    }

    var body: some View {
        HStack(spacing: 10) {
            thumbnail
            VStack(alignment: .leading, spacing: 1) {
                Text(project.name)
                    .lineLimit(1)
                if let running = runningLine {
                    Text(running)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                } else {
                    Text(subtitle)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }
            }
        }
        .padding(.vertical, 2)
        .task(id: thumbKey) {
            let key = thumbKey as NSString
            if let cached = Self.cache.object(forKey: key) { thumb = cached; return }
            let url = store.imageURL(for: project)
            let img = await Task.detached(priority: .utility) { Self.loadThumb(url) }.value
            if let img { Self.cache.setObject(img, forKey: key) }
            if !Task.isCancelled { thumb = img }
        }
    }

    /// Shared, auto-evicting thumbnail cache so scrolling never re-decodes.
    private static let cache = NSCache<NSString, NSImage>()

    private var thumbnail: some View {
        ZStack(alignment: .bottomTrailing) {
            Group {
                if let thumb {
                    Image(nsImage: thumb).resizable().scaledToFill()
                } else {
                    Image(systemName: "photo")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                }
            }
            .frame(width: 28, height: 28)
            .background(Color.primary.opacity(0.06))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(.white.opacity(0.08))
            )

            if let dot = statusDot {
                Circle()
                    .fill(dot)
                    .frame(width: 8, height: 8)
                    .overlay(Circle().strokeBorder(Color(NSColor.controlBackgroundColor), lineWidth: 1.5))
                    .offset(x: 3, y: 3)
            }
        }
    }

    /// Status text while a job runs — the §4.4/§4.5 stage names, 1:1.
    private var runningLine: String? {
        let shape = runtime.state.shapeState(project.id)
        if shape.isRunning {
            if case .denoising(let k, let n) = shape, n > 0 {
                return "Denoising \(k)/\(n)"
            }
            return AppReducer.shapeStageLabel(shape)
        }
        let paint = runtime.state.paintState(project.id)
        if paint.isRunning {
            if case .denoising(let k, let n) = paint, n > 0 {
                return "Painting \(k)/\(n)"
            }
            return AppReducer.paintStageLabel(paint)
        }
        return nil
    }

    private var statusDot: Color? {
        let shape = runtime.state.shapeState(project.id)
        let paint = runtime.state.paintState(project.id)
        if shape.isRunning || paint.isRunning { return .accentColor }
        if case .failed = shape { return .yellow }
        if case .failed = paint { return .yellow }
        return nil
    }

    private var subtitle: String {
        var s = project.shapeModel.label
        if project.quantization != .full { s += " · \(project.quantization.label)" }
        return s
    }

    /// Efficient downsampled thumbnail (decodes to ~72px, not the full image).
    /// nonisolated so it can run on the detached decode task without hopping back.
    private nonisolated static func loadThumb(_ url: URL?) -> NSImage? {
        guard let url, FileManager.default.fileExists(atPath: url.path),
              let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: 72,
            kCGImageSourceCreateThumbnailWithTransform: true,
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) else { return nil }
        return NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
    }
}
