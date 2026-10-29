import SwiftUI
import UniformTypeIdentifiers

/// Optional extra reference views (left / back / right) beside the main front image.
/// With any extra view, generation uses the multi-view model.
struct ReferenceViewsStrip: View {
    @Environment(ProjectStore.self) private var store
    let projectID: Project.ID
    var reloadKey: Int

    @State private var importing: ReferenceView?
    @State private var targeted: ReferenceView?

    private var project: Project? { store.project(projectID) }

    var body: some View {
        VStack(spacing: 6) {
            HStack(spacing: 10) {
                ForEach(ReferenceView.allCases) { view in slot(view) }
            }
            Text(project?.extraViews.isEmpty == false
                 ? "Multi-view model will be used"
                 : "Optional: add more angles for a more accurate back side")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
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
