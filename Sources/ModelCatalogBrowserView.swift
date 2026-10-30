import SwiftUI
import UniformTypeIdentifiers

/// Settings → Catalog: look through every model the app can install — search by name, use, license
/// or repo, filter by kind or install state — and install or remove with the same controls as the
/// Models tab.
struct ModelCatalogBrowserView: View {
    @Environment(AppRuntime.self) private var runtime
    @State private var query = ""
    @State private var filter: Filter = .all
    @State private var importTarget: ModelID?

    enum Filter: String, CaseIterable, Identifiable {
        case all, shape, paint, tools, installed, available
        var id: String { rawValue }
        var label: String {
            switch self {
            case .all: return "All"
            case .shape: return "Shape"
            case .paint: return "Paint"
            case .tools: return "Tools"
            case .installed: return "Installed"
            case .available: return "Not installed"
            }
        }
        var category: ModelCategory? {
            switch self {
            case .shape: return .shape
            case .paint: return .paint
            case .tools: return .tools
            default: return nil
            }
        }
    }

    private var results: [ModelID] {
        ModelCatalog.search(query, category: filter.category).filter { id in
            let installed = runtime.state.installState(id).isInstalled
            switch filter {
            case .installed: return installed
            case .available: return !installed
            default: return true
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            VStack(spacing: 8) {
                HStack(spacing: 6) {
                    Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                    TextField("Search models, uses, licenses…", text: $query).textFieldStyle(.plain)
                    if !query.isEmpty {
                        Button { query = "" } label: { Image(systemName: "xmark.circle.fill") }
                            .buttonStyle(.plain).foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal, 10).padding(.vertical, 7)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                Picker("Show", selection: $filter) {
                    ForEach(Filter.allCases) { Text($0.label).tag($0) }
                }
                .pickerStyle(.segmented).labelsHidden()
            }
            .padding(.horizontal, 16).padding(.top, 14).padding(.bottom, 10)
            Divider()
            if results.isEmpty {
                ContentUnavailableView("No matching models", systemImage: "magnifyingglass",
                                       description: Text("Try a different word or filter."))
                    .frame(maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(results, id: \.self) { id in
                            VStack(alignment: .leading, spacing: 0) {
                                ModelRow(model: id, importTarget: $importTarget)
                                info(id)
                            }
                            Divider().padding(.leading, 16)
                        }
                    }
                }
            }
            Divider()
            HStack {
                Text("\(results.count) of \(ModelID.allCases.count) models")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text("\(OnboardingView.gb(DiskSpace.free())) free").font(.caption).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 16).padding(.vertical, 8)
        }
        .frame(height: 620)
        .fileImporter(isPresented: Binding(get: { importTarget != nil }, set: { if !$0 { importTarget = nil } }),
                      allowedContentTypes: [.folder]) { result in
            if let target = importTarget, case .success(let url) = result { runtime.importWeights(target, from: url) }
            importTarget = nil
        }
    }

    private func info(_ id: ModelID) -> some View {
        let i = id.info
        return VStack(alignment: .leading, spacing: 3) {
            Text("Used for: \(i.usedFor)").font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 10) {
                Text("License: \(i.license)")
                Link("Hugging Face page", destination: ModelCatalog.model(id).pageURL)
            }
            .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 20).padding(.bottom, 12)
    }
}
