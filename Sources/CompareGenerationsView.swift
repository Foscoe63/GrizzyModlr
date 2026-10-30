import SwiftUI

/// Two saved versions side by side, each in its own orbitable viewer, with the settings that made
/// them — to judge whether more steps or a finer grid actually helped.
struct CompareGenerationsView: View {
    @Environment(ProjectStore.self) private var store
    let projectID: Project.ID
    var onClose: () -> Void

    @State private var leftID: UUID?
    @State private var rightID: UUID?

    private var generations: [Generation] { store.project(projectID)?.generations ?? [] }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Compare versions").font(.headline)
                Spacer()
                Button("Done", action: onClose).keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 16).padding(.vertical, 12)
            Divider()
            HStack(spacing: 0) {
                side(selection: $leftID)
                Divider()
                side(selection: $rightID)
            }
        }
        .frame(width: 1000, height: 640)
        .onAppear {
            let g = generations
            let current = store.project(projectID)?.currentGeneration?.id ?? g.last?.id
            rightID = rightID ?? current
            leftID = leftID ?? g.last(where: { $0.id != current })?.id ?? g.first?.id
        }
    }

    private func side(selection: Binding<UUID?>) -> some View {
        let gen = generations.first { $0.id == selection.wrappedValue }
        return VStack(spacing: 0) {
            Picker("Version", selection: selection) {
                ForEach(Array(generations.enumerated()), id: \.element.id) { index, g in
                    Text(Self.title(g, number: index + 1)).tag(Optional(g.id))
                }
            }
            .labelsHidden()
            .padding(10)
            ZStack {
                if let gen, let content = store.viewerContent(for: gen, in: projectID) {
                    MeshViewer(content: content)
                } else {
                    Text("No mesh for this version").foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.black.opacity(0.12))
            if let gen {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(details(gen), id: \.self) { Text($0).font(.caption).foregroundStyle(.secondary) }
                    Button("Use this version") {
                        store.selectGeneration(gen.id, for: projectID)
                        onClose()
                    }
                    .controlSize(.small).padding(.top, 4)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
            }
        }
        .frame(maxWidth: .infinity)
    }

    static func title(_ g: Generation, number: Int) -> String {
        let kind = g.kind == .paint ? "Paint" : "Shape"
        return "#\(number) · \(kind) · \(g.shapeModel.label) · \(g.createdAt.formatted(date: .abbreviated, time: .shortened))"
    }

    private func details(_ g: Generation) -> [String] {
        var lines: [String] = []
        if g.kind == .paint {
            lines.append("Texture \(g.paintTexRaw ?? 0) px · \(g.paintStepsRaw ?? 0) steps · \(g.paintFacesRaw.map { "\($0 / 1000)k tris" } ?? "")")
        } else {
            lines.append("\(g.steps) steps · grid \(g.octreeRaw ?? 256) · guidance \(String(format: "%.1f", g.guidanceRaw ?? 5))")
        }
        if case .mesh(let url)? = store.viewerContent(for: g, in: projectID), let faces = ViewLOD.faceCount(of: url) {
            var line = "\(faces.formatted()) triangles"
            if faces > ViewLOD.threshold { line += " (shown simplified)" }
            lines.append(line)
        }
        if let d = g.durationSeconds { lines.append(String(format: "Took %.0f s", d)) }
        return lines
    }
}
