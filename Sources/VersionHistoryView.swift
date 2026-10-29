import SwiftUI

/// Version history popover: every saved generation, newest first. Click to load it
/// into the viewer; right-click or the trash button to delete.
struct VersionHistoryView: View {
    @Environment(ProjectStore.self) private var store
    let projectID: Project.ID
    var onClose: () -> Void = {}

    private var project: Project? { store.project(projectID) }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Version History").font(.headline)
                Spacer()
                if let project {
                    Text("\(project.generations.count)")
                        .font(.callout).foregroundStyle(.secondary)
                        .monospacedDigit()
                }
            }
            .padding(.horizontal, 14).padding(.vertical, 11)

            Divider()

            if let project, !project.generations.isEmpty {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(Array(project.generations.reversed())) { gen in
                            row(gen, in: project)
                            Divider().opacity(0.5)
                        }
                    }
                }
            } else {
                Text("No generations yet")
                    .font(.callout).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity).padding(.vertical, 40)
            }
        }
        .frame(width: 340, height: 430)
    }

    private func row(_ gen: Generation, in project: Project) -> some View {
        let isSelected = gen.id == project.currentGeneration?.id
        return HStack(spacing: 11) {
            thumb(gen)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 5) {
                    if gen.kind == .paint {
                        Image(systemName: "paintbrush.fill").font(.caption2).foregroundStyle(.purple)
                    }
                    Text(gen.shapeModel.label).font(.callout.weight(.medium))
                    if gen.isPBR {
                        Text("PBR")
                            .font(.system(size: 9, weight: .bold))
                            .padding(.horizontal, 4).padding(.vertical, 1)
                            .background(Color.purple.opacity(0.22), in: RoundedRectangle(cornerRadius: 4, style: .continuous))
                            .foregroundStyle(.purple)
                    }
                }
                HStack(spacing: 5) {
                    Text(metaLine(gen))
                    if gen.removeBackground { Text("· cut out") }
                }
                .font(.caption2).foregroundStyle(.secondary)
                Text(subtitle(gen)).font(.caption2).foregroundStyle(.tertiary)
            }
            Spacer(minLength: 4)
            if isSelected {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.tint)
            }
            Button {
                store.restoreGeneration(gen.id, for: projectID)
                onClose()
            } label: {
                Image(systemName: "pencil").foregroundStyle(.secondary)
            }
            .buttonStyle(.borderless)
            .help("Load this version's image to edit & regenerate")
            Button {
                store.deleteGeneration(gen.id, for: projectID)
            } label: {
                Image(systemName: "trash").foregroundStyle(.secondary)
            }
            .buttonStyle(.borderless)
            .help("Delete this version")
        }
        .padding(.horizontal, 12).padding(.vertical, 9)
        .background(isSelected ? Color.accentColor.opacity(0.14) : Color.clear)
        .contentShape(Rectangle())
        .onTapGesture { store.selectGeneration(gen.id, for: projectID) }
    }

    private func thumb(_ gen: Generation) -> some View {
        Group {
            if let url = store.inputURL(for: gen, in: projectID), let img = NSImage(contentsOf: url) {
                Image(nsImage: img).resizable().scaledToFill()
            } else {
                Image(systemName: "cube.transparent").font(.title3).foregroundStyle(.secondary)
            }
        }
        .frame(width: 46, height: 46)
        .background(Color.black.opacity(0.18))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(.white.opacity(0.08)))
    }

    /// Metadata line; the default Full quant is omitted (it's the boring case).
    private func metaLine(_ gen: Generation) -> String {
        if gen.kind == .paint {
            if let edit = gen.editSummary { return "Edited · \(edit)" }
            return "Textured · \(gen.paintStepsRaw ?? 0) steps · \(gen.paintResRaw ?? 0)px"
        }
        if let edit = gen.editSummary { return "Edited · \(edit)" }
        return gen.quantization == .full
            ? "\(gen.steps) steps"
            : "\(gen.quantization.label) · \(gen.steps) steps"
    }

    private func subtitle(_ gen: Generation) -> String {
        let rel = gen.createdAt.formatted(.relative(presentation: .named))
        var parts = [rel]
        if let d = gen.durationSeconds { parts.append(String(format: "%.1fs", d)) }
        // The seed this run actually used (§3: reproducibility is part of
        // determinism) — a paint version shows its source shape's seed.
        if let seed = gen.seedRaw { parts.append("seed \(seed)") }
        return parts.joined(separator: " · ")
    }
}
