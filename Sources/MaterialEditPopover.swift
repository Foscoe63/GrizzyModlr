import SwiftUI

/// Global material tuning for the selected painted version. Apply saves a new paint
/// version (the mesh is shared; textures are rewritten), so the original stays.
struct MaterialEditPopover: View {
    @Environment(ProjectStore.self) private var store
    let projectID: Project.ID

    @State private var adj = MaterialAdjustments()
    @State private var busy = false
    @State private var message: String?

    private var isPBR: Bool { store.project(projectID)?.currentGeneration?.isPBR ?? false }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Edit material").font(.headline)

            row("Brightness", String(format: "%+.2f", adj.brightness)) {
                Slider(value: $adj.brightness, in: -0.5...0.5)
            }
            row("Saturation", String(format: "×%.2f", adj.saturation)) {
                Slider(value: $adj.saturation, in: 0...2)
            }
            row("Hue", String(format: "%+.0f°", adj.hue)) {
                Slider(value: $adj.hue, in: -180...180)
            }
            if isPBR {
                Divider()
                row("Roughness", String(format: "×%.2f", adj.roughness)) {
                    Slider(value: $adj.roughness, in: 0...2)
                }
                row("Metalness", String(format: "×%.2f", adj.metalness)) {
                    Slider(value: $adj.metalness, in: 0...2)
                }
            }

            HStack {
                Button("Reset") { adj = MaterialAdjustments(); message = nil }
                    .disabled(adj.isIdentity)
                Spacer()
                Button {
                    busy = true
                    message = nil
                    Task {
                        let result = await store.applyMaterialEdit(adj, for: projectID)
                        busy = false
                        if result == nil { adj = MaterialAdjustments() }
                        message = result ?? "Saved as a new version."
                    }
                } label: { Label("Apply", systemImage: "checkmark") }
                .buttonStyle(.borderedProminent)
                .disabled(adj.isIdentity)
            }
            if busy { ProgressView().controlSize(.small) }
            if let message { Text(message).font(.caption).foregroundStyle(.secondary) }
            Text(isPBR ? "Adjusts the whole texture." : "Color textures have no roughness or metalness map.")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(14)
        .frame(width: 270)
        .disabled(busy)
    }

    private func row<C: View>(_ title: String, _ value: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title).font(.callout.weight(.medium))
                Spacer()
                Text(value).font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            content()
        }
    }
}
