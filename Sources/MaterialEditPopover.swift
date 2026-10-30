import SwiftUI

/// Global material tuning for the selected painted version. Apply saves a new paint
/// version (the mesh is shared; textures are rewritten), so the original stays.
struct MaterialEditPopover: View {
    @Environment(ProjectStore.self) private var store
    let projectID: Project.ID

    @State private var adj = MaterialAdjustments()
    @State private var busy = false
    @State private var message: String?
    @State private var palette: [SIMD3<Float>] = []
    @State private var selectedSwatch: Int?
    @State private var tolerance = ToolDefaults.regionTolerance()
    @State private var feather = ToolDefaults.regionFeather()
    @State private var coverage: Float?
    @State private var sample: (rgba: [UInt8], size: Int)?

    private var isPBR: Bool { store.project(projectID)?.currentGeneration?.isPBR ?? false }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Edit material").font(.headline)

            regionSection
            Divider()

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
                Button("Reset") { let r = adj.region; adj = MaterialAdjustments(); adj.region = r; message = nil }
                    .disabled(adj.isIdentity)
                Spacer()
                Button {
                    busy = true
                    message = nil
                    Task {
                        let result = await store.applyMaterialEdit(adj, for: projectID)
                        busy = false
                        if result == nil { let r = adj.region; adj = MaterialAdjustments(); adj.region = r }
                        message = result ?? "Saved as a new version."
                    }
                } label: { Label("Apply", systemImage: "checkmark") }
                .buttonStyle(.borderedProminent)
                .disabled(adj.isIdentity)
            }
            if busy { ProgressView().controlSize(.small) }
            if let message { Text(message).font(.caption).foregroundStyle(.secondary) }
            Text(isPBR ? (adj.region == nil ? "Adjusts the whole texture." : "Adjusts only the selected color.") : "Color textures have no roughness or metalness map.")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(14)
        .frame(width: 270)
        .disabled(busy)
        .task { await loadPalette() }
        .onChange(of: selectedSwatch) { _, _ in syncRegion() }
        .onChange(of: tolerance) { _, _ in syncRegion() }
        .onChange(of: feather) { _, _ in syncRegion() }
    }

    @ViewBuilder private var regionSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Region").font(.callout.weight(.medium))
            HStack(spacing: 6) {
                swatchButton(nil)
                ForEach(palette.indices, id: \.self) { swatchButton($0) }
            }
            if selectedSwatch != nil {
                row("Tolerance", String(format: "%.0f%%", tolerance * 100)) { Slider(value: $tolerance, in: 0.02...0.5) }
                row("Softness", String(format: "%.0f%%", feather * 100)) { Slider(value: $feather, in: 0.01...0.4) }
                if let coverage {
                    Text(String(format: "Covers %.0f%% of the texture", coverage * 100))
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
    }

    private func swatchButton(_ index: Int?) -> some View {
        let on = selectedSwatch == index
        return Button { selectedSwatch = index } label: {
            ZStack {
                if let index {
                    let c = palette[index]
                    Circle().fill(Color(red: Double(c.x), green: Double(c.y), blue: Double(c.z)))
                } else {
                    Circle().strokeBorder(.secondary, style: StrokeStyle(lineWidth: 1, dash: [3]))
                    Image(systemName: "square.dashed").font(.system(size: 10)).foregroundStyle(.secondary)
                }
            }
            .frame(width: 24, height: 24)
            .overlay(Circle().strokeBorder(Color.accentColor, lineWidth: on ? 2.5 : 0).padding(-3))
        }
        .buttonStyle(.plain)
        .help(index == nil ? "Whole texture" : "Only this color")
    }

    private func loadPalette() async {
        guard let gen = store.project(projectID)?.currentGeneration,
              let name = gen.paintedTextureFileName else { return }
        let url = store.folderURL(for: projectID).appendingPathComponent(name)
        let loaded = await Task.detached(priority: .userInitiated) { () -> ([SIMD3<Float>], [UInt8])? in
            guard let d = MaterialOps.decode(url) else { return nil }
            // A strided copy keeps the live coverage readout instant on large textures.
            let step = max(1, d.rgba.count / 4 / 65536)
            var small: [UInt8] = []
            var i = 0
            while i < d.rgba.count / 4 { small.append(contentsOf: d.rgba[i * 4..<i * 4 + 4]); i += step }
            return (MaterialRegion.dominantColors(d.rgba), small)
        }.value
        if let loaded { palette = loaded.0; sample = (loaded.1, 0) }
    }

    private func syncRegion() {
        guard let i = selectedSwatch, palette.indices.contains(i) else {
            adj.region = nil; coverage = nil; return
        }
        let region = ColorRegion(color: palette[i], tolerance: Float(tolerance), feather: Float(feather))
        adj.region = region
        if let sample {
            coverage = MaterialRegion.coverage(of: MaterialRegion.mask(sample.rgba, region: region), rgba: sample.rgba)
        }
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
