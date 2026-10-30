import SwiftUI

/// Paint (texture) configuration — the Color/PBR model picker plus Normal (one
/// quality slider) / Advanced (steps, view resolution, texture size, mesh
/// detail, super-resolution). Mirrors ModelSettingsPopover.
struct PaintSettingsPopover: View {
    @Environment(ProjectStore.self) private var store
    let projectID: Project.ID

    private let resStops = [256, 384, 512]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let project = store.project(projectID) {
                modelPicker(project)
                Divider()
                if project.paintAdvanced {
                    advanced(project)
                } else {
                    normal(project)
                }

                Divider()

                Toggle("Advanced mode", isOn: advancedBinding.animation(.snappy(duration: 0.2)))
                    .toggleStyle(.switch)

                Button { store.applyMaximumDetail(for: projectID) } label: {
                    Label("Use maximum detail", systemImage: "wand.and.stars")
                }
                .controlSize(.small)
                .help("Paint: Max quality, 4096 texture, 400k triangles, super-resolution. Also sets the shape to its practical maximum.")
            }
        }
        .padding(16)
        .frame(width: 320)
    }

    /// Small = Color (RGB), Large = PBR (albedo + metallic-roughness), per §5.
    @ViewBuilder
    private func modelPicker(_ p: Project) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Texture type").fontWeight(.semibold)
                Spacer()
                Text(p.paintModel.detail).font(.caption).foregroundStyle(.tertiary)
            }
            Picker("", selection: modelBinding) {
                ForEach(PaintModel.allCases) { m in
                    Text(m.label).tag(m)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
        }
    }

    @ViewBuilder
    private func normal(_ p: Project) -> some View {
        steppedRow(title: "Texture quality", value: p.paintQuality.label, detail: p.paintQuality.detail,
                   index: qualityIndex, count: PaintQuality.ordered.count, low: "Fast", high: "Best")
    }

    @ViewBuilder
    private func advanced(_ p: Project) -> some View {
        valueRow(title: "Diffusion steps", value: "\(p.paintSteps)", binding: stepsBinding,
                 range: 4...30, detail: "More steps = cleaner views, slower")
        Divider()
        steppedRow(title: "View resolution", value: "\(p.paintRes)px", detail: "Multiview render size",
                   index: resIndex, count: resStops.count, low: "Low", high: "High")
        Divider()
        steppedRow(title: "Texture size", value: "\(p.paintTex)", detail: "Baked texture resolution",
                   index: texIndex, count: PaintQuality.texStops.count, low: "1K", high: "4K")
        Divider()
        steppedRow(title: "Mesh detail", value: "\(p.paintFaces / 1000)k tris",
                   detail: "Painted-mesh face budget — higher keeps more geometry, slower UV-unwrap",
                   index: facesIndex, count: PaintQuality.faceStops.count, low: "Coarse", high: "Fine")
        Divider()
        VStack(alignment: .leading, spacing: 3) {
            Toggle("Super-resolution", isOn: superresBinding).toggleStyle(.switch)
            Text("4× upscale views before baking — sharper, slower")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        Divider()
        seedRow(p)
    }

    /// Paint seed (§3: every paint version records its seed; pin to reproduce). The
    /// library now takes the initial-noise seed, so texturing is reproducible too.
    private func seedRow(_ p: Project) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Seed").fontWeight(.semibold)
                Spacer()
                if let last = lastRecordedSeed(p), p.paintSeed != last {
                    Button { store.setPaintSeed(last, for: projectID) } label: {
                        Image(systemName: "pin")
                    }
                    .buttonStyle(.borderless)
                    .help("Reuse the last paint run's seed (\(last))")
                }
                if p.paintSeed != nil {
                    Button { store.setPaintSeed(nil, for: projectID) } label: {
                        Image(systemName: "dice")
                    }
                    .buttonStyle(.borderless)
                    .help("Back to a random seed per run")
                }
                TextField("Random", text: seedBinding)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 120)
                    .multilineTextAlignment(.trailing)
                    .font(.callout.monospacedDigit())
            }
            Text("Empty = new random seed each run (recorded per version)")
                .font(.caption).foregroundStyle(.tertiary)
        }
    }

    /// The seed the newest paint generation actually ran with, if any.
    private func lastRecordedSeed(_ p: Project) -> UInt64? {
        p.generations.last(where: { $0.kind == .paint && $0.paintSeedRaw != nil })?.paintSeedRaw
    }

    // MARK: rows

    private func steppedRow(title: String, value: String, detail: String,
                            index: Binding<Int>, count: Int, low: String, high: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title).fontWeight(.semibold)
                Spacer()
                Text(value).foregroundStyle(.secondary)
            }
            HStack(spacing: 8) {
                Text(low).font(.caption2).foregroundStyle(.secondary)
                SteppedSlider(index: index, count: count)
                Text(high).font(.caption2).foregroundStyle(.secondary)
            }
            Text(detail).font(.caption).foregroundStyle(.tertiary)
        }
    }

    private func valueRow(title: String, value: String, binding: Binding<Double>,
                          range: ClosedRange<Double>, detail: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title).fontWeight(.semibold)
                Spacer()
                Text(value).foregroundStyle(.secondary).monospacedDigit()
            }
            Slider(value: binding, in: range)
            Text(detail).font(.caption).foregroundStyle(.tertiary)
        }
    }

    // MARK: bindings (live)

    private var advancedBinding: Binding<Bool> {
        Binding(get: { store.project(projectID)?.paintAdvanced ?? false },
                set: { store.setPaintAdvanced($0, for: projectID) })
    }
    private var modelBinding: Binding<PaintModel> {
        Binding(get: { store.project(projectID)?.paintModel ?? .small },
                set: { store.setPaintModel($0, for: projectID) })
    }
    private var qualityIndex: Binding<Int> {
        Binding(get: { PaintQuality.ordered.firstIndex(of: store.project(projectID)?.paintQuality ?? .fast) ?? 0 },
                set: { store.setPaintQuality(PaintQuality.ordered[$0], for: projectID) })
    }
    private var stepsBinding: Binding<Double> {
        Binding(get: { Double(store.project(projectID)?.paintSteps ?? 10) },
                set: { store.setPaintSteps(Int($0.rounded()), for: projectID) })
    }
    private var resIndex: Binding<Int> {
        Binding(get: { resStops.firstIndex(of: store.project(projectID)?.paintRes ?? 512) ?? 2 },
                set: { store.setPaintRes(resStops[$0], for: projectID) })
    }
    private var texIndex: Binding<Int> {
        Binding(get: { PaintQuality.texStops.firstIndex(of: store.project(projectID)?.paintTex ?? 2048) ?? 1 },
                set: { store.setPaintTex(PaintQuality.texStops[$0], for: projectID) })
    }
    private var superresBinding: Binding<Bool> {
        Binding(get: { store.project(projectID)?.paintSuperres ?? false },
                set: { store.setPaintSuperres($0, for: projectID) })
    }
    private var facesIndex: Binding<Int> {
        Binding(get: { PaintQuality.faceStops.firstIndex(of: store.project(projectID)?.paintFaces ?? 120_000) ?? 2 },
                set: { store.setPaintFaces(PaintQuality.faceStops[$0], for: projectID) })
    }
    private var seedBinding: Binding<String> {
        Binding(get: { store.project(projectID)?.paintSeed.map(String.init) ?? "" },
                set: { text in
                    let trimmed = text.trimmingCharacters(in: .whitespaces)
                    store.setPaintSeed(trimmed.isEmpty ? nil : UInt64(trimmed), for: projectID)
                })
    }
}
