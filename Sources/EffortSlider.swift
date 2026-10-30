import SwiftUI
import AppKit

/// A native macOS slider (NSSlider) with one tick mark per stop that snaps to ticks.
struct SteppedSlider: NSViewRepresentable {
    @Binding var index: Int
    let count: Int

    func makeNSView(context: Context) -> NSSlider {
        let slider = NSSlider(value: Double(index),
                              minValue: 0, maxValue: Double(max(count - 1, 1)),
                              target: context.coordinator,
                              action: #selector(Coordinator.changed(_:)))
        slider.numberOfTickMarks = count
        slider.allowsTickMarkValuesOnly = true
        slider.tickMarkPosition = .below
        slider.isContinuous = true
        return slider
    }

    func updateNSView(_ slider: NSSlider, context: Context) {
        context.coordinator.parent = self
        slider.maxValue = Double(max(count - 1, 1))
        slider.numberOfTickMarks = count
        if Int(slider.doubleValue.rounded()) != index {
            slider.doubleValue = Double(index)
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject {
        var parent: SteppedSlider
        init(_ parent: SteppedSlider) { self.parent = parent }
        @objc func changed(_ sender: NSSlider) {
            let i = Int(sender.doubleValue.rounded())
            if parent.index != i { parent.index = i }
        }
    }
}

/// Popover content: the Small/Large model picker plus the quality controls —
/// one effort slider in normal mode, every knob (steps/guidance/octree/
/// quantization/seed) in advanced mode. Per DESIGN.md §5.
struct ModelSettingsPopover: View {
    @Environment(ProjectStore.self) private var store
    let projectID: Project.ID

    private var project: Project? { store.project(projectID) }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let project {
                modelPicker(project)
                Divider()
                if project.advancedMode {
                    advancedControls(project)
                } else {
                    normalControls(project)
                }

                Divider()

                Toggle("Advanced mode", isOn: advancedBinding.animation(.snappy(duration: 0.2)))
                    .toggleStyle(.switch)

                Button { store.applyMaximumDetail(for: projectID) } label: {
                    Label("Use maximum detail", systemImage: "wand.and.stars")
                }
                .controlSize(.small)
                .help("Shape: top steps, full-precision weights, 768 grid. Paint: Max quality, 4096 texture, 400k triangles, super-resolution.")
            }
        }
        .padding(16)
        .frame(width: 320)
    }

    // MARK: model picker (both modes)

    @ViewBuilder
    private func modelPicker(_ p: Project) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Model").fontWeight(.semibold)
                Spacer()
                Text(p.shapeModel.detail).font(.caption).foregroundStyle(.tertiary)
            }
            Picker("", selection: modelBinding) {
                ForEach(ShapeModel.allCases) { m in
                    Text(m.label).tag(m)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
        }
    }

    // MARK: normal — one combined quality slider

    @ViewBuilder
    private func normalControls(_ p: Project) -> some View {
        steppedRow(title: "Quality", value: p.quality.label, detail: p.quality.detail(for: p.shapeModel),
                   index: qualityIndex, count: QualityPreset.ordered.count,
                   low: "Fast", high: "Best")
        Text("8-bit weights · higher = more detail, slower")
            .font(.caption2).foregroundStyle(.tertiary)
    }

    // MARK: advanced — every knob

    @ViewBuilder
    private func advancedControls(_ p: Project) -> some View {
        steppedRow(title: "Weights", value: p.quantization.label, detail: p.quantization.detail,
                   index: quantIndex, count: Quantization.ordered.count, low: "Smaller", high: "Best")
        Divider()
        valueRow(title: "Denoise steps", value: "\(p.steps)", binding: stepsBinding, range: 4...150,
                 detail: "More steps = finer shape, slower")
        Divider()
        valueRow(title: "Guidance", value: String(format: "%.1f", p.guidance), binding: guidanceBinding,
                 range: 0...10, detail: "Prompt adherence (CFG)")
        Divider()
        steppedRow(title: "Resolution", value: "\(p.octree)", detail: DetailEstimate.summary(octree: p.octree).map { "\($0) (rough)" } ?? "Mesh grid — finer = denser mesh",
                   index: octreeIndex, count: QualityPreset.octreeStops.count, low: "Coarse", high: "Fine")
        Divider()
        seedRow(p)
    }

    /// Seed pinning (§3: every generation records its seed; pin to reproduce).
    /// The pin button copies the last generation's recorded seed into the field so
    /// "regenerate exactly that one" is one click; the dice returns to random.
    private func seedRow(_ p: Project) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Seed").fontWeight(.semibold)
                Spacer()
                if let last = lastRecordedSeed(p), p.seed != last {
                    Button {
                        store.setSeed(last, for: projectID)
                    } label: {
                        Image(systemName: "pin")
                    }
                    .buttonStyle(.borderless)
                    .help("Reuse the last run's seed (\(last))")
                }
                if p.seed != nil {
                    Button {
                        store.setSeed(nil, for: projectID)
                    } label: {
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

    /// The seed the newest shape generation actually ran with, if any.
    private func lastRecordedSeed(_ p: Project) -> UInt64? {
        p.generations.last(where: { $0.kind == .shape && $0.seedRaw != nil })?.seedRaw
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

    // MARK: bindings (read live from the store)

    private var advancedBinding: Binding<Bool> {
        Binding(get: { store.project(projectID)?.advancedMode ?? false },
                set: { store.setAdvancedMode($0, for: projectID) })
    }
    private var modelBinding: Binding<ShapeModel> {
        Binding(get: { store.project(projectID)?.shapeModel ?? .small },
                set: { store.setShapeModel($0, for: projectID) })
    }
    private var qualityIndex: Binding<Int> {
        Binding(get: { QualityPreset.ordered.firstIndex(of: store.project(projectID)?.quality ?? .fast) ?? 1 },
                set: { store.setQuality(QualityPreset.ordered[$0], for: projectID) })
    }
    private var quantIndex: Binding<Int> {
        Binding(get: { Quantization.ordered.firstIndex(of: store.project(projectID)?.quantization ?? .full) ?? 0 },
                set: { store.setQuantization(Quantization.ordered[$0], for: projectID) })
    }
    private var stepsBinding: Binding<Double> {
        Binding(get: { Double(store.project(projectID)?.steps ?? 30) },
                set: { store.setSteps(Int($0.rounded()), for: projectID) })
    }
    private var guidanceBinding: Binding<Double> {
        Binding(get: { store.project(projectID)?.guidance ?? 5.0 },
                set: { store.setGuidance(($0 * 10).rounded() / 10, for: projectID) })
    }
    private var octreeIndex: Binding<Int> {
        Binding(get: { QualityPreset.octreeStops.firstIndex(of: store.project(projectID)?.octree ?? 256) ?? 2 },
                set: { store.setOctree(QualityPreset.octreeStops[$0], for: projectID) })
    }
    private var seedBinding: Binding<String> {
        Binding(get: { store.project(projectID)?.seed.map(String.init) ?? "" },
                set: { text in
                    let trimmed = text.trimmingCharacters(in: .whitespaces)
                    store.setSeed(trimmed.isEmpty ? nil : UInt64(trimmed), for: projectID)
                })
    }
}
