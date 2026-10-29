import SwiftUI

/// Post-generation mesh tools. Each action saves a new shape version, so the
/// original is never overwritten and every edit can be undone from history.
struct MeshEditPopover: View {
    @Environment(ProjectStore.self) private var store
    let projectID: Project.ID
    var onSculpt: () -> Void = {}

    @State private var targetK: Double = 50
    @State private var smoothIterations: Double = 3
    @State private var holeEdges: Double = 64
    @State private var busy = false
    @State private var booleanOp: BooleanOp = .subtract
    @State private var operandChoice = "sphere"          // a BooleanPrimitive raw value or a version UUID
    @State private var operandSize: Double = 0.5
    @State private var offsetX: Double = 0
    @State private var offsetY: Double = 0.6
    @State private var offsetZ: Double = 0
    @State private var booleanRes: Double = 160
    @State private var message: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Edit mesh").font(.headline)

            section("Decimate", detail: "\(Int(targetK))k triangles") {
                Slider(value: $targetK, in: 5...400, step: 5)
                action("Decimate", "arrow.down.right.and.arrow.up.left") {
                    .decimate(faceBudget: Int(targetK) * 1000)
                }
            }
            Divider()
            section("Smooth", detail: "\(Int(smoothIterations)) iterations") {
                Slider(value: $smoothIterations, in: 1...20, step: 1)
                action("Smooth", "scribble.variable") { .smooth(iterations: Int(smoothIterations)) }
            }
            Divider()
            section("Fill holes", detail: "up to \(Int(holeEdges)) edges") {
                Slider(value: $holeEdges, in: 4...256, step: 4)
                action("Fill holes", "circle.dashed") { .fillHoles(maxEdges: Int(holeEdges)) }
            }

            Divider()
            combineSection

            if busy { ProgressView().controlSize(.small) }
            if let message {
                Text(message).font(.caption).foregroundStyle(.secondary)
            }
            Text("Each edit becomes a new version in history.")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(14)
        .frame(width: 270)
        .disabled(busy)
    }

    private func section<C: View>(_ title: String, detail: String,
                                  @ViewBuilder content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title).font(.callout.weight(.medium))
                Spacer()
                Text(detail).font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            content()
        }
    }

    private func action(_ title: String, _ icon: String, edit: @escaping () -> MeshEdit) -> some View {
        Button {
            busy = true
            message = nil
            Task {
                let result = await store.applyMeshEdit(edit(), for: projectID)
                busy = false
                message = result ?? "Saved as a new version."
            }
        } label: {
            Label(title, systemImage: icon).frame(maxWidth: .infinity)
        }
    }
}

extension MeshEditPopover {
    /// Other shape versions of this project that can be combined with the selected one.
    private var otherVersions: [Generation] {
        guard let project = store.project(projectID) else { return [] }
        let currentBase = project.currentGeneration.map { $0.kind == .shape ? $0.id : $0.sourceShapeID }
        return project.generations.filter { $0.kind == .shape && $0.id != currentBase }
    }

    private var operandSpec: ProjectStore.BooleanOperandSpec {
        if let prim = BooleanPrimitive(rawValue: operandChoice) {
            return .primitive(prim, offset: SIMD3(Float(offsetX), Float(offsetY), Float(offsetZ)),
                              size: Float(operandSize))
        }
        return .version(UUID(uuidString: operandChoice) ?? UUID())
    }

    @ViewBuilder
    private var combineSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Combine").font(.callout.weight(.medium))
                Spacer()
                Text("\(Int(booleanRes)) voxels").font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            Picker("Operation", selection: $booleanOp) {
                ForEach(BooleanOp.allCases) { Text($0.label).tag($0) }
            }
            .pickerStyle(.segmented).labelsHidden()
            Picker("With", selection: $operandChoice) {
                ForEach(BooleanPrimitive.allCases) { Text($0.label).tag($0.rawValue) }
                if !otherVersions.isEmpty { Divider() }
                ForEach(Array(otherVersions.enumerated()), id: \.element.id) { n, g in
                    Text("Version \(n + 1)").tag(g.id.uuidString)
                }
            }
            if BooleanPrimitive(rawValue: operandChoice) != nil {
                miniSlider("Size", $operandSize, 0.05...1.5)
                miniSlider("X", $offsetX, -1.5...1.5)
                miniSlider("Y", $offsetY, -1.5...1.5)
                miniSlider("Z", $offsetZ, -1.5...1.5)
            }
            Slider(value: $booleanRes, in: 64...256, step: 16)
            Button {
                busy = true; message = nil
                Task {
                    let result = await store.applyBoolean(booleanOp, with: operandSpec,
                                                          resolution: Int(booleanRes), for: projectID)
                    busy = false
                    message = result ?? "Saved as a new version."
                }
            } label: { Label("Apply \(booleanOp.label.lowercased())", systemImage: "square.on.square.intersection.dashed").frame(maxWidth: .infinity) }
            Text("Resolution-limited: details finer than a voxel are lost.")
                .font(.caption2).foregroundStyle(.tertiary)
        }
    }

    private func miniSlider(_ title: String, _ value: Binding<Double>, _ range: ClosedRange<Double>) -> some View {
        HStack(spacing: 8) {
            Text(title).font(.caption).frame(width: 30, alignment: .leading)
            Slider(value: value, in: range)
        }
    }
}
