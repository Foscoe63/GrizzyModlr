import SwiftUI

/// Settings for one weight-painting session.
struct WeightBrush: Equatable {
    var joint: String
    var add: Bool
    var radius: Float
    var strength: Float
}

/// Post-generation mesh tools. Each action saves a new shape version, so the
/// original is never overwritten and every edit can be undone from history.
struct MeshEditPopover: View {
    @Environment(ProjectStore.self) private var store
    let projectID: Project.ID
    var onSculpt: () -> Void = {}
    var flowShown = false
    var flowBusy = false
    var onToggleFlow: () -> Void = {}
    var guides: [FlowGuide] = []
    var onDrawGuides: () -> Void = {}
    var onClearGuides: () -> Void = {}
    var onRigOverlay: ([FlowGuide]) -> Void = { _ in }
    var onPaintWeights: (WeightBrush) -> Void = { _ in }

    @State private var targetK: Double = ToolDefaults.decimateK()
    @State private var smoothIterations: Double = ToolDefaults.smoothIterations()
    @State private var holeEdges: Double = ToolDefaults.holeEdges()
    @State private var symmetryAxis: SymmetryAxis = ToolDefaults.symmetryAxis()
    @State private var symmetryKeepPositive: Bool = ToolDefaults.symmetryKeepPositive()
    @State private var remeshRes: Double = ToolDefaults.remeshResolution()
    @State private var lodText: String = UserDefaults.standard.string(forKey: ToolDefaults.Key.lodLevels) ?? ToolDefaults.fallbackLODLevels
    @State private var busy = false
    @State private var poseName = PoseLibrary.all[1].name
    @State private var rigJoint = "wristL"
    @State private var rigNote: String?
    @State private var brushAdd = true
    @State private var brushRadius = ToolDefaults.weightBrushRadius()
    @State private var brushStrength = ToolDefaults.weightBrushStrength()
    @State private var booleanOp: BooleanOp = .subtract
    @State private var operandChoice = "sphere"          // a BooleanPrimitive raw value or a version UUID
    @State private var operandSize: Double = 0.5
    @State private var offsetX: Double = 0
    @State private var offsetY: Double = 0.6
    @State private var offsetZ: Double = 0
    @State private var booleanRes: Double = ToolDefaults.booleanResolution()
    @State private var message: String?

    var body: some View {
        ScrollView { content }
            .frame(width: 290, height: 600)
    }

    private var content: some View {
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
            topologySection
            Divider()
            poseSection

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
        .frame(width: 290, alignment: .leading)
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
            Slider(value: $booleanRes, in: 64...384, step: 16)
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

extension MeshEditPopover {
    @ViewBuilder
    fileprivate var poseSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Pose").font(.callout.weight(.medium))
            Picker("Pose", selection: $poseName) {
                ForEach(PoseLibrary.all, id: \.name) { Text($0.name).tag($0.name) }
            }
            .labelsHidden()
            action("Bake pose into mesh", "figure.stand") { .pose(name: poseName, rig: store.project(projectID)?.rigConfig ?? RigConfig()) }
            rigTuning
            Text("Fits a humanoid skeleton to an upright figure and bends the mesh. Saved as a new version. For a rigged, animated file use Export ▸ GLB · rigged.")
                .font(.caption2).foregroundStyle(.tertiary)
        }
    }

    private var rigConfig: RigConfig { store.project(projectID)?.rigConfig ?? RigConfig() }

    private func editRig(_ change: (inout RigConfig) -> Void) {
        var c = rigConfig
        change(&c)
        store.setRigConfig(c, for: projectID)
        Task {
            let overlay = await store.rigPreview(for: projectID)
            rigNote = overlay == nil ? "This shape doesn't look like a standing figure." : nil
            onRigOverlay(overlay ?? [])
        }
    }

    @ViewBuilder
    fileprivate var rigTuning: some View {
        DisclosureGroup("Adjust rig") {
            VStack(alignment: .leading, spacing: 6) {
                Button("Show skeleton") { editRig { _ in } }.controlSize(.small)
                Picker("Joint", selection: $rigJoint) {
                    ForEach(Skeleton.names, id: \.self) { Text($0).tag($0) }
                }
                let off = rigConfig.offset(rigJoint)
                ForEach(0..<3, id: \.self) { axis in
                    miniSlider(["X", "Y", "Z"][axis],
                               Binding(get: { Double(off[axis]) }, set: { v in
                                   editRig { c in var o = c.offset(rigJoint); o[axis] = Float(v); c.setOffset(rigJoint, o) } }),
                               -0.15...0.15)
                }
                miniSlider("Reach", Binding(get: { Double(rigConfig.influence[rigJoint] ?? 1) }, set: { v in
                    editRig { $0.setInfluence(rigJoint, Float(v)) } }), 0.3...3)
                HStack {
                    Text("Smooth weights").font(.caption)
                    Stepper("\(rigConfig.smoothing)", value: Binding(get: { rigConfig.smoothing }, set: { v in
                        editRig { $0.smoothing = v } }), in: 0...6).font(.caption)
                }
                Picker("Brush", selection: $brushAdd) { Text("Add").tag(true); Text("Remove").tag(false) }
                    .pickerStyle(.segmented).labelsHidden()
                miniSlider("Size", $brushRadius, 0.02...0.25)
                miniSlider("Power", $brushStrength, 0.1...1)
                Button {
                    onPaintWeights(WeightBrush(joint: rigJoint, add: brushAdd, radius: Float(brushRadius), strength: Float(brushStrength)))
                } label: { Label("Paint \(rigJoint) weights on the model", systemImage: "paintbrush.pointed") }
                    .controlSize(.small)
                if !rigConfig.strokes.isEmpty {
                    Text("\(rigConfig.strokes.count) brush stroke\(rigConfig.strokes.count == 1 ? "" : "s")").font(.caption2).foregroundStyle(.secondary)
                }
                Button("Reset rig adjustments") { editRig { $0 = RigConfig() } }
                    .controlSize(.small).disabled(rigConfig.isDefault)
                if let rigNote { Text(rigNote).font(.caption2).foregroundStyle(.orange) }
                Text("Offsets move a joint (fraction of height). Reach scales how far its bones pull the surface. Smooth weights softens joints. Brush strokes pull the surface toward (or away from) the chosen joint; they follow the model, not vertex numbers.")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        }
        .font(.caption)
    }

    @ViewBuilder
    fileprivate var topologySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Topology").font(.callout.weight(.medium))
            HStack {
                Picker("Axis", selection: $symmetryAxis) {
                    ForEach(SymmetryAxis.allCases) { Text($0.rawValue.uppercased()).tag($0) }
                }
                .pickerStyle(.segmented).labelsHidden().frame(width: 100)
                Picker("Keep", selection: $symmetryKeepPositive) {
                    Text("+ half").tag(true); Text("− half").tag(false)
                }
                .pickerStyle(.segmented).labelsHidden()
            }
            action("Make symmetric", "arrow.left.and.right.righttriangle.left.righttriangle.right") {
                .symmetrize(axis: symmetryAxis, keepPositive: symmetryKeepPositive)
            }
            HStack {
                Text("Remesh").font(.caption)
                Spacer()
                Text("\(Int(remeshRes)) voxels").font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            Slider(value: $remeshRes, in: 48...320, step: 16)
            action("Remesh evenly", "square.grid.3x3") { .remesh(resolution: Int(remeshRes)) }
            action("Quad remesh (aligned)", "square.grid.2x2") { .quadRemesh(resolution: Int(remeshRes)) }
            action("Flow remesh (experimental)", "wind") {
                .flowRemesh(targetQuads: Int(ToolDefaults.flowTargetQuads()), featureAngle: Float(ToolDefaults.flowFeatureAngle()),
                            fallbackResolution: Int(remeshRes), guides: guides,
                            guideStrength: Float(ToolDefaults.flowGuideStrength()))
            }
            HStack {
                Button { onDrawGuides() } label: {
                    Label("Draw flow guides", systemImage: "scribble.variable").frame(maxWidth: .infinity)
                }
                .help("Drag on the model to draw curves the quad rows should follow (around an eye, along a limb). Flow remesh and the flow field view use them.")
                if !guides.isEmpty {
                    Button(role: .destructive) { onClearGuides() } label: {
                        Label("Clear \(guides.count)", systemImage: "xmark.circle")
                    }
                }
            }
            Button { onToggleFlow() } label: {
                Label(flowShown ? "Hide flow field" : (flowBusy ? "Computing flow field…" : "Show flow field"),
                      systemImage: "wind").frame(maxWidth: .infinity)
            }
            .disabled(flowBusy)
            .help("Draws the direction field that quad rows will follow: cyan lines, red and blue dots where it has singular points")
            HStack(spacing: 6) {
                Text("LOD").font(.caption)
                TextField("50, 25, 10", text: $lodText).textFieldStyle(.roundedBorder).controlSize(.small)
                Text("%").font(.caption).foregroundStyle(.secondary)
            }
            Button {
                busy = true; message = nil
                let levels = ToolDefaults.parseLODLevels(lodText)
                Task {
                    let result = await store.applyLODs(percents: levels, for: projectID)
                    busy = false
                    message = result ?? "Saved \(levels.count) LOD version\(levels.count == 1 ? "" : "s")."
                }
            } label: { Label("Generate LODs", systemImage: "square.stack.3d.down.right").frame(maxWidth: .infinity) }
            Text("Symmetry mirrors across the model's center. Remesh needs a closed shape and loses detail below a voxel.")
                .font(.caption2).foregroundStyle(.tertiary)
        }
    }
}
