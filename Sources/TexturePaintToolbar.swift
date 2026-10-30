import SwiftUI

/// Floating controls for an active texture-paint session.
struct TexturePaintToolbar: View {
    @Environment(AppRuntime.self) private var runtime
    @Bindable var paint: TexturePaintController
    let projectID: Project.ID
    var busy: Bool
    var onSave: () -> Void
    var onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Picker("Tool", selection: $paint.tool) {
                ForEach(PaintTool.allCases) { t in Label(t.label, systemImage: t.icon).tag(t) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            if paint.tool == .paint {
                ColorPicker("Color", selection: $paint.color, supportsOpacity: false).font(.caption)
            }
            slider("Size", $paint.radiusFraction, 0.01...0.2)
            if paint.tool == .paint { slider("Hardness", $paint.hardness, 0...1) }
            if paint.tool != .heal && paint.tool != .aiFill { slider(paint.tool == .paint ? "Opacity" : "Strength", $paint.opacity, 0.05...1) }

            if paint.tool == .aiFill { aiFillControls }

            HStack {
                Button { paint.undo() } label: { Label("Undo", systemImage: "arrow.uturn.backward") }
                    .disabled(!paint.canUndo || busy || filling)
                Spacer()
                Button("Cancel", action: onCancel).disabled(busy || filling)
                Button("Save", action: onSave)
                    .buttonStyle(.borderedProminent)
                    .disabled(!paint.isEdited || busy || filling)
            }
            Text(paint.tool == .heal ? "Heal fills blemishes from the surrounding colors"
                 : paint.tool == .aiFill ? "Paint over the area to replace, then Fill · hold ⌥ to orbit"
                 : "Drag to paint · hold ⌥ to orbit")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(12)
        .frame(width: 280)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(.white.opacity(0.12)))
    }

    private var filling: Bool { if case .running? = runtime.aiFill[projectID] { return true } else { return false } }

    @ViewBuilder private var aiFillControls: some View {
        TextField("Describe the surface", text: $paint.fillPrompt, axis: .vertical)
            .lineLimit(1...3).textFieldStyle(.roundedBorder).font(.caption)
            .disabled(filling)
        if case .running(let p)? = runtime.aiFill[projectID] {
            HStack {
                ProgressView(value: p).frame(maxWidth: .infinity)
                Button("Stop") { runtime.cancelAIFill(projectID) }
            }
        } else {
            HStack {
                Button("Clear") { paint.discardFillSelection() }
                    .disabled(!paint.hasFillSelection)
                Spacer()
                Button { runtime.startAIFill(paint, for: projectID) } label: { Label("Fill", systemImage: "sparkles") }
                    .buttonStyle(.borderedProminent)
                    .disabled(!paint.hasFillSelection || busy)
            }
        }
        if case .failed(let message)? = runtime.aiFill[projectID] {
            HStack(alignment: .top) {
                Text(message).font(.caption2).foregroundStyle(.orange)
                Spacer()
                Button("OK") { runtime.dismissAIFillFailure(projectID) }.controlSize(.small)
            }
        }
    }

    private func slider(_ title: String, _ value: Binding<Double>, _ range: ClosedRange<Double>) -> some View {
        HStack(spacing: 8) {
            Text(title).font(.caption).frame(width: 58, alignment: .leading)
            Slider(value: value, in: range)
        }
    }
}
