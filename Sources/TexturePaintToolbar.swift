import SwiftUI

/// Floating controls for an active texture-paint session.
struct TexturePaintToolbar: View {
    @Bindable var paint: TexturePaintController
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
            if paint.tool != .heal { slider(paint.tool == .paint ? "Opacity" : "Strength", $paint.opacity, 0.05...1) }

            HStack {
                Button { paint.undo() } label: { Label("Undo", systemImage: "arrow.uturn.backward") }
                    .disabled(!paint.canUndo || busy)
                Spacer()
                Button("Cancel", action: onCancel).disabled(busy)
                Button("Save", action: onSave)
                    .buttonStyle(.borderedProminent)
                    .disabled(!paint.isEdited || busy)
            }
            Text(paint.tool == .heal ? "Heal fills blemishes from the surrounding colors"
                                     : "Drag to paint · hold ⌥ to orbit")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(12)
        .frame(width: 280)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(.white.opacity(0.12)))
    }

    private func slider(_ title: String, _ value: Binding<Double>, _ range: ClosedRange<Double>) -> some View {
        HStack(spacing: 8) {
            Text(title).font(.caption).frame(width: 58, alignment: .leading)
            Slider(value: value, in: range)
        }
    }
}
