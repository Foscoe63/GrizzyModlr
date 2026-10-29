import SwiftUI

/// Floating controls for an active sculpt session: tool, brush size/strength, undo,
/// and Save (new version) / Cancel (discard).
struct SculptToolbar: View {
    @Bindable var sculpt: SculptController
    var busy: Bool
    var onSave: () -> Void
    var onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Picker("Tool", selection: $sculpt.tool) {
                ForEach(SculptTool.allCases) { t in Label(t.label, systemImage: t.icon).tag(t) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            HStack(spacing: 8) {
                Text("Size").font(.caption).frame(width: 52, alignment: .leading)
                Slider(value: $sculpt.radiusFraction, in: 0.02...0.4)
            }
            HStack(spacing: 8) {
                Text("Strength").font(.caption).frame(width: 52, alignment: .leading)
                Slider(value: $sculpt.strength, in: 0.05...1)
            }
            HStack {
                Button { sculpt.undo() } label: { Label("Undo", systemImage: "arrow.uturn.backward") }
                    .disabled(!sculpt.canUndo || busy)
                Spacer()
                Button("Cancel", action: onCancel).disabled(busy)
                Button("Save", action: onSave)
                    .buttonStyle(.borderedProminent)
                    .disabled(!sculpt.isEdited || busy)
            }
            Text("Drag to sculpt · hold ⌥ to orbit")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(12)
        .frame(width: 280)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(.white.opacity(0.12)))
    }
}
