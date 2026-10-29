import SwiftUI

/// Post-generation mesh tools. Each action saves a new shape version, so the
/// original is never overwritten and every edit can be undone from history.
struct MeshEditPopover: View {
    @Environment(ProjectStore.self) private var store
    let projectID: Project.ID

    @State private var targetK: Double = 50
    @State private var smoothIterations: Double = 3
    @State private var holeEdges: Double = 64
    @State private var busy = false
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
