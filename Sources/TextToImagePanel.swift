import SwiftUI

/// Prompt box for text-to-3D: describe an object, get a starting image, then generate the 3D
/// model from it as usual. Defaults (steps, guidance, style suffix, negative prompt, background
/// removal, auto-start) live in Settings → Tools.
struct TextToImagePanel: View {
    @Environment(AppRuntime.self) private var runtime
    let projectID: Project.ID
    var compact = false

    @State private var prompt = ""

    private var state: AppRuntime.ViewSynthState? { runtime.textToImage[projectID] }
    private var running: Bool { if case .running? = state { return true } else { return false } }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !compact {
                Label("Or describe it", systemImage: "text.cursor").font(.caption).foregroundStyle(.secondary)
            }
            HStack(alignment: .bottom, spacing: 8) {
                TextField("a small wooden chair…", text: $prompt, axis: .vertical)
                    .lineLimit(1...3)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(generate)
                    .disabled(running)
                if running {
                    Button("Cancel") { runtime.cancelTextToImage(projectID) }
                } else {
                    Button(action: generate) { Label("Generate image", systemImage: "sparkles") }
                        .buttonStyle(.borderedProminent)
                        .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        .help(ModelStore.isInstalled(.textToImage)
                              ? "Make a starting image from your description"
                              : "Installs the Stable Diffusion 2.1 text-to-image model (2.6 GB)")
                }
            }
            switch state {
            case .running(let p)?:
                HStack(spacing: 8) { ProgressView(value: p).frame(maxWidth: 220); Text("Generating image…").font(.caption).foregroundStyle(.secondary) }
            case .failed(let message)?:
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange)
                    Text(message).font(.caption)
                    Button { runtime.dismissTextToImageFailure(projectID) } label: { Image(systemName: "xmark") }.buttonStyle(.plain)
                }
            case nil:
                EmptyView()
            }
        }
        .padding(compact ? 0 : 12)
        .frame(maxWidth: compact ? 360 : 420)
        .background(compact ? AnyShapeStyle(.clear) : AnyShapeStyle(.ultraThinMaterial),
                    in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private func generate() { runtime.generateImage(fromText: prompt, for: projectID) }
}
