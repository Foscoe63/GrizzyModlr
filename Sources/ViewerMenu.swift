import SwiftUI

/// The small "camera" button in a viewport corner: camera presets, turntable, lighting, ground,
/// shadows and image export. Toggles write the same settings the Settings → Viewer tab edits.
struct ViewerMenuButton: View {
    var onCommand: (ViewerCommand) -> Void

    @AppStorage(ViewerDefaults.Key.environment) private var envName = ViewerDefaults.fallbackEnvironment
    @AppStorage(ViewerDefaults.Key.showGround) private var showGround = false
    @AppStorage(ViewerDefaults.Key.shadows) private var shadows = false
    @AppStorage(ViewerDefaults.Key.autoRotate) private var autoRotate = false
    @AppStorage(ViewerDefaults.Key.customEnvironmentPath) private var customPath = ""

    var body: some View {
        Menu {
            Section("Camera") {
                ForEach(CameraPreset.allCases) { preset in
                    Button { onCommand(ViewerCommand(kind: .camera(preset))) } label: {
                        Label(preset.label, systemImage: preset.icon)
                    }
                }
            }
            Section("Lighting") {
                Picker("Environment", selection: $envName) {
                    ForEach(ViewerEnvironment.allCases.filter { $0 != .custom || !customPath.isEmpty }) {
                        Text($0.label).tag($0.rawValue)
                    }
                }
                Toggle("Ground", isOn: $showGround)
                Toggle("Shadows", isOn: $shadows)
                Toggle("Turntable", isOn: $autoRotate)
            }
            Divider()
            Button { onCommand(ViewerCommand(kind: .snapshot)) } label: {
                Label("Save Image…", systemImage: "camera")
            }
        } label: {
            Image(systemName: "camera.viewfinder")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.primary)
                .frame(width: 28, height: 28)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous).strokeBorder(.white.opacity(0.12)))
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Camera, lighting and image export")
    }
}
