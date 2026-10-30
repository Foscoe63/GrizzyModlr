import SwiftUI

/// What the menu bar can ask the open project to do. `ProjectDetailView` publishes one as a
/// focused scene value; commands read it, so shortcuts act on the front window's project.
struct ProjectActions {
    var export: (MeshExportFormat) -> Void
    var runPlugin: (ToolPlugin) -> Void
    var camera: (CameraPreset) -> Void
    var toggleHistory: () -> Void
    var toggleScene: () -> Void
    var toggleMeshEdit: () -> Void
    var hasModel: Bool
    var hasHistory: Bool
}

private struct ProjectActionsKey: FocusedValueKey { typealias Value = ProjectActions }

extension FocusedValues {
    var projectActions: ProjectActions? {
        get { self[ProjectActionsKey.self] }
        set { self[ProjectActionsKey.self] = newValue }
    }
}

/// File ▸ Export, View ▸ Camera, and Edit-tool shortcuts.
struct ProjectCommands: Commands {
    @FocusedValue(\.projectActions) private var actions

    var body: some Commands {
        CommandGroup(after: .importExport) {
            Menu("Export Model") {
                ForEach(MeshExportFormat.allCases) { fmt in
                    Button(fmt.menuTitle) { actions?.export(fmt) }
                }
            }
            .disabled(actions?.hasModel != true)
            Button("Export as GLB…") { actions?.export(.glb) }
                .keyboardShortcut("e", modifiers: .command)
                .disabled(actions?.hasModel != true)
            Button("Export as USDZ…") { actions?.export(.usdz) }
                .keyboardShortcut("e", modifiers: [.command, .shift])
                .disabled(actions?.hasModel != true)
        }
        CommandMenu("Model") {
            Button("Edit Mesh…") { actions?.toggleMeshEdit() }
                .keyboardShortcut("m", modifiers: [.command, .option])
                .disabled(actions?.hasModel != true)
            Button("Version History") { actions?.toggleHistory() }
                .keyboardShortcut("h", modifiers: [.command, .option])
                .disabled(actions?.hasHistory != true)
            Button("Compose Scene…") { actions?.toggleScene() }
                .keyboardShortcut("k", modifiers: [.command, .option])
                .disabled(actions?.hasHistory != true)
            Menu("Tools") {
                let plugins = ToolPlugin.load()
                if plugins.isEmpty {
                    Text("No plugins installed")
                } else {
                    ForEach(plugins) { plugin in
                        Button(plugin.name) { actions?.runPlugin(plugin) }
                    }
                }
                Divider()
                Button("Open Plugins Folder") {
                    try? FileManager.default.createDirectory(at: ToolPlugin.folder, withIntermediateDirectories: true)
                    NSWorkspace.shared.open(ToolPlugin.folder)
                }
            }
            .disabled(actions?.hasModel != true)
            Divider()
            ForEach(Array(CameraPreset.allCases.enumerated()), id: \.element) { index, preset in
                Button("Camera: \(preset.label)") { actions?.camera(preset) }
                    .keyboardShortcut(KeyEquivalent(Character("\(index + 1)")), modifiers: .command)
                    .disabled(actions?.hasModel != true)
            }
        }
    }
}
