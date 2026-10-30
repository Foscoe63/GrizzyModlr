import SwiftUI

@main
struct GrizzyModlrApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var runtime = AppRuntime()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(runtime)
                .environment(runtime.store)
                .frame(minWidth: 480, minHeight: 480)
                .task {
                    runtime.bootIfNeeded()
                    SmokeRunner.startIfRequested(runtime: runtime)
                }
        }
        .defaultSize(width: 1180, height: 760)
        .commands {
            ProjectCommands()
            CommandGroup(replacing: .newItem) {
                Button("New Project") { runtime.store.newProject() }
                    .keyboardShortcut("n", modifiers: .command)
            }
            CommandMenu("Generate") {
                Button("Generate / Regenerate Shape") {
                    if let id = runtime.store.selection { runtime.requestGenerate(id) }
                }
                .keyboardShortcut("r", modifiers: .command)
                .disabled(runtime.store.selection == nil)

                Button("Paint Texture") {
                    if let id = runtime.store.selection { runtime.requestPaint(id) }
                }
                .keyboardShortcut("p", modifiers: [.command, .shift])
                .disabled(runtime.store.selection == nil)

                Divider()

                Button("Stop") {
                    if let id = runtime.store.selection {
                        runtime.cancelShape(id)
                        runtime.cancelPaint(id)
                    }
                }
                .keyboardShortcut(".", modifiers: .command)
                .disabled(runtime.store.selection == nil)

                Divider()

                Button("Manage Models…") { runtime.showModelManager() }
            }
        }

        Settings {
            SettingsView()
                .environment(runtime)
                .task { runtime.bootIfNeeded() }
        }
    }
}

/// Generation runs in-process (vendored MLX), so there are no worker
/// subprocesses to reap — the delegate only governs app lifecycle.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
