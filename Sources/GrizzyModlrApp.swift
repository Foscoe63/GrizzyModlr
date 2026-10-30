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

                Button("Batch Generate from Images…") { chooseBatchImages() }
                    .disabled(runtime.batch.running)
                if let summary = runtime.batch.summary {
                    Text(summary)
                    if runtime.batch.running {
                        Button("Cancel Batch") { runtime.batch.cancel(runtime: runtime) }
                    } else {
                        Button("Clear Batch") { runtime.batch.clearFinished() }
                    }
                }

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

extension GrizzyModlrApp {
    /// Picks images and queues a shape (and, per Settings, paint) run for each.
    @MainActor fileprivate func chooseBatchImages() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.message = "Choose images — one project is created and generated per image."
        guard panel.runModal() == .OK else { return }
        runtime.batch.start(images: panel.urls, paint: ToolDefaults.batchPaint(), runtime: runtime)
    }
}

/// Generation runs in-process (vendored MLX), so there are no worker
/// subprocesses to reap — the delegate only governs app lifecycle.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
