import AppKit
import Foundation
import UniformTypeIdentifiers

/// Env-triggered self-drive harness (DESIGN.md §8.4). When `MODELR_UI_SMOKE=1`,
/// the app walks its own real flow — onboarding → project → import image →
/// generate → paint → export GLB — capturing its window at key states, then
/// exits 0. Any failure prints to stderr and exits nonzero. No external UI
/// driving is involved: the runner dispatches the same events the UI would.
///
/// Environment:
///   MODELR_UI_SMOKE=1            enable
///   MODELR_SMOKE_OUT=<dir>       artifacts (screenshots + exported GLB); required
///   MODELR_SMOKE_MODEL=small|large|fast   shape+paint pair (default small; fast = TripoSR shape + small paint)
///   MODELR_SMOKE_IMAGE=<png>     demo image to import (required unless download mode)
///   MODELR_SMOKE_IMPORT=<root>   folder holding <root>/{shape-small,…} weight
///                                bundles; onboarding is completed by importing
///                                the needed pair from here ("import mode")
///   MODELR_SMOKE_DOWNLOAD=<slug> download-only mode: install one model through
///                                the real DownloadManager (resume-aware), then
///                                exit — exercises §4.3 against the live repo
@MainActor
enum SmokeRunner {
    private static let env = ProcessInfo.processInfo.environment

    static var isRequested: Bool { env["MODELR_UI_SMOKE"] == "1" }

    static func startIfRequested(runtime: AppRuntime) {
        guard isRequested else { return }
        Task { @MainActor in
            do {
                try await run(runtime: runtime)
                log("PASS")
                exit(0)
            } catch {
                capture("99-failure")
                FileHandle.standardError.write(Data("[smoke] FAIL: \(error.localizedDescription)\n".utf8))
                exit(1)
            }
        }
    }

    private struct SmokeError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    private static func fail(_ message: String) -> SmokeError { SmokeError(message: message) }

    private static func log(_ message: String) {
        print("[smoke] \(message)")
        fflush(stdout)
    }

    // MARK: - main walk

    private static func run(runtime: AppRuntime) async throws {
        guard let outPath = env["MODELR_SMOKE_OUT"], !outPath.isEmpty else {
            throw fail("MODELR_SMOKE_OUT is required")
        }
        outDir = URL(fileURLWithPath: outPath, isDirectory: true)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        // 1. Boot settles (scan → migrate → onboarding|ready).
        try await waitUntil("boot", 30) {
            runtime.state.phase == .onboarding || runtime.state.phase == .ready
        }
        log("boot: phase=\(runtime.state.phase)")

        if let slug = env["MODELR_SMOKE_DOWNLOAD"], !slug.isEmpty {
            try await runDownloadMode(runtime: runtime, slug: slug)
            return
        }

        let modelChoice = env["MODELR_SMOKE_MODEL"] ?? "small"
        guard ["small", "large", "fast"].contains(modelChoice) else {
            throw fail("MODELR_SMOKE_MODEL must be small, large or fast (got \(modelChoice))")
        }
        let shapeModel: ShapeModel = modelChoice == "large" ? .large : (modelChoice == "fast" ? .fast : .small)
        let paintModel: PaintModel = modelChoice == "large" ? .large : .small
        let needed = [shapeModel.modelID, paintModel.modelID]

        // 2. Onboarding: capture the sheet, then complete it in import mode —
        //    weights come from a local folder through the real import/verify path
        //    (the import runs whether or not onboarding appeared, so a warm
        //    container can still add a missing pair).
        let onboardingShown = runtime.state.phase == .onboarding
        if onboardingShown {
            await settle()
            capture("01-onboarding")
        }
        if let importRoot = env["MODELR_SMOKE_IMPORT"] {
            let root = URL(fileURLWithPath: importRoot, isDirectory: true)
            for model in needed where !runtime.state.installState(model).isInstalled {
                log("importing \(model.rawValue) from \(root.path)")
                runtime.importWeights(model, from: root.appendingPathComponent(model.slug))
                try await waitUntil("import \(model.rawValue)", 1800) {
                    switch runtime.state.installState(model) {
                    case .installed, .failed: return true
                    default: return false
                    }
                }
                if case .failed(let message) = runtime.state.installState(model) {
                    throw fail("import \(model.rawValue) failed: \(message)")
                }
                log("imported \(model.rawValue)")
            }
        }
        if onboardingShown {
            runtime.dispatch(.onboardingSkipped)
            try await waitUntil("onboarding complete", 10) { runtime.state.phase == .ready }
            log("onboarding: complete")
        }

        for model in needed where !runtime.state.installState(model).isInstalled {
            throw fail("\(model.rawValue) is not installed; set MODELR_SMOKE_IMPORT or preinstall")
        }

        // 3. Project + demo image.
        guard let imagePath = env["MODELR_SMOKE_IMAGE"], !imagePath.isEmpty else {
            throw fail("MODELR_SMOKE_IMAGE is required")
        }
        let store = runtime.store
        let project = store.newProject()
        store.rename(project.id, to: "Smoke \(modelChoice)")
        store.setShapeModel(shapeModel, for: project.id)
        store.setPaintModel(paintModel, for: project.id)
        store.setImage(fromURL: URL(fileURLWithPath: imagePath), for: project.id)
        if let importError = store.importErrors[project.id] {
            throw fail("image import failed: \(importError)")
        }
        guard store.project(project.id)?.inputImageName != nil else {
            throw fail("image import produced no input.png")
        }
        log("project: created, image imported, model=\(modelChoice)")

        // 4. Generate (shape).
        runtime.requestGenerate(project.id)
        var capturedGenerating = false
        try await waitUntil("shape run", 1800) {
            let state = runtime.state.shapeState(project.id)
            if case .denoising = state, !capturedGenerating,
               runtime.shapePreviews[project.id] != nil {
                capturedGenerating = true
                capture("02-generating-preview")
            }
            if case .failed(let stage, let message) = state {
                lastFailure = "shape failed at \(stage): \(message)"
                return true
            }
            return state == .idle && store.project(project.id)?.currentGeneration?.kind == .shape
        }
        if let failure = lastFailure { throw fail(failure) }
        if !capturedGenerating { log("note: shape finished before a preview frame was captured") }
        guard let shapeGen = store.project(project.id)?.currentGeneration, shapeGen.kind == .shape else {
            throw fail("no shape generation was committed")
        }
        log(String(format: "shape: committed in %.1fs (seed %@)",
                   shapeGen.durationSeconds ?? -1,
                   shapeGen.seedRaw.map(String.init) ?? "?"))
        await settle()
        capture("03-shape-done")

        // 5. Paint.
        runtime.requestPaint(project.id)
        var capturedPainting = false
        try await waitUntil("paint run", 3600) {
            let state = runtime.state.paintState(project.id)
            if case .denoising = state, !capturedPainting,
               runtime.paintViewPreviews[project.id] != nil {
                capturedPainting = true
                capture("04-painting-views")
            }
            if case .failed(let stage, let message) = state {
                lastFailure = "paint failed at \(stage): \(message)"
                return true
            }
            return state == .idle && store.project(project.id)?.currentGeneration?.kind == .paint
        }
        if let failure = lastFailure { throw fail(failure) }
        guard let paintGen = store.project(project.id)?.currentGeneration, paintGen.kind == .paint else {
            throw fail("no paint generation was committed")
        }
        log(String(format: "paint: committed in %.1fs (PBR %@)",
                   paintGen.durationSeconds ?? -1, paintGen.isPBR ? "yes" : "no"))
        if paintModel == .large && !paintGen.isPBR {
            throw fail("large paint committed without a metallic-roughness map")
        }
        await settle()
        capture("05-painted")

        // 6. Export GLB (same MeshExporter call the export menu makes).
        guard let projectNow = store.project(project.id),
              let content = store.currentViewerContent(for: projectNow) else {
            throw fail("no viewer content to export")
        }
        let dest = outDir.appendingPathComponent("modelr-\(modelChoice)-painted.glb")
        switch content {
        case .pbrMesh(let mesh, let albedo, let mr):
            try MeshExporter.export(meshURL: mesh, texture: albedo, metallicRoughness: mr,
                                    format: .glb, to: dest)
        case .texturedMesh(let mesh, let tex):
            try MeshExporter.export(meshURL: mesh, texture: tex, format: .glb, to: dest)
        case .mesh, .points:
            throw fail("viewer content is untextured after paint — expected a textured mesh")
        }
        let bytes = (try? dest.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        guard bytes > 8 else { throw fail("exported GLB is empty") }
        log("export: \(dest.lastPathComponent) (\(bytes) bytes)")
        capture("06-exported")
    }

    // MARK: - download-only mode (§4.3 against the live repo)

    private static func runDownloadMode(runtime: AppRuntime, slug: String) async throws {
        guard let model = ModelID(rawValue: slug) else {
            throw fail("MODELR_SMOKE_DOWNLOAD must be one of \(ModelID.allCases.map(\.rawValue))")
        }
        if runtime.state.phase == .onboarding {
            await settle()
            capture("01-onboarding")
            runtime.dispatch(.onboardingSkipped)
            try await waitUntil("onboarding complete", 10) { runtime.state.phase == .ready }
        }
        let initial = runtime.state.installState(model)
        log("download \(slug): initial state \(describe(initial))")
        switch initial {
        case .installed:
            log("download \(slug): already installed")
            return
        case .paused:
            runtime.resumeInstall(model)
        default:
            runtime.install(model)
        }

        var lastLogged = ""
        var lastProgressLog = Date.distantPast
        try await waitUntil("download \(slug)", 7200) {
            let state = runtime.state.installState(model)
            let line = describe(state)
            if case .downloading(let p) = state {
                // First progress line after a resume shows fileBytes > 0 — the
                // §4.3 Range-resume evidence.
                if line != lastLogged || Date().timeIntervalSince(lastProgressLog) > 5 {
                    log("download \(slug): file \(p.fileIndex)/\(p.fileCount) \(p.currentFileName) \(p.fileBytes)/\(p.fileTotal) total \(p.totalBytes)/\(p.totalExpected)")
                    lastProgressLog = Date()
                }
            } else if line != lastLogged {
                log("download \(slug): \(line)")
            }
            lastLogged = line
            switch state {
            case .installed, .failed: return true
            default: return false
            }
        }
        if case .failed(let message) = runtime.state.installState(model) {
            throw fail("download \(slug) failed: \(message)")
        }
        log("download \(slug): installed (size + sha256 verified)")
    }

    private static func describe(_ state: ModelInstallState) -> String {
        switch state {
        case .notInstalled: return "notInstalled"
        case .queued: return "queued"
        case .downloading(let p): return "downloading(file \(p.fileIndex)/\(p.fileCount))"
        case .paused(let bytes): return "paused(resumeBytes: \(bytes))"
        case .verifying: return "verifying"
        case .installed: return "installed"
        case .failed(let m): return "failed(\(m))"
        }
    }

    // MARK: - plumbing

    private static var outDir = URL(fileURLWithPath: NSTemporaryDirectory())
    private static var lastFailure: String?

    /// Poll on the main actor until `condition` or timeout.
    private static func waitUntil(_ what: String, _ timeout: TimeInterval,
                                  _ condition: () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() {
            guard Date() < deadline else { throw fail("timed out waiting for \(what) (\(Int(timeout))s)") }
            try? await Task.sleep(nanoseconds: 250_000_000)
        }
    }

    /// Give SwiftUI a couple of runloop turns so the state just reached is on screen.
    private static func settle() async {
        for _ in 0..<3 { try? await Task.sleep(nanoseconds: 150_000_000) }
    }

    /// Self-capture the app's own window (allowed without screen-recording
    /// permission for windows this process owns). Falls back to a view snapshot
    /// if the CG capture is unavailable.
    private static func capture(_ name: String) {
        guard let window = NSApp.windows.first(where: { $0.isVisible && $0.contentView != nil })
            ?? NSApp.windows.first else {
            log("capture \(name): no window")
            return
        }
        // Composite the window with any attached sheet (a sheet is its own child
        // NSWindow, so a single-window capture would show only the dimmed parent).
        var windowIDs: [CGWindowID] = [CGWindowID(window.windowNumber)]
        if let sheet = window.attachedSheet {
            windowIDs.append(CGWindowID(sheet.windowNumber))
        }
        // CGWindowListCreateImageFromArray wants raw window IDs in the CFArray
        // (not boxed numbers); first element = topmost, so the sheet leads.
        var rawIDs = windowIDs.reversed().map { UnsafeRawPointer(bitPattern: UInt($0)) }
        let idArray = CFArrayCreate(kCFAllocatorDefault, &rawIDs, rawIDs.count, nil)
        var cg = idArray.flatMap {
            CGImage(windowListFromArrayScreenBounds: .null, windowArray: $0,
                    imageOption: [.boundsIgnoreFraming, .bestResolution])
        }
        if cg == nil {
            cg = CGWindowListCreateImage(.null, .optionIncludingWindow,
                                         CGWindowID(window.windowNumber),
                                         [.boundsIgnoreFraming, .bestResolution])
        }
        if cg == nil, let view = window.contentView,
           let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) {
            view.cacheDisplay(in: view.bounds, to: rep)
            cg = rep.cgImage
        }
        guard let image = cg else {
            log("capture \(name): failed")
            return
        }
        let url = outDir.appendingPathComponent("\(name).png")
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            log("capture \(name): destination failed")
            return
        }
        CGImageDestinationAddImage(dest, image, nil)
        CGImageDestinationFinalize(dest)
        log("capture \(name): \(image.width)x\(image.height) -> \(url.lastPathComponent)")
    }
}
