import Hy3DMLX
import Foundation
import Observation

/// The impure shell around the pure core: owns AppState, feeds every event
/// through AppReducer, and executes the returned effects against the real world
/// (ProjectStore files, DownloadManager network, EngineArbiter + engines).
/// Views observe this object; all UI intents come through `dispatch` or the
/// convenience methods below.
@MainActor
@Observable
final class AppRuntime {
    private(set) var state = AppState()
    let store: ProjectStore

    /// Latest streamed preview mesh per project (shape denoise). Transient view
    /// data, token-guarded against stale engine callbacks.
    private(set) var shapePreviews: [Project.ID: URL] = [:]
    /// Latest streamed multiview grid per project (paint denoise).
    private(set) var paintViewPreviews: [Project.ID: URL] = [:]
    /// Bumped when generation is requested without weights — the UI routes to
    /// Settings → Models (§4.9 weightsMissing: never a bare error).
    private(set) var modelManagerSignal = 0

    /// What the most recent run per project was configured with (recorded at staging,
    /// kept after failure) — feeds the failure-details popover (§4.9 engineFailed).
    struct RunDetails: Equatable {
        var model: String
        var seed: UInt64?
    }
    private(set) var lastShapeRunDetails: [Project.ID: RunDetails] = [:]
    private(set) var lastPaintRunDetails: [Project.ID: RunDetails] = [:]

    /// Zero123++ view synthesis runs outside the reducer (a short, user-initiated job that
    /// only ever runs while no shape or paint job holds the GPU).
    enum ViewSynthState: Equatable {
        case running(Double)
        case failed(String)
    }
    private(set) var viewSynth: [Project.ID: ViewSynthState] = [:]

    /// Moving the models folder to another drive (copy → verify → switch → delete old copy).
    enum RelocationState: Equatable {
        case running(copied: Int64, total: Int64, file: String)
        case failed(String)
    }
    private(set) var relocation: RelocationState?
    @ObservationIgnored private var relocationFlag: RelocationFlag?
    /// Bumped each time a synthesis finishes so views can show the fresh candidates.
    private(set) var viewSynthRevision: [Project.ID: Int] = [:]

    @ObservationIgnored private let shapeEngine = ShapeEngine()
    @ObservationIgnored private let paintEngine = PaintEngine()
    @ObservationIgnored private let viewSynthEngine = ViewSynthEngine()
    @ObservationIgnored private var viewSynthRuns: [Project.ID: ViewSynthEngine.Run] = [:]
    @ObservationIgnored private let arbiter: EngineArbiter
    @ObservationIgnored private var downloads: DownloadManager!
    @ObservationIgnored private let bridge = EventBridge()

    @ObservationIgnored private var stagedShapes: [Project.ID: (token: UInt64, run: ProjectStore.StagedShapeRun)] = [:]
    @ObservationIgnored private var stagedPaints: [Project.ID: (token: UInt64, run: ProjectStore.StagedPaintRun)] = [:]
    @ObservationIgnored private var engineRuns: [JobKey: any CancellableRun] = [:]

    @ObservationIgnored private var eventQueue: [AppEvent] = []
    @ObservationIgnored private var draining = false
    @ObservationIgnored private var booted = false

    private static let onboardingKey = "onboardingComplete"
    private static let intentsKey = "installIntents"

    init(store: ProjectStore? = nil) {
        self.store = store ?? ProjectStore()
        let shape = shapeEngine
        let paint = paintEngine
        arbiter = EngineArbiter(evictors: [
            .shape: { await shape.evictAndWait() },
            .paint: { await paint.evictAndWait() },
        ])
        downloads = DownloadManager(
            configuration: .init(rootDir: ModelStore.modelsRoot),
            emit: { [bridge] event in bridge.send(event) })
        bridge.runtime = self
    }

    /// Forwards events from background subsystems onto the main actor without
    /// retaining the runtime before it finishes initializing.
    final class EventBridge: @unchecked Sendable {
        weak var runtime: AppRuntime?
        func send(_ event: AppEvent) {
            Task { @MainActor in self.runtime?.dispatch(event) }
        }
    }

    // MARK: - boot (§4.1)

    func bootIfNeeded() {
        guard !booted else { return }
        booted = true
        let downloads = downloads!
        Task.detached(priority: .userInitiated) { [bridge] in
            let legacy = ModelStore.legacyLayoutExists
            let report = Self.scanBootReport(downloads: downloads)
            bridge.send(.bootScanned(legacyLayoutDetected: legacy, report: report))
        }
    }

    private nonisolated static func scanBootReport(downloads: DownloadManager) -> BootReport {
        var report = BootReport()
        report.onboardingComplete = UserDefaults.standard.bool(forKey: onboardingKey)
        let intents = Set((UserDefaults.standard.stringArray(forKey: intentsKey) ?? [])
            .compactMap(ModelID.init(rawValue:)))
        for model in ModelID.allCases {
            if downloads.isInstalledOnDisk(model) {
                report.installed.insert(model)
            } else if intents.contains(model) || downloads.hasPartialData(for: model) {
                report.resumable[model] = downloads.resumableBytes(for: model)
            }
        }
        return report
    }

    // MARK: - dispatch loop

    /// FIFO event processing: effects may synchronously produce follow-up events
    /// (e.g. staging), which queue behind the current reduction instead of
    /// recursing — keeps ordering deterministic.
    func dispatch(_ event: AppEvent) {
        eventQueue.append(event)
        guard !draining else { return }
        draining = true
        while !eventQueue.isEmpty {
            let next = eventQueue.removeFirst()
            let effects = AppReducer.reduce(&state, next)
            for effect in effects { perform(effect) }
        }
        draining = false
    }

    // MARK: - UI conveniences

    func requestGenerate(_ id: Project.ID) {
        guard let project = store.project(id) else { return }
        let model: ModelID = project.extraViews.isEmpty ? project.resolvedSettings.model.modelID : .shapeMultiView
        dispatch(.shapeGenerateRequested(project: id, model: model))
    }

    func requestPaint(_ id: Project.ID) {
        guard let project = store.project(id) else { return }
        dispatch(.paintRequested(project: id, model: project.resolvedPaintSettings.model.modelID))
    }

    /// Generates six candidate side views for the project's main image.
    func synthesizeViews(_ id: Project.ID) {
        guard viewSynth[id] == nil, let project = store.project(id), let image = store.imageURL(for: project) else { return }
        guard let weights = ModelStore.viewSynthesisDir() else { showModelManager(); return }
        let busy = state.shape.values.contains { $0.state.isRunning } || state.paint.values.contains { $0.state.isRunning }
        guard !busy else {
            viewSynth[id] = .failed("Wait for the current generation to finish first.")
            return
        }
        viewSynth[id] = .running(0)
        let outputs = store.synthViewURLs(for: id)
        let engine = viewSynthEngine, shape = shapeEngine, paint = paintEngine
        Task { [weak self] in
            await shape.evictAndWait()                    // free the other engines' weights first
            await paint.evictAndWait()
            guard let self else { return }
            let run = engine.synthesize(
                imageURL: image, weightsDir: weights, seed: UInt64.random(in: 0 ..< UInt64(UInt32.max)),
                outputs: outputs,
                onProgress: { p in Task { @MainActor in
                    if case .running = self.viewSynth[id] { self.viewSynth[id] = .running(p) }
                } },
                onFinish: { outcome in Task { @MainActor in
                    self.viewSynthRuns[id] = nil
                    switch outcome {
                    case .success:
                        self.viewSynth[id] = nil
                        self.viewSynthRevision[id, default: 0] += 1
                    case .failure(let message):
                        self.viewSynth[id] = message == "Cancelled" ? nil : .failed(message)
                    }
                } })
            self.viewSynthRuns[id] = run
        }
    }

    func cancelViewSynthesis(_ id: Project.ID) { viewSynthRuns[id]?.cancel() }
    func dismissViewSynthFailure(_ id: Project.ID) { if case .failed = viewSynth[id] { viewSynth[id] = nil } }

    // MARK: - models location

    /// Moving or re-pointing the library is only safe while nothing reads or writes it.
    var canChangeModelsLocation: Bool {
        if case .running? = relocation { return false }
        return state.activeInstall == nil && state.installQueue.isEmpty
            && !state.shape.values.contains { $0.state.isRunning }
            && !state.paint.values.contains { $0.state.isRunning }
            && viewSynth.values.allSatisfy { if case .running = $0 { return false } else { return true } }
    }

    /// Copies the library into `picked` (a "Modelr Models" folder is created inside it), then
    /// switches to it and removes the old copy once every file is verified.
    func moveModels(toFolder picked: URL) {
        guard canChangeModelsLocation else { return }
        let dst = ModelRelocation.resolveRoot(picked: picked)
        startRelocation(from: ModelStore.modelsRoot, to: dst,
                        customAfter: dst.standardizedFileURL == ModelStore.defaultRoot.standardizedFileURL ? nil : dst)
    }

    /// Uses a folder that already holds models (e.g. copied earlier, or from another Mac).
    func useExistingModels(at picked: URL) {
        guard canChangeModelsLocation else { return }
        let nested = picked.appendingPathComponent("Modelr Models", isDirectory: true)
        let root: URL
        if !ModelRelocation.items(in: picked).isEmpty { root = picked }
        else if !ModelRelocation.items(in: nested).isEmpty { root = nested }
        else { relocation = .failed("No models were found in that folder."); return }
        switchModelsRoot(to: root)
    }

    /// Back to the built-in location; optionally copies the library back first.
    func resetModelsLocation(moveBack: Bool) {
        guard canChangeModelsLocation, ModelStore.customRoot != nil else { return }
        if moveBack, ModelStore.rootIsAvailable {
            startRelocation(from: ModelStore.modelsRoot, to: ModelStore.defaultRoot, customAfter: nil)
        } else {
            switchModelsRoot(to: ModelStore.defaultRoot)
        }
    }

    func cancelRelocation() { relocationFlag?.cancel() }
    func dismissRelocationFailure() { if case .failed? = relocation { relocation = nil } }

    private func switchModelsRoot(to root: URL) {
        let isDefault = root.standardizedFileURL == ModelStore.defaultRoot.standardizedFileURL
        ModelStore.setCustomRoot(isDefault ? nil : root)
        downloads.setRootDir(root)
        relocation = nil
        bridge.send(.modelsRescanned(report: Self.scanBootReport(downloads: downloads)))
    }

    private func startRelocation(from src: URL, to dst: URL, customAfter: URL?) {
        guard src.standardizedFileURL != dst.standardizedFileURL else {
            relocation = .failed("That's already the models folder."); return
        }
        guard ModelStore.rootIsAvailable else { relocation = .failed("The models drive isn't connected."); return }
        let items = ModelRelocation.items(in: src)
        guard !items.isEmpty else { switchModelsRoot(to: customAfter ?? ModelStore.defaultRoot); return }
        let total = items.reduce(0) { $0 + $1.bytes }
        let free = ModelRelocation.freeBytes(at: dst)
        let needed = ModelRelocation.sameVolume(src, dst) ? 0 : total      // a same-volume copy is a clone
        guard needed <= free else {
            relocation = .failed(ModelRelocation.Failure.notEnoughSpace(needed: needed, free: free).localizedDescription)
            return
        }
        relocation = .running(copied: 0, total: total, file: "")
        let flag = RelocationFlag()
        relocationFlag = flag
        Task.detached(priority: .userInitiated) { [weak self] in
            var lastUpdate = Date.distantPast
            do {
                try FileManager.default.createDirectory(at: dst, withIntermediateDirectories: true)
                try ModelRelocation.copy(items, from: src, to: dst, progress: { done, name in
                    let now = Date()
                    guard now.timeIntervalSince(lastUpdate) > 0.15 || done == total else { return }
                    lastUpdate = now
                    Task { @MainActor in
                        if case .running? = self?.relocation { self?.relocation = .running(copied: done, total: total, file: name) }
                    }
                }, isCancelled: { flag.cancelled })
                await MainActor.run { self?.switchModelsRoot(to: customAfter ?? dst) }
                ModelRelocation.removeSources(items, from: src)            // only after the switch
            } catch let f as ModelRelocation.Failure where f == .cancelled {
                await MainActor.run { self?.relocation = nil }
            } catch {
                await MainActor.run { self?.relocation = .failed(error.localizedDescription) }
            }
        }
    }

    func cancelShape(_ id: Project.ID) { dispatch(.shapeCancelRequested(project: id)) }
    func cancelPaint(_ id: Project.ID) { dispatch(.paintCancelRequested(project: id)) }

    /// Cancel whatever is legal to cancel, then remove the project + its files.
    func deleteProject(_ id: Project.ID) {
        dispatch(.shapeCancelRequested(project: id))
        dispatch(.paintCancelRequested(project: id))
        shapePreviews[id] = nil
        paintViewPreviews[id] = nil
        lastShapeRunDetails[id] = nil
        lastPaintRunDetails[id] = nil
        store.delete(id)
    }

    func install(_ model: ModelID) { dispatch(.installRequested(model)) }
    func pauseInstall(_ model: ModelID) { dispatch(.installPauseRequested(model)) }
    func resumeInstall(_ model: ModelID) { dispatch(.installResumeRequested(model)) }
    func removeInstall(_ model: ModelID) { dispatch(.installRemoveRequested(model)) }
    func retryInstall(_ model: ModelID) { dispatch(.installRetryRequested(model)) }
    func importWeights(_ model: ModelID, from folder: URL) {
        dispatch(.importWeightsRequested(model, folder: folder))
    }

    func showModelManager() { modelManagerSignal += 1 }

    var shapeJobs: [Project.ID: ShapeJob] { state.shape }
    var paintJobs: [Project.ID: PaintJob] { state.paint }

    // MARK: - effect execution

    private func perform(_ effect: AppEffect) {
        switch effect {
        // boot / persistence
        case .performMigration:
            let downloads = downloads!
            Task.detached(priority: .userInitiated) { [bridge] in
                ModelStore.performLegacyMigration()
                let report = Self.scanBootReport(downloads: downloads)
                bridge.send(.migrationFinished(report: report))
            }

        case .persistOnboardingComplete:
            UserDefaults.standard.set(true, forKey: Self.onboardingKey)

        case .persistInstallIntents(let intents):
            UserDefaults.standard.set(intents.map(\.rawValue).sorted(), forKey: Self.intentsKey)

        // model install
        case .startDownload(let model, let attempt):
            guard ModelStore.rootIsAvailable else {
                dispatch(.downloadFailed(model, attempt: attempt, message: "The models drive isn't connected."))
                break
            }
            downloads.start(model, attempt: attempt)

        case .pauseDownload(let model):
            downloads.pause(model)

        case .verifyFiles(let model, let attempt):
            downloads.verify(model, attempt: attempt)

        case .removeModelFiles(let model):
            downloads.removeFiles(for: model)

        case .importWeights(let model, let folder):
            downloads.importWeights(for: model, from: folder)

        // shape job
        case .stageShape(let project, let token):
            do {
                let staged = try store.stageShapeRun(for: project)
                stagedShapes[project] = (token, staged)
                shapePreviews[project] = nil
                lastShapeRunDetails[project] = RunDetails(model: staged.settings.model.label,
                                                          seed: staged.seed)
                dispatch(.shapeStaged(project: project, token: token, error: nil))
            } catch {
                dispatch(.shapeStaged(project: project, token: token,
                                      error: error.localizedDescription))
            }

        case .startShapeEngine(let project, let token):
            startShapeEngine(project: project, token: token)

        case .commitShape(let project, let token):
            guard let (stagedToken, staged) = stagedShapes[project], stagedToken == token else {
                dispatch(.shapeCommitFinished(project: project, token: token,
                                              error: "The run's staged files were lost."))
                break
            }
            stagedShapes[project] = nil
            let error = store.commitShapeRun(staged)
            shapePreviews[project] = nil
            store.clearStreamFiles(for: project)
            dispatch(.shapeCommitFinished(project: project, token: token, error: error))

        case .discardShapeStaging(let project, let token):
            if let (stagedToken, staged) = stagedShapes[project], stagedToken == token {
                stagedShapes[project] = nil
                store.discardShapeRun(staged)
            }
            shapePreviews[project] = nil
            store.clearStreamFiles(for: project)

        // paint job
        case .stagePaint(let project, let token):
            do {
                let staged = try store.stagePaintRun(for: project)
                stagedPaints[project] = (token, staged)
                paintViewPreviews[project] = nil
                // The resolved paint seed (the initial-noise seed the pipeline runs
                // with) — recorded on the version and shown in the failure details.
                lastPaintRunDetails[project] = RunDetails(model: staged.settings.model.label,
                                                          seed: staged.seed)
                dispatch(.paintStaged(project: project, token: token, error: nil))
            } catch {
                dispatch(.paintStaged(project: project, token: token,
                                      error: error.localizedDescription))
            }

        case .unwrapPaintMesh(let project, let token):
            // §4.5 prep: QEM-decimate the shape mesh to the run's face budget before
            // the engine, whose internal xatlas unwrap + rasterizer cost scales with
            // triangle count (the known #1 perf issue — ~238 s to paint a 240k-vert
            // mesh, dominated by unwrap+raster). On decimation failure the original
            // mesh is painted instead — slow but correct. The pipeline still unwraps
            // internally; this stage owns the face budget + early mesh validation.
            guard let (stagedToken, staged) = stagedPaints[project], stagedToken == token else {
                dispatch(.paintUnwrapFinished(project: project, token: token,
                                              error: "The run's staged files were lost."))
                break
            }
            let src = staged.shapeMesh
            let dst = staged.prepMesh
            let budget = staged.settings.faces
            Task.detached(priority: .userInitiated) { [bridge] in
                let outcome = MeshDecimator.decimateMeshFile(at: src, faceBudget: budget, to: dst)
                let error: String?
                switch outcome {
                case .decimated, .unchanged:
                    error = nil
                case .fallback:
                    // Undecimated fallback (§4.5): scrub any partial prep file so
                    // the engine picks up the original mesh.
                    try? FileManager.default.removeItem(at: dst)
                    error = nil
                case .unreadable(let message):
                    error = message                      // genuinely bad mesh → Failed
                }
                bridge.send(.paintUnwrapFinished(project: project, token: token, error: error))
            }

        case .startPaintEngine(let project, let token):
            startPaintEngine(project: project, token: token)

        case .commitPaint(let project, let token):
            guard let (stagedToken, staged) = stagedPaints[project], stagedToken == token else {
                dispatch(.paintCommitFinished(project: project, token: token,
                                              error: "The run's staged files were lost."))
                break
            }
            stagedPaints[project] = nil
            let error = store.commitPaintRun(staged)
            paintViewPreviews[project] = nil
            store.clearPaintStreamFiles(for: project)
            dispatch(.paintCommitFinished(project: project, token: token, error: error))

        case .discardPaintStaging(let project, let token):
            if let (stagedToken, staged) = stagedPaints[project], stagedToken == token {
                stagedPaints[project] = nil
                store.discardPaintRun(staged)
            }
            paintViewPreviews[project] = nil
            store.clearPaintStreamFiles(for: project)

        // engine arbiter
        case .grantEngine(let key):
            Task { [arbiter, bridge] in
                await arbiter.grant(key)                   // evict-then-grant, sequenced
                bridge.send(.engineGranted(key))
            }

        case .cancelEngine(let key):
            engineRuns[key]?.cancel()

        case .releaseEngine(let key):
            engineRuns[key] = nil
            Task { [arbiter] in await arbiter.release(key) }

        // UX routing
        case .openModelManager:
            modelManagerSignal += 1
        }
    }

    // MARK: - engine adapters (callbacks → token-carrying events)

    private func startShapeEngine(project: Project.ID, token: UInt64) {
        let key = JobKey(kind: .shape, project: project, token: token)
        guard let (stagedToken, staged) = stagedShapes[project], stagedToken == token else {
            dispatch(.shapeEngineFinished(project: project, token: token,
                                          result: .failure("The run's staged files were lost.")))
            return
        }
        let multiView = !staged.extraViews.isEmpty
        guard let weights = multiView ? ModelStore.multiViewWeightsFile()
                                      : ModelStore.shapeWeightsFile(for: staged.settings.model) else {
            let name = multiView ? "Multi-view" : staged.settings.model.label
            dispatch(.shapeEngineFinished(project: project, token: token,
                                          result: .failure("\(name) weights are no longer installed.")))
            return
        }
        let angles: [ReferenceView: ViewAngle] = [.left: .left, .back: .back, .right: .right]
        let extraViews = ReferenceView.allCases.compactMap { v in
            staged.extraViews[v].flatMap { url in angles[v].map { (angle: $0, url: url) } }
        }
        let bridge = bridge
        var run: ShapeEngine.Run!
        run = shapeEngine.generate(
            imageURL: staged.input,
            extraViews: extraViews,
            output: staged.mesh,
            weightsURL: weights,
            quantize: staged.settings.quant.flag,
            steps: staged.settings.steps,
            guidance: Float(staged.settings.guidance),
            resolution: staged.settings.octree,
            seed: staged.seed,
            onProgress: { stage, _, fraction in
                if let mapped = Self.mapShapeStage(stage) {
                    bridge.send(.shapeEngineStage(project: project, token: token,
                                                  stage: mapped, fraction: fraction))
                }
            },
            onPreview: { url in
                Task { @MainActor in
                    guard let self = bridge.runtime,
                          self.state.shape[project]?.token == token else { return }
                    self.shapePreviews[project] = url
                }
            },
            onFinish: { outcome in
                let result: EngineResult
                switch outcome {
                case .success: result = .success
                case .failure(let message):
                    result = run.cancelled ? .cancelled : .failure(message)
                }
                bridge.send(.shapeEngineFinished(project: project, token: token, result: result))
            })
        engineRuns[key] = run
    }

    private func startPaintEngine(project: Project.ID, token: UInt64) {
        let key = JobKey(kind: .paint, project: project, token: token)
        guard let (stagedToken, staged) = stagedPaints[project], stagedToken == token else {
            dispatch(.paintEngineFinished(project: project, token: token,
                                          result: .failure("The run's staged files were lost.")))
            return
        }
        guard let weightsRoot = ModelStore.paintWeightsRoot(for: staged.settings.model) else {
            dispatch(.paintEngineFinished(project: project, token: token,
                                          result: .failure("Paint weights are no longer installed.")))
            return
        }
        let bridge = bridge
        let viewsDir = staged.outMesh.deletingLastPathComponent()
        let onProgress: (String, Double?) -> Void = { stage, fraction in
            if let mapped = Self.mapPaintStage(stage) {
                bridge.send(.paintEngineStage(project: project, token: token,
                                              stage: mapped, fraction: fraction))
            }
        }
        let onViews: (URL) -> Void = { url in
            Task { @MainActor in
                guard let self = bridge.runtime,
                      self.state.paint[project]?.token == token else { return }
                self.paintViewPreviews[project] = url
            }
        }
        var run: PaintEngine.Run!
        let onFinish: (GenerationOutcome) -> Void = { outcome in
            let result: EngineResult
            switch outcome {
            case .success: result = .success
            case .failure(let message):
                result = run.cancelled ? .cancelled : .failure(message)
            }
            bridge.send(.paintEngineFinished(project: project, token: token, result: result))
        }
        // Large → 2.1 PBR (albedo + metallic-roughness); Small → 2.0 Color (RGB).
        if staged.settings.model == .large {
            run = paintEngine.paintPBR(
                meshURL: staged.engineMesh, imageURL: staged.image,
                output: staged.outMesh, texture: staged.outTexture, mrTexture: staged.outMR,
                weightsRoot: weightsRoot,
                res: staged.settings.res, steps: staged.settings.steps, tex: staged.settings.tex,
                superres: staged.settings.superres, seed: staged.seed, viewsDir: viewsDir,
                onProgress: onProgress, onViews: onViews, onFinish: onFinish)
        } else {
            run = paintEngine.paint(
                meshURL: staged.engineMesh,      // decimated prep mesh when present (§4.5)
                imageURL: staged.image,
                output: staged.outMesh, texture: staged.outTexture,
                weightsRoot: weightsRoot,
                res: staged.settings.res, steps: staged.settings.steps, tex: staged.settings.tex,
                superres: staged.settings.superres, seed: staged.seed, viewsDir: viewsDir,
                onProgress: onProgress, onViews: onViews, onFinish: onFinish)
        }
        engineRuns[key] = run
    }

    // MARK: - engine progress → §4.4/§4.5 stages

    /// Hy3DMLX emits: "Conditioning image", "Denoising (k/N)", "Decoding shape",
    /// "Building mesh", "Done" (+ "Loading model…" on cache miss).
    static func mapShapeStage(_ stage: String) -> ShapeStage? {
        if stage.hasPrefix("Conditioning") { return .conditioning }
        if stage.hasPrefix("Denoising") {
            let (step, total) = parseProgress(stage)
            return .denoising(step: step, total: total)
        }
        if stage.hasPrefix("Decoding") { return .decoding }
        if stage.hasPrefix("Building mesh") { return .meshing }
        return nil                              // Loading model… / Done
    }

    /// HunyuanPaintMLX emits: "Loading paint model", "Unwrapping UVs", "Rendering
    /// control maps", "Painting (k/N)", "Decoding views", "Super-resolving",
    /// "Baking texture", "Done". "Unwrapping UVs" arrives with weights already
    /// resident, so it marks the Rendering stage boundary (§4.5 LoadingModel →
    /// Rendering: weights resident); inpainting has no separate engine signal and
    /// is passed through on completion.
    static func mapPaintStage(_ stage: String) -> PaintStage? {
        if stage.hasPrefix("Unwrapping") || stage.hasPrefix("Rendering") { return .rendering }
        if stage.hasPrefix("Painting") {
            let (step, total) = parseProgress(stage)
            return .denoising(step: step, total: total)
        }
        if stage.hasPrefix("Decoding") { return .decoding }
        if stage.hasPrefix("Super-resolving") { return .upscaling }
        if stage.hasPrefix("Baking") { return .baking }
        return nil                              // Loading paint model / Done
    }

    /// Extract "(k/N)" from an engine stage string; (0, 0) when absent.
    private static func parseProgress(_ stage: String) -> (Int, Int) {
        guard let open = stage.firstIndex(of: "("), let close = stage.firstIndex(of: ")"),
              open < close else { return (0, 0) }
        let parts = stage[stage.index(after: open)..<close].split(separator: "/")
        guard parts.count == 2, let k = Int(parts[0]), let n = Int(parts[1]) else { return (0, 0) }
        return (k, n)
    }
}

/// Cooperative cancel flag for a models-folder copy.
final class RelocationFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var _cancelled = false
    var cancelled: Bool { lock.lock(); defer { lock.unlock() }; return _cancelled }
    func cancel() { lock.lock(); _cancelled = true; lock.unlock() }
}
