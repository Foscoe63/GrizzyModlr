import Foundation

/// The single pure transition function (DESIGN.md §3): every state change flows
/// through `reduce(state, event) -> effects`. No I/O in here — the runtime
/// executes the returned effects and feeds their results back as events.
///
/// Faithfulness notes (§4 tables implemented exactly; gaps resolved as follows):
/// - Engine stage events may skip intermediate states (an engine that doesn't
///   signal a stage passes through it in zero time); the walk only ever moves
///   FORWARD along the table's edges, never backward and never across.
/// - Paint `Preparing → Failed` (staging error, e.g. mesh file missing) is added,
///   mirroring §4.4 shape — the §4.5 table has no failure edge before Unwrapping.
/// - `Paused → Queued` happens when resume is requested while another model holds
///   the single download slot; the table's direct `Paused → Downloading` edge is
///   the slot-free case.
/// - Paint cancel is accepted ONLY from Rendering/Denoising, exactly per §4.5.
/// - Shape cancel is NOT accepted from Meshing/Committing, exactly per §4.4.
enum AppReducer {

    static func reduce(_ state: inout AppState, _ event: AppEvent) -> [AppEffect] {
        var effects: [AppEffect] = []
        let intentsBefore = installIntents(state)

        switch event {
        // MARK: - boot (§4.1)

        case .bootScanned(let legacy, let report):
            guard state.phase == .boot else { break }
            if legacy {
                state.phase = .migrating
                effects.append(.performMigration)
            } else {
                applyBootReport(&state, report)
            }

        case .modelsRescanned(let report):
            // Only between installs: an active/queued download owns its model states.
            guard state.phase == .ready, state.activeInstall == nil, state.installQueue.isEmpty else { break }
            applyModelStates(&state, report)

        case .migrationFinished(let report):
            guard state.phase == .migrating else { break }
            applyBootReport(&state, report)

        // MARK: - onboarding (§4.2)

        case .onboardingAdvanced:
            guard state.phase == .onboarding, state.onboarding.step == .welcome else { break }
            state.onboarding.step = .chooseModels

        case .onboardingStartDownload(let selection):
            guard state.phase == .onboarding, state.onboarding.step == .chooseModels,
                  !selection.isEmpty else { break }
            state.onboarding.selection = selection
            state.onboarding.step = .downloading
            for model in orderedModels(selection) {
                effects += enqueueInstall(&state, model)
            }
            effects += checkOnboardingComplete(&state)   // everything already installed

        case .onboardingSkipped:
            guard state.phase == .onboarding, state.onboarding.step != .done else { break }
            state.onboarding.step = .done
            state.phase = .ready
            effects.append(.persistOnboardingComplete)

        case .onboardingRetried:
            guard state.phase == .onboarding, case .failed = state.onboarding.step else { break }
            state.onboarding.step = .downloading
            for model in orderedModels(state.onboarding.selection)
            where !state.installState(model).isInstalled {
                effects += enqueueInstall(&state, model)
            }

        case .onboardingFinished:
            guard state.phase == .onboarding, state.onboarding.step == .done else { break }
            state.phase = .ready

        // MARK: - model install (§4.3)

        case .installRequested(let model), .installRetryRequested(let model),
             .installResumeRequested(let model):
            effects += enqueueInstall(&state, model)

        case .installPauseRequested(let model):
            guard case .downloading(let progress) = state.installState(model) else { break }
            state.models[model] = .paused(resumeBytes: progress.totalBytes)
            effects.append(.pauseDownload(model))
            if state.activeInstall == model { state.activeInstall = nil }
            effects += startNextInstallIfFree(&state)

        case .installRemoveRequested(let model):
            // Remove an installed model, or cancel a paused download — both
            // discard the install folder (partials included) and reset to zero.
            switch state.installState(model) {
            case .installed, .paused:
                state.models[model] = .notInstalled
                effects.append(.removeModelFiles(model))
            default:
                break
            }

        case .importWeightsRequested(let model, let folder):
            switch state.installState(model) {
            case .notInstalled, .failed, .paused:
                state.models[model] = .verifying
                effects.append(.importWeights(model, folder: folder))
            default:
                break
            }

        case .importWeightsFinished(let model, let error):
            guard state.installState(model) == .verifying, state.activeInstall != model else { break }
            state.models[model] = error.map { .failed($0) } ?? .installed
            effects += checkOnboardingComplete(&state)

        case .downloadProgressed(let model, let attempt, let progress):
            guard state.activeInstall == model, attempt == state.installAttempt,
                  case .downloading = state.installState(model) else { break }
            state.models[model] = .downloading(progress)

        case .downloadCompleted(let model, let attempt):
            guard state.activeInstall == model, attempt == state.installAttempt,
                  case .downloading = state.installState(model) else { break }
            state.models[model] = .verifying
            effects.append(.verifyFiles(model, attempt: attempt))

        case .downloadFailed(let model, let attempt, let message):
            guard state.activeInstall == model, attempt == state.installAttempt,
                  case .downloading = state.installState(model) else { break }
            state.models[model] = .failed(message)
            state.activeInstall = nil
            effects += startNextInstallIfFree(&state)
            effects += propagateOnboardingFailure(&state, model, message)

        case .verifyFinished(let model, let attempt, let error):
            guard state.activeInstall == model, attempt == state.installAttempt,
                  state.installState(model) == .verifying else { break }
            state.activeInstall = nil
            if let error {
                state.models[model] = .failed(error)     // mismatched file already deleted
                effects += startNextInstallIfFree(&state)
                effects += propagateOnboardingFailure(&state, model, error)
            } else {
                state.models[model] = .installed
                effects += startNextInstallIfFree(&state)
                effects += checkOnboardingComplete(&state)
            }

        // MARK: - shape job (§4.4)

        case .shapeGenerateRequested(let project, let model):
            switch state.shapeState(project) {
            case .idle, .failed: break
            default: return effects                      // already running — UI shows Stop
            }
            guard state.installState(model).isInstalled else {
                effects.append(.openModelManager)        // §4.9 weightsMissing
                break
            }
            let token = state.mintToken()
            state.shape[project] = ShapeJob(token: token, state: .preparing)
            effects.append(.stageShape(project: project, token: token))

        case .shapeStaged(let project, let token, let error):
            guard var job = state.shape[project], job.token == token else { break }
            switch job.state {
            case .preparing where error == nil:
                job.state = .waitingForEngine
                state.shape[project] = job
                effects += enqueueEngineJob(&state, JobKey(kind: .shape, project: project, token: token))
            case .preparing:
                job.state = .failed(stage: "Preparing", message: error ?? "Couldn't prepare the run.")
                state.shape[project] = job
                effects.append(.discardShapeStaging(project: project, token: token))
            case .cancelling:                            // cancel arrived while staging
                job.state = .idle
                state.shape[project] = job
                effects.append(.discardShapeStaging(project: project, token: token))
            default:
                break
            }

        case .shapeEngineStage(let project, let token, let stage, let fraction):
            guard var job = state.shape[project], job.token == token else { break }
            guard let next = advancedShapeState(from: job.state, to: stage) else { break }
            job.state = next
            job.fraction = fraction
            state.shape[project] = job

        case .shapeEngineFinished(let project, let token, let result):
            guard var job = state.shape[project], job.token == token else { break }
            let key = JobKey(kind: .shape, project: project, token: token)
            switch job.state {
            case .cancelling:
                job.state = .idle
                job.fraction = nil
                state.shape[project] = job
                effects.append(.discardShapeStaging(project: project, token: token))
                effects += releaseEngine(&state, key)
            case .loadingModel, .conditioning, .denoising, .decoding, .meshing:
                switch result {
                case .success:
                    job.state = .committing
                    state.shape[project] = job
                    effects += releaseEngine(&state, key)
                    effects.append(.commitShape(project: project, token: token))
                case .failure(let message):
                    job.state = .failed(stage: shapeStageLabel(job.state), message: message)
                    job.fraction = nil
                    state.shape[project] = job
                    effects.append(.discardShapeStaging(project: project, token: token))
                    effects += releaseEngine(&state, key)
                case .cancelled:
                    job.state = .idle
                    job.fraction = nil
                    state.shape[project] = job
                    effects.append(.discardShapeStaging(project: project, token: token))
                    effects += releaseEngine(&state, key)
                }
            default:
                break
            }

        case .shapeCommitFinished(let project, let token, let error):
            guard var job = state.shape[project], job.token == token,
                  job.state == .committing else { break }
            if let error {
                job.state = .failed(stage: "Committing", message: error)
                state.shape[project] = job
                effects.append(.discardShapeStaging(project: project, token: token))
            } else {
                job.state = .idle                        // Generation appended (Done badge)
                job.fraction = nil
                state.shape[project] = job
            }

        case .shapeCancelRequested(let project):
            guard var job = state.shape[project] else { break }
            let key = JobKey(kind: .shape, project: project, token: job.token)
            switch job.state {
            case .preparing:
                job.state = .cancelling                  // resolves when shapeStaged lands
                state.shape[project] = job
            case .waitingForEngine:
                job.state = .idle                        // dequeue
                job.fraction = nil
                state.shape[project] = job
                state.engine.queue.removeAll { $0 == key }
                effects.append(.discardShapeStaging(project: project, token: job.token))
                effects += releaseEngine(&state, key)    // no-op unless a grant is in flight
            case .loadingModel, .conditioning, .denoising, .decoding:
                job.state = .cancelling                  // takes effect after load returns
                state.shape[project] = job
                effects.append(.cancelEngine(key))
            default:
                break                                    // no cancel edge (§4.4)
            }

        case .shapeFailureDismissed(let project):
            guard var job = state.shape[project], case .failed = job.state else { break }
            job.state = .idle
            job.fraction = nil
            state.shape[project] = job

        // MARK: - paint job (§4.5)

        case .paintRequested(let project, let model):
            switch state.paintState(project) {
            case .idle, .failed: break
            default: return effects
            }
            guard state.installState(model).isInstalled else {
                effects.append(.openModelManager)
                break
            }
            let token = state.mintToken()
            state.paint[project] = PaintJob(token: token, state: .preparing)
            effects.append(.stagePaint(project: project, token: token))

        case .paintStaged(let project, let token, let error):
            guard var job = state.paint[project], job.token == token,
                  job.state == .preparing else { break }
            if let error {
                job.state = .failed(stage: "Preparing", message: error)
                state.paint[project] = job
                effects.append(.discardPaintStaging(project: project, token: token))
            } else {
                job.state = .unwrapping
                state.paint[project] = job
                effects.append(.unwrapPaintMesh(project: project, token: token))
            }

        case .paintUnwrapFinished(let project, let token, let error):
            guard var job = state.paint[project], job.token == token,
                  job.state == .unwrapping else { break }
            if let error {
                job.state = .failed(stage: "Unwrapping", message: error)
                state.paint[project] = job
                effects.append(.discardPaintStaging(project: project, token: token))
            } else {
                job.state = .waitingForEngine
                state.paint[project] = job
                effects += enqueueEngineJob(&state, JobKey(kind: .paint, project: project, token: token))
            }

        case .paintEngineStage(let project, let token, let stage, let fraction):
            guard var job = state.paint[project], job.token == token else { break }
            guard let next = advancedPaintState(from: job.state, to: stage) else { break }
            job.state = next
            job.fraction = fraction
            state.paint[project] = job

        case .paintEngineFinished(let project, let token, let result):
            guard var job = state.paint[project], job.token == token else { break }
            let key = JobKey(kind: .paint, project: project, token: token)
            switch job.state {
            case .cancelling:
                job.state = .idle
                job.fraction = nil
                state.paint[project] = job
                effects.append(.discardPaintStaging(project: project, token: token))
                effects += releaseEngine(&state, key)
            case .loadingModel, .rendering, .denoising, .decoding, .upscaling, .baking, .inpainting:
                switch result {
                case .success:
                    job.state = .committing
                    state.paint[project] = job
                    effects += releaseEngine(&state, key)
                    effects.append(.commitPaint(project: project, token: token))
                case .failure(let message):
                    job.state = .failed(stage: paintStageLabel(job.state), message: message)
                    job.fraction = nil
                    state.paint[project] = job
                    effects.append(.discardPaintStaging(project: project, token: token))
                    effects += releaseEngine(&state, key)
                case .cancelled:
                    job.state = .idle
                    job.fraction = nil
                    state.paint[project] = job
                    effects.append(.discardPaintStaging(project: project, token: token))
                    effects += releaseEngine(&state, key)
                }
            default:
                break
            }

        case .paintCommitFinished(let project, let token, let error):
            guard var job = state.paint[project], job.token == token,
                  job.state == .committing else { break }
            if let error {
                job.state = .failed(stage: "Committing", message: error)
                state.paint[project] = job
                effects.append(.discardPaintStaging(project: project, token: token))
            } else {
                job.state = .idle
                job.fraction = nil
                state.paint[project] = job
            }

        case .paintCancelRequested(let project):
            guard var job = state.paint[project] else { break }
            switch job.state {
            case .rendering, .denoising:                 // the only cancel edges in §4.5
                job.state = .cancelling
                state.paint[project] = job
                effects.append(.cancelEngine(JobKey(kind: .paint, project: project, token: job.token)))
            default:
                break
            }

        case .paintFailureDismissed(let project):
            guard var job = state.paint[project], case .failed = job.state else { break }
            job.state = .idle
            job.fraction = nil
            state.paint[project] = job

        // MARK: - engine arbiter (§4.6)

        case .engineGranted(let key):
            let valid: Bool
            switch key.kind {
            case .shape:
                valid = state.shape[key.project]?.token == key.token
                    && state.shapeState(key.project) == .waitingForEngine
            case .paint:
                valid = state.paint[key.project]?.token == key.token
                    && state.paintState(key.project) == .waitingForEngine
            }
            guard valid else {
                // Job was cancelled while the grant was in flight. The arbiter
                // physically performed this grant, so it must always be told to
                // release the stale key (idempotent there); reducer-side
                // ownership may already have moved on.
                effects.append(.releaseEngine(key))
                if state.engine.owner == .held(key) {
                    state.engine.owner = .free
                    effects += grantNextIfFree(&state)
                }
                break
            }
            switch key.kind {
            case .shape:
                state.shape[key.project]!.state = .loadingModel
                effects.append(.startShapeEngine(project: key.project, token: key.token))
            case .paint:
                state.paint[key.project]!.state = .loadingModel
                effects.append(.startPaintEngine(project: key.project, token: key.token))
            }
        }

        // Persist install intents whenever the unfinished-install set changes,
        // so a relaunch can re-derive Paused/resumable models.
        let intentsAfter = installIntents(state)
        if intentsAfter != intentsBefore {
            effects.append(.persistInstallIntents(intentsAfter))
        }
        return effects
    }

    // MARK: - helpers (all pure)

    private static func applyBootReport(_ state: inout AppState, _ report: BootReport) {
        applyModelStates(&state, report)
        if report.onboardingComplete || !report.installed.isEmpty {
            state.phase = .ready
        } else {
            state.phase = .onboarding
            state.onboarding = OnboardingState()
        }
    }

    private static func applyModelStates(_ state: inout AppState, _ report: BootReport) {
        for model in ModelID.allCases {
            if report.installed.contains(model) {
                state.models[model] = .installed
            } else if let bytes = report.resumable[model] {
                state.models[model] = .paused(resumeBytes: bytes)
            } else {
                state.models[model] = .notInstalled
            }
        }
    }

    /// Deterministic ordering for a selection set (catalog order).
    private static func orderedModels(_ selection: Set<ModelID>) -> [ModelID] {
        ModelID.allCases.filter { selection.contains($0) }
    }

    /// NotInstalled/Failed → Queued (install/retry) and Paused → Downloading|Queued
    /// (resume), then claim the single download slot if it's free.
    private static func enqueueInstall(_ state: inout AppState, _ model: ModelID) -> [AppEffect] {
        switch state.installState(model) {
        case .notInstalled, .failed, .paused:
            state.models[model] = .queued
            if !state.installQueue.contains(model) { state.installQueue.append(model) }
            return startNextInstallIfFree(&state)
        default:
            return []                                    // already queued/active/installed
        }
    }

    /// Queued → Downloading when the app-wide slot frees up (one at a time).
    private static func startNextInstallIfFree(_ state: inout AppState) -> [AppEffect] {
        guard state.activeInstall == nil else { return [] }
        while let next = state.installQueue.first {
            state.installQueue.removeFirst()
            guard state.installState(next) == .queued else { continue }
            state.activeInstall = next
            state.installAttempt += 1
            var progress = DownloadProgress()
            progress.fileCount = ModelCatalog.model(next).files.count
            progress.totalExpected = ModelCatalog.model(next).totalBytes
            state.models[next] = .downloading(progress)
            return [.startDownload(next, attempt: state.installAttempt)]
        }
        return []
    }

    /// The set of models with an unfinished install (drives boot-time Paused).
    private static func installIntents(_ state: AppState) -> Set<ModelID> {
        Set(ModelID.allCases.filter { model in
            switch state.installState(model) {
            case .queued, .downloading, .paused, .verifying: return true
            default: return false
            }
        })
    }

    private static func checkOnboardingComplete(_ state: inout AppState) -> [AppEffect] {
        guard state.phase == .onboarding, state.onboarding.step == .downloading,
              !state.onboarding.selection.isEmpty,
              state.onboarding.selection.allSatisfy({ state.installState($0).isInstalled })
        else { return [] }
        state.onboarding.step = .done
        return [.persistOnboardingComplete]
    }

    private static func propagateOnboardingFailure(
        _ state: inout AppState, _ model: ModelID, _ message: String
    ) -> [AppEffect] {
        guard state.phase == .onboarding, state.onboarding.step == .downloading,
              state.onboarding.selection.contains(model) else { return [] }
        state.onboarding.step = .failed(message)
        return []
    }

    // MARK: engine queue

    /// FIFO-append a job that reached WaitingForEngine; grant if the engine is free.
    private static func enqueueEngineJob(_ state: inout AppState, _ key: JobKey) -> [AppEffect] {
        state.engine.queue.append(key)
        return grantNextIfFree(&state)
    }

    private static func grantNextIfFree(_ state: inout AppState) -> [AppEffect] {
        guard state.engine.owner == .free else { return [] }
        while let next = state.engine.queue.first {
            let waiting: Bool
            switch next.kind {
            case .shape:
                waiting = state.shape[next.project]?.token == next.token
                    && state.shapeState(next.project) == .waitingForEngine
            case .paint:
                waiting = state.paint[next.project]?.token == next.token
                    && state.paintState(next.project) == .waitingForEngine
            }
            guard waiting else {
                state.engine.queue.removeFirst()         // defensively skip stale entries
                continue
            }
            state.engine.queue.removeFirst()             // granted jobs leave the queue
            state.engine.owner = .held(next)
            return [.grantEngine(next)]
        }
        return []
    }

    /// Held → Free when the given job owns the engine, then grant the next in FIFO.
    private static func releaseEngine(_ state: inout AppState, _ key: JobKey) -> [AppEffect] {
        guard state.engine.owner == .held(key) else { return [] }
        state.engine.owner = .free
        return [.releaseEngine(key)] + grantNextIfFree(&state)
    }

    // MARK: forward-walk stage maps

    /// §4.4 chain: LoadingModel → Conditioning → Denoising → Decoding → Meshing.
    private static func advancedShapeState(from current: ShapeJobState, to stage: ShapeStage)
    -> ShapeJobState? {
        func ordinal(_ s: ShapeJobState) -> Int? {
            switch s {
            case .loadingModel: return 0
            case .conditioning: return 1
            case .denoising: return 2
            case .decoding: return 3
            case .meshing: return 4
            default: return nil                          // not in an engine stage
            }
        }
        let target: ShapeJobState
        switch stage {
        case .conditioning: target = .conditioning
        case .denoising(let step, let total): target = .denoising(step: step, total: total)
        case .decoding: target = .decoding
        case .meshing: target = .meshing
        }
        guard let cur = ordinal(current), let tgt = ordinal(target), tgt >= cur else { return nil }
        if case .denoising = current, case .denoising = target { return target }   // step k/N
        return tgt > cur ? target : nil                  // same-ordinal repeats are dropped
    }

    /// §4.5 chain: LoadingModel → Rendering → Denoising → Decoding → Upscaling →
    /// Baking → Inpainting.
    private static func advancedPaintState(from current: PaintJobState, to stage: PaintStage)
    -> PaintJobState? {
        func ordinal(_ s: PaintJobState) -> Int? {
            switch s {
            case .loadingModel: return 0
            case .rendering: return 1
            case .denoising: return 2
            case .decoding: return 3
            case .upscaling: return 4
            case .baking: return 5
            case .inpainting: return 6
            default: return nil
            }
        }
        let target: PaintJobState
        switch stage {
        case .rendering: target = .rendering
        case .denoising(let step, let total): target = .denoising(step: step, total: total)
        case .decoding: target = .decoding
        case .upscaling: target = .upscaling
        case .baking: target = .baking
        case .inpainting: target = .inpainting
        }
        guard let cur = ordinal(current), let tgt = ordinal(target), tgt >= cur else { return nil }
        if case .denoising = current, case .denoising = target { return target }
        return tgt > cur ? target : nil
    }

    // MARK: stage labels (§4.9 engineFailed(stage, message) + status pills)

    static func shapeStageLabel(_ s: ShapeJobState) -> String {
        switch s {
        case .idle: return "Idle"
        case .preparing: return "Preparing"
        case .waitingForEngine: return "Waiting for engine"
        case .loadingModel: return "Loading model"
        case .conditioning: return "Conditioning"
        case .denoising: return "Denoising"
        case .decoding: return "Decoding"
        case .meshing: return "Meshing"
        case .committing: return "Committing"
        case .cancelling: return "Cancelling"
        case .failed: return "Failed"
        }
    }

    static func paintStageLabel(_ s: PaintJobState) -> String {
        switch s {
        case .idle: return "Idle"
        case .preparing: return "Preparing"
        case .unwrapping: return "Unwrapping"
        case .waitingForEngine: return "Waiting for engine"
        case .loadingModel: return "Loading model"
        case .rendering: return "Rendering"
        case .denoising: return "Denoising"
        case .decoding: return "Decoding"
        case .upscaling: return "Upscaling"
        case .baking: return "Baking"
        case .inpainting: return "Inpainting"
        case .committing: return "Committing"
        case .cancelling: return "Cancelling"
        case .failed: return "Failed"
        }
    }
}
