import Foundation

/// Everything that can change AppState. Events come from the UI (intents), the
/// download manager, the engine arbiter, and the engines' callbacks. Job events
/// carry the job token; the reducer drops any event whose token is stale.
enum AppEvent: Equatable {
    // MARK: boot (§4.1)

    /// The runtime scanned Application Support at launch.
    case bootScanned(legacyLayoutDetected: Bool, report: BootReport)
    /// Legacy `models/shape/**`+`models/paint/**` were migrated (§2.3); fresh scan attached.
    case migrationFinished(report: BootReport)

    /// The models folder changed (moved / pointed at another drive); re-derive every model's
    /// install state from the new location.
    case modelsRescanned(report: BootReport)

    // MARK: onboarding (§4.2)

    case onboardingAdvanced                          // Welcome → ChooseModels
    case onboardingStartDownload(Set<ModelID>)       // ChooseModels → Downloading
    case onboardingSkipped                           // "Later" from any step
    case onboardingRetried                           // Failed → Downloading
    case onboardingFinished                          // Done screen dismissed → Ready

    // MARK: model install (§4.3)

    case installRequested(ModelID)
    case installPauseRequested(ModelID)
    case installResumeRequested(ModelID)
    case installRemoveRequested(ModelID)
    case installRetryRequested(ModelID)
    /// User picked a local folder to import (offline install).
    case importWeightsRequested(ModelID, folder: URL)
    case importWeightsFinished(ModelID, error: String?)

    /// Byte progress from the download manager (attempt guards staleness).
    case downloadProgressed(ModelID, attempt: UInt64, progress: DownloadProgress)
    /// Every file fetched → Verifying.
    case downloadCompleted(ModelID, attempt: UInt64)
    case downloadFailed(ModelID, attempt: UInt64, message: String)
    case verifyFinished(ModelID, attempt: UInt64, error: String?)

    // MARK: shape job (§4.4)

    /// UI intent; carries the shape model the project resolves to so the reducer
    /// can gate on weights without reading the project store.
    case shapeGenerateRequested(project: UUID, model: ModelID)
    case shapeStaged(project: UUID, token: UInt64, error: String?)
    case shapeEngineStage(project: UUID, token: UInt64, stage: ShapeStage, fraction: Double?)
    case shapeEngineFinished(project: UUID, token: UInt64, result: EngineResult)
    case shapeCommitFinished(project: UUID, token: UInt64, error: String?)
    case shapeCancelRequested(project: UUID)
    case shapeFailureDismissed(project: UUID)

    // MARK: paint job (§4.5)

    case paintRequested(project: UUID, model: ModelID)
    case paintStaged(project: UUID, token: UInt64, error: String?)
    /// Pre-engine unwrap outcome. Today's engine performs xatlas inside the run,
    /// so the runtime reports an immediate pass-through success; wave 2b moves the
    /// real QEM-decimate + xatlas work here.
    case paintUnwrapFinished(project: UUID, token: UInt64, error: String?)
    case paintEngineStage(project: UUID, token: UInt64, stage: PaintStage, fraction: Double?)
    case paintEngineFinished(project: UUID, token: UInt64, result: EngineResult)
    case paintCommitFinished(project: UUID, token: UInt64, error: String?)
    case paintCancelRequested(project: UUID)
    case paintFailureDismissed(project: UUID)

    // MARK: engine arbiter (§4.6)

    /// The arbiter finished the residency handoff (other engine evicted).
    case engineGranted(JobKey)
}

/// Terminal outcome of an engine run.
enum EngineResult: Equatable {
    case success
    case failure(String)
    case cancelled
}

/// What the boot scan found on disk (plus persisted markers).
struct BootReport: Equatable {
    /// Models whose every catalog file is present with the exact byte size.
    var installed: Set<ModelID> = []
    /// Models with leftover `.partial` bytes and/or a persisted install intent —
    /// they boot as Paused (resume from byte offset) per §4.3 "app quit".
    var resumable: [ModelID: Int64] = [:]
    /// The onboarding-complete marker.
    var onboardingComplete: Bool = false
}
