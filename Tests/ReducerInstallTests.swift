import XCTest

/// §4.3 model install — per-model machine plus the app-wide single-download-slot
/// rule, resume/pause paths, verify outcomes, and boot reconciliation.
final class ReducerInstallTests: XCTestCase {

    private func readyState() -> AppState { .ready(installed: []) }

    @discardableResult
    private func reduce(_ state: inout AppState, _ event: AppEvent) -> [AppEffect] {
        AppReducer.reduce(&state, event)
    }

    // MARK: install → queued → downloading

    func testInstallClaimsTheFreeSlot() {
        var s = readyState()
        let effects = reduce(&s, .installRequested(.shapeSmall))
        XCTAssertEqual(effects, [.startDownload(.shapeSmall, attempt: 1),
                                 .persistInstallIntents([.shapeSmall])])
        guard case .downloading(let p) = s.installState(.shapeSmall) else {
            return XCTFail("expected downloading")
        }
        XCTAssertEqual(p.fileCount, ModelCatalog.model(.shapeSmall).files.count)
        XCTAssertEqual(p.totalExpected, ModelCatalog.model(.shapeSmall).totalBytes)
        XCTAssertEqual(s.activeInstall, .shapeSmall)
    }

    func testSecondInstallQueuesBehindTheFirst() {
        var s = readyState()
        reduce(&s, .installRequested(.shapeSmall))
        let effects = reduce(&s, .installRequested(.paintSmall))
        XCTAssertEqual(effects, [.persistInstallIntents([.shapeSmall, .paintSmall])])
        XCTAssertEqual(s.installState(.paintSmall), .queued)
        XCTAssertEqual(s.installQueue, [.paintSmall])
    }

    func testDuplicateInstallRequestIsIdempotent() {
        var s = readyState()
        reduce(&s, .installRequested(.shapeSmall))
        XCTAssertEqual(reduce(&s, .installRequested(.shapeSmall)), [])
        XCTAssertEqual(s.installQueue, [])
    }

    // MARK: progress / completion / verify

    func testProgressUpdatesPayload() {
        var s = readyState()
        reduce(&s, .installRequested(.shapeSmall))
        var p = DownloadProgress()
        p.fileIndex = 2; p.fileCount = 2; p.fileBytes = 100; p.fileTotal = 200
        p.totalBytes = 300; p.totalExpected = 400; p.currentFileName = "model.fp16.safetensors"
        reduce(&s, .downloadProgressed(.shapeSmall, attempt: 1, progress: p))
        XCTAssertEqual(s.installState(.shapeSmall), .downloading(p))
    }

    func testStaleAttemptProgressIsDropped() {
        var s = readyState()
        reduce(&s, .installRequested(.shapeSmall))
        var p = DownloadProgress()
        p.totalBytes = 999
        XCTAssertEqual(reduce(&s, .downloadProgressed(.shapeSmall, attempt: 7, progress: p)), [])
        if case .downloading(let cur) = s.installState(.shapeSmall) {
            XCTAssertNotEqual(cur.totalBytes, 999)
        } else {
            XCTFail("expected downloading")
        }
    }

    func testCompletionEntersVerifying() {
        var s = readyState()
        reduce(&s, .installRequested(.shapeSmall))
        let effects = reduce(&s, .downloadCompleted(.shapeSmall, attempt: 1))
        XCTAssertEqual(s.installState(.shapeSmall), .verifying)
        XCTAssertEqual(effects, [.verifyFiles(.shapeSmall, attempt: 1)])
    }

    func testVerifyOKInstallsAndStartsNextInQueue() {
        var s = readyState()
        reduce(&s, .installRequested(.shapeSmall))
        reduce(&s, .installRequested(.paintSmall))
        reduce(&s, .downloadCompleted(.shapeSmall, attempt: 1))
        let effects = reduce(&s, .verifyFinished(.shapeSmall, attempt: 1, error: nil))
        XCTAssertEqual(s.installState(.shapeSmall), .installed)
        XCTAssertEqual(s.installState(.paintSmall), .downloading(paintSmallInitialProgress()))
        XCTAssertEqual(effects, [.startDownload(.paintSmall, attempt: 2),
                                 .persistInstallIntents([.paintSmall])])
    }

    private func paintSmallInitialProgress() -> DownloadProgress {
        var p = DownloadProgress()
        p.fileCount = ModelCatalog.model(.paintSmall).files.count
        p.totalExpected = ModelCatalog.model(.paintSmall).totalBytes
        return p
    }

    func testVerifyMismatchFails() {
        var s = readyState()
        reduce(&s, .installRequested(.shapeSmall))
        reduce(&s, .downloadCompleted(.shapeSmall, attempt: 1))
        reduce(&s, .verifyFinished(.shapeSmall, attempt: 1, error: "model.fp16.safetensors failed checksum verification."))
        XCTAssertEqual(s.installState(.shapeSmall),
                       .failed("model.fp16.safetensors failed checksum verification."))
        XCTAssertNil(s.activeInstall)
    }

    // MARK: failure / retry

    func testDownloadFailureThenRetryRequeues() {
        var s = readyState()
        reduce(&s, .installRequested(.shapeSmall))
        reduce(&s, .downloadFailed(.shapeSmall, attempt: 1, message: "HTTP 500"))
        XCTAssertEqual(s.installState(.shapeSmall), .failed("HTTP 500"))
        XCTAssertNil(s.activeInstall)

        let effects = reduce(&s, .installRetryRequested(.shapeSmall))
        XCTAssertEqual(effects, [.startDownload(.shapeSmall, attempt: 2),
                                 .persistInstallIntents([.shapeSmall])])
        if case .downloading = s.installState(.shapeSmall) {} else { XCTFail("expected downloading") }
    }

    func testStaleFailureAfterRetryIsDropped() {
        var s = readyState()
        reduce(&s, .installRequested(.shapeSmall))            // attempt 1
        reduce(&s, .downloadFailed(.shapeSmall, attempt: 1, message: "x"))
        reduce(&s, .installRetryRequested(.shapeSmall))       // attempt 2
        XCTAssertEqual(reduce(&s, .downloadFailed(.shapeSmall, attempt: 1, message: "late")), [])
        if case .downloading = s.installState(.shapeSmall) {} else { XCTFail("expected downloading") }
    }

    // MARK: pause / resume

    func testPauseKeepsBytesAndFreesTheSlot() {
        var s = readyState()
        reduce(&s, .installRequested(.shapeSmall))
        reduce(&s, .installRequested(.paintSmall))
        var p = DownloadProgress(); p.totalBytes = 12_345
        reduce(&s, .downloadProgressed(.shapeSmall, attempt: 1, progress: p))

        let effects = reduce(&s, .installPauseRequested(.shapeSmall))
        XCTAssertEqual(s.installState(.shapeSmall), .paused(resumeBytes: 12_345))
        // Pause frees the slot; the queued model starts.
        XCTAssertEqual(effects, [.pauseDownload(.shapeSmall),
                                 .startDownload(.paintSmall, attempt: 2)])
        XCTAssertEqual(s.activeInstall, .paintSmall)
    }

    func testPauseHasNoEdgeFromQueuedOrVerifying() {
        var s = readyState()
        reduce(&s, .installRequested(.shapeSmall))
        reduce(&s, .installRequested(.paintSmall))
        XCTAssertEqual(reduce(&s, .installPauseRequested(.paintSmall)), [])
        XCTAssertEqual(s.installState(.paintSmall), .queued)

        reduce(&s, .downloadCompleted(.shapeSmall, attempt: 1))
        XCTAssertEqual(reduce(&s, .installPauseRequested(.shapeSmall)), [])
        XCTAssertEqual(s.installState(.shapeSmall), .verifying)
    }

    func testResumeWithFreeSlotGoesStraightToDownloading() {
        var s = readyState()
        reduce(&s, .installRequested(.shapeSmall))
        reduce(&s, .installPauseRequested(.shapeSmall))
        let effects = reduce(&s, .installResumeRequested(.shapeSmall))
        // Paused and downloading are both "unfinished" — the intent set is
        // unchanged, so only the download start is effected.
        XCTAssertEqual(effects, [.startDownload(.shapeSmall, attempt: 2)])
        if case .downloading = s.installState(.shapeSmall) {} else { XCTFail("expected downloading") }
    }

    func testResumeWhileSlotBusyParksInQueued() {
        var s = readyState()
        reduce(&s, .installRequested(.shapeSmall))
        reduce(&s, .installPauseRequested(.shapeSmall))
        reduce(&s, .installRequested(.paintSmall))            // takes the slot
        let effects = reduce(&s, .installResumeRequested(.shapeSmall))
        XCTAssertEqual(s.installState(.shapeSmall), .queued)
        XCTAssertEqual(effects, [])                           // intent set unchanged
    }

    // MARK: remove

    func testRemoveOnlyFromInstalled() {
        var s = AppState.ready(installed: [.shapeSmall])
        let effects = reduce(&s, .installRemoveRequested(.shapeSmall))
        XCTAssertEqual(s.installState(.shapeSmall), .notInstalled)
        XCTAssertEqual(effects, [.removeModelFiles(.shapeSmall)])

        // Removing a paused download cancels it and discards the partials.
        var s2 = readyState()
        reduce(&s2, .installRequested(.paintSmall))
        reduce(&s2, .installPauseRequested(.paintSmall))
        XCTAssertEqual(reduce(&s2, .installRemoveRequested(.paintSmall)),
                       [.removeModelFiles(.paintSmall), .persistInstallIntents([])])
        XCTAssertEqual(s2.installState(.paintSmall), .notInstalled)

        // No remove edge from a never-installed model.
        var s3 = readyState()
        XCTAssertEqual(reduce(&s3, .installRemoveRequested(.paintSmall)), [])
    }

    // MARK: import (offline install)

    func testImportVerifiesThenInstalls() {
        var s = readyState()
        let folder = URL(fileURLWithPath: "/tmp/weights")
        let effects = reduce(&s, .importWeightsRequested(.paintLarge, folder: folder))
        XCTAssertEqual(s.installState(.paintLarge), .verifying)
        XCTAssertEqual(effects, [.importWeights(.paintLarge, folder: folder),
                                 .persistInstallIntents([.paintLarge])])

        let done = reduce(&s, .importWeightsFinished(.paintLarge, error: nil))
        XCTAssertEqual(s.installState(.paintLarge), .installed)
        XCTAssertEqual(done, [.persistInstallIntents([])])
    }

    func testImportFailureLandsInFailed() {
        var s = readyState()
        reduce(&s, .importWeightsRequested(.paintLarge, folder: URL(fileURLWithPath: "/tmp/w")))
        reduce(&s, .importWeightsFinished(.paintLarge, error: "unet/config.json is missing or has the wrong size."))
        XCTAssertEqual(s.installState(.paintLarge),
                       .failed("unet/config.json is missing or has the wrong size."))
    }

    func testImportIgnoredWhileDownloading() {
        var s = readyState()
        reduce(&s, .installRequested(.shapeSmall))
        XCTAssertEqual(reduce(&s, .importWeightsRequested(.shapeSmall, folder: URL(fileURLWithPath: "/x"))), [])
        if case .downloading = s.installState(.shapeSmall) {} else { XCTFail("expected downloading") }
    }

    // MARK: boot reconciliation (§4.1 + §4.3 re-derive from disk)

    func testBootAppliesInstalledAndResumable() {
        var s = AppState()
        _ = AppReducer.reduce(&s, .bootScanned(legacyLayoutDetected: false, report: BootReport(
            installed: [.shapeSmall],
            resumable: [.paintSmall: 1_234],
            onboardingComplete: true)))
        XCTAssertEqual(s.phase, .ready)
        XCTAssertEqual(s.installState(.shapeSmall), .installed)
        XCTAssertEqual(s.installState(.paintSmall), .paused(resumeBytes: 1_234))
        XCTAssertEqual(s.installState(.shapeLarge), .notInstalled)
    }

    func testBootFirstRunEntersOnboarding() {
        var s = AppState()
        _ = AppReducer.reduce(&s, .bootScanned(legacyLayoutDetected: false,
                                               report: BootReport()))
        XCTAssertEqual(s.phase, .onboarding)
        XCTAssertEqual(s.onboarding.step, .welcome)
    }

    func testBootWithModelsButNoMarkerGoesReady() {
        var s = AppState()
        _ = AppReducer.reduce(&s, .bootScanned(legacyLayoutDetected: false, report: BootReport(
            installed: [.shapeLarge], resumable: [:], onboardingComplete: false)))
        XCTAssertEqual(s.phase, .ready)
    }

    func testLegacyLayoutRoutesThroughMigrating() {
        var s = AppState()
        let effects = AppReducer.reduce(&s, .bootScanned(legacyLayoutDetected: true,
                                                         report: BootReport()))
        XCTAssertEqual(s.phase, .migrating)
        XCTAssertEqual(effects, [.performMigration])

        _ = AppReducer.reduce(&s, .migrationFinished(report: BootReport(
            installed: [.shapeSmall], resumable: [:], onboardingComplete: false)))
        XCTAssertEqual(s.phase, .ready)
        XCTAssertEqual(s.installState(.shapeSmall), .installed)
    }

    func testMigrationCanStillLandInOnboarding() {
        var s = AppState()
        _ = AppReducer.reduce(&s, .bootScanned(legacyLayoutDetected: true, report: BootReport()))
        _ = AppReducer.reduce(&s, .migrationFinished(report: BootReport()))
        XCTAssertEqual(s.phase, .onboarding)
    }

    // MARK: rescan after the models folder changes

    func testRescanRederivesInstallStatesWhenIdle() {
        var s = AppState.ready(installed: [.shapeSmall])
        let report = BootReport(installed: [.paintSmall], resumable: [.shapeLarge: 1234], onboardingComplete: true)
        // The paused model becomes a persisted install intent (resume after relaunch).
        XCTAssertEqual(reduce(&s, .modelsRescanned(report: report)), [.persistInstallIntents([.shapeLarge])])
        XCTAssertEqual(s.installState(.shapeSmall), .notInstalled)
        XCTAssertEqual(s.installState(.paintSmall), .installed)
        XCTAssertEqual(s.installState(.shapeLarge), .paused(resumeBytes: 1234))
        XCTAssertEqual(s.phase, .ready)
    }

    func testRescanIgnoredWhileADownloadIsActive() {
        var s = readyState()
        reduce(&s, .installRequested(.shapeSmall))
        reduce(&s, .modelsRescanned(report: BootReport(installed: [.paintLarge], onboardingComplete: true)))
        XCTAssertEqual(s.installState(.paintLarge), .notInstalled)
        if case .downloading = s.installState(.shapeSmall) {} else { XCTFail("download should be untouched") }
    }
}
