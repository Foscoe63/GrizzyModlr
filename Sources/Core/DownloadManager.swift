import Foundation
import CryptoKit

/// Executes model downloads (DESIGN.md §4.3): sequential per model, one model at
/// a time app-wide (the reducer owns the queue; this manager only ever runs the
/// one model it's told to). Every file:
///
///   1. streams into `<file>.partial` (HTTP Range resume from the partial's byte
///      count — after pause, relaunch, or retry),
///   2. is sha256-hashed *while* streaming (no second read pass; on resume only
///      the already-present prefix is re-read to seed the hash),
///   3. on completion: hash checked against the catalog → fsync → atomic rename.
///      A mismatch deletes the bytes and fails the install.
///
/// Results are reported exclusively as AppEvents through `emit` (the runtime
/// forwards them into the reducer, which drops stale attempts by token).
final class DownloadManager: @unchecked Sendable {

    /// Test seams: catalog lookup, remote URL resolution, and the byte sink
    /// (production writes append-to-file; tests can inject quota failures).
    struct Configuration {
        var rootDir: URL                                   // …/Modelr/models
        var catalog: @Sendable (ModelID) -> CatalogModel = { ModelCatalog.model($0) }
        var remoteURL: @Sendable (CatalogModel, CatalogFile) -> URL = { $0.remoteURL(for: $1) }
        var makeSink: @Sendable (URL) throws -> DownloadSink = { try FileSink(url: $0) }
        var session: URLSession = .shared
        /// Emit a progress event at most every N bytes (plus file boundaries).
        var progressByteInterval: Int64 = 4 << 20
    }

    private let config: Configuration
    /// Mutable copy of `config.rootDir`: the models folder can be moved to another drive.
    private var rootDir: URL
    private let emit: @Sendable (AppEvent) -> Void

    private let lock = NSLock()
    private var currentTask: Task<Void, Never>?
    private var currentModel: ModelID?
    private var pausing = false
    /// Files (model/relPath) whose streamed hash matched this session — verify
    /// can skip re-reading them.
    private var sessionVerified = Set<String>()

    init(configuration: Configuration, emit: @escaping @Sendable (AppEvent) -> Void) {
        self.config = configuration
        self.rootDir = configuration.rootDir
        self.emit = emit
    }

    // MARK: - paths

    func installDir(for model: ModelID) -> URL {
        lock.lock(); let root = rootDir; lock.unlock()
        return root.appendingPathComponent(model.slug, isDirectory: true)
    }

    /// Points the manager at a different models folder (only while no download is active).
    func setRootDir(_ url: URL) {
        lock.lock(); rootDir = url; sessionVerified.removeAll(); lock.unlock()
    }

    private func finalURL(_ model: ModelID, _ file: CatalogFile) -> URL {
        installDir(for: model).appendingPathComponent(file.path)
    }

    private func partialURL(_ model: ModelID, _ file: CatalogFile) -> URL {
        installDir(for: model).appendingPathComponent(file.path + ".partial")
    }

    private static func fileSize(_ url: URL) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? nil
    }

    /// Bytes already on disk for a model: complete files + partials. Used for the
    /// boot report (resume offsets) and the Paused row's "x of y GB".
    func resumableBytes(for model: ModelID) -> Int64 {
        let cat = config.catalog(model)
        var total: Int64 = 0
        for file in cat.files {
            if let s = Self.fileSize(finalURL(model, file)), s == file.bytes { total += s }
            else if let p = Self.fileSize(partialURL(model, file)) { total += p }
        }
        return total
    }

    func hasPartialData(for model: ModelID) -> Bool {
        let cat = config.catalog(model)
        return cat.files.contains { Self.fileSize(partialURL(model, $0)) != nil }
    }

    /// True when every catalog file is present with the exact byte size (§4.3:
    /// installed-state is re-derived from disk by size check only).
    func isInstalledOnDisk(_ model: ModelID) -> Bool {
        config.catalog(model).files.allSatisfy { Self.fileSize(finalURL(model, $0)) == $0.bytes }
    }

    func bytesOnDisk(for model: ModelID) -> Int64 {
        config.catalog(model).files.reduce(0) { $0 + (Self.fileSize(finalURL(model, $1)) ?? 0) }
    }

    // MARK: - control (effect executors)

    /// `.startDownload` — spawn the sequential per-file download for one model.
    func start(_ model: ModelID, attempt: UInt64) {
        lock.lock()
        currentTask?.cancel()
        pausing = false
        currentModel = model
        lock.unlock()
        let task = Task.detached(priority: .utility) { [weak self] in
            guard let self else { return }
            await self.run(model, attempt: attempt)
        }
        lock.lock(); currentTask = task; lock.unlock()
    }

    /// `.pauseDownload` — cancel the transfer, keep `.partial` bytes on disk.
    func pause(_ model: ModelID) {
        lock.lock()
        guard currentModel == model else { lock.unlock(); return }
        pausing = true
        let task = currentTask
        lock.unlock()
        task?.cancel()
    }

    /// `.removeModelFiles` — delete the whole install folder (incl. partials).
    func removeFiles(for model: ModelID) {
        try? FileManager.default.removeItem(at: installDir(for: model))
        lock.lock()
        sessionVerified = sessionVerified.filter { !$0.hasPrefix("\(model.rawValue)/") }
        lock.unlock()
    }

    // MARK: - download loop

    private enum Failure: Error {
        case http(Int)
        case hashMismatch(String)
        case shortRead
        case message(String)
    }

    private func run(_ model: ModelID, attempt: UInt64) async {
        let cat = config.catalog(model)
        do {
            var completedBytes: Int64 = 0
            for (index, file) in cat.files.enumerated() {
                let final = finalURL(model, file)
                if Self.fileSize(final) == file.bytes {
                    completedBytes += file.bytes           // already present — skip
                    continue
                }
                try FileManager.default.createDirectory(
                    at: final.deletingLastPathComponent(), withIntermediateDirectories: true)
                try await downloadFile(model: model, catalog: cat, file: file,
                                       fileIndex: index, completedBytes: &completedBytes,
                                       attempt: attempt)
                lock.lock(); sessionVerified.insert("\(model.rawValue)/\(file.path)"); lock.unlock()
            }
            emit(.downloadCompleted(model, attempt: attempt))
        } catch is CancellationError {
            // pause() — reducer already moved the model to Paused; stay silent.
        } catch let urlError as URLError where urlError.code == .cancelled {
            // same: cooperative pause surfaces as a cancelled URL task
        } catch {
            let paused: Bool = { lock.lock(); defer { lock.unlock() }; return pausing }()
            if !paused {
                emit(.downloadFailed(model, attempt: attempt, message: describe(error)))
            }
        }
        lock.lock()
        if currentModel == model { currentModel = nil; currentTask = nil }
        lock.unlock()
    }

    private func downloadFile(model: ModelID, catalog cat: CatalogModel, file: CatalogFile,
                              fileIndex: Int, completedBytes: inout Int64,
                              attempt: UInt64) async throws {
        let partial = partialURL(model, file)
        let final = finalURL(model, file)
        let fm = FileManager.default

        // Seed the streaming hash with whatever partial bytes survive from a
        // previous attempt (pause / app quit / network drop).
        var hasher = SHA256()
        var offset: Int64 = 0
        if let existing = Self.fileSize(partial), existing > 0, existing <= file.bytes {
            offset = try Self.hashPrefix(of: partial, into: &hasher)
        } else if fm.fileExists(atPath: partial.path) {
            try? fm.removeItem(at: partial)                // oversized/corrupt partial
        }

        var request = URLRequest(url: config.remoteURL(cat, file))
        if offset > 0 { request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }

        let (stream, response) = try await httpStream(for: request)
        if offset > 0 && response.statusCode == 200 {      // server ignored Range → restart
            try? fm.removeItem(at: partial)
            offset = 0
            hasher = SHA256()
        } else if !(response.statusCode == 206 || response.statusCode == 200) {
            throw Failure.http(response.statusCode)
        }

        if !fm.fileExists(atPath: partial.path) {
            fm.createFile(atPath: partial.path, contents: nil)
        }
        let sink = try config.makeSink(partial)
        var written = offset
        var lastReported = written

        func report() {
            var p = DownloadProgress()
            p.fileIndex = fileIndex + 1
            p.fileCount = cat.files.count
            p.currentFileName = (file.path as NSString).lastPathComponent
            p.fileBytes = written
            p.fileTotal = file.bytes
            p.totalBytes = completedBytes + written
            p.totalExpected = cat.totalBytes
            emit(.downloadProgressed(model, attempt: attempt, progress: p))
        }
        report()

        do {
            for try await chunk in stream {
                try Task.checkCancellation()
                hasher.update(data: chunk)
                try sink.write(chunk)
                written += Int64(chunk.count)
                if written - lastReported >= config.progressByteInterval {
                    lastReported = written
                    report()
                }
            }
        } catch {
            try? sink.finalize()                            // keep the partial for resume
            throw error
        }

        guard written == file.bytes else {
            try? sink.finalize()
            throw Failure.shortRead
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard digest == file.sha256 else {
            try? sink.finalize()
            try? fm.removeItem(at: partial)                 // §4.3: mismatch → delete
            throw Failure.hashMismatch(file.path)
        }
        try sink.finalize()                                 // fsync
        try? fm.removeItem(at: final)
        try fm.moveItem(at: partial, to: final)             // atomic rename
        report()                                            // written == file.bytes here
        completedBytes += file.bytes
    }

    /// Read an existing partial through the hasher; returns its byte count.
    private static func hashPrefix(of url: URL, into hasher: inout SHA256) throws -> Int64 {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var total: Int64 = 0
        while let data = try handle.read(upToCount: 4 << 20), !data.isEmpty {
            hasher.update(data: data)
            total += Int64(data.count)
        }
        return total
    }

    // MARK: - verify (§4.3 Verifying)

    /// `.verifyFiles` — after the last file completes: every file must exist with
    /// the exact size, and every file's sha256 must be known-good. Files streamed
    /// this session were hashed on the wire; anything else (migrated / imported /
    /// pre-existing) is hashed now. A mismatch deletes the file and fails.
    func verify(_ model: ModelID, attempt: UInt64) {
        lock.lock(); currentModel = model; lock.unlock()
        let task = Task.detached(priority: .utility) { [self] in
            let cat = config.catalog(model)
            var error: String?
            for file in cat.files {
                let final = finalURL(model, file)
                guard Self.fileSize(final) == file.bytes else {
                    error = "\(file.path) is missing or truncated."
                    break
                }
                let key = "\(model.rawValue)/\(file.path)"
                let alreadyVerified: Bool = { lock.lock(); defer { lock.unlock() }
                                              return sessionVerified.contains(key) }()
                if alreadyVerified { continue }
                if let digest = try? Self.sha256(of: final), digest == file.sha256 {
                    lock.lock(); sessionVerified.insert(key); lock.unlock()
                } else {
                    try? FileManager.default.removeItem(at: final)
                    error = "\(file.path) failed checksum verification."
                    break
                }
            }
            lock.lock()
            if currentModel == model { currentModel = nil; currentTask = nil }
            lock.unlock()
            emit(.verifyFinished(model, attempt: attempt, error: error))
        }
        lock.lock(); currentTask = task; lock.unlock()
    }

    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let data = try handle.read(upToCount: 4 << 20), !data.isEmpty {
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - import (offline install)

    /// `.importWeights` — copy a user-picked folder into the slot. Files are
    /// size-validated before copying and hashed as they land (same §4.3 contract).
    func importWeights(for model: ModelID, from folder: URL) {
        Task.detached(priority: .utility) { [self] in
            let cat = config.catalog(model)
            let fm = FileManager.default
            let scoped = folder.startAccessingSecurityScopedResource()
            defer { if scoped { folder.stopAccessingSecurityScopedResource() } }

            for file in cat.files {                        // size validation first
                let src = folder.appendingPathComponent(file.path)
                guard Self.fileSize(src) == file.bytes else {
                    emit(.importWeightsFinished(
                        model, error: "\(file.path) is missing or has the wrong size."))
                    return
                }
            }
            do {
                for file in cat.files {
                    let src = folder.appendingPathComponent(file.path)
                    let dst = finalURL(model, file)
                    try fm.createDirectory(at: dst.deletingLastPathComponent(),
                                           withIntermediateDirectories: true)
                    try? fm.removeItem(at: dst)
                    try fm.copyItem(at: src, to: dst)
                    guard try Self.sha256(of: dst) == file.sha256 else {
                        try? fm.removeItem(at: dst)
                        emit(.importWeightsFinished(
                            model, error: "\(file.path) failed checksum verification."))
                        return
                    }
                    lock.lock(); sessionVerified.insert("\(model.rawValue)/\(file.path)"); lock.unlock()
                }
                emit(.importWeightsFinished(model, error: nil))
            } catch {
                emit(.importWeightsFinished(model, error: describe(error)))
            }
        }
    }

    // MARK: - transport

    /// Async chunked HTTP body via a data-task delegate (real Data chunks at wire
    /// speed — URLSession.AsyncBytes iterates per byte, too slow for multi-GB).
    private func httpStream(for request: URLRequest)
    async throws -> (AsyncThrowingStream<Data, Error>, HTTPURLResponse) {
        let streamer = ChunkStreamer(session: config.session, request: request)
        let response = try await streamer.start()
        return (streamer.chunks, response)
    }

    private func describe(_ error: Error) -> String {
        switch error {
        case Failure.http(let code): return "The server returned HTTP \(code)."
        case Failure.hashMismatch(let path): return "\(path) failed checksum verification."
        case Failure.shortRead: return "The connection ended before the file was complete."
        case Failure.message(let m): return m
        default: return error.localizedDescription
        }
    }
}

// MARK: - byte sinks

/// Destination for streamed download bytes; injectable so tests can simulate
/// disk-full deterministically.
protocol DownloadSink: Sendable {
    func write(_ data: Data) throws
    /// Flush to stable storage (fsync) and close.
    func finalize() throws
}

/// Production sink: appends to the partial file, fsyncs on finalize.
final class FileSink: DownloadSink, @unchecked Sendable {
    private let handle: FileHandle

    init(url: URL) throws {
        handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
    }

    func write(_ data: Data) throws {
        try handle.write(contentsOf: data)
    }

    func finalize() throws {
        try handle.synchronize()
        try handle.close()
    }
}

// MARK: - chunked transport

/// Bridges URLSessionDataTask delegate callbacks into an AsyncThrowingStream of
/// Data chunks, exposing the HTTPURLResponse before the body starts.
private final class ChunkStreamer: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    let chunks: AsyncThrowingStream<Data, Error>
    private let chunkContinuation: AsyncThrowingStream<Data, Error>.Continuation
    private let session: URLSession
    private let request: URLRequest
    private var task: URLSessionDataTask?
    private let lock = NSLock()
    private var responseContinuation: CheckedContinuation<HTTPURLResponse, Error>?

    init(session: URLSession, request: URLRequest) {
        var continuation: AsyncThrowingStream<Data, Error>.Continuation!
        chunks = AsyncThrowingStream { continuation = $0 }
        chunkContinuation = continuation
        self.session = session
        self.request = request
        super.init()
    }

    func start() async throws -> HTTPURLResponse {
        // A session per transfer so `self` can be its delegate without retain
        // cycles into a shared session; invalidated on completion.
        let owned = URLSession(configuration: session.configuration, delegate: self, delegateQueue: nil)
        let task = owned.dataTask(with: request)
        self.task = task
        // When the consuming task is cancelled (pause), the stream terminates —
        // stop the transfer immediately rather than letting it drain unseen.
        chunkContinuation.onTermination = { _ in task.cancel() }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { cont in
                lock.lock(); responseContinuation = cont; lock.unlock()
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        lock.lock()
        let cont = responseContinuation
        responseContinuation = nil
        lock.unlock()
        if let http = response as? HTTPURLResponse {
            cont?.resume(returning: http)
        } else {
            cont?.resume(throwing: URLError(.badServerResponse))
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        chunkContinuation.yield(data)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        lock.lock()
        let cont = responseContinuation
        responseContinuation = nil
        lock.unlock()
        if let error {
            cont?.resume(throwing: error)
            chunkContinuation.finish(throwing: error)
        } else {
            chunkContinuation.finish()
        }
        session.finishTasksAndInvalidate()
    }
}
