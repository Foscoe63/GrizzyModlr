import Foundation

/// Copies the installed model library to another folder (typically a faster external drive)
/// and removes the old copy only once every file is verified at the destination.
enum ModelRelocation {
    struct Item: Equatable {
        let relativePath: String          // e.g. "shape-small/model.fp16.safetensors"
        let bytes: Int64
    }

    enum Failure: LocalizedError, Equatable {
        case cancelled
        case notEnoughSpace(needed: Int64, free: Int64)
        case copyFailed(String)
        case verifyFailed(String)

        var errorDescription: String? {
            switch self {
            case .cancelled: return "Cancelled."
            case .notEnoughSpace(let n, let f):
                return "Not enough space on that drive: \(n / 1_000_000_000) GB needed, \(f / 1_000_000_000) GB free."
            case .copyFailed(let p): return "Couldn't copy \(p)."
            case .verifyFailed(let p): return "\(p) didn't copy correctly."
            }
        }
    }

    /// Every file under the model slots of `root` (including in-flight `.partial` downloads).
    /// Anything else in the folder — and the `.compat` link farm — is left alone.
    static func items(in root: URL) -> [Item] {
        let fm = FileManager.default
        var out: [Item] = []
        for model in ModelID.allCases {
            let slot = root.appendingPathComponent(model.slug, isDirectory: true)
            guard let e = fm.enumerator(at: slot, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else { continue }
            // The enumerator may hand back symlink-resolved paths (/var → /private/var), so
            // compare against the resolved slot path rather than the one we built.
            let slotPath = slot.resolvingSymlinksInPath().path
            for case let url as URL in e {
                let v = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
                guard v?.isRegularFile == true else { continue }
                let inside = String(url.resolvingSymlinksInPath().path.dropFirst(slotPath.count + 1))
                out.append(Item(relativePath: model.slug + "/" + inside, bytes: Int64(v?.fileSize ?? 0)))
            }
        }
        return out.sorted { $0.relativePath < $1.relativePath }
    }

    static func sameVolume(_ a: URL, _ b: URL) -> Bool {
        func id(_ u: URL) -> NSObject? {
            // `b` may not exist yet: walk up to the nearest existing ancestor.
            var url = u
            while !FileManager.default.fileExists(atPath: url.path), url.path != "/" { url = url.deletingLastPathComponent() }
            return (try? url.resourceValues(forKeys: [.volumeIdentifierKey]))?.volumeIdentifier as? NSObject
        }
        guard let x = id(a), let y = id(b) else { return false }
        return x.isEqual(y)
    }

    static func freeBytes(at url: URL) -> Int64 {
        var u = url
        while !FileManager.default.fileExists(atPath: u.path), u.path != "/" { u = u.deletingLastPathComponent() }
        let v = try? u.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return v?.volumeAvailableCapacityForImportantUsage ?? 0
    }

    /// Copies `items` from `src` to `dst`. Files already present at the destination with the
    /// right size are skipped (so an interrupted copy resumes). `progress` gets total bytes done.
    /// Throws `.cancelled` after cleaning up the file in flight; the source is never touched.
    static func copy(_ items: [Item], from src: URL, to dst: URL,
                     progress: (Int64, String) -> Void, isCancelled: () -> Bool) throws {
        let fm = FileManager.default
        let clone = sameVolume(src, dst)         // APFS clone: instant, no extra space until sources go
        var done: Int64 = 0
        for item in items {
            if isCancelled() { throw Failure.cancelled }
            let from = src.appendingPathComponent(item.relativePath)
            let to = dst.appendingPathComponent(item.relativePath)
            try fm.createDirectory(at: to.deletingLastPathComponent(), withIntermediateDirectories: true)
            if size(to) == item.bytes { done += item.bytes; progress(done, item.relativePath); continue }
            try? fm.removeItem(at: to)
            if clone {
                do { try fm.copyItem(at: from, to: to) } catch { throw Failure.copyFailed(item.relativePath) }
                done += item.bytes; progress(done, item.relativePath)
            } else {
                let base = done
                try chunkedCopy(from: from, to: to, name: item.relativePath, isCancelled: isCancelled) { n in
                    progress(base + n, item.relativePath)
                }
                done += item.bytes
            }
            guard size(to) == item.bytes else { try? fm.removeItem(at: to); throw Failure.verifyFailed(item.relativePath) }
        }
    }

    private static func chunkedCopy(from: URL, to: URL, name: String, isCancelled: () -> Bool,
                                    progress: (Int64) -> Void) throws {
        let fm = FileManager.default
        let tmp = to.appendingPathExtension("copying")
        try? fm.removeItem(at: tmp)
        guard fm.createFile(atPath: tmp.path, contents: nil),
              let reader = try? FileHandle(forReadingFrom: from),
              let writer = try? FileHandle(forWritingTo: tmp) else { throw Failure.copyFailed(name) }
        defer { try? reader.close(); try? writer.close() }
        var copied: Int64 = 0
        do {
            while true {
                if isCancelled() { throw Failure.cancelled }
                guard let chunk = try reader.read(upToCount: 8 << 20), !chunk.isEmpty else { break }
                try writer.write(contentsOf: chunk)
                copied += Int64(chunk.count)
                progress(copied)
            }
            try writer.close()
            try fm.moveItem(at: tmp, to: to)
        } catch let f as Failure {
            try? fm.removeItem(at: tmp); throw f
        } catch {
            try? fm.removeItem(at: tmp); throw Failure.copyFailed(name)
        }
    }

    /// Removes the old copy: the copied files, now-empty slot folders and the `.compat` links.
    static func removeSources(_ items: [Item], from root: URL) {
        let fm = FileManager.default
        for item in items { try? fm.removeItem(at: root.appendingPathComponent(item.relativePath)) }
        for model in ModelID.allCases {
            let slot = root.appendingPathComponent(model.slug, isDirectory: true)
            removeEmptyDirs(slot)
        }
        try? fm.removeItem(at: root.appendingPathComponent(".compat"))
    }

    private static func removeEmptyDirs(_ dir: URL) {
        let fm = FileManager.default
        guard let kids = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.isDirectoryKey]) else { return }
        for k in kids where (try? k.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true { removeEmptyDirs(k) }
        if (try? fm.contentsOfDirectory(atPath: dir.path))?.isEmpty == true { try? fm.removeItem(at: dir) }
    }

    private static func size(_ url: URL) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? nil
    }

    /// Folder to use inside a user-picked folder: the folder itself if it already holds model
    /// slots (or is already named "Modelr Models"), else a "Modelr Models" subfolder.
    static func resolveRoot(picked: URL) -> URL {
        if picked.lastPathComponent == "Modelr Models" || !items(in: picked).isEmpty { return picked }
        return picked.appendingPathComponent("Modelr Models", isDirectory: true)
    }
}
