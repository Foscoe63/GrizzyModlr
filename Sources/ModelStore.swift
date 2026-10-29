import Foundation

/// Resolves where model weights live. The app is fully self-contained: ONLY the
/// container layout under Application Support is consulted (DESIGN.md §2.3) —
/// the old hardcoded dev-repo fallbacks are gone. Offline installs go through
/// "Import weights folder…", which copies size-validated files into the slots.
///
/// Container layout:
///   <AppSupport>/Modelr/models/{shape-small,shape-large,paint-small,paint-large}/…
///       exactly the HF repo layouts from ModelCatalog
///   <file>.partial       in-flight downloads (resumable)
///   models/.compat/…     symlink bridges to the vendored pipelines' expected paths
enum ModelStore {
    /// Container-side models root (created on demand).
    static var modelsRoot: URL {
        let base = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Modelr/models", isDirectory: true)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    static func installDir(for model: ModelID) -> URL {
        modelsRoot.appendingPathComponent(model.slug, isDirectory: true)
    }

    private static func fileSize(_ url: URL) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? nil
    }

    /// §4.3: installed-state is re-derived from disk at every boot — a fast size
    /// check of every catalog file (hashes were verified when the bytes landed).
    static func isInstalled(_ model: ModelID) -> Bool {
        let dir = installDir(for: model)
        return ModelCatalog.model(model).files.allSatisfy {
            fileSize(dir.appendingPathComponent($0.path)) == $0.bytes
        }
    }

    /// Bytes of catalog files currently present (complete files only).
    static func bytesOnDisk(_ model: ModelID) -> Int64 {
        let dir = installDir(for: model)
        return ModelCatalog.model(model).files.reduce(0) {
            $0 + (fileSize(dir.appendingPathComponent($1.path)) ?? 0)
        }
    }

    /// Everything under models/ (installed weights + partials + compat links).
    static func totalBytesOnDisk() -> Int64 {
        guard let e = FileManager.default.enumerator(
            at: modelsRoot, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
            options: [.skipsHiddenFiles]) else { return 0 }
        var total: Int64 = 0
        for case let url as URL in e {
            let v = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if v?.isRegularFile == true { total += Int64(v?.fileSize ?? 0) }
        }
        return total
    }

    // MARK: - engine path resolution

    /// The single safetensors checkpoint the in-process `ShapeGenerator` loads.
    /// nil unless the model is fully installed.
    static func shapeWeightsFile(for model: ShapeModel) -> URL? {
        guard isInstalled(model.modelID) else { return nil }
        return installDir(for: model.modelID).appendingPathComponent("model.fp16.safetensors")
    }

    /// Install directory of the Zero123++ weights (diffusers layout); nil unless fully installed.
    static func viewSynthesisDir() -> URL? {
        isInstalled(.viewSynthesis) ? installDir(for: .viewSynthesis) : nil
    }

    /// The multi-view (2mv) checkpoint; nil unless fully installed.
    static func multiViewWeightsFile() -> URL? {
        guard isInstalled(.shapeMultiView) else { return nil }
        return installDir(for: .shapeMultiView).appendingPathComponent("model.fp16.safetensors")
    }

    /// The weights root handed to the vendored `PaintPipeline`. The pipeline
    /// hardcodes repo-era subpaths (`hunyuan3d-paint-v2-0/…`, `dinov2-giant/…`)
    /// that differ from the §2.3 on-disk layout, and Packages/ must not change —
    /// so we bridge with a per-model directory of symlinks, rebuilt on demand.
    static func paintWeightsRoot(for model: PaintModel) -> URL? {
        guard isInstalled(model.modelID) else { return nil }
        let install = installDir(for: model.modelID)
        let compat = modelsRoot.appendingPathComponent(".compat/\(model.modelID.slug)", isDirectory: true)
        try? FileManager.default.createDirectory(at: compat, withIntermediateDirectories: true)

        func link(_ name: String, to target: URL) {
            let at = compat.appendingPathComponent(name)
            let existing = try? FileManager.default.destinationOfSymbolicLink(atPath: at.path)
            guard existing != target.path else { return }
            try? FileManager.default.removeItem(at: at)
            try? FileManager.default.createSymbolicLink(atPath: at.path,
                                                        withDestinationPath: target.path)
        }
        switch model {
        case .small:
            // paintRGB loads root/hunyuan3d-paint-v2-0/{unet,vae} + root/realesrgan.
            link("hunyuan3d-paint-v2-0", to: install)
            link("realesrgan", to: install.appendingPathComponent("realesrgan"))
        case .large:
            // PBR loads root/hunyuan3d-paintpbr-v2-1/unet, the reused v2-0 VAE at
            // root/hunyuan3d-paint-v2-0/vae, root/dinov2-giant, root/realesrgan.
            link("hunyuan3d-paintpbr-v2-1", to: install)
            link("hunyuan3d-paint-v2-0", to: install)
            link("dinov2-giant", to: install.appendingPathComponent("dinov2"))
            link("realesrgan", to: install.appendingPathComponent("realesrgan"))
        }
        return compat
    }

    // MARK: - legacy migration (§2.3)

    private static var legacyShapeDir: URL { modelsRoot.appendingPathComponent("shape", isDirectory: true) }
    private static var legacyPaintDir: URL { modelsRoot.appendingPathComponent("paint", isDirectory: true) }

    static var legacyLayoutExists: Bool {
        FileManager.default.fileExists(atPath: legacyShapeDir.path)
            || FileManager.default.fileExists(atPath: legacyPaintDir.path)
    }

    /// Move size-matching legacy files into the new slots, delete the rest, and
    /// remove the legacy roots. Driven by the pure LegacyMigration planner.
    static func performLegacyMigration() {
        let fm = FileManager.default
        var files: [(path: String, bytes: Int64)] = []
        for root in [legacyShapeDir, legacyPaintDir] where fm.fileExists(atPath: root.path) {
            guard let e = fm.enumerator(at: root, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else { continue }
            for case let url as URL in e {
                let v = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
                guard v?.isRegularFile == true else { continue }
                let rel = url.path.replacingOccurrences(of: modelsRoot.path + "/", with: "")
                files.append((rel, Int64(v?.fileSize ?? 0)))
            }
        }
        let plan = LegacyMigration.plan(files: files)
        for move in plan.moves {
            let src = modelsRoot.appendingPathComponent(move.from)
            let dst = installDir(for: move.model).appendingPathComponent(move.to)
            if fileSize(dst) == ModelCatalog.model(move.model).files.first(where: { $0.path == move.to })?.bytes {
                try? fm.removeItem(at: src)                // slot already populated
                continue
            }
            try? fm.createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? fm.removeItem(at: dst)
            try? fm.moveItem(at: src, to: dst)
        }
        for deletion in plan.deletions {
            try? fm.removeItem(at: modelsRoot.appendingPathComponent(deletion))
        }
        try? fm.removeItem(at: legacyShapeDir)
        try? fm.removeItem(at: legacyPaintDir)
    }
}
