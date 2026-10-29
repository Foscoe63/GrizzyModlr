import Foundation
import Observation
import AppKit

/// The file + persistence service: owns projects, their on-disk folders, image
/// processing, and the stage/commit lifecycle of generation files. All LIVE job
/// state (running/failed/progress, tokens, downloads) lives in AppState and is
/// driven by AppRuntime through the reducer — this store performs no engine or
/// network work.
@MainActor
@Observable
final class ProjectStore {
    private(set) var projects: [Project] = []
    var selection: Project.ID?

    /// Bumped whenever input.png is rewritten, so the preview reloads from disk.
    private var inputVersions: [Project.ID: Int] = [:]
    /// Transient import/processing error per project (image pipeline only).
    private(set) var importErrors: [Project.ID: String] = [:]

    func clearImportError(_ id: Project.ID) { importErrors[id] = nil }

    private let rootDir: URL

    /// `rootDir` defaults to Application Support; tests inject a temp directory for
    /// isolation (nothing else varies by root).
    init(rootDir: URL? = nil) {
        self.rootDir = rootDir ?? FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Modelr", isDirectory: true)
        try? FileManager.default.createDirectory(at: projectsDir, withIntermediateDirectories: true)
        load()
    }

    // MARK: - paths

    private var indexFile: URL { rootDir.appendingPathComponent("projects.json") }
    private var projectsDir: URL { rootDir.appendingPathComponent("projects", isDirectory: true) }
    private func folder(for id: Project.ID) -> URL {
        projectsDir.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    func imageURL(for project: Project) -> URL? {
        guard let name = project.inputImageName else { return nil }
        let url = folder(for: project.id).appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    func inputVersion(_ id: Project.ID) -> Int { inputVersions[id] ?? 0 }

    private func clearInputFiles(in folderURL: URL) {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(atPath: folderURL.path) else { return }
        for item in items where item.hasPrefix("input.") || item.hasPrefix("source.") || item.hasPrefix("mask.") {
            try? fm.removeItem(at: folderURL.appendingPathComponent(item))
        }
    }

    /// Delete streamed preview meshes for a project (runtime calls this when a
    /// shape run ends, cancels, or fails).
    func clearStreamFiles(for id: Project.ID) {
        let fm = FileManager.default
        let dir = folder(for: id)
        guard let items = try? fm.contentsOfDirectory(atPath: dir.path) else { return }
        for item in items where (item.hasPrefix("preview_") && item.hasSuffix(".mesh"))
            || (item.hasPrefix("points_") && item.hasSuffix(".bin")) {
            try? fm.removeItem(at: dir.appendingPathComponent(item))
        }
    }

    /// Delete streamed paint view grids for a project.
    func clearPaintStreamFiles(for id: Project.ID) {
        let fm = FileManager.default
        let dir = folder(for: id)
        guard let items = try? fm.contentsOfDirectory(atPath: dir.path) else { return }
        for item in items where item.hasPrefix("paint_views_") && item.hasSuffix(".png") {
            try? fm.removeItem(at: dir.appendingPathComponent(item))
        }
    }

    func meshURL(for project: Project) -> URL? {
        if let gen = project.currentGeneration {
            let url = folder(for: project.id).appendingPathComponent(gen.meshFileName)
            return FileManager.default.fileExists(atPath: url.path) ? url : nil
        }
        // legacy fallback (pre-migration)
        guard let name = project.outputMeshName else { return nil }
        let url = folder(for: project.id).appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// The input-image snapshot for a saved generation (for history thumbnails).
    func inputURL(for generation: Generation, in id: Project.ID) -> URL? {
        guard !generation.inputFileName.isEmpty else { return nil }
        let url = folder(for: id).appendingPathComponent(generation.inputFileName)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// The untextured shape geometry for the current selection — for a paint
    /// version, the shape it was textured from. Always the "model".
    func shapeMeshURL(for project: Project) -> URL? {
        guard let gen = project.currentGeneration else { return nil }
        let shapeName: String
        if gen.kind == .shape {
            shapeName = gen.meshFileName
        } else if let sid = gen.sourceShapeID,
                  let s = project.generations.first(where: { $0.id == sid }) {
            shapeName = s.meshFileName
        } else {
            return nil
        }
        let url = folder(for: project.id).appendingPathComponent(shapeName)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Viewer content for the selected version: textured for a paint version, the
    /// plain shape mesh otherwise.
    func currentViewerContent(for project: Project) -> ViewerContent? {
        guard let gen = project.currentGeneration else { return nil }
        let dir = folder(for: project.id)
        let mesh = dir.appendingPathComponent(gen.meshFileName)
        guard FileManager.default.fileExists(atPath: mesh.path) else { return nil }
        if gen.kind == .paint, let texName = gen.paintedTextureFileName {
            let tex = dir.appendingPathComponent(texName)
            if FileManager.default.fileExists(atPath: tex.path) {
                // PBR version: hand the viewer the albedo + metallic-roughness pair
                // for physically-based lighting; Color versions stay flat-textured.
                if let mrName = gen.paintedMRFileName {
                    let mr = dir.appendingPathComponent(mrName)
                    if FileManager.default.fileExists(atPath: mr.path) {
                        return .pbrMesh(mesh, albedo: tex, metallicRoughness: mr)
                    }
                }
                return .texturedMesh(mesh, tex)
            }
        }
        return .mesh(mesh)
    }

    func selectGeneration(_ genID: UUID, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].selectedGenerationID = genID
        save()
    }

    func deleteGeneration(_ genID: UUID, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }),
              projects[i].generations.contains(where: { $0.id == genID }) else { return }
        let dir = folder(for: id)
        // Cascade: deleting a shape also removes the paint versions that textured it,
        // so no paint version is left pointing at a missing source.
        let toRemove = projects[i].generations.filter { $0.id == genID || $0.sourceShapeID == genID }
        let removedIDs = Set(toRemove.map { $0.id })
        func fileNames(_ gen: Generation) -> [String] {
            [gen.meshFileName, gen.inputFileName, gen.sourceFileName, gen.maskFileName,
             gen.paintedMeshFileName, gen.paintedTextureFileName, gen.paintedMRFileName]
                .compactMap { $0 }.filter { !$0.isEmpty }
        }
        // Paint versions share their input/source/mask snapshots with the shape they
        // textured — never delete a file a surviving generation still references.
        let stillReferenced = Set(projects[i].generations
            .filter { !removedIDs.contains($0.id) }
            .flatMap(fileNames))
        for gen in toRemove {
            for name in fileNames(gen) where !stillReferenced.contains(name) {
                try? FileManager.default.removeItem(at: dir.appendingPathComponent(name))
            }
        }
        let removed = removedIDs
        projects[i].generations.removeAll { removed.contains($0.id) }
        if let sel = projects[i].selectedGenerationID, removed.contains(sel) {
            projects[i].selectedGenerationID = projects[i].generations.last?.id
        }
        save()
    }

    /// Load a saved version's image (source + mask + cutout) back into the project's
    /// working files, so it can be edited and regenerated into a new version.
    func restoreGeneration(_ genID: UUID, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }),
              let gen = projects[i].generations.first(where: { $0.id == genID }) else { return }
        let dir = folder(for: id)
        let fm = FileManager.default
        clearInputFiles(in: dir)   // clears working input/source/mask, not the gen_* snapshots

        if let srcName = gen.sourceFileName, fm.fileExists(atPath: dir.appendingPathComponent(srcName).path) {
            let ext = (srcName as NSString).pathExtension.isEmpty ? "png" : (srcName as NSString).pathExtension
            let dest = dir.appendingPathComponent("source.\(ext)")
            try? fm.copyItem(at: dir.appendingPathComponent(srcName), to: dest)
            projects[i].sourceImageName = dest.lastPathComponent
        } else {
            projects[i].sourceImageName = nil   // old version without a source snapshot
        }
        if let maskName = gen.maskFileName, fm.fileExists(atPath: dir.appendingPathComponent(maskName).path) {
            try? fm.copyItem(at: dir.appendingPathComponent(maskName), to: dir.appendingPathComponent("mask.png"))
            projects[i].removeBackground = true
        } else {
            projects[i].removeBackground = false
        }
        let inputSnap = dir.appendingPathComponent(gen.inputFileName)
        if fm.fileExists(atPath: inputSnap.path) {
            try? fm.copyItem(at: inputSnap, to: dir.appendingPathComponent("input.png"))
            projects[i].inputImageName = "input.png"
        } else {
            projects[i].inputImageName = nil   // snapshot missing → no working input
        }
        inputVersions[id, default: 0] += 1
        save()
    }

    func project(_ id: Project.ID) -> Project? { projects.first { $0.id == id } }

    // MARK: - CRUD

    @discardableResult
    func newProject() -> Project {
        let project = Project(name: nextUntitledName())
        try? FileManager.default.createDirectory(at: folder(for: project.id), withIntermediateDirectories: true)
        projects.insert(project, at: 0)
        selection = project.id
        save()
        return project
    }

    /// Remove the project and its folder. The runtime cancels any live jobs
    /// before calling this.
    func delete(_ id: Project.ID) {
        try? FileManager.default.removeItem(at: folder(for: id))
        projects.removeAll { $0.id == id }
        importErrors[id] = nil
        if selection == id { selection = projects.first?.id }
        save()
    }

    func rename(_ id: Project.ID, to name: String) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        projects[i].name = trimmed
        save()
    }

    func setShapeModel(_ model: ShapeModel, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].shapeModel = model
        save()
    }

    func setPaintModel(_ model: PaintModel, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].paintModel = model
        save()
    }

    func setQuantization(_ quant: Quantization, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].quantization = quant
        save()
    }

    func setAdvancedMode(_ on: Bool, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        // Entering advanced: seed the fields from the current normal preset so the
        // configuration carries over instead of jumping to unrelated defaults.
        if on && !projects[i].advancedMode {
            let s = projects[i].resolvedSettings   // normal preset (advancedMode still false)
            projects[i].shapeModel = s.model
            projects[i].quantization = s.quant
            projects[i].steps = s.steps
            projects[i].guidance = s.guidance
            projects[i].octree = s.octree
        }
        projects[i].advancedMode = on
        save()
    }
    func setQuality(_ q: QualityPreset, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].quality = q
        save()
    }
    func setSteps(_ n: Int, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].steps = n
        save()
    }
    func setGuidance(_ g: Double, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].guidance = g
        save()
    }
    func setOctree(_ o: Int, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].octree = o
        save()
    }
    func setSeed(_ s: UInt64?, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].seed = s
        save()
    }

    func setPaintAdvanced(_ on: Bool, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        if on && !projects[i].paintAdvanced {
            let s = projects[i].resolvedPaintSettings   // seed from the normal preset
            projects[i].paintRes = s.res; projects[i].paintSteps = s.steps
            projects[i].paintTex = s.tex; projects[i].paintSuperres = s.superres
            projects[i].paintFaces = s.faces
        }
        projects[i].paintAdvanced = on
        save()
    }
    func setPaintQuality(_ q: PaintQuality, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].paintQuality = q; save()
    }
    func setPaintSteps(_ n: Int, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].paintSteps = n; save()
    }
    func setPaintRes(_ n: Int, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].paintRes = n; save()
    }
    func setPaintTex(_ n: Int, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].paintTex = n; save()
    }
    func setPaintSuperres(_ on: Bool, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].paintSuperres = on; save()
    }
    func setPaintFaces(_ n: Int, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].paintFaces = n; save()
    }
    func setPaintSeed(_ s: UInt64?, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].paintSeed = s; save()
    }

    // MARK: - image input

    func setImage(fromURL src: URL, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        let ext = src.pathExtension.isEmpty ? "png" : src.pathExtension
        let dest = folder(for: id).appendingPathComponent("source.\(ext)")
        do {
            try FileManager.default.createDirectory(at: folder(for: id), withIntermediateDirectories: true)
            clearInputFiles(in: folder(for: id))
            try FileManager.default.copyItem(at: src, to: dest)
            projects[i].sourceImageName = dest.lastPathComponent
            applyImage(id)
        } catch {
            importErrors[id] = "Couldn't import image: \(error.localizedDescription)"
        }
    }

    func setImage(_ image: NSImage, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        guard let png = image.pngData() else {
            importErrors[id] = "Unsupported image."; return
        }
        let dest = folder(for: id).appendingPathComponent("source.png")
        do {
            try FileManager.default.createDirectory(at: folder(for: id), withIntermediateDirectories: true)
            clearInputFiles(in: folder(for: id))
            try png.write(to: dest)
            projects[i].sourceImageName = dest.lastPathComponent
            applyImage(id)
        } catch {
            importErrors[id] = "Couldn't import image: \(error.localizedDescription)"
        }
    }

    func setRemoveBackground(_ on: Bool, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        projects[i].removeBackground = on
        save()
        if projects[i].sourceImageName != nil { applyImage(id) }   // re-process
    }

    /// Produce the model input from the source image (optionally background-removed
    /// via an editable mask).
    private func applyImage(_ id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }),
              let srcName = projects[i].sourceImageName else { return }
        let dir = folder(for: id)
        let source = dir.appendingPathComponent(srcName)
        let input = dir.appendingPathComponent("input.png")
        let maskFile = dir.appendingPathComponent("mask.png")
        try? FileManager.default.removeItem(at: input)
        importErrors[id] = nil

        var wrote = false
        if projects[i].removeBackground, let original = BackgroundRemover.loadCGImage(source) {
            // an edited mask if present, else compute (and cache) the Vision mask
            var mask = BackgroundRemover.loadCGImage(maskFile)
            if mask == nil {
                mask = BackgroundRemover.visionMask(for: original)
                if let m = mask, let data = BackgroundRemover.pngData(m) { try? data.write(to: maskFile) }
            }
            if let mask, let cutout = BackgroundRemover.cutoutPNG(original: original, mask: mask) {
                wrote = (try? cutout.write(to: input)) != nil
            }
        }
        if !wrote {
            try? FileManager.default.removeItem(at: maskFile)
            if let cg = BackgroundRemover.loadCGImage(source), let png = BackgroundRemover.pngData(cg) {
                wrote = (try? png.write(to: input)) != nil   // upright (bakes EXIF orientation)
            } else if let img = NSImage(contentsOf: source), let png = img.pngData() {
                wrote = (try? png.write(to: input)) != nil
            }
        }
        guard wrote else {
            projects[i].inputImageName = nil   // input.png was removed above; don't leave a dangling ref
            inputVersions[id, default: 0] += 1
            save()
            importErrors[id] = "Couldn't process the image."
            return
        }
        projects[i].inputImageName = "input.png"
        projects[i].outputMeshName = nil
        inputVersions[id, default: 0] += 1   // force the left preview to reload from disk
        save()
        // Generation is started explicitly by the user (Start button).
    }

    /// Save a hand-edited mask and recomposite.
    func applyEditedMask(_ mask: CGImage, for id: Project.ID) {
        guard let i = projects.firstIndex(where: { $0.id == id }) else { return }
        if let data = BackgroundRemover.pngData(mask) {
            try? data.write(to: folder(for: id).appendingPathComponent("mask.png"))
        }
        projects[i].removeBackground = true
        save()
        applyImage(id)
    }

    /// The original image + current mask for the touch-up editor.
    func maskEditingInputs(for id: Project.ID) -> (original: CGImage, mask: CGImage)? {
        guard let project = projects.first(where: { $0.id == id }),
              let srcName = project.sourceImageName,
              let original = BackgroundRemover.loadCGImage(folder(for: id).appendingPathComponent(srcName))
        else { return nil }
        let maskFile = folder(for: id).appendingPathComponent("mask.png")
        guard let mask = BackgroundRemover.loadCGImage(maskFile) ?? BackgroundRemover.visionMask(for: original)
        else { return nil }
        return (original, mask)
    }

    // MARK: - shape run staging (files for one run; committed or discarded by the runtime)

    struct StagedShapeRun {
        let project: Project.ID
        let genID: UUID
        let mesh: URL
        let input: URL
        let source: URL?
        let mask: URL?
        let settings: RunSettings
        /// The concrete seed this run uses (resolved from settings.seed or random).
        let seed: UInt64
        let removeBackground: Bool
        let startedAt: Date
    }

    enum StagingError: LocalizedError {
        case projectMissing
        case noInputImage
        case noShapeMesh
        case snapshotFailed(String)

        var errorDescription: String? {
            switch self {
            case .projectMissing: return "The project no longer exists."
            case .noInputImage: return "Add an image first."
            case .noShapeMesh: return "Generate a shape before painting."
            case .snapshotFailed(let what): return "Couldn't snapshot \(what) for this run."
            }
        }
    }

    /// Snapshot the exact input used (image + source + mask) into per-run files,
    /// so a mid-run input change can't corrupt this version (§4.4 Preparing).
    func stageShapeRun(for id: Project.ID) throws -> StagedShapeRun {
        guard let project = projects.first(where: { $0.id == id }) else {
            throw StagingError.projectMissing
        }
        guard let image = imageURL(for: project) else { throw StagingError.noInputImage }
        let settings = project.resolvedSettings

        let genID = UUID()
        let dir = folder(for: id)
        let output = dir.appendingPathComponent("gen_\(genID.uuidString).mesh")
        let genInput = dir.appendingPathComponent("gen_\(genID.uuidString)_input.png")
        try? FileManager.default.removeItem(at: output)
        try? FileManager.default.removeItem(at: genInput)
        do {
            try FileManager.default.copyItem(at: image, to: genInput)
        } catch {
            throw StagingError.snapshotFailed("the input image")
        }

        // Snapshot the source + mask too, so this version's image can be re-edited.
        var genSource: URL?
        if let srcName = project.sourceImageName {
            let ext = (srcName as NSString).pathExtension.isEmpty ? "png" : (srcName as NSString).pathExtension
            let snap = dir.appendingPathComponent("gen_\(genID.uuidString)_source.\(ext)")
            if (try? FileManager.default.copyItem(at: dir.appendingPathComponent(srcName), to: snap)) != nil {
                genSource = snap
            }
        }
        var genMask: URL?
        let maskFile = dir.appendingPathComponent("mask.png")
        if FileManager.default.fileExists(atPath: maskFile.path) {
            let snap = dir.appendingPathComponent("gen_\(genID.uuidString)_mask.png")
            if (try? FileManager.default.copyItem(at: maskFile, to: snap)) != nil { genMask = snap }
        }

        clearStreamFiles(for: id)
        return StagedShapeRun(project: id, genID: genID, mesh: output, input: genInput,
                              source: genSource, mask: genMask, settings: settings,
                              seed: settings.seed ?? UInt64.random(in: 0..<UInt64(UInt32.max)),
                              removeBackground: project.removeBackground, startedAt: Date())
    }

    /// Turn a finished run into a saved generation, selected for viewing.
    /// Returns an error message (and cleans up) if the output is unusable.
    func commitShapeRun(_ staged: StagedShapeRun) -> String? {
        guard let i = projects.firstIndex(where: { $0.id == staged.project }) else {
            discardShapeRun(staged)
            return "The project no longer exists."
        }
        // A 0-byte (or sub-header) file means a crash mid-write — discard it.
        let size = (try? staged.mesh.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        guard FileManager.default.fileExists(atPath: staged.mesh.path), size > 8 else {
            discardShapeRun(staged)
            return "The model didn't produce a valid mesh."
        }
        let s = staged.settings
        let gen = Generation(id: staged.genID, createdAt: staged.startedAt,
                             modelRaw: s.model.rawValue, quantRaw: s.quant.rawValue,
                             steps: s.steps, removeBackground: staged.removeBackground,
                             meshFileName: staged.mesh.lastPathComponent,
                             inputFileName: staged.input.lastPathComponent,
                             durationSeconds: Date().timeIntervalSince(staged.startedAt),
                             guidanceRaw: s.guidance, octreeRaw: s.octree,
                             seedRaw: staged.seed,
                             sourceFileName: staged.source?.lastPathComponent,
                             maskFileName: staged.mask?.lastPathComponent)
        projects[i].generations.append(gen)
        projects[i].selectedGenerationID = gen.id
        projects[i].outputMeshName = nil
        save()
        return nil
    }

    /// Applies a mesh edit to the selected shape (for a paint version, the shape it
    /// textured) and saves the result as a new, selected shape version. The heavy work
    /// runs off the main actor. Returns a user-facing message: nil on success.
    func applyMeshEdit(_ edit: MeshEdit, for id: Project.ID) async -> String? {
        guard let project = project(id), let cur = project.currentGeneration else {
            return "There's no shape to edit."
        }
        let baseID = cur.kind == .shape ? cur.id : cur.sourceShapeID
        guard let base = project.generations.first(where: { $0.id == baseID }) else {
            return "The source shape for this version is missing."
        }
        let dir = folder(for: id)
        let src = dir.appendingPathComponent(base.meshFileName)
        guard FileManager.default.fileExists(atPath: src.path) else { return "The shape mesh is missing." }
        let newID = UUID()
        let dst = dir.appendingPathComponent("gen_\(newID.uuidString).mesh")
        let result = await Task.detached(priority: .userInitiated) {
            MeshOps.applyToFile(edit, from: src, to: dst)
        }.value
        guard let result else { return "Couldn't apply the edit to this mesh." }
        guard result.changed else { return "Nothing to change: \(result.note)." }
        guard let i = projects.firstIndex(where: { $0.id == id }) else {
            try? FileManager.default.removeItem(at: dst)
            return "The project no longer exists."
        }
        appendEditedShape(from: base, id: newID, mesh: dst, summary: "\(edit.label) (\(result.note))", at: i)
        return nil
    }

    /// Saves `mesh` as a new, selected shape version derived from `base`.
    private func appendEditedShape(from base: Generation, id newID: UUID, mesh: URL,
                                   summary: String, at i: Int) {
        let gen = Generation(id: newID, createdAt: Date(), modelRaw: base.modelRaw, quantRaw: base.quantRaw,
                             steps: base.steps, removeBackground: base.removeBackground,
                             meshFileName: mesh.lastPathComponent, inputFileName: base.inputFileName,
                             durationSeconds: nil, guidanceRaw: base.guidanceRaw, octreeRaw: base.octreeRaw,
                             seedRaw: base.seedRaw, sourceFileName: base.sourceFileName,
                             maskFileName: base.maskFileName,
                             editSummary: summary, editedFromID: base.id)
        projects[i].generations.append(gen)
        projects[i].selectedGenerationID = newID
        save()
    }

    /// Loads the selected shape (for a paint version, the shape it textured) for
    /// sculpting. nil if it can't be read.
    func beginSculpt(for id: Project.ID) async -> SculptController? {
        guard let project = project(id), let url = shapeMeshURL(for: project) else { return nil }
        let mesh = await Task.detached(priority: .userInitiated) { () -> SculptMesh? in
            guard let raw = try? Data(contentsOf: url), let m = MeshDecimator.parse(raw),
                  !m.faces.isEmpty else { return nil }
            return SculptMesh(positions: m.positions, faces: m.faces)
        }.value
        guard let mesh else { return nil }
        return SculptController(mesh: mesh)
    }

    /// Saves a finished sculpt as a new shape version. Returns an error message, or nil.
    func commitSculpt(_ sculpt: SculptController, for id: Project.ID) async -> String? {
        guard let project = project(id), let cur = project.currentGeneration else {
            return "There's no shape to edit."
        }
        let baseID = cur.kind == .shape ? cur.id : cur.sourceShapeID
        guard let base = project.generations.first(where: { $0.id == baseID }) else {
            return "The source shape for this version is missing."
        }
        let newID = UUID()
        let dst = folder(for: id).appendingPathComponent("gen_\(newID.uuidString).mesh")
        let m = sculpt.mesh
        let ok = await Task.detached(priority: .userInitiated) { () -> Bool in
            let data = MeshDecimator.serialize(positions: m.positions, faces: m.faces)
            return (try? data.write(to: dst, options: .atomic)) != nil
        }.value
        guard ok else { return "Couldn't save the sculpted mesh." }
        guard let i = projects.firstIndex(where: { $0.id == id }) else {
            try? FileManager.default.removeItem(at: dst)
            return "The project no longer exists."
        }
        appendEditedShape(from: base, id: newID, mesh: dst, summary: "sculpted", at: i)
        return nil
    }

    /// Applies global material adjustments to the selected paint version and saves them
    /// as a new paint version. The mesh is shared with the source version; only the
    /// albedo (and, for PBR, the metallic-roughness map) are rewritten.
    func applyMaterialEdit(_ adj: MaterialAdjustments, for id: Project.ID) async -> String? {
        guard let project = project(id), let base = project.currentGeneration,
              base.kind == .paint, let albedoName = base.paintedTextureFileName else {
            return "Select a painted version to edit its material."
        }
        if adj.isIdentity { return "Nothing to change." }
        let dir = folder(for: id)
        let newID = UUID()
        let albedoSrc = dir.appendingPathComponent(albedoName)
        let albedoDst = dir.appendingPathComponent("painted_\(newID.uuidString)_albedo.png")
        let mrSrc = base.paintedMRFileName.map { dir.appendingPathComponent($0) }
        let mrDst = mrSrc == nil ? nil : dir.appendingPathComponent("painted_\(newID.uuidString)_mr.png")
        let ok = await Task.detached(priority: .userInitiated) { () -> Bool in
            if adj.altersAlbedo || mrSrc == nil || !adj.altersMR {
                guard MaterialOps.processFile(albedoSrc, to: albedoDst, isMR: false, adj) else { return false }
            } else {
                guard (try? FileManager.default.copyItem(at: albedoSrc, to: albedoDst)) != nil else { return false }
            }
            if let mrSrc, let mrDst {
                guard MaterialOps.processFile(mrSrc, to: mrDst, isMR: true, adj) else { return false }
            }
            return true
        }.value
        guard ok else {
            try? FileManager.default.removeItem(at: albedoDst)
            if let mrDst { try? FileManager.default.removeItem(at: mrDst) }
            return "Couldn't read this version's textures."
        }
        guard let i = projects.firstIndex(where: { $0.id == id }) else {
            try? FileManager.default.removeItem(at: albedoDst)
            if let mrDst { try? FileManager.default.removeItem(at: mrDst) }
            return "The project no longer exists."
        }
        let gen = Generation(id: newID, createdAt: Date(), modelRaw: base.modelRaw, quantRaw: base.quantRaw,
                         steps: base.steps, removeBackground: base.removeBackground,
                         meshFileName: base.meshFileName, inputFileName: base.inputFileName,
                         durationSeconds: nil, guidanceRaw: base.guidanceRaw, octreeRaw: base.octreeRaw,
                         seedRaw: base.seedRaw, sourceFileName: base.sourceFileName,
                         maskFileName: base.maskFileName,
                         paintedTextureFileName: albedoDst.lastPathComponent,
                         paintedMRFileName: mrDst?.lastPathComponent,
                         kindRaw: "paint", sourceShapeID: base.sourceShapeID,
                         paintModelRaw: base.paintModelRaw, paintResRaw: base.paintResRaw,
                         paintStepsRaw: base.paintStepsRaw, paintTexRaw: base.paintTexRaw,
                         paintFacesRaw: base.paintFacesRaw, paintSuperresRaw: base.paintSuperresRaw,
                         paintSeedRaw: base.paintSeedRaw,
                         editSummary: "material: \(adj.summary)", editedFromID: base.id)
        projects[i].generations.append(gen)
        projects[i].selectedGenerationID = newID
        save()
        return nil
    }

    /// Drop a staged run's files (cancelled / failed) so they don't accumulate.
    func discardShapeRun(_ staged: StagedShapeRun) {
        for url in [staged.mesh, staged.input, staged.source, staged.mask].compactMap({ $0 }) {
            try? FileManager.default.removeItem(at: url)
        }
        clearStreamFiles(for: staged.project)
    }

    // MARK: - paint run staging

    struct StagedPaintRun {
        let project: Project.ID
        let paintID: UUID
        /// The immutable shape generation being textured.
        let source: Generation
        let shapeMesh: URL
        /// Where the §4.5 prep stage writes the QEM-decimated mesh when the shape
        /// exceeds the face budget. Exists only if decimation ran and succeeded: the
        /// engine paints `prepMesh` when present, else the original `shapeMesh`
        /// (slow-but-correct fallback). Per-run scratch — deleted on commit + discard.
        let prepMesh: URL
        let image: URL
        let outMesh: URL
        let outTexture: URL
        /// The metallic-roughness map, written by the PBR (Large) path only. Always
        /// staged; left unwritten (and required-absent at commit) for the Color path.
        let outMR: URL
        let settings: PaintSettings
        /// The concrete seed this run uses (resolved from settings.seed or random).
        let seed: UInt64
        let startedAt: Date

        /// A PBR (Large) run — produces the extra metallic-roughness map.
        var isPBR: Bool { settings.model == .large }

        /// The geometry the paint engine should consume.
        var engineMesh: URL {
            FileManager.default.fileExists(atPath: prepMesh.path) ? prepMesh : shapeMesh
        }
    }

    /// Resolve the shape to texture (the selected version, or the shape a selected
    /// paint version came from) and stage the output files.
    func stagePaintRun(for id: Project.ID) throws -> StagedPaintRun {
        guard let project = projects.first(where: { $0.id == id }) else {
            throw StagingError.projectMissing
        }
        guard let current = project.currentGeneration else { throw StagingError.noShapeMesh }
        let shape: Generation
        if current.kind == .shape {
            shape = current
        } else if let sid = current.sourceShapeID,
                  let s = project.generations.first(where: { $0.id == sid }) {
            shape = s
        } else {
            throw StagingError.noShapeMesh
        }
        let dir = folder(for: id)
        let meshURL = dir.appendingPathComponent(shape.meshFileName)
        guard FileManager.default.fileExists(atPath: meshURL.path) else {
            throw StagingError.noShapeMesh
        }
        guard let image = inputURL(for: shape, in: id) ?? imageURL(for: project) else {
            throw StagingError.noInputImage
        }

        let paintID = UUID()
        let outMesh = dir.appendingPathComponent("painted_\(paintID.uuidString).tmesh")
        let outTex = dir.appendingPathComponent("painted_\(paintID.uuidString)_texture.png")
        let outMR = dir.appendingPathComponent("painted_\(paintID.uuidString)_mr.png")
        let prepMesh = dir.appendingPathComponent("painted_\(paintID.uuidString)_prep.mesh")
        try? FileManager.default.removeItem(at: outMesh)
        try? FileManager.default.removeItem(at: outTex)
        try? FileManager.default.removeItem(at: outMR)
        try? FileManager.default.removeItem(at: prepMesh)
        clearPaintStreamFiles(for: id)

        // Prefer the shape's immutable input snapshot; if it's missing, snapshot the
        // live image so a mid-paint input change can't corrupt this run.
        var paintInput = image
        if inputURL(for: shape, in: id) == nil {
            let copy = dir.appendingPathComponent("painted_\(paintID.uuidString)_input.png")
            if (try? FileManager.default.copyItem(at: image, to: copy)) != nil { paintInput = copy }
        }
        let settings = project.resolvedPaintSettings
        return StagedPaintRun(project: id, paintID: paintID, source: shape,
                              shapeMesh: meshURL, prepMesh: prepMesh, image: paintInput,
                              outMesh: outMesh, outTexture: outTex, outMR: outMR,
                              settings: settings,
                              seed: settings.seed ?? UInt64.random(in: 0..<UInt64(UInt32.max)),
                              startedAt: Date())
    }

    /// Record a finished paint run as a saved generation. Returns an error message
    /// (and cleans up) if the outputs are unusable.
    func commitPaintRun(_ staged: StagedPaintRun) -> String? {
        guard let i = projects.firstIndex(where: { $0.id == staged.project }) else {
            discardPaintRun(staged)
            return "The project no longer exists."
        }
        // Require BOTH the mesh and the texture, each non-trivially sized (a crash
        // mid-write leaves a 0-byte file) before recording the version.
        func sized(_ url: URL) -> Bool {
            ((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) > 8
        }
        guard sized(staged.outMesh), sized(staged.outTexture) else {
            discardPaintRun(staged)
            return "Paint didn't produce a usable texture."
        }
        // A PBR run must also produce the metallic-roughness map; without it the
        // version would silently degrade to Color at load/export time.
        if staged.isPBR, !sized(staged.outMR) {
            discardPaintRun(staged)
            return "PBR paint didn't produce a metallic-roughness map."
        }
        let src = staged.source
        let s = staged.settings
        let gen = Generation(
            id: staged.paintID, createdAt: staged.startedAt,
            modelRaw: src.modelRaw, quantRaw: src.quantRaw, steps: src.steps,
            removeBackground: src.removeBackground,
            meshFileName: staged.outMesh.lastPathComponent,
            inputFileName: src.inputFileName,
            durationSeconds: Date().timeIntervalSince(staged.startedAt),
            guidanceRaw: src.guidanceRaw, octreeRaw: src.octreeRaw,
            seedRaw: src.seedRaw,
            // Carry the shape's source/mask snapshots so restoring a paint version
            // rebuilds the same editable image set as restoring its source shape.
            sourceFileName: src.sourceFileName,
            maskFileName: src.maskFileName,
            paintedTextureFileName: staged.outTexture.lastPathComponent,
            paintedMRFileName: staged.isPBR ? staged.outMR.lastPathComponent : nil,
            kindRaw: "paint", sourceShapeID: src.id,
            paintModelRaw: s.model.rawValue,
            paintResRaw: s.res, paintStepsRaw: s.steps,
            paintTexRaw: s.tex, paintFacesRaw: s.faces, paintSuperresRaw: s.superres,
            paintSeedRaw: staged.seed)
        projects[i].generations.append(gen)
        projects[i].selectedGenerationID = gen.id
        try? FileManager.default.removeItem(at: staged.prepMesh)   // per-run scratch
        save()
        return nil
    }

    func discardPaintRun(_ staged: StagedPaintRun) {
        try? FileManager.default.removeItem(at: staged.outMesh)
        try? FileManager.default.removeItem(at: staged.outTexture)
        try? FileManager.default.removeItem(at: staged.outMR)
        try? FileManager.default.removeItem(at: staged.prepMesh)
        // Remove the per-run input snapshot only if we created one (it lives in
        // the project folder with the painted_ prefix).
        if staged.image.lastPathComponent.hasPrefix("painted_") {
            try? FileManager.default.removeItem(at: staged.image)
        }
        clearPaintStreamFiles(for: staged.project)
    }

    // MARK: - persistence

    private func load() {
        guard let data = try? Data(contentsOf: indexFile) else { return }   // first launch, no file
        guard let decoded = try? JSONDecoder().decode([Project].self, from: data) else {
            // Don't silently wipe everything on a malformed file: preserve it for recovery
            // before the next save() overwrites projects.json.
            let backup = indexFile.deletingPathExtension()
                .appendingPathExtension("corrupt-\(Int(Date().timeIntervalSince1970)).json")
            try? FileManager.default.copyItem(at: indexFile, to: backup)
            return
        }
        projects = decoded.sorted { $0.createdAt > $1.createdAt }
        selection = projects.first?.id

        // Migrate a legacy single output.mesh into the generations list, so an
        // existing project's result becomes its first saved version.
        var healed = false
        for i in projects.indices {
            if projects[i].generations.isEmpty {
                let dir = folder(for: projects[i].id)
                let legacy = dir.appendingPathComponent("output.mesh")
                if FileManager.default.fileExists(atPath: legacy.path) {
                    func snapshot(_ name: String?, as base: String) -> String? {
                        guard let name, !name.isEmpty else { return nil }
                        let src = dir.appendingPathComponent(name)
                        guard FileManager.default.fileExists(atPath: src.path) else { return nil }
                        let ext = (name as NSString).pathExtension.isEmpty ? "png" : (name as NSString).pathExtension
                        let snap = dir.appendingPathComponent("\(base).\(ext)")
                        try? FileManager.default.removeItem(at: snap)
                        return (try? FileManager.default.copyItem(at: src, to: snap)) != nil ? snap.lastPathComponent : nil
                    }
                    let inputName = snapshot(projects[i].inputImageName, as: "gen_legacy_input") ?? ""
                    let sourceName = snapshot(projects[i].sourceImageName, as: "gen_legacy_source")
                    let maskName = FileManager.default.fileExists(atPath: dir.appendingPathComponent("mask.png").path)
                        ? snapshot("mask.png", as: "gen_legacy_mask") : nil
                    let gen = Generation(id: UUID(), createdAt: projects[i].createdAt,
                                         modelRaw: projects[i].shapeModel.rawValue,
                                         quantRaw: projects[i].quantization.rawValue,
                                         steps: projects[i].shapeModel.defaultSteps,
                                         removeBackground: projects[i].removeBackground,
                                         meshFileName: "output.mesh",
                                         inputFileName: inputName,
                                         durationSeconds: nil,
                                         sourceFileName: sourceName,
                                         maskFileName: maskName)
                    projects[i].generations = [gen]
                    projects[i].selectedGenerationID = gen.id
                    projects[i].outputMeshName = nil
                    healed = true
                }
            }
        }
        if healed { save() }
        sweepOrphanGenFiles()
    }

    /// Delete gen_*.mesh / gen_*_input.png files no Generation references — orphans
    /// left by a crash or a cancelled run whose worker wrote after cleanup.
    private func sweepOrphanGenFiles() {
        let fm = FileManager.default
        for project in projects {
            let dir = folder(for: project.id)
            guard let items = try? fm.contentsOfDirectory(atPath: dir.path) else { continue }
            var referenced = Set<String>()
            for gen in project.generations {
                for name in [gen.meshFileName, gen.inputFileName, gen.sourceFileName, gen.maskFileName,
                             gen.paintedMeshFileName, gen.paintedTextureFileName, gen.paintedMRFileName]
                    .compactMap({ $0 }).filter({ !$0.isEmpty }) {
                    referenced.insert(name)
                }
            }
            for item in items where (item.hasPrefix("gen_") || item.hasPrefix("painted_"))
                && !referenced.contains(item) {
                try? fm.removeItem(at: dir.appendingPathComponent(item))
            }
        }
    }

    private func save() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(projects) {
            try? data.write(to: indexFile)
        }
    }

    private func nextUntitledName() -> String {
        let base = "Untitled"
        let used = Set(projects.map(\.name))
        if !used.contains(base) { return base }
        var n = 2
        while used.contains("\(base) \(n)") { n += 1 }
        return "\(base) \(n)"
    }
}
