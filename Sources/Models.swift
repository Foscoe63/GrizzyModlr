import Foundation

/// The shape checkpoint to generate with (DESIGN.md §2): Small = 2mini (0.6B,
/// 30-step CFG), Large = 2.0-turbo (1.1B distilled, 8-step consistency).
enum ShapeModel: String, CaseIterable, Identifiable {
    case small, large, fast

    var id: String { rawValue }

    var label: String {
        switch self {
        case .small: return "Small"
        case .large: return "Large"
        case .fast: return "Fast"
        }
    }

    var detail: String {
        switch self {
        case .small: return "2mini · 0.6B · fastest"
        case .large: return "2.0 turbo · 1.1B · best detail"
        case .fast: return "TripoSR · seconds · lower detail"
        }
    }

    var modelID: ModelID {
        switch self {
        case .small: return .shapeSmall
        case .large: return .shapeLarge
        case .fast: return .triposr
        }
    }

    /// Distilled (turbo) checkpoints run a consistency schedule in far fewer steps.
    var defaultSteps: Int {
        switch self {
        case .small: return 30
        case .large: return 8
        case .fast: return 1
        }
    }

    /// Migration-aware decoding: projects saved before the 2×2 lineup carry the
    /// old four-value raws (mini/miniTurbo → small; standard/standardTurbo → large).
    init(legacyRaw: String?) {
        switch legacyRaw {
        case "small", "mini", "miniTurbo": self = .small
        case "large", "standard", "standardTurbo": self = .large
        case "fast": self = .fast
        default: self = .small
        }
    }
}

/// The paint checkpoint (§2): Small = 2.0 RGB color texture, Large = 2.1 PBR
/// (albedo + metallic-roughness).
enum PaintModel: String, CaseIterable, Identifiable {
    case small, large

    var id: String { rawValue }

    var label: String {
        switch self {
        case .small: return "Color"
        case .large: return "PBR"
        }
    }

    var detail: String {
        switch self {
        case .small: return "RGB texture · 2048 atlas"
        case .large: return "Albedo + metal-rough · 4096 atlas"
        }
    }

    var modelID: ModelID {
        switch self {
        case .small: return .paintSmall
        case .large: return .paintLarge
        }
    }

    init(legacyRaw: String?) {
        self = legacyRaw.flatMap(PaintModel.init(rawValue:)) ?? .small
    }
}

/// Weight quantization of the DiT + DINO block linears (VAE always stays fp16).
enum Quantization: String, CaseIterable, Identifiable {
    case full, int8, int4

    var id: String { rawValue }

    var label: String {
        switch self {
        case .full: return "Full"
        case .int8: return "8-bit"
        case .int4: return "4-bit"
        }
    }

    var detail: String {
        switch self {
        case .full: return "fp16 · best quality"
        case .int8: return "near-lossless"
        case .int4: return "smallest · faster"
        }
    }

    /// Engine value for quantize (0 = off).
    var flag: Int {
        switch self {
        case .full: return 0
        case .int8: return 8
        case .int4: return 4
        }
    }

    /// Smaller/faster → best-quality ordering for the slider.
    static let ordered: [Quantization] = [.int4, .int8, .full]
}

/// Normal-mode effort ladder: one slider bundling steps + octree for the chosen
/// model (the model itself is a separate Small/Large picker per DESIGN.md §5).
/// Quantization is fixed at 8-bit in normal mode.
enum QualityPreset: String, CaseIterable, Identifiable {
    case fastest, fast, balanced, high, max

    var id: String { rawValue }

    var label: String {
        switch self {
        case .fastest:  return "Fastest"
        case .fast:     return "Fast"
        case .balanced: return "Balanced"
        case .high:     return "High"
        case .max:      return "Max"
        }
    }

    /// Denoise steps scale with the model's schedule (CFG vs consistency).
    func steps(for model: ShapeModel) -> Int {
        let index = Self.ordered.firstIndex(of: self) ?? 2
        switch model {
        case .small: return [12, 20, 30, 40, 50][index]
        case .large: return [4, 6, 8, 10, 12][index]
        case .fast: return [1, 1, 1, 1, 1][index]
        }
    }

    var octree: Int {
        switch self {
        case .fastest:  return 192
        case .fast:     return 256
        case .balanced: return 256
        case .high:     return 320
        case .max:      return 384
        }
    }

    func detail(for model: ShapeModel) -> String {
        "\(steps(for: model)) steps · grid \(octree)"
    }

    static let ordered: [QualityPreset] = [.fastest, .fast, .balanced, .high, .max]
    static let octreeStops = [128, 192, 256, 320, 384, 448, 512]
}

/// The fully-resolved parameters for one run (from normal preset or advanced
/// fields). `seed` nil = pick a random seed at staging time (it's recorded on
/// the Generation either way — reproducibility is part of determinism, §3).
struct RunSettings {
    let model: ShapeModel
    let quant: Quantization
    let steps: Int
    let guidance: Double
    let octree: Int
    let seed: UInt64?
}

/// Normal-mode paint quality ladder (one slider): bundles render res, diffusion
/// steps, texture size, and super-resolution.
enum PaintQuality: String, CaseIterable, Identifiable {
    case fast, balanced, high, max

    var id: String { rawValue }

    var label: String {
        switch self {
        case .fast:     return "Fast"
        case .balanced: return "Balanced"
        case .high:     return "High"
        case .max:      return "Max"
        }
    }
    var res: Int {
        switch self {
        case .fast: return 384
        default:    return 512
        }
    }
    var steps: Int {
        switch self {
        case .fast: return 8
        case .balanced: return 10
        case .high: return 15
        case .max: return 20
        }
    }
    var tex: Int {
        switch self {
        case .fast: return 1024
        case .balanced, .high: return 2048
        case .max: return 4096
        }
    }
    var superres: Bool {
        switch self {
        case .fast, .balanced: return false
        case .high, .max: return true
        }
    }
    /// Mesh-detail budget: the painted mesh is decimated to this many faces before
    /// UV-unwrap. The GPU rasterizer is cheap; xatlas (CPU) is the cost that scales.
    var faces: Int {
        switch self {
        case .fast:     return 40_000
        case .balanced: return 120_000
        case .high:     return 250_000
        case .max:      return 400_000
        }
    }
    var detail: String { "\(res)px · \(steps) steps · \(faces / 1000)k tris\(superres ? " · 4× SR" : "")" }

    static let ordered: [PaintQuality] = [.fast, .balanced, .high, .max]
    static let texStops = [1024, 2048, 4096]
    static let faceStops = [40_000, 80_000, 120_000, 200_000, 250_000, 400_000]
}

struct PaintSettings {
    let model: PaintModel
    let res: Int
    let steps: Int
    let tex: Int
    let superres: Bool
    let faces: Int
    /// nil = a fresh random seed per run (the resolved value is recorded on the
    /// paint Generation). Passed through to the pipeline's seeded initial noise.
    let seed: UInt64?
}

/// A reference view beyond the main (front) image, used by the multi-view model.
enum ReferenceView: String, Codable, CaseIterable, Identifiable {
    case left, back, right
    var id: String { rawValue }
    var label: String { rawValue.capitalized }
}

/// Whether a saved version is an untextured shape or a textured paint result.
enum GenerationKind: String, Codable { case shape, paint }

/// A single image → 3D shape project. Persisted as JSON; its files live in a
/// per-project folder under Application Support.
struct Project: Identifiable, Codable, Hashable {
    let id: UUID
    var name: String
    var createdAt: Date
    var sourceImageName: String?   // the original imported image
    var inputImageName: String?    // processed image fed to the model (e.g. bg-removed)
    var outputMeshName: String?    // legacy single output (migrated into generations)
    var modelRaw: String?
    var quantRaw: String?
    var removeBackgroundRaw: Bool?
    var generationsRaw: [Generation]?
    var selectedGenerationID: UUID?
    var advancedModeRaw: Bool?
    var qualityRaw: String?
    var stepsRaw: Int?
    var guidanceRaw: Double?
    var octreeRaw: Int?
    var seedRaw: UInt64?
    var paintModelRaw: String?
    var paintAdvancedRaw: Bool?
    var paintQualityRaw: String?
    var paintStepsRaw: Int?
    var paintResRaw: Int?
    var paintTexRaw: Int?
    var paintSuperresRaw: Bool?
    var paintFacesRaw: Int?
    var paintSeedRaw: UInt64?
    /// Extra reference views (view name → file in the project folder). The main image is
    /// always the front view; with any extra view the multi-view model is used.
    var extraViewsRaw: [String: String]?
    /// Objects placed in this project's composed scene (see SceneComposition).
    var sceneItemsRaw: [SceneItem]?
    /// Flow guide curves drawn on this project's shape (mesh coordinates), used by flow remesh.
    var flowGuidesRaw: [FlowGuide]?
    /// Hand adjustments to the auto-rig (joint offsets, bone influence, weight smoothing).
    var rigConfigRaw: RigConfig?

    init(id: UUID = UUID(), name: String, createdAt: Date = Date()) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
    }

    var rigConfig: RigConfig {
        get { rigConfigRaw ?? RigConfig() }
        set { rigConfigRaw = newValue.isDefault ? nil : newValue }
    }

    var flowGuides: [FlowGuide] {
        get { flowGuidesRaw ?? [] }
        set { flowGuidesRaw = newValue.isEmpty ? nil : newValue }
    }

    var sceneItems: [SceneItem] {
        get { sceneItemsRaw ?? [] }
        set { sceneItemsRaw = newValue.isEmpty ? nil : newValue }
    }

    var extraViews: [ReferenceView: String] {
        (extraViewsRaw ?? [:]).reduce(into: [:]) { acc, kv in
            if let v = ReferenceView(rawValue: kv.key) { acc[v] = kv.value }
        }
    }

    var removeBackground: Bool {
        get { removeBackgroundRaw ?? false }
        set { removeBackgroundRaw = newValue }
    }

    /// All saved generations, oldest → newest.
    var generations: [Generation] {
        get { generationsRaw ?? [] }
        set { generationsRaw = newValue }
    }

    /// The generation currently shown in the viewer (selected, else the newest).
    var currentGeneration: Generation? {
        if let id = selectedGenerationID, let g = generations.first(where: { $0.id == id }) { return g }
        return generations.last
    }

    /// Migration-aware (old four-value raws map onto Small/Large).
    var shapeModel: ShapeModel {
        get { ShapeModel(legacyRaw: modelRaw) }
        set { modelRaw = newValue.rawValue }
    }

    var paintModel: PaintModel {
        get { PaintModel(legacyRaw: paintModelRaw) }
        set { paintModelRaw = newValue.rawValue }
    }

    var quantization: Quantization {
        get { quantRaw.flatMap(Quantization.init(rawValue:)) ?? .full }
        set { quantRaw = newValue.rawValue }
    }

    var advancedMode: Bool {
        get { advancedModeRaw ?? false }
        set { advancedModeRaw = newValue }
    }
    var quality: QualityPreset {
        get { qualityRaw.flatMap(QualityPreset.init(rawValue:)) ?? .fast }
        set { qualityRaw = newValue.rawValue }
    }
    var steps: Int {
        get { stepsRaw ?? shapeModel.defaultSteps }
        set { stepsRaw = newValue }
    }
    var guidance: Double {
        get { guidanceRaw ?? 5.0 }
        set { guidanceRaw = newValue }
    }
    var octree: Int {
        get { octreeRaw ?? 256 }
        set { octreeRaw = newValue }
    }
    /// nil = a fresh random seed per run (the resolved value is recorded).
    var seed: UInt64? {
        get { seedRaw }
        set { seedRaw = newValue }
    }

    /// Effective run parameters from the current mode (normal preset or advanced fields).
    var resolvedSettings: RunSettings {
        if advancedMode {
            return RunSettings(model: shapeModel, quant: quantization, steps: steps,
                               guidance: guidance, octree: octree, seed: seed)
        }
        let p = quality
        return RunSettings(model: shapeModel, quant: .int8, steps: p.steps(for: shapeModel),
                           guidance: 5.0, octree: p.octree, seed: nil)
    }

    // MARK: paint config

    var paintAdvanced: Bool {
        get { paintAdvancedRaw ?? false }
        set { paintAdvancedRaw = newValue }
    }
    var paintQuality: PaintQuality {
        get { paintQualityRaw.flatMap(PaintQuality.init(rawValue:)) ?? .fast }
        set { paintQualityRaw = newValue.rawValue }
    }
    var paintSteps: Int {
        get { paintStepsRaw ?? 10 }
        set { paintStepsRaw = newValue }
    }
    var paintRes: Int {
        get { paintResRaw ?? 512 }
        set { paintResRaw = newValue }
    }
    var paintTex: Int {
        get { paintTexRaw ?? 2048 }
        set { paintTexRaw = newValue }
    }
    var paintSuperres: Bool {
        get { paintSuperresRaw ?? false }
        set { paintSuperresRaw = newValue }
    }
    var paintFaces: Int {
        get { paintFacesRaw ?? 120_000 }
        set { paintFacesRaw = newValue }
    }
    /// nil = a fresh random seed per paint run (the resolved value is recorded).
    var paintSeed: UInt64? {
        get { paintSeedRaw }
        set { paintSeedRaw = newValue }
    }

    var resolvedPaintSettings: PaintSettings {
        if paintAdvanced {
            return PaintSettings(model: paintModel, res: paintRes, steps: paintSteps,
                                 tex: paintTex, superres: paintSuperres, faces: paintFaces,
                                 seed: paintSeed)
        }
        let q = paintQuality
        return PaintSettings(model: paintModel, res: q.res, steps: q.steps, tex: q.tex,
                             superres: q.superres, faces: q.faces, seed: nil)
    }
}

/// One saved model output, with the settings + input that produced it. Immutable.
struct Generation: Identifiable, Codable, Hashable {
    let id: UUID
    var createdAt: Date
    var modelRaw: String
    var quantRaw: String
    var steps: Int
    var removeBackground: Bool
    var meshFileName: String        // gen_<id>.mesh
    var inputFileName: String       // snapshot of the input that was fed to the model
    var durationSeconds: Double?
    var guidanceRaw: Double? = nil
    var octreeRaw: Int? = nil
    var seedRaw: UInt64? = nil          // the seed this run actually used
    var sourceFileName: String? = nil   // original image snapshot (for re-editing)
    var maskFileName: String? = nil     // mask snapshot (nil if no background removal)
    var paintedMeshFileName: String? = nil    // (legacy) textured .tmesh
    var paintedTextureFileName: String? = nil // for a paint version: the baked albedo/base-color
    var paintedMRFileName: String? = nil      // for a PBR version: the metallic-roughness map
    var kindRaw: String? = nil          // shape (default) or paint
    var sourceShapeID: UUID? = nil      // for a paint version: the shape it textured
    var paintModelRaw: String? = nil
    var paintResRaw: Int? = nil
    var paintStepsRaw: Int? = nil
    var paintTexRaw: Int? = nil
    var paintFacesRaw: Int? = nil
    var paintSuperresRaw: Bool? = nil
    var paintSeedRaw: UInt64? = nil     // the seed this paint run actually used
    var editSummary: String? = nil      // for a mesh-edit version: what was done (e.g. "smoothed ×3")
    var editedFromID: UUID? = nil       // the version this edit was derived from
    var extraViewFiles: [String: String]? = nil  // multi-view run: view name → snapshot file

    var kind: GenerationKind { GenerationKind(rawValue: kindRaw ?? "shape") ?? .shape }
    var isPainted: Bool { kind == .paint }
    /// A PBR paint version — it carries a metallic-roughness map alongside the
    /// albedo. Drives physically-based viewer lighting, the two-texture GLB export,
    /// and the history "PBR" badge.
    var isPBR: Bool { kind == .paint && paintedMRFileName != nil }
    var paintModel: PaintModel { PaintModel(legacyRaw: paintModelRaw) }

    var shapeModel: ShapeModel { ShapeModel(legacyRaw: modelRaw) }
    var quantization: Quantization { Quantization(rawValue: quantRaw) ?? .full }
    var guidance: Double { guidanceRaw ?? 5.0 }
    var octree: Int { octreeRaw ?? 256 }
}
