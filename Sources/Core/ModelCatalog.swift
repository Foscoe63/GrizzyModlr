import Foundation

/// The installable models (DESIGN.md §2): four core slots plus the optional multi-view shape model.
enum ModelID: String, CaseIterable, Codable, Hashable, Identifiable, Sendable {
    case shapeSmall = "shape-small"
    case shapeLarge = "shape-large"
    case paintSmall = "paint-small"
    case paintLarge = "paint-large"
    /// Hunyuan3D-2mv: optional, used automatically when a project has 2+ reference views.
    case shapeMultiView = "shape-multiview"
    /// Zero123++ v1.2: optional, turns the main image into six candidate side views.
    case viewSynthesis = "view-synthesis"
    /// Stable Diffusion 2.1 base: optional, turns a text prompt into the starting image.
    case textToImage = "text-to-image"
    /// Depth Anything V2 Small: optional, powers depth relief meshes.
    case depthEstimation = "depth-estimation"
    /// TripoSR: a fast single-pass shape model (seconds); the "Fast" shape option.
    case triposr = "triposr"

    var id: String { rawValue }

    /// Directory name under `Application Support/Modelr/models/` (§2.3).
    var slug: String { rawValue }

    var kind: EngineKind {
        switch self {
        case .shapeSmall, .shapeLarge, .shapeMultiView, .triposr: return .shape
        case .paintSmall, .paintLarge, .viewSynthesis, .textToImage, .depthEstimation: return .paint
        }
    }
}

/// One file in a model's weight bundle. Download correctness = size + sha256 match.
struct CatalogFile: Hashable, Sendable {
    /// Path relative to both the HF repo root and the local install dir.
    let path: String
    let bytes: Int64
    let sha256: String
    /// Path in the HF repo when it differs from the local install path (nil = same).
    var remotePath: String? = nil
}

struct CatalogModel: Sendable {
    let id: ModelID
    let displayName: String
    let detail: String
    /// HuggingFace repo id (owned/curated — byte layout is a contract we control).
    let repo: String
    /// Pinned revision; nil resolves to `main` until the upload pins a commit.
    let revision: String?
    let files: [CatalogFile]

    var totalBytes: Int64 { files.reduce(0) { $0 + $1.bytes } }

    /// Resolve the download URL for one file.
    func remoteURL(for file: CatalogFile) -> URL {
        let rev = revision ?? "main"
        return URL(string: "https://huggingface.co/\(repo)/resolve/\(rev)/\(file.remotePath ?? file.path)?download=true")!
    }
}

/// Static, baked manifest (DESIGN.md §2.2): repo ids and file-level byte sizes +
/// sha256 digests for every model. No runtime manifest fetch.
///
/// REGENERATION: the constants below are generated from `model_manifest.json`
/// at the repo root (produced at HF upload time from the staged bundles).
/// To regenerate after re-uploading weights, run:
///
///     swift scripts/gen_catalog.swift model_manifest.json
///
/// and replace the `all` array below with its output.
enum ModelCatalog {
    static let all: [CatalogModel] = [
        CatalogModel(
            id: .shapeSmall,
            displayName: "Shape · Small",
            detail: "Hunyuan3D 2mini · 0.6B · ~5 s per shape",
            repo: "zimengxiong/hunyuan3d-mlx-shape-small",
            revision: "b7536809d38ad13fe6a9b7769a41fd5d42e520df",
            files: [
                CatalogFile(path: "config.yaml", bytes: 1_628,
                            sha256: "cabcba7f6115752c8fe5b370e12bf714936f70377a8a80f151872f76c2d64609"),
                CatalogFile(path: "model.fp16.safetensors", bytes: 3_819_958_234,
                            sha256: "3cc66f3bea33e4062b7dbc875ffe1d70c4888914aec3e91b60f94e9bd01b522b"),
            ]),
        CatalogModel(
            id: .shapeLarge,
            displayName: "Shape · Large",
            detail: "Hunyuan3D 2.0 turbo · 1.1B distilled · ~17 s per shape",
            repo: "zimengxiong/hunyuan3d-mlx-shape-large",
            revision: "5ad04593521ff80875ac007ef8185209368f18bc",
            files: [
                CatalogFile(path: "config.yaml", bytes: 1_663,
                            sha256: "4712421fe32d57c678be3be2b608acdb8871985f0b81646a883d136abc63feb7"),
                CatalogFile(path: "model.fp16.safetensors", bytes: 4_930_777_530,
                            sha256: "5ee5a81e4df08a1c65b79910bf5b145a90376e526794f4607a4d5d068d62f269"),
            ]),
        CatalogModel(
            id: .paintSmall,
            displayName: "Paint · Small",
            detail: "Color texture (SD2.1) · 2048 atlas",
            repo: "zimengxiong/hunyuan3d-mlx-paint-small",
            revision: "29bab9dbf2a4a4f9c6988a41b0e891156a517a23",
            files: [
                CatalogFile(path: "realesrgan/rrdbnet_mlx.safetensors", bytes: 66_857_885,
                            sha256: "67fbfb07607e3457154dddf9d29dfa6c65f393c2c531af700ae4370499c978ff"),
                CatalogFile(path: "unet/config.json", bytes: 911,
                            sha256: "ce0c6d379e3b1d3e1f79338de70c80a1e36f17a6372439a8249b6fb0dfa1b608"),
                CatalogFile(path: "unet/diffusion_pytorch_model.safetensors", bytes: 3_662_636_472,
                            sha256: "1c5ce434ba976b30bbb51a080917ecd39f8d8761691887b5df3be765ef4bd9e9"),
                CatalogFile(path: "vae/config.json", bytes: 553,
                            sha256: "424117cb534ce03497c41305ed868980123917b2b6abba4bbaa615e968772903"),
                CatalogFile(path: "vae/diffusion_pytorch_model.safetensors", bytes: 167_335_310,
                            sha256: "abcec86e499e1ce9f05d1630725d386dc533b61fe0947ab034f07f89042e7a61"),
            ]),
        CatalogModel(
            id: .paintLarge,
            displayName: "Paint · Large",
            detail: "PBR texture (albedo + metallic-roughness) · 4096 atlas",
            repo: "zimengxiong/hunyuan3d-mlx-paint-large",
            revision: "b56e8b86b4d1e62b0bb3bbef7e2070d6ec22620e",
            files: [
                CatalogFile(path: "dinov2/config.json", bytes: 548,
                            sha256: "15d2e68b709674c0eb9bb5cc8b73e57ebe9a169a425bef224f46ae10053806a2"),
                CatalogFile(path: "dinov2/model.safetensors", bytes: 4_546_005_432,
                            sha256: "917d3c470db999d32a312f8542149be91c7cbac61ee8fb4b67ae3d82b79ce21f"),
                CatalogFile(path: "dinov2/preprocessor_config.json", bytes: 436,
                            sha256: "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"),
                CatalogFile(path: "realesrgan/rrdbnet_mlx.safetensors", bytes: 66_857_885,
                            sha256: "67fbfb07607e3457154dddf9d29dfa6c65f393c2c531af700ae4370499c978ff"),
                CatalogFile(path: "scheduler/scheduler_config.json", bytes: 387,
                            sha256: "9a6a7268e4d7c1bc5000450927070e338dc57a51f4530078627c88f5828e792d"),
                CatalogFile(path: "unet/config.json", bytes: 911,
                            sha256: "ce0c6d379e3b1d3e1f79338de70c80a1e36f17a6372439a8249b6fb0dfa1b608"),
                CatalogFile(path: "unet/diffusion_pytorch_model.safetensors", bytes: 3_924_737_160,
                            sha256: "7b5bbe5ce5e40b816e55e0d0e3b8236759bdbd8b4cbaa90d497b3278fa68f10c"),
                CatalogFile(path: "vae/config.json", bytes: 553,
                            sha256: "424117cb534ce03497c41305ed868980123917b2b6abba4bbaa615e968772903"),
                CatalogFile(path: "vae/diffusion_pytorch_model.safetensors", bytes: 167_335_310,
                            sha256: "abcec86e499e1ce9f05d1630725d386dc533b61fe0947ab034f07f89042e7a61"),
            ]),
        CatalogModel(
            id: .shapeMultiView,
            displayName: "Shape · Multi-view",
            detail: "Hunyuan3D 2mv · 1.1B · front, left, back, right reference views",
            repo: "tencent/Hunyuan3D-2mv",
            revision: "3a761b539b29fe4ff64714813aa9560fd66f5de0",
            files: [
                CatalogFile(path: "config.yaml", bytes: 1_608,
                            sha256: "315fd5bf601d1d103130b9fda202f1bd28eda495e68ed1621bc823d5519c5e5b",
                            remotePath: "hunyuan3d-dit-v2-mv/config.yaml"),
                CatalogFile(path: "model.fp16.safetensors", bytes: 4_928_151_562,
                            sha256: "d36f5881bcdc56726b73e517cd444c13c60732431622da7268145355c8d38e9c",
                            remotePath: "hunyuan3d-dit-v2-mv/model.fp16.safetensors"),
            ]),
        CatalogModel(
            id: .viewSynthesis,
            displayName: "View synthesis",
            detail: "Zero123++ v1.2 · six novel views from one image · weights CC-BY-NC 4.0 (non-commercial)",
            repo: "sudo-ai/zero123plus-v1.2",
            revision: "2da07e89919e1a130c9b5add1584c70c7aa065fd",
            files: [
                CatalogFile(path: "text_encoder/model.safetensors", bytes: 680_820_392,
                            sha256: "bc1827c465450322616f06dea41596eac7d493f4e95904dcb51f0fc745c4e13f"),
                CatalogFile(path: "unet/diffusion_pytorch_model.safetensors", bytes: 3_463_726_504,
                            sha256: "481a8c8d268810095c6bc85ea71b44a0bb9a5ea6457d6c8792275f43989442a3"),
                CatalogFile(path: "vae/diffusion_pytorch_model.safetensors", bytes: 167_335_342,
                            sha256: "3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a"),
                CatalogFile(path: "vision_encoder/model.safetensors", bytes: 1_264_217_240,
                            sha256: "ae616c24393dd1854372b0639e5541666f7521cbe219669255e865cb7f89466a"),
            ]),
        CatalogModel(
            id: .textToImage,
            displayName: "Text to image",
            detail: "Stable Diffusion 2.1 base · 512² · makes the starting image for text-to-3D (OpenRAIL++ license)",
            repo: "sd2-community/stable-diffusion-2-1-base",
            revision: "4e63672c03103b6c636b8fb4119ba982469b2955",
            files: [
                CatalogFile(path: "text_encoder/model.fp16.safetensors", bytes: 680_821_096,
                            sha256: "681c555376658c81dc273f2d737a2aeb23ddb6d1d8e5b3a7064636d359a22668"),
                CatalogFile(path: "tokenizer/merges.txt", bytes: 524_619,
                            sha256: "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"),
                CatalogFile(path: "tokenizer/vocab.json", bytes: 1_059_962,
                            sha256: "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"),
                CatalogFile(path: "unet/diffusion_pytorch_model.fp16.safetensors", bytes: 1_731_904_736,
                            sha256: "28ec9cf3b239c0751c201b1f6fb46b551df5862731b30a37aa1360101cb3fbab"),
                CatalogFile(path: "vae/diffusion_pytorch_model.fp16.safetensors", bytes: 167_335_342,
                            sha256: "3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a"),
            ]),
        CatalogModel(
            id: .depthEstimation,
            displayName: "Depth estimation",
            detail: "Depth Anything V2 Small · 25M · single-image depth for relief meshes (Apache-2.0)",
            repo: "depth-anything/Depth-Anything-V2-Small-hf",
            revision: "5426e4f0f36572d16453bbda7a8389317b1bef99",
            files: [
                CatalogFile(path: "model.safetensors", bytes: 99_173_660,
                            sha256: "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"),
            ]),
        CatalogModel(
            id: .triposr,
            displayName: "Shape · Fast",
            detail: "TripoSR · 0.4B · shape in seconds, lower detail (MIT license)",
            repo: "stabilityai/TripoSR",
            revision: "5b521936b01fbe1890f6f9baed0254ab6351c04a",
            files: [
                CatalogFile(path: "model.ckpt", bytes: 1_677_246_742,
                            sha256: "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee"),
            ]),
    ]

    static func model(_ id: ModelID) -> CatalogModel {
        all.first { $0.id == id }!
    }

    // MARK: onboarding bundles (§4.2)

    /// Fast start: small pair (~7.7 GB).
    static let fastStart: Set<ModelID> = [.shapeSmall, .paintSmall]
    /// Best quality: large pair (~13.6 GB).
    static let bestQuality: Set<ModelID> = [.shapeLarge, .paintLarge]
    /// Everything in the core set (~21.3 GB). The optional multi-view model is installed
    /// on demand from model management, not offered during onboarding.
    static let everything: Set<ModelID> = [.shapeSmall, .shapeLarge, .paintSmall, .paintLarge]

    static func totalBytes(of models: Set<ModelID>) -> Int64 {
        models.reduce(0) { $0 + model($1).totalBytes }
    }
}
