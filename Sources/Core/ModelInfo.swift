import Foundation

/// How the catalog browser groups models.
enum ModelCategory: String, CaseIterable, Identifiable, Sendable {
    case shape, paint, tools
    var id: String { rawValue }
    var label: String {
        switch self {
        case .shape: return "Shape"
        case .paint: return "Paint"
        case .tools: return "Tools"
        }
    }
}

/// What a model is for and under which license, shown beside each catalog entry.
struct ModelInfo: Sendable {
    let category: ModelCategory
    let usedFor: String
    let license: String
}

extension ModelID {
    var info: ModelInfo {
        let tencent = "Tencent Hunyuan Community"
        switch self {
        case .shapeSmall:     return ModelInfo(category: .shape, usedFor: "Generate · the Small shape model (fastest)", license: tencent)
        case .shapeLarge:     return ModelInfo(category: .shape, usedFor: "Generate · the Large shape model (turbo, high detail)", license: tencent)
        case .shapeFull:      return ModelInfo(category: .shape, usedFor: "Generate · the Full shape model (most detail, many steps, slowest)", license: tencent)
        case .shapeMultiView: return ModelInfo(category: .shape, usedFor: "Generate · used automatically when a project has two or more reference views", license: tencent)
        case .triposr:        return ModelInfo(category: .shape, usedFor: "Generate · the Fast shape model", license: "MIT")
        case .shapeTrellis2:  return ModelInfo(category: .shape, usedFor: "Generate · the TRELLIS.2 shape model (512 level, geometry only; needs DINOv3)", license: "MIT")
        case .shapePixal3d:   return ModelInfo(category: .shape, usedFor: "Generate · the Pixal3D shape model (pixel-aligned to the photo, 512 stage; needs DINOv3). Includes NAF (Apache-2.0)", license: "MIT (third-party parts keep their licenses)")
        case .dinov3:         return ModelInfo(category: .tools, usedFor: "Image encoder for TRELLIS.2 and Pixal3D · gated, request access then use Import", license: "DINOv3 License (Meta)")
        case .paintSmall:     return ModelInfo(category: .paint, usedFor: "Paint · color textures", license: tencent)
        case .paintLarge:     return ModelInfo(category: .paint, usedFor: "Paint · PBR textures (albedo + metallic-roughness)", license: tencent)
        case .viewSynthesis:  return ModelInfo(category: .tools, usedFor: "Synthesize views · side and back views from one photo", license: "CC-BY-NC 4.0 (non-commercial)")
        case .textToImage:    return ModelInfo(category: .tools, usedFor: "Text to 3D starting image and AI Fill", license: "OpenRAIL++")
        case .depthEstimation: return ModelInfo(category: .tools, usedFor: "Depth relief · a 2.5D solid from an image", license: "Apache-2.0")
        }
    }
}

extension CatalogModel {
    /// The model's Hugging Face page.
    var pageURL: URL { URL(string: "https://huggingface.co/\(repo)")! }
}

extension ModelCatalog {
    /// Catalog entries matching `query` (every word must appear in the name, description, use, license
    /// or repo) and, when given, `category`. Catalog order is kept.
    static func search(_ query: String, category: ModelCategory? = nil) -> [ModelID] {
        let words = query.lowercased().split(whereSeparator: { $0.isWhitespace }).map(String.init)
        return all.compactMap { entry -> ModelID? in
            let info = entry.id.info
            if let category, info.category != category { return nil }
            let haystack = [entry.displayName, entry.detail, entry.repo, info.usedFor, info.license, info.category.label]
                .joined(separator: " ").lowercased()
            return words.allSatisfy { haystack.contains($0) } ? entry.id : nil
        }
    }
}
