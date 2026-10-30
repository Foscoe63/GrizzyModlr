import Foundation

/// A preset for a new project: shape and paint quality plus background removal. Built-ins
/// are fixed; users can save a project's current settings as their own.
struct ProjectTemplate: Identifiable, Codable, Equatable {
    var name: String
    var detail: String
    var icon: String = "doc"
    var quality: String            // QualityPreset raw value
    var paintQuality: String       // PaintQuality raw value
    var removeBackground: Bool
    var id: String { name }
    var isBuiltIn: Bool { Self.builtIn.contains { $0.name == name } }

    static let builtIn: [ProjectTemplate] = [
        ProjectTemplate(name: "Quick preview", detail: "Fastest shape, fast paint — to judge an idea.",
                        icon: "hare", quality: QualityPreset.fastest.rawValue, paintQuality: PaintQuality.fast.rawValue, removeBackground: true),
        ProjectTemplate(name: "Balanced", detail: "A good default for most images.",
                        icon: "scale.3d", quality: QualityPreset.balanced.rawValue, paintQuality: PaintQuality.balanced.rawValue, removeBackground: true),
        ProjectTemplate(name: "3D print", detail: "High shape detail; paint stays fast (print ignores texture).",
                        icon: "printer", quality: QualityPreset.high.rawValue, paintQuality: PaintQuality.fast.rawValue, removeBackground: true),
        ProjectTemplate(name: "Character", detail: "Balanced shape, high paint, background removed — for rigging.",
                        icon: "figure.stand", quality: QualityPreset.balanced.rawValue, paintQuality: PaintQuality.high.rawValue, removeBackground: true),
        ProjectTemplate(name: "Maximum detail", detail: "Slowest and best, shape and paint.",
                        icon: "sparkles", quality: QualityPreset.max.rawValue, paintQuality: PaintQuality.max.rawValue, removeBackground: true),
        ProjectTemplate(name: "Keep background", detail: "Balanced, without background removal (scenes, objects on a plain ground).",
                        icon: "photo", quality: QualityPreset.balanced.rawValue, paintQuality: PaintQuality.balanced.rawValue, removeBackground: false),
    ]

    private static let key = "tools.customTemplates"

    static func custom(_ d: UserDefaults = .standard) -> [ProjectTemplate] {
        (d.data(forKey: key)).flatMap { try? JSONDecoder().decode([ProjectTemplate].self, from: $0) } ?? []
    }
    static func saveCustom(_ list: [ProjectTemplate], _ d: UserDefaults = .standard) {
        d.set(try? JSONEncoder().encode(list), forKey: key)
    }
    static func all(_ d: UserDefaults = .standard) -> [ProjectTemplate] { builtIn + custom(d) }

    /// The template named in Settings for plain "New Project"; nil = the app's own defaults.
    static func defaultTemplate(_ d: UserDefaults = .standard) -> ProjectTemplate? {
        guard let name = d.string(forKey: ToolDefaults.Key.defaultTemplate), name != "none" else { return nil }
        return all(d).first { $0.name == name }
    }

    /// Adds or replaces a custom template (built-in names are protected).
    static func upsert(_ t: ProjectTemplate, _ d: UserDefaults = .standard) -> Bool {
        guard !builtIn.contains(where: { $0.name == t.name }), !t.name.trimmingCharacters(in: .whitespaces).isEmpty else { return false }
        var list = custom(d).filter { $0.name != t.name }
        list.append(t)
        saveCustom(list, d)
        return true
    }
}
