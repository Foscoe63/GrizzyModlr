import Foundation

/// User-configurable defaults for the editing tools (Settings → Tools). Popovers and tool
/// sessions read these as their starting values; the Settings page writes them through
/// `@AppStorage` with the same keys and fallbacks.
enum ToolDefaults {
    enum Key {
        static let decimateK = "tools.decimateK"
        static let smoothIterations = "tools.smoothIterations"
        static let holeEdges = "tools.holeEdges"
        static let booleanResolution = "tools.booleanResolution"
        static let remeshResolution = "tools.remeshResolution"
        static let symmetryAxis = "tools.symmetryAxis"
        static let symmetryKeepPositive = "tools.symmetryKeepPositive"
        static let lodLevels = "tools.lodLevels"
        static let sculptSize = "tools.sculptSize"
        static let sculptStrength = "tools.sculptStrength"
        static let paintSize = "tools.paintSize"
        static let paintHardness = "tools.paintHardness"
        static let paintOpacity = "tools.paintOpacity"
    }

    static let fallbackDecimateK = 50.0
    static let fallbackSmoothIterations = 3.0
    static let fallbackHoleEdges = 64.0
    static let fallbackBooleanResolution = 160.0
    static let fallbackRemeshResolution = 128.0
    static let fallbackSymmetryAxis = "x"
    static let fallbackLODLevels = "50, 25, 10"
    static let fallbackSculptSize = 0.12
    static let fallbackSculptStrength = 0.4
    static let fallbackPaintSize = 0.05
    static let fallbackPaintHardness = 0.6
    static let fallbackPaintOpacity = 1.0

    private static func double(_ key: String, _ fallback: Double, _ d: UserDefaults) -> Double {
        d.object(forKey: key) as? Double ?? fallback
    }

    static func decimateK(_ d: UserDefaults = .standard) -> Double { double(Key.decimateK, fallbackDecimateK, d) }
    static func smoothIterations(_ d: UserDefaults = .standard) -> Double { double(Key.smoothIterations, fallbackSmoothIterations, d) }
    static func holeEdges(_ d: UserDefaults = .standard) -> Double { double(Key.holeEdges, fallbackHoleEdges, d) }
    static func booleanResolution(_ d: UserDefaults = .standard) -> Double { double(Key.booleanResolution, fallbackBooleanResolution, d) }
    static func remeshResolution(_ d: UserDefaults = .standard) -> Double { double(Key.remeshResolution, fallbackRemeshResolution, d) }
    static func sculptSize(_ d: UserDefaults = .standard) -> Double { double(Key.sculptSize, fallbackSculptSize, d) }
    static func sculptStrength(_ d: UserDefaults = .standard) -> Double { double(Key.sculptStrength, fallbackSculptStrength, d) }
    static func paintSize(_ d: UserDefaults = .standard) -> Double { double(Key.paintSize, fallbackPaintSize, d) }
    static func paintHardness(_ d: UserDefaults = .standard) -> Double { double(Key.paintHardness, fallbackPaintHardness, d) }
    static func paintOpacity(_ d: UserDefaults = .standard) -> Double { double(Key.paintOpacity, fallbackPaintOpacity, d) }

    static func symmetryAxis(_ d: UserDefaults = .standard) -> SymmetryAxis {
        SymmetryAxis(rawValue: d.string(forKey: Key.symmetryAxis) ?? fallbackSymmetryAxis) ?? .x
    }
    static func symmetryKeepPositive(_ d: UserDefaults = .standard) -> Bool {
        d.object(forKey: Key.symmetryKeepPositive) as? Bool ?? true
    }

    /// "50, 25, 10" → `[50, 25, 10]` percent of the original triangle count. Entries outside
    /// 1…95 or that aren't numbers are dropped; duplicates removed; highest detail first.
    static func parseLODLevels(_ text: String) -> [Int] {
        let nums = text.split(whereSeparator: { ", ;/".contains($0) })
            .compactMap { Int($0.trimmingCharacters(in: CharacterSet(charactersIn: "% "))) }
            .filter { (1...95).contains($0) }
        return Array(Set(nums)).sorted(by: >)
    }
    static func lodLevels(_ d: UserDefaults = .standard) -> [Int] {
        let levels = parseLODLevels(d.string(forKey: Key.lodLevels) ?? fallbackLODLevels)
        return levels.isEmpty ? parseLODLevels(fallbackLODLevels) : levels
    }
}
