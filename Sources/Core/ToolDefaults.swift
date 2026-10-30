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
        static let booleanSharp = "tools.booleanSharp"
        static let remeshResolution = "tools.remeshResolution"
        static let symmetryAxis = "tools.symmetryAxis"
        static let symmetryKeepPositive = "tools.symmetryKeepPositive"
        static let lodLevels = "tools.lodLevels"
        static let sculptSize = "tools.sculptSize"
        static let sculptStrength = "tools.sculptStrength"
        static let paintSize = "tools.paintSize"
        static let paintHardness = "tools.paintHardness"
        static let paintOpacity = "tools.paintOpacity"
        static let sceneSpacing = "tools.sceneSpacing"
        static let exportQuads = "tools.exportQuads"
        static let aiFillPrompt = "tools.aiFillPrompt"
        static let aiFillSteps = "tools.aiFillSteps"
        static let aiFillStrength = "tools.aiFillStrength"
        static let aiFillGuidance = "tools.aiFillGuidance"
        static let regionTolerance = "tools.regionTolerance"
        static let regionFeather = "tools.regionFeather"
        static let tsrThreshold = "tools.tsrThreshold"
        static let tsrForegroundRatio = "tools.tsrForegroundRatio"
        static let tsrKeepLargest = "tools.tsrKeepLargest"
        static let reliefDepth = "tools.reliefDepth"
        static let reliefResolution = "tools.reliefResolution"
        static let reliefSmoothing = "tools.reliefSmoothing"
        static let ttiSteps = "tools.ttiSteps"
        static let ttiGuidance = "tools.ttiGuidance"
        static let ttiNegative = "tools.ttiNegative"
        static let ttiSuffix = "tools.ttiSuffix"
        static let ttiRemoveBackground = "tools.ttiRemoveBackground"
        static let ttiAutoGenerate = "tools.ttiAutoGenerate"
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
    static let fallbackSceneSpacing = 2.5
    static let fallbackAIFillPrompt = "seamless surface texture, same material and colors as the surrounding area"
    static let fallbackAIFillSteps = 25.0
    static let fallbackAIFillStrength = 0.85
    static let fallbackAIFillGuidance = 6.0
    static let fallbackRegionTolerance = 0.12
    static let fallbackRegionFeather = 0.10
    static let fallbackTSRThreshold = 25.0
    static let fallbackTSRForegroundRatio = 0.85
    static let fallbackReliefDepth = 0.25
    static let fallbackReliefResolution = 256.0
    static let fallbackReliefSmoothing = 2.0
    static let fallbackTTISteps = 25.0
    static let fallbackTTIGuidance = 7.5
    static let fallbackTTINegative = "blurry, low quality, deformed, multiple objects, text, watermark, cropped"
    static let fallbackTTISuffix = ", a single object, centered, plain light background, soft studio lighting, 3d render"

    private static func double(_ key: String, _ fallback: Double, _ d: UserDefaults) -> Double {
        d.object(forKey: key) as? Double ?? fallback
    }

    static func decimateK(_ d: UserDefaults = .standard) -> Double { double(Key.decimateK, fallbackDecimateK, d) }
    static func smoothIterations(_ d: UserDefaults = .standard) -> Double { double(Key.smoothIterations, fallbackSmoothIterations, d) }
    static func holeEdges(_ d: UserDefaults = .standard) -> Double { double(Key.holeEdges, fallbackHoleEdges, d) }
    static func booleanResolution(_ d: UserDefaults = .standard) -> Double { double(Key.booleanResolution, fallbackBooleanResolution, d) }
    static func booleanSharp(_ d: UserDefaults = .standard) -> Bool { d.object(forKey: Key.booleanSharp) as? Bool ?? true }
    static func remeshResolution(_ d: UserDefaults = .standard) -> Double { double(Key.remeshResolution, fallbackRemeshResolution, d) }
    static func sculptSize(_ d: UserDefaults = .standard) -> Double { double(Key.sculptSize, fallbackSculptSize, d) }
    static func sculptStrength(_ d: UserDefaults = .standard) -> Double { double(Key.sculptStrength, fallbackSculptStrength, d) }
    static func paintSize(_ d: UserDefaults = .standard) -> Double { double(Key.paintSize, fallbackPaintSize, d) }
    static func paintHardness(_ d: UserDefaults = .standard) -> Double { double(Key.paintHardness, fallbackPaintHardness, d) }
    static func paintOpacity(_ d: UserDefaults = .standard) -> Double { double(Key.paintOpacity, fallbackPaintOpacity, d) }
    static func sceneSpacing(_ d: UserDefaults = .standard) -> Double { double(Key.sceneSpacing, fallbackSceneSpacing, d) }
    static func aiFillPrompt(_ d: UserDefaults = .standard) -> String {
        let t = d.string(forKey: Key.aiFillPrompt) ?? ""
        return t.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? fallbackAIFillPrompt : t
    }
    static func aiFillSteps(_ d: UserDefaults = .standard) -> Int { Int(double(Key.aiFillSteps, fallbackAIFillSteps, d)) }
    static func aiFillStrength(_ d: UserDefaults = .standard) -> Double { double(Key.aiFillStrength, fallbackAIFillStrength, d) }
    static func aiFillGuidance(_ d: UserDefaults = .standard) -> Double { double(Key.aiFillGuidance, fallbackAIFillGuidance, d) }
    static func regionTolerance(_ d: UserDefaults = .standard) -> Double { double(Key.regionTolerance, fallbackRegionTolerance, d) }
    static func regionFeather(_ d: UserDefaults = .standard) -> Double { double(Key.regionFeather, fallbackRegionFeather, d) }
    static func exportQuads(_ d: UserDefaults = .standard) -> Bool { d.bool(forKey: Key.exportQuads) }
    static func tsrThreshold(_ d: UserDefaults = .standard) -> Float { Float(double(Key.tsrThreshold, fallbackTSRThreshold, d)) }
    static func tsrForegroundRatio(_ d: UserDefaults = .standard) -> Float { Float(double(Key.tsrForegroundRatio, fallbackTSRForegroundRatio, d)) }
    static func tsrKeepLargest(_ d: UserDefaults = .standard) -> Bool { d.object(forKey: Key.tsrKeepLargest) as? Bool ?? true }
    static func reliefDepth(_ d: UserDefaults = .standard) -> Double { double(Key.reliefDepth, fallbackReliefDepth, d) }
    static func reliefResolution(_ d: UserDefaults = .standard) -> Int { Int(double(Key.reliefResolution, fallbackReliefResolution, d)) }
    static func reliefSmoothing(_ d: UserDefaults = .standard) -> Int { Int(double(Key.reliefSmoothing, fallbackReliefSmoothing, d)) }
    static func ttiSteps(_ d: UserDefaults = .standard) -> Int { Int(double(Key.ttiSteps, fallbackTTISteps, d)) }
    static func ttiGuidance(_ d: UserDefaults = .standard) -> Double { double(Key.ttiGuidance, fallbackTTIGuidance, d) }
    static func ttiNegative(_ d: UserDefaults = .standard) -> String { d.string(forKey: Key.ttiNegative) ?? fallbackTTINegative }
    static func ttiSuffix(_ d: UserDefaults = .standard) -> String { d.string(forKey: Key.ttiSuffix) ?? fallbackTTISuffix }
    static func ttiRemoveBackground(_ d: UserDefaults = .standard) -> Bool { d.object(forKey: Key.ttiRemoveBackground) as? Bool ?? true }
    static func ttiAutoGenerate(_ d: UserDefaults = .standard) -> Bool { d.bool(forKey: Key.ttiAutoGenerate) }

    /// The full positive prompt: the user's text plus the configured style suffix.
    static func composePrompt(_ text: String, suffix: String) -> String {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? "" : t + suffix
    }

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
