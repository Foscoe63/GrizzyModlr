import Foundation
import simd

/// How the 3D viewport is lit and staged (Settings → Viewer; quick toggles in the viewport menu
/// write the same keys).
enum ViewerEnvironment: String, CaseIterable, Identifiable {
    case none, studio, outdoor, sunset, neutral, custom
    var id: String { rawValue }
    var label: String {
        switch self {
        case .none: return "Default lights"
        case .studio: return "Studio"
        case .outdoor: return "Outdoor"
        case .sunset: return "Sunset"
        case .neutral: return "Neutral gray"
        case .custom: return "Custom HDR…"
        }
    }
}

/// A named viewpoint, as a direction from the model's center toward the camera.
enum CameraPreset: String, CaseIterable, Identifiable {
    case front, back, left, right, top, threeQuarter
    var id: String { rawValue }
    var label: String {
        switch self {
        case .front: return "Front"
        case .back: return "Back"
        case .left: return "Left"
        case .right: return "Right"
        case .top: return "Top"
        case .threeQuarter: return "Three-quarter"
        }
    }
    var icon: String {
        switch self {
        case .front: return "person.fill.viewfinder"
        case .back: return "arrow.uturn.backward"
        case .left: return "arrow.left"
        case .right: return "arrow.right"
        case .top: return "arrow.up"
        case .threeQuarter: return "cube"
        }
    }

    /// Unit vector from the center toward the camera (Y up, +Z toward the viewer at "front").
    var direction: SIMD3<Float> {
        switch self {
        case .front: return [0, 0, 1]
        case .back: return [0, 0, -1]
        case .left: return [-1, 0, 0]
        case .right: return [1, 0, 0]
        case .top: return [0, 1, 0]
        case .threeQuarter:
            let az = Float(35) * .pi / 180, el = Float(20) * .pi / 180
            return [sin(az) * cos(el), sin(el), cos(az) * cos(el)]
        }
    }
    /// Up vector to use with `direction` (a straight-down view can't use +Y).
    var up: SIMD3<Float> { self == .top ? [0, 0, -1] : [0, 1, 0] }
}

enum ViewerDefaults {
    enum Key {
        static let environment = "viewer.environment"
        static let environmentIntensity = "viewer.environmentIntensity"
        static let environmentBackground = "viewer.environmentBackground"
        static let showGround = "viewer.showGround"
        static let shadows = "viewer.shadows"
        static let autoRotate = "viewer.autoRotate"
        static let customEnvironmentPath = "viewer.customEnvironmentPath"
    }
    static let fallbackEnvironment = ViewerEnvironment.none.rawValue
    static let fallbackIntensity = 1.0

    /// Everything the stage needs, read once per update.
    struct Options: Equatable {
        var environment: ViewerEnvironment = .none
        var intensity: Double = fallbackIntensity
        var environmentBackground = false
        var showGround = false
        var shadows = false
        var autoRotate = false
        var customPath = ""

        /// True when anything beyond the original plain look is requested.
        var isStaged: Bool { environment != .none || showGround || shadows }
    }

    static func options(_ d: UserDefaults = .standard) -> Options {
        Options(environment: ViewerEnvironment(rawValue: d.string(forKey: Key.environment) ?? fallbackEnvironment) ?? .none,
                intensity: d.object(forKey: Key.environmentIntensity) as? Double ?? fallbackIntensity,
                environmentBackground: d.bool(forKey: Key.environmentBackground),
                showGround: d.bool(forKey: Key.showGround),
                shadows: d.bool(forKey: Key.shadows),
                autoRotate: d.bool(forKey: Key.autoRotate),
                customPath: d.string(forKey: Key.customEnvironmentPath) ?? "")
    }
}

/// A one-shot request from the viewport menu to the SceneKit view (identified so the view
/// acts on each request exactly once).
struct ViewerCommand: Equatable {
    enum Kind: Equatable { case camera(CameraPreset), snapshot }
    let id = UUID()
    let kind: Kind
}
