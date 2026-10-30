import Foundation
import Vision

/// Whether a photo shows a person, so the reference-views strip can suggest synthesizing the side
/// and back views a single photo can't show. Runs Vision's on-device face and body detectors.
enum PersonDetector {
    static func containsPerson(at url: URL) async -> Bool {
        await Task.detached(priority: .utility) {
            let faces = VNDetectFaceRectanglesRequest()
            let bodies = VNDetectHumanRectanglesRequest()
            let handler = VNImageRequestHandler(url: url, options: [:])
            do { try handler.perform([faces, bodies]) } catch { return false }
            return !(faces.results ?? []).isEmpty || !(bodies.results ?? []).isEmpty
        }.value
    }
}
