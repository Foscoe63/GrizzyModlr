import XCTest
import simd

final class ViewerDefaultsTests: XCTestCase {
    func testCameraDirectionsAreUnitAndDistinct() {
        for p in CameraPreset.allCases {
            XCTAssertEqual(simd_length(p.direction), 1, accuracy: 1e-5)
            XCTAssertEqual(simd_length(p.up), 1, accuracy: 1e-5)
            XCTAssertLessThan(abs(simd_dot(p.direction, p.up)), 0.99)          // never parallel to up: a valid look-at
        }
        XCTAssertEqual(Set(CameraPreset.allCases.map { "\($0.direction)" }).count, CameraPreset.allCases.count)
        XCTAssertEqual(CameraPreset.front.direction, SIMD3<Float>(0, 0, 1))
        XCTAssertGreaterThan(CameraPreset.threeQuarter.direction.y, 0)
    }

    func testOptionsFallbackAndStoredValues() {
        let name = "vd-\(UUID().uuidString)"
        let d = UserDefaults(suiteName: name)!
        defer { d.removePersistentDomain(forName: name) }
        var o = ViewerDefaults.options(d)
        XCTAssertEqual(o.environment, .none)
        XCTAssertFalse(o.isStaged)
        d.set("studio", forKey: ViewerDefaults.Key.environment)
        d.set(true, forKey: ViewerDefaults.Key.shadows)
        d.set(1.5, forKey: ViewerDefaults.Key.environmentIntensity)
        o = ViewerDefaults.options(d)
        XCTAssertEqual(o.environment, .studio)
        XCTAssertTrue(o.shadows)
        XCTAssertEqual(o.intensity, 1.5)
        XCTAssertTrue(o.isStaged)
        d.set("bogus", forKey: ViewerDefaults.Key.environment)
        XCTAssertEqual(ViewerDefaults.options(d).environment, .none)
    }

    func testCommandsAreUnique() {
        XCTAssertNotEqual(ViewerCommand(kind: .snapshot), ViewerCommand(kind: .snapshot))
    }
}
