import XCTest

final class ToolDefaultsTests: XCTestCase {
    private func suite() -> UserDefaults {
        let name = "td-\(UUID().uuidString)"
        let d = UserDefaults(suiteName: name)!
        addTeardownBlock { d.removePersistentDomain(forName: name) }
        return d
    }

    func testFallbacksWhenNothingIsSet() {
        let d = suite()
        XCTAssertEqual(ToolDefaults.decimateK(d), 50)
        XCTAssertEqual(ToolDefaults.booleanResolution(d), 160)
        XCTAssertEqual(ToolDefaults.symmetryAxis(d), .x)
        XCTAssertTrue(ToolDefaults.symmetryKeepPositive(d))
        XCTAssertEqual(ToolDefaults.lodLevels(d), [50, 25, 10])
    }

    func testStoredValuesWin() {
        let d = suite()
        d.set(120.0, forKey: ToolDefaults.Key.decimateK)
        d.set("z", forKey: ToolDefaults.Key.symmetryAxis)
        d.set(false, forKey: ToolDefaults.Key.symmetryKeepPositive)
        XCTAssertEqual(ToolDefaults.decimateK(d), 120)
        XCTAssertEqual(ToolDefaults.symmetryAxis(d), .z)
        XCTAssertFalse(ToolDefaults.symmetryKeepPositive(d))
    }

    func testLODParsingIsForgiving() {
        XCTAssertEqual(ToolDefaults.parseLODLevels("50, 25, 10"), [50, 25, 10])
        XCTAssertEqual(ToolDefaults.parseLODLevels("10% 75;25 / 25, abc, 0, 99, 200"), [75, 25, 10])
        XCTAssertEqual(ToolDefaults.parseLODLevels("nonsense"), [])
        let d = suite()
        d.set("nonsense", forKey: ToolDefaults.Key.lodLevels)
        XCTAssertEqual(ToolDefaults.lodLevels(d), [50, 25, 10])             // falls back, never empty
    }
}
