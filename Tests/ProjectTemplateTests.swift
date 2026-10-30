import XCTest

final class ProjectTemplateTests: XCTestCase {
    private func defaults() -> UserDefaults {
        let d = UserDefaults(suiteName: "tpl-\(UUID().uuidString)")!
        return d
    }

    func testBuiltInsReferenceRealPresets() {
        for t in ProjectTemplate.builtIn {
            XCTAssertNotNil(QualityPreset(rawValue: t.quality), t.name)
            XCTAssertNotNil(PaintQuality(rawValue: t.paintQuality), t.name)
        }
        XCTAssertEqual(Set(ProjectTemplate.builtIn.map(\.name)).count, ProjectTemplate.builtIn.count)
    }

    func testCustomTemplatesRoundTripAndProtectBuiltIns() {
        let d = defaults()
        let mine = ProjectTemplate(name: "Mine", detail: "", quality: "high", paintQuality: "fast", removeBackground: false)
        XCTAssertTrue(ProjectTemplate.upsert(mine, d))
        XCTAssertFalse(ProjectTemplate.upsert(ProjectTemplate(name: "Balanced", detail: "", quality: "fast", paintQuality: "fast", removeBackground: true), d))
        XCTAssertFalse(ProjectTemplate.upsert(ProjectTemplate(name: "  ", detail: "", quality: "fast", paintQuality: "fast", removeBackground: true), d))
        XCTAssertEqual(ProjectTemplate.custom(d), [mine])
        XCTAssertEqual(ProjectTemplate.all(d).count, ProjectTemplate.builtIn.count + 1)
        d.set("Mine", forKey: ToolDefaults.Key.defaultTemplate)
        XCTAssertEqual(ProjectTemplate.defaultTemplate(d), mine)
        d.set("none", forKey: ToolDefaults.Key.defaultTemplate)
        XCTAssertNil(ProjectTemplate.defaultTemplate(d))
    }

    @MainActor func testNewProjectAppliesTemplate() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tpl-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ProjectStore(rootDir: root)
        let t = ProjectTemplate.builtIn.first { $0.name == "Maximum detail" }!
        let p = store.newProject(template: t)
        XCTAssertEqual(store.project(p.id)?.quality, .max)
        XCTAssertEqual(store.project(p.id)?.paintQuality, .max)
        XCTAssertEqual(store.project(p.id)?.removeBackground, true)
        XCTAssertTrue(store.saveTemplate(from: p.id, named: "ZZ test template"))
        defer { ProjectTemplate.saveCustom(ProjectTemplate.custom().filter { $0.name != "ZZ test template" }) }
        XCTAssertTrue(ProjectTemplate.custom().contains { $0.name == "ZZ test template" })
    }
}
