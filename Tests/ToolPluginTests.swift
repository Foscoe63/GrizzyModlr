import XCTest

final class ToolPluginTests: XCTestCase {
    private var dir: URL!
    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("pl-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: dir) }

    func testManifestsLoadAndBadOnesAreSkipped() throws {
        try Data(#"{"name":"A","command":"cp","args":["{input}","{output}"],"input":"stl","output":"stl"}"#.utf8)
            .write(to: dir.appendingPathComponent("a.json"))
        try Data("not json".utf8).write(to: dir.appendingPathComponent("b.json"))
        try Data(#"{"name":"","command":"x"}"#.utf8).write(to: dir.appendingPathComponent("c.json"))
        let plugins = ToolPlugin.load(from: dir)
        XCTAssertEqual(plugins.map(\.name), ["A"])
        XCTAssertEqual(plugins[0].inputFormat, .stl)
    }

    func testPluginRunsOnExportedModel() throws {
        var d = Data()
        func le<T>(_ v: T) { var x = v; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
        le(Int32(4)); le(Int32(4))
        MeshExporterTests.verts.forEach { le($0) }; MeshExporterTests.normals.forEach { le($0) }
        MeshExporterTests.faces.forEach { le($0) }
        let mesh = dir.appendingPathComponent("m.mesh"); try d.write(to: mesh)
        let plugin = ToolPlugin(name: "copy", command: "cp", args: ["{input}", "{output}"], input: "stl", output: "stl")
        let result = try plugin.run(meshURL: mesh, texture: nil, metallicRoughness: nil, projectName: "bust")
        XCTAssertEqual(result.status, 0, result.log)
        let produced = try XCTUnwrap(result.produced)
        XCTAssertEqual(try Data(contentsOf: produced).count, 84 + 4 * 50)   // binary STL of 4 faces
    }

    func testFailingPluginReportsStatus() throws {
        var d = Data()
        func le<T>(_ v: T) { var x = v; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
        le(Int32(4)); le(Int32(4))
        MeshExporterTests.verts.forEach { le($0) }; MeshExporterTests.normals.forEach { le($0) }
        MeshExporterTests.faces.forEach { le($0) }
        let mesh = dir.appendingPathComponent("m.mesh"); try d.write(to: mesh)
        let plugin = ToolPlugin(name: "bad", command: "false")
        let result = try plugin.run(meshURL: mesh, texture: nil, metallicRoughness: nil, projectName: "x")
        XCTAssertNotEqual(result.status, 0)
        XCTAssertNil(result.produced)
    }

    func testRiggedInputFormatResolves() {
        let p = ToolPlugin(name: "x", command: "true", input: "glbRigged")
        XCTAssertEqual(p.inputFormat, .glbRigged)
        XCTAssertEqual(p.inputFormat.ext, "glb")
    }
}
