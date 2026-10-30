import XCTest

final class PrintExportTests: XCTestCase {
    private var dir: URL!
    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("pe-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: dir) }

    private func tetra(uvs: Bool) throws -> URL {
        let verts = MeshExporterTests.verts, normals = MeshExporterTests.normals, faces = MeshExporterTests.faces
        var d = Data()
        func le<T>(_ v: T) { var x = v; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
        le(Int32(4)); le(Int32(4))
        verts.forEach { le($0) }; normals.forEach { le($0) }
        if uvs { MeshExporterTests.uvs.forEach { le($0) } }
        faces.forEach { le($0) }
        let url = dir.appendingPathComponent(uvs ? "m.tmesh" : "m.mesh")
        try d.write(to: url)
        return url
    }

    func testCRC32KnownValue() {
        XCTAssertEqual(Zip.crc32(Data("123456789".utf8)), 0xCBF43926)
    }

    func testThreeMFIsAValidZipWithScaledGeometry() throws {
        let dest = dir.appendingPathComponent("t.3mf")
        let src = try tetra(uvs: false)
        let mesh = try MeshExporter.read(meshURL: src, textureURL: nil)
        try ThreeMFExporter.encode(mesh, maxDimensionMM: 50).write(to: dest)

        let test = Process()
        test.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        test.arguments = ["-tq", dest.path]
        let pipe = Pipe(); test.standardOutput = pipe; test.standardError = pipe
        try test.run(); test.waitUntilExit()
        XCTAssertEqual(test.terminationStatus, 0, String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "")

        let xml = Process()
        xml.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        xml.arguments = ["-p", dest.path, "3D/3dmodel.model"]
        let out = Pipe(); xml.standardOutput = out
        try xml.run()
        let data = out.fileHandleForReading.readDataToEndOfFile(); xml.waitUntilExit()
        let counter = ModelCounter()
        let parser = XMLParser(data: data); parser.delegate = counter
        XCTAssertTrue(parser.parse())
        XCTAssertEqual(counter.vertices.count, 4)
        XCTAssertEqual(counter.triangles, 4)
        let extent = (0..<3).map { axis in
            (counter.vertices.map { $0[axis] }.max()! - counter.vertices.map { $0[axis] }.min()!)
        }.max()!
        XCTAssertEqual(extent, 50, accuracy: 1e-3)                       // longest side scaled to 50 mm
        XCTAssertEqual(counter.vertices.map { $0[2] }.min()!, 0, accuracy: 1e-5)   // sits on the bed
    }

    func testFBXHasClosedPolygonsAndUVs() throws {
        let src = try tetra(uvs: true)
        let mesh = try MeshExporter.read(meshURL: src, textureURL: dir.appendingPathComponent("x.png"))
        let text = String(decoding: FBXExporter.encode(mesh, name: "t", textureName: "t.png"), as: UTF8.self)
        XCTAssertTrue(text.contains("Vertices: *12"))
        XCTAssertTrue(text.contains("PolygonVertexIndex: *12"))
        XCTAssertTrue(text.contains("LayerElementUV"))
        XCTAssertTrue(text.contains("RelativeFilename: \"t.png\""))
        // Every third index is a negative end-of-polygon marker.
        let line = text.components(separatedBy: "PolygonVertexIndex: *12 {\n\t\t\ta: ")[1].components(separatedBy: "\n")[0]
        let idx = line.split(separator: ",").compactMap { Int($0) }
        XCTAssertEqual(idx.count, 12)
        for (i, v) in idx.enumerated() { XCTAssertEqual(v < 0, i % 3 == 2) }
    }

    func testSlicerListIsStable() {
        XCTAssertEqual(Slicer.allCases.count, 4)
        XCTAssertNil(Slicer.preferred("nonexistent").flatMap { Slicer.installed.contains($0) ? nil : $0 })
    }
}

private final class ModelCounter: NSObject, XMLParserDelegate {
    var vertices: [[Float]] = []
    var triangles = 0
    func parser(_ parser: XMLParser, didStartElement name: String, namespaceURI: String?,
                qualifiedName: String?, attributes a: [String: String]) {
        if name == "vertex", let x = Float(a["x"] ?? ""), let y = Float(a["y"] ?? ""), let z = Float(a["z"] ?? "") {
            vertices.append([x, y, z])
        }
        if name == "triangle" { triangles += 1 }
    }
}
