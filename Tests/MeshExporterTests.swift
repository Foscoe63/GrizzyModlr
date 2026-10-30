import XCTest
import SceneKit
// MeshExporter lives in Sources/Core, compiled into this test bundle — same module.

/// §4.8 export flow: all four formats end-to-end against the app's binary
/// .mesh/.tmesh layouts, GLB structure re-parsed per glTF 2.0 (including the
/// PBR-ready two-texture baseColor + metallicRoughness form).
final class MeshExporterTests: XCTestCase {

    // MARK: - fixtures

    /// Tiny valid 1×1 PNG (what matters to the exporter is the bytes, not pixels).
    static let tinyPNG = Data(base64Encoded:
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")!
    /// A second, distinguishable PNG blob standing in for the metallic-roughness map.
    static let tinyMRPNG = tinyPNG + Data([0x4D, 0x52])   // trailing junk is fine for embedding

    /// A tetrahedron: 4 verts, 4 faces — closed, minimal, easy to assert against.
    static let verts: [Float] = [
        0, 0, 0,   1, 0, 0,   0, 1, 0,   0, 0, 1,
    ]
    static let normals: [Float] = [
        -1, -1, -1,   1, 0, 0,   0, 1, 0,   0, 0, 1,
    ]
    static let uvs: [Float] = [
        0, 0,   1, 0,   0, 1,   1, 1,
    ]
    static let faces: [UInt32] = [
        0, 2, 1,   0, 1, 3,   0, 3, 2,   1, 2, 3,
    ]

    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mex-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    /// Write the app's binary `.mesh` (or `.tmesh` when `withUVs`).
    private func writeMeshFile(named name: String, withUVs: Bool) throws -> URL {
        var data = Data()
        func le32(_ v: Int32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func leF(_ v: Float) { var x = v.bitPattern.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func leU(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        le32(4); le32(4)
        Self.verts.forEach(leF)
        Self.normals.forEach(leF)
        if withUVs { Self.uvs.forEach(leF) }
        Self.faces.forEach(leU)
        let url = dir.appendingPathComponent(name)
        try data.write(to: url)
        return url
    }

    private func writePNG(named name: String, _ bytes: Data) throws -> URL {
        let url = dir.appendingPathComponent(name)
        try bytes.write(to: url)
        return url
    }

    // MARK: - GLB re-parse helper (glTF 2.0 binary container)

    struct ParsedGLB {
        let json: [String: Any]
        let bin: Data
        let totalLength: Int
    }

    private func parseGLB(_ url: URL) throws -> ParsedGLB {
        let data = try Data(contentsOf: url)
        func u32(_ off: Int) -> UInt32 { data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: off, as: UInt32.self) } }
        XCTAssertEqual(u32(0), 0x4654_6C67, "GLB magic 'glTF'")
        XCTAssertEqual(u32(4), 2, "glTF container version")
        let total = Int(u32(8))
        XCTAssertEqual(total, data.count, "declared length matches the file")

        let jsonLen = Int(u32(12))
        XCTAssertEqual(u32(16), 0x4E4F_534A, "first chunk is JSON")
        XCTAssertEqual(jsonLen % 4, 0, "JSON chunk 4-byte aligned")
        let jsonData = data.subdata(in: 20 ..< 20 + jsonLen)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: jsonData) as? [String: Any])

        let binHeader = 20 + jsonLen
        let binLen = Int(u32(binHeader))
        XCTAssertEqual(u32(binHeader + 4), 0x004E_4942, "second chunk is BIN\\0")
        XCTAssertEqual(binLen % 4, 0, "BIN chunk 4-byte aligned")
        let bin = data.subdata(in: binHeader + 8 ..< binHeader + 8 + binLen)
        return ParsedGLB(json: json, bin: bin, totalLength: total)
    }

    // MARK: - STL

    func testSTLStructure() throws {
        let mesh = try writeMeshFile(named: "t.mesh", withUVs: false)
        let dest = dir.appendingPathComponent("t.stl")
        try MeshExporter.export(meshURL: mesh, texture: nil, format: .stl, to: dest)
        let data = try Data(contentsOf: dest)
        XCTAssertEqual(data.count, 84 + 4 * 50, "80B header + count + 4 × 50B facets")
        let count = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 80, as: UInt32.self) }
        XCTAssertEqual(count, 4)
    }

    // MARK: - PLY

    func testPLYStructure() throws {
        let mesh = try writeMeshFile(named: "t.mesh", withUVs: false)
        let dest = dir.appendingPathComponent("t.ply")
        try MeshExporter.export(meshURL: mesh, texture: nil, format: .ply, to: dest)
        let data = try Data(contentsOf: dest)
        let header = String(decoding: data.prefix(300), as: UTF8.self)
        XCTAssertTrue(header.hasPrefix("ply"))
        XCTAssertTrue(header.contains("format binary_little_endian 1.0"))
        XCTAssertTrue(header.contains("element vertex 4"))
        XCTAssertTrue(header.contains("element face 4"))
        // body = 4 verts × 6 floats + 4 faces × (1 + 12) bytes
        let bodyStart = try XCTUnwrap(data.range(of: Data("end_header\n".utf8))).upperBound
        XCTAssertEqual(data.count - bodyStart, 4 * 24 + 4 * 13)
    }

    // MARK: - OBJ

    func testOBJWithTextureWritesMTLAndPNG() throws {
        let mesh = try writeMeshFile(named: "t.tmesh", withUVs: true)
        let tex = try writePNG(named: "t_texture.png", Self.tinyPNG)
        let dest = dir.appendingPathComponent("model.obj")
        try MeshExporter.export(meshURL: mesh, texture: tex, format: .obj, to: dest)

        let obj = try String(contentsOf: dest, encoding: .utf8)
        XCTAssertTrue(obj.contains("mtllib model.mtl"))
        XCTAssertEqual(obj.components(separatedBy: "\nv ").count - 1 + (obj.hasPrefix("v ") ? 1 : 0), 4)
        XCTAssertEqual(obj.components(separatedBy: "\nvt ").count - 1, 4)
        XCTAssertEqual(obj.components(separatedBy: "\nvn ").count - 1, 4)
        XCTAssertEqual(obj.components(separatedBy: "\nf ").count - 1, 4)
        // V flipped for OBJ's bottom-left origin: our (0,0) → vt 0 1.
        XCTAssertTrue(obj.contains("vt 0.0 1.0"))

        let mtlURL = dir.appendingPathComponent("model.mtl")
        let mtl = try String(contentsOf: mtlURL, encoding: .utf8)
        XCTAssertTrue(mtl.contains("map_Kd model.png"))
        XCTAssertEqual(try Data(contentsOf: dir.appendingPathComponent("model.png")), Self.tinyPNG)
    }

    /// A PBR export to OBJ carries albedo only (no MTL slot for metallic-roughness),
    /// and the MTL notes the dropped map so it isn't a silent loss (§4.8).
    func testOBJPBRWritesAlbedoOnlyWithNote() throws {
        let mesh = try writeMeshFile(named: "t.tmesh", withUVs: true)
        let albedo = try writePNG(named: "albedo.png", Self.tinyPNG)
        let mr = try writePNG(named: "mr.png", Self.tinyMRPNG)
        let dest = dir.appendingPathComponent("pbr.obj")
        try MeshExporter.export(meshURL: mesh, texture: albedo, metallicRoughness: mr,
                                format: .obj, to: dest)

        let mtl = try String(contentsOf: dir.appendingPathComponent("pbr.mtl"), encoding: .utf8)
        XCTAssertTrue(mtl.contains("map_Kd pbr.png"))
        XCTAssertFalse(mtl.contains("mr"), "OBJ/MTL has no metallic-roughness slot")
        XCTAssertTrue(mtl.lowercased().contains("albedo"), "MTL should note it's albedo-only")
        // Only the albedo PNG is written next to the OBJ — no MR sidecar.
        XCTAssertEqual(try Data(contentsOf: dir.appendingPathComponent("pbr.png")), Self.tinyPNG)
    }

    func testOBJUntexturedHasNoMaterial() throws {
        let mesh = try writeMeshFile(named: "t.mesh", withUVs: false)
        let dest = dir.appendingPathComponent("plain.obj")
        try MeshExporter.export(meshURL: mesh, texture: nil, format: .obj, to: dest)
        let obj = try String(contentsOf: dest, encoding: .utf8)
        XCTAssertFalse(obj.contains("mtllib"))
        XCTAssertFalse(obj.contains("vt "))
        XCTAssertTrue(obj.contains("f 1//1 3//3 2//2"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("plain.mtl").path))
    }

    // MARK: - GLB

    func testGLBUntexturedStructure() throws {
        let mesh = try writeMeshFile(named: "t.mesh", withUVs: false)
        let dest = dir.appendingPathComponent("t.glb")
        try MeshExporter.export(meshURL: mesh, texture: nil, format: .glb, to: dest)
        let glb = try parseGLB(dest)

        let accessors = try XCTUnwrap(glb.json["accessors"] as? [[String: Any]])
        XCTAssertEqual(accessors.count, 3)                 // POSITION, NORMAL, indices
        XCTAssertNil(glb.json["materials"])
        XCTAssertNil(glb.json["images"])
        let meshes = try XCTUnwrap(glb.json["meshes"] as? [[String: Any]])
        let prim = try XCTUnwrap((meshes[0]["primitives"] as? [[String: Any]])?.first)
        let attrs = try XCTUnwrap(prim["attributes"] as? [String: Any])
        XCTAssertNotNil(attrs["POSITION"]); XCTAssertNotNil(attrs["NORMAL"])
        XCTAssertNil(attrs["TEXCOORD_0"])
        // POSITION accessor count and min/max present.
        XCTAssertEqual(accessors[0]["count"] as? Int, 4)
        XCTAssertNotNil(accessors[0]["min"]); XCTAssertNotNil(accessors[0]["max"])
        // BIN holds verts + normals + indices (+ padding).
        XCTAssertGreaterThanOrEqual(glb.bin.count, 4*12 + 4*12 + 12*4)
    }

    func testGLBTexturedStructure() throws {
        let mesh = try writeMeshFile(named: "t.tmesh", withUVs: true)
        let tex = try writePNG(named: "t_texture.png", Self.tinyPNG)
        let dest = dir.appendingPathComponent("t.glb")
        try MeshExporter.export(meshURL: mesh, texture: tex, format: .glb, to: dest)
        let glb = try parseGLB(dest)

        let images = try XCTUnwrap(glb.json["images"] as? [[String: Any]])
        XCTAssertEqual(images.count, 1)
        let materials = try XCTUnwrap(glb.json["materials"] as? [[String: Any]])
        let pbr = try XCTUnwrap(materials[0]["pbrMetallicRoughness"] as? [String: Any])
        XCTAssertNotNil(pbr["baseColorTexture"])
        XCTAssertNil(pbr["metallicRoughnessTexture"])
        XCTAssertEqual(pbr["metallicFactor"] as? Double, 0.0)   // matte without an MR map

        // The embedded PNG bytes round-trip exactly.
        let bufferViews = try XCTUnwrap(glb.json["bufferViews"] as? [[String: Any]])
        let imgBV = try XCTUnwrap(images[0]["bufferView"] as? Int)
        let off = try XCTUnwrap(bufferViews[imgBV]["byteOffset"] as? Int)
        let len = try XCTUnwrap(bufferViews[imgBV]["byteLength"] as? Int)
        XCTAssertEqual(glb.bin.subdata(in: off ..< off + len), Self.tinyPNG)

        let meshes = try XCTUnwrap(glb.json["meshes"] as? [[String: Any]])
        let prim = try XCTUnwrap((meshes[0]["primitives"] as? [[String: Any]])?.first)
        let attrs = try XCTUnwrap(prim["attributes"] as? [String: Any])
        XCTAssertNotNil(attrs["TEXCOORD_0"])
    }

    /// The PBR-ready two-texture form: baseColor + metallicRoughness per glTF 2.0.
    func testGLBWithMetallicRoughnessTwoTextureStructure() throws {
        let mesh = try writeMeshFile(named: "t.tmesh", withUVs: true)
        let albedo = try writePNG(named: "albedo.png", Self.tinyPNG)
        let mr = try writePNG(named: "mr.png", Self.tinyMRPNG)
        let dest = dir.appendingPathComponent("pbr.glb")
        try MeshExporter.export(meshURL: mesh, texture: albedo, metallicRoughness: mr,
                                format: .glb, to: dest)
        let glb = try parseGLB(dest)

        let images = try XCTUnwrap(glb.json["images"] as? [[String: Any]])
        let textures = try XCTUnwrap(glb.json["textures"] as? [[String: Any]])
        XCTAssertEqual(images.count, 2)
        XCTAssertEqual(textures.count, 2)
        XCTAssertEqual(textures[0]["source"] as? Int, 0)
        XCTAssertEqual(textures[1]["source"] as? Int, 1)

        let materials = try XCTUnwrap(glb.json["materials"] as? [[String: Any]])
        let pbr = try XCTUnwrap(materials[0]["pbrMetallicRoughness"] as? [String: Any])
        XCTAssertEqual((pbr["baseColorTexture"] as? [String: Any])?["index"] as? Int, 0)
        XCTAssertEqual((pbr["metallicRoughnessTexture"] as? [String: Any])?["index"] as? Int, 1)
        XCTAssertEqual(pbr["metallicFactor"] as? Double, 1.0)   // map is authoritative
        XCTAssertEqual(pbr["roughnessFactor"] as? Double, 1.0)

        // Both PNGs land intact at their (4-aligned) bufferView offsets.
        let bufferViews = try XCTUnwrap(glb.json["bufferViews"] as? [[String: Any]])
        for (img, expected) in zip(images, [Self.tinyPNG, Self.tinyMRPNG]) {
            let bv = try XCTUnwrap(img["bufferView"] as? Int)
            let off = try XCTUnwrap(bufferViews[bv]["byteOffset"] as? Int)
            let len = try XCTUnwrap(bufferViews[bv]["byteLength"] as? Int)
            XCTAssertEqual(glb.bin.subdata(in: off ..< off + len), expected)
        }
        // MR bufferView is 4-aligned (PNG lengths aren't multiples of 4).
        let mrBV = try XCTUnwrap(images[1]["bufferView"] as? Int)
        XCTAssertEqual(try XCTUnwrap(bufferViews[mrBV]["byteOffset"] as? Int) % 4, 0)
    }

    // MARK: - USDZ

    func testUSDZExportsAndReloads() throws {
        let painted = try writeMeshFile(named: "u.tmesh", withUVs: true)
        let tex = try writePNG(named: "u_texture.png", Self.tinyPNG)
        let dest = dir.appendingPathComponent("u.usdz")
        try MeshExporter.export(meshURL: painted, texture: tex, format: .usdz, to: dest)
        let scene = try SCNScene(url: dest, options: nil)
        let node = try XCTUnwrap(scene.rootNode.childNodes(passingTest: { n, _ in n.geometry != nil }).first)
        XCTAssertGreaterThan(node.geometry?.elements.first?.primitiveCount ?? 0, 0)
    }

    /// End-to-end sweep: every format exports from real .mesh/.tmesh files and the
    /// results re-parse (§4.8 verification for the current pipeline outputs).
    func testAllFormatsEndToEnd() throws {
        let shape = try writeMeshFile(named: "gen.mesh", withUVs: false)
        let painted = try writeMeshFile(named: "painted.tmesh", withUVs: true)
        let tex = try writePNG(named: "painted_texture.png", Self.tinyPNG)

        for format in MeshExportFormat.allCases where format != .glbRigged {   // needs a humanoid; see RigTests
            // Untextured shape export.
            let d1 = dir.appendingPathComponent("shape-e2e.\(format.ext)")
            try MeshExporter.export(meshURL: shape, texture: nil, format: format, to: d1)
            XCTAssertGreaterThan(try Data(contentsOf: d1).count, 8, "\(format.ext) shape export empty")

            // Painted (textured) export.
            let d2 = dir.appendingPathComponent("paint-e2e.\(format.ext)")
            try MeshExporter.export(meshURL: painted, texture: tex, format: format, to: d2)
            XCTAssertGreaterThan(try Data(contentsOf: d2).count, 8, "\(format.ext) paint export empty")

            if format == .glb {
                _ = try parseGLB(d1)
                let glb = try parseGLB(d2)
                XCTAssertNotNil(glb.json["materials"])
            }
        }
    }
}
