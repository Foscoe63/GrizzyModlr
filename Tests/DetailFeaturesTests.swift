import XCTest
import simd

@MainActor
final class DetailFeaturesTests: XCTestCase {
    func testEstimateOnlyShownAboveDefaultsAndGrowsWithGrid() {
        XCTAssertNil(DetailEstimate.summary(octree: 512))
        let a = DetailEstimate.memoryGB(octree: 1024), b = DetailEstimate.memoryGB(octree: 2048)
        XCTAssertEqual(b / a, 4, accuracy: 1e-9)
        XCTAssertEqual(DetailEstimate.relativeTime(octree: 1024), 4, accuracy: 1e-9)
        XCTAssertTrue(DetailEstimate.summary(octree: 2048)!.contains("GB"))
        XCTAssertTrue(DetailEstimate.summary(octree: 2048)!.contains("triangles"))
    }

    func testMaximumDetailNeverLowersHigherValues() {
        let low = MaximumDetail.shape(model: .small, steps: 10, octree: 256)
        XCTAssertEqual(low.steps, QualityPreset.max.steps(for: .small))
        XCTAssertEqual(low.octree, MaximumDetail.octree)
        let high = MaximumDetail.shape(model: .large, steps: 90, octree: 1536)
        XCTAssertEqual(high.steps, 90)
        XCTAssertEqual(high.octree, 1536)
    }

    func testApplyMaximumDetailSetsShapeAndPaint() {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("md-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ProjectStore(rootDir: root)
        let p = store.newProject()
        store.applyMaximumDetail(for: p.id)
        let q = store.project(p.id)!
        XCTAssertTrue(q.advancedMode); XCTAssertTrue(q.paintAdvanced)
        XCTAssertEqual(q.octree, MaximumDetail.octree)
        XCTAssertEqual(q.paintTex, PaintQuality.max.tex)
        XCTAssertEqual(q.paintFaces, PaintQuality.max.faces)
        XCTAssertEqual(q.quantization, .full)
    }

    func testViewLODPassesSmallMeshesThroughAndReadsHeader() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("lod-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let mesh = dir.appendingPathComponent("gen_x.mesh")
        let positions: [SIMD3<Float>] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        try MeshDecimator.serialize(positions: positions, faces: [[0, 1, 2]]).write(to: mesh)
        XCTAssertEqual(ViewLOD.faceCount(of: mesh), 1)
        XCTAssertFalse(ViewLOD.isSimplified(mesh))
        XCTAssertEqual(ViewLOD.displayURL(for: mesh), mesh)
        XCTAssertEqual(ViewLOD.previewURL(for: mesh).lastPathComponent, "gen_x.view.mesh")
        XCTAssertNil(ViewLOD.faceCount(of: dir.appendingPathComponent("missing.mesh")))
    }

    func testViewLODBuildsAndReusesCachedPreview() throws {
        // UV sphere (radius 1) just over the viewport threshold.
        let lat = 640, lon = 800
        var pos: [SIMD3<Float>] = []
        for i in 0...lat { for j in 0..<lon {
            let t = Float.pi * Float(i) / Float(lat), u = 2 * Float.pi * Float(j) / Float(lon)
            pos.append([sin(t) * cos(u), cos(t), sin(t) * sin(u)])
        } }
        var faces: [SIMD3<UInt32>] = []
        for i in 0..<lat { for j in 0..<lon {
            let a = UInt32(i * lon + j), b = UInt32(i * lon + (j + 1) % lon)
            let c = UInt32((i + 1) * lon + j), d = UInt32((i + 1) * lon + (j + 1) % lon)
            faces.append([a, c, b]); faces.append([b, c, d])
        } }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("lod2-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertGreaterThan(faces.count, ViewLOD.threshold)
        let mesh = dir.appendingPathComponent("gen_big.mesh")
        try MeshDecimator.serialize(positions: pos, faces: faces).write(to: mesh)
        XCTAssertTrue(ViewLOD.isSimplified(mesh))
        let shown = ViewLOD.displayURL(for: mesh)
        XCTAssertEqual(shown, ViewLOD.previewURL(for: mesh))
        let shownFaces = try XCTUnwrap(ViewLOD.faceCount(of: shown))
        XCTAssertLessThan(shownFaces, ViewLOD.threshold)
        XCTAssertGreaterThan(shownFaces, 100_000)
        let parsed = try XCTUnwrap(MeshDecimator.parse(try Data(contentsOf: shown)))
        let meanRadius = parsed.positions.map { simd_length($0) }.reduce(0, +) / Float(parsed.positions.count)
        XCTAssertEqual(meanRadius, 1, accuracy: 0.02)
        let stamp = (try FileManager.default.attributesOfItem(atPath: shown.path))[.modificationDate] as? Date
        XCTAssertEqual(ViewLOD.displayURL(for: mesh), shown)                       // reused, not rebuilt
        XCTAssertEqual((try FileManager.default.attributesOfItem(atPath: shown.path))[.modificationDate] as? Date, stamp)
        XCTAssertEqual(ViewLOD.faceCount(of: mesh), faces.count)                    // original untouched
    }
}
