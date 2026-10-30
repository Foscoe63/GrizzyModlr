import Foundation

/// Rough cost of a shape grid resolution, shown beside the Resolution control. Measured on a sphere
/// through the block-sparse decoder (1024 → about 0.8 GB and 3.5M triangles); real figures have more
/// surface, so the memory figure is padded. Estimates, not guarantees.
enum DetailEstimate {
    static func memoryGB(octree r: Int) -> Double { 2.0 * pow(Double(r) / 1024, 2) }
    /// Decode time relative to the same model at 512 (the near-surface band grows with the square).
    static func relativeTime(octree r: Int) -> Double { pow(Double(r) / 512, 2) }
    static func triangles(octree r: Int) -> Int { Int(3_500_000 * pow(Double(r) / 1024, 2)) }

    /// "≈ 2 GB · ~4× slower than 512 · ~3.5M triangles", or nil where it isn't worth showing.
    static func summary(octree r: Int) -> String? {
        guard r > 512 else { return nil }
        let gb = memoryGB(octree: r)
        let mem = gb < 10 ? String(format: "%.1f GB", gb) : String(format: "%.0f GB", gb)
        let tris = Double(triangles(octree: r)) / 1_000_000
        return "≈ \(mem) · ~\(String(format: "%.0f", relativeTime(octree: r)))× slower than 512 · ~"
            + String(format: tris < 10 ? "%.1fM" : "%.0fM", tris) + " triangles"
    }
}

/// What "Maximum detail" sets: the best settings that stay practical, for shape and paint together.
enum MaximumDetail {
    static let octree = 768

    /// Shape steps/grid for `model`: at least the Max preset's steps and grid, never lowering a value
    /// the user already pushed higher.
    static func shape(model: ShapeModel, steps: Int, octree current: Int) -> (steps: Int, octree: Int) {
        (max(steps, QualityPreset.max.steps(for: model)), max(current, octree))
    }
}

/// Display-only simplification for very dense shape meshes: the viewport shows a vertex-clustered copy
/// (cached beside the mesh) while exports and edits keep using the full file.
enum ViewLOD {
    static let threshold = 1_000_000
    static let budget = 400_000

    /// Face count from a `.mesh` header (8 bytes), without reading the file.
    static func faceCount(of url: URL) -> Int? {
        guard let h = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? h.close() }
        guard let d = try? h.read(upToCount: 8), d.count == 8 else { return nil }
        return Int(d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: Int32.self) })
    }

    static func isSimplified(_ url: URL) -> Bool { (faceCount(of: url) ?? 0) > threshold }

    static func previewURL(for url: URL) -> URL {
        url.deletingPathExtension().appendingPathExtension("view.mesh")
    }

    /// The file the viewport should load: `url` itself when small, else a cached decimated copy
    /// (made on first use, reused while it is newer than the source).
    static func displayURL(for url: URL) -> URL {
        guard let faces = faceCount(of: url), faces > threshold else { return url }
        let preview = previewURL(for: url)
        let fm = FileManager.default
        func modified(_ u: URL) -> Date? { (try? fm.attributesOfItem(atPath: u.path))?[.modificationDate] as? Date }
        if let p = modified(preview), let s = modified(url), p >= s { return preview }
        guard let data = try? Data(contentsOf: url), let mesh = MeshDecimator.parse(data) else { return url }
        let r = VertexClustering.simplify(positions: mesh.positions, faces: mesh.faces, targetFaces: budget)
        guard !r.faces.isEmpty, (try? MeshDecimator.serialize(positions: r.positions, faces: r.faces).write(to: preview)) != nil else { return url }
        return preview
    }
}
