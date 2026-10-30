import Foundation
import simd
import MLX
import HunyuanPaintMLX

/// In-process MLX paint/texture generation — the native-Swift replacement for the
/// Python paint worker. Holds a resident `PaintPipeline` (weights loaded once) and
/// runs the mesh+image → textured-mesh pipeline off-main on a serial queue.
final class PaintEngine {
    private let queue = DispatchQueue(label: "com.zimeng.Modelr.paint", qos: .userInitiated)
    private var cachedKey: String?
    private var cachedPipe: PaintPipeline?

    final class Run: CancellableRun {
        private let lock = NSLock()
        private var _cancelled = false
        var cancelled: Bool { lock.lock(); defer { lock.unlock() }; return _cancelled }
        func cancel() { lock.lock(); _cancelled = true; lock.unlock() }
    }

    /// 2.0 RGB (Color) paint. `viewsDir` is where streamed preview grids are written
    /// (the caller cleans them up). `seed` is the resolved concrete seed for this run.
    @discardableResult
    func paint(meshURL: URL, imageURL: URL, output: URL, texture: URL, weightsRoot: URL,
               res: Int, steps: Int, tex: Int, superres: Bool, seed: UInt64, viewsDir: URL,
               onProgress: @escaping (String, Double?) -> Void,
               onViews: @escaping (URL) -> Void,
               onFinish: @escaping (GenerationOutcome) -> Void) -> Run {
        let run = Run()
        queue.async { [weak self] in
            guard let self else { return }
            if run.cancelled { onFinish(.failure("Cancelled")); return }
            guard let loaded = Self.loadShapeMesh(meshURL) else {
                onFinish(.failure("Couldn't read the shape mesh.")); return
            }
            let pipe = self.residentPipe(weightsRoot: weightsRoot, res: res, steps: steps,
                                         tex: tex, superres: superres, onProgress: onProgress)
            if run.cancelled { onFinish(.failure("Cancelled")); return }

            var viewIdx = 0
            let result: PaintResult?
            do {
                result = try pipe.paintRGB(
                    mesh: loaded, imagePath: imageURL.path, guidance: 2.0, seed: seed,
                    onProgress: { stage, frac in onProgress(stage, Double(frac)) },
                    isCancelled: { run.cancelled },
                    onViews: { data in
                        let u = viewsDir.appendingPathComponent("paint_views_\(viewIdx).png")
                        viewIdx += 1
                        try? data.write(to: u)
                        onViews(u)
                    })
            } catch {
                self.cachedKey = nil; self.cachedPipe = nil      // failed load → retry fresh next time
                onFinish(.failure("Couldn't load the paint model: \(error.localizedDescription)"))
                return
            }

            if run.cancelled { onFinish(.failure("Cancelled")); return }
            guard let result else { onFinish(.failure("Couldn't unwrap or paint this mesh.")); return }
            do {
                try PaintMeshWriter.write(vertices: result.vertices, faces: result.faces,
                                          uvs: result.uvs, to: output)
                try result.albedoPNG.write(to: texture)
                if run.cancelled {                                // closed the post-write cancel window
                    try? FileManager.default.removeItem(at: output)
                    try? FileManager.default.removeItem(at: texture)
                    onFinish(.failure("Cancelled")); return
                }
                onFinish(.success)
            } catch {
                onFinish(.failure("Couldn't write the painted mesh: \(error.localizedDescription)"))
            }
        }
        return run
    }

    /// 2.1 PBR (Large) paint — same contract as `paint`, plus a metallic-roughness
    /// map written to `mrTexture` (G = roughness, B = metallic). Guidance is fixed at
    /// the 2.1 value (3.0); `seed` is the resolved concrete seed. The stage callbacks
    /// use the same strings the RGB path emits, so `mapPaintStage` maps them into the
    /// same reducer stages ("Baking textures" still matches the "Baking" prefix).
    @discardableResult
    func paintPBR(meshURL: URL, imageURL: URL, output: URL, texture: URL, mrTexture: URL,
                  weightsRoot: URL, res: Int, steps: Int, tex: Int, superres: Bool, seed: UInt64,
                  viewsDir: URL,
                  onProgress: @escaping (String, Double?) -> Void,
                  onViews: @escaping (URL) -> Void,
                  onFinish: @escaping (GenerationOutcome) -> Void) -> Run {
        let run = Run()
        queue.async { [weak self] in
            guard let self else { return }
            if run.cancelled { onFinish(.failure("Cancelled")); return }
            guard let loaded = Self.loadShapeMesh(meshURL) else {
                onFinish(.failure("Couldn't read the shape mesh.")); return
            }
            let pipe = self.residentPipe(weightsRoot: weightsRoot, res: res, steps: steps,
                                         tex: tex, superres: superres, onProgress: onProgress)
            if run.cancelled { onFinish(.failure("Cancelled")); return }

            var viewIdx = 0
            let result: PBRPaintResult?
            do {
                result = try pipe.paintPBR(
                    mesh: loaded, imagePath: imageURL.path, guidance: 3.0, seed: seed,
                    onProgress: { stage, frac in onProgress(stage, Double(frac)) },
                    isCancelled: { run.cancelled },
                    onViews: { data in
                        let u = viewsDir.appendingPathComponent("paint_views_\(viewIdx).png")
                        viewIdx += 1
                        try? data.write(to: u)
                        onViews(u)
                    })
            } catch {
                self.cachedKey = nil; self.cachedPipe = nil      // failed load → retry fresh next time
                onFinish(.failure("Couldn't load the paint model: \(error.localizedDescription)"))
                return
            }

            if run.cancelled { onFinish(.failure("Cancelled")); return }
            guard let result else { onFinish(.failure("Couldn't unwrap or paint this mesh.")); return }
            do {
                try PaintMeshWriter.write(vertices: result.vertices, faces: result.faces,
                                          uvs: result.uvs, to: output)
                try result.albedoPNG.write(to: texture)
                try result.metallicRoughnessPNG.write(to: mrTexture)
                if run.cancelled {                                // closed the post-write cancel window
                    for u in [output, texture, mrTexture] { try? FileManager.default.removeItem(at: u) }
                    onFinish(.failure("Cancelled")); return
                }
                onFinish(.success)
            } catch {
                onFinish(.failure("Couldn't write the painted mesh: \(error.localizedDescription)"))
            }
        }
        return run
    }

    /// Fetch (or build) the resident pipeline for `weightsRoot`. Keys only on what
    /// changes the LOADED WEIGHTS (root + whether super-res loads); res/steps/tex are
    /// per-run knobs set on the cached pipe so changing quality — or switching between
    /// the Color and PBR checkpoints (different roots) — doesn't needlessly reload GBs.
    /// Must be called on `queue`.
    private func residentPipe(weightsRoot: URL, res: Int, steps: Int, tex: Int, superres: Bool,
                              onProgress: (String, Double?) -> Void) -> PaintPipeline {
        let key = "\(weightsRoot.path)#\(superres)"
        let pipe: PaintPipeline
        if cachedKey == key, let p = cachedPipe {
            pipe = p
        } else {
            onProgress("Loading paint model…", nil)
            cachedKey = nil
            cachedPipe = nil
            MLX.GPU.clearCache()
            pipe = PaintPipeline(weightsRoot: weightsRoot.path, res: res, steps: steps,
                                 tex: tex, superRes: superres)
            cachedKey = key
            cachedPipe = pipe
        }
        pipe.res = res; pipe.steps = steps; pipe.tex = tex
        return pipe
    }

    /// Drop the resident paint pipeline and free GPU buffers (called when the shape
    /// engine starts, so both model sets aren't resident at once).
    func evict() {
        queue.async {
            self.cachedKey = nil
            self.cachedPipe = nil
            MLX.GPU.clearCache()
        }
    }

    /// Eviction the EngineArbiter can await: returns only after the resident
    /// pipeline has actually been dropped on the engine queue.
    func evictAndWait() async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async {
                self.cachedKey = nil
                self.cachedPipe = nil
                MLX.GPU.clearCache()
                cont.resume()
            }
        }
    }

    /// Parse GrizzyModlr's `.mesh` (verts + normals + faces) into the paint package's
    /// `LoadedMesh` (positions + faces only).
    private static func loadShapeMesh(_ url: URL) -> LoadedMesh? {
        guard let d = try? Data(contentsOf: url), d.count >= 8 else { return nil }
        let n = Int(d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: Int32.self) })
        let m = Int(d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: Int32.self) })
        guard n > 0, m > 0 else { return nil }
        let vBytes = n * 12, nBytes = n * 12, fBytes = m * 12
        guard d.count >= 8 + vBytes + nBytes + fBytes else { return nil }
        var verts = [Float](repeating: 0, count: n * 3)
        var faces = [UInt32](repeating: 0, count: m * 3)
        d.withUnsafeBytes { raw in
            for i in 0..<(n * 3) { verts[i] = raw.loadUnaligned(fromByteOffset: 8 + i * 4, as: Float.self) }
            let foff = 8 + vBytes + nBytes
            for i in 0..<(m * 3) { faces[i] = raw.loadUnaligned(fromByteOffset: foff + i * 4, as: UInt32.self) }
        }
        return LoadedMesh(vertices: verts, faces: faces)
    }
}

/// Serializes a `PaintResult` into GrizzyModlr's `.tmesh`
/// (`[i32 nV][i32 nF][f32 verts][f32 normals][f32 uvs][i32 faces]`), computing
/// area-weighted vertex normals.
enum PaintMeshWriter {
    /// Serialize unwrapped geometry + UVs (shared by the RGB and PBR paint paths —
    /// both produce the same geometry contract, differing only in their baked maps).
    static func write(vertices verts: [Float], faces: [UInt32], uvs: [Float], to url: URL) throws {
        let n = verts.count / 3, m = faces.count / 3
        guard n > 0, m > 0, uvs.count == n * 2 else {
            throw NSError(domain: "PaintMeshWriter", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "malformed paint result"])
        }

        var normals = [Float](repeating: 0, count: n * 3)
        func vert(_ i: Int) -> SIMD3<Float> { SIMD3(verts[i*3], verts[i*3+1], verts[i*3+2]) }
        for f in 0..<m {
            let i0 = Int(faces[f*3]), i1 = Int(faces[f*3+1]), i2 = Int(faces[f*3+2])
            let fn = simd_cross(vert(i1) - vert(i0), vert(i2) - vert(i0))   // area-weighted
            for i in [i0, i1, i2] { normals[i*3]+=fn.x; normals[i*3+1]+=fn.y; normals[i*3+2]+=fn.z }
        }
        for i in 0..<n {
            let v = SIMD3<Float>(normals[i*3], normals[i*3+1], normals[i*3+2])
            let l = simd_length(v)
            let u = l > 1e-12 ? v / l : SIMD3<Float>(0, 0, 1)
            normals[i*3] = u.x; normals[i*3+1] = u.y; normals[i*3+2] = u.z
        }

        var data = Data()
        data.reserveCapacity(8 + n * 12 + n * 12 + n * 8 + m * 12)
        appendI32(&data, Int32(n)); appendI32(&data, Int32(m))
        for i in 0..<(n * 3) { appendF(&data, verts[i]) }
        for i in 0..<(n * 3) { appendF(&data, normals[i]) }
        for i in 0..<(n * 2) { appendF(&data, uvs[i]) }
        for i in 0..<(m * 3) { appendU32(&data, faces[i]) }
        try data.write(to: url)
    }

    private static func appendF(_ d: inout Data, _ v: Float) {
        var x = v.bitPattern.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) }
    }
    private static func appendI32(_ d: inout Data, _ v: Int32) {
        var x = v.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) }
    }
    private static func appendU32(_ d: inout Data, _ v: UInt32) {
        var x = v.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) }
    }
}
