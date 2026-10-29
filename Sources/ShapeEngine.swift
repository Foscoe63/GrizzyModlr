import Foundation
import CoreGraphics
import ImageIO
import simd
import MLX
import Hy3DMLX

/// A cancellable handle to an in-flight generation.
protocol CancellableRun: AnyObject {
    func cancel()
}

/// Result of a generation (shape or paint).
enum GenerationOutcome {
    case success
    case failure(String)
}

/// In-process MLX shape generation — the native-Swift replacement for the Python
/// shape worker. Loads a `ShapeGenerator` (the vendored Hy3DMLX pipeline) once and
/// keeps it resident across runs; all heavy work runs off the main thread on a
/// serial queue (which also makes the weight cache access race-free).
final class ShapeEngine {
    private let queue = DispatchQueue(label: "com.zimeng.Modelr.shape", qos: .userInitiated)
    private var cachedKey: String?
    private var cachedGen: ShapeGenerator?

    /// Cooperative cancellation token. `ShapeGenerator.generate` polls `cancelled`
    /// per denoise step and at each stage boundary, so a cancel stops work promptly
    /// (no orphaned GPU compute).
    final class Run: CancellableRun {
        private let lock = NSLock()
        private var _cancelled = false
        var cancelled: Bool { lock.lock(); defer { lock.unlock() }; return _cancelled }
        func cancel() { lock.lock(); _cancelled = true; lock.unlock() }
    }

    /// Mirrors the old `GenerationService.run` contract: callbacks may arrive on a
    /// background queue; the caller hops to the main actor.
    @discardableResult
    func generate(imageURL: URL, extraViews: [(angle: ViewAngle, url: URL)] = [],
                  output: URL, weightsURL: URL, quantize: Int,
                  steps: Int, guidance: Float, resolution: Int, seed: UInt64,
                  onProgress: @escaping (String, String?, Double?) -> Void,
                  onPreview: @escaping (URL) -> Void,
                  onFinish: @escaping (GenerationOutcome) -> Void) -> Run {
        let run = Run()
        queue.async { [weak self] in
            guard let self else { return }
            if run.cancelled { onFinish(.failure("Cancelled")); return }
            guard let cg = Self.loadCGImage(imageURL) else {
                onFinish(.failure("Couldn't read the input image.")); return
            }

            // Reuse the resident model when the weights + quantization match.
            let key = "\(weightsURL.path)#\(quantize)"
            let gen: ShapeGenerator
            if self.cachedKey == key, let g = self.cachedGen {
                gen = g
            } else {
                onProgress("Loading model…", nil, nil)
                self.cachedKey = nil
                self.cachedGen = nil                 // free the previous model before loading a new one
                do {
                    gen = try ShapeGenerator(weightsURL: weightsURL, quantize: quantize)
                } catch {
                    onFinish(.failure("Couldn't load the model: \(error.localizedDescription)"))
                    return
                }
                self.cachedKey = key
                self.cachedGen = gen
            }
            if run.cancelled { onFinish(.failure("Cancelled")); return }

            let dir = output.deletingLastPathComponent()
            var previewIdx = 0
            let previewHandler: (Mesh) -> Void = { m in
                let u = dir.appendingPathComponent("preview_\(previewIdx).mesh")
                previewIdx += 1
                try? ShapeMeshWriter.writeMesh(m, to: u)
                onPreview(u)
            }
            let progressHandler: (ShapeGenerator.Progress) -> Void = { p in
                onProgress(p.stage, nil, Double(p.fraction))
            }
            let mesh: Mesh?
            if extraViews.isEmpty {
                mesh = gen.generate(image: cg, steps: steps, guidance: guidance, seed: seed,
                                    resolution: resolution, octree: true,
                                    isCancelled: { run.cancelled },
                                    onPreview: previewHandler, onProgress: progressHandler)
            } else {
                guard gen.isMultiView else {
                    onFinish(.failure("These weights don't support multiple views.")); return
                }
                var views: [(angle: ViewAngle, image: CGImage)] = [(.front, cg)]
                for extra in extraViews {
                    guard let img = Self.loadCGImage(extra.url) else {
                        onFinish(.failure("Couldn't read a reference view.")); return
                    }
                    views.append((extra.angle, img))
                }
                mesh = gen.generate(views: views, steps: steps, guidance: guidance, seed: seed,
                                    resolution: resolution, octree: true,
                                    isCancelled: { run.cancelled },
                                    onPreview: previewHandler, onProgress: progressHandler)
            }

            if run.cancelled { onFinish(.failure("Cancelled")); return }
            guard let mesh else { onFinish(.failure("The model didn't produce a mesh.")); return }
            do {
                try ShapeMeshWriter.writeMesh(mesh, to: output)
                if run.cancelled {                                 // close the post-write cancel window
                    try? FileManager.default.removeItem(at: output)
                    onFinish(.failure("Cancelled")); return
                }
                onFinish(.success)
            } catch {
                onFinish(.failure("Couldn't write the mesh: \(error.localizedDescription)"))
            }
        }
        return run
    }

    /// Drop the resident model and free GPU buffers. Called when the paint engine
    /// starts, so shape + paint weights aren't both resident on a constrained machine.
    func evict() {
        queue.async {
            self.cachedKey = nil
            self.cachedGen = nil
            MLX.GPU.clearCache()
        }
    }

    /// Eviction the EngineArbiter can await: returns only after the resident
    /// model has actually been dropped on the engine queue, making the
    /// evict-then-load handoff deterministic (no cross-queue overlap).
    func evictAndWait() async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async {
                self.cachedKey = nil
                self.cachedGen = nil
                MLX.GPU.clearCache()
                cont.resume()
            }
        }
    }

    private static func loadCGImage(_ url: URL) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }
}

/// Serializes a Hy3DMLX `Mesh` (positions + faces) into Modelr's `.mesh` binary
/// (`[i32 nV][i32 nF][f32 verts][f32 normals][i32 faces]`), computing area-weighted
/// vertex normals the marching-cubes output doesn't carry.
enum ShapeMeshWriter {
    static func writeMesh(_ mesh: Mesh, to url: URL) throws {
        let verts = mesh.vertices
        let faces = mesh.faces
        let n = verts.count, m = faces.count
        guard n > 0, m > 0 else {
            throw NSError(domain: "ShapeMeshWriter", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "empty mesh"])
        }

        var normals = [SIMD3<Float>](repeating: .zero, count: n)
        for f in faces {
            let i0 = Int(f.0), i1 = Int(f.1), i2 = Int(f.2)
            let fn = simd_cross(verts[i1] - verts[i0], verts[i2] - verts[i0])   // area-weighted
            normals[i0] += fn; normals[i1] += fn; normals[i2] += fn
        }

        var data = Data()
        data.reserveCapacity(8 + n * 24 + m * 12)
        appendI32(&data, Int32(n)); appendI32(&data, Int32(m))
        for v in verts { appendF(&data, v.x); appendF(&data, v.y); appendF(&data, v.z) }
        for nrm in normals {
            let len = simd_length(nrm)
            let u = len > 1e-12 ? nrm / len : SIMD3<Float>(0, 0, 1)
            appendF(&data, u.x); appendF(&data, u.y); appendF(&data, u.z)
        }
        for f in faces { appendU32(&data, f.0); appendU32(&data, f.1); appendU32(&data, f.2) }
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
