import XCTest
import MLX
@testable import Hy3DMLX

/// Parity with scripts/trellis2/make_pixal3d_fixture.py: the projection sampler and the structure DiT with
/// per-block proj_linear, on the real Pixal3D checkpoint.
final class Pixal3DTests: XCTestCase {
    private struct Fixture: Decodable {
        let fov, dist, scale, t: Float
        let pin: Int
        let fmap, samp, x, cond, proj, out: [Float]
        let coords: [Int32]
    }

    private let root = ProcessInfo.processInfo.environment["PIXAL3D_DIR"] ?? "/Volumes/Models/Modelr/pixal3d"

    private func load() throws -> Fixture {
        let fx = URL(fileURLWithPath: root + "/fixtures/pixal3d_fixture.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: fx.path), "Pixal3D fixture not installed")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fx))
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> (Double, Float) {
        var d = 0.0, na = 0.0, nb = 0.0
        for (x, y) in zip(a, b) { d += Double(x * y); na += Double(x * x); nb += Double(y * y) }
        return (d / (na.squareRoot() * nb.squareRoot()), zip(a, b).map { abs($0 - $1) }.max() ?? 0)
    }

    func testProjectionSamplerMatchesTorch() throws {
        let f = try load()
        let s = Pixal3DProjection.sample(features: MLXArray(f.fmap, [32, 32, 16]), grid: 16, imageSize: 512,
                                         fov: f.fov, distance: f.dist, meshScale: f.scale)
        eval(s)
        let (c, m) = cosine(s.asArray(Float.self), f.samp)
        print("pixal3d projection cosine \(c) maxAbsDiff \(m)")
        XCTAssertGreaterThan(c, 0.99999)
        XCTAssertLessThan(m, 1e-3)
    }

    func testStructureDiTWithProjMatchesTorch() throws {
        let f = try load()
        let w = URL(fileURLWithPath: root + "/Pixal3D/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: w.path), "Pixal3D weights not installed")
        let dit = TRELLIS2FlowDiT(weights: try loadArrays(url: w))
        let out = dit(x: MLXArray(f.x, [4096, 8]), coords: MLXArray(f.coords, [4096, 3]), t: f.t,
                      cond: MLXArray(f.cond, [5, 1024]), proj: MLXArray(f.proj, [4096, f.pin]))
        eval(out)
        let (c, m) = cosine(out.asArray(Float.self), f.out)
        print("pixal3d ss DiT cosine \(c) maxAbsDiff \(m)")
        XCTAssertGreaterThan(c, 0.999)
    }
}
