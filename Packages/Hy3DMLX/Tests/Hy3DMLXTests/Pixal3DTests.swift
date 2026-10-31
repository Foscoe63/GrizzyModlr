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

/// Parity with scripts/trellis2/make_naf_fixture.py on the real NAF weights.
final class NAFTests: XCTestCase {
    private struct Fixture: Decodable { let h, c, hk: Int; let image, feats, out: [Float] }
    private let root = ProcessInfo.processInfo.environment["PIXAL3D_DIR"] ?? "/Volumes/Models/Modelr/pixal3d"

    func testNAFMatchesTorch() throws {
        let fx = URL(fileURLWithPath: root + "/fixtures/naf_fixture.json")
        let wp = URL(fileURLWithPath: root + "/naf/naf.safetensors")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: fx.path) && FileManager.default.fileExists(atPath: wp.path), "NAF not installed")
        let f = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fx))
        let naf = NAFUpsampler(weights: try loadArrays(url: wp))
        // fixture tensors are NCHW; the Swift path is NHWC
        let img = MLXArray(f.image, [1, 3, f.h, f.h]).transposed(0, 2, 3, 1)
        let feats = MLXArray(f.feats, [1, f.c, f.hk, f.hk]).transposed(0, 2, 3, 1)
        let out = naf(image: img, features: feats, outputSize: (f.h, f.h)).transposed(0, 3, 1, 2)
        eval(out)
        var d = 0.0, na = 0.0, nb = 0.0
        let got = out.asArray(Float.self)
        for (a, b) in zip(got, f.out) { d += Double(a * b); na += Double(a * a); nb += Double(b * b) }
        let cos = d / (na.squareRoot() * nb.squareRoot())
        let m = zip(got, f.out).map { abs($0 - $1) }.max() ?? 0
        print("NAF cosine \(cos) maxAbsDiff \(m)")
        XCTAssertGreaterThan(cos, 0.9999)
    }
}
