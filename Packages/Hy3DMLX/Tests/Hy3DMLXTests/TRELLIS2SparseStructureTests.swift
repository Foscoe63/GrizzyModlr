import XCTest
import MLX
@testable import Hy3DMLX

/// Parity with scripts/trellis2/make_ssdec_sampler_fixture.py: the real structure decoder weights, and the sampler
/// schedule / guidance math driven by a toy model.
final class TRELLIS2SparseStructureTests: XCTestCase {
    private struct Fixture: Decodable {
        let z, out, noise, sampled: [Float]
        let steps: Int
        let rescale_t, strength, rescale: Float
        let interval: [Float]
    }

    private let root = ProcessInfo.processInfo.environment["TRELLIS2_DIR"] ?? "/Volumes/Models/Modelr/trellis2"

    private func load() throws -> Fixture {
        let fx = URL(fileURLWithPath: root + "/fixtures/ssdec_sampler_fixture.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: fx.path), "TRELLIS.2 fixtures not installed")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fx))
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> (Double, Float) {
        var d = 0.0, na = 0.0, nb = 0.0
        for (x, y) in zip(a, b) { d += Double(x * y); na += Double(x * x); nb += Double(y * y) }
        return (d / (na.squareRoot() * nb.squareRoot()), zip(a, b).map { abs($0 - $1) }.max() ?? 0)
    }

    func testStructureDecoderMatchesTorch() throws {
        let f = try load()
        let w = URL(fileURLWithPath: root + "/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: w.path), "weights not installed")
        let dec = TRELLIS2SparseStructureDecoder(weights: try loadArrays(url: w))
        // fixture z is [1, 8, 16, 16, 16] channels-first
        let z = MLXArray(f.z, [1, 8, 16, 16, 16]).transposed(0, 2, 3, 4, 1)
        let out = dec.decode(z)
        eval(out)
        XCTAssertEqual(out.shape, [1, 64, 64, 64, 1])
        let got = out.asArray(Float.self)       // channel dim is 1, so NDHWC order == NCDHW order
        let (c, m) = cosine(got, f.out)
        print("ssdec cosine \(c) maxAbsDiff \(m)")
        XCTAssertGreaterThan(c, 0.999)
    }

    func testSamplerMatchesTorch() throws {
        let f = try load()
        var p = FlowEulerSampler.Params()
        p.steps = f.steps; p.rescaleT = f.rescale_t; p.guidanceStrength = f.strength
        p.guidanceRescale = f.rescale; p.guidanceInterval = (f.interval[0], f.interval[1])
        let sampler = FlowEulerSampler(params: p)
        let out = sampler.sample(noise: MLXArray(f.noise, [1, 8, 4, 4, 4])) { x, t1000, cond in
            tanh(0.7 * x + (cond ? 0.3 : -0.2) + 0.5 * t1000 / 1000)
        }
        let (c, m) = cosine(out.asArray(Float.self), f.sampled)
        print("sampler cosine \(c) maxAbsDiff \(m)")
        XCTAssertGreaterThan(c, 0.99999)
        XCTAssertLessThan(m, 1e-4)
    }

    func testOccupiedCoordsOrder() {
        var v = [Float](repeating: -1, count: 4 * 4 * 4)
        v[(1 * 4 + 2) * 4 + 3] = 1        // x=1, y=2, z=3
        let c = TRELLIS2SparseStructureDecoder.occupiedCoords(MLXArray(v, [1, 4, 4, 4, 1]))
        XCTAssertEqual(c.asArray(Int32.self), [0, 1, 2, 3])
    }
}
