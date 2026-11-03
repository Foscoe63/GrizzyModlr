import XCTest
import MLX
@testable import Hy3DMLX

/// Parity of the Swift shape decoder against scripts/trellis2/make_decoder_fixture.py (real checkpoint,
/// PyTorch CPU fp32). Needs the weights and the fixture under /Volumes/Models/Modelr/trellis2 (not in git).
final class TRELLIS2ShapeDecoderTests: XCTestCase {
    private struct Fixture: Decodable {
        let latent_coords: [Int32]
        let latent: [Float]
        let n_latent: Int
        let final_coords: [Int32]
        let final: [Float]
        let n_final: Int
        let stage_counts: [Int]
        let sub0: [Float]
    }

    private let root = ProcessInfo.processInfo.environment["TRELLIS2_DIR"] ?? "/Volumes/Models/Modelr/trellis2"

    private func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var d = 0.0, na = 0.0, nb = 0.0
        for (x, y) in zip(a, b) { d += Double(x * y); na += Double(x * x); nb += Double(y * y) }
        return d / (na.squareRoot() * nb.squareRoot())
    }

    func testDecoderMatchesTorch() throws {
        let ckpt = URL(fileURLWithPath: root + "/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors")
        let fx = URL(fileURLWithPath: root + "/fixtures/decoder_fixture.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: ckpt.path) && FileManager.default.fileExists(atPath: fx.path),
                          "TRELLIS.2 decoder weights/fixture not installed")
        let f = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fx))
        let dec = TRELLIS2ShapeDecoder(weights: try loadArrays(url: ckpt))
        let out = dec.decode(coords: MLXArray(f.latent_coords, [f.n_latent, 4]), latent: MLXArray(f.latent, [f.n_latent, 32]))
        eval(out.feats, out.coords)
        XCTAssertEqual(out.stageCounts, f.stage_counts)
        XCTAssertEqual(out.coords.asArray(Int32.self), f.final_coords)
        let got = out.feats.asArray(Float.self)
        let c = cosine(got, f.final)
        let maxd = zip(got, f.final).map { abs($0 - $1) }.max() ?? 0
        print("decoder cosine \(c) maxAbsDiff \(maxd) voxels \(f.n_final)")
        XCTAssertGreaterThan(c, 0.999)
        XCTAssertEqual(MLX.argMax(out.subdivLogits[0] .> 0).item(Int.self) >= 0, true)
    }
}
