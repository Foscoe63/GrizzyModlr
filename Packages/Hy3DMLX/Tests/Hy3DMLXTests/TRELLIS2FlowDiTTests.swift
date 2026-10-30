import XCTest
import MLX
@testable import Hy3DMLX

/// Parity of the Swift flow DiT with scripts/trellis2/make_dit_fixture.py on the real checkpoints.
final class TRELLIS2FlowDiTTests: XCTestCase {
    private struct Fixture: Decodable {
        let n, cin, n_cond: Int
        let t: Float
        let coords: [Int32]
        let x, cond, out: [Float]
    }

    private let root = ProcessInfo.processInfo.environment["TRELLIS2_DIR"] ?? "/Volumes/Models/Modelr/trellis2"

    private func check(_ name: String, ckpt: String, minCos: Double = 0.999) throws {
        let w = URL(fileURLWithPath: "\(root)/TRELLIS.2-4B/ckpts/\(ckpt).safetensors")
        let fx = URL(fileURLWithPath: "\(root)/fixtures/dit_\(name)_fixture.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: w.path) && FileManager.default.fileExists(atPath: fx.path),
                          "TRELLIS.2 DiT weights/fixture not installed")
        let f = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fx))
        let dit = TRELLIS2FlowDiT(weights: try loadArrays(url: w))
        let out = dit(x: MLXArray(f.x, [f.n, f.cin]), coords: MLXArray(f.coords, [f.n, 3]), t: f.t,
                      cond: MLXArray(f.cond, [f.n_cond, 1024]))
        eval(out)
        let got = out.asArray(Float.self)
        var d = 0.0, na = 0.0, nb = 0.0
        for (a, b) in zip(got, f.out) { d += Double(a * b); na += Double(a * a); nb += Double(b * b) }
        let cos = d / (na.squareRoot() * nb.squareRoot())
        let maxd = zip(got, f.out).map { abs($0 - $1) }.max() ?? 0
        print("DiT \(name) cosine \(cos) maxAbsDiff \(maxd) tokens \(f.n)")
        XCTAssertGreaterThan(cos, minCos)
    }

    func testSparseStructureFlowMatchesTorch() throws { try check("ss", ckpt: "ss_flow_img_dit_1_3B_64_bf16") }
    func testShapeSLatFlowMatchesTorch() throws { try check("slat", ckpt: "slat_flow_img2shape_dit_1_3B_512_bf16") }
}
