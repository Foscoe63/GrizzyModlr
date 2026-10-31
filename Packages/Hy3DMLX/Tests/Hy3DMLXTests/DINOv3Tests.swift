import XCTest
import MLX
@testable import Hy3DMLX

/// Architecture parity with scripts/trellis2/make_dinov3_fixture.py (random weights; the real DINOv3 checkpoint is gated).
final class DINOv3Tests: XCTestCase {
    private struct Fixture: Decodable { let h, w, heads, tokens: Int; let x, out: [Float] }
    private let root = ProcessInfo.processInfo.environment["TRELLIS2_DIR"] ?? "/Volumes/Models/Modelr/trellis2"

    private func check(_ name: String) throws {
        let wURL = URL(fileURLWithPath: "\(root)/fixtures/dinov3_\(name).safetensors")
        let fURL = URL(fileURLWithPath: "\(root)/fixtures/dinov3_\(name).json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: wURL.path) && FileManager.default.fileExists(atPath: fURL.path),
                          "DINOv3 fixtures not installed")
        let f = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fURL))
        let net = DINOv3FeatureExtractor(weights: try loadArrays(url: wURL), heads: f.heads)
        // fixture x is NCHW [1,3,H,W]
        let x = MLXArray(f.x, [1, 3, f.h, f.w]).transposed(0, 2, 3, 1)
        let out = net(x)
        eval(out)
        XCTAssertEqual(out.dim(1), f.tokens)
        let got = out.asArray(Float.self)
        var d = 0.0, na = 0.0, nb = 0.0
        for (a, b) in zip(got, f.out) { d += Double(a * b); na += Double(a * a); nb += Double(b * b) }
        let cos = d / (na.squareRoot() * nb.squareRoot())
        let maxd = zip(got, f.out).map { abs($0 - $1) }.max() ?? 0
        print("DINOv3 \(name) cosine \(cos) maxAbsDiff \(maxd)")
        XCTAssertGreaterThan(cos, 0.9999)
    }

    func testSmallMatchesTorch() throws { try check("small") }
    func testWideNonSquareMatchesTorch() throws { try check("wide") }
}

/// Real `facebook/dinov3-vitl16-pretrain-lvd1689m` weights (gated, installed by the user) against
/// scripts/trellis2/make_dinov3_real_fixture.py on a 512×512 input.
final class DINOv3RealWeightsTests: XCTestCase {
    func testViTLMatchesTorch() throws {
        let root = ProcessInfo.processInfo.environment["TRELLIS2_DIR"] ?? "/Volumes/Models/Modelr/trellis2"
        let w = URL(fileURLWithPath: root + "/dinov3-vitl16/model.safetensors")
        let xURL = URL(fileURLWithPath: root + "/fixtures/dinov3_real_x.f32")
        let oURL = URL(fileURLWithPath: root + "/fixtures/dinov3_real_out.f32")
        try XCTSkipUnless([w, xURL, oURL].allSatisfy { FileManager.default.fileExists(atPath: $0.path) },
                          "DINOv3 ViT-L weights/fixtures not installed")
        func floats(_ u: URL) throws -> [Float] {
            let d = try Data(contentsOf: u)
            return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        let net = DINOv3FeatureExtractor(weights: try loadArrays(url: w), heads: 16)
        XCTAssertEqual(net.layers, 24)
        XCTAssertEqual(net.hidden, 1024)
        let x = MLXArray(try floats(xURL), [1, 3, 512, 512]).transposed(0, 2, 3, 1)
        let out = net(x)
        eval(out)
        XCTAssertEqual(out.shape, [1, 1029, 1024])
        let ref = try floats(oURL)
        let got = out.asArray(Float.self)
        var d = 0.0, na = 0.0, nb = 0.0
        for (a, b) in zip(got, ref) { d += Double(a * b); na += Double(a * a); nb += Double(b * b) }
        let cos = d / (na.squareRoot() * nb.squareRoot())
        let maxd = zip(got, ref).map { abs($0 - $1) }.max() ?? 0
        print("DINOv3 ViT-L real cosine \(cos) maxAbsDiff \(maxd)")
        XCTAssertGreaterThan(cos, 0.999)
    }
}
