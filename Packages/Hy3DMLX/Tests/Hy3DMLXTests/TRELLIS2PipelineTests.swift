import XCTest
import MLX
@testable import Hy3DMLX

/// End-to-end 512 run on the real weights (slow, memory heavy). Writes an OBJ for inspection.
final class TRELLIS2PipelineTests: XCTestCase {
    private let root = ProcessInfo.processInfo.environment["TRELLIS2_DIR"] ?? "/Volumes/Models/Modelr/trellis2"

    static func writeOBJ(_ v: [Float], _ f: [Int32], to path: String) throws {
        var s = ""
        for i in 0 ..< v.count / 3 { s += "v \(v[i * 3]) \(v[i * 3 + 1]) \(v[i * 3 + 2])\n" }
        for i in 0 ..< f.count / 3 { s += "f \(f[i * 3] + 1) \(f[i * 3 + 1] + 1) \(f[i * 3 + 2] + 1)\n" }
        try s.write(toFile: path, atomically: true, encoding: .utf8)
    }

    func testImageToMesh512() throws {
        let img = ProcessInfo.processInfo.environment["TRELLIS2_IMAGE"] ?? root + "/ref-src/assets/example_image/0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp"
        let ck = root + "/TRELLIS.2-4B/ckpts/", ss = root + "/TRELLIS-image-large/ckpts/"
        let paths = TRELLIS2Pipeline.Paths(
            dino: root + "/dinov3-vitl16/model.safetensors",
            ssFlow: ck + "ss_flow_img_dit_1_3B_64_bf16.safetensors",
            ssDecoder: ss + "ss_dec_conv3d_16l8_fp16.safetensors",
            shapeFlow512: ck + "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
            shapeDecoder: ck + "shape_dec_next_dc_f16c32_fp16.safetensors")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: paths.shapeFlow512) && FileManager.default.fileExists(atPath: img),
                          "TRELLIS.2 weights/image not installed")
        let px = try XCTUnwrap(Preprocess.dinoPixels(img, size: 512, border: 0, background: 0))
        let t0 = Date()
        let m = try TRELLIS2Pipeline(paths: paths).run(pixels: px)
        print("TRELLIS2 512: \(m.vertices.count / 3) verts, \(m.faces.count / 3) faces, \(Date().timeIntervalSince(t0)) s, peak MLX \(GPU.peakMemory >> 20) MB")
        let out = root + "/out"; try FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)
        try Self.writeOBJ(m.vertices, m.faces, to: out + "/trellis2_512.obj")
        XCTAssertGreaterThan(m.faces.count, 1000)
    }

    func testPixal3DImageToMesh512() throws {
        let img = ProcessInfo.processInfo.environment["TRELLIS2_IMAGE"] ?? root + "/ref-src/assets/example_image/0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp"
        let px = ProcessInfo.processInfo.environment["PIXAL3D_DIR"] ?? "/Volumes/Models/Modelr/pixal3d"
        let paths = Pixal3DPipeline.Paths(
            dino: root + "/dinov3-vitl16/model.safetensors",
            ssFlow: px + "/Pixal3D/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
            ssDecoder: root + "/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
            shapeFlow512: px + "/Pixal3D/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
            shapeDecoder: root + "/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
            naf: px + "/naf/naf.safetensors")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: paths.shapeFlow512) && FileManager.default.fileExists(atPath: img), "Pixal3D weights/image not installed")
        let border: Float = 1 - 1 / 1.1
        let npx = try XCTUnwrap(Preprocess.dinoPixels(img, size: 512, border: border, background: 0))
        let raw = try XCTUnwrap(Preprocess.dinoPixels(img, size: 512, border: border, background: 0, normalize: false))
        let t0 = Date()
        var o = Pixal3DPipeline.Options()
        let env = ProcessInfo.processInfo.environment
        if let v = env["PIXAL3D_STEPS"], let n = Int(v) { o.steps = n }
        if let v = env["PIXAL3D_SEED"], let n = UInt64(v) { o.seed = n }
        if let v = env["PIXAL3D_FOV"], let x = Float(v) { o.fov = x }
        let m = try Pixal3DPipeline(paths: paths).run(pixels: npx, image01: raw, options: o)
        print("Pixal3D 512: \(m.vertices.count / 3) verts, \(m.faces.count / 3) faces, \(Date().timeIntervalSince(t0)) s, peak MLX \(GPU.peakMemory >> 20) MB")
        let out = root + "/out"; try FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)
        try Self.writeOBJ(m.vertices, m.faces, to: out + "/pixal3d_512" + (env["PIXAL3D_TAG"] ?? "") + ".obj")
        XCTAssertGreaterThan(m.faces.count, 1000)
    }
}
