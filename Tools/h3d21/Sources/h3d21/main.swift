import Foundation
import MLX
import Hy3DMLX

// Developer harness for the Hunyuan3D 2.1 port (reads weights from the external models drive, which
// the XCTest host cannot open). `h3d21 parity` compares each module with the PyTorch fixtures.
func sub(_ w: [String: MLXArray], _ prefix: String) -> [String: MLXArray] {
    var o: [String: MLXArray] = [:]
    for (k, v) in w where k.hasPrefix(prefix) { o[String(k.dropFirst(prefix.count))] = v }
    return o
}
func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.asType(.float32).reshaped([-1]), y = b.asType(.float32).reshaped([-1])
    return ((x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()))).item(Float.self)
}
func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float { abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self) }

let args = CommandLine.arguments
let weightsPath = ProcessInfo.processInfo.environment["H3D21_WEIGHTS"] ?? "/Volumes/Models/Modelr/shape-v21/model.fp16.safetensors"
let fxPath = ProcessInfo.processInfo.environment["H3D21_FIXTURES"] ?? "/Volumes/Models/Modelr/ref21/work/fixtures.safetensors"

switch args.count > 1 ? args[1] : "" {
case "parity":
    let w = try loadArrays(url: URL(fileURLWithPath: weightsPath))
    let fx = try loadArrays(url: URL(fileURLWithPath: fxPath))
    let dino = DINOv2(weights: sub(w, "conditioner.main_image_encoder.model."), heads: 16, layers: 24, bits: 4)
    let img = (fx["dino_img"]! + 1) / 2
    let mean = MLXArray([0.485, 0.456, 0.406] as [Float]), std = MLXArray([0.229, 0.224, 0.225] as [Float])
    let d = dino(((img.transposed(0, 2, 3, 1) - mean) / std).asType(.float16))
    print("DINOv2-large  cosine \(cosine(d, fx["dino_out"]!))  maxabs \(maxAbs(d, fx["dino_out"]!))")
    let vae = VAE(weights: sub(w, "vae."), scaleFactor: 1.0039506)
    let kv = vae.decode(fx["vae_lat"]!.asType(.float16))
    let lg = vae.geoDecoder(fx["vae_q"]!.asType(.float16), fx["vae_kv"]!.asType(.float16))
    print("VAE decode    cosine \(cosine(kv, fx["vae_kv"]!))  maxabs \(maxAbs(kv, fx["vae_kv"]!))")
    print("VAE geo SDF   cosine \(cosine(lg, fx["vae_logits"]!))  maxabs \(maxAbs(lg, fx["vae_logits"]!))")
    let dw = sub(w, "model.")
    for (name, eps) in [("fp32 eps", Float(1.1920929e-7)), ("fp16 eps", Float(9.765625e-4))] {
        let dit = DiT21(weights: dw, rmsEps: eps)
        let o = dit(fx["dit_x"]!.asType(.float16), fx["dit_t"]!, fx["dit_cond"]!.asType(.float16))
        print("DiT21 (\(name)) cosine \(cosine(o, fx["dit_out"]!))  maxabs \(maxAbs(o, fx["dit_out"]!))")
    }
case "e2e":
    // h3d21 e2e <name> [resolution] [quantize]: run the full on-device pipeline from the reference's
    // preprocessed pixels + noise; report latent parity and write an OBJ for Chamfer comparison.
    let name = args.count > 2 ? args[2] : "004"
    let res = args.count > 3 ? Int(args[3])! : 128
    let q = args.count > 4 ? Int(args[4])! : 0
    let work = "/Volumes/Models/Modelr/ref21/work"
    let ref = try loadArrays(url: URL(fileURLWithPath: "\(work)/e2e_\(name).safetensors"))
    let t0 = Date()
    let gen = try ShapeGenerator(weightsURL: URL(fileURLWithPath: weightsPath), quantize: q)
    print("loaded in \(Date().timeIntervalSince(t0))s")
    let t1 = Date()
    guard let mesh = gen.generate(pixels: ref["pix_nhwc"]!.asType(.float16), noise: ref["noise"]!, steps: 30,
                                  guidance: 5.0, resolution: res, octree: true,
                                  onProgress: { p in if p.stage.hasPrefix("Denoising") && p.fraction.truncatingRemainder(dividingBy: 0.2) < 0.02 { print(p.stage) } }) else {
        print("generate failed"); exit(1)
    }
    print("generated in \(Date().timeIntervalSince(t1))s: \(mesh.vertices.count) verts, \(mesh.faces.count) faces")
    var obj = ""
    for v in mesh.vertices { obj += "v \(v.x) \(v.y) \(v.z)\n" }
    for f in mesh.faces { obj += "f \(f.0 + 1) \(f.1 + 1) \(f.2 + 1)\n" }
    try obj.write(toFile: "\(work)/swift_\(name)_r\(res)_q\(q).obj", atomically: true, encoding: .utf8)
default:
    print("usage: h3d21 parity | e2e <name> [res] [quantize]")
}
