import Foundation
import MLX

/// TRELLIS.2 512-resolution image → mesh (geometry only), mirroring `Trellis2ImageTo3DPipeline.run` with
/// pipeline_type "512": DINOv3 conditioning → structure flow → structure decoder (64³ occupancy, pooled to 32³) →
/// shape SLat flow → de-normalise → shape decoder (flexible dual grid at 512) → mesher.
public final class TRELLIS2Pipeline {
    public struct Paths {
        public var dino: String            // dinov3 model.safetensors
        public var ssFlow: String          // ss_flow_img_dit_1_3B_64_bf16.safetensors
        public var ssDecoder: String       // ss_dec_conv3d_16l8_fp16.safetensors
        public var shapeFlow512: String    // slat_flow_img2shape_dit_1_3B_512_bf16.safetensors
        public var shapeDecoder: String    // shape_dec_next_dc_f16c32_fp16.safetensors
        public init(dino: String, ssFlow: String, ssDecoder: String, shapeFlow512: String, shapeDecoder: String) {
            self.dino = dino; self.ssFlow = ssFlow; self.ssDecoder = ssDecoder
            self.shapeFlow512 = shapeFlow512; self.shapeDecoder = shapeDecoder
        }
    }

    public struct Options {
        public var seed: UInt64 = 42
        public var ssSteps = 12, shapeSteps = 12
        public var ssGuidance: Float = 7.5, shapeGuidance: Float = 7.5
        public init() {}
    }

    /// Intermediates for stage-by-stage comparison.
    struct Stages {
        var cond: MLXArray, ssLatent: MLXArray, ssLogits: MLXArray, coords: MLXArray
        var shapeLatent: MLXArray, decoded: TRELLIS2ShapeDecoder.Output
    }

    static let shapeMean: [Float] = [0.781296, 0.018091, -0.495192, -0.558457, 1.06053, 0.093252, 1.518149, -0.933218, -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944, 0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667, -0.059621, 2.22078, 1.621212, 0.87723, 0.567247, -3.175944, -3.186688, 1.578665]
    static let shapeStd: [Float] = [5.972266, 4.706852, 5.44501, 5.209927, 5.32022, 4.547237, 5.020802, 5.444004, 5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578, 4.805265, 5.124013, 5.530808, 5.619001, 5.10393, 5.41767, 5.269677, 5.547194, 5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691]

    private let paths: Paths
    private let dtype: DType
    public init(paths: Paths, dtype: DType = .float32) { self.paths = paths; self.dtype = dtype }

    /// `pixels`: ImageNet-normalised NHWC [1, 512, 512, 3] (see `Preprocess.dinoPixels(_:size:)`).
    public func run(pixels: MLXArray, options: Options = Options()) throws -> (vertices: [Float], faces: [Int32]) {
        let s = try stages(pixels: pixels, options: options)
        let m = FlexibleDualGridMesh.extract(coords: s.decoded.coords.asArray(Int32.self),
                                             feats: s.decoded.feats.asArray(Float.self), grid: 512)
        return (m.vertices, m.faces)
    }

    func stages(pixels: MLXArray, options: Options, noiseSS: MLXArray? = nil, noiseSlat: MLXArray? = nil) throws -> Stages {
        let key = MLXRandom.key(options.seed)

        // 1. conditioning (weights freed after use)
        var cond: MLXArray
        do {
            let dino = DINOv3FeatureExtractor(weights: try loadArrays(url: URL(fileURLWithPath: paths.dino)), heads: 16, ropeTheta: 100)
            cond = dino(pixels)[0]                        // [1029, 1024]
            eval(cond)
        }
        let neg = MLXArray.zeros(like: cond)

        // 2. sparse structure
        let ssLatent: MLXArray
        do {
            let dit = TRELLIS2FlowDiT(weights: try loadArrays(url: URL(fileURLWithPath: paths.ssFlow)), dtype: dtype)
            var p = FlowEulerSampler.Params()
            p.steps = options.ssSteps; p.rescaleT = 5; p.guidanceStrength = options.ssGuidance; p.guidanceRescale = 0.7
            let g = (0 ..< 16).flatMap { x in (0 ..< 16).flatMap { y in (0 ..< 16).flatMap { z in [Int32(x), Int32(y), Int32(z)] } } }
            let grid = MLXArray(g, [4096, 3])
            let noise = noiseSS ?? MLXRandom.normal([1, 16, 16, 16, 8], key: MLXRandom.split(key: key, into: 2)[0])
            ssLatent = FlowEulerSampler(params: p).sample(noise: noise) { x, t1000, c in
                let out = dit(x: x.reshaped(4096, 8), coords: grid, t: t1000, cond: c ? cond : neg)
                return out.reshaped(1, 16, 16, 16, 8)
            }
            eval(ssLatent)
        }
        let ssLogits = TRELLIS2SparseStructureDecoder(weights: try loadArrays(url: URL(fileURLWithPath: paths.ssDecoder))).decode(ssLatent)
        let coords = TRELLIS2SparseStructureDecoder.occupiedCoords(ssLogits, resolution: 32)
        eval(coords)
        let n = coords.dim(0)

        // 3. shape SLat
        var latent: MLXArray
        do {
            let dit = TRELLIS2FlowDiT(weights: try loadArrays(url: URL(fileURLWithPath: paths.shapeFlow512)), dtype: dtype)
            var p = FlowEulerSampler.Params()
            p.steps = options.shapeSteps; p.rescaleT = 3; p.guidanceStrength = options.shapeGuidance; p.guidanceRescale = 0.5
            let xyz = coords[0..., 1...]
            let noise = noiseSlat ?? MLXRandom.normal([n, 32], key: MLXRandom.split(key: key, into: 2)[1])
            latent = FlowEulerSampler(params: p).sample(noise: noise) { x, t1000, c in
                dit(x: x, coords: xyz, t: t1000, cond: c ? cond : neg)
            }
            eval(latent)
        }
        let denorm = latent * MLXArray(Self.shapeStd) + MLXArray(Self.shapeMean)

        // 4. decode
        let decoder = TRELLIS2ShapeDecoder(weights: try loadArrays(url: URL(fileURLWithPath: paths.shapeDecoder)))
        let out = decoder.decode(coords: coords, latent: denorm, size: 32)
        eval(out.coords, out.feats)
        return Stages(cond: cond, ssLatent: ssLatent, ssLogits: ssLogits, coords: coords, shapeLatent: latent, decoded: out)
    }
}
