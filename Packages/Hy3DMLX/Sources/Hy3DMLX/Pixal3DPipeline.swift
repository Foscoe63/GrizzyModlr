import Foundation
import MLX

/// Pixal3D (TencentARC) geometry at the 512 level, single view: the TRELLIS.2 stages with pixel-aligned
/// conditioning. Structure stage: 16³ projected DINOv3 features. Shape stage: 32³ projected features, low-res
/// DINOv3 concatenated with NAF-upsampled DINOv3 (2048-d). Global cross-attention tokens are CLS + 4 registers only.
/// The released pipeline runs this stage as the low-resolution half of a 1024/1536 cascade; the 1024 stage is not ported.
public final class Pixal3DPipeline {
    public struct Paths {
        public var dino, ssFlow, ssDecoder, shapeFlow512, shapeDecoder, naf: String
        public init(dino: String, ssFlow: String, ssDecoder: String, shapeFlow512: String, shapeDecoder: String, naf: String) {
            self.dino = dino; self.ssFlow = ssFlow; self.ssDecoder = ssDecoder
            self.shapeFlow512 = shapeFlow512; self.shapeDecoder = shapeDecoder; self.naf = naf
        }
    }

    public struct Options {
        public var seed: UInt64 = 42
        public var steps = 12
        public var guidance: Float = 7.5
        /// Pixal3D's defaults when no FOV estimate is available (MoGe-2 is not ported).
        public var fov: Float = 0.8575560450553894
        /// nil: derived from the FOV exactly as Pixal3D's `distance_from_fov` does (left edge of the object box at the left image edge).
        public var distance: Float? = nil
        public var meshScale: Float = 1
        public init() {}
    }

    /// `distance_from_fov` of the reference inference script (extend_pixel 0, image resolution 512).
    static func distance(fov: Float, meshScale: Float, imageSize: Int = 512) -> Float {
        let f = Float(imageSize) / (2 * tan(fov / 2))
        return f * (0.5 / meshScale) / (Float(imageSize) / 2)
    }

    struct Stages { var coords: MLXArray, shapeLatent: MLXArray, decoded: TRELLIS2ShapeDecoder.Output }

    private let paths: Paths
    private let dtype: DType
    public init(paths: Paths, dtype: DType = .float32) { self.paths = paths; self.dtype = dtype }

    private func load(_ p: String) throws -> [String: MLXArray] { try loadArrays(url: URL(fileURLWithPath: p)) }

    /// `pixels`: [1,512,512,3] ImageNet-normalised (object on black, 10% margin, see `Preprocess.dinoPixels`);
    /// `image01`: the same crop unnormalised in [0,1] (NAF's guide image).
    public func run(pixels: MLXArray, image01: MLXArray, options: Options = Options(), hooks: PipelineHooks = PipelineHooks()) throws -> (vertices: [Float], faces: [Int32]) {
        let s = try stages(pixels: pixels, image01: image01, options: options, hooks: hooks)
        let m = FlexibleDualGridMesh.extract(coords: s.decoded.coords.asArray(Int32.self),
                                             feats: s.decoded.feats.asArray(Float.self), grid: 512)
        return (m.vertices, m.faces)
    }

    func stages(pixels: MLXArray, image01: MLXArray, options o: Options, hooks: PipelineHooks = PipelineHooks()) throws -> Stages {
        let key = MLXRandom.key(o.seed)
        func check() throws { if hooks.isCancelled() { throw CancellationError() } }
        hooks.progress("Encoding image", 0.02)
        let keys = MLXRandom.split(key: key, into: 2)
        let dist = o.distance ?? Self.distance(fov: o.fov, meshScale: o.meshScale)

        // DINOv3 once: global tokens (CLS + 4 registers) and the 32×32 patch map
        var global: MLXArray, patch: MLXArray, hr: MLXArray
        do {
            let dino = DINOv3FeatureExtractor(weights: try load(paths.dino), heads: 16, ropeTheta: 100)
            let t = dino(pixels)[0]
            global = t[..<5]
            patch = t[5...].reshaped(32, 32, -1)
            eval(global, patch)
        }
        do {
            hr = NAFUpsampler(weights: try load(paths.naf))(image: image01, features: patch.expandedDimensions(axis: 0),
                                                            outputSize: (512, 512))[0]
            eval(hr)
        }

        try check()
        func sampler(rescaleT: Float, rescale: Float) -> FlowEulerSampler {
            var p = FlowEulerSampler.Params()
            p.steps = o.steps; p.rescaleT = rescaleT; p.guidanceStrength = o.guidance; p.guidanceRescale = rescale
            return FlowEulerSampler(params: p)
        }
        let zerosG = MLXArray.zeros(like: global)

        // structure stage (16³ projection grid, DINOv3 features only: 1024-d)
        let ssLatent: MLXArray
        do {
            let projSS = Pixal3DProjection.sample(features: patch, grid: 16, imageSize: 512, fov: o.fov, distance: dist, meshScale: o.meshScale)
            let zerosP = MLXArray.zeros(like: projSS)
            let dit = TRELLIS2FlowDiT(weights: try load(paths.ssFlow), dtype: dtype)
            let g = (0 ..< 16).flatMap { x in (0 ..< 16).flatMap { y in (0 ..< 16).flatMap { z in [Int32(x), Int32(y), Int32(z)] } } }
            let grid = MLXArray(g, [4096, 3])
            ssLatent = sampler(rescaleT: 5, rescale: 0.7).sample(noise: MLXRandom.normal([1, 16, 16, 16, 8], key: keys[0]), onStep: { i, n in
                hooks.progress("Sampling structure", 0.1 + 0.3 * Double(i) / Double(n)); return !hooks.isCancelled()
            }) { x, t1000, c in
                dit(x: x.reshaped(4096, 8), coords: grid, t: t1000, cond: c ? global : zerosG, proj: c ? projSS : zerosP)
                    .reshaped(1, 16, 16, 16, 8)
            }
            eval(ssLatent)
        }
        try check()
        hooks.progress("Decoding structure", 0.42)
        let logits = TRELLIS2SparseStructureDecoder(weights: try load(paths.ssDecoder)).decode(ssLatent)
        let coords = TRELLIS2SparseStructureDecoder.occupiedCoords(logits, resolution: 32)
        eval(coords)
        let n = coords.dim(0)

        // shape stage (32³ grid, [DINOv3 | NAF-upsampled DINOv3] = 2048-d, gathered at the occupied voxels)
        let lr = Pixal3DProjection.sample(features: patch, grid: 32, imageSize: 512, fov: o.fov, distance: dist, meshScale: o.meshScale)
        let hi = Pixal3DProjection.sample(features: hr, grid: 32, imageSize: 512, fov: o.fov, distance: dist, meshScale: o.meshScale)
        let c3 = coords[0..., 1...].asType(.int32)
        let lin = c3[0..., 0] * 1024 + c3[0..., 1] * 32 + c3[0..., 2]
        let projShape = concatenated([lr, hi], axis: -1)[lin]
        let zerosS = MLXArray.zeros(like: projShape)
        eval(projShape)

        let latent: MLXArray
        do {
            let dit = TRELLIS2FlowDiT(weights: try load(paths.shapeFlow512), dtype: dtype)
            latent = sampler(rescaleT: 3, rescale: 0.5).sample(noise: MLXRandom.normal([n, 32], key: keys[1]), onStep: { i, nn in
                hooks.progress("Sampling shape", 0.45 + 0.4 * Double(i) / Double(nn)); return !hooks.isCancelled()
            }) { x, t1000, c in
                dit(x: x, coords: c3, t: t1000, cond: c ? global : zerosG, proj: c ? projShape : zerosS)
            }
            eval(latent)
        }
        try check()
        hooks.progress("Decoding shape", 0.88)
        let denorm = latent * MLXArray(TRELLIS2Pipeline.shapeStd) + MLXArray(TRELLIS2Pipeline.shapeMean)
        let out = TRELLIS2ShapeDecoder(weights: try load(paths.shapeDecoder)).decode(coords: coords, latent: denorm, size: 32)
        eval(out.coords, out.feats)
        return Stages(coords: coords, shapeLatent: latent, decoded: out)
    }
}
