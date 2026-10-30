import Foundation

/// Geometry for filling a masked area of a large texture with a 512² generative model: pick a
/// square window around the mask, resample it to the model size, and blend the result back
/// inside the mask only (everything else stays byte-identical).
enum InpaintPatch {
    static let modelSize = 512

    struct Window: Equatable {
        var x: Int, y: Int, side: Int
    }

    /// Bounding box of texels with mask > 0, or nil when nothing is selected.
    static func maskBounds(_ mask: [Float], width: Int, height: Int) -> (x0: Int, y0: Int, x1: Int, y1: Int)? {
        var x0 = width, y0 = height, x1 = -1, y1 = -1
        for y in 0..<height {
            for x in 0..<width where mask[y * width + x] > 0 {
                x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
            }
        }
        return x1 < 0 ? nil : (x0, y0, x1, y1)
    }

    /// A square around the mask with `context`× its size of surroundings (so the model sees the
    /// material it must match), never smaller than `minSide`, shifted to stay inside the texture.
    static func window(bounds b: (x0: Int, y0: Int, x1: Int, y1: Int), width: Int, height: Int,
                       context: Float = 2.0, minSide: Int = 192) -> Window {
        let w = b.x1 - b.x0 + 1, h = b.y1 - b.y0 + 1
        let want = max(minSide, Int((Float(max(w, h)) * context).rounded(.up)))
        let side = min(want, min(width, height))
        let cx = (b.x0 + b.x1 + 1) / 2, cy = (b.y0 + b.y1 + 1) / 2
        let x = max(0, min(width - side, cx - side / 2))
        let y = max(0, min(height - side, cy - side / 2))
        return Window(x: x, y: y, side: side)
    }

    /// Bilinear resample of a square window of straight RGBA8 to `size`², as RGB floats 0…1.
    static func extractRGB(_ rgba: [UInt8], width: Int, window win: Window, size: Int = modelSize) -> [Float] {
        var out = [Float](repeating: 0, count: size * size * 3)
        let scale = Float(win.side) / Float(size)
        for y in 0..<size {
            let sy = (Float(y) + 0.5) * scale - 0.5 + Float(win.y)
            for x in 0..<size {
                let sx = (Float(x) + 0.5) * scale - 0.5 + Float(win.x)
                let px = bilinear(rgba, width: width, window: win, sx: sx, sy: sy)
                let o = (y * size + x) * 3
                out[o] = px.0; out[o + 1] = px.1; out[o + 2] = px.2
            }
        }
        return out
    }

    /// The mask over the window resampled to `size`², 0…1.
    static func extractMask(_ mask: [Float], width: Int, window win: Window, size: Int = modelSize) -> [Float] {
        var out = [Float](repeating: 0, count: size * size)
        let scale = Float(win.side) / Float(size)
        for y in 0..<size {
            let sy = min(win.y + win.side - 1, win.y + Int((Float(y) + 0.5) * scale))
            for x in 0..<size {
                let sx = min(win.x + win.side - 1, win.x + Int((Float(x) + 0.5) * scale))
                out[y * size + x] = mask[sy * width + sx]
            }
        }
        return out
    }

    /// Mask with a softened edge: dilated by `grow` texels then box-blurred by `blur`, 0…1.
    static func feathered(_ mask: [Float], width: Int, height: Int, grow: Int, blur: Int) -> [Float] {
        var m = mask
        if grow > 0 { m = boxMax(m, width, height, grow) }
        if blur > 0 { m = boxBlur(m, width, height, blur) }
        return m
    }

    /// The mask restricted to the window, `side`² texels.
    static func cropMask(_ mask: [Float], width: Int, window win: Window) -> [Float] {
        var out = [Float](repeating: 0, count: win.side * win.side)
        for y in 0..<win.side {
            let src = (win.y + y) * width + win.x
            out.replaceSubrange(y * win.side..<(y + 1) * win.side, with: mask[src..<src + win.side])
        }
        return out
    }

    /// Writes `result` (model-size RGB 0…1) back into `pixels` over the window, weighted by `weight`
    /// (per-texel 0…1, window-sized `side`²). Texels with weight 0 are left untouched; alpha is kept.
    static func composite(_ pixels: inout [UInt8], width: Int, result: [Float], window win: Window,
                          weight: [Float], size: Int = modelSize) {
        let scale = Float(size) / Float(win.side)
        for y in 0..<win.side {
            let sy = (Float(y) + 0.5) * scale - 0.5
            for x in 0..<win.side {
                let w = weight[y * win.side + x]
                guard w > 0 else { continue }
                let sx = (Float(x) + 0.5) * scale - 0.5
                let c = bilinearRGB(result, size: size, sx: sx, sy: sy)
                let i = ((win.y + y) * width + win.x + x) * 4
                pixels[i] = blend(pixels[i], c.0, w)
                pixels[i + 1] = blend(pixels[i + 1], c.1, w)
                pixels[i + 2] = blend(pixels[i + 2], c.2, w)
            }
        }
    }

    // MARK: - helpers

    private static func blend(_ old: UInt8, _ new: Float, _ w: Float) -> UInt8 {
        UInt8(max(0, min(255, Float(old) * (1 - w) + new * 255 * w + 0.5)))
    }

    private static func bilinear(_ rgba: [UInt8], width: Int, window win: Window,
                                 sx: Float, sy: Float) -> (Float, Float, Float) {
        let x0 = Int(sx.rounded(.down)), y0 = Int(sy.rounded(.down))
        let fx = sx - Float(x0), fy = sy - Float(y0)
        func at(_ x: Int, _ y: Int, _ c: Int) -> Float {
            let cx = max(win.x, min(win.x + win.side - 1, x)), cy = max(win.y, min(win.y + win.side - 1, y))
            return Float(rgba[(cy * width + cx) * 4 + c]) / 255
        }
        func ch(_ c: Int) -> Float {
            let top = at(x0, y0, c) * (1 - fx) + at(x0 + 1, y0, c) * fx
            let bot = at(x0, y0 + 1, c) * (1 - fx) + at(x0 + 1, y0 + 1, c) * fx
            return top * (1 - fy) + bot * fy
        }
        return (ch(0), ch(1), ch(2))
    }

    private static func bilinearRGB(_ rgb: [Float], size: Int, sx: Float, sy: Float) -> (Float, Float, Float) {
        let x0 = Int(sx.rounded(.down)), y0 = Int(sy.rounded(.down))
        let fx = sx - Float(x0), fy = sy - Float(y0)
        func at(_ x: Int, _ y: Int, _ c: Int) -> Float {
            rgb[(max(0, min(size - 1, y)) * size + max(0, min(size - 1, x))) * 3 + c]
        }
        func ch(_ c: Int) -> Float {
            let top = at(x0, y0, c) * (1 - fx) + at(x0 + 1, y0, c) * fx
            let bot = at(x0, y0 + 1, c) * (1 - fx) + at(x0 + 1, y0 + 1, c) * fx
            return top * (1 - fy) + bot * fy
        }
        return (ch(0), ch(1), ch(2))
    }

    /// Separable box max (dilation) with radius `r`.
    private static func boxMax(_ m: [Float], _ w: Int, _ h: Int, _ r: Int) -> [Float] {
        func pass(_ src: [Float], horizontal: Bool) -> [Float] {
            var out = src
            for y in 0..<h { for x in 0..<w {
                var best: Float = 0
                for d in -r...r {
                    let xx = horizontal ? x + d : x, yy = horizontal ? y : y + d
                    guard xx >= 0, xx < w, yy >= 0, yy < h else { continue }
                    best = max(best, src[yy * w + xx])
                }
                out[y * w + x] = best
            } }
            return out
        }
        return pass(pass(m, horizontal: true), horizontal: false)
    }

    /// Separable box blur with radius `r` (edges clamp), via running sums.
    private static func boxBlur(_ m: [Float], _ w: Int, _ h: Int, _ r: Int) -> [Float] {
        func pass(_ src: [Float], horizontal: Bool) -> [Float] {
            var out = src
            let n = horizontal ? w : h, lines = horizontal ? h : w
            for line in 0..<lines {
                func idx(_ i: Int) -> Int { horizontal ? line * w + i : i * w + line }
                var sum: Float = 0
                for d in -r...r { sum += src[idx(max(0, min(n - 1, d)))] }
                for i in 0..<n {
                    out[idx(i)] = sum / Float(2 * r + 1)
                    sum += src[idx(min(n - 1, i + r + 1))] - src[idx(max(0, i - r))]
                }
            }
            return out
        }
        return pass(pass(m, horizontal: true), horizontal: false)
    }
}
