import Foundation
import simd
import ImageIO
import UniformTypeIdentifiers

/// Bakes a tangent-space normal map from a full-resolution mesh onto a decimated, UV-unwrapped
/// one ("simplify, then bake"): the low-poly surface keeps the silhouette and the map carries the
/// fine relief the decimation removed. Conventions match glTF 2.0: UVs have a top-left origin
/// (image row = v · height), the green channel points toward decreasing v, and the bitangent is
/// `w · cross(N, T)`.
enum NormalBaker {
    struct Options {
        var size: Int = 2048
        /// Ray cage half-depth as a fraction of the mesh's bounding-box diagonal.
        var cage: Float = 0.015
        /// 1 = the baked relief as is; below 1 flattens it, above 1 exaggerates it.
        var strength: Float = 1
        /// Texels the baked islands are grown by so filtering never samples empty texture.
        var padding: Int = 8
    }

    struct HighMesh {
        var positions: [SIMD3<Float>]
        var faces: [SIMD3<UInt32>]
    }

    // MARK: - Public entry points

    /// RGB8 pixels (top-left row-major, `size · size · 3` bytes), or nil when the inputs are unusable.
    static func bake(low: TexturedMesh, high: HighMesh, options: Options,
                     isCancelled: () -> Bool = { false }) -> [UInt8]? {
        let size = options.size
        guard size >= 16, !low.faces.isEmpty, !high.faces.isEmpty,
              low.uvs.count == low.positions.count, low.normals.count == low.positions.count else { return nil }

        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
        for p in high.positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let diag = simd_length(hi - lo)
        guard diag > 1e-9 else { return nil }
        let cage = max(options.cage, 1e-4) * diag

        let bvh = BVH(positions: high.positions, faces: high.faces)
        let highNormals = smoothNormals(positions: high.positions, faces: high.faces)
        if isCancelled() { return nil }

        let frames = tangentFrames(low)
        let owner = rasterize(low, size: size)
        if isCancelled() { return nil }

        var pixels = [UInt8](repeating: 0, count: size * size * 3)
        var covered = [Bool](repeating: false, count: size * size)
        let sizeF = Float(size)
        let strength = options.strength

        pixels.withUnsafeMutableBufferPointer { pix in
            covered.withUnsafeMutableBufferPointer { cov in
                owner.withUnsafeBufferPointer { own in
                    let pixBase = pix.baseAddress!, covBase = cov.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: size) { row in
                        for col in 0..<size {
                            let f = Int(own[row * size + col])
                            guard f >= 0 else { continue }
                            let face = low.faces[f]
                            let i0 = Int(face.x), i1 = Int(face.y), i2 = Int(face.z)
                            let uv = SIMD2<Float>((Float(col) + 0.5) / sizeF, (Float(row) + 0.5) / sizeF)
                            guard let b = barycentric(uv, low.uvs[i0], low.uvs[i1], low.uvs[i2]) else { continue }
                            let p = low.positions[i0] * b.x + low.positions[i1] * b.y + low.positions[i2] * b.z
                            var n = low.normals[i0] * b.x + low.normals[i1] * b.y + low.normals[i2] * b.z
                            let nl = simd_length(n)
                            n = nl > 1e-9 ? n / nl : SIMD3(0, 0, 1)
                            var t = frames.tangents[i0] * b.x + frames.tangents[i1] * b.y + frames.tangents[i2] * b.z
                            t -= n * simd_dot(n, t)
                            let tl = simd_length(t)
                            t = tl > 1e-9 ? t / tl : frames.tangents[i0]
                            let w = frames.signs[i0]
                            let bt = simd_cross(n, t) * w

                            var tangentSpace = SIMD3<Float>(0, 0, 1)
                            if let hit = bvh.closestHit(origin: p + n * cage, direction: -n, maxDistance: 2 * cage) {
                                let hf = high.faces[hit.face]
                                var hn = highNormals[Int(hf.x)] * (1 - hit.u - hit.v)
                                    + highNormals[Int(hf.y)] * hit.u + highNormals[Int(hf.z)] * hit.v
                                let hl = simd_length(hn)
                                if hl > 1e-9 {
                                    hn /= hl
                                    var ts = SIMD3<Float>(simd_dot(hn, t), simd_dot(hn, bt), simd_dot(hn, n))
                                    ts.x *= strength; ts.y *= strength
                                    ts.z = max(ts.z, 0.05)
                                    tangentSpace = simd_normalize(ts)
                                }
                            }
                            let o = (row * size + col) * 3
                            pixBase[o] = encode(tangentSpace.x)
                            pixBase[o + 1] = encode(tangentSpace.y)
                            pixBase[o + 2] = encode(tangentSpace.z)
                            covBase[row * size + col] = true
                        }
                    }
                }
            }
        }
        if isCancelled() { return nil }
        pad(&pixels, covered: &covered, size: size, passes: options.padding)
        return pixels
    }

    /// Bakes and encodes straight to PNG bytes.
    static func bakePNG(low: TexturedMesh, high: HighMesh, options: Options,
                        isCancelled: () -> Bool = { false }) -> Data? {
        guard let rgb = bake(low: low, high: high, options: options, isCancelled: isCancelled) else { return nil }
        return pngData(rgb: rgb, size: options.size)
    }

    // MARK: - Frames

    /// Per-vertex tangent and handedness from UV derivatives, accumulated over the faces sharing
    /// the vertex (vertices are already split along UV seams).
    static func tangentFrames(_ mesh: TexturedMesh) -> (tangents: [SIMD3<Float>], signs: [Float]) {
        let n = mesh.positions.count
        var dPdu = [SIMD3<Float>](repeating: .zero, count: n)
        var dPdv = [SIMD3<Float>](repeating: .zero, count: n)
        for f in mesh.faces {
            let i0 = Int(f.x), i1 = Int(f.y), i2 = Int(f.z)
            let e1 = mesh.positions[i1] - mesh.positions[i0], e2 = mesh.positions[i2] - mesh.positions[i0]
            let d1 = mesh.uvs[i1] - mesh.uvs[i0], d2 = mesh.uvs[i2] - mesh.uvs[i0]
            let det = d1.x * d2.y - d2.x * d1.y
            guard abs(det) > 1e-20 else { continue }
            let u = (e1 * d2.y - e2 * d1.y) / det
            let v = (e2 * d1.x - e1 * d2.x) / det
            let weight = simd_length(simd_cross(e1, e2))      // area weighting
            let uw = simd_normalize(u) * weight, vw = simd_normalize(v) * weight
            for i in [i0, i1, i2] { dPdu[i] += uw; dPdv[i] += vw }
        }
        var tangents = [SIMD3<Float>](repeating: SIMD3(1, 0, 0), count: n)
        var signs = [Float](repeating: 1, count: n)
        for i in 0..<n {
            let nrm = mesh.normals[i]
            var t = dPdu[i] - nrm * simd_dot(nrm, dPdu[i])
            let tl = simd_length(t)
            if tl < 1e-9 {
                let axis: SIMD3<Float> = abs(nrm.x) < 0.9 ? SIMD3(1, 0, 0) : SIMD3(0, 1, 0)
                t = simd_normalize(simd_cross(axis, nrm))
            } else { t /= tl }
            tangents[i] = t
            // glTF v runs downward, so the image's +Y (green) is toward decreasing v.
            signs[i] = simd_dot(simd_cross(nrm, t), -dPdv[i]) < 0 ? -1 : 1
        }
        return (tangents, signs)
    }

    /// Area-weighted smooth vertex normals.
    static func smoothNormals(positions: [SIMD3<Float>], faces: [SIMD3<UInt32>]) -> [SIMD3<Float>] {
        var normals = [SIMD3<Float>](repeating: .zero, count: positions.count)
        for f in faces {
            let a = Int(f.x), b = Int(f.y), c = Int(f.z)
            let fn = simd_cross(positions[b] - positions[a], positions[c] - positions[a])
            normals[a] += fn; normals[b] += fn; normals[c] += fn
        }
        return normals.map { let l = simd_length($0); return l > 1e-12 ? $0 / l : SIMD3(0, 0, 1) }
    }

    // MARK: - Rasterization

    /// Face index covering each texel center (−1 = none), top-left row-major.
    static func rasterize(_ mesh: TexturedMesh, size: Int) -> [Int32] {
        var owner = [Int32](repeating: -1, count: size * size)
        let s = Float(size)
        for (fi, f) in mesh.faces.enumerated() {
            let a = mesh.uvs[Int(f.x)] * s, b = mesh.uvs[Int(f.y)] * s, c = mesh.uvs[Int(f.z)] * s
            let minX = max(0, Int((min(a.x, b.x, c.x) - 0.5).rounded(.down)))
            let maxX = min(size - 1, Int((max(a.x, b.x, c.x) - 0.5).rounded(.up)))
            let minY = max(0, Int((min(a.y, b.y, c.y) - 0.5).rounded(.down)))
            let maxY = min(size - 1, Int((max(a.y, b.y, c.y) - 0.5).rounded(.up)))
            if minX > maxX || minY > maxY { continue }
            for y in minY...maxY {
                for x in minX...maxX {
                    let p = SIMD2<Float>(Float(x) + 0.5, Float(y) + 0.5)
                    if barycentric(p, a, b, c) != nil { owner[y * size + x] = Int32(fi) }
                }
            }
        }
        return owner
    }

    /// Barycentric weights of `p` in triangle abc, or nil outside it / when degenerate.
    static func barycentric(_ p: SIMD2<Float>, _ a: SIMD2<Float>, _ b: SIMD2<Float>, _ c: SIMD2<Float>)
        -> SIMD3<Float>? {
        let v0 = b - a, v1 = c - a, v2 = p - a
        let den = v0.x * v1.y - v1.x * v0.y
        guard abs(den) > 1e-20 else { return nil }
        let v = (v2.x * v1.y - v1.x * v2.y) / den
        let w = (v0.x * v2.y - v2.x * v0.y) / den
        let u = 1 - v - w
        let eps: Float = -1e-5
        return (u >= eps && v >= eps && w >= eps) ? SIMD3(u, v, w) : nil
    }

    // MARK: - Padding and encoding

    private static func encode(_ v: Float) -> UInt8 { UInt8(max(0, min(255, ((v * 0.5 + 0.5) * 255).rounded()))) }

    /// Grows covered texels outward so bilinear / mip sampling near island borders stays valid;
    /// whatever is still empty afterwards becomes the flat normal.
    static func pad(_ pixels: inout [UInt8], covered: inout [Bool], size: Int, passes: Int) {
        for _ in 0..<max(0, passes) {
            let snapshot = covered
            var changed = false
            for y in 0..<size {
                for x in 0..<size where !snapshot[y * size + x] {
                    var found = -1
                    search: for dy in -1...1 {
                        for dx in -1...1 where dx != 0 || dy != 0 {
                            let nx = x + dx, ny = y + dy
                            if nx >= 0, ny >= 0, nx < size, ny < size, snapshot[ny * size + nx] {
                                found = ny * size + nx; break search
                            }
                        }
                    }
                    if found >= 0 {
                        let o = (y * size + x) * 3, s = found * 3
                        pixels[o] = pixels[s]; pixels[o + 1] = pixels[s + 1]; pixels[o + 2] = pixels[s + 2]
                        covered[y * size + x] = true
                        changed = true
                    }
                }
            }
            if !changed { break }
        }
        for i in 0..<(size * size) where !covered[i] {
            pixels[i * 3] = 128; pixels[i * 3 + 1] = 128; pixels[i * 3 + 2] = 255
        }
    }

    static func pngData(rgb: [UInt8], size: Int) -> Data? {
        guard let provider = CGDataProvider(data: Data(rgb) as CFData),
              let image = CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 24,
                                  bytesPerRow: size * 3, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent) else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, image, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }

    // MARK: - BVH

    struct Hit { var t: Float; var face: Int; var u: Float; var v: Float }

    /// Flat bounding-volume hierarchy over triangles (median split on the longest centroid axis).
    final class BVH {
        private struct Node {
            var lo: SIMD3<Float>, hi: SIMD3<Float>
            var left: Int32      // child index, or first triangle slot for a leaf
            var right: Int32     // child index, or -count for a leaf
        }
        private var nodes: [Node] = []
        private var order: [Int32]
        private let positions: [SIMD3<Float>]
        private let faces: [SIMD3<UInt32>]

        init(positions: [SIMD3<Float>], faces: [SIMD3<UInt32>]) {
            self.positions = positions
            self.faces = faces
            order = (0..<faces.count).map { Int32($0) }
            let centroids: [SIMD3<Float>] = faces.map {
                (positions[Int($0.x)] + positions[Int($0.y)] + positions[Int($0.z)]) / 3
            }
            nodes.reserveCapacity(faces.count / 2 + 1)
            _ = build(0, faces.count, centroids)
        }

        private func bounds(_ start: Int, _ end: Int) -> (SIMD3<Float>, SIMD3<Float>) {
            var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
            for k in start..<end {
                let f = faces[Int(order[k])]
                for v in [f.x, f.y, f.z] { lo = simd_min(lo, positions[Int(v)]); hi = simd_max(hi, positions[Int(v)]) }
            }
            return (lo, hi)
        }

        private func build(_ start: Int, _ end: Int, _ centroids: [SIMD3<Float>]) -> Int {
            let (lo, hi) = bounds(start, end)
            let index = nodes.count
            nodes.append(Node(lo: lo, hi: hi, left: Int32(start), right: Int32(-(end - start))))
            if end - start <= 4 { return index }

            var clo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), chi = -clo
            for k in start..<end { clo = simd_min(clo, centroids[Int(order[k])]); chi = simd_max(chi, centroids[Int(order[k])]) }
            let ext = chi - clo
            let axis = ext.x > ext.y ? (ext.x > ext.z ? 0 : 2) : (ext.y > ext.z ? 1 : 2)
            if ext[axis] < 1e-12 { return index }
            let mid = (clo[axis] + chi[axis]) / 2

            // in-place partition around the centroid midpoint
            var i = start, j = end - 1
            while i <= j {
                if centroids[Int(order[i])][axis] < mid { i += 1 } else { order.swapAt(i, j); j -= 1 }
            }
            var split = i
            if split == start || split == end { split = (start + end) / 2 }   // degenerate: halve by count

            let l = build(start, split, centroids)
            let r = build(split, end, centroids)
            nodes[index].left = Int32(l)
            nodes[index].right = Int32(r)
            return index
        }

        /// Closest hit along a ray, double-sided, within `maxDistance`.
        func closestHit(origin: SIMD3<Float>, direction: SIMD3<Float>, maxDistance: Float) -> Hit? {
            guard !nodes.isEmpty else { return nil }
            var best: Hit? = nil
            var tMax = maxDistance
            let inv = SIMD3<Float>(1 / direction.x, 1 / direction.y, 1 / direction.z)
            var stack = [Int32](repeating: 0, count: 64)
            var sp = 0
            stack[0] = 0; sp = 1
            while sp > 0 {
                sp -= 1
                let node = nodes[Int(stack[sp])]
                // slab test
                let t0 = (node.lo - origin) * inv, t1 = (node.hi - origin) * inv
                let tmin = simd_min(t0, t1), tmax = simd_max(t0, t1)
                let near = max(max(tmin.x, tmin.y), max(tmin.z, 0))
                let far = min(min(tmax.x, tmax.y), min(tmax.z, tMax))
                if near > far { continue }
                if node.right < 0 {
                    let first = Int(node.left), count = Int(-node.right)
                    for k in first..<(first + count) {
                        let fi = Int(order[k])
                        let f = faces[fi]
                        if let h = Self.intersect(origin, direction, positions[Int(f.x)], positions[Int(f.y)],
                                                  positions[Int(f.z)]), h.t < tMax {
                            tMax = h.t
                            best = Hit(t: h.t, face: fi, u: h.u, v: h.v)
                        }
                    }
                } else if sp + 2 <= stack.count {
                    stack[sp] = node.left; stack[sp + 1] = node.right; sp += 2
                }
            }
            return best
        }

        /// Möller–Trumbore, both faces.
        private static func intersect(_ o: SIMD3<Float>, _ d: SIMD3<Float>, _ a: SIMD3<Float>,
                                      _ b: SIMD3<Float>, _ c: SIMD3<Float>) -> (t: Float, u: Float, v: Float)? {
            let e1 = b - a, e2 = c - a
            let p = simd_cross(d, e2)
            let det = simd_dot(e1, p)
            if abs(det) < 1e-12 { return nil }
            let invDet = 1 / det
            let s = o - a
            let u = simd_dot(s, p) * invDet
            if u < 0 || u > 1 { return nil }
            let q = simd_cross(s, e1)
            let v = simd_dot(d, q) * invDet
            if v < 0 || u + v > 1 { return nil }
            let t = simd_dot(e2, q) * invDet
            return t >= 0 ? (t, u, v) : nil
        }
    }
}
