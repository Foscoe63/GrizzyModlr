import Foundation
import MLX

/// Pixal3D view-aligned feature sampling (`ProjGrid.forward` in TencentARC/Pixal3D): every voxel centre of an
/// R³ grid in [-1, 1]³ is projected through a pinhole camera looking at the object from the front, and the
/// image feature map is bilinearly sampled there (grid_sample, align_corners=False, border padding).
enum Pixal3DProjection {
    /// - Parameters:
    ///   - features: [H, W, C] feature map (DINOv3 patch tokens, or an upsampled map).
    ///   - grid: voxel grid resolution R.
    ///   - imageSize: the pixel size the camera model uses (512 at the 512 levels).
    ///   - fov: horizontal field of view in radians. distance: camera distance. meshScale: object scale.
    /// - Returns: [R³, C], flattened in (x, y, z) row-major order.
    static func sample(features: MLXArray, grid r: Int, imageSize: Int, fov: Float, distance: Float, meshScale: Float) -> MLXArray {
        let h = features.dim(0), w = features.dim(1)
        let lin: [Float] = (0 ..< r).map { r == 1 ? 0 : -1 + 2 * Float($0) / Float(r - 1) }
        var px = [Float](), py = [Float](), pz = [Float]()
        for x in lin { for y in lin { for z in lin {
            // rotation to Blender axes: (x, y, z) -> (x, -z, y), then scaled by 1 / (meshScale * 2)
            let k = 1 / (meshScale * 2)
            px.append(x * k); py.append(-z * k); pz.append(y * k)
        } } }
        let X = MLXArray(px), Y = MLXArray(py), Z = MLXArray(pz)
        // front-view camera: x_cam = x, y_cam = z, depth = y + distance
        let depth = Y + distance
        let f = Float(imageSize) / (2 * tan(fov / 2))
        let xPix = f * X / (depth + 1e-8) + Float(imageSize) / 2
        let yPix = -f * Z / (depth + 1e-8) + Float(imageSize) / 2
        let gx = (xPix + 0.5) / Float(imageSize) * 2 - 1
        let gy = (yPix + 0.5) / Float(imageSize) * 2 - 1
        // grid_sample unnormalisation (align_corners=False) + border clamp
        let ix = clip(((gx + 1) * Float(w) - 1) / 2, min: 0, max: Float(w - 1))
        let iy = clip(((gy + 1) * Float(h) - 1) / 2, min: 0, max: Float(h - 1))
        let x0 = floor(ix), y0 = floor(iy)
        let x1 = minimum(x0 + 1, Float(w - 1)), y1 = minimum(y0 + 1, Float(h - 1))
        let wx = (ix - x0).expandedDimensions(axis: 1), wy = (iy - y0).expandedDimensions(axis: 1)
        let flat = features.reshaped(h * w, -1)
        func at(_ yy: MLXArray, _ xx: MLXArray) -> MLXArray { flat[(yy * Float(w) + xx).asType(.int32)] }
        let top = at(y0, x0) * (1 - wx) + at(y0, x1) * wx
        let bot = at(y1, x0) * (1 - wx) + at(y1, x1) * wx
        return top * (1 - wy) + bot * wy
    }
}
