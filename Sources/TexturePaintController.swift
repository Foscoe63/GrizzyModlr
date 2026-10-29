import SwiftUI
import simd

/// Drives a texture-painting session on a painted version: the editable albedo canvas,
/// the mesh (for UVs), and brush settings. The viewer reads it on every dab.
@MainActor @Observable
final class TexturePaintController {
    var canvas: TextureCanvas
    let mesh: TexturedMesh
    var tool: PaintTool = .paint
    var color: Color = .red
    /// Brush radius as a fraction of the model's bounding radius.
    var radiusFraction: Double = ToolDefaults.paintSize()
    var hardness: Double = ToolDefaults.paintHardness()
    var opacity: Double = ToolDefaults.paintOpacity()
    private(set) var revision = 0

    init(canvas: TextureCanvas, mesh: TexturedMesh) {
        self.canvas = canvas
        self.mesh = mesh
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for p in mesh.positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        boundsRadius = max(simd_length(hi - lo) / 2, 1e-4)
    }

    var canUndo: Bool { _ = revision; return canvas.canUndo }
    var isEdited: Bool { _ = revision; return canvas.isEdited }
    func touch() { revision += 1 }
    func undo() { canvas.undo(); touch() }

    /// Bounding radius of the mesh, to scale the brush.
    @ObservationIgnored let boundsRadius: Float
    var worldRadius: Float { Float(radiusFraction) * boundsRadius }

    var rgba: SIMD4<Float> {
        let c = NSColor(color).usingColorSpace(.deviceRGB) ?? .red
        return SIMD4(Float(c.redComponent) * 255, Float(c.greenComponent) * 255, Float(c.blueComponent) * 255, 255)
    }

    /// Applies the active tool at a surface hit.
    func dab(uv: SIMD2<Float>, face: Int) {
        let texels = mesh.texelsPerUnit(face: face, textureWidth: canvas.width)
        let radius = max(worldRadius * texels, 1.5)
        switch tool {
        case .paint:
            canvas.paint(u: uv.x, v: uv.y, radius: radius, color: rgba,
                         hardness: Float(hardness), opacity: Float(opacity))
        case .soften:
            canvas.soften(u: uv.x, v: uv.y, radius: radius, strength: Float(opacity) * 0.6)
        case .heal:
            canvas.heal(u: uv.x, v: uv.y, radius: radius)
        }
    }
}
