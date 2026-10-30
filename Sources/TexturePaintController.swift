import SwiftUI
import simd

/// Drives a texture-painting session on a painted version: the editable albedo canvas,
/// the mesh (for UVs), and brush settings. The viewer reads it on every dab.
@MainActor @Observable
final class TexturePaintController {
    var canvas: TextureCanvas
    let mesh: TexturedMesh
    var tool: PaintTool = .paint {
        didSet { if oldValue == .aiFill, tool != .aiFill { discardFillSelection() } }
    }
    var color: Color = .red
    /// Brush radius as a fraction of the model's bounding radius.
    var radiusFraction: Double = ToolDefaults.paintSize()
    var hardness: Double = ToolDefaults.paintHardness()
    var opacity: Double = ToolDefaults.paintOpacity()
    private(set) var revision = 0

    // AI Fill: the brush marks a selection (tinted on the canvas, remembered here); Fill then
    // regenerates it. `fillOriginal` is the texture without the tint.
    var fillPrompt = ToolDefaults.aiFillPrompt()
    private(set) var fillMask: [Float] = []
    @ObservationIgnored private var fillOriginal: [UInt8]?
    var hasFillSelection: Bool { _ = revision; return fillOriginal != nil && fillMask.contains { $0 > 0 } }

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
        case .aiFill:
            markFill(uv: uv, radius: radius)
        }
    }

    private func markFill(uv: SIMD2<Float>, radius: Float) {
        if fillOriginal == nil {
            fillOriginal = canvas.pixels
            fillMask = [Float](repeating: 0, count: canvas.width * canvas.height)
        }
        let (w, h) = (canvas.width, canvas.height)
        let cx = uv.x * Float(w), cy = uv.y * Float(h)
        let x0 = max(0, Int(cx - radius)), x1 = min(w - 1, Int(cx + radius + 1))
        let y0 = max(0, Int(cy - radius)), y1 = min(h - 1, Int(cy + radius + 1))
        guard x0 <= x1, y0 <= y1 else { return }
        for y in y0...y1 { for x in x0...x1 {
            let dx = Float(x) + 0.5 - cx, dy = Float(y) + 0.5 - cy
            if dx * dx + dy * dy < radius * radius { fillMask[y * w + x] = 1 }
        } }
        canvas.paintUntracked(u: uv.x, v: uv.y, radius: radius, color: SIMD4(255, 40, 200, 255),
                              hardness: 1, opacity: 0.45)
    }

    /// Clears the selection and removes its tint.
    func discardFillSelection() {
        if let original = fillOriginal { canvas.resetPixels(original) }
        fillOriginal = nil
        fillMask = []
        touch()
    }

    /// Everything the generator needs: the untinted texture and the selection, or nil.
    func fillRequest() -> (pixels: [UInt8], mask: [Float], width: Int, height: Int)? {
        guard let original = fillOriginal, fillMask.contains(where: { $0 > 0 }) else { return nil }
        return (original, fillMask, canvas.width, canvas.height)
    }

    /// Installs a generated fill as one undoable edit and clears the selection.
    func applyFill(_ pixels: [UInt8]) {
        if let original = fillOriginal { canvas.resetPixels(original) }
        canvas.replaceUndoable(pixels)
        fillOriginal = nil
        fillMask = []
        touch()
    }
}
