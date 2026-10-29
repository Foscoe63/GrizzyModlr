import SwiftUI

/// Edits UV islands of a painted version: pick an island on the layout, then drag to
/// move it or use the sliders to scale and rotate it. Applying re-maps the albedo (and
/// metallic-roughness map) so the surface looks the same, and saves a new version.
@MainActor @Observable
final class UVEditorModel {
    let mesh: TexturedMesh
    let islands: [UVLayout.Island]
    let texture: NSImage?
    let outlines: [Int: [(SIMD2<Float>, SIMD2<Float>)]]
    var selected: Int?
    var transforms: [Int: IslandTransform] = [:]

    init(mesh: TexturedMesh, albedo: URL) {
        self.mesh = mesh
        let isl = UVLayout.islands(faces: mesh.faces, uvs: mesh.uvs)
        islands = isl
        texture = NSImage(contentsOf: albedo)
        outlines = Dictionary(uniqueKeysWithValues: isl.map { i in
            (i.id, UVLayout.outline(of: i, faces: mesh.faces).map { (mesh.uvs[$0.0], mesh.uvs[$0.1]) })
        })
    }

    var editedUVs: [SIMD2<Float>] { UVLayout.apply(transforms, islands: islands, to: mesh.uvs) }
    var changed: Set<Int> { Set(transforms.filter { !$0.value.isIdentity }.keys) }
    var isEdited: Bool { !changed.isEmpty }

    var warning: String? {
        guard isEdited else { return nil }
        let uvs = editedUVs
        if UVLayout.leavesUnitSquare(uvs, islands: islands, changed: changed) { return "An island is outside the texture." }
        if UVLayout.overlapsOthers(faces: mesh.faces, uvs: uvs, islands: islands, changed: changed) {
            return "Islands overlap; one will overwrite the other."
        }
        return nil
    }

    func transform(_ id: Int) -> IslandTransform { transforms[id] ?? IslandTransform() }
    func update(_ id: Int, _ edit: (inout IslandTransform) -> Void) {
        var t = transform(id); edit(&t); transforms[id] = t.isIdentity ? nil : t
    }

    /// The island whose triangles contain a UV point (frontmost = last), if any.
    func hit(_ p: SIMD2<Float>) -> Int? {
        let uvs = editedUVs
        for island in islands.reversed() where p.x >= island.lo.x - 0.5 && p.x <= island.hi.x + 0.5 {
            for fi in island.faces {
                let f = mesh.faces[fi]
                let a = uvs[Int(f.x)], b = uvs[Int(f.y)], c = uvs[Int(f.z)]
                let d1 = (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y)
                let d2 = (p.x - c.x) * (b.y - c.y) - (b.x - c.x) * (p.y - c.y)
                let d3 = (p.x - a.x) * (c.y - a.y) - (c.x - a.x) * (p.y - a.y)
                if !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0)) { return island.id }
            }
        }
        return nil
    }
}

struct UVEditorView: View {
    @Environment(ProjectStore.self) private var store
    @State var model: UVEditorModel
    let projectID: Project.ID
    var onClose: () -> Void

    @State private var busy = false
    @State private var message: String?
    @State private var dragStart: (uv: SIMD2<Float>, du: Float, dv: Float)?

    private let side: CGFloat = 520

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            canvas
            controls.frame(width: 240)
        }
        .padding(16)
        .disabled(busy)
    }

    private var canvas: some View {
        ZStack {
            Color.black.opacity(0.25)
            if let img = model.texture {
                Image(nsImage: img).resizable().frame(width: side, height: side).opacity(0.85)
            }
            Canvas { ctx, size in
                for island in model.islands {
                    let t = model.transform(island.id)
                    let isSel = model.selected == island.id
                    var path = Path()
                    for (a, b) in model.outlines[island.id] ?? [] {
                        path.move(to: point(a, island: island, t: t, size: size))
                        path.addLine(to: point(b, island: island, t: t, size: size))
                    }
                    ctx.stroke(path, with: .color(isSel ? .yellow : (t.isIdentity ? .white.opacity(0.55) : .cyan)),
                               lineWidth: isSel ? 2 : 1)
                }
            }
            .frame(width: side, height: side)
        }
        .frame(width: side, height: side)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(.white.opacity(0.15)))
        .contentShape(Rectangle())
        .gesture(DragGesture(minimumDistance: 0)
            .onChanged { value in
                let uv = SIMD2(Float(value.startLocation.x / side), Float(value.startLocation.y / side))
                if dragStart == nil {
                    if let hit = model.hit(uv) { model.selected = hit }
                    guard let sel = model.selected else { return }
                    let t = model.transform(sel)
                    dragStart = (uv, t.du, t.dv)
                }
                guard let start = dragStart, let sel = model.selected else { return }
                let dx = Float((value.location.x - value.startLocation.x) / side)
                let dy = Float((value.location.y - value.startLocation.y) / side)
                model.update(sel) { $0.du = start.du + dx; $0.dv = start.dv + dy }
            }
            .onEnded { _ in dragStart = nil })
    }

    /// Transformed screen point of an original UV in `island`.
    private func point(_ uv: SIMD2<Float>, island: UVLayout.Island, t: IslandTransform, size: CGSize) -> CGPoint {
        let c = island.center
        let cs = cos(t.rotation), sn = sin(t.rotation)
        let p = (uv - c) * t.scale
        let q = c + SIMD2(p.x * cs - p.y * sn, p.x * sn + p.y * cs) + SIMD2(t.du, t.dv)
        return CGPoint(x: CGFloat(q.x) * size.width, y: CGFloat(q.y) * size.height)
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("UV editor").font(.headline)
            Text("\(model.islands.count) islands · click one, then drag to move")
                .font(.caption).foregroundStyle(.secondary)
            if let sel = model.selected {
                let t = model.transform(sel)
                row("Scale", String(format: "×%.2f", t.scale)) {
                    Slider(value: Binding(get: { Double(t.scale) }, set: { v in model.update(sel) { $0.scale = Float(v) } }),
                           in: 0.25...2)
                }
                row("Rotate", String(format: "%.0f°", t.rotation * 180 / .pi)) {
                    Slider(value: Binding(get: { Double(t.rotation) * 180 / .pi },
                                          set: { v in model.update(sel) { $0.rotation = Float(v * .pi / 180) } }),
                           in: -180...180, step: 5)
                }
                Button("Reset island") { model.transforms[sel] = nil }.disabled(t.isIdentity)
            } else {
                Text("No island selected").font(.callout).foregroundStyle(.secondary)
            }
            if let w = model.warning {
                Label(w, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
            }
            if let message { Text(message).font(.caption).foregroundStyle(.secondary) }
            Spacer()
            HStack {
                Button("Close", action: onClose)
                Spacer()
                Button {
                    busy = true; message = nil
                    Task {
                        let err = await store.applyUVEdit(model.transforms, for: projectID)
                        busy = false
                        if err == nil { onClose() } else { message = err }
                    }
                } label: { Label("Apply", systemImage: "checkmark") }
                .buttonStyle(.borderedProminent)
                .disabled(!model.isEdited || model.warning?.hasPrefix("An island is outside") == true)
            }
            Text("Applying re-maps the texture so the model looks the same, and saves a new version.")
                .font(.caption2).foregroundStyle(.tertiary)
        }
    }

    private func row<C: View>(_ title: String, _ value: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack { Text(title).font(.callout.weight(.medium)); Spacer()
                Text(value).font(.caption).foregroundStyle(.secondary).monospacedDigit() }
            content()
        }
    }
}
