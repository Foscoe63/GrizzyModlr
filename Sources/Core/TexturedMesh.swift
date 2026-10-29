import Foundation
import simd

/// Parsed `.tmesh`: `[i32 nV][i32 nF][f32 verts 3nV][f32 normals 3nV][f32 uvs 2nV][i32 faces 3nF]`.
struct TexturedMesh {
    var positions: [SIMD3<Float>]
    var normals: [SIMD3<Float>]
    var uvs: [SIMD2<Float>]
    var faces: [SIMD3<UInt32>]

    static func parse(_ data: Data) -> TexturedMesh? {
        guard data.count >= 8 else { return nil }
        let n = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: Int32.self) })
        let m = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: Int32.self) })
        guard n > 0, m > 0, data.count >= 8 + n * 32 + m * 12 else { return nil }
        func floats(_ offset: Int, _ count: Int) -> [Float] {
            data.withUnsafeBytes { raw in
                (0..<count).map { raw.loadUnaligned(fromByteOffset: offset + $0 * 4, as: Float.self) }
            }
        }
        let v = floats(8, n * 3), nr = floats(8 + n * 12, n * 3), uv = floats(8 + n * 24, n * 2)
        let fi: [UInt32] = data.withUnsafeBytes { raw in
            (0..<(m * 3)).map { raw.loadUnaligned(fromByteOffset: 8 + n * 32 + $0 * 4, as: UInt32.self) }
        }
        guard fi.allSatisfy({ Int($0) < n }) else { return nil }
        return TexturedMesh(
            positions: (0..<n).map { SIMD3(v[$0 * 3], v[$0 * 3 + 1], v[$0 * 3 + 2]) },
            normals: (0..<n).map { SIMD3(nr[$0 * 3], nr[$0 * 3 + 1], nr[$0 * 3 + 2]) },
            uvs: (0..<n).map { SIMD2(uv[$0 * 2], uv[$0 * 2 + 1]) },
            faces: (0..<m).map { SIMD3(fi[$0 * 3], fi[$0 * 3 + 1], fi[$0 * 3 + 2]) })
    }

    func serialize() -> Data {
        var d = Data()
        d.reserveCapacity(8 + positions.count * 32 + faces.count * 12)
        func i32(_ v: Int32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
        func f32(_ v: Float) { var x = v.bitPattern.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
        i32(Int32(positions.count)); i32(Int32(faces.count))
        for p in positions { f32(p.x); f32(p.y); f32(p.z) }
        for n in normals { f32(n.x); f32(n.y); f32(n.z) }
        for t in uvs { f32(t.x); f32(t.y) }
        for f in faces { i32(Int32(f.x)); i32(Int32(f.y)); i32(Int32(f.z)) }
        return d
    }

    /// Texels per world unit around a triangle (for sizing a brush in texture space).
    func texelsPerUnit(face: Int, textureWidth: Int) -> Float {
        guard face >= 0, face < faces.count else { return 0 }
        let f = faces[face]
        let idx = [Int(f.x), Int(f.y), Int(f.z)]
        var ratio: Float = 0, count: Float = 0
        for k in 0..<3 {
            let a = idx[k], b = idx[(k + 1) % 3]
            let world = simd_length(positions[a] - positions[b])
            let uv = simd_length(uvs[a] - uvs[b]) * Float(textureWidth)
            if world > 1e-9 { ratio += uv / world; count += 1 }
        }
        return count > 0 ? ratio / count : 0
    }
}
