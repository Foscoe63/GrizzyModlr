import Foundation
import simd

/// 3MF export (3D printing). A 3MF file is a zip holding `3D/3dmodel.model` (XML); entries
/// are written uncompressed ("stored"), which every reader accepts. Geometry only: slicers
/// ignore texture, and 3MF's base-color extension isn't read by the common ones.
enum ThreeMFExporter {
    /// `maxDimensionMM > 0` scales the model so its longest bounding-box side has that length
    /// (generated meshes are unit-sized, which a slicer would otherwise read as 1 mm).
    static func encode(_ mesh: MeshExporter.MeshData, maxDimensionMM: Double = 0) -> Data {
        var scale: Float = 1
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for i in 0..<mesh.vertCount {
            let p = SIMD3(mesh.verts[i * 3], mesh.verts[i * 3 + 1], mesh.verts[i * 3 + 2])
            lo = simd_min(lo, p); hi = simd_max(hi, p)
        }
        let extent = max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
        if maxDimensionMM > 0, extent > 0, extent.isFinite { scale = Float(maxDimensionMM) / extent }
        // Slicers want the model sitting on the bed with +Z up; ours is Y-up. Rotate Y→Z, rest on z = 0.
        func point(_ i: Int) -> SIMD3<Float> {
            let x = mesh.verts[i * 3], y = mesh.verts[i * 3 + 1], z = mesh.verts[i * 3 + 2]
            return SIMD3((x - lo.x) * scale, -(z - lo.z) * scale, (y - lo.y) * scale)
        }

        var xml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        xml += "<model unit=\"millimeter\" xml:lang=\"en-US\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n"
        xml += " <metadata name=\"Application\">GrizzyModlr</metadata>\n"
        xml += " <resources>\n  <object id=\"1\" type=\"model\">\n   <mesh>\n    <vertices>\n"
        xml.reserveCapacity(mesh.vertCount * 60 + mesh.faceCount * 40)
        for i in 0..<mesh.vertCount {
            let p = point(i)
            xml += "     <vertex x=\"\(p.x)\" y=\"\(p.y)\" z=\"\(p.z)\"/>\n"
        }
        xml += "    </vertices>\n    <triangles>\n"
        for f in 0..<mesh.faceCount {
            // The Y→Z rotation above is proper (det +1), so winding is preserved.
            xml += "     <triangle v1=\"\(mesh.indices[f * 3])\" v2=\"\(mesh.indices[f * 3 + 1])\" v3=\"\(mesh.indices[f * 3 + 2])\"/>\n"
        }
        xml += "    </triangles>\n   </mesh>\n  </object>\n </resources>\n"
        xml += " <build>\n  <item objectid=\"1\"/>\n </build>\n</model>\n"

        let types = """
        <?xml version="1.0" encoding="UTF-8"?>
        <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
         <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
         <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
        </Types>
        """
        let rels = """
        <?xml version="1.0" encoding="UTF-8"?>
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
         <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
        </Relationships>
        """
        return Zip.stored([("[Content_Types].xml", Data(types.utf8)),
                           ("_rels/.rels", Data(rels.utf8)),
                           ("3D/3dmodel.model", Data(xml.utf8))])
    }
}

/// Minimal zip writer: uncompressed entries, no zip64 (fine below 4 GB).
enum Zip {
    static func stored(_ entries: [(name: String, data: Data)]) -> Data {
        var out = Data()
        var central = Data()
        for (name, data) in entries {
            let nameBytes = Data(name.utf8)
            let crc = crc32(data)
            let offset = UInt32(out.count)
            out.le(UInt32(0x04034b50)); out.le(UInt16(20)); out.le(UInt16(0)); out.le(UInt16(0))
            out.le(UInt16(0)); out.le(UInt16(0x21))                       // time, date (1980-01-01)
            out.le(crc); out.le(UInt32(data.count)); out.le(UInt32(data.count))
            out.le(UInt16(nameBytes.count)); out.le(UInt16(0))
            out.append(nameBytes); out.append(data)
            central.le(UInt32(0x02014b50)); central.le(UInt16(20)); central.le(UInt16(20))
            central.le(UInt16(0)); central.le(UInt16(0)); central.le(UInt16(0)); central.le(UInt16(0x21))
            central.le(crc); central.le(UInt32(data.count)); central.le(UInt32(data.count))
            central.le(UInt16(nameBytes.count)); central.le(UInt16(0)); central.le(UInt16(0))
            central.le(UInt16(0)); central.le(UInt16(0)); central.le(UInt32(0)); central.le(offset)
            central.append(nameBytes)
        }
        let centralOffset = UInt32(out.count)
        out.append(central)
        out.le(UInt32(0x06054b50)); out.le(UInt16(0)); out.le(UInt16(0))
        out.le(UInt16(entries.count)); out.le(UInt16(entries.count))
        out.le(UInt32(central.count)); out.le(centralOffset); out.le(UInt16(0))
        return out
    }

    static func crc32(_ data: Data) -> UInt32 {
        var c: UInt32 = 0xFFFF_FFFF
        for b in data { c = table[Int((c ^ UInt32(b)) & 0xFF)] ^ (c >> 8) }
        return c ^ 0xFFFF_FFFF
    }

    private static let table: [UInt32] = (0..<256).map { n in
        var c = UInt32(n)
        for _ in 0..<8 { c = c & 1 == 1 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        return c
    }
}

private extension Data {
    mutating func le<T: FixedWidthInteger>(_ v: T) {
        var x = v.littleEndian
        Swift.withUnsafeBytes(of: &x) { append(contentsOf: $0) }
    }
}
