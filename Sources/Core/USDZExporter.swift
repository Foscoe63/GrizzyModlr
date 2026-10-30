import Foundation
import SceneKit

/// USDZ export (Apple AR Quick Look). Builds a one-node SceneKit scene from the mesh and
/// lets SceneKit's USD writer pack it, embedding the base color and, when present, the
/// metallic-roughness maps.
enum USDZExporter {
    static func write(_ mesh: MeshExporter.MeshData, to dest: URL) throws {
        let vertices = SCNGeometrySource(
            data: mesh.rawVerts, semantic: .vertex, vectorCount: mesh.vertCount,
            usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4,
            dataOffset: 0, dataStride: 12)
        let normals = SCNGeometrySource(
            data: mesh.rawNormals, semantic: .normal, vectorCount: mesh.vertCount,
            usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4,
            dataOffset: 0, dataStride: 12)
        var sources = [vertices, normals]
        if let uvData = mesh.rawUVs {
            sources.append(SCNGeometrySource(
                data: uvData, semantic: .texcoord, vectorCount: mesh.vertCount,
                usesFloatComponents: true, componentsPerVector: 2, bytesPerComponent: 4,
                dataOffset: 0, dataStride: 8))
        }
        let element = SCNGeometryElement(
            data: mesh.rawFaces, primitiveType: .triangles, primitiveCount: mesh.faceCount,
            bytesPerIndex: 4)
        let geometry = SCNGeometry(sources: sources, elements: [element])

        let material = SCNMaterial()
        material.lightingModel = .physicallyBased
        material.isDoubleSided = true
        if let png = mesh.texturePNG, mesh.uvs != nil, let image = image(png) {
            material.diffuse.contents = image
        } else {
            material.diffuse.contents = CGColor(gray: 0.8, alpha: 1)
        }
        if let png = mesh.metallicRoughnessPNG, mesh.uvs != nil, let image = image(png) {
            // glTF packs roughness in G and metallic in B; SceneKit samples one channel each.
            material.roughness.contents = image
            material.metalness.contents = image
        } else {
            material.roughness.contents = 0.6
            material.metalness.contents = 0.0
        }
        geometry.materials = [material]

        let scene = SCNScene()
        scene.rootNode.addChildNode(SCNNode(geometry: geometry))
        if FileManager.default.fileExists(atPath: dest.path) {
            try FileManager.default.removeItem(at: dest)
        }
        guard scene.write(to: dest, options: nil, delegate: nil, progressHandler: nil),
              FileManager.default.fileExists(atPath: dest.path)
        else { throw MeshExporter.ExportError.unreadable }
    }

    private static func image(_ png: Data) -> CGImage? {
        guard let src = CGImageSourceCreateWithData(png as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }
}
