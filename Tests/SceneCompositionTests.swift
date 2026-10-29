import XCTest
import simd

final class SceneCompositionTests: XCTestCase {
    private func box(center: SIMD3<Float>, half: Float) -> ([SIMD3<Float>], [SIMD3<UInt32>]) {
        let s: [SIMD3<Float>] = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
        let f: [SIMD3<UInt32>] = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[2,3,7],[2,7,6],[1,2,6],[1,6,5],[3,0,4],[3,4,7]]
        return (s.map { center + $0 * half }, f)
    }
    private func item(x: Double = 0, y: Double = 0, rot: Double = 0, scale: Double = 1) -> SceneItem {
        SceneItem(projectID: UUID(), generationID: UUID(), x: x, y: y, z: 0, rotationY: rot, scale: scale)
    }

    func testNormalizationMakesAnyModelUnitRadiusAtItsPosition() {
        let (p, f) = box(center: [10, 20, 30], half: 5)
        let merged = SceneComposition.merge([(p, f, item(x: 3))])
        let (c, r) = SceneComposition.bounds(merged.positions)
        XCTAssertEqual(c.x, 3, accuracy: 1e-4); XCTAssertEqual(c.y, 0, accuracy: 1e-4); XCTAssertEqual(c.z, 0, accuracy: 1e-4)
        XCTAssertEqual(r, 1, accuracy: 1e-4)
    }

    func testScaleAndRotationApplyAboutTheObjectCenter() {
        let (p, f) = box(center: .zero, half: 1)
        let big = SceneComposition.merge([(p, f, item(scale: 2))])
        XCTAssertEqual(SceneComposition.bounds(big.positions).radius, 2, accuracy: 1e-4)
        // A 90° turn about Y maps the box's +X corner to −Z (counter-clockwise seen from above).
        let turned = SceneComposition.merge([(p, f, item(rot: 90))])
        let corner = turned.positions[1]                                  // originally (1,-1,-1)/√3
        XCTAssertEqual(corner.x, -1 / Float(3).squareRoot(), accuracy: 1e-4)
        XCTAssertEqual(corner.z, -1 / Float(3).squareRoot(), accuracy: 1e-4)
    }

    func testMergeReindexesFacesAndKeepsAllGeometry() {
        let (p, f) = box(center: .zero, half: 1)
        let m = SceneComposition.merge([(p, f, item(x: -2)), (p, f, item(x: 2))])
        XCTAssertEqual(m.positions.count, 16)
        XCTAssertEqual(m.faces.count, 24)
        XCTAssertTrue(m.faces.suffix(12).allSatisfy { $0.x >= 8 && $0.y >= 8 && $0.z >= 8 })
        let xs = m.positions.map(\.x)
        XCTAssertEqual(xs.min()! + xs.max()!, 0, accuracy: 1e-4)             // centered between the two
        // each box is normalized to a unit bounding sphere (half-width 1/√3), placed at x = ±2
        XCTAssertEqual(xs.max()!, 2 + 1 / Float(3).squareRoot(), accuracy: 1e-3)
    }

    func testNextSlotAndCodableRoundTrip() throws {
        XCTAssertEqual(SceneComposition.nextSlotX(existing: [], spacing: 2.5), 0)
        XCTAssertEqual(SceneComposition.nextSlotX(existing: [item(x: 0), item(x: 2.5)], spacing: 2.5), 5)
        let it = item(x: 1, y: 2, rot: 30, scale: 1.5)
        XCTAssertEqual(try JSONDecoder().decode(SceneItem.self, from: JSONEncoder().encode(it)), it)
    }
}
