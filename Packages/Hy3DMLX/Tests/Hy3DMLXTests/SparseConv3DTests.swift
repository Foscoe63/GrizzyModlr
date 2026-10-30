import XCTest
import MLX
@testable import Hy3DMLX

/// Parity with the PyTorch reference (scripts/trellis2/make_sparse_conv_fixture.py): dense conv3d masked
/// to the active sites for the submanifold conv, and TRELLIS.2's own Spatial2Channel code for the rest.
final class SparseConv3DTests: XCTestCase {
    private struct Fixture: Decodable {
        let batch, size, cin, cout, n: Int
        let coords: [Int32]
        let feats, weight, bias, out_dil1, out_dil2: [Float]
        let s2c_n: Int
        let s2c_coords: [Int32]
        let s2c_feats: [Float]
    }

    private func load() throws -> Fixture {
        let url = Bundle.module.resourceURL!.appendingPathComponent("Fixtures/sparse_conv_fixture.json")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: [Float]) -> Float {
        let x = a.asArray(Float.self)
        XCTAssertEqual(x.count, b.count)
        return zip(x, b).map { abs($0 - $1) }.max() ?? 0
    }

    private func voxels(_ f: Fixture) -> SparseVoxels {
        SparseVoxels(feats: MLXArray(f.feats, [f.n, f.cin]), coords: MLXArray(f.coords, [f.n, 4]),
                     size: f.size, batch: f.batch)
    }

    func testSubmanifoldConvMatchesTorch() throws {
        let f = try load()
        let x = voxels(f)
        let w = MLXArray(f.weight, [f.cout, 3, 3, 3, f.cin]), b = MLXArray(f.bias)
        for (dil, ref) in [(1, f.out_dil1), (2, f.out_dil2)] {
            let nbr = submanifoldNeighbors(coords: x.coords, size: f.size, batch: f.batch, dilation: dil)
            let out = submanifoldConv3d(x, weight: w, bias: b, neighbors: nbr)
            eval(out)
            XCTAssertLessThan(maxAbsDiff(out, ref), 1e-4, "dilation \(dil)")
        }
    }

    func testSpatialToChannelMatchesTorchAndRoundTrips() throws {
        let f = try load()
        let x = voxels(f)
        let s = spatialToChannel(x)
        eval(s.coords, s.feats)
        XCTAssertEqual(s.coords.dim(0), f.s2c_n)
        XCTAssertEqual(s.coords.asArray(Int32.self), f.s2c_coords)
        XCTAssertLessThan(maxAbsDiff(s.feats, f.s2c_feats), 1e-6)
        let back = channelToSpatial(s.feats, parent: s.parent, slot: s.slot)
        XCTAssertLessThan(maxAbsDiff(back, f.feats), 1e-6)
    }

    /// Speed at decoder-like scale: 400k active voxels on a 256³ lattice, 128 → 128 channels.
    func testConvPerformanceAtDecoderScale() throws {
        let n = 400_000, size = 256, c = 128
        var seen = Set<Int32>(), pts: [Int32] = []
        var rng = SystemRandomNumberGenerator()
        while pts.count < 4 * n {
            // thin shell of a sphere so neighbourhoods are realistic
            let u = Float.random(in: 0 ..< 1, using: &rng), v = Float.random(in: 0 ..< 1, using: &rng)
            let th = 2 * Float.pi * u, ph = acos(2 * v - 1)
            let r = Float(100) + Float.random(in: -1.5 ... 1.5, using: &rng)
            let x = Int32(128 + r * sin(ph) * cos(th)), y = Int32(128 + r * sin(ph) * sin(th)), z = Int32(128 + r * cos(ph))
            let key = (x * 256 + y) * 256 + z
            if seen.insert(key).inserted { pts += [0, x, y, z] }
        }
        let coords = MLXArray(pts, [n, 4])
        let x = SparseVoxels(feats: MLXRandom.normal([n, c]), coords: coords, size: size, batch: 1)
        let w = MLXRandom.normal([c, 3, 3, 3, c]) * 0.02
        eval(x.feats, w)
        let t0 = Date()
        let nbr = submanifoldNeighbors(coords: coords, size: size, batch: 1)
        eval(nbr)
        let t1 = Date()
        let out = submanifoldConv3d(x, weight: w, bias: nil, neighbors: nbr)
        eval(out)
        let t2 = Date()
        print(String(format: "sparse conv perf: N=%d C=%d neighbor-map %.3fs, conv %.3fs", n, c,
                     t1.timeIntervalSince(t0), t2.timeIntervalSince(t1)))
        XCTAssertEqual(out.shape, [n, c])
    }
}
