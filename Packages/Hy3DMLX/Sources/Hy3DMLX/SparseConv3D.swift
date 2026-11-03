import Foundation
import MLX

/// Sparse voxel tensor for TRELLIS.2: `feats` is [N, C], `coords` is int32 [N, 4] as (batch, x, y, z).
struct SparseVoxels {
    var feats: MLXArray
    var coords: MLXArray
    /// Edge length of the (cubic) lattice the coords live in, and the batch size.
    var size: Int
    var batch: Int

    var count: Int { coords.dim(0) }
}

/// Neighbour table for a submanifold conv: row i, column k is the index of the active voxel at
/// `coords[i] + (k - 1) * dilation` in kernel order (kx, ky, kz row-major), or -1 where there is none.
/// Built from a dense index lookup over batch*size³ cells, so it suits lattices up to ~512³; beyond that
/// swap the lookup for a sorted-key search (see docs/research/trellis2-feasibility.md).
func submanifoldNeighbors(coords: MLXArray, size: Int, batch: Int, dilation: Int = 1) -> MLXArray {
    let n = coords.dim(0)
    let cells = batch * size * size * size
    var lookup = MLXArray.full([cells], values: MLXArray(Int32(-1)), type: Int32.self)
    let c = coords.asType(Int32.self)
    let b = c[0..., 0], x = c[0..., 1], y = c[0..., 2], z = c[0..., 3]
    let s = Int32(size)
    lookup[((b * s + x) * s + y) * s + z] = MLXArray(0 ..< Int32(n))

    var columns: [MLXArray] = []
    for kx in -1 ... 1 { for ky in -1 ... 1 { for kz in -1 ... 1 {
        let nx = x + Int32(kx * dilation), ny = y + Int32(ky * dilation), nz = z + Int32(kz * dilation)
        let inside = (nx .>= 0) .&& (nx .< s) .&& (ny .>= 0) .&& (ny .< s) .&& (nz .>= 0) .&& (nz .< s)
        let key = ((b * s + nx) * s + ny) * s + nz
        let safe = MLX.where(inside, key, MLXArray(Int32(0)))
        columns.append(MLX.where(inside, lookup[safe], MLXArray(Int32(-1))))
    } } }
    return stacked(columns, axis: 1)
}

/// Submanifold sparse 3×3×3 convolution: output sites are exactly the input sites. `weight` is
/// TRELLIS.2's layout [Co, 3, 3, 3, Ci]; out[i] = bias + Σ_k W[:, k, :] · feats[neighbor(i, k)].
/// One gather + matmul per kernel tap so peak memory stays at O(N·C) rather than O(27·N·C).
func submanifoldConv3d(_ x: SparseVoxels, weight: MLXArray, bias: MLXArray?, neighbors: MLXArray) -> MLXArray {
    let co = weight.dim(0), ci = weight.dim(4)
    let taps = weight.reshaped(co, 27, ci)
    let n = x.count
    // A zero row at index n absorbs the -1 (missing neighbour) entries.
    let padded = concatenated([x.feats, MLXArray.zeros([1, ci], dtype: x.feats.dtype)], axis: 0)
    let idx = MLX.where(neighbors .< 0, MLXArray(Int32(n)), neighbors)
    var out: MLXArray? = nil
    for k in 0 ..< 27 {
        let part = matmul(padded[idx[0..., k]], taps[0..., k, 0...].transposed())
        out = out.map { $0 + part } ?? part
    }
    guard let o = out else { return MLXArray.zeros([n, co]) }
    return bias.map { o + $0 } ?? o
}

/// Spatial → channel (the strided half of TRELLIS.2's down blocks): voxels sharing a `factor`³ parent are
/// merged into one voxel whose features are the children's concatenated in `sub = Σ (c % f)·f^axis` slots
/// (absent children are zero). Returns the parent coords/features plus what `channelToSpatial` needs.
struct SpatialToChannel {
    var coords: MLXArray      // [M, 4] parent coords
    var feats: MLXArray       // [M, f³·C]
    var parent: MLXArray      // [N] index of each child's parent row
    var slot: MLXArray        // [N] child slot inside the parent
}

func spatialToChannel(_ x: SparseVoxels, factor: Int = 2) -> SpatialToChannel {
    let f = Int32(factor)
    let c = x.coords.asType(Int32.self)
    let parentSize = Int32((x.size + factor - 1) / factor)
    let px = floorDivide(c[0..., 1], f), py = floorDivide(c[0..., 2], f), pz = floorDivide(c[0..., 3], f)
    let slot = (c[0..., 1] % f) + (c[0..., 2] % f) * f + (c[0..., 3] % f) * f * f
    let code = ((c[0..., 0] * parentSize + px) * parentSize + py) * parentSize + pz
    // unique + inverse via sort: codes are small enough for Int32 up to 512³ lattices at batch 1
    let order = argSort(code)
    let sorted = code[order]
    let isNew = concatenated([MLXArray([true]), sorted[1...] .!= sorted[..<(sorted.dim(0) - 1)]], axis: 0)
    let rank = cumsum(isNew.asType(Int32.self)) - 1
    let m = Int(rank[rank.dim(0) - 1].item(Int32.self)) + 1
    var parent = MLXArray.zeros([code.dim(0)], dtype: .int32)
    parent[order] = rank
    let firstOfGroup = which(isNew)
    let uniqueCodes = sorted[firstOfGroup]
    let ux = floorDivide(uniqueCodes, parentSize * parentSize) % parentSize
    let uy = floorDivide(uniqueCodes, parentSize) % parentSize
    let uz = uniqueCodes % parentSize
    let ub = floorDivide(uniqueCodes, parentSize * parentSize * parentSize)
    let newCoords = stacked([ub, ux, uy, uz], axis: 1)
    let cdim = x.feats.dim(1), f3 = factor * factor * factor
    var grid = MLXArray.zeros([m * f3, cdim], dtype: x.feats.dtype)
    grid[parent * Int32(f3) + slot] = x.feats
    return SpatialToChannel(coords: newCoords, feats: grid.reshaped(m, f3 * cdim), parent: parent, slot: slot)
}

/// Channel → spatial: the inverse gather, restoring the original children.
func channelToSpatial(_ feats: MLXArray, parent: MLXArray, slot: MLXArray, factor: Int = 2) -> MLXArray {
    let f3 = factor * factor * factor
    let rows = feats.reshaped(feats.dim(0) * f3, -1)
    return rows[parent * Int32(f3) + slot]
}

private func which(_ mask: MLXArray) -> MLXArray {
    // indices of true entries (host round trip; N is small relative to the conv cost)
    let m = mask.asArray(Bool.self)
    return MLXArray(m.enumerated().compactMap { $1 ? Int32($0) : nil })
}
