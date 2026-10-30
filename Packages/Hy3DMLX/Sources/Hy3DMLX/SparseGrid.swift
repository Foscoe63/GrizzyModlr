import Foundation

let sparseBlockBits = 3
let sparseBlockEdge = 1 << sparseBlockBits
let sparseBlockCells = sparseBlockEdge * sparseBlockEdge * sparseBlockEdge

/// Block-sparse (n,n,n) array. 8³ blocks are allocated only where something is stored, so a thin
/// shell around a surface costs memory proportional to the shell, not to the whole cube — this is
/// what lets the shape decoder reach 2048³ lattices. Cells in an absent block read as `fill`.
struct SparseBlocks<T> {
    let n: Int
    let nb: Int
    let fill: T
    private(set) var table: [Int32]          // block index -> slot, -1 = absent
    private(set) var slotBlock: [Int32]      // slot -> block index
    var data: [T]                            // slot * 512 + offset

    var slots: Int { slotBlock.count }

    init(n: Int, fill: T) {
        self.n = n
        self.nb = (n + sparseBlockEdge - 1) >> sparseBlockBits
        self.fill = fill
        self.table = [Int32](repeating: -1, count: nb * nb * nb)
        self.slotBlock = []
        self.data = []
    }

    /// Same blocks as `other`, every cell set to `fill`.
    init<U>(layoutOf other: SparseBlocks<U>, fill: T) {
        n = other.n; nb = other.nb; self.fill = fill
        table = other.table; slotBlock = other.slotBlock
        data = [T](repeating: fill, count: other.slotBlock.count * sparseBlockCells)
    }

    @inline(__always) func blockIndex(_ i: Int, _ j: Int, _ k: Int) -> Int {
        (((i >> sparseBlockBits) * nb) + (j >> sparseBlockBits)) * nb + (k >> sparseBlockBits)
    }
    @inline(__always) static func offset(_ i: Int, _ j: Int, _ k: Int) -> Int {
        ((i & 7) << 6) | ((j & 7) << 3) | (k & 7)
    }
    @inline(__always) func slot(_ i: Int, _ j: Int, _ k: Int) -> Int { Int(table[blockIndex(i, j, k)]) }

    /// Lattice coordinates of a slot's block corner.
    @inline(__always) func origin(of slot: Int) -> (Int, Int, Int) {
        let b = Int(slotBlock[slot])
        return ((b / (nb * nb)) << sparseBlockBits, ((b / nb) % nb) << sparseBlockBits, (b % nb) << sparseBlockBits)
    }

    @discardableResult
    mutating func ensure(_ i: Int, _ j: Int, _ k: Int) -> Int {
        let b = blockIndex(i, j, k)
        let s = Int(table[b])
        if s >= 0 { return s }
        let ns = slotBlock.count
        table[b] = Int32(ns)
        slotBlock.append(Int32(b))
        data.append(contentsOf: repeatElement(fill, count: sparseBlockCells))
        return ns
    }

    @inline(__always) func get(_ i: Int, _ j: Int, _ k: Int) -> T {
        let s = slot(i, j, k)
        return s < 0 ? fill : data[s * sparseBlockCells + Self.offset(i, j, k)]
    }

    @inline(__always) mutating func set(_ i: Int, _ j: Int, _ k: Int, _ v: T) {
        let s = ensure(i, j, k)
        data[s * sparseBlockCells + Self.offset(i, j, k)] = v
    }
}

typealias SparseSDFGrid = SparseBlocks<Float>

extension SparseBlocks where T == Float {
    /// Dense x-slowest copy (tests, small grids). Absent cells are NaN.
    func toDense() -> [Float] {
        var out = [Float](repeating: .nan, count: n * n * n)
        for s in 0 ..< slots {
            let (bi, bj, bk) = origin(of: s)
            for a in 0 ..< sparseBlockEdge where bi + a < n {
                for b in 0 ..< sparseBlockEdge where bj + b < n {
                    for c in 0 ..< sparseBlockEdge where bk + c < n {
                        out[((bi + a) * n + bj + b) * n + bk + c] = data[s * sparseBlockCells + ((a << 6) | (b << 3) | c)]
                    }
                }
            }
        }
        return out
    }
}

extension SparseBlocks where T == Float {
    /// Dense x-slowest -> sparse with every block present.
    init(dense g: [Float], n: Int) {
        self.init(n: n, fill: .nan)
        for i in 0 ..< n { for j in 0 ..< n { for k in 0 ..< n { set(i, j, k, g[(i * n + j) * n + k]) } } }
    }
}

/// FlashVDM near-surface refinement over block-sparse storage. Same algorithm and results as the
/// dense `OctreeSDF.dense` reference; only voxels in the near-surface band are ever allocated.
enum OctreeSDF {
    typealias Decode = (_ pts: [Float], _ count: Int) -> [Float]

    static func levels(resolution R: Int, minResolution: Int) -> [Int] {
        var list: [Int] = []
        var r = R
        while r >= minResolution { list.append(r); r /= 2 }
        list.reverse()                                                   // [coarse ... fine]
        return list.isEmpty ? [R] : list
    }

    /// Coarse dense lattice coordinates, i slowest.
    private static func coarsePoints(_ n0: Int, bmin: Float, step: Float) -> [Float] {
        var coarse = [Float](repeating: 0, count: n0 * n0 * n0 * 3)
        var p = 0
        for i in 0 ..< n0 {
            let x = bmin + Float(i) * step
            for j in 0 ..< n0 {
                let y = bmin + Float(j) * step
                for k in 0 ..< n0 {
                    coarse[p] = x; coarse[p + 1] = y; coarse[p + 2] = bmin + Float(k) * step; p += 3
                }
            }
        }
        return coarse
    }

    /// The old dense algorithm, kept as the reference (and used for the small preview grids).
    static func dense(resolution R: Int, bound: Float, minResolution: Int, mcLevel: Float,
                      decode: Decode) -> (grid: [Float], n: Int) {
        let bmin = -bound, bsize = 2 * bound
        let resList = levels(resolution: R, minResolution: minResolution)
        let r0 = resList[0], n0 = r0 + 1
        var grid = decode(coarsePoints(n0, bmin: bmin, step: bsize / Float(r0)), n0 * n0 * n0)
        var gn = n0
        for li in 1 ..< resList.count {
            let res = resList[li], gs = res + 1, step = bsize / Float(res)
            var near = Pipeline.nearSurfaceMask(grid, gn, mcLevel)
            let expand = (res == resList.last!) ? 0 : 1
            for _ in 0 ..< expand { near = Pipeline.dilate(near, gn) }
            var nxt = [Bool](repeating: false, count: gs * gs * gs)
            for ci in 0 ..< gn {
                for cj in 0 ..< gn {
                    for ck in 0 ..< gn where near[(ci * gn + cj) * gn + ck] {
                        let fi = Swift.min(ci * 2, gs - 1), fj = Swift.min(cj * 2, gs - 1), fk = Swift.min(ck * 2, gs - 1)
                        nxt[(fi * gs + fj) * gs + fk] = true
                    }
                }
            }
            for _ in 0 ..< (2 - expand) { nxt = Pipeline.dilate(nxt, gs) }
            var nidx: [Int] = []
            var pts: [Float] = []
            for fi in 0 ..< gs {
                let x = bmin + Float(fi) * step
                for fj in 0 ..< gs {
                    let y = bmin + Float(fj) * step
                    for fk in 0 ..< gs where nxt[(fi * gs + fj) * gs + fk] {
                        nidx.append((fi * gs + fj) * gs + fk)
                        pts.append(x); pts.append(y); pts.append(bmin + Float(fk) * step)
                    }
                }
            }
            let vals = decode(pts, nidx.count)
            var newGrid = [Float](repeating: Float.nan, count: gs * gs * gs)
            for m in 0 ..< nidx.count { newGrid[nidx[m]] = vals[m] }
            grid = newGrid; gn = gs
        }
        return (grid, gn)
    }

    /// Cells bracketing the iso-surface (see `Pipeline.nearSurfaceMask`), flags share `g`'s layout.
    @_optimize(speed)
    static func nearSurface(_ g: SparseSDFGrid, alpha: Float) -> SparseBlocks<UInt8> {
        var near = SparseBlocks<UInt8>(layoutOf: g, fill: 0)
        let n = g.n
        @inline(__always) func sgn(_ v: Float) -> Float { let x = v + alpha; return x > 0 ? 1 : (x < 0 ? -1 : 0) }
        for s in 0 ..< g.slots {
            let (bi, bj, bk) = g.origin(of: s)
            let base = s * sparseBlockCells
            for a in 0 ..< sparseBlockEdge {
                let i = bi + a
                if i >= n { break }
                for b in 0 ..< sparseBlockEdge {
                    let j = bj + b
                    if j >= n { break }
                    for c in 0 ..< sparseBlockEdge {
                        let k = bk + c
                        if k >= n { break }
                        let o = base + ((a << 6) | (b << 3) | c)
                        let v = g.data[o]
                        if v.isNaN { continue }
                        if abs(v + alpha) < 0.95 { near.data[o] = 1 }
                        let sa = sgn(v)
                        if i + 1 < n { let w = g.get(i + 1, j, k); if !w.isNaN && sa != sgn(w) { near.data[o] = 1; near.set(i + 1, j, k, 1) } }
                        if j + 1 < n { let w = g.get(i, j + 1, k); if !w.isNaN && sa != sgn(w) { near.data[o] = 1; near.set(i, j + 1, k, 1) } }
                        if k + 1 < n { let w = g.get(i, j, k + 1); if !w.isNaN && sa != sgn(w) { near.data[o] = 1; near.set(i, j, k + 1, 1) } }
                    }
                }
            }
        }
        return near
    }

    /// 3x3x3 box dilation as three 1-D passes (same result as the dense 26-neighbourhood dilate).
    @_optimize(speed)
    static func dilate(_ m: SparseBlocks<UInt8>) -> SparseBlocks<UInt8> {
        var cur = m
        let n = m.n
        for axis in 0 ..< 3 {
            var out = SparseBlocks<UInt8>(n: n, fill: 0)
            for s in 0 ..< cur.slots {
                let (bi, bj, bk) = cur.origin(of: s)
                let base = s * sparseBlockCells
                for a in 0 ..< sparseBlockEdge {
                    let i = bi + a
                    if i >= n { break }
                    for b in 0 ..< sparseBlockEdge {
                        let j = bj + b
                        if j >= n { break }
                        for c in 0 ..< sparseBlockEdge {
                            let k = bk + c
                            if k >= n { break }
                            if cur.data[base + ((a << 6) | (b << 3) | c)] == 0 { continue }
                            out.set(i, j, k, 1)
                            switch axis {
                            case 0:
                                if i > 0 { out.set(i - 1, j, k, 1) }
                                if i + 1 < n { out.set(i + 1, j, k, 1) }
                            case 1:
                                if j > 0 { out.set(i, j - 1, k, 1) }
                                if j + 1 < n { out.set(i, j + 1, k, 1) }
                            default:
                                if k > 0 { out.set(i, j, k - 1, 1) }
                                if k + 1 < n { out.set(i, j, k + 1, 1) }
                            }
                        }
                    }
                }
            }
            cur = out
        }
        return cur
    }

    /// Block-sparse refinement. `progress` gets 0…1; returns nil if `isCancelled` fires.
    static func sparse(resolution R: Int, bound: Float, minResolution: Int, mcLevel: Float,
                       batch: Int = 1 << 19, isCancelled: () -> Bool = { false },
                       progress: ((Float) -> Void)? = nil, decode: Decode) -> SparseSDFGrid? {
        let bmin = -bound, bsize = 2 * bound
        let resList = levels(resolution: R, minResolution: minResolution)
        let r0 = resList[0], n0 = r0 + 1
        if isCancelled() { return nil }
        let first = decode(coarsePoints(n0, bmin: bmin, step: bsize / Float(r0)), n0 * n0 * n0)
        var grid = SparseSDFGrid(dense: first, n: n0)
        // Finer levels cost ~4x the previous one, so weight progress that way.
        let weights = (1 ..< max(resList.count, 2)).map { Float(1 << (2 * $0)) }
        let totalWeight = weights.reduce(0, +)
        var doneWeight: Float = 0
        progress?(0)

        for li in 1 ..< resList.count {
            let res = resList[li], gs = res + 1, step = bsize / Float(res)
            let w = weights[li - 1]
            let expand = (res == resList.last!) ? 0 : 1
            var m = nearSurface(grid, alpha: mcLevel)
            grid = SparseSDFGrid(n: 1, fill: .nan)                       // drop the coarse values early
            for _ in 0 ..< expand { m = dilate(m) }
            var nxt = SparseBlocks<UInt8>(n: gs, fill: 0)
            for s in 0 ..< m.slots {
                let (bi, bj, bk) = m.origin(of: s)
                let base = s * sparseBlockCells
                for a in 0 ..< sparseBlockEdge { for b in 0 ..< sparseBlockEdge { for c in 0 ..< sparseBlockEdge
                    where m.data[base + ((a << 6) | (b << 3) | c)] != 0 {
                    nxt.set(Swift.min((bi + a) * 2, gs - 1), Swift.min((bj + b) * 2, gs - 1), Swift.min((bk + c) * 2, gs - 1), 1)
                } } }
            }
            m = SparseBlocks<UInt8>(n: 1, fill: 0)
            for _ in 0 ..< (2 - expand) { nxt = dilate(nxt) }

            var total = 0
            for f in nxt.data where f != 0 { total += 1 }
            var out = SparseSDFGrid(layoutOf: nxt, fill: .nan)
            var pts: [Float] = []; pts.reserveCapacity(batch * 3)
            var dest: [Int] = []; dest.reserveCapacity(batch)
            var processed = 0
            var cancelled = false
            func flush() {
                guard !dest.isEmpty else { return }
                let vals = decode(pts, dest.count)
                for t in 0 ..< dest.count { out.data[dest[t]] = vals[t] }
                processed += dest.count
                pts.removeAll(keepingCapacity: true); dest.removeAll(keepingCapacity: true)
                progress?((doneWeight + w * Float(processed) / Float(max(total, 1))) / totalWeight)
                if isCancelled() { cancelled = true }
            }
            scan: for s in 0 ..< nxt.slots {
                let (bi, bj, bk) = nxt.origin(of: s)
                let base = s * sparseBlockCells
                for a in 0 ..< sparseBlockEdge {
                    let x = bmin + Float(bi + a) * step
                    for b in 0 ..< sparseBlockEdge {
                        let y = bmin + Float(bj + b) * step
                        for c in 0 ..< sparseBlockEdge {
                            let o = base + ((a << 6) | (b << 3) | c)
                            if nxt.data[o] == 0 { continue }
                            pts.append(x); pts.append(y); pts.append(bmin + Float(bk + c) * step)
                            dest.append(o)
                            if dest.count == batch { flush(); if cancelled { break scan } }
                        }
                    }
                }
            }
            if !cancelled { flush() }
            if cancelled { return nil }
            grid = out
            doneWeight += w
        }
        progress?(1)
        return grid
    }
}
