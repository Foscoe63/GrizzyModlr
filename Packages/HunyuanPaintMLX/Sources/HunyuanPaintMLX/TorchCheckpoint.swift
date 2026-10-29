import Foundation

/// Minimal reader for PyTorch `.ckpt` / `.pt` files saved with `torch.save` (a zip of a pickle plus
/// raw tensor storages). It understands only the tiny opcode subset torch emits for a state dict,
/// builds plain values (never imports or executes anything from the file), and rejects anything
/// else — so loading an untrusted checkpoint cannot run code.
public enum TorchCheckpoint {
    public struct TensorRef {
        public let name: String
        public let dtype: String                 // "float32", "float16", "bfloat16", "int64", …
        public let shape: [Int]
        public let strides: [Int]
        public let storageOffset: Int            // in elements
        public let storage: Data                 // raw bytes of the whole storage
        public var elementSize: Int { Self.size(of: dtype) }
        public var count: Int { shape.reduce(1, *) }

        static func size(of dtype: String) -> Int {
            switch dtype {
            case "float64", "int64": return 8
            case "float32", "int32": return 4
            case "float16", "bfloat16", "int16": return 2
            default: return 1
            }
        }

        public var isContiguous: Bool {
            var expected = 1
            for (s, d) in zip(strides, shape).reversed() where d != 1 {
                if s != expected { return false }
                expected *= d
            }
            return true
        }

        /// Contiguous little-endian bytes of this tensor (copies only when strided).
        public func bytes() -> Data {
            let es = elementSize
            if isContiguous {
                let start = storageOffset * es
                return storage.subdata(in: storage.startIndex + start ..< storage.startIndex + start + count * es)
            }
            var out = Data(count: count * es)
            var index = [Int](repeating: 0, count: shape.count)
            out.withUnsafeMutableBytes { dst in
                storage.withUnsafeBytes { src in
                    for n in 0 ..< count {
                        var off = storageOffset
                        for (i, st) in zip(index, strides) { off += i * st }
                        for b in 0 ..< es { dst[n * es + b] = src[off * es + b] }
                        var d = shape.count - 1
                        while d >= 0 { index[d] += 1; if index[d] < shape[d] { break }; index[d] = 0; d -= 1 }
                    }
                }
            }
            return out
        }
    }

    public enum Failure: Error, LocalizedError {
        case notAZip, missing(String), unsupported(String), malformed(String)
        public var errorDescription: String? {
            switch self {
            case .notAZip: return "Not a PyTorch checkpoint (zip) file."
            case .missing(let n): return "Checkpoint entry missing: \(n)."
            case .unsupported(let s): return "Unsupported checkpoint content: \(s)."
            case .malformed(let s): return "Malformed checkpoint: \(s)."
            }
        }
    }

    /// Loads every tensor of a state dict (a top-level dict of name → tensor).
    public static func load(_ url: URL) throws -> [String: TensorRef] {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        let entries = try zipEntries(data)
        guard let pkl = entries.first(where: { $0.name.hasSuffix("/data.pkl") || $0.name == "data.pkl" }) else {
            throw Failure.missing("data.pkl")
        }
        let prefix = String(pkl.name.dropLast("data.pkl".count))                    // e.g. "archive/"
        var unpickler = Unpickler(bytes: data.subdata(in: pkl.range), storage: { key in
            guard let e = entries.first(where: { $0.name == "\(prefix)data/\(key)" }) else { return nil }
            return data.subdata(in: e.range)
        })
        let root = try unpickler.run()
        guard case .dict(let items) = root else { throw Failure.malformed("top level is not a dict") }
        var out: [String: TensorRef] = [:]
        for (k, v) in items {
            guard case .string(let name) = k else { continue }
            if case .tensor(let t) = v { out[name] = TensorRef(name: name, dtype: t.dtype, shape: t.shape, strides: t.strides,
                                                               storageOffset: t.offset, storage: t.storage) }
        }
        return out
    }

    // MARK: zip

    private struct ZipEntry { let name: String; let range: Range<Int> }

    /// Central-directory walk for stored (uncompressed) entries — what torch writes.
    private static func zipEntries(_ d: Data) throws -> [ZipEntry] {
        func u16(_ o: Int) -> Int { Int(d[d.startIndex + o]) | Int(d[d.startIndex + o + 1]) << 8 }
        func u32(_ o: Int) -> Int { u16(o) | u16(o + 2) << 16 }
        func u64(_ o: Int) -> Int { u32(o) | u32(o + 4) << 32 }
        guard d.count > 22 else { throw Failure.notAZip }
        var eocd = -1
        var i = d.count - 22
        while i >= max(0, d.count - 22 - 65_535) {
            if u32(i) == 0x0605_4b50 { eocd = i; break }
            i -= 1
        }
        guard eocd >= 0 else { throw Failure.notAZip }
        var count = u16(eocd + 10), cdOffset = u32(eocd + 16)
        if cdOffset == 0xFFFF_FFFF || count == 0xFFFF {                              // zip64
            let loc = eocd - 20
            guard loc >= 0, u32(loc) == 0x0706_4b50 else { throw Failure.notAZip }
            let z64 = u64(loc + 8)
            guard u32(z64) == 0x0606_4b50 else { throw Failure.notAZip }
            count = u64(z64 + 32); cdOffset = u64(z64 + 48)
        }
        var entries: [ZipEntry] = []
        var p = cdOffset
        for _ in 0 ..< count {
            guard u32(p) == 0x0201_4b50 else { throw Failure.malformed("bad central directory") }
            let method = u16(p + 10)
            var compressed = u32(p + 20), uncompressed = u32(p + 24)
            let nameLen = u16(p + 28), extraLen = u16(p + 30), commentLen = u16(p + 32)
            var local = u32(p + 42)
            let name = String(decoding: d.subdata(in: d.startIndex + p + 46 ..< d.startIndex + p + 46 + nameLen), as: UTF8.self)
            // zip64 extra field: 8-byte sizes / offset appear in order for the 0xFFFFFFFF fields
            if compressed == 0xFFFF_FFFF || uncompressed == 0xFFFF_FFFF || local == 0xFFFF_FFFF {
                var e = p + 46 + nameLen, remaining = extraLen
                while remaining >= 4 {
                    let id = u16(e), size = u16(e + 2)
                    if id == 1 {
                        var q = e + 4
                        if uncompressed == 0xFFFF_FFFF { uncompressed = u64(q); q += 8 }
                        if compressed == 0xFFFF_FFFF { compressed = u64(q); q += 8 }
                        if local == 0xFFFF_FFFF { local = u64(q) }
                    }
                    e += 4 + size; remaining -= 4 + size
                }
            }
            guard method == 0 else { throw Failure.unsupported("compressed zip entry \(name)") }
            guard u32(local) == 0x0403_4b50 else { throw Failure.malformed("bad local header") }
            let start = local + 30 + u16(local + 26) + u16(local + 28)
            entries.append(ZipEntry(name: name, range: d.startIndex + start ..< d.startIndex + start + uncompressed))
            p += 46 + nameLen + extraLen + commentLen
        }
        return entries
    }

    // MARK: pickle

    indirect enum Value {
        case none, bool(Bool), int(Int), float(Double), string(String)
        case tuple([Value]), list([Value]), dict([(Value, Value)])
        case global(String)
        case storage(key: String, dtype: String)
        case tensor(Tensor)
        struct Tensor { var storage: Data; var dtype: String; var shape: [Int]; var strides: [Int]; var offset: Int }
    }

    private struct Unpickler {
        let bytes: Data
        let storage: (String) -> Data?
        private var pos = 0
        private var stack: [Value] = []
        private var marks: [Int] = []
        private var memo: [Int: Value] = [:]
        private var memoCounter = 0

        init(bytes: Data, storage: @escaping (String) -> Data?) { self.bytes = bytes; self.storage = storage }

        private mutating func byte() throws -> UInt8 {
            guard pos < bytes.count else { throw Failure.malformed("truncated pickle") }
            defer { pos += 1 }; return bytes[bytes.startIndex + pos]
        }
        private mutating func read(_ n: Int) throws -> Data {
            guard pos + n <= bytes.count else { throw Failure.malformed("truncated pickle") }
            defer { pos += n }
            return bytes.subdata(in: bytes.startIndex + pos ..< bytes.startIndex + pos + n)
        }
        private mutating func uint(_ n: Int) throws -> Int {
            let d = try read(n); var v = 0
            for (i, b) in d.enumerated() { v |= Int(b) << (8 * i) }
            return v
        }
        private mutating func line() throws -> String {
            var out = [UInt8]()
            while true { let b = try byte(); if b == 0x0A { break }; out.append(b) }
            return String(decoding: out, as: UTF8.self)
        }
        private mutating func pop() throws -> Value {
            guard let v = stack.popLast() else { throw Failure.malformed("stack underflow") }; return v
        }
        private mutating func popMark() throws -> [Value] {
            guard let m = marks.popLast() else { throw Failure.malformed("no mark") }
            let items = Array(stack[m...]); stack.removeSubrange(m...); return items
        }

        static func dtype(forStorage g: String) -> String? {
            switch g.split(separator: ".").last.map(String.init) ?? g {
            case "FloatStorage": return "float32"
            case "HalfStorage": return "float16"
            case "BFloat16Storage": return "bfloat16"
            case "DoubleStorage": return "float64"
            case "LongStorage": return "int64"
            case "IntStorage": return "int32"
            case "ShortStorage": return "int16"
            case "CharStorage", "ByteStorage", "BoolStorage": return "uint8"
            default: return nil
            }
        }

        mutating func run() throws -> Value {
            while true {
                let op = try byte()
                switch op {
                case 0x80: _ = try byte()                                              // PROTO
                case 0x95: _ = try read(8)                                             // FRAME
                case UInt8(ascii: "}"): stack.append(.dict([]))
                case UInt8(ascii: "]"): stack.append(.list([]))
                case UInt8(ascii: ")"): stack.append(.tuple([]))
                case UInt8(ascii: "N"): stack.append(.none)
                case 0x88: stack.append(.bool(true))
                case 0x89: stack.append(.bool(false))
                case UInt8(ascii: "("): marks.append(stack.count)
                case UInt8(ascii: "q"): memo[try uint(1)] = stack.last
                case UInt8(ascii: "r"): memo[try uint(4)] = stack.last
                case 0x94: memo[memoCounter] = stack.last; memoCounter += 1             // MEMOIZE
                case UInt8(ascii: "h"): stack.append(try memoGet(try uint(1)))
                case UInt8(ascii: "j"): stack.append(try memoGet(try uint(4)))
                case UInt8(ascii: "X"): let n = try uint(4); stack.append(.string(String(decoding: try read(n), as: UTF8.self)))
                case 0x8c: let n = try uint(1); stack.append(.string(String(decoding: try read(n), as: UTF8.self)))
                case UInt8(ascii: "J"): let v = try uint(4); stack.append(.int(Int(Int32(truncatingIfNeeded: v))))
                case UInt8(ascii: "K"): stack.append(.int(try uint(1)))
                case UInt8(ascii: "M"): stack.append(.int(try uint(2)))
                case 0x8a:
                    let n = try uint(1); let d = try read(n); var v = 0
                    for (i, b) in d.enumerated() { v |= Int(b) << (8 * i) }
                    if n > 0, n < 8, d.last! & 0x80 != 0 { v -= 1 << (8 * n) }
                    stack.append(.int(v))
                case UInt8(ascii: "G"):
                    let d = try read(8); var bits: UInt64 = 0
                    for b in d { bits = bits << 8 | UInt64(b) }
                    stack.append(.float(Double(bitPattern: bits)))
                case UInt8(ascii: "c"):
                    let module = try line(), name = try line(); stack.append(.global("\(module).\(name)"))
                case 0x93:
                    guard case .string(let name) = try pop(), case .string(let module) = try pop() else { throw Failure.malformed("STACK_GLOBAL") }
                    stack.append(.global("\(module).\(name)"))
                case UInt8(ascii: "t"): stack.append(.tuple(try popMark()))
                case 0x85: stack.append(.tuple([try pop()]))
                case 0x86: let b = try pop(), a = try pop(); stack.append(.tuple([a, b]))
                case 0x87: let c = try pop(), b = try pop(), a = try pop(); stack.append(.tuple([a, b, c]))
                case UInt8(ascii: "s"):
                    let v = try pop(), k = try pop()
                    guard case .dict(var items) = try pop() else { throw Failure.malformed("SETITEM") }
                    items.append((k, v)); stack.append(.dict(items))
                case UInt8(ascii: "u"):
                    let flat = try popMark()
                    guard case .dict(var items) = try pop() else { throw Failure.malformed("SETITEMS") }
                    var i = 0
                    while i + 1 < flat.count { items.append((flat[i], flat[i + 1])); i += 2 }
                    stack.append(.dict(items))
                case UInt8(ascii: "a"):
                    let v = try pop()
                    guard case .list(var l) = try pop() else { throw Failure.malformed("APPEND") }
                    l.append(v); stack.append(.list(l))
                case UInt8(ascii: "e"):
                    let flat = try popMark()
                    guard case .list(var l) = try pop() else { throw Failure.malformed("APPENDS") }
                    l.append(contentsOf: flat); stack.append(.list(l))
                case UInt8(ascii: "Q"):                                                // BINPERSID
                    guard case .tuple(let pid) = try pop(), pid.count >= 3, case .string("storage") = pid[0],
                          case .global(let type) = pid[1], case .string(let key) = pid[2],
                          let dtype = Self.dtype(forStorage: type) else { throw Failure.unsupported("persistent id") }
                    stack.append(.storage(key: key, dtype: dtype))
                case UInt8(ascii: "R"): // REDUCE: only a few torch/collections constructors are honoured
                    let args = try pop(), callee = try pop()
                    stack.append(try reduce(callee, args))
                case UInt8(ascii: "b"):                                                // BUILD: ignore (OrderedDict metadata)
                    _ = try pop()
                case UInt8(ascii: "."): return try pop()
                default: throw Failure.unsupported(String(format: "pickle opcode 0x%02x", op))
                }
            }
        }

        private func memoGet(_ i: Int) throws -> Value {
            guard let v = memo[i] else { throw Failure.malformed("bad memo reference") }; return v
        }

        private func ints(_ v: Value) throws -> [Int] {
            guard case .tuple(let t) = v else { throw Failure.malformed("expected tuple") }
            return try t.map { guard case .int(let i) = $0 else { throw Failure.malformed("expected int") }; return i }
        }

        private func reduce(_ callee: Value, _ args: Value) throws -> Value {
            guard case .global(let name) = callee else { throw Failure.unsupported("calling a non-global") }
            switch name {
            case "collections.OrderedDict": return .dict([])
            case "torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_tensor":
                guard case .tuple(let a) = args, a.count >= 4, case .storage(let key, let dtype) = a[0],
                      case .int(let offset) = a[1], let data = storage(key) else {
                    throw Failure.malformed("tensor rebuild arguments")
                }
                return .tensor(.init(storage: data, dtype: dtype, shape: try ints(a[2]), strides: try ints(a[3]), offset: offset))
            case "torch._utils._rebuild_parameter":
                guard case .tuple(let a) = args, let first = a.first else { throw Failure.malformed("parameter rebuild") }
                return first
            default:
                throw Failure.unsupported("callable \(name)")
            }
        }
    }
}
