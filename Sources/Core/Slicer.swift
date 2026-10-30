import AppKit

/// Hands a 3MF export to an installed slicer.
enum Slicer: String, CaseIterable, Identifiable {
    case orca, prusa, bambu, cura

    var id: String { rawValue }
    var title: String {
        switch self {
        case .orca: return "OrcaSlicer"
        case .prusa: return "PrusaSlicer"
        case .bambu: return "Bambu Studio"
        case .cura: return "UltiMaker Cura"
        }
    }
    var bundleIDs: [String] {
        switch self {
        case .orca: return ["com.softfever3d.orca-slicer", "com.orcaslicer.OrcaSlicer"]
        case .prusa: return ["com.prusa3d.prusaslicer", "com.prusa3d.PrusaSlicer"]
        case .bambu: return ["com.bambulab.bambu-studio", "com.bambulab.BambuStudio"]
        case .cura: return ["nl.ultimaker.cura", "com.ultimaker.cura"]
        }
    }

    /// App bundle for this slicer, found by bundle id and then by the usual install paths.
    var appURL: URL? {
        for id in bundleIDs { if let u = NSWorkspace.shared.urlForApplication(withBundleIdentifier: id) { return u } }
        let name = title.replacingOccurrences(of: "UltiMaker ", with: "")
        for dir in ["/Applications", NSHomeDirectory() + "/Applications"] {
            for n in [title, name] {
                let p = "\(dir)/\(n).app"
                if FileManager.default.fileExists(atPath: p) { return URL(fileURLWithPath: p) }
            }
        }
        return nil
    }

    static var installed: [Slicer] { allCases.filter { $0.appURL != nil } }

    /// The slicer named in Settings ("auto" = first installed in `allCases` order).
    static func preferred(_ setting: String = ToolDefaults.slicer()) -> Slicer? {
        if let s = Slicer(rawValue: setting), s.appURL != nil { return s }
        return installed.first
    }

    /// Opens `file` (a 3MF) in `slicer`; calls `done` with an error message, or nil on success.
    func open(_ file: URL, done: @escaping (String?) -> Void) {
        guard let app = appURL else { done("\(title) isn't installed."); return }
        NSWorkspace.shared.open([file], withApplicationAt: app, configuration: NSWorkspace.OpenConfiguration()) { _, err in
            DispatchQueue.main.async { done(err.map { "Couldn't open \(self.title): \($0.localizedDescription)" }) }
        }
    }
}
