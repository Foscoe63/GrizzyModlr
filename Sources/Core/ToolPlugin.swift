import Foundation

/// A user-installed external tool, declared by a JSON manifest in the plugins folder
/// (`~/Library/Application Support/GrizzyModlr/Plugins/*.json`):
///
///     { "name": "Decimate with MeshLab", "command": "/usr/local/bin/meshlabserver",
///       "args": ["-i", "{input}", "-o", "{output}"], "input": "obj", "output": "obj" }
///
/// Placeholders: `{input}` (the exported current model), `{output}` (a file path in the run
/// folder, extension from `output`), `{dir}` (the run folder), `{name}` (project name).
/// A plugin runs only when picked from the Model ▸ Tools menu; it can't run on its own.
struct ToolPlugin: Identifiable, Codable, Equatable {
    var name: String
    var command: String
    var args: [String] = []
    /// Export format handed to the tool: glb, obj, stl, ply, 3mf, fbx, usdz.
    var input: String = "glb"
    /// Extension of `{output}`; when set and the tool produces that file it is revealed in Finder.
    var output: String?
    var id: String { name }

    static var folder: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("GrizzyModlr/Plugins", isDirectory: true)
    }

    static func load(from dir: URL = folder) -> [ToolPlugin] {
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return files.filter { $0.pathExtension == "json" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
            .compactMap { (try? Data(contentsOf: $0)).flatMap { try? JSONDecoder().decode(ToolPlugin.self, from: $0) } }
            .filter { !$0.name.isEmpty && !$0.command.isEmpty }
    }

    var inputFormat: MeshExportFormat { MeshExportFormat(rawValue: input == "3mf" ? "threemf" : input) ?? .glb }

    /// Arguments with placeholders substituted.
    func resolvedArgs(input: URL, output: URL, dir: URL, name: String) -> [String] {
        args.map {
            $0.replacingOccurrences(of: "{input}", with: input.path)
                .replacingOccurrences(of: "{output}", with: output.path)
                .replacingOccurrences(of: "{dir}", with: dir.path)
                .replacingOccurrences(of: "{name}", with: name)
        }
    }

    struct Result { var status: Int32; var log: String; var produced: URL? }

    /// Exports the model, runs the tool in a fresh folder, returns its exit status and output tail.
    func run(meshURL: URL, texture: URL?, metallicRoughness: URL?, projectName: String) throws -> Result {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("GrizzyModlr-plugin-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let safe = projectName.replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: "-")
        let inputURL = dir.appendingPathComponent("\(safe).\(inputFormat.ext)")
        try MeshExporter.export(meshURL: meshURL, texture: texture, metallicRoughness: metallicRoughness,
                                format: inputFormat, to: inputURL)
        let outputURL = dir.appendingPathComponent("\(safe)-out.\(output ?? "out")")
        let process = Process()
        // A bare command name is looked up on the usual tool paths, like a shell would.
        if command.contains("/") { process.executableURL = URL(fileURLWithPath: command) } else {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        }
        process.arguments = (command.contains("/") ? [] : [command])
            + resolvedArgs(input: inputURL, output: outputURL, dir: dir, name: safe)
        process.currentDirectoryURL = dir
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = (env["PATH"] ?? "") + ":/opt/homebrew/bin:/usr/local/bin"
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe; process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        let log = String(decoding: data.suffix(2000), as: UTF8.self)
        let produced = output != nil && FileManager.default.fileExists(atPath: outputURL.path) ? outputURL : nil
        return Result(status: process.terminationStatus, log: log, produced: produced)
    }
}
