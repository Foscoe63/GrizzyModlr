import SwiftUI

/// Runs shape (and optionally paint) generation for many images, one project per image,
/// strictly one job at a time — the GPU holds one model's weights at once.
@Observable
final class BatchQueue {
    struct Item: Identifiable, Equatable {
        enum Step: Equatable { case waiting, shape, paint, done, failed(String) }
        let id: Project.ID
        var name: String
        var step: Step = .waiting
    }

    private(set) var items: [Item] = []
    private(set) var running = false
    private var task: Task<Void, Never>?

    var summary: String? {
        guard !items.isEmpty else { return nil }
        let done = items.filter { $0.step == .done }.count
        let failed = items.filter { if case .failed = $0.step { return true } else { return false } }.count
        return "Batch: \(done) of \(items.count) done" + (failed > 0 ? ", \(failed) failed" : "")
    }

    /// Creates one project per image and works through them in order.
    @MainActor func start(images: [URL], paint: Bool, runtime: AppRuntime) {
        guard !running, !images.isEmpty else { return }
        let store = runtime.store
        items = images.map { url in
            let project = store.newProject()
            store.rename(project.id, to: url.deletingPathExtension().lastPathComponent)
            store.setImage(fromURL: url, for: project.id)
            return Item(id: project.id, name: url.deletingPathExtension().lastPathComponent)
        }
        running = true
        task = Task { @MainActor [weak self] in
            guard let self else { return }
            for index in self.items.indices {
                if Task.isCancelled { break }
                let id = self.items[index].id
                self.items[index].step = .shape
                var failure = await Self.drive(runtime, id: id, kind: .shape)
                if failure == nil, paint, !Task.isCancelled {
                    self.items[index].step = .paint
                    failure = await Self.drive(runtime, id: id, kind: .paint)
                }
                self.items[index].step = failure.map { .failed($0) } ?? .done
            }
            self.running = false
        }
    }

    @MainActor func cancel(runtime: AppRuntime) {
        task?.cancel()
        for item in items where item.step == .shape || item.step == .paint {
            runtime.cancelShape(item.id); runtime.cancelPaint(item.id)
        }
    }

    func clearFinished() { if !running { items = [] } }

    private enum Kind { case shape, paint }

    /// Starts one job and waits for it to end. Returns an error message, or nil on success.
    @MainActor private static func drive(_ runtime: AppRuntime, id: Project.ID, kind: Kind) async -> String? {
        func isRunning() -> Bool {
            switch kind {
            case .shape: return runtime.state.shape[id]?.state.isRunning ?? false
            case .paint: return runtime.state.paint[id]?.state.isRunning ?? false
            }
        }
        func failure() -> String? {
            switch kind {
            case .shape: if case .failed(_, let m)? = runtime.state.shape[id]?.state { return m } else { return nil }
            case .paint: if case .failed(_, let m)? = runtime.state.paint[id]?.state { return m } else { return nil }
            }
        }
        // Wait for any other job (a manual one, say) to release the GPU.
        while runtime.state.shape.values.contains(where: { $0.state.isRunning })
            || runtime.state.paint.values.contains(where: { $0.state.isRunning }) {
            if Task.isCancelled { return "Cancelled" }
            try? await Task.sleep(for: .seconds(1))
        }
        let before = runtime.store.project(id)?.generations.count ?? 0
        switch kind {
        case .shape: runtime.requestGenerate(id)
        case .paint: runtime.requestPaint(id)
        }
        // It must show as running within a few seconds, otherwise the request was refused
        // (e.g. weights missing).
        var waited = 0
        while !isRunning() && waited < 10 {
            if (runtime.store.project(id)?.generations.count ?? 0) > before { return nil }
            try? await Task.sleep(for: .milliseconds(500)); waited += 1
        }
        guard isRunning() else {
            return failure() ?? ((runtime.store.project(id)?.generations.count ?? 0) > before ? nil
                                 : "Didn't start — are the models installed?")
        }
        while isRunning() {
            if Task.isCancelled { return "Cancelled" }
            try? await Task.sleep(for: .seconds(1))
        }
        if let message = failure() { return message }
        return (runtime.store.project(id)?.generations.count ?? 0) > before ? nil : "No result was produced."
    }
}
