# grizzymodlr

native macos app for local image → 3d. both the shape and texture pipelines run in-process on [mlx swift](https://github.com/zimengxiong/hunyuan3d-swift) (no python, no pytorch, nothing leaves the machine) and render live in scenekit as they generate.

https://github.com/user-attachments/assets/555b455d-5e79-429d-8f7f-b768ad8bf8da

## models

grizzymodlr ships four models: one small and one large per stage. interested in integrating hunyuan3d into your app? checkout [hunyuan3d-swift](https://github.com/zimengxiong/hunyuan3d-swift)

times below are measured in-app on an M4 Max (with mesh decimation before paint).

| slot | checkpoint | notes | time |
|---|---|---|---|
| shape · small | `hunyuan3d-dit-v2-mini` (0.6b) | fastest, low ram | ~52 s |
| shape · large | `hunyuan3d-dit-v2-0-turbo` (1.1b) | 8-step consistency | ~57 s |
| paint · small | `hunyuan3d-paint-v2-0` | rgb color, 2048 atlas | ~94 s |
| paint · large | `hunyuan3d-paintpbr-v2-1` | pbr (albedo + metallic-roughness), 4096 atlas | ~139 s |

## architecture!!

grizzymodlr is a simple app, but a lot of time went into designing a nice architecture for it, since generation, downloading, and painting take a lot of time, everything is designed to be responsive, including cancellations, no hung processes 😁. grizzymodlr is built around a single pure reducer: `reduce(state, event) -> (state', [Effect])`,
so the entire transition table (onboarding, downloads, shape, paint, cancel/fail edges,
stale-token protection) is unit-testable with no UI, network, or GPU. read
[`DESIGN.md`](DESIGN.md) for agents.

```
MainActor
  AppState (reducer + state)          Sources/Core/AppReducer.swift, AppState.swift,
    - AppPhase (onboarding/ready)                  AppEvent.swift, AppEffect.swift
    - ModelInstallStore [4 models]
    - ProjectStore [shape/paint jobs, versions]    Sources/ProjectStore.swift
        │ effects
   ┌────┴─────────────┬───────────────────┐
   DownloadManager    EngineArbiter        (swiftui views = pure f(state))
   URLSession +       actor: exclusive     Sources/ContentView.swift,
   Range resume +     GPU owner, single    OnboardingView, ModelManagerView,
   sha256 verify      model residency      ProjectDetailView, MeshViewer, …
   (Core/)            (Core/)
                         │
                ┌────────┴────────┐
                ShapeEngine        PaintEngine        (serial, off-main)
                → Hy3DMLX          → HunyuanPaintMLX   (vendored in Packages/)
```

## misc
everything lives under `~/Library/Application Support/Modelr/`:

## install

grizzymodlr currently ships an apple-silicon build for macos 14 or later. download
[`GrizzyModlr.zip`](https://github.com/Foscoe63/GrizzyModlr/releases/latest/download/GrizzyModlr.zip),
unzip it, and move `GrizzyModlr.app` to Applications. remove from quarentine:

```sh
xattr -dr com.apple.quarantine /Applications/GrizzyModlr.app
```
