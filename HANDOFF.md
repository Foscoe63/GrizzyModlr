# Modelr handoff

Written 2026-10-29. Everything below is committed on `main` locally (nothing pushed). Read this first, then `DESIGN.md` and `ROADMAP.md`.

## Where things stand

**Roadmap Phase 1: complete. Phase 2: complete except two items** (see "Not done").

Last commit: `4bc462a` (TripoSR Fast shape model). Full test suite: **204 tests, 0 failures**.

Uncommitted/untracked: `.codegraph/` (tool index) and `ROADMAP.md` (the plan; its checkboxes are not updated). `HANDOFF.md` is new.

### What was built this session (newest first)

| Area | What | Main files |
|---|---|---|
| Fast shape | TripoSR as a 3rd shape model ("Fast" in the model menu). Torch `.ckpt` reader, DINO ViT + triplane transformer + NeRF decoder, surface-nets meshing, forced Y-up axis remap | `Packages/HunyuanPaintMLX/.../TripoSR.swift`, `TorchCheckpoint.swift`, `Sources/Core/TripoMesh.swift`, `Sources/ShapeEngine.swift` (`runTripo`) |
| Depth relief | Depth Anything V2 Small to a watertight relief solid | `.../DepthAnything.swift`, `Sources/Core/ReliefMesh.swift`, `Sources/DepthEngine.swift` |
| Text to 3D | Stable Diffusion 2.1 base text-to-image (CLIP BPE tokenizer, DDIM), prompt panel on empty projects | `.../TextToImage.swift`, `Sources/TextToImageEngine.swift`, `Sources/TextToImagePanel.swift` |
| Quad export | Greedy triangle pairing for OBJ | `Sources/Core/QuadPairing.swift`, `MeshExporter.export(quads:)` |
| Scenes | Place versions from any project, transform, merge into a shape version | `Sources/SceneComposerView.swift`, `Sources/Core/SceneComposition.swift` |
| Viewer | Environments (procedural or custom HDR), ground, shadows, camera presets, turntable, image export | `Sources/ViewerStage.swift`, `ViewerMenu.swift`, `Sources/Core/ViewerDefaults.swift` |
| Topology | Symmetrize, evenly-sized remesh, LOD sets | `Sources/Core/MeshOps.swift`, `ProjectStore.applyLODs` |
| Models location | Move or point the whole model library at another drive (verified copy, rescan) | `Sources/Core/ModelRelocation.swift`, `AppRuntime` (relocation), Settings → Models |
| View synthesis | Zero123++ v1.2 six candidate views, assign to Left/Back/Right | `.../Zero123Plus.swift`, `Sources/ViewSynthEngine.swift`, `ReferenceViewsStrip.swift` |
| Multi-view input | Hunyuan3D-2mv model + reference slots | `Hy3DMLX/MultiView.swift`, `ShapeGenerator.generate(views:)` |
| Phase 1 tools | Mesh edits (decimate/smooth/fill holes), sculpt brush, material editor, texture paint (paint/soften/heal), booleans, UV editor | `MeshEditPopover`, `SculptView`, `TexturePaintController`, `MeshBoolean`, `UVEditorView` |

All new edits save as **new versions** (never overwrite). The shape/paint job state machine (`AppReducer`) is untouched except a `.modelsRescanned` event; the newer jobs (view synthesis, text-to-image, depth relief, relocation) run **outside the reducer** in `AppRuntime` with a busy guard (no GPU job while a shape/paint job runs).

### Settings (rule: new options go in the Settings window)

Settings has three tabs: **Models** (installs + models location), **Tools** (mesh/topology/scene/Fast shape/depth relief/text-to-3D/export/brush defaults), **Viewer**. Defaults live in `Sources/Core/ToolDefaults.swift` and `ViewerDefaults.swift` (`@AppStorage` keys `tools.*` / `viewer.*`). Add a section there for any new feature with options.

## Not done

1. **Quad retopology (partial)**: "Quad remesh (aligned)" in the mesh edit popover fits the model's own axis frame (`MeshFrame`), remeshes on a grid in that frame with sharp creases, and reports how much pairs into quads (OBJ export pairs them). Still missing: a real curvature/cross-field flow, edge loops that follow organic shapes, singularity handling, and user-drawn flow guides. Curved organic models get an axis-aligned grid, not flowing loops.
2. **Material editing by region**: limited to one dominant color of the texture (swatches in the material popover, soft edge, Settings → Tools → Material regions). No per-UV-island or paint-a-region selection yet.
3. **AI Fill tool** (texture paint → AI Fill): mark an area, Fill regenerates it with Stable Diffusion 2.1 base (blended latent diffusion in a 512² window; no dedicated inpainting checkpoint; needs the Text to 3D model; Settings → Tools → AI Fill). Smoke check: `MODELR_UI_SMOKE=1 MODELR_SMOKE_INPAINT=1 MODELR_SMOKE_OUT=<dir>` (about 7 s Release). Result on stripes continued the pattern with a faint smudge. The old Heal brush is still classical diffusion.
4. **Booleans**: still voxel-based, but result vertices are now placed on creases (QEF over field gradients, Settings → Tools → Combine → Keep sharp edges), so cuts stay crisp at moderate detail; detail slider goes to 384 voxels (memory heavy above 256). Features thinner than a voxel are still lost.
5. **Phase 3** (rigging, FBX/USDZ/3MF, batch, plugins, shortcuts) not started. Cheap wins listed in `ROADMAP.md`: USDZ export, keyboard shortcuts.

## Not yet verified by hand in the running app

Unit tests and harnesses cover the logic; the app was only driven for: the Fast-shape smoke run (shape → paint → GLB, PASS), the Scene composer, viewer staging, Settings tabs, and the models-location row. **Never clicked through live:** sculpt brush feel on a large mesh, texture paint, UV editor drag, booleans on a real generated shape, reference slots + Synthesize views, depth relief button, text-to-image panel generation, LOD/symmetry/remesh buttons. A hands-on pass is the best first task.

## Open decisions / blockers

- **Git push returns 403.** `origin` is `ZimengXiong/Modelr`, signed in as `Foscoe63` (no write access). Either fork + PR or get collaborator access. All work is local commits only.
- Decide whether to update `ROADMAP.md` (untracked) and commit it.

## Environment and gotchas

- **Models live on `/Volumes/Models/Modelr`** (UserDefaults key `modelsRootPath`, domain `com.zimeng.Modelr`). Folders: `shape-small`, `shape-large`, `shape-multiview`, `paint-small`, `paint-large`, `view-synthesis`, `text-to-image`, `depth-estimation`, `triposr`. ~38 GB. Drive unplugged ⇒ models show "not installed" and downloads fail cleanly. Projects stay in `~/Library/Application Support/Modelr/projects` (internal).
- **`xcodebuild test` must use a derived-data path outside `/Volumes/Storage`** (the test bundle won't load from there). Working command:
  `xcodebuild -project Modelr.xcodeproj -scheme Modelr -derivedDataPath <somewhere on the internal disk> CODE_SIGNING_ALLOWED=NO test`
  Regenerate the project after adding files: `xcodegen generate`. New Swift files that tests need (or that `MeshViewer`/`ProjectStore` reference) must be listed under the `ModelrTests` sources in `project.yml`.
- **Smoke harness** (`make smoke`-style): `MODELR_UI_SMOKE=1 MODELR_SMOKE_MODEL=small|large|fast MODELR_SMOKE_OUT=<dir> MODELR_SMOKE_IMAGE=<png> .derived/Build/Products/Debug/Modelr.app/Contents/MacOS/Modelr`. It creates a project in the real store: back up `projects.json` and remove the new project folder afterwards.
- **App window position:** the debug app has opened on the user's second display. Move it with `osascript` (System Events → set position) and capture only the app window (`screencapture -l <windowID>`), not the whole screen. **Avoid scripted clicks/typing** unless the app is certainly frontmost: one earlier scripted keystroke missed and may have typed into another window.
- Debug builds are slow for CPU-heavy steps (surface-nets meshing ~26 s Debug vs ~0.4 s Release); judge speed on Release.
- Do not run downloaded third-party code. Python references in this session used only the installed libraries (diffusers/transformers) and `torch.load(weights_only=True)`.

## Licenses to remember

- Zero123++ weights: **CC-BY-NC 4.0** (non-commercial). Hunyuan3D: Tencent non-commercial license.
- Stable Diffusion 2.1: OpenRAIL++ (use restrictions). Taken from the `sd2-community` mirror (Stability's original repo is gone).
- TripoSR: MIT. Depth Anything V2 Small: Apache-2.0.
- The Zero123++ and text-to-image notes appear in the catalog descriptions and UI; keep them if the UI changes.

## Verification record (how ports were checked)

Each model port was compared against an independent reference built from library classes, with the exact weights:

| Port | Result |
|---|---|
| Zero123++ | final latents cosine 0.9999997, decoded image 68.9 dB PSNR (fp32) |
| SD 2.1 text-to-image | tokenizer 21/21 vs diffusers; fp32 latents cosine 0.99994 |
| Depth Anything V2 Small | cosine 1.0 vs transformers |
| TripoSR | ViT tokens, triplanes, density, color all cosine 1.0; preprocessing exact |
| Hunyuan3D-2mv | loads and generates (no numeric reference; conditioning math follows upstream) |

The Python reference scripts and harness packages were in the session scratchpad (`/private/tmp/claude-501/.../scratchpad`, temporary): `z123/ref*.py`, `sdpar`, `dapar`, `tsrpar`, `tsre2e`. They are not in the repo. If they are needed again, recreate from the notes above (venv with torch 2.2.2, diffusers 0.27.2, transformers 4.38.2).

## Suggested start tomorrow

1. Decide push strategy (fork + PR vs access) and push.
2. Hands-on pass over the unverified tools above; file bugs as you go.
3. Pick from: USDZ export and keyboard shortcuts (Phase 3 quick wins), per-region material editing, or a first cut at quad retopology.
4. Update `ROADMAP.md` checkboxes and commit it.
