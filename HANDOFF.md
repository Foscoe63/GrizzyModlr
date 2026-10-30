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

1. **Quad retopology (partial)**: "Quad remesh (aligned)" in the mesh edit popover fits the model's own axis frame (`MeshFrame`), remeshes on a grid in that frame with sharp creases, and reports how much pairs into quads (OBJ export pairs them). Flow-field work (plan M1–M8): done M1 (`MeshTopology`), M2 (`CrossField`: curvature + crease constraints, annealed smoothing, singularity indices; correct on cylinder, box creases, sphere χ=2, torus χ=0), M3 (Mesh edit → Show flow field draws it; Settings → Tools → Flow field; smoke: `MODELR_SMOKE_FLOW=<.mesh>`). Finding: on a real generated scan the field is coherent in smooth areas but has hundreds of singularities in folds/hair (about 5% of faces), so M4–M5 (position field, quad extraction) will give irregular results on noisy meshes. M4–M6 are partly done: `PositionField` (coarse-to-fine lattice agreement) and `FlowRemesh` (extraction, projection, validation) exist, and Mesh edit → "Flow remesh (experimental)" (Settings → Tools → Flow field → Flow remesh quads) runs them. **Status:** the sphere now closes (every boundary loop is capped; touching loops are split at pinch vertices): mostly quads, ~36% valence-4 vertices, vertices within 0.03 of the surface. Torus also closes (335 quads at a 400 target, 79% valence-4). A real generated scan (2000-quad target) now closes too, after a repair loop that drops quads on duplicate edges and re-caps: ~1013 quads + ~936 fill triangles (holes ≤80 edges use a minimum-length triangulation, bigger ones a centre fan; max valence 12, was 70; some long flat diagonals remain in big holes) (~1/3 of faces are triangles), 47% valence-4, ~8.5 s in a Release test build (~88 s Debug; front view at `docs/e2e/flow-remesh-scan-front.png`: green quads / red triangles, after ring-peeling of big holes and quads for 4-edge holes: ~1827 quads (target 2000), ~19% of faces triangles, 54% valence-4, ~9 s Release, no long fans; some overlapping slivers remain where ring points project onto a different part of a concave surface; `refineLongEdges` keeps the working mesh evenly sampled; `FLOW_MESH=<.mesh>` opt-in test `testRealScanReport`) and it only reached ~half the 2000-quad target. Smoke check of the button's edit path on a real bust (`MODELR_UI_SMOKE=1 MODELR_SMOKE_OUT=<dir> MODELR_SMOKE_FLOWREMESH=<.mesh>`, writes flow-remesh-*.png): 1812 quads / 81% of faces, ~10 s; the bust is recognisable and mostly smooth and a fold-repair pass (`relaxFolds`: faces whose normal opposes the input surface are relaxed and re-projected) took folded faces from 364 to 27 of ~2700, which removed the dark patches except one tiny one on the lower torso; ~15 s Release. NOTE: the earlier `flow-remesh-scan-front.png` is a top view (the model is Y-up). The button itself has not been clicked in the UI. Cylinder/rounded box not checked. So the button still falls back to the aligned remesh whenever the result isn't a closed manifold. Next: torus/cylinder/rounded-box checks, quality metrics (M7), guides (M8). M7 done: `FlowRemesh.metrics` plus `testMetricsOnCleanShapes` (sphere/cylinder/torus/rounded box; e.g. torus 95% quads, 87% valence-4, 6° corner error; cylinder 90% / 79% / 9°; rounded box 96% / 84% / 6°; sphere is weakest at 73% / 35% / 25° because of unavoidable singularities). M8 done: draw flow guides (Mesh edit → Draw flow guides, drag on the model, ⌥ orbits; Undo/Done in the banner; Clear N in the popover). Guides pin the cross field near the curve and the lattice itself (`PositionField` pins vertices within 0.6ρ so a lattice row runs through the curve: on the bust, edges lying on the guide went from 9 of 13 aligned to 19 of 19). Guides are saved with the project (`Project.flowGuides`). Also fixed: the aligned quad remesh no longer grows a bar out to the grid edge on scans that aren't watertight (odd ray-parity columns in `meshField`).  Earlier text: Guides pin the cross field near the curve (`CrossField.Options.guides`, strength in Settings → Tools → Flow field), apply to Flow remesh and Show flow field, and (since this update) are saved with the project. Drawing was verified live; the banner's Done button could not be pressed by accessibility automation (it resolved to the wand button), so press it with the mouse. Measured on the real bust (opt-in `testGuideBendsRealScan`, `FLOW_MESH=<.mesh>`, a slanted guide across the chest): cross-field faces near the guide aligned within 15°: 55% without, 82% with; remeshed edges near the guide aligned within 25°: 22% without, 52% with (28 of 54 edges). The field plateaus around 82% whatever the strength (suspected neighbouring surface layers such as arms/hair; not confirmed), and the quad rows follow the guide only about half the time, so treat guides as a strong hint, not a guarantee. Still missing: a real curvature/cross-field flow, edge loops that follow organic shapes, singularity handling, and user-drawn flow guides. Curved organic models get an axis-aligned grid, not flowing loops.
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

## Phase 3 round (UV-island materials, print/FBX export, batch, plugins)

- Material edit can be limited to UV islands (checklist in the popover; `UVLayout.footprintMask`, padded by Settings → "UV island padding"); combines with the color region.
- Export: `.threemf` (Sources/Core/ThreeMFExporter.swift, own stored-zip writer; scaled via Settings "3MF longest side", Y-up→Z-up, on the bed) and `.fbx` (FBXExporter.swift, ASCII 7.4, static mesh + UVs + texture ref). **FBX has only been structure-tested, not opened in Blender/Maya** — verify in a real importer.
- `Slicer` (Sources/Core/Slicer.swift): "Open in <slicer>" in the export menu writes a temp 3MF and opens it; Settings picks the slicer. Not tested against a real slicer here (none installed).
- Batch: `BatchQueue` (Sources/BatchQueue.swift), Generate ▸ Batch Generate from Images…; one project per image, shape then optional paint, sequential. Driven by polling `runtime.state`; **not run against real models here** — only compiles/unit-covered indirectly.
- Plugins: `ToolPlugin` (Sources/Core/ToolPlugin.swift), Model ▸ Tools; manifest format in docs/PLUGINS.md. Tested with cp/false.
- **Not done:** rigging, pose library, weight painting, animation export, project templates.

## Rigging (minimal)

- `Skeleton.fit` (Sources/Core/Rig.swift): 22-joint humanoid placed from the bbox + width at shoulder height (T-pose if reach > 0.45h, else A/hanging). `SkinWeights.compute`: inverse-distance to bone segments, top 4. `RigPose` is direction-based (bone points along a world vector), `PoseLibrary` has poses; `MeshEdit.pose` bakes one into a new version (popover → Pose).
- `RigGLB` (RigGLB.swift) post-processes `encodeGLB`: skin, JOINTS_0/WEIGHTS_0, IBMs, rotation animation. Export format "GLB · rigged humanoid + animation clip"; clip + speed in Settings → Rigging.
- **Verified only on a synthetic box-figure** (tests/RigTests) — not on a real character scan, not loaded in Blender/Unity. Arms touching the torso share weights; non-humanoids fit badly.
- **Not done:** weight painting, hand-adjusting joints, retargeting, FBX animation, project templates.

### Rig on the real bust scan (result)

The bust (head + torso, h/width ≈ 1.5) is not a figure: the fitted skeleton put hips/legs inside the torso and "Arms down" changed nothing visible. `Skeleton.fit` now returns nil for non-T-pose shapes with h/width < 2, so the pose edit says "couldn't fit a skeleton" and the rigged GLB export throws, instead of producing nonsense. The rig has therefore still never run on a real full-body character — needs one to validate. Opt-in test: `TEST_RUNNER_RIG_MESH=<.mesh> [TEST_RUNNER_RIG_OUT=<dir>]` (the dir must be writable by the test sandbox, e.g. the scratchpad, not docs/).

## Templates + rig tuning (round 3)

- `ProjectTemplate` (built-ins + custom saved via sidebar menu; Settings picks the default): sets quality, paint quality, background removal at `newProject(template:)`.
- `RigConfig` (per project, saved in `Project.rigConfigRaw`): joint offsets, per-joint reach, weight smoothing; "Adjust rig" in the Pose section shows the skeleton in the viewer (drawn through the guide-line layer, so it's yellow). Applied to Bake pose and rigged GLB export.
- `JointNaming` (Settings → Rigging): generic / Mixamo / Unity names in the rigged GLB. Our skeleton has no clavicles or fingers.
- Weight painting: brush strokes (`WeightStroke`, Settings → Rigging for default size/power) ride the guide-drawing recorder; replayed over computed weights in `SkinWeights.paint`. **The brush UI has never been exercised in the live app** (needs a standing figure; the bust is rejected). Painted-weights display is only the stroke lines, not a heat map. **FBX animation** not implemented; plugin recipe in docs/PLUGINS.md (Blender, untested).
- Still unvalidated on a real full-body character.

GLB checks: `testRiggedGLBInvariants` verifies spec-level structure (accessor ranges, joint indices, weights sum, acyclic nodes, animation counts). ModelIO on this macOS can't import GLB, so there is no outside importer check; run Khronos `gltf-validator` or Blender on a rigged export when available.

Validated the rigged GLBs (Walk, Wave, Arms up on the test figure, Mixamo names) with Khronos `gltf-validator` (npm): 0 errors, 0 warnings, 0 infos after setting `target` on the JOINTS_0/WEIGHTS_0 buffer views. That checks file structure only — not how Blender/Unity animate it, and still not on a real character. Re-run: `TEST_RUNNER_RIG_GLB_OUT=<dir>` with `RigTests/testWriteRiggedGLBForValidation`, then validate the files.

### Live-app click-through (Debug build, computer-use)

Verified in the running app: File ▸ New Project from Template (Maximum detail → qualityRaw/paintQualityRaw max, saved to projects.json); Generate ▸ Batch Generate from Images… and Model ▸ Tools menu items exist; Edit-mesh popover shows the Pose section and "Adjust rig" (joint picker, X/Y/Z, reach, smooth, brush, paint); on the bust the figure warning appears and Paint is disabled (the rig now fits when the section opens); Settings → Tools shows 3D printing, Project templates, Rigging, Batch and plugins, Material regions (UV island padding).
Fixed from this pass: guide / weight-paint banners moved to the bottom of the pane (their Done button sat under the toolbar's wand button).
NOT exercised: UV-island checklist (needs a painted version), batch run, a real paint-mode stroke, the template save alert, slicer hand-off. One unexplained blank 3D viewport right after opening/closing Settings with the popover open; re-selecting the project redrew it — not reproduced since. An empty "Untitled 2" test project (Maximum detail template) was left in the bust's project list; delete it.
