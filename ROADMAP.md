# GrizzyModlr Feature Roadmap

Comprehensive plan to evolve from image→3D to a full local 3D creation studio.

## Current State

### Core Strengths
- ✅ Native macOS app (SwiftUI + SceneKit) — fast, polished UI
- ✅ Runs entirely on-device via MLX Swift (no Python)
- ✅ Clean reducer architecture — testable, maintainable
- ✅ Hunyuan3D models (shape + paint with PBR)
- ✅ Background removal + mask touch-up editor
- ✅ Export: GLB, OBJ, STL, PLY
- ✅ Version history for iterations
- ✅ Mesh decimation before paint

## Competitive Landscape

### Modly
- **Stack:** Electron + Python + React
- ✓ Node-based workflows
- ✓ Multi-model support (TripoSG, Trellis, Hunyuan3D)
- ✓ Mesh smoothing/decimation
- ✓ Extension system
- ✓ Gaussian splat viewing
- ✗ Heavy dependencies (Electron + Python)

### ComfyUI / Comfy Desktop
- **Stack:** Python + Node-based UI
- ✓ Huge ecosystem
- ✓ 3D via extensions (Hunyuan3D, TripoSplat)
- ✓ Flexible workflows
- ✗ Complex setup
- ✗ Not native UI

### TripoSR / InstantMesh
- **Stack:** Python CLI + Gradio
- ✓ Very fast (<1s on A100)
- ✓ MIT licensed
- ✗ No editing capabilities
- ✗ Texture/UV issues
- ✗ Web/CLI only

## Common User Pain Points

| Issue | Frequency |
|-------|-----------|
| Poor back-side quality from single image | Very High |
| No mesh topology editing/cleanup | High |
| Texture/UV mapping problems | High |
| Can't control poly count output | High |
| No rigging/animation support | Medium |
| No multi-image input for better accuracy | Medium |
| No real-world scale inference | Medium |

## Proposed Feature Roadmap

### Phase 1: Essential Editing (P1)

#### 1.1 Mesh Editing Tools
- ✅ **Interactive decimation** — slider to reduce poly count post-generation
- ✅ **Mesh smoothing** — Laplacian smoothing with iterations control
- ✅ **Hole filling** — auto-detect and fill holes in generated meshes
- ✅ **Vertex sculpting** — brush-based push/pull, smooth, flatten
- ✅ **Boolean operations** — union, subtract, intersect for combining meshes

#### 1.2 Texture Editing
- ✅ **Texture paint** — paint directly on 3D model surface
- ✅ **Texture inpainting** — AI-assisted fill for bad regions
- ✅ **UV editor** — view and manually adjust UV maps
- ✅ **Material editor** — adjust roughness, metalness, color per-region

#### 1.3 Multi-View Generation
- ✅ **Multi-image input** — use 2-4 reference images for better accuracy
- ✅ **View synthesis** — generate missing views automatically
- ✅ **Reference view alignment** — help user position reference images

### Phase 2: Advanced Modeling (P2)

#### 2.1 Topology Tools
- **Auto-retopology** — clean quad mesh from dense triangles
- **Edge flow editing** — for animation-ready models
- ✅ **LOD generation** — automatic Level of Detail variants
- ✅ **Symmetry enforcement** — mirror edits, fix asymmetry

#### 2.2 Scene Composition
- ✅ **Multi-object scenes** — generate and arrange multiple objects
- ✅ **Ground plane + shadows** — proper scene lighting
- ✅ **HDR environment maps** — realistic lighting preview
- ✅ **Camera presets** — product shots, turntable, etc.

#### 2.3 Additional AI Models
- ✅ **TripoSR** — fast alternative model (MLX port needed)
- ✅ **Stable Zero123** — novel view synthesis
- ✅ **Text-to-3D** — prompt-based generation
- ✅ **Depth estimation** — improve generation from single images

### Phase 3: Pro Features (P3)

#### 3.1 Rigging & Animation
- **Auto-rigging** — AI-based skeleton generation for characters
- **Pose library** — preset poses and animation clips
- **Weight painting** — manual skin weight adjustment
- **Animation export** — FBX with animations

#### 3.2 Extended Export
- ✅ **USDZ** — Apple AR Quick Look
- **FBX** — industry standard with animations
- **3MF** — modern 3D printing format
- **Slicer integration** — direct send to OrcaSlicer/PrusaSlicer

#### 3.3 Workflow Enhancements
- **Batch processing** — process multiple images
- **Project templates** — presets for common use cases
- **Plugin/extension system** — user-contributed tools
- **Keyboard shortcuts** — power-user workflow

## Implementation Notes

### Technical Considerations
- `MeshViewer` already uses SceneKit — extend with Metal shaders for editing
- `MeshDecimator` exists — build smoothing/hole-fill on same pattern
- `MaskEditor` pattern can inform texture paint UX
- Reducer architecture supports undo/redo naturally — leverage for all edits
- Consider `libigl` (via C++ bridge) for advanced mesh ops
- Texture paint could use Metal compute shaders for real-time preview

## Quick Wins (Low Effort, High Value)

| Feature | Effort | Impact |
|---------|--------|--------|
| Interactive decimation slider | Low | High — users want poly control |
| Mesh smoothing | Low | Medium — quick quality fix |
| USDZ export | Low | High — native AR on iOS/macOS |
| Keyboard shortcuts | Low | Medium — power users love this |
| Drag-and-drop multiple images | Medium | High — better quality output |

## Differentiators to Maintain

- **Native macOS** — don't go Electron, stay SwiftUI
- **No Python** — MLX Swift is the competitive advantage
- **Local-first** — privacy and offline capability
- **Clean architecture** — keep the reducer pattern
- **Apple Silicon optimized** — leverage unified memory

## Research Sources

- GitHub: lightningpixel/modly (8.4k stars)
- GitHub: Comfy-Org/Comfy-Desktop
- GitHub: VAST-AI-Research/TripoSR (7.2k stars)
- GitHub: TencentARC/InstantMesh (4.5k stars)
- GitHub: Tencent/Hunyuan3D-2 (15k stars)
- GitHub: xxlong0/Wonder3D (5.4k stars)
- User issues from TripoSR, InstantMesh, Wonder3D repositories

## Status notes (2026-10-30)

Done: Phase 1, Phase 2 except true flow-field retopology and edge-flow editing (axis-aligned quad remesh is in), plus USDZ export. Partial: AI Fill uses SD 2.1 base blended diffusion (not a dedicated inpainting model), material editor can be limited to one texture color (no island or painted regions), booleans are voxel-based (sharp-edge placement, up to 384 voxels). Not started: rigging, FBX, 3MF, slicer integration, batch, templates, plugins.
