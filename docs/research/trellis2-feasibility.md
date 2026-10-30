# TRELLIS.2 → MLX Swift: feasibility

Date: 2026-10-30. Machine: Apple M3 Ultra, 96 GiB unified memory (`hw.memsize` = 103,079,215,104).
Status of claims: **verified** = measured/tested here; **sourced** = read from the reference code or a
published README; **estimate** = my arithmetic, not measured.

## Verdict

**Feasible, and cheaper than it first looked, but a multi-week port rather than a multi-day one if done
to the standard of the Hunyuan path.** Three things decide this:

1. The one operator I expected to be a blocker, sparse 3D convolution, is not. TRELLIS.2's
   `flex_gemm` backend supports only stride-1 submanifold 3×3×3 convolution (`conv_flex_gemm.py` asserts
   `stride == 1`). Down/up-sampling is done by rearranging voxels between space and channels
   (`SparseSpatial2Channel` / `SparseChannel2Spatial`), which is plain index arithmetic. I implemented
   both in MLX Swift and they match PyTorch (Phase 1 below, **verified**).
2. Attention in the 1.3B flow DiTs is dense full self-attention over one token sequence (`attn_mode='full'`),
   so `MLXFast.scaledDotProductAttention` covers it. No varlen/windowed kernel is needed at batch 1.
3. **Prior art exists, including an MLX-native port** (`lyonsno/trellis2mlx`, MIT, Python, very active,
   500+ commits) and a PyTorch-MPS port (`shivampkumar/trellis-mac`, MIT). Both show the model runs on a Mac.
   The MLX port is the best reference for a Swift port and a source of ground truth for parity tests.

The real obstacles are not numerical. They are (a) **licensing/gating of two non-TRELLIS inputs** and
(b) **the amount of surrounding code** (two DINOv3/BiRefNet-class vision models, a 2-stage cascade,
dual-grid meshing). See "Risks".

## Pipeline, stage by stage (geometry-only marked ★)

Files are in `microsoft/TRELLIS.2-4B` (revision `af44b45f2e35a493886929c6d786e563ec68364d`, MIT) unless noted.
Sizes are the LFS sizes from the Hugging Face API (**sourced**).

| # | Stage | Weights | Size | Params (approx) | Geometry? |
|---|-------|---------|------|-----------------|-----------|
| 0 | Background removal, BiRefNet | `briaai/RMBG-2.0` | ~0.9 GB (fp32) | ~220 M | ★ (any matting works) |
| 1 | Image conditioner, DINOv3 ViT-L/16 | `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated) | ~1.2 GB (fp32) | ~300 M | ★ |
| 2 | Sparse-structure flow DiT (16³×8 latent, 4096 tokens) | `ckpts/ss_flow_img_dit_1_3B_64_bf16` | 2.58 GB | 1.3 B | ★ |
| 3 | Sparse-structure decoder (dense Conv3d, from `microsoft/TRELLIS-image-large`) | `ckpts/ss_dec_conv3d_16l8_fp16` | 0.15 GB | ~75 M | ★ |
| 4 | Shape structured-latent flow DiT, 512 (32³ grid) | `ckpts/slat_flow_img2shape_dit_1_3B_512_bf16` | 2.58 GB | 1.3 B | ★ |
| 4b | Same at 1024, used for the cascade second pass | `ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16` | 2.58 GB | 1.3 B | ★ (for ≥1024) |
| 5 | Shape decoder, FlexiDualGrid VAE (sparse ConvNeXt U-Net, 16× up) | `ckpts/shape_dec_next_dc_f16c32_fp16` | 0.95 GB | ~470 M | ★ |
| 6 | O-Voxel (flexible dual grid) → mesh | none (`o_voxel` C++/CUDA) | – | – | ★ |
| – | Shape encoder | `shape_enc_next_dc_f16c32_fp16` | 0.71 GB | | not needed (training / shape-conditioned texturing) |
| – | Texture flow DiT 512 / 1024 | `slat_flow_imgshape2tex_*` | 2.58 GB ×2 | 1.3 B ×2 | out of scope |
| – | Texture decoder / encoder | `tex_dec_*`, `tex_enc_*` | 0.95 / 0.71 GB | | out of scope |

Geometry-only download: about **10–11 GB** (2.58×3 + 0.95 + 0.15 + DINOv3 + RMBG-2.0), versus ~16 GB for
the full repo plus the two gated models. Parameter counts for the two non-flow stages are from the model
cards/configs, not counted from tensors (**estimate**).

Pipeline defaults (`pipeline.json`, **sourced**): 12 Euler steps per stage; classifier-free guidance 7.5
(rescale 0.7 for structure, 0.5 for shape) applied only while t∈[0.6, 1.0]; `rescale_t` 5.0 / 3.0;
shape latents are normalised with a fixed 32-channel mean/std table in `pipeline.json`;
default pipeline type `1024_cascade` (512 pass → upsample/prune coords → 1024 pass).

Mesh output: the decoder emits per-voxel 7 channels: 3 dual-vertex offsets (`1.5·sigmoid − 0.5`),
3 edge-intersection logits (`>0`), 1 quad-split weight (`softplus`). `flexible_dual_grid_to_mesh` joins
the dual vertices of voxels sharing an intersected edge into quads, split into triangles by the weight.
No marching cubes, and it can produce open/non-manifold surfaces.

## Operators that are CUDA-only or not in MLX

| Reference dependency | Used for | Mac plan | Status |
|---|---|---|---|
| `flex_gemm` `sparse_submanifold_conv3d` (Triton) | every conv in the decoder (~70 sparse convs, 5 stages) | gather/scatter: neighbour table from a lookup of active voxels, then 27 gather+matmul taps accumulated | **done, parity-tested** (below) |
| `spconv` / `torchsparse` | alternative conv backends | not needed | – |
| `SparseSpatial2Channel` / `Channel2Spatial` | down/up-sampling | sort-based unique + scatter/gather | **done, parity-tested** |
| `flash_attn` varlen | sparse-token attention | batch 1 ⇒ one sequence: `MLXFast.scaledDotProductAttention` (fused, no N² matrix) | not yet written; standard op |
| 3D RoPE on voxel coords (`rope.py`) | DiT blocks | elementwise sin/cos on coordinates, easy | not yet written |
| Dense `Conv3d` (structure decoder) | stage 3 | `MLXNN.Conv3d` | not yet written |
| `o_voxel` hashmap + `flexible_dual_grid_to_mesh` | O-Voxel → mesh | port to Swift on CPU (hash dictionary) or MLX sort-join. `trellis-mac` does it in ~8 s of pure Python for ~800k tris; Swift will be faster | not yet written |
| `cumesh` (hole fill, simplify, UV unwrap) | post-processing | optional. Skip for first version (both Mac ports skip hole filling). Existing app mesh cleanup/simplification can be reused. | – |
| `nvdiffrast`, `nvdiffrec` | PBR render/bake | texturing is out of scope | – |
| Neighbour lookup at ≥1024³ | decoder at 1024/1536 | dense index grid is 4 GB at 1024³ (OK here, wasteful). Replace with sorted-key binary search or a small Metal hash kernel via `MLXFast.metalKernel`; `SparseGrid.swift` block-sparse tables in this repo are an alternative | prototype uses the dense grid (≤ ~512³) |

## Phase 1: sparse convolution prototype (**verified**)

Code: `Packages/Hy3DMLX/Sources/Hy3DMLX/SparseConv3D.swift`
(`submanifoldNeighbors`, `submanifoldConv3d`, `spatialToChannel`, `channelToSpatial`).
Tests: `Packages/Hy3DMLX/Tests/Hy3DMLXTests/SparseConv3DTests.swift`.
Reference: `scripts/trellis2/make_sparse_conv_fixture.py`, run with PyTorch 2.15 (CPU) from a venv at
`/Volumes/Models/Modelr/venvs/trellis2-ref`; output committed as a 250 KB JSON fixture
(2 batches, 318 voxels, 8→6 channels, TRELLIS.2 weight layout `[Co,3,3,3,Ci]`).

* Submanifold conv, dilation 1 and 2: max |Δ| < 1e-4 against `F.conv3d` masked to active sites
  (the definition of submanifold conv). Spatial2Channel: parent coordinates identical, features within 1e-6,
  and `channelToSpatial` round-trips exactly. All pass; the whole package suite (8 tests, 1 pre-existing skip) is green.
* Speed (M3 Ultra, fp32, **measured**): 400,000 active voxels on a 256³ shell, 128→128 channels:
  neighbour table 0.04 s, convolution 0.12 s (≈ 3 TFLOP/s effective).

Caveats, stated plainly:
* I could not run `flex_gemm` (CUDA/Triton), so the reference is the mathematical definition via dense `conv3d`,
  not the `flex_gemm` output itself. The kernel orientation assumption is that `weight[:, kd, kh, kw, :]`
  multiplies the neighbour at `coord + (k−1)` on axes (1,2,3), which matches `spconv` and cross-correlation.
  It must be re-checked against real checkpoint weights in Phase 2 (a decoder parity test on a known latent
  will catch a flipped kernel immediately).
* Not yet optimised: fp32 only, no fp16/bf16, no neighbour-table cache across the ~25 convs sharing a lattice,
  and the neighbour table uses a dense lookup. Each is a mechanical change.

## Memory and time on this Mac (**estimates** unless stated)

Reference timings (**sourced**, README, H100): 512³ ≈ 3 s total (2 s shape + 1 s material); 1024³ ≈ 17 s
(10 s + 7 s); 1536³ ≈ 60 s. Mac measurements published by the ports (**sourced**, not reproduced here):
`trellis-mac` on M4 Pro 24 GB, 512 pipeline, shape+texture: 12-step shape SLat 22 s, shape decoder ~20 s,
~18 GB peak. `trellis2mlx` on M4 Max: ~3 min for a 512 textured object; a 768 cascade reached 63.8 GiB peak.

My arithmetic:

* Flow DiT forward ≈ 2·1.3e9·N flops for the linear layers plus 4·N²·1536·30 for attention. At N≈20k tokens
  (512 grid, typical object) that is ~125 TFLOP per forward; ×12 steps, with guidance only in the
  last 40% of the trajectory, ≈ 1,800–2,400 TFLOP per shape stage. M3 Ultra bf16 matmul realistically sustains
  10–20 TFLOP/s ⇒ **~1.5–3 min for the 512 shape stage; the 1024 pass (N≈80k, attention-dominated) maybe 6–15 min.**
  Roughly 25–40× slower than H100, consistent with H100's 2 s / 10 s figures.
* Structure stage (4096 tokens) is cheap: seconds to tens of seconds.
* Shape decoder ≈ 50–60 TFLOP at 512 (my voxel counts per level are guesses) ⇒ ~10–20 s; ~4× that at 1024.
  `trellis-mac` measured ~20 s on an M4 Pro, which supports the order of magnitude.
* Peak memory: weights 5–8 GB resident at a time (stages can be unloaded between uses), activations of the
  decoder at 1024 (millions of voxels × 64–128 ch) 10–30 GB. **512: ~15–20 GB; 1024 cascade: ~30–60 GB.** Fits
  this 96 GiB machine; the 1024+ path would not be comfortable below 48 GB.

Net: usable for **512 and cascaded 1024 on a high-memory Mac as a patient, minutes-per-asset mode**, not as an
interactive default. For comparison the app's Hunyuan3D path is far faster.

## Prior art

| Project | What it is | Reuse |
|---|---|---|
| [lyonsno/trellis2mlx](https://github.com/lyonsno/trellis2mlx) | MLX-native Python port: DINOv3 conditioning, sparse structure, shape/texture flow with cascade, mesh extraction, UV/bake, GLB. MIT. Last commit 2026-10-25. Unit-tested; claims parity work (`docs/validation.md`, `architecture-map.md`). | **Best reference.** Port module by module to Swift, use its saved intermediates as test fixtures. Its quantisation experiment (4-bit flow, 6.4× smaller) did not speed up the M2 Pro flow stage. |
| [shivampkumar/trellis-mac](https://github.com/shivampkumar/trellis-mac) | PyTorch-MPS patches around the reference repo; `conv_none.py` gather/scatter sparse conv, pure-Python dual-grid mesher | Documents the same sparse-conv approach and the mesh-extraction algorithm. Good source for the reference PyTorch-on-MPS environment for parity tests. |
| pedronaugusto `mtlgemm` / `trellis2-apple` / `mtlmesh` / `mtlbvh` | Metal sparse conv, CPU fork of `o_voxel`, mesh tools | Metal sparse-conv kernels are a ready template if the gather/matmul path proves too slow. Note `mtlgemm` ships an MSL 4.0 metallib (macOS 26+). |
| Microsoft upstream | CUDA only; open issue asking for non-CUDA | none |

## Risks and blockers

1. **Licences / gating for the app.** TRELLIS.2 itself is MIT. But the stock pipeline also needs
   `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated, Meta custom licence, requires accepting terms and an HF login)
   and `briaai/RMBG-2.0` (CC BY-NC 4.0, **non-commercial**). The app's catalog downloads pinned public files;
   gated weights need an authenticated download flow. DINOv3 cannot be swapped for the existing DINOv2: the DiT was
   trained on DINOv3 features. RMBG-2.0 can be replaced by any matting step the app already has (the DiT only needs
   an RGBA cutout), which removes the non-commercial dependency.
2. **Cost of the long tail.** Besides the DiTs and decoder there is a new ViT (DINOv3: RoPE, register tokens,
   LayerScale) and, if kept, BiRefNet (Swin + deformable convs). Then the cascade glue (upsample decoder subdivision
   → prune → recompute coordinates → re-run at 1024) and the dual-grid mesher. `trellis2mlx` needed 500 commits,
   mostly in finishing; its README reports Metal command-buffer failures on complex runs.
3. **Numerical drift.** bf16 flow models with 12-step Euler and CFG rescale are chaotic; "cosine > 0.999 per module" is
   achievable per block, but end-to-end the outputs will not be bit-equal. Test per module and per step with fixed inputs.
4. **Output quality is not proven for this app.** The model's detail claim is from its paper; I did not generate any mesh.

## Recommended Phase 2 order (not started)

1. Load `shape_dec` weights and run the decoder on a latent captured from the PyTorch/`trellis2mlx` run (needs `SparseConvNeXtBlock3d`,
   `SparseResBlockC2S3d`, LayerNorm/GroupNorm variants on sparse features). Gate: cosine > 0.999 on the 7 output channels.
   This also settles the kernel-orientation caveat above.
2. Dual-grid → mesh in Swift; compare vertex/face counts against the reference on a decoder output.
3. SLat flow DiT 512 (modulated blocks, 3D RoPE, cross-attention to 1024-dim DINOv3 tokens), one forward-pass parity test.
4. Sparse-structure DiT + dense decoder; Euler/CFG sampler (reuse `Sampler.swift` patterns).
5. DINOv3 + a matting step.
6. Cascade to 1024, then app wiring following `git show 8b68c2a` (`ModelID`, `CatalogModel` with pinned revision and sha256,
   `ShapeModel` case, `ModelInfo` noting **MIT** for TRELLIS.2 and the gated/NC terms of the two helpers, install gating, tests),
   with the resolution, steps and guidance exposed in Settings.

Rough effort for a competent port using `trellis2mlx` as the oracle: steps 1–2 about 3–5 days, step 3–4 about 4–6 days,
step 5 about 3–4 days, step 6 about 3–5 days. This is an **estimate**, not something I tested.

## Cheapest alternatives if the port is not worth it

* Run `trellis2mlx` as an external Python tool and import its GLB. Zero porting, same model, but needs a Python/uv
  environment and an HF login for DINOv3.
* Keep investing in the existing Hunyuan3D 2.0 path (already running natively in this app), whose detail is already
  being improved by the sparse 2048³ decoder.
