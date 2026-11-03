# Pixal3D assessment

Date: 2026-10-30. Branch `claude/pixal3d-cc9f55`.

**Verdict: Pixal3D is not an independent model. It is TRELLIS.2 with a different image-conditioning path, and its only released weights are TRELLIS.2-format checkpoints. A Swift port is the TRELLIS.2 port plus a small delta, so it should not be started separately.** No code beyond this document was written, and no weights were downloaded.

What was read directly, and what was not, is marked throughout. Anything in the "Estimates" section is not a measurement.

## Sources

- Official repo: <https://github.com/TencentARC/Pixal3D> (MIT, 2.6k stars, `master` is the TRELLIS.2 variant, `paper` is the Direct3D-S2 variant).
- Paper: arXiv 2605.10922, <https://arxiv.org/abs/2605.10922>, HTML at <https://arxiv.org/html/2605.10922v1>. SIGGRAPH 2026, accepted Apr 2026.
- Project page: <https://ldyang694.github.io/projects/pixal3d/>
- Weights: <https://huggingface.co/TencentARC/Pixal3D> (the project page also links `Pixal3D-T`, which redirects to the same repo).
- Existing Apple Silicon port, unofficial: <https://github.com/pawel-mazurkiewicz/Pixal3D-mac>. It replaces the CUDA ops with Metal packages. I read only its README.

## License (verified)

- Repo `LICENSE` begins "MIT License, Copyright (c) 2026 Tencent". The HF model has tag `license:mit`, and its `LICENSE` file is 1.06 kB. The HF `NOTICE` (12.5 kB) says third-party components keep their own licenses. I did not read the NOTICE text.
- The pipeline pulls in other models with their own terms that I did not check. They are DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`, gated on HF, Meta DINOv3 license), RMBG-2.0 (`briaai/RMBG-2.0`, Bria's non-commercial license, as I recall) and MoGe (`microsoft/MoGe`). Check these before shipping the app.

## What it is

Per the paper abstract and README (paper text read, code read in part):

- Most image-to-3D models generate in a canonical pose and inject the image through cross-attention. Pixal3D instead generates the shape in the **input camera's frame** ("pixel-aligned").
- Conditioning is by **back-projection**. Each voxel `(i,j,k)` of the generation grid is projected onto the image plane with a pinhole camera. Image features are bilinearly sampled there (multi-scale: DINO patch features plus a feature-upsampled full-resolution map, averaged). The resulting 3D feature volume is **added to the noise volume**. A global DINO token still goes through cross-attention.
- At inference no camera is given. A small FOV and a unit cube are used, and the camera distance is set so the rays through the four image corners hit the four vertices of the cube's back face. The repo calls MoGe for the FOV (`--fov` overrides it, per the Mac port's README; I did not verify this in the upstream code).
- Multi-view works by averaging the back-projected volumes of several views. It needs camera poses and separate `*_mv` weights.
- In code, the delta from TRELLIS.2 is `SparseProjectAttention`: `proj_linear(proj_feats) + cross_attn(global)` in each block, with `proj_in_channels: 2048` in the shape DiT config. This is in `pixal3d/modules/sparse/attention/proj_attention.py` (read).

## Architecture of the released weights (verified from HF configs and `pipeline.json`)

`pipeline.json` has `"name": "Trellis2ImageTo3DPipeline"`. The README install step 1 is "Follow TRELLIS.2 installation". The README acknowledgements say the code is "heavily built upon Trellis.2 and Direct3D-S2". Pixal3D is a three-stage cascade:

| Stage | Model | Config |
|---|---|---|
| 1. Sparse structure | `SparseStructureFlowModel` (dense flow DiT) at 16³ latent, 8 channels, then `ss_dec_conv3d` decodes to a 64³ occupancy grid | 30 blocks, 1536 ch, 12 heads, `image_attn_mode: proj`, 12 Euler steps, CFG 7.5 |
| 2. Shape | `ElasticSLatFlowModel`, sparse flow DiT at 512 and 1024 (cascade, optional 1536), 32-ch structured latent | 30 blocks, 1536 ch, 12 heads, rope, qk-RMSnorm, `proj` conditioning |
| 2b. Shape decode | `FlexiDualGridVaeDecoder` (ConvNeXt blocks, 4 conv-to-subdivide upsamples, 1024/512/256/128/64 ch) | outputs 7 channels per voxel |
| 3. Texture (skipped for us) | `slat_flow_imgshape2tex` DiT plus `tex_dec` | PBR attributes |

**The geometry representation is not an SDF.** The decoder emits a Flexible Dual Grid per sparse voxel: 3 vertex offsets (`sigmoid`), 3 edge-intersection logits, and 1 quad-lerp weight (`softplus`). `o_voxel.convert.flexible_dual_grid_to_mesh` turns these into a mesh. This is TRELLIS.2's "O-Voxel" format. `SparseGrid.swift` and `MarchingCubesSparse.swift` therefore **do not apply**. The mesher is a different algorithm (dual-grid face construction, not marching cubes). The decoder also subdivides, so intermediate sparse structures must be produced.

## Weights (verified from the HF API)

Repo revision `b0cb2e1b794cab9aa0ac38a95d794a4d9337437f`. Sizes are in bytes.

| File | Size |
|---|---|
| `ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors` | 5,359,822,584 |
| `ckpts/ss_dec_conv3d_16l8_fp16.safetensors` | 147,591,972 |
| `ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors` | 5,546,764,048 |
| `ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors` | 5,546,764,048 |
| `ckpts/shape_dec_next_dc_f16c32_fp16.safetensors` | 948,490,494 |
| `ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors` (texture) | 5,546,960,656 |
| `ckpts/tex_dec_next_dc_f16c32_fp16.safetensors` (texture) | 948,458,812 |
| `*_mv` variants of the ss / 512 / 1024 / tex DiTs | same sizes as above, one per stage |

- Geometry-only, single-view needs about **17.5 GB** (ss 5.36, ss dec 0.15, shape 512 5.55, shape 1024 5.55, shape decoder 0.95). Add DINOv3 ViT-L (about 0.6 GB fp16, estimate), RMBG-2.0 (about 0.9 GB, estimate), MoGe-2, and the feature upsampler.
- Correction (verified later): the Pixal3D DiT files are 5.5 GB because they are stored in fp32, not because they are bigger. The matching TRELLIS.2 bf16 file is 2,584,574,424 bytes; doubling it and adding 30 per-block `proj_linear` layers (2048x1536, fp32, about 0.38 GB) gives about 5.55 GB, which matches. So each DiT is about 1.3 B parameters, as the name says. Pixal3D's `shape_dec` file has the same size and LFS oid as TRELLIS.2's, so that decoder is identical.
- File names and sizes are verified. The SHA-256 values are the LFS oids and would need to be recorded when the files are actually downloaded.

## Does it depend on TRELLIS.2?

Yes, completely. The `main` branch is the TRELLIS.2 pipeline and the only released weights are for it. The stack is: sparse flow DiTs with elastic mixin, RoPE and qk-RMSnorm, the sparse ConvNeXt/C2S decoder, the FDG mesher, DINOv3 conditioning and the sampler (`FlowEulerGuidanceIntervalSampler`). It matches the TRELLIS.2 stack except for the `proj_linear` additions, the projection sampler, and the camera selection.

What a TRELLIS.2 port (`trellis2-port`) would share with Pixal3D:
- Sparse tensor type, sparse attention (full and windowed), sparse conv, the sparse DiT blocks.
- The dense and sparse flow models, samplers (Euler with guidance interval), and the FDG decoder.
- The FDG mesher, DINOv3 and background removal.

What Pixal3D adds on top:
1. Per-block `proj_linear` (2048 to 1536) in the DiTs.
2. A voxel-to-image projection and bilinear feature sampler (small, dense, easily done in MLX).
3. A feature upsampler for the full-resolution feature map (the paper cites Chambon et al. 2025 and the Mac port lists `natten-mps` for it, so it likely needs neighborhood attention; not verified).
4. MoGe FOV estimation, or a fixed FOV.
5. Different checkpoint tensors. The `proj` DiT checkpoints have extra keys, so the TRELLIS.2 Swift module definitions need an optional `proj_linear`.

Recommendation: do the TRELLIS.2 port first. Pixal3D geometry then costs roughly the delta above (items 1, 2, 5, and 3 and 4 if wanted). I did not start this because that session owns the shared code, and building it in parallel would duplicate or conflict with it.

## Operators and the Metal / MLX route

| CUDA op | Used for | MLX / Metal route |
|---|---|---|
| `flex_gemm` submanifold sparse conv | the whole shape decoder, and the VAE | Not in MLX. Needs a hash-map neighbor build plus gather-GEMM. A Metal version already exists: `mtlgemm` (pedronaugusto), vendored by the Mac port. In MLX it is a gather over a coordinate hash then batched matmul, doable but the main performance risk. |
| `flash_attn` / varlen sparse attention | the DiTs | `mx.fast.scaled_dot_product_attention` with per-sample padding or chunking. The Mac port reports an MPS fused-SDPA accuracy problem above about 18k tokens and works around it with chunked fp32 attention. Plan for fp32 accumulation or chunking at the large stages. |
| windowed sparse attention | the shape DiT (if enabled by config) | Re-expressible with gather and padding. |
| `o_voxel` / `cumesh` meshing | `flexible_dual_grid_to_mesh`, cleanup | CUDA only. Needs a reimplementation (CPU Swift is acceptable for geometry-only; the Mac port uses `mtlmesh`). |
| `natten` neighborhood attention | feature upsampler (inferred) | MLX has no op. Needs a custom kernel or a dense-with-mask fallback. The Mac port ships `natten-mps`. |
| `nvdiffrast` | PBR texturing only | Out of scope for geometry-only. |

## Estimates (not measured)

- Weights memory, geometry-only, bf16: about 17.5 GB of DiTs and decoders plus about 2 GB of image models. This fits comfortably in this machine's 96 GB (`hw.memsize` is 103,079,215,104). It would be tight on a 16 GB Mac.
- Time: the Mac port author reports "several minutes" for a 1024 cascade and up to 50 minutes for the 1536 cascade with a 2048² texture on an M5 Max with 128 GB. These are their numbers, and they include texturing. A geometry-only 1024 run in MLX is likely minutes, dominated by 12 steps × 2 (CFG) × the 1024-level sparse DiT. I have no measurement of my own.

## Quality claims (authors' claims, not measured by us)

From the paper's Table 1 (Toys4K, normal-map comparison, rendered in the input camera frame): Hunyuan3D-2.1 gets IoU 83.33 and mean normal error 21.19°, TRELLIS gets 79.48 and 25.00°. The Pixal3D row was cut off in my fetch, but the text says it improves "across all metrics". Table 2 (150 in-the-wild images, user study from 1 to 5) gives fidelity 4.91 and quality 4.74 for Pixal3D, against 2.77 and 2.50 for Hunyuan3D-2.1. Uni3D 42.11 and ULIP2 45.04 versus 41.15 and 44.65 for Hunyuan3D-2.1.

- **The paper does not compare to TRELLIS.2.** Its baselines are TRELLIS (v1), TripoSG, Hunyuan3D-2.1 and Direct3D-S2. The paper variant is built on Direct3D-S2, while the released `main` is on TRELLIS.2 and the README only says "better performance".
- The advantage is mostly a fidelity advantage: the mesh lines up with the input image in the input view. That is what the metrics measure, and it does not directly measure geometry quality in occluded regions.
- The multi-view numbers (Chamfer 4.73e-4 for 4 views vs 18.35e-4 for TRELLIS) are for Toys4K and need known cameras.

## The paper branch

The `paper` branch uses Direct3D-S2 (a sparse SDF model, which would suit our SDF-oriented code). I found no weights for it on HF: `TencentARC/Pixal3D` holds only TRELLIS.2-format checkpoints. I did not search elsewhere. So it is not usable.

## Decision

Not feasible as a standalone port. It is feasible as an add-on after the TRELLIS.2 port, with a moderate delta. The hard parts are the same as for TRELLIS.2: sparse conv, sparse attention at scale and O-Voxel meshing.

Cheapest ways to get the benefit now:
1. **Do TRELLIS.2 first**, with `proj_linear` and the projection sampler designed in as optional (so the same Swift DiT loads either checkpoint family). Then Pixal3D geometry is a weights-catalog entry plus the delta.
2. **Or run the existing Mac port as an external helper process** for evaluation only, to judge the quality before spending porting effort. Its stack is Python with Metal externs, so it would not be shippable inside the app, and I have not run it.

## Not done / untested

- Nothing was downloaded or run. No numerical parity, no timing, no memory measurement.
- Not read: the Mac port's `PIPELINE_MAP.md` and kernel code, `o_voxel`, the NOTICE text, the licenses of DINOv3, RMBG-2.0 and MoGe, and the details of the feature upsampler.

## NAF upsampler (needed by the shape stages), assessed 2026-10-31

Pixal3D's shape DiTs take 2048-d `proj` features: DINOv3 patch features concatenated with the same features upsampled by NAF (`valeoai/NAF`, `use_naf_upsample=True`, target 512 px at the 512 level). The structure stage does not use it.

- Source read from <https://github.com/valeoai/NAF> (Apache-2.0; `src/layers/rope.py` carries a Meta header saying it follows the DINOv3 License Agreement, which I have not reviewed). Weights: `naf_release.pth`, 2,664,431 bytes (GitHub release `model`). That is about 2.7 MB, not hundreds of MB.
- Architecture is small and has no learned q/k/v projections: two conv encoders on the RGB image (1x1-kernel and 3x3-kernel branches, each a conv + 2 GroupNorm/SiLU/conv blocks, 128 channels each, reflect padding) concatenated to 256 channels, adaptive-avg-pooled to the output size, axial RoPE (base 100, coordinate rescale only in training), then 4-head 9x9 neighbourhood cross-attention. Queries are the RoPE'd image features at output size, keys are the same features pooled to the DINOv3 grid, values are the DINOv3 features. Keys and values are nearest-exact resized to the query grid, and the attention uses dilation = query size / key size.
- The only awkward piece is the `natten` kernel (CUDA/CPU extension, which we do not have). Its edge rule clamps the 9x9 window inside the map instead of zero padding; that must be reproduced exactly. In MLX it is a gather over 81 neighbours per query. At 512x512 that is 262k queries x 81 x 4 heads x 64 dims, which is workable in chunks, and Pixal3D only samples the map at the voxel projections, so a lazy per-voxel version is possible.
- Verdict: feasible, roughly a day of work including a standalone PyTorch reference (needs a pure-PyTorch replacement for `natten` in the reference, checked against the clamp rule). Not started.

## Port status (2026-10-31, branch claude/pixal3d-cc9f55)

Measured on this Mac (M-series, 96 GB), fp32, one example image (a crown), seed 42, 12+12 steps, CFG on. Single runs, not benchmarks.

| Pipeline | Time | Peak MLX memory | Output |
|---|---|---|---|
| TRELLIS.2 512 (black background, no margin) | 55 s | 8.7 GB | 1.67 M verts, 3.83 M faces, clean crown |
| Pixal3D 512 stage (FOV 0.8576, distance derived 1.09) | 39 s | 8.2 GB | 0.74 M verts, 1.51 M faces, coherent crown |

- Parity vs PyTorch CPU fp32 (cosine): NAF 1.0 (against my reference, with NATTEN's window rule as I read it), projection sampler 1.0, Pixal3D structure DiT with `proj_linear` 1.0. Earlier modules are listed in the commit history. The shape DiT with `proj_linear` is run end to end but has no separate parity fixture.
- End-to-end quality was judged by eye from a depth-shaded render, no numeric comparison against the PyTorch pipeline's mesh. Pixal3D's mesh is in its own y-up frame (TRELLIS.2's is z-up).
- A first Pixal3D run with a placeholder camera distance of 2.0 gave fragmented geometry. Pixal3D derives the distance from the FOV (`distance_from_fov`), and that fixed it.
- Not ported: MoGe-2 (FOV estimation; `Ruicheng/moge-2-vitl`, MIT, `model.pt` 1,305,030,700 bytes, DINOv2-L encoder plus conv heads plus a focal solver), the 1024/1536 cascade levels (needs the 1024 DiT, 5.55 GB), texturing, bf16, background removal (the example image already had alpha), multi-view, and all app wiring.
