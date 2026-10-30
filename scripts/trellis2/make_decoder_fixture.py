"""Standalone PyTorch reference for the TRELLIS.2 shape decoder (FlexiDualGridVaeDecoder trunk), CPU fp32.

Written from microsoft/TRELLIS.2 `sparse_unet_vae.py` (SparseConvNeXtBlock3d, SparseResBlockC2S3d,
SparseUnetVaeDecoder.forward) without importing that repo. Loads the real checkpoint and writes a small
fixture (input latent, intermediate stage outputs, subdivision logits, final 7 channels) for the Swift test.

  /Volumes/Models/Modelr/venvs/trellis2-ref/bin/python -I scripts/trellis2/make_decoder_fixture.py CKPT.safetensors OUT_DIR
"""
import json, sys, os
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ckpt, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
cfg = dict(model_channels=[1024, 512, 256, 128, 64], num_blocks=[4, 16, 8, 4, 0])
W = {k: v.float() for k, v in load_file(ckpt).items()}


def subm_conv(coords, feats, w, b):
    """Submanifold 3x3x3 conv, w is (Co,3,3,3,Ci): out[i] = b + sum_k w[:,k] @ feats[nbr(i,k)]."""
    n = coords.shape[0]
    S = int(coords[:, 1:].max()) + 2
    key = lambda c: ((c[:, 0] * S + c[:, 1]) * S + c[:, 2]) * S + c[:, 3]
    k0 = key(coords); order = torch.argsort(k0); sk = k0[order]
    out = torch.zeros(n, w.shape[0])
    for kx in (-1, 0, 1):
        for ky in (-1, 0, 1):
            for kz in (-1, 0, 1):
                nc = coords.clone(); nc[:, 1] += kx; nc[:, 2] += ky; nc[:, 3] += kz
                ok = (nc[:, 1:] >= 0).all(1)
                nk = key(nc.clamp(min=0)); pos = torch.searchsorted(sk, nk).clamp(max=n - 1)
                hit = ok & (sk[pos] == nk)
                f = torch.where(hit[:, None], feats[order[pos]], torch.zeros(()))
                out += f @ w[:, kx + 1, ky + 1, kz + 1, :].T
    return out + b


def layer_norm(x, p=None, eps=1e-6):
    return F.layer_norm(x, x.shape[-1:], None if p is None else W[p + '.weight'], None if p is None else W[p + '.bias'], eps)


def linear(x, p):
    return x @ W[p + '.weight'].T + W[p + '.bias']


def convnext(coords, x, p):
    h = subm_conv(coords, x, W[p + '.conv.weight'], W[p + '.conv.bias'])
    h = layer_norm(h, p + '.norm')
    h = F.silu(linear(h, p + '.mlp.0'))
    h = linear(h, p + '.mlp.2')
    return h + x


def c2s_block(coords, x, p, cin, cout):
    sub = linear(x, p + '.to_subdiv')                       # [N, 8] logits
    h = F.silu(layer_norm(x, p + '.norm1'))
    h = subm_conv(coords, h, W[p + '.conv1.weight'], W[p + '.conv1.bias'])   # [N, cout*8]
    keep = sub > 0                                           # [N, 8]
    ridx, sidx = keep.nonzero(as_tuple=True)                 # row-major (parent, slot)
    new_coords = coords[ridx].clone()
    new_coords[:, 1:] *= 2
    for i in range(3):
        new_coords[:, i + 1] += (sidx // 2 ** i) % 2
    hn = h.reshape(h.shape[0] * 8, -1)[ridx * 8 + sidx]      # [M, cout]
    xn = x.reshape(x.shape[0] * 8, -1)[ridx * 8 + sidx]      # [M, cin/8]
    hn = F.silu(layer_norm(hn))                              # norm2 has no affine
    hn = subm_conv(new_coords, hn, W[p + '.conv2.weight'], W[p + '.conv2.bias'])
    skip = xn.repeat_interleave(cout // (cin // 8), dim=1)
    return new_coords, hn + skip, sub


# deterministic latent on the 16^3 grid
g = torch.Generator().manual_seed(11)
pts = torch.unique(torch.randint(4, 12, (36, 3), generator=g), dim=0)
coords = torch.cat([torch.zeros(len(pts), 1, dtype=torch.long), pts], 1)
z = torch.randn(len(pts), 32, generator=g)

stages, subs = [], []
h = linear(z, 'from_latent')
mc, nb = cfg['model_channels'], cfg['num_blocks']
for i, n in enumerate(nb):
    for j in range(n):
        h = convnext(coords, h, f'blocks.{i}.{j}')
    if i < len(nb) - 1:
        coords, h, s = c2s_block(coords, h, f'blocks.{i}.{n}', mc[i], mc[i + 1])
        subs.append(s)
    stages.append((coords.clone(), h.clone()))
h = F.layer_norm(h, h.shape[-1:])
out = linear(h, 'output_layer')


def arr(t):
    return t.detach().numpy().astype(np.float32).ravel().tolist() if t.is_floating_point() else t.numpy().astype(np.int32).ravel().tolist()


fx = dict(latent_coords=arr(torch.cat([coords[:0]]) if False else torch.cat([torch.zeros(len(pts), 1, dtype=torch.long), pts], 1)),
          latent=arr(z), n_latent=len(pts),
          final_coords=arr(coords), final=arr(out), n_final=int(coords.shape[0]),
          sub_counts=[int(s.shape[0]) for s in subs], stage_counts=[int(c.shape[0]) for c, _ in stages],
          sub0=arr(subs[0]), stage0_feats=arr(stages[0][1]))
json.dump(fx, open(os.path.join(out_dir, 'decoder_fixture.json'), 'w'))
print('latent voxels', len(pts), 'stage counts', fx['stage_counts'], 'final', fx['n_final'])
print('out std', out.std().item(), 'sub0 pos frac', (subs[0] > 0).float().mean().item())
