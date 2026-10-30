"""Standalone PyTorch references for (1) the TRELLIS.2 sparse-structure decoder (dense Conv3d U-Net, CPU fp32,
real checkpoint) and (2) the flow-Euler sampler with guidance interval + CFG rescale, driven by a toy model so the
timestep schedule and guidance math are tested in isolation.

Written from microsoft/TRELLIS.2 `sparse_structure_vae.py`, `modules/spatial.py` and `pipelines/samplers/*`
without importing that repo.

  /Volumes/Models/Modelr/venvs/trellis2-ref/bin/python -I scripts/trellis2/make_ssdec_sampler_fixture.py CKPT OUT_DIR
"""
import json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ckpt, out_dir = sys.argv[1], sys.argv[2]
W = {k: v.float() for k, v in load_file(ckpt).items()}


def cnorm(x, p):                      # ChannelLayerNorm32: layer norm over channels, eps 1e-5
    return F.layer_norm(x.permute(0, 2, 3, 4, 1), (x.shape[1],), W[p + '.weight'], W[p + '.bias'], 1e-5).permute(0, 4, 1, 2, 3)


def conv(x, p, pad=1):
    return F.conv3d(x, W[p + '.weight'], W[p + '.bias'], padding=pad)


def res(x, p):
    h = conv(F.silu(cnorm(x, p + '.norm1')), p + '.conv1')
    h = conv(F.silu(cnorm(h, p + '.norm2')), p + '.conv2')
    return h + x


def shuffle(x, s=2):
    B, C, H, Wd, D = x.shape
    c = C // s ** 3
    x = x.reshape(B, c, s, s, s, H, Wd, D).permute(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(B, c, H * s, Wd * s, D * s)


g = torch.Generator().manual_seed(3)
z = torch.randn(1, 8, 16, 16, 16, generator=g)
h = conv(z, 'input_layer')
for i in range(2):
    h = res(h, f'middle_block.{i}')
for i in range(8):
    if i in (2, 5):
        h = shuffle(conv(h, f'blocks.{i}.conv'))
    else:
        h = res(h, f'blocks.{i}')
h = F.silu(F.layer_norm(h.permute(0, 2, 3, 4, 1), (32,), W['out_layer.0.weight'], W['out_layer.0.bias'], 1e-5).permute(0, 4, 1, 2, 3))
out = conv(h, 'out_layer.2')
print('ssdec out', tuple(out.shape), 'occupied', (out > 0).float().mean().item())

# ---- sampler ----
sig = 1e-5


def toy(x, t1000, cond):
    return torch.tanh(0.7 * x + (0.3 if cond else -0.2) + 0.5 * t1000 / 1000.0)


def pred_to_x0(x, t, p):
    return (1 - sig) * x - (sig + (1 - sig) * t) * p


def x0_to_pred(x, t, x0):
    return ((1 - sig) * x - x0) / (sig + (1 - sig) * t)


def sample(noise, steps, rescale_t, strength, rescale, interval):
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = (rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)).tolist()
    x = noise
    for i in range(steps):
        t, tp = t_seq[i], t_seq[i + 1]
        s = strength if interval[0] <= t <= interval[1] else 1.0
        if s == 1:
            pred = toy(x, 1000 * t, True)
        elif s == 0:
            pred = toy(x, 1000 * t, False)
        else:
            pos, neg = toy(x, 1000 * t, True), toy(x, 1000 * t, False)
            pred = s * pos + (1 - s) * neg
            if rescale > 0:
                x0p, x0c = pred_to_x0(x, t, pos), pred_to_x0(x, t, pred)
                x0r = x0c * (x0p.std() / x0c.std())
                pred = x0_to_pred(x, t, rescale * x0r + (1 - rescale) * x0c)
        x = x - (t - tp) * pred
    return x


noise = torch.randn(1, 8, 4, 4, 4, generator=g)
res_s = sample(noise, 12, 5.0, 7.5, 0.7, (0.6, 1.0))
json.dump(dict(z=z.reshape(-1).tolist(), out=out.reshape(-1).tolist(),
               noise=noise.reshape(-1).tolist(), sampled=res_s.reshape(-1).tolist(),
               steps=12, rescale_t=5.0, strength=7.5, rescale=0.7, interval=[0.6, 1.0]),
          open(os.path.join(out_dir, 'ssdec_sampler_fixture.json'), 'w'))
print('sampler out std', res_s.std().item())
