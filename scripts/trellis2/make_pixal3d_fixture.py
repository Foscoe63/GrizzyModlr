"""Standalone PyTorch reference for Pixal3D: the projection sampler (ProjGrid) and the structure flow DiT with
per-block proj_linear, CPU fp32, real checkpoint. Written from TencentARC/Pixal3D as a spec, not imported.

Written from microsoft/TRELLIS.2 `sparse_structure_flow.py`, `structured_latent_flow.py`,
`modules/transformer/modulated.py` and the RoPE / RMS-norm attention modules, without importing that repo.
Writes inputs + output for the Swift parity test.

  /Volumes/Models/Modelr/venvs/trellis2-ref/bin/python -I scripts/trellis2/make_pixal3d_fixture.py PIXAL3D_SS_CKPT.safetensors OUT_DIR
"""
import json, math, os, sys
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ckpt, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
C, H, NB, HD = 1536, 12, 30, 128


def run(W, x, coords, t, cond, proj):
    ln = lambda h, p=None: F.layer_norm(h, (C,), None if p is None else W[p + '.weight'], None if p is None else W[p + '.bias'], 1e-6)
    lin = lambda h, p: h @ W[p + '.weight'].T + W[p + '.bias']
    # rope phases
    freq_dim = HD // 2 // 3
    freqs = 1.0 / (10000.0 ** (torch.arange(freq_dim, dtype=torch.float32) / freq_dim))
    ang = (coords.float()[:, :, None] * freqs[None, None]).reshape(coords.shape[0], -1)          # [N, 63]
    ang = torch.cat([ang, torch.zeros(ang.shape[0], HD // 2 - ang.shape[1])], 1)                  # pad to 64
    cs, sn = torch.cos(ang)[:, None], torch.sin(ang)[:, None]                                     # [N,1,64]

    def rope(v):                                    # v [N,H,HD]
        p = v.reshape(*v.shape[:-1], HD // 2, 2)
        re, im = p[..., 0], p[..., 1]
        return torch.stack([re * cs - im * sn, re * sn + im * cs], -1).reshape(v.shape)

    def rms(v, g):                                  # g [H,HD]
        return F.normalize(v.float(), dim=-1) * g * math.sqrt(HD)

    half = 128
    fr = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    a = t * fr
    temb = torch.cat([torch.cos(a), torch.sin(a)])[None]
    temb = lin(F.silu(lin(temb, 't_embedder.mlp.0')), 't_embedder.mlp.2')
    mod = lin(F.silu(temb), 'adaLN_modulation.1')                                                 # [1, 6C]
    h = lin(x, 'input_layer')
    N = h.shape[0]
    for i in range(NB):
        p = f'blocks.{i}'
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = (W[p + '.modulation'] + mod).chunk(6, dim=1)
        y = ln(h) * (1 + sc_a) + sh_a
        qkv = lin(y, p + '.self_attn.to_qkv').reshape(N, 3, H, HD)
        q, k, v = qkv.unbind(1)
        q, k = rope(rms(q, W[p + '.self_attn.q_rms_norm.gamma'])), rope(rms(k, W[p + '.self_attn.k_rms_norm.gamma']))
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)).transpose(0, 1).reshape(N, C)
        h = h + lin(o, p + '.self_attn.to_out') * g_a
        y = ln(h, p + '.norm2')
        ca = p + '.cross_attn.cross_attn_block'
        q = rms(lin(y, ca + '.to_q').reshape(N, H, HD), W[ca + '.q_rms_norm.gamma'])
        kv = lin(cond, ca + '.to_kv').reshape(cond.shape[0], 2, H, HD)
        k, v = rms(kv[:, 0], W[ca + '.k_rms_norm.gamma']), kv[:, 1]
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)).transpose(0, 1).reshape(N, C)
        h = h + (lin(o, ca + '.to_out') + lin(proj, p + '.cross_attn.proj_linear'))
        y = ln(h) * (1 + sc_m) + sh_m
        y = lin(F.gelu(lin(y, p + '.mlp.mlp.0'), approximate='tanh'), p + '.mlp.mlp.2')
        h = h + y * g_m
    h = F.layer_norm(h, (C,))
    return lin(h, 'out_layer')



def proj_sample(fmap, R, res, fov, dist, scale):
    """ProjGrid.forward: fmap [H,W,C] -> [R^3, C]."""
    one = torch.linspace(-1, 1, R)
    pts = torch.stack(torch.meshgrid(one, one, one, indexing='ij'), -1)
    rot = torch.tensor([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    pts = (pts @ rot.T).reshape(-1, 3) / scale / 2
    T = torch.tensor([[1., 0, 0, 0], [0, 0, -1, -dist], [0, 1, 0, 0], [0, 0, 0, 1]])
    pc = (torch.cat([pts, torch.ones(len(pts), 1)], 1) @ torch.linalg.inv(T).T)[:, :3]
    f = 16.0 / math.tan(fov / 2) * res / 32.0
    xp = f * pc[:, 0] / (-pc[:, 2] + 1e-8) + res / 2
    yp = -f * pc[:, 1] / (-pc[:, 2] + 1e-8) + res / 2
    grid = (torch.stack([xp, yp], -1) + 0.5) / res * 2 - 1
    out = F.grid_sample(fmap.permute(2, 0, 1)[None], grid.view(1, -1, 1, 2), mode='bilinear', align_corners=False, padding_mode='border')
    return out[0, :, :, 0].T


def arr(t):
    return t.reshape(-1).tolist()


g = torch.Generator().manual_seed(9)
fmap = torch.randn(32, 32, 16, generator=g)
fov, dist, scale = 0.8575560450553894, 2.0, 1.0
samp = proj_sample(fmap, 16, 512, fov, dist, scale)
W = {k: v.float() for k, v in load_file(ckpt).items()}
pin = W['blocks.0.cross_attn.proj_linear.weight'].shape[1]
coords = torch.stack(torch.meshgrid(*[torch.arange(16)] * 3, indexing='ij'), -1).reshape(-1, 3)
x = torch.randn(4096, 8, generator=g)
cond = torch.randn(5, 1024, generator=g)
proj = torch.randn(4096, pin, generator=g)
with torch.no_grad():
    out = run(W, x, coords, 600.0, cond, proj)
json.dump(dict(fmap=arr(fmap), samp=arr(samp), fov=fov, dist=dist, scale=scale, pin=pin, x=arr(x), cond=arr(cond),
               proj=arr(proj), out=arr(out), coords=coords.int().reshape(-1).tolist(), t=600.0),
          open(os.path.join(out_dir, 'pixal3d_fixture.json'), 'w'))
print('proj_in', pin, 'out std', out.std().item(), 'samp std', samp.std().item())
