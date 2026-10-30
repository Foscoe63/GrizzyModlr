"""Standalone PyTorch reference for the TRELLIS.2 flow DiTs (SparseStructureFlowModel and SLatFlowModel share
one block layout), CPU fp32, loaded from the real checkpoints.

Written from microsoft/TRELLIS.2 `sparse_structure_flow.py`, `structured_latent_flow.py`,
`modules/transformer/modulated.py` and the RoPE / RMS-norm attention modules, without importing that repo.
Writes inputs + output for the Swift parity test.

  /Volumes/Models/Modelr/venvs/trellis2-ref/bin/python -I scripts/trellis2/make_dit_fixture.py CKPT_DIR OUT_DIR
"""
import json, math, os, sys
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ckpt_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
C, H, NB, HD = 1536, 12, 30, 128


def run(W, x, coords, t, cond):
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
        q = rms(lin(y, p + '.cross_attn.to_q').reshape(N, H, HD), W[p + '.cross_attn.q_rms_norm.gamma'])
        kv = lin(cond, p + '.cross_attn.to_kv').reshape(cond.shape[0], 2, H, HD)
        k, v = rms(kv[:, 0], W[p + '.cross_attn.k_rms_norm.gamma']), kv[:, 1]
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)).transpose(0, 1).reshape(N, C)
        h = h + lin(o, p + '.cross_attn.to_out')
        y = ln(h) * (1 + sc_m) + sh_m
        y = lin(F.gelu(lin(y, p + '.mlp.mlp.0'), approximate='tanh'), p + '.mlp.mlp.2')
        h = h + y * g_m
    h = F.layer_norm(h, (C,))
    return lin(h, 'out_layer')


def arr(t):
    return t.reshape(-1).tolist()


g = torch.Generator().manual_seed(5)
cond = torch.randn(96, 1024, generator=g)
for name, ckpt, cin, coords in [
    ('ss', 'ss_flow_img_dit_1_3B_64_bf16', 8,
     torch.stack(torch.meshgrid(*[torch.arange(16)] * 3, indexing='ij'), -1).reshape(-1, 3)),
    ('slat', 'slat_flow_img2shape_dit_1_3B_512_bf16', 32,
     torch.unique(torch.randint(0, 32, (400, 3), generator=g), dim=0)),
]:
    W = {k: v.float() for k, v in load_file(os.path.join(ckpt_dir, ckpt + '.safetensors')).items()}
    n = coords.shape[0]
    x = torch.randn(n, cin, generator=g)
    t = 600.0
    with torch.no_grad():
        out = run(W, x, coords, t, cond)
    json.dump(dict(n=n, cin=cin, t=t, coords=coords.int().reshape(-1).tolist(), x=arr(x), cond=arr(cond), n_cond=cond.shape[0],
                   out=arr(out)), open(os.path.join(out_dir, f'dit_{name}_fixture.json'), 'w'))
    print(name, n, 'out std', out.std().item())
