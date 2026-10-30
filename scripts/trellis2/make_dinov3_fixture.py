"""Standalone PyTorch reference for the DINOv3 ViT feature extractor as TRELLIS.2 uses it (CPU fp32, RANDOM weights).

Mirrors HF `DINOv3ViTModel` (patch conv, cls + register tokens, axial RoPE on patch tokens only, pre-norm blocks
with LayerScale, GELU MLP, key projection without bias) and `DinoV3FeatureExtractor.extract_features`: all tokens out,
followed by a plain (non-affine) layer norm. The real checkpoint is gated, so this validates the architecture and the
weight-key layout, not the pretrained values. Two small configs cover RoPE with head_dim 16 and 64 and a non-square image.

  /Volumes/Models/Modelr/venvs/trellis2-ref/bin/python -I scripts/trellis2/make_dinov3_fixture.py OUT_DIR
"""
import json, math, os, sys
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
g = torch.Generator().manual_seed(21)


def make(name, hidden, heads, layers, inter, regs, h_img, w_img):
    P = 16
    W = {}
    r = lambda *s, sc=0.05: torch.randn(*s, generator=g) * sc
    W['embeddings.cls_token'] = r(1, 1, hidden, sc=0.5)
    W['embeddings.register_tokens'] = r(1, regs, hidden, sc=0.5)
    W['embeddings.mask_token'] = torch.zeros(1, 1, hidden)
    W['embeddings.patch_embeddings.weight'] = r(hidden, 3, P, P, sc=0.03)
    W['embeddings.patch_embeddings.bias'] = r(hidden)
    for i in range(layers):
        p = f'layer.{i}.'
        for n in ('norm1', 'norm2'):
            W[p + n + '.weight'] = 1 + r(hidden); W[p + n + '.bias'] = r(hidden)
        for n in ('q_proj', 'v_proj', 'o_proj'):
            W[p + f'attention.{n}.weight'] = r(hidden, hidden); W[p + f'attention.{n}.bias'] = r(hidden)
        W[p + 'attention.k_proj.weight'] = r(hidden, hidden)          # key_bias = False
        W[p + 'mlp.up_proj.weight'] = r(inter, hidden); W[p + 'mlp.up_proj.bias'] = r(inter)
        W[p + 'mlp.down_proj.weight'] = r(hidden, inter); W[p + 'mlp.down_proj.bias'] = r(hidden)
        W[p + 'layer_scale1.lambda1'] = 0.5 + torch.rand(hidden, generator=g)
        W[p + 'layer_scale2.lambda1'] = 0.5 + torch.rand(hidden, generator=g)
    save_file({k: v.contiguous() for k, v in W.items()}, os.path.join(out_dir, f'dinov3_{name}.safetensors'))

    x = torch.rand(1, 3, h_img, w_img, generator=g)
    hd = hidden // heads
    ph, pw = h_img // P, w_img // P
    # embeddings
    pe = F.conv2d(x, W['embeddings.patch_embeddings.weight'], W['embeddings.patch_embeddings.bias'], stride=P)
    pe = pe.flatten(2).transpose(1, 2)
    h = torch.cat([W['embeddings.cls_token'], W['embeddings.register_tokens'], pe], 1)
    # rope
    ch = (torch.arange(0.5, ph) / ph); cw = (torch.arange(0.5, pw) / pw)
    coords = 2.0 * torch.stack(torch.meshgrid(ch, cw, indexing='ij'), -1).flatten(0, 1) - 1.0   # [N,2] (y,x)
    inv = 1 / 100.0 ** torch.arange(0, 1, 4 / hd, dtype=torch.float32)
    ang = (2 * math.pi * coords[:, :, None] * inv[None, None]).flatten(1, 2).tile(2)
    cos, sin = torch.cos(ang), torch.sin(ang)

    def rot_half(t):
        a, b = t[..., :t.shape[-1] // 2], t[..., t.shape[-1] // 2:]
        return torch.cat([-b, a], -1)

    lin = lambda t, p, bias=True: t @ W[p + '.weight'].T + (W[p + '.bias'] if bias else 0)
    for i in range(layers):
        p = f'layer.{i}.'
        res = h
        y = F.layer_norm(h, (hidden,), W[p + 'norm1.weight'], W[p + 'norm1.bias'], 1e-5)
        T = y.shape[1]
        sp = lambda t: t.view(1, T, heads, hd).transpose(1, 2)
        q, k, v = sp(lin(y, p + 'attention.q_proj')), sp(lin(y, p + 'attention.k_proj', False)), sp(lin(y, p + 'attention.v_proj'))
        npre = T - ph * pw
        def ro(t):
            pre, pat = t[:, :, :npre], t[:, :, npre:]
            return torch.cat([pre, pat * cos + rot_half(pat) * sin], 2)
        q, k = ro(q), ro(k)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, T, hidden)
        h = res + lin(a, p + 'attention.o_proj') * W[p + 'layer_scale1.lambda1']
        res = h
        y = F.layer_norm(h, (hidden,), W[p + 'norm2.weight'], W[p + 'norm2.bias'], 1e-5)
        y = lin(F.gelu(lin(y, p + 'mlp.up_proj')), p + 'mlp.down_proj')
        h = res + y * W[p + 'layer_scale2.lambda1']
    out = F.layer_norm(h, (hidden,))
    json.dump(dict(h=h_img, w=w_img, heads=heads, x=x.reshape(-1).tolist(), out=out.reshape(-1).tolist(), tokens=out.shape[1]),
              open(os.path.join(out_dir, f'dinov3_{name}.json'), 'w'))
    print(name, tuple(out.shape))


make('small', hidden=64, heads=4, layers=2, inter=128, regs=4, h_img=64, w_img=64)
make('wide', hidden=128, heads=2, layers=2, inter=256, regs=4, h_img=48, w_img=80)
