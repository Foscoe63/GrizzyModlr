"""Real-weights reference for DINOv3 ViT-L/16 as TRELLIS.2 conditions on it (CPU fp32), same math as
make_dinov3_fixture.py. Input is a deterministic 512x512 image; writes raw float32 files (x NCHW, out [1,T,1024]).

  /Volumes/Models/Modelr/venvs/trellis2-ref/bin/python -I scripts/trellis2/make_dinov3_real_fixture.py WEIGHTS.safetensors OUT_DIR
"""
import math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ckpt, out_dir = sys.argv[1], sys.argv[2]
W = load_file(ckpt)
hidden, heads, P = 1024, 16, 16
layers = 1 + max(int(k.split('.')[1]) for k in W if k.startswith('layer.'))
g = torch.Generator().manual_seed(2)
# smooth-ish image: low-frequency noise upsampled, so features are not pure noise
x = F.interpolate(torch.rand(1, 3, 16, 16, generator=g), size=(512, 512), mode='bilinear', align_corners=False)
x = (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

hd = hidden // heads
ph = pw = 512 // P
pe = F.conv2d(x, W['embeddings.patch_embeddings.weight'], W['embeddings.patch_embeddings.bias'], stride=P).flatten(2).transpose(1, 2)
h = torch.cat([W['embeddings.cls_token'], W['embeddings.register_tokens'], pe], 1)
ch = torch.arange(0.5, ph) / ph; cw = torch.arange(0.5, pw) / pw
coords = 2.0 * torch.stack(torch.meshgrid(ch, cw, indexing='ij'), -1).flatten(0, 1) - 1.0
inv = 1 / 100.0 ** torch.arange(0, 1, 4 / hd, dtype=torch.float32)
ang = (2 * math.pi * coords[:, :, None] * inv[None, None]).flatten(1, 2).tile(2)
cos, sin = torch.cos(ang), torch.sin(ang)
rot_half = lambda t: torch.cat([-t[..., t.shape[-1] // 2:], t[..., :t.shape[-1] // 2]], -1)
lin = lambda t, p, b=True: t @ W[p + '.weight'].T + (W[p + '.bias'] if b else 0)
T = h.shape[1]; npre = T - ph * pw
with torch.no_grad():
    for i in range(layers):
        p = f'layer.{i}.'
        res = h
        y = F.layer_norm(h, (hidden,), W[p + 'norm1.weight'], W[p + 'norm1.bias'], 1e-5)
        sp = lambda t: t.view(1, T, heads, hd).transpose(1, 2)
        q, k, v = sp(lin(y, p + 'attention.q_proj')), sp(lin(y, p + 'attention.k_proj', False)), sp(lin(y, p + 'attention.v_proj'))
        ro = lambda t: torch.cat([t[:, :, :npre], t[:, :, npre:] * cos + rot_half(t[:, :, npre:]) * sin], 2)
        a = F.scaled_dot_product_attention(ro(q), ro(k), v).transpose(1, 2).reshape(1, T, hidden)
        h = res + lin(a, p + 'attention.o_proj') * W[p + 'layer_scale1.lambda1']
        res = h
        y = F.layer_norm(h, (hidden,), W[p + 'norm2.weight'], W[p + 'norm2.bias'], 1e-5)
        h = res + lin(F.gelu(lin(y, p + 'mlp.up_proj')), p + 'mlp.down_proj') * W[p + 'layer_scale2.lambda1']
    out = F.layer_norm(h, (hidden,))
x.numpy().astype(np.float32).tofile(os.path.join(out_dir, 'dinov3_real_x.f32'))
out.numpy().astype(np.float32).tofile(os.path.join(out_dir, 'dinov3_real_out.f32'))
print('layers', layers, 'tokens', T, 'out std', out.std().item())
