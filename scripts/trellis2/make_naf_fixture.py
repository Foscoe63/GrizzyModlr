"""Standalone PyTorch reference for the NAF feature upsampler (valeoai/NAF, Apache-2.0), CPU fp32, real weights.

Written from NAF `src/model/naf.py`, `layers/{convolutions,attentions,rope}.py` as a spec, not imported. The
`natten` kernel is replaced by an explicit gather that follows NATTEN's dilated neighbourhood rule (window of 9
over the stride-d sub-grid, centred on the query's sub-grid index and shifted to stay inside the map). That rule
is my reading of NATTEN, not checked against the extension itself.

  /Volumes/Models/Modelr/venvs/trellis2-ref/bin/python -I scripts/trellis2/make_naf_fixture.py naf_release.pth OUT_DIR
"""
import json, math, os, sys
import torch
import torch.nn.functional as F

ckpt, out_dir = sys.argv[1], sys.argv[2]
W = torch.load(ckpt, map_location='cpu', weights_only=True)
K, HEADS, DIM = 9, 4, 256


def conv(x, p, k):
    pad = k // 2
    if pad: x = F.pad(x, (pad,) * 4, mode='reflect')
    return F.conv2d(x, W[p + '.weight'], W[p + '.bias'])


def enc(x, p, k):
    x = conv(x, p + '.0', k)
    for i in (1, 2):
        b = f'{p}.{i}'
        h = F.silu(F.group_norm(x, 8, W[b + '.norm1.weight'], W[b + '.norm1.bias']))
        h = conv(h, b + '.conv1', k)
        h = F.silu(F.group_norm(h, 8, W[b + '.norm2.weight'], W[b + '.norm2.bias']))
        x = conv(h, b + '.conv2', k)
    return x


def rope(x):                                   # x [1, 256, H, Wd] -> same
    _, _, h, w = x.shape
    dh = DIM // HEADS
    x = x.reshape(HEADS, dh, h * w).permute(0, 2, 1)                 # [n, HW, d]
    ch = (torch.arange(h) + 0.5) / h; cw = (torch.arange(w) + 0.5) / w
    coords = 2 * torch.stack(torch.meshgrid(ch, cw, indexing='ij'), -1).reshape(-1, 2) - 1
    ang = (2 * math.pi * coords[:, :, None] / W['image_encoder.rope.periods'][None, None]).flatten(1).tile(2)
    cos, sin = torch.cos(ang), torch.sin(ang)
    x1, x2 = x.chunk(2, -1)
    x = x * cos + torch.cat([-x2, x1], -1) * sin
    return x.permute(0, 2, 1).reshape(1, DIM, h, w)


def naf(image, feats, out_size):
    x = torch.cat([enc(image, 'image_encoder.encoder', 1), enc(image, 'image_encoder.sem_encoder', 3)], 1)
    x = rope(F.adaptive_avg_pool2d(x, out_size))
    hq, wq = x.shape[-2:]
    hk, wk = feats.shape[-2:]
    keys = F.adaptive_avg_pool2d(x, (hk, wk))
    dy, dx = hq // hk, wq // wk
    q = x.reshape(HEADS, DIM // HEADS, hq, wq)
    kk = keys.reshape(HEADS, DIM // HEADS, hk, wk)
    C = feats.shape[1]
    vv = feats.reshape(HEADS, C // HEADS, hk, wk)
    sy = (torch.arange(hq) // dy - K // 2).clamp(0, hk - K)
    sx = (torch.arange(wq) // dx - K // 2).clamp(0, wk - K)
    iy = sy[:, None] + torch.arange(K)[None]                          # [hq, K]
    ix = sx[:, None] + torch.arange(K)[None]                          # [wq, K]
    kn = kk[:, :, iy][:, :, :, :, ix]                                 # [n, d, hq, K, wq, K]
    vn = vv[:, :, iy][:, :, :, :, ix]
    s = torch.einsum('ndhw,ndhawb->nhwab', q, kn) * (DIM // HEADS) ** -0.5
    a = s.reshape(HEADS, hq, wq, K * K).softmax(-1).reshape(HEADS, hq, wq, K, K)
    o = torch.einsum('nhwab,ncha wb->nchw'.replace(' ', ''), a, vn)
    return o.reshape(1, C, hq, wq)


g = torch.Generator().manual_seed(21)
H = 128
image = torch.rand(1, 3, H, H, generator=g)
feats = torch.randn(1, 24, 8, 8, generator=g)
with torch.no_grad():
    out = naf(image, feats, (H, H))
json.dump(dict(h=H, c=24, hk=8, image=image.reshape(-1).tolist(), feats=feats.reshape(-1).tolist(), out=out.reshape(-1).tolist()),
          open(os.path.join(out_dir, 'naf_fixture.json'), 'w'))
print('out', tuple(out.shape), 'std', out.std().item())
