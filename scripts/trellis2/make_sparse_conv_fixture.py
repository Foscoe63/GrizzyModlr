"""PyTorch reference for the TRELLIS.2 sparse ops, written as a JSON fixture for the Swift tests.

Submanifold conv (what flex_gemm's sparse_submanifold_conv3d computes): a dense cross-correlation
masked to the active sites. Weight layout is TRELLIS.2's (Co, Kd, Kh, Kw, Ci). Spatial2Channel /
Channel2Spatial are re-implemented line for line from trellis2/modules/sparse/spatial/spatial2channel.py.

  /Volumes/Models/Modelr/venvs/trellis2-ref/bin/python -I scripts/trellis2/make_sparse_conv_fixture.py OUT.json
"""
import json, sys
import torch
import torch.nn.functional as F

g = torch.Generator().manual_seed(7)
B, S, Ci, Co = 2, 12, 8, 6   # batch, edge, channels


def make_coords():
    pts = []
    for b in range(B):
        # blob + a plane so there are both dense neighbourhoods and isolated voxels
        blob = torch.randint(2, 9, (160, 3), generator=g)
        plane = torch.stack([torch.randint(0, S, (40,), generator=g), torch.randint(0, S, (40,), generator=g),
                             torch.full((40,), 11)], 1)
        c = torch.unique(torch.cat([blob, plane]), dim=0)
        pts.append(torch.cat([torch.full((len(c), 1), b), c], 1))
    return torch.cat(pts).int()


def submconv(coords, feats, weight, bias, dilation):
    # weight (Co, Kd, Kh, Kw, Ci) -> conv3d layout (Co, Ci, Kd, Kh, Kw)
    w = weight.permute(0, 4, 1, 2, 3).contiguous()
    dense = torch.zeros(B, Ci, S, S, S)
    c = coords.long()
    dense[c[:, 0], :, c[:, 1], c[:, 2], c[:, 3]] = feats
    pad = dilation  # kernel 3
    out = F.conv3d(dense, w, bias, padding=pad, dilation=dilation)
    return out[c[:, 0], :, c[:, 1], c[:, 2], c[:, 3]]


def s2c(coords, feats, factor=2):
    coord = list(coords.unbind(-1))
    for i in range(3):
        coord[i + 1] = coord[i + 1] // factor
    subidx = coords[:, 1:] % factor
    subidx = sum(subidx[..., i] * factor ** i for i in range(3))
    MAX = [(S + factor - 1) // factor] * 3
    OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
    code = sum(c * o for c, o in zip(coord, OFFSET))
    code, idx = code.unique(return_inverse=True)
    new_coords = torch.stack([code // OFFSET[0]] + [(code // OFFSET[i + 1]) % MAX[i] for i in range(3)], -1)
    nf = torch.zeros(new_coords.shape[0] * factor ** 3, feats.shape[1])
    nf[idx * factor ** 3 + subidx] = feats
    return new_coords, nf.reshape(new_coords.shape[0], -1), (idx, subidx)


def c2s(feats, cache, factor=2):
    idx, subidx = cache
    xf = feats.reshape(feats.shape[0] * factor ** 3, -1)
    return xf[idx * factor ** 3 + subidx]


coords = make_coords()
feats = torch.randn(len(coords), Ci, generator=g)
weight = torch.randn(Co, 3, 3, 3, Ci, generator=g) * 0.2
bias = torch.randn(Co, generator=g)
weight_c = torch.randn(Co, 3, 3, 3, Co, generator=g) * 0.2

cases = {}
for dil in (1, 2):
    cases[f'dil{dil}'] = submconv(coords, feats, weight, bias, dil)
nc, nf, cache = s2c(coords, feats)
back = c2s(nf, cache)
assert torch.equal(back, feats)

fl = lambda t: t.flatten().tolist()
json.dump({
    'batch': B, 'size': S, 'cin': Ci, 'cout': Co, 'n': len(coords),
    'coords': fl(coords), 'feats': fl(feats), 'weight': fl(weight), 'bias': fl(bias),
    'out_dil1': fl(cases['dil1']), 'out_dil2': fl(cases['dil2']),
    's2c_n': len(nc), 's2c_coords': fl(nc.int()), 's2c_feats': fl(nf),
}, open(sys.argv[1], 'w'))
print('N', len(coords), 'S2C N', len(nc))
