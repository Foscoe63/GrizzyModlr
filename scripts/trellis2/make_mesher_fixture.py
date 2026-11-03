"""Standalone PyTorch reference for TRELLIS.2's flexible-dual-grid -> mesh step (inference path).

Written from o-voxel `flexible_dual_grid_to_mesh` (train=False, split_weight given) and the activations in
FlexiDualGridVaeDecoder.forward (vertex = 2*sigmoid-0.5, intersected = logit>0, quad weight = softplus),
using sorted-key lookup instead of the CUDA hashmap. Reads the decoder fixture's final features.

  /Volumes/Models/Modelr/venvs/trellis2-ref/bin/python -I scripts/trellis2/make_mesher_fixture.py FIXTURE_DIR GRID
"""
import json, sys, os
import torch
import torch.nn.functional as F

d, grid = sys.argv[1], int(sys.argv[2])
fx = json.load(open(os.path.join(d, 'decoder_fixture.json')))
n = fx['n_final']
coords = torch.tensor(fx['final_coords'], dtype=torch.long).reshape(n, 4)[:, 1:]
h = torch.tensor(fx['final']).reshape(n, 7)

dual = 2.0 * torch.sigmoid(h[:, 0:3]) - 0.5          # (1 + 2*margin)*sigmoid - margin, margin = 0.5
inter = h[:, 3:6] > 0
sw = F.softplus(h[:, 6])
voxel = 1.0 / grid
verts = (coords.float() + dual) * voxel - 0.5          # aabb [-0.5, 0.5]

S = grid + 2
key = lambda c: (c[..., 0] * S + c[..., 1]) * S + c[..., 2]
k0 = key(coords); order = torch.argsort(k0); sk = k0[order]
off = torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
                    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]])
nb = coords.reshape(n, 1, 1, 3) + off.unsqueeze(0)       # (N,3,4,3)
conn = nb[inter]                                         # (M,4,3), voxel-major then axis
m = conn.shape[0]
flat = conn.reshape(-1, 3)
inb = ((flat >= 0) & (flat < grid)).all(1)
kk = key(flat.clamp(0, grid - 1)); pos = torch.searchsorted(sk, kk).clamp(max=n - 1)
hit = inb & (sk[pos] == kk)
idx = torch.where(hit, order[pos], torch.full_like(pos, -1)).reshape(m, 4)
quad = idx[(idx >= 0).all(1)]
w02 = sw[quad[:, 0]] * sw[quad[:, 2]]; w13 = sw[quad[:, 1]] * sw[quad[:, 3]]
s1 = torch.tensor([0, 1, 2, 0, 2, 3]); s2 = torch.tensor([0, 1, 3, 3, 1, 2])
tri = torch.where((w02 > w13)[:, None], quad[:, s1], quad[:, s2]).reshape(-1, 3)
margin_gap = ((w02 - w13).abs() / (w02 + w13)).min().item()
out = dict(grid=grid, n_verts=n, n_faces=int(tri.shape[0]), verts=verts.reshape(-1).tolist(), faces=tri.int().reshape(-1).tolist())
json.dump(out, open(os.path.join(d, 'mesher_fixture.json'), 'w'))
print('faces', tri.shape[0], 'quads', quad.shape[0], 'min relative split margin', margin_gap)
