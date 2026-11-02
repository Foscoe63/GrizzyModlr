import torch
from safetensors.torch import save_file
ck = torch.load("/Volumes/Models/Modelr/ref21/ckpt/model.fp16.ckpt", map_location="cpu", weights_only=True, mmap=True)
out = {}
for k, v in ck["model"].items(): out["model." + k] = v.contiguous()
for k, v in ck["vae"].items(): out["vae." + k] = v.contiguous()
for k, v in ck["conditioner"].items(): out["conditioner." + k] = v.contiguous()
save_file(out, "/Volumes/Models/Modelr/shape-v21/model.fp16.safetensors")
print(len(out))
for k in out:
    if (".blocks.0." in k or ".blocks.20." in k or ".blocks.11." in k or "layer.0." in k) or not any(s in k for s in ["blocks.","layer.","resblocks","encoder."]): print(k, tuple(out[k].shape))
