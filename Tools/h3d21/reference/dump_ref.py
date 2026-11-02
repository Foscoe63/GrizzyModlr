"""Phase 1: dump parity fixtures from the Hunyuan3D-2.1 PyTorch reference (CPU, fp32)."""
import sys, yaml, torch
from safetensors.torch import save_file
sys.path.insert(0, "/Volumes/Models/Modelr/ref21/repo/hy3dshape")
from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
from hy3dshape.models.autoencoders.model import ShapeVAE
from hy3dshape.models.conditioner import SingleImageEncoder

R = "/Volumes/Models/Modelr/ref21"
torch.manual_seed(0)
ck = torch.load(f"{R}/ckpt/model.fp16.ckpt", map_location="cpu", weights_only=True, mmap=True)
cfg = yaml.safe_load(open(f"{R}/cfg/dit.yaml"))
out = {}

dit = HunYuanDiTPlain(**cfg["model"]["params"])
dit.load_state_dict(ck["model"]); dit = dit.float().eval()
x = torch.randn(1, 4096, 64); t = torch.tensor([0.5]); cond = torch.randn(1, 1370, 1024)
with torch.no_grad():
    y = dit(x, t, {"main": cond})
out.update(dit_x=x, dit_t=t, dit_cond=cond, dit_out=y)
print("dit", y.shape, y.abs().mean().item())

vae = ShapeVAE(**cfg["vae"]["params"])
vae.load_state_dict(ck["vae"]); vae = vae.float().eval()
lat = torch.randn(1, 4096, 64); q = torch.rand(1, 2048, 3) * 2 - 1
with torch.no_grad():
    kv = vae.decode(lat)
    logits = vae.geo_decoder(queries=q, latents=kv)
out.update(vae_lat=lat, vae_kv=kv, vae_q=q, vae_logits=logits)
print("vae", kv.shape, logits.shape)

cc = cfg["conditioner"]["params"]["main_image_encoder"]["kwargs"]
enc = SingleImageEncoder(main_image_encoder={"type": "DinoImageEncoder", "kwargs": cc})
def rn(k):
    k = k.replace("main_image_encoder.model.", "")
    for a, b in [("attention.attention.query", "attention.q_proj"), ("attention.attention.key", "attention.k_proj"),
                 ("attention.attention.value", "attention.v_proj"), ("attention.output.dense", "attention.o_proj")]:
        k = k.replace(a, b)
    return k
sd = {rn(k): v for k, v in ck["conditioner"].items()}
enc.main_image_encoder.model.load_state_dict(sd); enc = enc.float().eval()
img = torch.rand(1, 3, 518, 518) * 2 - 1  # value_range (-1,1)
with torch.no_grad():
    h = enc(img)["main"]
out.update(dino_img=img, dino_out=h)
print("dino", h.shape)

save_file({k: v.contiguous().float() for k, v in out.items()}, f"{R}/work/fixtures.safetensors")
print("saved")
