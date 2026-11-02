"""Phase 3 reference: image -> latents (Euler flow-matching, CFG) -> dense SDF -> marching cubes, CPU fp32."""
import sys, time, yaml, numpy as np, torch
from PIL import Image
from safetensors.torch import save_file
sys.path.insert(0, "/Volumes/Models/Modelr/ref21/repo/hy3dshape")
from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
from hy3dshape.models.autoencoders.model import ShapeVAE
from hy3dshape.models.conditioner import SingleImageEncoder
from hy3dshape.preprocessors import ImageProcessorV2
import skimage.measure, trimesh

R = "/Volumes/Models/Modelr/ref21"
name = sys.argv[1] if len(sys.argv) > 1 else "004"
STEPS, GUIDE, RES = 30, 5.0, 128
torch.set_num_threads(10)
ck = torch.load(f"{R}/ckpt/model.fp16.ckpt", map_location="cpu", weights_only=True, mmap=True)
cfg = yaml.safe_load(open(f"{R}/cfg/dit.yaml"))
dit = HunYuanDiTPlain(**cfg["model"]["params"]); dit.load_state_dict(ck["model"]); dit = dit.float().eval()
vae = ShapeVAE(**cfg["vae"]["params"]); vae.load_state_dict(ck["vae"]); vae = vae.float().eval()
cc = cfg["conditioner"]["params"]["main_image_encoder"]["kwargs"]
enc = SingleImageEncoder(main_image_encoder={"type": "DinoImageEncoder", "kwargs": cc})
def rn(k):
    k = k.replace("main_image_encoder.model.", "")
    for a, b in [("attention.attention.query", "attention.q_proj"), ("attention.attention.key", "attention.k_proj"),
                 ("attention.attention.value", "attention.v_proj"), ("attention.output.dense", "attention.o_proj")]:
        k = k.replace(a, b)
    return k
enc.main_image_encoder.model.load_state_dict({rn(k): v for k, v in ck["conditioner"].items()}); enc = enc.float().eval()

img = Image.open(f"{R}/repo/assets/example_images/{name}.png")
proc = ImageProcessorV2(size=512, border_ratio=0.15)
image = proc(img)["image"].float()                                   # [1,3,512,512] in [-1,1]
tf = enc.main_image_encoder.transform
x01 = (image + 1) / 2
pix = tf(x01)                                                          # [1,3,518,518] normalised
with torch.no_grad():
    cond = enc(image)["main"]
    cond = torch.cat([cond, enc.unconditional_embedding(1)["main"]], 0)
g = torch.Generator().manual_seed(1234)
noise = torch.randn(1, 4096, 64, generator=g)
sig = np.linspace(0, 1, STEPS).astype(np.float32)
lat = noise.clone(); t0 = time.time()
with torch.no_grad():
    for i in range(STEPS):
        nxt = sig[i + 1] if i + 1 < STEPS else 1.0
        v = dit(torch.cat([lat, lat]), torch.full((2,), float(sig[i])), {"main": cond})
        vc, vu = v.chunk(2)
        lat = lat + (nxt - sig[i]) * (vu + GUIDE * (vc - vu))
        print("step", i, round(time.time() - t0), flush=True)
save_file({"pix_nhwc": pix.permute(0, 2, 3, 1).contiguous(), "noise": noise, "latents": lat.contiguous(), "cond": cond[:1].contiguous()},
          f"{R}/work/e2e_{name}.safetensors")
print("latents saved", flush=True)
sf = cfg["vae"]["params"]["scale_factor"]
with torch.no_grad():
    kv = vae.decode(lat / sf)
    ax = torch.linspace(-1.01, 1.01, RES + 1)
    grid = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    out = torch.cat([vae.geo_decoder(queries=grid[s:s + 20000][None], latents=kv)[0, :, 0] for s in range(0, len(grid), 20000)])
vol = out.reshape(RES + 1, RES + 1, RES + 1).numpy()
v, f, _, _ = skimage.measure.marching_cubes(vol, level=0.0, spacing=(2.02 / RES,) * 3)
v = v - 1.01
trimesh.Trimesh(v, f).export(f"{R}/work/ref_{name}_r{RES}.obj")
print("mesh", len(v), len(f), flush=True)
