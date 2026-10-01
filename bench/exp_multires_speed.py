"""MultiResNet speed per checkpoint (line_GSRP vs shadow_GSRP) on identical inputs (official vs lean).

  python bench/exp_multires_speed.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from diffusers import AutoencoderKL
from cobra_utils.utils import MultiHiddenResNetModel
from cobra_utils import postprocess as pp

dtype = torch.float16
model_path = snapshot_download(repo_id="JunhaoZhuang/Cobra", cache_dir="./Cobra/", repo_type="model")
vae = AutoencoderKL.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="vae", torch_dtype=dtype).to("cuda").eval()
img = Image.open("examples/line/example0/input.png").convert("RGB").resize((1008, 1440))
with torch.inference_mode():
    x = pp.pil_to_normalized_tensor(img, "cuda", dtype)
    _, hidden = vae._encode(x, return_dict=False, hidden_flag=True)
    feats = [torch.cat((h, h.flip(-1)), dim=1) for h in hidden]
    del hidden, x

models = {}
for style in ("line_GSRP", "shadow_GSRP"):
    m = MultiHiddenResNetModel([128, 128, 256, 512, 512], 5)
    m.load_state_dict(torch.load(os.path.join(model_path, style, "MultiResNetModel.bin"), map_location="cpu"), strict=True)
    w16 = torch.cat([p.detach().half().flatten().float() for p in m.parameters()])
    sub = ((w16 != 0) & (w16.abs() < 6.104e-5)).float().mean().item() * 100
    print(f"{style}: fp16 subnormal weights {sub:.2f}%")
    models[f"{style} lean"] = pp.LeanMultiHiddenResNetModel(m).to("cuda", dtype)
    models[f"{style} official"] = m.to("cuda", dtype)

times = {k: [] for k in models}
with torch.inference_mode():
    for rep in range(4):
        for name, mr in models.items():
            inp = [f.clone() for f in feats]
            torch.cuda.synchronize(); t = time.perf_counter()
            out = mr(inp)
            torch.cuda.synchronize()
            times[name].append((time.perf_counter() - t) * 1000)
            if rep == 0:
                bad = sum((~torch.isfinite(o)).sum().item() for o in out)
                maxv = max(o.abs().max().item() for o in out)
                print(f"{name}: non-finite outputs {bad}, max |out| {maxv:.1f}")
            del out, inp
for name, ts in times.items():
    print(f"{name}: {[round(t) for t in ts]} ms (median of last 3: {sorted(ts[1:])[1]:.0f})")
