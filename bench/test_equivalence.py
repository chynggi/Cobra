"""Numerical equivalence checks for the optimized post-processing building blocks.

  python bench/test_equivalence.py

Loads only the VAE and MultiResNetModel (no diffusion models) and checks:
  1. pil_to_normalized_tensor == torchvision transform (bit-exact)
  2. tensor_to_pil == official tensor -> PIL conversion (bit-exact)
  3. encode_skip_features == Encoder(hidden_flag=True) hidden list (bit-exact)
  4. LeanMultiHiddenResNetModel == MultiHiddenResNetModel as used by app.py (bit-exact)
  5. decode_with_skip_features == vae._decode(hidden_list=...) (bit-exact)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import snapshot_download

from diffusers import AutoencoderKL
from cobra_utils.utils import MultiHiddenResNetModel
from cobra_utils import postprocess as pp

dtype = torch.float16
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
model_path = snapshot_download(repo_id="JunhaoZhuang/Cobra", cache_dir="./Cobra/", repo_type="model")
img = Image.open("examples/line/example0/input.png").convert("RGB").resize((1008, 1440))
ok = True

# 1
a = transform(img).unsqueeze(0).to("cuda", dtype=dtype)
b = pp.pil_to_normalized_tensor(img, "cuda", dtype)
eq = torch.equal(a, b)
ok &= eq
print("1. pil_to_normalized_tensor bit-exact:", eq)

# 2
out = torch.randn(1, 3, 1440, 1008, device="cuda", dtype=dtype) * 1.2
ref_out = out.clone()
ref_out[ref_out > 1] = 1
ref_out[ref_out < -1] = -1
ref = Image.fromarray(((ref_out[0] * 0.5 + 0.5).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
mine = pp.tensor_to_pil(out)
eq = np.array_equal(np.asarray(ref), np.asarray(mine))
ok &= eq
print("2. tensor_to_pil bit-exact:", eq, "max diff", int(np.abs(np.asarray(ref, int) - np.asarray(mine, int)).max()))

# 3
vae = AutoencoderKL.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="vae", torch_dtype=dtype).to("cuda").eval()
with torch.no_grad():
    _, hidden_ref = vae._encode(a, return_dict=False, hidden_flag=True)
with torch.inference_mode():
    hidden_mine = pp.encode_skip_features(vae.encoder, a)
eq = len(hidden_ref) == len(hidden_mine) and all(torch.equal(x, y) for x, y in zip(hidden_ref, hidden_mine))
ok &= eq
print("3. encode_skip_features bit-exact:", eq, [tuple(h.shape) for h in hidden_mine])

# 4 (smaller input to keep the test light)
del hidden_ref, hidden_mine, a, b
with torch.no_grad():
    small = pp.pil_to_normalized_tensor(img.resize((512, 768)), "cuda", dtype)
    _, hidden_ref = vae._encode(small, return_dict=False, hidden_flag=True)
torch.cuda.empty_cache()
for style in ("line_GSRP", "shadow_GSRP"):
    m = MultiHiddenResNetModel([128, 128, 256, 512, 512], 5)
    m.load_state_dict(torch.load(os.path.join(model_path, style, "MultiResNetModel.bin"), map_location="cpu"), strict=True)
    # the official app never calls .eval(): BatchNorm runs with per-image statistics
    lean = pp.LeanMultiHiddenResNetModel(m).to("cuda", dtype)
    m16 = m.to("cuda", dtype)  # official
    with torch.inference_mode():
        x = [torch.cat((h, h.flip(-1)), dim=1) for h in hidden_ref]
        y16 = m16([t.clone() for t in x])
        yl = lean([t.clone() for t in x])
    eq = m16.training and lean.training and all(torch.equal(p, q) for p, q in zip(y16, yl))
    ok &= eq
    print(f"4. {style} LeanMultiHiddenResNetModel bit-exact vs official (training-mode BN):", eq)

# 5. decode_with_skip_features == vae._decode(hidden_list=...)
with torch.inference_mode():
    post, hidden_c = vae._encode(small, return_dict=False, hidden_flag=True)
    z = post.mode()
    feats = lean([torch.cat((h, h), dim=1) for h in hidden_c])
    ref_dec = vae._decode(z, return_dict=False, hidden_list=[f.clone() for f in feats])[0]
    my_dec = pp.decode_with_skip_features(vae, z, [f.clone() for f in feats])
eq = torch.equal(ref_dec, my_dec)
ok &= eq
print("5. decode_with_skip_features bit-exact:", eq)
print("ALL OK" if ok else "MISMATCH")
