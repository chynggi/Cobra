"""Isolated experiments on the official models.

  python bench/run_experiments.py [--repeats 2] [--cases line_ex0,shadow_ex0] [--skip-official]

* post-processing ablation ladder (official -> each optimization), scale sweep
* VAE slicing / tiling, empty_cache cost
* precision: CLIP image encoder fp32/fp16/bf16 (top-k equality), line model fp32/fp16/bf16,
  post-processing fp16 vs bf16
* CLIP CPU<->GPU offload cost

Every post-processing variant gets the same diffusion output, line art and RNG seed
(h_color.sample() uses the global CUDA RNG) and starts from an emptied allocator cache.

Note: bench/results/experiments.json was produced by an earlier revision of this script that
measured BN folded into the convolutions in eval mode ("2 +BN fused"; not equivalent: the
official MultiResNet runs BatchNorm in training mode), a level-0 strip-tiled MultiResNet
("3 +level-0 strips") and a line-art feature cache ("3b"). All three were dropped afterwards.
"""
import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import (CASES, RESULTS, GpuMemSampler, apply_hints, delta_e, hint_error, import_app,
                          line_darkness, load_case, median, psnr, save_json, sharpness)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SAMPLER = None


def timed(fn, repeats, warmup=1, seed=1234, clean=True):
    """Median wall ms (synchronized), peak allocated/reserved MB, WDDM spill, last result."""
    if clean:
        torch.cuda.empty_cache()
    for _ in range(warmup):
        torch.manual_seed(seed); fn()
    times = []
    t_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        torch.cuda.synchronize()
        torch.manual_seed(seed)
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    t_end = time.perf_counter()
    time.sleep(1.5)
    res = dict(ms=median(times), runs=[round(x, 1) for x in times],
               peak_alloc_mb=torch.cuda.max_memory_allocated() / 2**20,
               peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20)
    res.update(SAMPLER.window(t_start, t_end))
    return res, out


def img_diff(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    return dict(psnr=psnr(a, b), max_abs_diff=int(np.abs(a - b).max()), pct_px_diff=float((a != b).any(-1).mean() * 100))


def main():
    global SAMPLER
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cases", default="line_ex0,shadow_ex0")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    app = import_app("app")
    from bench.baseline_profiled import colorize_image_profiled, official_postprocess
    from cobra_utils import postprocess as pp
    SAMPLER = GpuMemSampler()
    img_dir = os.path.join(RESULTS, "images")
    os.makedirs(img_dir, exist_ok=True)
    report = {}

    names = sorted(args.cases.split(","), key=lambda n: CASES[n]["style"] != app.cur_style)
    for name in names:
        c, inp, refs = load_case(name)
        if app.cur_style != c["style"]:
            app.change_ckpt(c["style"])
        r = report[name] = {}
        extracted, hint_color, hint_mask, query_origin, extracted_ori, resolution = app.extract_sketch_line_image(inp, c["style"])
        hint_color, hint_mask, placed = apply_hints(app, hint_color, hint_mask, refs, c["hints"])
        call = (extracted, refs, resolution, c["seed"], 10, args.top_k, hint_mask, hint_color, query_origin, extracted_ori)
        _, colorized, topk_official = colorize_image_profiled(app, pp.NULL_TIMER, *call, run_postprocess=False)
        colorized.save(os.path.join(img_dir, f"{name}_exp_diffusion.png"))
        line_size = tuple(resolution)
        vae = app.pipeline.vae
        multires_official = app.MultiResNetModel  # fp16 on GPU, BN unfused (official)
        multires_cpu32 = copy.deepcopy(multires_official).float().cpu()

        def lean_mr(dtype=torch.float16):  # lean MultiResNet, same (training) mode as the official one
            return pp.LeanMultiHiddenResNetModel(multires_cpu32).to("cuda", dtype)

        def official(scale):
            size = (int(resolution[0] * scale), int(resolution[1] * scale))
            return lambda: official_postprocess(app, pp.NULL_TIMER, colorized, extracted_ori.resize(size))

        def optimized(scale, multires):
            return lambda: pp.postprocess_colorized_image(colorized, extracted_ori, resolution, vae, multires, scale=scale)

        # ---------------- ablation ladder @1.5 (CLIP fp32 stays on the GPU, as in app.py)
        v = {}
        t, base_img = timed(official(1.5), args.repeats)
        v["0 official"] = t
        base_img.save(os.path.join(img_dir, f"{name}_exp_official_1.50.png"))
        print(name, "official", t, flush=True)

        def lib_unfused(xs):  # official module (its inputs are not released)
            return multires_official(xs)
        t, img = timed(optimized(1.5, lib_unfused), args.repeats)
        v["1 +skip bw mid-block, GPU convert, decode frees skips"] = dict(t, **img_diff(img, base_img))
        mr = lean_mr()
        t, img_opt = timed(optimized(1.5, mr), args.repeats)
        v["2 +in-place ReLU, inputs released"] = dict(t, **img_diff(img_opt, base_img))
        print(name, "optimized", v["2 +in-place ReLU, inputs released"], flush=True)

        clip_dev = app.image_encoder
        clip_dev.half()
        t, img = timed(optimized(1.5, mr), args.repeats)
        v["4 +CLIP fp16 (frees 1.2 GB)"] = dict(t, **img_diff(img, base_img))
        clip_dev.to("cpu")
        t, img = timed(optimized(1.5, mr), args.repeats)
        v["5 CLIP offloaded to CPU during post-processing"] = dict(t, **img_diff(img, base_img))
        clip_dev.to("cuda").float()

        torch.backends.cudnn.benchmark = True
        t, _ = timed(optimized(1.5, mr), args.repeats, warmup=2)
        v["3c +cudnn.benchmark"] = t
        torch.backends.cudnn.benchmark = False

        mr_cl = copy.deepcopy(mr).to(memory_format=torch.channels_last)
        def cl(xs):
            return [y.contiguous() for y in mr_cl([x.contiguous(memory_format=torch.channels_last) for x in xs])]
        t, _ = timed(optimized(1.5, cl), args.repeats)
        v["3d +channels_last MultiResNet"] = t
        del mr_cl

        def with_empty_cache():
            out = optimized(1.5, mr)()
            torch.cuda.empty_cache()
            return out
        t, _ = timed(with_empty_cache, args.repeats)
        v["3e + empty_cache after each run"] = t
        torch.cuda.synchronize(); t0 = time.perf_counter(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        v["empty_cache call (warm, nothing cached) ms"] = (time.perf_counter() - t0) * 1000

        # bf16 post-processing (VAE + MultiResNet)
        vae.to(torch.bfloat16)
        mr_bf = lean_mr(dtype=torch.bfloat16)
        t, img = timed(optimized(1.5, mr_bf), args.repeats)
        v["6 bf16 VAE + MultiResNet"] = dict(t, **img_diff(img, base_img), psnr_vs_fp16_optimized=psnr(img, img_opt))
        img.save(os.path.join(img_dir, f"{name}_exp_bf16_1.50.png"))
        vae.to(torch.float16)
        del mr_bf

        # VAE slicing / tiling (official algorithm)
        vae.enable_slicing()
        t, img = timed(optimized(1.5, mr), args.repeats)
        v["7 +vae.enable_slicing()"] = dict(t, **img_diff(img, img_opt))
        vae.disable_slicing()
        vae.enable_tiling()
        t, img = timed(official(1.5), 1)
        v["8 official + vae.enable_tiling()"] = dict(t, **img_diff(img, base_img))
        img.save(os.path.join(img_dir, f"{name}_exp_official_tiling.png"))
        vae.disable_tiling()
        r["postprocess_ablation"] = v
        save_json(report, "experiments.json")

        # ---------------- scale sweep (quality relative to official 1.5 output)
        scales = {}
        for s in (1.5, 1.25, 1.0, 0.875):
            t_off = timed(official(s), args.repeats)[0] if s in (1.25, 1.0) else None
            t_opt, img = timed(optimized(s, mr), args.repeats)
            img.save(os.path.join(img_dir, f"{name}_exp_opt_{s:.3f}.png"))
            scales[str(s)] = dict(official=t_off, optimized=t_opt, output_size=list(img.size),
                                  psnr_vs_official15=psnr(img.resize(base_img.size, Image.BICUBIC), base_img),
                                  deltaE_vs_official15_at_1x=delta_e(img, base_img, line_size),
                                  sharpness_at_1x=sharpness(img.resize(line_size, Image.BICUBIC)),
                                  hint_err=hint_error(img, placed, line_size),
                                  line_darkness=line_darkness(img, extracted_ori, line_size))
            print(name, "scale", s, scales[str(s)], flush=True)
        scales["no_postprocess"] = dict(output_size=list(colorized.size),
                                        psnr_vs_official15=psnr(colorized.resize(base_img.size, Image.BICUBIC), base_img),
                                        deltaE_vs_official15_at_1x=delta_e(colorized, base_img, line_size),
                                        sharpness_at_1x=sharpness(colorized),
                                        hint_err=hint_error(colorized, placed, line_size),
                                        line_darkness=line_darkness(colorized, extracted_ori, line_size))
        scales["official_1.5"] = dict(sharpness_at_1x=sharpness(base_img.resize(line_size, Image.BICUBIC)),
                                      hint_err=hint_error(base_img, placed, line_size),
                                      line_darkness=line_darkness(base_img, extracted_ori, line_size))
        r["scales"] = scales
        del mr
        save_json(report, "experiments.json")

        # ---------------- retrieval: CLIP precision, argsort vs topk, offload cost
        from cobra_utils.utils import process_image, process_image_Q_varres, process_image_ref_varres
        tw, th = resolution
        q = process_image_Q_varres(query_origin.resize((tw, th)), tw, th)
        refp = []
        for ri in app.process_multi_images(refs):
            refp += process_image_ref_varres(process_image(ri, tw, th), tw, th)
        proc = app.image_processor
        t_pre, (q_px, r_px) = timed(lambda: (proc(images=q, return_tensors="pt").pixel_values,
                                            proc(images=refp, return_tensors="pt").pixel_values), args.repeats)
        enc = app.image_encoder

        def sims(dtype):
            with torch.inference_mode():
                qe = enc(q_px.to("cuda", dtype)).image_embeds.float()
                re = enc(r_px.to("cuda", dtype)).image_embeds.float()
                return F.cosine_similarity(qe.unsqueeze(1), re.unsqueeze(0), dim=-1)
        k = min(args.top_k, len(refp))
        ret = dict(n_query=len(q), n_ref_patches=len(refp), clip_preprocess_ms=t_pre["ms"])
        t32, s32 = timed(lambda: sims(torch.float32), args.repeats)
        arg = torch.argsort(s32, descending=True, dim=1).tolist()
        arg_k = [row[:k] for row in arg]
        ret["topk_equals_argsort"] = torch.topk(s32, k, dim=1).indices.tolist() == arg_k
        ret["official_pipeline_topk"] = topk_official
        ret["fp32"] = dict(ms=t32["ms"])
        for dname, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            enc.to(dt)
            t_, s_ = timed(lambda: sims(dt), args.repeats)
            idx = torch.topk(s_, k, dim=1).indices.tolist()
            full = torch.argsort(s_, descending=True, dim=1).tolist()
            ret[dname] = dict(ms=t_["ms"], topk_identical=idx == arg_k,
                              topk_same_set=all(sorted(a) == sorted(b) for a, b in zip(idx, arg_k)),
                              full_ranking_identical=full == arg,
                              max_abs_sim_diff=float((s_ - s32).abs().max()))
            enc.float()
        enc.half()
        torch.cuda.synchronize(); t0 = time.perf_counter(); enc.to("cpu"); torch.cuda.synchronize()
        ret["fp16_gpu_to_cpu_ms"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); enc.to("cuda"); torch.cuda.synchronize()
        ret["fp16_cpu_to_gpu_ms"] = (time.perf_counter() - t0) * 1000
        enc.float()
        r["retrieval"] = ret
        print(name, "retrieval", ret, flush=True)

        # ---------------- line extraction precision
        src = inp.resize(tuple(resolution)).convert("L").convert("RGB")
        t32, l32 = timed(lambda: app.extract_lines(src), args.repeats)
        le = dict(fp32_ms=t32["ms"])
        line_model = app.line_model
        for dname, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            line_model.to(dt)
            app.line_model = lambda x, _dt=dt: line_model(x.to(_dt)).float()  # extract_lines uses the module global
            try:
                t_, l_ = timed(lambda: app.extract_lines(src), args.repeats)
            finally:
                app.line_model = line_model
            le[dname] = dict(ms=t_["ms"], **img_diff(l_, l32))
            line_model.float()
        r["line_extraction"] = le
        print(name, "line_extraction", le, flush=True)
        save_json(report, "experiments.json")
    SAMPLER.close()
    save_json(report, "experiments.json")


if __name__ == "__main__":
    main()
