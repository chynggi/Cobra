"""Post-processing experiments on app_optimized (fp16 CLIP).

  python bench/run_experiments2.py [--repeats 3] [--cases line_ex0,shadow_ex0]

* official post-processing algorithm vs optimized (bit-exact path) in the same process
* scale sweep with quality metrics
"""
import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import (CASES, RESULTS, GpuMemSampler, apply_hints, delta_e, hint_error, import_app,
                          line_darkness, load_case, psnr, save_json, sharpness)
import bench.run_experiments as exp1

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cases", default="line_ex0,shadow_ex0")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    app = import_app("app_optimized")
    from bench.baseline_profiled import official_postprocess
    from cobra_utils import postprocess as pp
    exp1.SAMPLER = GpuMemSampler()
    timed, img_diff = exp1.timed, exp1.img_diff
    img_dir = os.path.join(RESULTS, "images")
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
        colorized = app.run_colorization(*call, enable_postprocess=False)[0]
        colorized.save(os.path.join(img_dir, f"{name}_exp2_diffusion.png"))
        line_size = tuple(resolution)
        vae = app.pipeline.vae

        official_mr = copy.deepcopy(app.MultiResNetModel).to("cuda", torch.float16)  # official module (training mode)
        lean = app.FastMultiResNetModel
        # official_postprocess() reads these attributes; give it the official fp16 MultiResNet
        shim = type("A", (), dict(pipeline=app.pipeline, MultiResNetModel=official_mr, transform=app.transform,
                                  weight_dtype=app.weight_dtype))

        def official(scale):
            size = (int(resolution[0] * scale), int(resolution[1] * scale))
            return lambda: official_postprocess(shim, pp.NULL_TIMER, colorized, extracted_ori.resize(size))

        def optimized(scale, multires):
            return lambda: pp.postprocess_colorized_image(colorized, extracted_ori, resolution, vae, multires, scale=scale)

        v = {}
        t, off = timed(official(1.5), args.repeats)
        v["official algorithm (CLIP fp16)"] = t
        off.save(os.path.join(img_dir, f"{name}_exp2_official_1.50.png"))
        t, img_lean = timed(optimized(1.5, lean), args.repeats)
        v["lean (bit-exact path)"] = dict(t, **img_diff(img_lean, off))
        img_lean.save(os.path.join(img_dir, f"{name}_exp2_lean_1.50.png"))
        # run-to-run determinism of the official algorithm (cuDNN algorithm choice)
        t, off2 = timed(official(1.5), 1, warmup=0)
        v["official run-to-run"] = img_diff(off2, off)
        print(name, v, flush=True)
        r["postprocess"] = v
        save_json(report, "experiments2.json")

        scales = {}
        for s in (1.5, 1.25, 1.0, 0.875):
            t, img = timed(optimized(s, lean), args.repeats)
            img.save(os.path.join(img_dir, f"{name}_exp_opt_{s:.3f}.png"))
            scales[str(s)] = dict(t, output_size=list(img.size),
                                  psnr_vs_official15=psnr(img.resize(off.size, Image.BICUBIC), off),
                                  deltaE_vs_official15_at_1x=delta_e(img, off, line_size),
                                  sharpness_at_1x=sharpness(img.resize(line_size, Image.BICUBIC)),
                                  hint_err=hint_error(img, placed, line_size),
                                  line_darkness=line_darkness(img, extracted_ori, line_size))
            print(name, "scale", s, scales[str(s)], flush=True)
        scales["no_postprocess"] = dict(output_size=list(colorized.size),
                                        psnr_vs_official15=psnr(colorized.resize(off.size, Image.BICUBIC), off),
                                        deltaE_vs_official15_at_1x=delta_e(colorized, off, line_size),
                                        sharpness_at_1x=sharpness(colorized),
                                        hint_err=hint_error(colorized, placed, line_size),
                                        line_darkness=line_darkness(colorized, extracted_ori, line_size))
        scales["official_1.5"] = dict(sharpness_at_1x=sharpness(off.resize(line_size, Image.BICUBIC)),
                                      hint_err=hint_error(off, placed, line_size),
                                      line_darkness=line_darkness(off, extracted_ori, line_size))
        r["scales"] = scales
        # comparison sheet inputs (same naming as make_comparisons.py)
        off.save(os.path.join(img_dir, f"{name}_exp_official_1.50.png"))
        img_lean.save(os.path.join(img_dir, f"{name}_exp_optimized_1.50.png"))
        colorized.save(os.path.join(img_dir, f"{name}_exp_diffusion.png"))
        del official_mr
        torch.cuda.empty_cache()
        save_json(report, "experiments2.json")
    exp1.SAMPLER.close()
    save_json(report, "experiments2.json")


if __name__ == "__main__":
    main()
