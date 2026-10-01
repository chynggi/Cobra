"""Diffusion-only experiment: PEFT LoRA (r=128, unmerged) vs LoRA merged into the base weights.

  python bench/exp_lora_merge.py [--repeats 3] [--cases shadow_ex0,line_ex0]

The merge is computed in fp32 (W + scale * B @ A) and cast back to fp16 once.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import CASES, RESULTS, apply_hints, import_app, load_case, median, psnr, save_json

import torch


def merge_lora_fp32(model):
    from peft.tuners.tuners_utils import BaseTunerLayer
    n = 0
    for m in model.modules():
        if isinstance(m, BaseTunerLayer) and not m.merged:
            dtype = m.get_base_layer().weight.dtype
            m.float()
            m.merge()
            m.to(dtype)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cases", default="shadow_ex0,line_ex0")
    args = ap.parse_args()
    app = import_app("app_optimized")
    report = {}
    names = sorted(args.cases.split(","), key=lambda n: CASES[n]["style"] != app.cur_style)
    for name in names:
        c, inp, refs = load_case(name)
        if app.cur_style != c["style"]:
            app.change_ckpt(c["style"])  # reloads the (unmerged) LoRA weights
        extracted, hint_color, hint_mask, query_origin, extracted_ori, resolution = app.extract_sketch_line_image(inp, c["style"])
        hint_color, hint_mask, _ = apply_hints(app, hint_color, hint_mask, refs, c["hints"])
        call = (extracted, refs, resolution, c["seed"], 10, 3, hint_mask, hint_color, query_origin, extracted_ori)

        def run():
            app.run_colorization(*call, enable_postprocess=False)  # warm-up
            ts = []
            for _ in range(args.repeats):
                torch.cuda.synchronize(); t = time.perf_counter()
                img = app.run_colorization(*call, enable_postprocess=False)[0]
                torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000)
            return median(ts), img

        t_unmerged, img_u = run()
        from peft.tuners.tuners_utils import BaseTunerLayer
        tuners = [m for m in app.pipeline.transformer.modules() if isinstance(m, BaseTunerLayer)]
        backup = [m.get_base_layer().weight.detach().cpu().clone() for m in tuners]
        n = merge_lora_fp32(app.pipeline.transformer)
        t_merged, img_m = run()
        report[name] = dict(lora_layers=n, unmerged_ms=t_unmerged, merged_ms=t_merged,
                            psnr_merged_vs_unmerged=psnr(img_m, img_u))
        img_m.save(os.path.join(RESULTS, "images", f"{name}_lora_merged_diffusion.png"))
        img_u.save(os.path.join(RESULTS, "images", f"{name}_lora_unmerged_diffusion.png"))
        print(name, report[name], flush=True)
        # restore the exact unmerged base weights for the next case
        for m, w in zip(tuners, backup):
            m.get_base_layer().weight.data.copy_(w)
            m.merged_adapters.clear()
        del backup
    save_json(report, "exp_lora_merge.json")


if __name__ == "__main__":
    main()
