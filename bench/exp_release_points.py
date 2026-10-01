"""Ablation: release the unused CUDA cache before diffusion AND post-processing (app_optimized
default) vs only before post-processing.

  python bench/exp_release_points.py [--repeats 3]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import CASES, GpuMemSampler, apply_hints, gpu_mem, import_app, load_case, median, save_json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    app = import_app("app_optimized")
    sampler = GpuMemSampler()
    release = app.release_unused_cache
    calls = {"n": 0}

    def post_only(threshold_gb):  # run_colorization calls it before diffusion, then before post-processing
        calls["n"] += 1
        return release(threshold_gb) if calls["n"] % 2 == 0 else False

    report = {}
    names = sorted(CASES, key=lambda n: CASES[n]["style"] != app.cur_style)
    for variant, fn in (("both", release), ("post_only", post_only)):
        app.release_unused_cache = fn
        for name in names:
            c, inp, refs = load_case(name)
            if app.cur_style != c["style"]:
                app.change_ckpt(c["style"])
            extracted, hint_color, hint_mask, query_origin, extracted_ori, resolution = app.extract_sketch_line_image(inp, c["style"])
            hint_color, hint_mask, _ = apply_hints(app, hint_color, hint_mask, refs, c["hints"])
            call = (extracted, refs, resolution, c["seed"], 10, 3, hint_mask, hint_color, query_origin, extracted_ori)
            calls["n"] = 0
            app.run_colorization(*call)
            ts, spills = [], []
            for _ in range(args.repeats):
                calls["n"] = 0
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
                app.run_colorization(*call)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                ts.append((t1 - t) * 1000)
                time.sleep(1.5)
                spills.append(sampler.window(t, t1)["nonlocal_max_mb"])
            report.setdefault(variant, {})[name] = dict(e2e_ms=median(ts), runs=ts, spill_mb=max(spills), mem=gpu_mem())
            print(variant, name, report[variant][name], flush=True)
    sampler.close()
    save_json(report, "exp_release_points.json")


if __name__ == "__main__":
    main()
