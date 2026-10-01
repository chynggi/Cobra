"""End-to-end + per-stage benchmark of the official app.py or app_optimized.py.

  python bench/run_baseline.py --impl app            [--top-k 3] [--repeats 3] [--steps-sweep]
  python bench/run_baseline.py --impl app_optimized  [--postprocess-scale 1.5] [--no-postprocess]

--top-k overrides the example value (default 3 = the Gradio UI default). Each implementation
runs in its own process with identical inputs/hints/seeds. Results: bench/results/<tag>.json
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import (CASES, RESULTS, GpuMemSampler, apply_hints, gpu_mem, import_app, load_case, median,
                          save_json)

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="app", choices=["app", "app_optimized"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cases", default="line_ex0,line_ex2,shadow_ex0,shadow_ex2")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--steps-sweep", action="store_true")
    ap.add_argument("--postprocess-scale", type=float, default=1.5)
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--no-profile", action="store_true", help="only end-to-end timing")
    ap.add_argument("--no-release-cache", action="store_true",
                    help="app_optimized: disable the conditional cache release at phase boundaries")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    optimized = args.impl == "app_optimized"
    tag = args.tag or (f"{args.impl}_k{args.top_k}" + (f"_s{args.postprocess_scale}" if optimized else "")
                       + ("_nopp" if args.no_postprocess else "") + ("_norelease" if args.no_release_cache else ""))

    t0 = time.perf_counter()
    app = import_app(args.impl)
    load_s = time.perf_counter() - t0
    if args.no_release_cache:
        app.RELEASE_CACHE_THRESHOLD_GB = None
    from bench.baseline_profiled import colorize_image_profiled
    from cobra_utils.postprocess import NULL_TIMER, StageTimer

    def run(timer, call, steps=args.steps):
        """Returns (final image, diffusion image or None)."""
        call = call[:4] + (steps,) + call[5:]
        if optimized:
            gallery = app.run_colorization(*call, enable_postprocess=not args.no_postprocess,
                                           postprocess_scale=args.postprocess_scale, timer=timer or NULL_TIMER)
            return gallery[0], None
        if timer is None:
            return app.colorize_image(*call)[0], None
        hi, colorized, _ = colorize_image_profiled(app, timer, *call, run_postprocess=not args.no_postprocess)
        return hi, colorized

    torch.cuda.synchronize()
    sampler = GpuMemSampler()
    report = {"impl": args.impl, "args": vars(args), "model_load_s": load_s, "after_load_mem": gpu_mem(),
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cases": {}}
    print(f"[{tag}] model load {load_s:.1f}s", report["after_load_mem"], flush=True)
    img_dir = os.path.join(RESULTS, "images")
    os.makedirs(img_dir, exist_ok=True)

    names = sorted(args.cases.split(","), key=lambda n: CASES[n]["style"] != app.cur_style)
    for name in names:
        c, inp, refs = load_case(name)
        r = report["cases"][name] = {"style": c["style"], "seed": c["seed"], "top_k": args.top_k}

        # style switch (Load Model) is timed separately from line extraction
        if app.cur_style != c["style"]:
            torch.cuda.synchronize(); t = time.perf_counter()
            app.change_ckpt(c["style"])
            torch.cuda.synchronize()
            r["change_ckpt_s"] = time.perf_counter() - t

        ext = []
        for _ in range(args.repeats):
            torch.cuda.synchronize(); t = time.perf_counter()
            outs = app.extract_sketch_line_image(inp, c["style"])
            torch.cuda.synchronize(); ext.append((time.perf_counter() - t) * 1000)
        r["line_extraction_ms"] = median(ext)
        extracted, hint_color, hint_mask, query_origin, extracted_ori, resolution = outs
        hint_color, hint_mask, placed = apply_hints(app, hint_color, hint_mask, refs, c["hints"])
        r["resolution"], r["hints"] = list(resolution), placed
        hint_color.save(os.path.join(img_dir, f"{name}_hints.png"))
        extracted_ori.save(os.path.join(img_dir, f"{name}_lineart.png"))
        call = (extracted, refs, resolution, c["seed"], args.steps, args.top_k, hint_mask, hint_color, query_origin, extracted_ori)

        # warm-up (cuDNN algorithm selection, allocator growth)
        t = time.perf_counter()
        run(None, call)
        r["first_run_ms"] = (time.perf_counter() - t) * 1000

        # 1) end-to-end, production code path (no extra synchronization inside)
        e2e, mems, spills = [], [], []
        for _ in range(args.repeats):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
            img, _ = run(None, call)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            e2e.append((t1 - t) * 1000)
            mems.append(gpu_mem())
            time.sleep(1.5)  # let the 1 s counter sampler cover the run window
            spills.append(sampler.window(t, t1))
        r["e2e_total_ms"] = median(e2e)
        r["e2e_runs_ms"] = e2e
        r["e2e_mem"] = mems[-1]
        r["e2e_wddm"] = spills
        img.save(os.path.join(img_dir, f"{name}_{tag}.png"))
        r["output_size"] = list(img.size)
        print(f"[{tag}] {name} e2e {r['e2e_runs_ms']} ms  peak alloc {mems[-1]['peak_alloc_mb']:.0f} MB "
              f"reserved {mems[-1]['peak_reserved_mb']:.0f} MB  wddm {spills[-1]}", flush=True)

        # 2) per-stage (synchronized around every stage)
        if not args.no_profile:
            stage_runs = []
            for _ in range(args.repeats):
                timer = StageTimer(enabled=True, sync=True)
                torch.cuda.reset_peak_memory_stats()
                t = time.perf_counter()
                hi, colorized = run(timer, call)
                total = (time.perf_counter() - t) * 1000
                res = timer.results()
                res["_total"] = {"wall_ms": total}
                res["_peak"] = dict(zip(("alloc_mb", "reserved_mb"), [v / 2**20 for v in timer.overall_peaks()]))
                stage_runs.append(res)
            r["stages"] = {k: {m: median([run_[k][m] for run_ in stage_runs]) for m in stage_runs[0][k]}
                           for k in stage_runs[0]}
            if colorized is not None:
                colorized.save(os.path.join(img_dir, f"{name}_{tag}_diffusion_only.png"))
            print(f"[{tag}] {name} stages(gpu ms) " + ", ".join(
                f"{k}={v.get('gpu_ms', v.get('wall_ms', 0)):.0f}" for k, v in r["stages"].items() if k[0] != "_")
                  + f"  total={r['stages']['_total']['wall_ms']:.0f}  peak={r['stages']['_peak']}", flush=True)

        if args.steps_sweep and name in ("line_ex0", "shadow_ex0"):
            sweep = {}
            for steps in (5, 8, 10, 12, 15):
                runs = []
                for _ in range(max(2, args.repeats - 1)):
                    timer = StageTimer(enabled=True, sync=True)
                    t = time.perf_counter()
                    img, _ = run(timer, call, steps=steps)
                    total = (time.perf_counter() - t) * 1000
                    runs.append((timer.results()["diffusion"]["wall_ms"], total))
                sweep[steps] = {"diffusion_ms": median([a for a, _ in runs]), "full_ms": median([b for _, b in runs])}
                img.save(os.path.join(img_dir, f"{name}_{tag}_steps{steps}.png"))
                print(f"[{tag}] {name} steps {steps} {sweep[steps]}", flush=True)
            r["steps_sweep"] = sweep
        save_json(report, f"{tag}.json")
    sampler.close()
    save_json(report, f"{tag}.json")


if __name__ == "__main__":
    main()
