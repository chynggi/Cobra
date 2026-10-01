"""Builds the markdown benchmark tables from bench/results/*.json.

  python bench/summarize.py > bench/results/summary.md
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import RESULTS

CASES = ["line_ex0", "line_ex2", "shadow_ex0", "shadow_ex2"]
POST = ("A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.")


def load(name):
    p = os.path.join(RESULTS, name + ".json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def sec(ms):
    return "-" if ms is None else f"{ms / 1000:.2f}"


def stage_split(case):
    st = case.get("stages")
    if not st:
        return None, None
    diff = st["diffusion"]["wall_ms"]
    post = sum(v["wall_ms"] for k, v in st.items() if k.startswith(POST))
    return diff, post


def spill(case):
    vals = [w["nonlocal_max_mb"] for w in case.get("e2e_wddm", []) if w.get("nonlocal_max_mb") is not None]
    return max(vals) if vals else None


def main():
    base = load("app_k3")
    configs = [
        ("Baseline (official app.py)", "app_k3", "1.50"),
        ("Optimized", "app_optimized_k3_s1.5", "1.50"),
        ("Optimized A", "app_optimized_k3_s1.25", "1.25"),
        ("Optimized B", "app_optimized_k3_s1.0", "1.00"),
        ("No postprocess", "app_optimized_k3_s1.5_nopp", "-"),
    ]
    print("## End-to-end (Colorize click), top_k=3, 10 steps, median of runs\n")
    print("Diffusion and post-processing come from the separate synchronized per-stage run ('-' = not profiled). "
          "Total e2e is the production code path without extra synchronization.\n")
    print("| Configuration | Case | Postprocess scale | Diffusion, profiled (s) | Postprocess, profiled (s) | Total e2e (s) | e2e runs (s) | Peak alloc / reserved (GB) | Spill to shared mem (MB, idle≈78) | Output |")
    print("|---|---|---:|---:|---:|---:|---|---:|---:|---|")
    for label, tag, scale in configs:
        d = load(tag)
        if not d:
            continue
        for c in CASES:
            case = d["cases"].get(c)
            if not case:
                continue
            diff, post = stage_split(case)
            m = case["e2e_mem"]
            runs = ", ".join(f"{x / 1000:.1f}" for x in case["e2e_runs_ms"])
            print(f"| {label} | {c} | {scale} | {sec(diff)} | {sec(post)} | **{sec(case['e2e_total_ms'])}** | {runs} | "
                  f"{m['peak_alloc_mb'] / 1024:.2f} / {m['peak_reserved_mb'] / 1024:.2f} | {spill(case):.0f} | {case['output_size'][0]}x{case['output_size'][1]} |")

    print("\n## Improvement vs baseline (median e2e)\n")
    print("| Case | Baseline (s) | Optimized 1.5x (s) | Improvement | Optimized 1.25x (s) | Optimized 1.0x (s) | No postprocess (s) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    opt = {t: load(t) for _, t, _ in configs[1:]}
    for c in CASES:
        b = base["cases"][c]["e2e_total_ms"] if base and c in base["cases"] else None
        vals = [opt[t]["cases"][c]["e2e_total_ms"] if opt[t] and c in opt[t]["cases"] else None for _, t, _ in configs[1:]]
        imp = f"{(1 - vals[0] / b) * 100:.1f}% ({b / vals[0]:.2f}x)" if b and vals[0] else "-"
        print(f"| {c} | {sec(b)} | {sec(vals[0])} | {imp} | {sec(vals[1])} | {sec(vals[2])} | {sec(vals[3])} |")

    print("\n## Per-stage profile at 1.5x (synchronized, GPU ms, median)\n")
    o = load("app_optimized_k3_s1.5")
    for c in CASES:
        if not (base and o and c in base["cases"] and c in o["cases"]):
            continue
        sb, so = base["cases"][c]["stages"], o["cases"][c]["stages"]
        names = [k for k in sb if not k.startswith("_")] + [k for k in so if not k.startswith("_") and k not in sb]
        print(f"\n**{c}** — baseline total {sec(sb['_total']['wall_ms'])} s, optimized total {sec(so['_total']['wall_ms'])} s\n")
        print("| Stage | Baseline GPU ms | Baseline stage peak alloc MB | Optimized GPU ms | Optimized stage peak alloc MB |")
        print("|---|---:|---:|---:|---:|")
        for k in names:
            a, bb = sb.get(k), so.get(k)
            print(f"| {k} | {a['gpu_ms']:.0f} | {a['peak_alloc_mb']:.0f} |" if a else f"| {k} | - | - |", end="")
            print(f" {bb['gpu_ms']:.0f} | {bb['peak_alloc_mb']:.0f} |" if bb else " - | - |")

    print("\n## Steps sweep (optimized, 1.5x)\n")
    print("| Case | Steps | Diffusion only (s) | Full (s) |")
    print("|---|---:|---:|---:|")
    for c in CASES:
        sw = o["cases"].get(c, {}).get("steps_sweep") if o else None
        for steps, v in (sw or {}).items():
            print(f"| {c} | {steps} | {sec(v['diffusion_ms'])} | {sec(v['full_ms'])} |")

    print("\n## Output quality vs the official output (same inputs, hints, seed)\n")
    print("PSNR at the official 1.5x size (lower-res outputs bicubic-upscaled); colour difference (mean CIE76 dE) and "
          "sharpness (Laplacian variance) at the diffusion resolution; hint error = RGB distance between each hint "
          "colour and the output inside its 16x16 square; line darkness = mean luminance on line-art pixels.\n")
    print("| Case | Output | PSNR vs official (dB) | max abs diff | identical px | dE vs official | Sharpness | Hint error | Line darkness |")
    print("|---|---|---:|---:|---:|---:|---:|---|---:|")
    from PIL import Image
    import numpy as np
    from bench.common import delta_e, hint_error, line_darkness, psnr, sharpness
    img_dir = os.path.join(RESULTS, "images")
    outputs = [("official app.py 1.5x", "app_k3"), ("optimized 1.5x", "app_optimized_k3_s1.5"),
               ("optimized 1.25x", "app_optimized_k3_s1.25"), ("optimized 1.0x", "app_optimized_k3_s1.0"),
               ("no post-process", "app_optimized_k3_s1.5_nopp")]
    for c in CASES:
        if not (base and c in base["cases"]):
            continue
        res = tuple(base["cases"][c]["resolution"])
        placed = [((x, y), tuple(col)) for (x, y), col in base["cases"][c]["hints"]]
        ref = Image.open(os.path.join(img_dir, f"{c}_app_k3.png")).convert("RGB")
        line = Image.open(os.path.join(img_dir, f"{c}_lineart.png"))
        for label, tag in outputs:
            p = os.path.join(img_dir, f"{c}_{tag}.png")
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB")
            up = np.asarray(im.resize(ref.size, Image.BICUBIC), int)
            r = np.asarray(ref, int)
            same = (up == r).all(-1).mean() * 100 if im.size == ref.size else float("nan")
            print(f"| {c} | {label} | {psnr(up, r):.2f} | {np.abs(up - r).max()} | {same:.1f}% | {delta_e(im, ref, res):.2f} | "
                  f"{sharpness(im.resize(res, Image.BICUBIC)):.0f} | {', '.join(f'{e:.0f}' for e in hint_error(im, placed, res))} | "
                  f"{line_darkness(im, line, res):.1f} |")

    print("\n## Top K = 20 (80 reference patches), 1 run\n")
    print("| Implementation | Case | Total e2e (s) | Peak alloc / reserved (GB) | Spill (MB) |")
    print("|---|---|---:|---:|---:|")
    for label, tag in (("Baseline", "app_k20"), ("Optimized", "app_optimized_k20_s1.5")):
        d = load(tag)
        for c, case in (d or {}).get("cases", {}).items():
            if "e2e_total_ms" not in case:
                continue
            m = case["e2e_mem"]
            print(f"| {label} | {c} | {sec(case['e2e_total_ms'])} | {m['peak_alloc_mb'] / 1024:.2f} / {m['peak_reserved_mb'] / 1024:.2f} | {spill(case):.0f} |")


if __name__ == "__main__":
    main()
