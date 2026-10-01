"""Shared helpers for the Cobra benchmarks (inputs, hints, metrics)."""
import json
import os
import statistics
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "bench", "results")


def import_app(module_name):
    """Imports app.py / app_optimized.py without starting the Gradio server."""
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import gradio as gr
    gr.Blocks.launch = lambda self, *a, **k: None
    return __import__(module_name)


# (input, refs, style, seed, steps, top_k) taken from the official Quick Examples, plus
# deterministic colour hints: (hint position on line art as fraction, colour pick position on ref 0)
CASES = {
    "line_ex0": dict(dir="examples/line/example0", n_refs=3, style="line", seed=0, top_k=6,
                     hints=[((0.50, 0.30), (0.50, 0.30)), ((0.40, 0.65), (0.45, 0.65)), ((0.62, 0.15), (0.55, 0.15))]),
    "line_ex2": dict(dir="examples/line/example2", n_refs=4, style="line", seed=1, top_k=20,
                     hints=[((0.50, 0.30), (0.50, 0.30)), ((0.45, 0.70), (0.45, 0.70))]),
    "shadow_ex0": dict(dir="examples/shadow/example0", n_refs=4, style="line + shadow", seed=1, top_k=20,
                       hints=[((0.50, 0.30), (0.50, 0.30)), ((0.45, 0.70), (0.45, 0.70))]),
    "shadow_ex2": dict(dir="examples/shadow/example2", n_refs=1, style="line + shadow", seed=4, top_k=3,
                       hints=[((0.50, 0.35), (0.50, 0.35)), ((0.40, 0.70), (0.40, 0.70))]),
}


def load_case(name):
    c = CASES[name]
    inp = Image.open(os.path.join(REPO, c["dir"], "input.png"))
    refs = [SimpleNamespace(name=os.path.join(REPO, c["dir"], f"reference_image_{i}.png")) for i in range(c["n_refs"])]
    return c, inp, refs


def apply_hints(app, hint_color, hint_mask, refs, hints):
    """Places hints exactly like the UI: get_color_value on the reference, draw_square on the line art."""
    ref0 = np.array(Image.open(refs[0].name).convert("RGB"))
    placed = []
    for (hx, hy), (rx, ry) in hints:
        evt_ref = SimpleNamespace(index=(int(rx * ref0.shape[1]), int(ry * ref0.shape[0])))
        _, color = app.get_color_value(ref0, evt_ref)
        w, h = hint_color.size
        evt = SimpleNamespace(index=(int(hx * w), int(hy * h)))
        hint_color, hint_mask = app.draw_square(hint_color, hint_mask, color, evt)
        placed.append((evt.index, tuple(int(v) for v in color)))
    return hint_color, hint_mask, placed


# ------------------------------------------------------------------ metrics
def psnr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def _lab(img):
    import cv2
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float64)


def delta_e(a, b, size):
    """Mean CIE76 colour difference after resizing both to `size` (colour fidelity, not sharpness)."""
    la, lb = _lab(a.resize(size, Image.BICUBIC)), _lab(b.resize(size, Image.BICUBIC))
    # OpenCV 8-bit Lab: L*255/100, a+128, b+128
    la[..., 0] *= 100 / 255; lb[..., 0] *= 100 / 255
    return float(np.mean(np.sqrt(np.sum((la - lb) ** 2, axis=-1))))


def sharpness(img):
    import cv2
    g = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def hint_error(img, placed, line_size):
    """Mean RGB distance between each hint colour and the output inside its 16x16 square."""
    arr = np.asarray(img.convert("RGB").resize(line_size, Image.BICUBIC), dtype=np.float64)
    errs = []
    for (x, y), color in placed:
        patch = arr[max(0, y - 8):y + 8, max(0, x - 8):x + 8].reshape(-1, 3)
        errs.append(float(np.linalg.norm(patch.mean(0) - np.array(color))))
    return errs


def line_darkness(img, line_img, line_size):
    """Mean luminance of the output on pixels that are dark in the line art (line preservation)."""
    out = np.asarray(img.convert("L").resize(line_size, Image.BICUBIC), dtype=np.float64)
    mask = np.asarray(line_img.convert("L").resize(line_size), dtype=np.float64) < 100
    return float(out[mask].mean())


def median(xs):
    return statistics.median(xs)


def save_json(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class GpuMemSampler:
    """Samples this process' WDDM GPU memory once per second via Windows performance counters.

    'Dedicated Usage' = VRAM; 'Non Local Usage' = GPU allocations that the driver placed in
    system RAM (CUDA sysmem fallback when VRAM is exhausted -> very slow kernels).
    """

    def __init__(self):
        import subprocess
        import threading
        import time as _time
        self._time = _time
        pid = os.getpid()
        cmd = ("Get-Counter -Counter '\\GPU Process Memory(pid_%d_*)\\Dedicated Usage','\\GPU Process Memory(pid_%d_*)\\Non Local Usage' "
               "-SampleInterval 1 -Continuous | ForEach-Object { $d=0; $n=0; foreach($s in $_.CounterSamples) "
               "{ if ($s.Path -like '*dedicated usage') { $d += $s.CookedValue } else { $n += $s.CookedValue } }; "
               "Write-Output \"$d $n\" }") % (pid, pid)
        self.samples = []
        self.proc = subprocess.Popen(["powershell", "-NoProfile", "-Command", cmd], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                d, n = (float(v) for v in line.split())
                self.samples.append((self._time.perf_counter(), d / 2**20, n / 2**20))
            except ValueError:
                pass

    def window(self, t0, t1):
        """Max dedicated / non-local MB sampled in [t0, t1] (perf_counter seconds)."""
        s = [x for x in self.samples if t0 <= x[0] <= t1 + 1.5]
        if not s:
            return {"dedicated_max_mb": None, "nonlocal_max_mb": None}
        return {"dedicated_max_mb": max(x[1] for x in s), "nonlocal_max_mb": max(x[2] for x in s)}

    def close(self):
        self.proc.kill()


def gpu_mem():
    return dict(
        alloc_mb=torch.cuda.memory_allocated() / 2**20,
        reserved_mb=torch.cuda.memory_reserved() / 2**20,
        peak_alloc_mb=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20,
    )
