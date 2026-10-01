"""Side-by-side quality comparison images from bench/results/images.

  python bench/make_comparisons.py

For every case writes bench/results/compare_<case>.png:
  row 1: line art + hints | official 1.5x | optimized 1.5x | optimized 1.25x | optimized 1.0x | no post-process
         (end-to-end outputs written by bench/run_baseline.py)
  row 2: zoomed crop (face / hint region) of the same panels
  row 3: |official - X| x10 difference maps (X resized to the official size)
"""
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import RESULTS

IMG = os.path.join(RESULTS, "images")
PANEL_H = 480


def load(path):
    return Image.open(path).convert("RGB") if os.path.exists(path) else None


def fit(img, h=PANEL_H):
    return img.resize((int(img.width * h / img.height), h), Image.BICUBIC)


def main():
    cases = sorted({os.path.basename(p).split("_app_k3.png")[0] for p in glob.glob(os.path.join(IMG, "*_app_k3.png"))})
    for case in cases:
        # end-to-end outputs of bench/run_baseline.py: same inputs, hints, seed and top_k
        panels = [
            ("line art + hints", load(os.path.join(IMG, f"{case}_hints.png"))),
            ("official app.py 1.5x", load(os.path.join(IMG, f"{case}_app_k3.png"))),
            ("optimized 1.5x", load(os.path.join(IMG, f"{case}_app_optimized_k3_s1.5.png"))),
            ("optimized 1.25x", load(os.path.join(IMG, f"{case}_app_optimized_k3_s1.25.png"))),
            ("optimized 1.0x", load(os.path.join(IMG, f"{case}_app_optimized_k3_s1.0.png"))),
            ("no post-process", load(os.path.join(IMG, f"{case}_app_optimized_k3_s1.5_nopp.png"))),
        ]
        panels = [(t, p) for t, p in panels if p is not None]
        ref = panels[1][1]
        # crop: upper-centre region (faces in the examples) in relative coordinates
        box = (0.25, 0.08, 0.75, 0.42)
        rows = [[], [], []]
        for title, p in panels:
            full = fit(p)
            crop = p.crop(tuple(int(v * s) for v, s in zip(box, (p.width, p.height, p.width, p.height))))
            crop = crop.resize(full.size, Image.LANCZOS)
            pr = p.resize(ref.size, Image.BICUBIC)
            diff = np.clip(np.abs(np.asarray(pr, int) - np.asarray(ref, int)) * 10, 0, 255).astype(np.uint8)
            diff = fit(Image.fromarray(diff))
            for row, im in zip(rows, (full, crop, diff)):
                d = ImageDraw.Draw(im)
                d.rectangle((0, 0, im.width, 22), fill="black")
                d.text((4, 4), title, fill="white")
                row.append(im)
        width = sum(im.width for im in rows[0])
        sheet = Image.new("RGB", (width, PANEL_H * 3), "white")
        for r, row in enumerate(rows):
            x = 0
            for im in row:
                sheet.paste(im, (x, r * PANEL_H))
                x += im.width
        out = os.path.join(RESULTS, f"compare_{case}.png")
        sheet.save(out)
        print(out)


if __name__ == "__main__":
    main()
