"""Where does model loading time go? (answers: would pre-converted fp16 checkpoints help?)

  python bench/profile_load.py [app|app_optimized]
"""
import cProfile
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import import_app

module = sys.argv[1] if len(sys.argv) > 1 else "app_optimized"
prof = cProfile.Profile()
t = time.perf_counter()
prof.enable()
import_app(module)
prof.disable()
print(f"total import/load: {time.perf_counter() - t:.1f} s")

stats = pstats.Stats(prof)
interesting = ("load_state_dict", "torch/serialization.py", "from_pretrained", "init_causal_dit", "add_adapter",
               "snapshot_download", "module.py:1340(to)", "load_ckpt", "__init__", "safetensors", "res_skip",
               "_load_from_state_dict", "_apply")
rows = []
for (fn, line, name), (cc, nc, tt, ct, callers) in stats.stats.items():
    key = f"{fn}:{line}({name})"
    if ct > 0.5 and any(s in key for s in interesting):
        rows.append((ct, nc, key))
for ct, nc, key in sorted(rows, reverse=True)[:30]:
    print(f"{ct:7.2f} s  {nc:6d} calls  {key[-110:]}")
