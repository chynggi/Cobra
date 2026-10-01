"""VRAM housekeeping for Windows (WDDM) GPUs.

With the default "CUDA sysmem fallback" driver policy, cudaMalloc beyond physical VRAM does
not fail: the driver places the allocation in shared system memory and every kernel touching
it becomes very slow. PyTorch's caching allocator therefore never reaches its own
"out of memory -> free cached blocks -> retry" path; blocks cached by one phase (diffusion)
that do not fit the next phase's tensors (post-processing) stay reserved while new ones are
allocated on top (measured with app.py: 11.6 GB allocated, 13.5+ GB reserved on a 12 GB card).

A hard cap (torch.cuda.set_per_process_memory_fraction) would stop that, but it cannot be
lifted again and turns workloads that genuinely need more than the card (e.g. Top K 20) into
CUDA OOM errors. Instead the cache is released at phase boundaries when it holds a lot of
unused memory, and memory pressure is estimated before diffusion.
"""
import torch


def release_unused_cache(threshold_gb=1.0, device=0):
    """``torch.cuda.empty_cache()`` only if more than ``threshold_gb`` is reserved but unused."""
    unused = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    if unused > threshold_gb * 2**30:
        torch.cuda.empty_cache()
        return True
    return False


def available_vram(device=0):
    """Bytes this process can still use without spilling: free VRAM + its own unused cache."""
    free, _ = torch.cuda.mem_get_info(device)
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
