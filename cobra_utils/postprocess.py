"""Optimized Cobra post-processing (high-resolution VAE + MultiResNet reconstruction)
and profiling helpers.

The post-processing stage re-encodes the colorized diffusion output and the original
line art with the VAE encoder, merges the encoder skip features with
MultiHiddenResNetModel and decodes the colorized latent with those features injected.
"""
import copy
import time
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# --------------------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------------------
class StageTimer:
    """Per-stage timer. Records CPU wall-clock and GPU stream time (torch.cuda.Event).

    With ``sync=True`` (benchmark mode) the device is synchronized around every stage so
    wall-clock numbers are attributable to that stage, and per-stage peak VRAM is tracked.
    With ``enabled=False`` it is a no-op and adds no synchronization to production code.
    """

    def __init__(self, enabled=True, sync=True):
        self.enabled = enabled
        self.sync = sync
        self.records = []
        self.peak_allocated = 0
        self.peak_reserved = 0

    def __call__(self, name):
        return self._stage(name) if self.enabled else nullcontext()

    @contextmanager
    def _stage(self, name):
        if self.sync:
            torch.cuda.synchronize()
            self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated())
            self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved())
            torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            end.record()
            stage_peak = None
            if self.sync:
                torch.cuda.synchronize()
                stage_peak = torch.cuda.max_memory_allocated()
            self.records.append([name, (time.perf_counter() - t0) * 1000.0, start, end, stage_peak])

    def results(self):
        """Returns {name: {'wall_ms', 'gpu_ms', 'peak_alloc_mb'}} (summed over repeats of a name)."""
        torch.cuda.synchronize()
        out = {}
        for name, wall, start, end, peak in self.records:
            r = out.setdefault(name, {"wall_ms": 0.0, "gpu_ms": 0.0, "peak_alloc_mb": 0.0})
            r["wall_ms"] += wall
            r["gpu_ms"] += start.elapsed_time(end)
            if peak is not None:
                r["peak_alloc_mb"] = max(r["peak_alloc_mb"], peak / 2**20)
        return out

    def overall_peaks(self):
        torch.cuda.synchronize()
        return (max(self.peak_allocated, torch.cuda.max_memory_allocated()),
                max(self.peak_reserved, torch.cuda.max_memory_reserved()))


NULL_TIMER = StageTimer(enabled=False)


# --------------------------------------------------------------------------------------
# Tensor conversion (kept on the GPU; bit-identical to the torchvision/numpy versions)
# --------------------------------------------------------------------------------------
def pil_to_normalized_tensor(image, device, dtype):
    """Same result as transforms.Normalize([.5]*3, [.5]*3)(transforms.ToTensor()(image)),
    but the uint8 -> float conversion and normalization run on the GPU (4x less H2D traffic)."""
    arr = torch.from_numpy(np.array(image.convert("RGB"))).to(device)
    x = arr.permute(2, 0, 1).unsqueeze(0).float().div_(255).sub_(0.5).div_(0.5)
    return x.to(dtype)


def tensor_to_pil(output):
    """Same result as the original
    ``output[output > 1] = 1; output[output < -1] = -1``
    ``Image.fromarray(((output[0]*0.5+0.5).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))``
    but scaling/quantization run on the GPU and only uint8 data is copied to the host."""
    out = output[0].clamp(-1, 1)
    out = (out * 0.5 + 0.5).permute(1, 2, 0) * 255
    return Image.fromarray(out.to(torch.uint8).cpu().numpy()).convert("RGB")


# --------------------------------------------------------------------------------------
# VAE helpers
# --------------------------------------------------------------------------------------
def encode_skip_features(encoder, x):
    """VAE encoder skip features only (conv_in + down blocks).

    Identical to the ``hidden_list`` returned by ``Encoder.forward(x, hidden_flag=True)``
    in eval mode, but skips the mid block (attention), conv_out and quant_conv whose
    outputs are discarded for the line-art branch.
    """
    sample = encoder.conv_in(x)
    hidden_list = [sample]
    for down_block in encoder.down_blocks:
        sample = down_block(sample)
        hidden_list.append(sample)
    return hidden_list


def decode_with_skip_features(vae, z, hidden_list):
    """``vae._decode(z, return_dict=False, hidden_list=hidden_list)[0]`` (eval path, identical
    operations), but each skip feature is released as soon as it has been added, so the
    full-resolution feature is not alive while the last (largest) up block runs.
    ``hidden_list`` is consumed (its entries are set to None)."""
    decoder = vae.decoder
    hidden_list.reverse()
    sample = decoder.conv_in(vae.post_quant_conv(z))
    upscale_dtype = next(iter(decoder.up_blocks.parameters())).dtype
    sample += hidden_list[0]
    hidden_list[0] = None
    sample = decoder.mid_block(sample, None)
    sample = sample.to(upscale_dtype)
    for idx, up_block in enumerate(decoder.up_blocks, start=1):
        sample += hidden_list[idx]
        hidden_list[idx] = None
        sample = up_block(sample, None)
    sample = decoder.conv_norm_out(sample)
    sample = decoder.conv_act(sample)
    return decoder.conv_out(sample)


# --------------------------------------------------------------------------------------
# MultiHiddenResNetModel, memory-lean
# --------------------------------------------------------------------------------------
class _LeanResNetBlock(nn.Module):
    """ResNetBlock forward with in-place ReLU; same modules and operations as the original."""

    def __init__(self, block):
        super().__init__()
        self.conv1, self.bn1, self.conv2, self.bn2 = block.conv1, block.bn1, block.conv2, block.bn2
        self.shortcut = block.shortcut

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class LeanMultiHiddenResNetModel(nn.Module):
    """MultiHiddenResNetModel with in-place ReLU whose ``forward`` consumes ``tensor_list``
    (each input is released once processed) to lower peak VRAM. Bit-exact with the original.

    The copy keeps the source model's train/eval mode. Note: the official app never calls
    ``MultiResNetModel.eval()``, so its BatchNorm layers normalize with per-image batch
    statistics; keep that mode to reproduce the official output (eval mode or folding BN
    into the convolutions changes the result, ~36 dB PSNR vs the official output).
    """

    def __init__(self, model):
        super().__init__()
        model = copy.deepcopy(model).float()
        self.two_layer_resnets = nn.ModuleList([
            nn.Sequential(*[_LeanResNetBlock(getattr(m, f"block{i}")) for i in range(1, 5)])
            for m in model.two_layer_resnets
        ])
        self.train(model.training)

    def forward(self, tensor_list):
        out = []
        for i, resnet in enumerate(self.two_layer_resnets):
            out.append(resnet(tensor_list[i]))
            tensor_list[i] = None
        return out


# --------------------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------------------
def postprocess_size(resolution, scale):
    """Target size of the post-processing pass; multiples of 8 so the VAE skip features
    line up with the decoder. Equals (int(w*1.5), int(h*1.5)) for the default scale."""
    tar_width, tar_height = resolution
    return (int(round(tar_width * scale / 8)) * 8, int(round(tar_height * scale / 8)) * 8)


@torch.inference_mode()
def postprocess_colorized_image(colorized_image, original_image, resolution, vae, multires_model,
                                scale=1.5, timer=NULL_TIMER):
    """High-resolution reconstruction of the diffusion output.

    Args:
        colorized_image: PIL output of the Cobra pipeline (at ``resolution``).
        original_image: PIL line art (``extracted_image_ori``) guiding the reconstruction.
        resolution: (width, height) used for diffusion.
        vae: pipeline.vae
        multires_model: MultiHiddenResNetModel (or LeanMultiHiddenResNetModel).
        scale: post-processing resolution relative to ``resolution`` (official: 1.5).
    """
    device, dtype = vae.device, vae.dtype
    size = postprocess_size(resolution, scale)

    with timer("A.resize"):
        up_img = colorized_image.resize(size)
    with timer("B.test_low_color"):
        test_low_color = pil_to_normalized_tensor(up_img, device, dtype)

    with timer("C.query_image_vae_"):
        query_image_vae_ = pil_to_normalized_tensor(original_image.resize(size), device, dtype)
    with timer("E.vae_encode_bw"):
        hidden_list_bw = encode_skip_features(vae.encoder, query_image_vae_)
        del query_image_vae_

    with timer("D.vae_encode_color"):
        h_color, hidden_list_color = vae._encode(test_low_color, return_dict=False, hidden_flag=True)
        del test_low_color
    with timer("F.concat"):
        hidden_list_double = [torch.cat((c, b), dim=1) for c, b in zip(hidden_list_color, hidden_list_bw)]
        del hidden_list_color, hidden_list_bw
    with timer("G.multires"):
        hidden_list = multires_model(hidden_list_double)
        del hidden_list_double
    with timer("H.vae_decode"):
        output = decode_with_skip_features(vae, h_color.sample(), hidden_list)
        del hidden_list
    with timer("I.to_pil"):
        return tensor_to_pil(output)
