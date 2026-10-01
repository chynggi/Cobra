import contextlib
import gc
import json
import logging
import math
import os
import random
import shutil
import sys
import time
import itertools
import copy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from safetensors.torch import load_model
from peft import LoraConfig
import gradio as gr
import pandas as pd

import transformers
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    CLIPProcessor,
)

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PixArtTransformer2DModel,
    CausalSparseDiTModel,
    CausalSparseDiTControlModel,
    CobraPixArtAlphaPipeline,
    UniPCMultistepScheduler,
)
from cobra_utils.utils import *
from cobra_utils.postprocess import LeanMultiHiddenResNetModel, StageTimer, NULL_TIMER, postprocess_colorized_image
from cobra_utils.vram import available_vram, release_unused_cache

from huggingface_hub import snapshot_download

# ---- performance options -------------------------------------------------------------
ENABLE_POSTPROCESS = True   # high-resolution VAE + MultiResNet reconstruction (official: always on)
POSTPROCESS_SCALE = 1.5     # reconstruction resolution relative to the diffusion resolution (official: 1.5)
# COBRA_BENCHMARK=1 synchronizes around every stage and prints per-stage timings/VRAM.
# Leave it off for normal use: the extra synchronization only exists for profiling.
BENCHMARK_MODE = os.environ.get("COBRA_BENCHMARK", "0") == "1"
CLIP_DTYPE = torch.float16  # retrieval image encoder (official: float32)
# Before diffusion and before post-processing, release PyTorch's cache when more than this is
# reserved but unused, so it cannot grow into the driver's slow shared-memory fallback
# (see cobra_utils/vram.py). None disables it.
RELEASE_CACHE_THRESHOLD_GB = 1.0
# Diffusion working memory besides the reference K/V cache (measured ~1.6 GB at Top K 3).
DIFFUSION_WORKSPACE_GB = 2.0

model_global_path = snapshot_download(repo_id="JunhaoZhuang/Cobra", cache_dir='./Cobra/', repo_type="model")
print(model_global_path)
examples = [
    [
        "./examples/shadow/example0/input.png", 
        ["./examples/shadow/example0/reference_image_0.png", 
        "./examples/shadow/example0/reference_image_1.png", 
        "./examples/shadow/example0/reference_image_2.png",
        "./examples/shadow/example0/reference_image_3.png"], 
        "line + shadow", # style
        1, # seed
        10, # step
        20, # top k
    ],
    [
        "./examples/shadow/example1/input.png", 
        ["./examples/shadow/example1/reference_image_0.png", 
        "./examples/shadow/example1/reference_image_1.png", 
        "./examples/shadow/example1/reference_image_2.png",
        "./examples/shadow/example1/reference_image_3.png",
        "./examples/shadow/example1/reference_image_4.png",
        "./examples/shadow/example1/reference_image_5.png"], 
        "line + shadow", # style
        1, # seed
        10, # step
        20, # top k
    ],
    [
        "./examples/shadow/example2/input.png", 
        ["./examples/shadow/example2/reference_image_0.png"], 
        "line + shadow", # style
        4, # seed
        10, # step
        3, # top k
    ],
    [
        "./examples/line/example2/input.png", 
        ["./examples/line/example2/reference_image_0.png", 
        "./examples/line/example2/reference_image_1.png", 
        "./examples/line/example2/reference_image_2.png",
        "./examples/line/example2/reference_image_3.png"], 
        "line", # style
        1, # seed
        10, # step
        20, # top k
    ],
    [
        "./examples/line/example0/input.png", 
        ["./examples/line/example0/reference_image_0.png", 
        "./examples/line/example0/reference_image_1.png", 
        "./examples/line/example0/reference_image_2.png"],
        "line", # style
        0, # seed
        10, # step
        6, # top k
    ],
    [
        "./examples/line/example1/input.png", 
        ["./examples/line/example1/reference_image_0.png",],
        "line", # style
        0, # seed
        10, # step
        3, # top k
    ],
    [
        "./examples/line/example3/input.png", 
        ["./examples/line/example3/reference_image_0.png",],
        "line", # style
        0, # seed
        10, # step
        3, # top k
    ],]

ratio_list = [[800, 800], [768, 896], [704, 928], [672, 960], [640, 1024], [608, 1056], [576, 1088], [576, 1184]]
ratio_list += [[896, 768], [928, 704], [960, 672], [1024, 640], [1056, 608], [1088, 576], [1184, 576]]

def get_rate(image):
    # 计算输入图片的长宽比
    input_rate = image.size[0] / image.size[1]
    # 计算与每个预设比例的差距
    min_diff = float('inf')
    best_idx = 0
    
    for i, ratio in enumerate(ratio_list):
        ratio_rate = ratio[0] / ratio[1]
        diff = abs(input_rate - ratio_rate)
        if diff < min_diff:
            min_diff = diff
            best_idx = i
            
    return ratio_list[best_idx]


transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])
weight_dtype = torch.float16

# line model
line_model_path = os.path.join(model_global_path, 'LE', 'erika.pth')
line_model = res_skip()
line_model.load_state_dict(torch.load(line_model_path))
line_model.eval()
line_model.cuda()


# image encoder
image_processor = CLIPImageProcessor()
# fp16 instead of fp32: same top-k patches and the same full similarity ranking on the
# benchmark cases, ~3x faster retrieval and 1.26 GB less VRAM.
image_encoder = CLIPVisionModelWithProjection.from_pretrained(os.path.join(model_global_path, 'image_encoder'), torch_dtype=CLIP_DTYPE).to('cuda')




global pipeline
global MultiResNetModel
global FastMultiResNetModel

def build_fast_multires():
    # MultiResNetModel keeps the fp32 weights on the CPU; the GPU copy computes the same fp16
    # result with in-place ReLU and releases each input as soon as it is processed. Like the
    # official app it is left in training mode (BatchNorm uses per-image statistics).
    global FastMultiResNetModel
    FastMultiResNetModel = None
    FastMultiResNetModel = LeanMultiHiddenResNetModel(MultiResNetModel).to('cuda', dtype=weight_dtype)

def load_ckpt():
    global pipeline
    global MultiResNetModel
    global causal_dit
    global controlnet
    weight_dtype = torch.float16

    block_out_channels = [128, 128, 256, 512, 512]
    MultiResNetModel = MultiHiddenResNetModel(block_out_channels, len(block_out_channels))
    MultiResNetModel.load_state_dict(torch.load(os.path.join(model_global_path, 'shadow_GSRP', 'MultiResNetModel.bin'), map_location='cpu'), strict=True)
    build_fast_multires()


    # transformer
    transform = transforms.Compose([
                transforms.ToTensor(),  
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
            ])
    # seed = 43
    lora_rank = 128
    pretrained_model_name_or_path = "PixArt-alpha/PixArt-XL-2-1024-MS"

    transformer = PixArtTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="transformer", revision=None, variant=None
        )
    pixart_config = get_pixart_config()
    causal_dit = CausalSparseDiTModel(num_attention_heads=pixart_config.get("num_attention_heads"),
                        attention_head_dim=pixart_config.get("attention_head_dim"),
                        in_channels=pixart_config.get("in_channels"),
                        out_channels=pixart_config.get("out_channels"),
                        num_layers=pixart_config.get("num_layers"),
                        dropout=pixart_config.get("dropout"),
                        norm_num_groups=pixart_config.get("norm_num_groups"),
                        cross_attention_dim=pixart_config.get("cross_attention_dim"),
                        attention_bias=pixart_config.get("attention_bias"),
                        sample_size=pixart_config.get("sample_size"),
                        patch_size=pixart_config.get("patch_size"),
                        activation_fn=pixart_config.get("activation_fn"),
                        num_embeds_ada_norm=pixart_config.get("num_embeds_ada_norm"),
                        upcast_attention=pixart_config.get("upcast_attention"),
                        norm_type=pixart_config.get("norm_type"),
                        norm_elementwise_affine=pixart_config.get("norm_elementwise_affine"),
                        norm_eps=pixart_config.get("norm_eps"),
                        caption_channels=pixart_config.get("caption_channels"),
                        attention_type=pixart_config.get("attention_type"))

    causal_dit = init_causal_dit(causal_dit, transformer)
    print('loaded causal_dit')
    controlnet = CausalSparseDiTControlModel(num_attention_heads=pixart_config.get("num_attention_heads"),
                                    attention_head_dim=pixart_config.get("attention_head_dim"),
                                    in_channels=pixart_config.get("in_channels"),
                                    cond_chanels = 9,
                                    out_channels=pixart_config.get("out_channels"),
                                    num_layers=pixart_config.get("num_layers"),
                                    dropout=pixart_config.get("dropout"),
                                    norm_num_groups=pixart_config.get("norm_num_groups"),
                                    cross_attention_dim=pixart_config.get("cross_attention_dim"),
                                    attention_bias=pixart_config.get("attention_bias"),
                                    sample_size=pixart_config.get("sample_size"),
                                    patch_size=pixart_config.get("patch_size"),
                                    activation_fn=pixart_config.get("activation_fn"),
                                    num_embeds_ada_norm=pixart_config.get("num_embeds_ada_norm"),
                                    upcast_attention=pixart_config.get("upcast_attention"),
                                    norm_type=pixart_config.get("norm_type"),
                                    norm_elementwise_affine=pixart_config.get("norm_elementwise_affine"),
                                    norm_eps=pixart_config.get("norm_eps"),
                                    caption_channels=pixart_config.get("caption_channels"),
                                    attention_type=pixart_config.get("attention_type")
                                )
    # controlnet = init_controlnet(controlnet, causal_dit)
    del transformer
    transformer_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            # use_dora=True,
            init_lora_weights="gaussian",
            target_modules=["to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
                "proj",
                "linear",
                "linear_1",
                "linear_2"],
        )
    causal_dit.add_adapter(transformer_lora_config)

    
    lora_state_dict = torch.load(os.path.join(model_global_path, 'shadow_ckpt', 'transformer_lora_pos.bin'), map_location='cpu')
    causal_dit.load_state_dict(lora_state_dict, strict=False)
    controlnet_state_dict = torch.load(os.path.join(model_global_path, 'shadow_ckpt', 'controlnet.bin'), map_location='cpu')
    controlnet.load_state_dict(controlnet_state_dict, strict=True)

    causal_dit.to('cuda', dtype=weight_dtype)
    controlnet.to('cuda', dtype=weight_dtype)

    pipeline = CobraPixArtAlphaPipeline.from_pretrained(
            pretrained_model_name_or_path,
            transformer=causal_dit,
            controlnet=controlnet,
            safety_checker=None,    
            revision=None,
            variant=None,
            torch_dtype=weight_dtype,
        )

    pipeline = pipeline.to("cuda")

global cur_style
cur_style = 'line + shadow'
def change_ckpt(style):
    global pipeline
    global MultiResNetModel
    global cur_style
    weight_dtype = torch.float16

    if style == 'line':
        MultiResNetModel_path = os.path.join(model_global_path, 'line_GSRP', 'MultiResNetModel.bin')
        causal_dit_lora_path = os.path.join(model_global_path, 'line_ckpt', 'transformer_lora_pos.bin')
        controlnet_path = os.path.join(model_global_path, 'line_ckpt', 'controlnet.bin')
    elif style == 'line + shadow':
        MultiResNetModel_path = os.path.join(model_global_path, 'shadow_GSRP', 'MultiResNetModel.bin')
        causal_dit_lora_path = os.path.join(model_global_path, 'shadow_ckpt', 'transformer_lora_pos.bin')
        controlnet_path = os.path.join(model_global_path, 'shadow_ckpt', 'controlnet.bin')
    else:
        raise ValueError("Invalid style: {}".format(style))

    cur_style = style

    MultiResNetModel.load_state_dict(torch.load(MultiResNetModel_path, map_location='cpu'), strict=True)
    build_fast_multires()


    lora_state_dict = torch.load(causal_dit_lora_path, map_location='cpu')
    pipeline.transformer.load_state_dict(lora_state_dict, strict=False)
    controlnet_state_dict = torch.load(controlnet_path, map_location='cpu')
    pipeline.controlnet.load_state_dict(controlnet_state_dict, strict=True)

    pipeline.transformer.to('cuda', dtype=weight_dtype)
    pipeline.controlnet.to('cuda', dtype=weight_dtype)

    print('loaded {} ckpt'.format(style))

    return style

    
  
load_ckpt()

def reference_kv_cache_bytes(resolution, top_k):
    # The DiT keeps K and V of every reference token for all its blocks. 4*top_k reference
    # patches of (w/2, h/2) -> VAE /8 -> 2x2 patches -> tokens of width heads*head_dim (fp16).
    config = pipeline.transformer.config
    tokens = 4 * top_k * (resolution[0] // 2 // 16) * (resolution[1] // 2 // 16)
    return 2 * config.num_layers * tokens * config.num_attention_heads * config.attention_head_dim * 2

def fix_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def process_multi_images(files):
    images = [Image.open(file.name) for file in files]
    imgs = []
    for i, img in enumerate(images):
        imgs.append(img)
    return imgs 

def extract_lines(image):
    src = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    rows = int(np.ceil(src.shape[0] / 16)) * 16
    cols = int(np.ceil(src.shape[1] / 16)) * 16

    patch = np.ones((1, 1, rows, cols), dtype="float32")
    patch[0, 0, 0:src.shape[0], 0:src.shape[1]] = src

    tensor = torch.from_numpy(patch).cuda()

    with torch.no_grad():
        y = line_model(tensor)

    yc = y.cpu().numpy()[0, 0, :, :]
    yc[yc > 255] = 255
    yc[yc < 0] = 0

    outimg = yc[0:src.shape[0], 0:src.shape[1]]
    outimg = outimg.astype(np.uint8)
    outimg = Image.fromarray(outimg)
    torch.cuda.empty_cache()
    return outimg

def extract_line_image(query_image_, resolution):
    tar_width, tar_height = resolution
    query_image = query_image_.resize((tar_width, tar_height))
    query_image = query_image.convert('L').convert('RGB')
    extracted_line = extract_lines(query_image)
    extracted_line = extracted_line.convert('L').convert('RGB')
    torch.cuda.empty_cache()
    return extracted_line, Image.new('RGB', (tar_width, tar_height), 'black')

def extract_sketch_line_image(query_image_, input_style):
    global cur_style
    if input_style != cur_style:
        change_ckpt(input_style)

    resolution = get_rate(query_image_)
    extracted_line, hint_mask = extract_line_image(query_image_, resolution)
    extracted_sketch = extracted_line
    extracted_sketch_line = Image.blend(extracted_sketch, extracted_line, 0.5)

    extracted_sketch_line_ori = copy.deepcopy(extracted_sketch_line)

    extracted_sketch_line_np = np.array(extracted_sketch_line)
    extracted_sketch_line = Image.fromarray(np.uint8(extracted_sketch_line_np))
    if input_style == 'line + shadow':
        print('line + shadow sketch')
        black_rate = 74
        black_value = 18
        gary_rate = 155
        up_bound = 145
        ori_np = np.array(extracted_sketch_line_ori)
        query_image_np = np.array(query_image_.resize(resolution).convert('L').convert('RGB'))
        extracted_sketch_line_np = np.array(extracted_sketch_line.convert('L').convert('RGB'))
        ori_np[query_image_np <= black_rate] = black_value
        ori_np[(ori_np > gary_rate) & (query_image_np < up_bound) & (query_image_np > black_rate)] = gary_rate
        extracted_sketch_line_ori = Image.fromarray(np.uint8(ori_np))

        extracted_sketch_line_np[query_image_np <= black_rate] = black_value
        extracted_sketch_line_np[(extracted_sketch_line_np > gary_rate) & (query_image_np < up_bound) & (query_image_np > black_rate)] = gary_rate
        extracted_sketch_line = Image.fromarray(np.uint8(extracted_sketch_line_np))

    return extracted_sketch_line.convert('RGB'), extracted_sketch_line.convert('RGB'), hint_mask, query_image_, extracted_sketch_line_ori.convert('RGB'), resolution

def colorize_image(extracted_line, reference_images, resolution, seed, num_inference_steps, top_k, hint_mask=None, hint_color=None, query_image_origin=None, extracted_image_ori=None, enable_postprocess=ENABLE_POSTPROCESS, postprocess_scale=POSTPROCESS_SCALE):
    if extracted_line is None:
        gr.Info("Please preprocess the image first")
        raise ValueError("Please preprocess the image first")
    timer = StageTimer(enabled=True, sync=True) if BENCHMARK_MODE else NULL_TIMER
    output_gallery = run_colorization(extracted_line, reference_images, resolution, seed, num_inference_steps, top_k,
                                      hint_mask, hint_color, query_image_origin, extracted_image_ori,
                                      enable_postprocess, postprocess_scale, timer=timer, notify=gr.Info)
    if BENCHMARK_MODE:
        for name, r in timer.results().items():
            print(f"[benchmark] {name:24s} wall {r['wall_ms']:8.1f} ms  gpu {r['gpu_ms']:8.1f} ms  peak {r['peak_alloc_mb']:7.0f} MB")
        peak_alloc, peak_reserved = timer.overall_peaks()
        print(f"[benchmark] peak allocated {peak_alloc / 2**20:.0f} MB, peak reserved {peak_reserved / 2**20:.0f} MB")
    return output_gallery


def run_colorization(extracted_line, reference_images, resolution, seed, num_inference_steps, top_k, hint_mask, hint_color,
                     query_image_origin, extracted_image_ori, enable_postprocess=ENABLE_POSTPROCESS,
                     postprocess_scale=POSTPROCESS_SCALE, timer=NULL_TIMER, notify=lambda msg: None):
    global pipeline
    global FastMultiResNetModel
    reference_images = process_multi_images(reference_images)
    fix_random_seeds(seed)

    tar_width, tar_height = resolution

    notify("Image retrieval in progress...")

    query_image_bw = extracted_line.resize((tar_width, tar_height))
    query_image = query_image_bw.convert('RGB')

    query_image_origin = query_image_origin.resize((tar_width, tar_height))

    with timer("retrieval.prep"):
        reference_images = [process_image(ref_image, tar_width, tar_height) for ref_image in reference_images]
        query_patches_pil = process_image_Q_varres(query_image_origin, tar_width, tar_height)
        reference_patches_pil = []

        for reference_image in reference_images:
            reference_patches_pil += process_image_ref_varres(reference_image, tar_width, tar_height)
    with torch.inference_mode():
        with timer("retrieval.clip"):
            if image_encoder.device.type != 'cuda':
                image_encoder.to('cuda')
            clip_img = image_processor(images=query_patches_pil, return_tensors="pt").pixel_values.to(image_encoder.device, dtype=image_encoder.dtype)
            query_embeddings = image_encoder(clip_img).image_embeds
            reference_patches_pil_gray = [rimg.convert('RGB').convert('RGB') for rimg in reference_patches_pil]
            clip_img = image_processor(images=reference_patches_pil_gray, return_tensors="pt").pixel_values.to(image_encoder.device, dtype=image_encoder.dtype)
            reference_embeddings = image_encoder(clip_img).image_embeds
        with timer("retrieval.select"):
            cosine_similarities = F.cosine_similarity(query_embeddings.unsqueeze(1), reference_embeddings.unsqueeze(0), dim=-1)
            # Only the first top_k entries of the official full argsort are used. Like the
            # official code, top_k larger than the number of reference patches raises.
            top_k_indices = torch.topk(cosine_similarities, top_k, dim=1).indices.tolist()
            available_ref_patches = [[],[],[],[]]
            for i in range(len(top_k_indices)):
                for j in range(top_k):
                    available_ref_patches[i].append(reference_patches_pil[top_k_indices[i][j]].resize((tar_width//2, tar_height//2)).convert('RGB'))

            flat_available_ref_patches = [item for sublist in available_ref_patches for item in sublist]

    grid_N = int(np.ceil(np.sqrt(len(flat_available_ref_patches))))
    small_tar_width = tar_width//grid_N
    small_tar_height = tar_height//grid_N
    grid_img = Image.new('RGB', (grid_N*small_tar_width, grid_N*small_tar_height), 'black')
    for i in range(len(flat_available_ref_patches)):
        grid_img.paste(flat_available_ref_patches[i].resize((small_tar_width, small_tar_height)), (i%grid_N*small_tar_width, int(i/grid_N)*small_tar_height))

    draw = ImageDraw.Draw(grid_img)
    draw.text((0, 0), "Reference Images", fill='red', font_size=50)

    notify("Model inference in progress...")
    hint_mask = hint_mask.resize((tar_width//8, tar_height//8)).convert('RGB')
    hint_color = hint_color.convert('RGB')

    def diffusion():
        generator = torch.Generator(device='cuda').manual_seed(seed)
        return pipeline(
                cond_input=query_image_bw.convert('RGB'),
                cond_refs=available_ref_patches,
                hint_mask=hint_mask,
                hint_color=hint_color,
                num_inference_steps=num_inference_steps,
                generator = generator,
            )[0][0]

    with timer("diffusion"):
        if RELEASE_CACHE_THRESHOLD_GB is not None:
            release_unused_cache(RELEASE_CACHE_THRESHOLD_GB)
        # Only under memory pressure (e.g. Top K 20), ComfyUI-style: the retrieval encoder waits on
        # the CPU during diffusion (it returns for the next retrieval), and the 4*Top K reference
        # patches are VAE-encoded one at a time instead of as one batch (Top K 20: a single 3.08 GiB
        # activation). Both stay off otherwise: slicing changes cuDNN's rounding of the reference
        # latents, so the official output is kept unless memory requires it.
        low_vram = reference_kv_cache_bytes(resolution, top_k) + DIFFUSION_WORKSPACE_GB * 2**30 > available_vram()
        if low_vram:
            print("[vram] low VRAM for diffusion: retrieval encoder offloaded, reference VAE encode sliced")
            image_encoder.to('cpu')
            pipeline.vae.enable_slicing()
        else:
            pipeline.vae.disable_slicing()
        colorized_image = diffusion()
    if enable_postprocess:
        notify("Post-processing image...")
        if RELEASE_CACHE_THRESHOLD_GB is not None:
            with timer("release_cache"):
                release_unused_cache(RELEASE_CACHE_THRESHOLD_GB)
        high_res_image = postprocess_colorized_image(
            colorized_image, extracted_image_ori, resolution, pipeline.vae, FastMultiResNetModel,
            scale=postprocess_scale, timer=timer)
    else:
        high_res_image = colorized_image
    notify("Colorization complete!")

    output_gallery = [high_res_image, query_image_bw, hint_mask, hint_color, grid_img]
    return output_gallery


# Function to get color value from reference image
def get_color_value(reference_image, evt: gr.SelectData):
    if reference_image is None:
        return "Please upload a reference image first."
    x, y = evt.index
    color_value = reference_image[y, x]
    return f"Get Color value: {color_value}", color_value

# Function to draw a square on the line drawing image
def draw_square(line_drawing_image_pil, hint_mask, color_value, evt: gr.SelectData):
    line_drawing_image = np.array(line_drawing_image_pil)
    hint_mask = np.array(hint_mask)
    if line_drawing_image is None:
        return "Please upload a line drawing image first."
    if color_value is None:
        return "Please pick a color from the reference image first."
    x, y = evt.index
    # Calculate square boundaries
    start_x = max(0, x - 8)
    start_y = max(0, y - 8)
    end_x = min(line_drawing_image.shape[1], x + 8)
    end_y = min(line_drawing_image.shape[0], y + 8)
    # Draw the square
    line_drawing_image[start_y:end_y, start_x:end_x] = color_value
    line_drawing_image_pil = Image.fromarray(np.uint8(line_drawing_image))
    hint_mask[start_y:end_y, start_x:end_x] = 255
    hint_mask_pil = Image.fromarray(np.uint8(hint_mask))
    return line_drawing_image_pil, hint_mask_pil


with gr.Blocks() as demo:
    gr.HTML(
    """
<div style="text-align: center;">
    <h1 style="text-align: center; font-size: 3em;">🎨 Cobra:</h1>
    <h3 style="text-align: center; font-size: 1.8em;">Efficient Line Art COlorization with BRoAder References</h3>
    <p style="text-align: center; font-weight: bold;">
        <a href="https://zhuang2002.github.io/Cobra/">Project Page</a> | 
        <a href="https://arxiv.org">ArXiv Preprint</a> | 
        <a href="https://github.com/zhuang2002/Cobra">GitHub Repository</a>
    </p>
    <p style="text-align: center; font-weight: bold;">
        NOTE: Each time you switch the input style, the corresponding model will be reloaded, which may take some time. Please be patient.
    </p>
    <p style="text-align: left; font-size: 1.1em;">
        Welcome to the demo of <strong>Cobra</strong>. Follow the steps below to explore the capabilities of our model:
    </p>
</div>
<div style="text-align: left; margin: 0 auto;">
    <ol style="font-size: 1.1em;">
        <li>Choose your input style: either line + shadow or line only.</li>
        <li>Upload your image: Click the 'Upload' button to select the image you want to colorize.</li>
        <li>Preprocess the image: Click the 'Preprocess' button to extract the line art from your image.</li>
        <li>(Optional) Obtain color values and add color hints: Upload an image to the left area and click to get color values; then, add color hints to the line art on the right.</li>
        <li>Upload reference images: Upload several reference images to help guide the colorization process.</li>
        <li>(Optional) Set inference parameters: Adjust the inference settings as needed.</li>
        <li>Run: Click the <b>Colorize</b> button to start the process.</li>
    </ol>
    <p>
        ⏱️ <b>ZeroGPU Time Limit</b>: Hugging Face ZeroGPU has an inference time limit of 180 seconds. You may need to log in with a free account to use this demo. Large sampling steps might lead to timeout (GPU Abort). In that case, please consider logging in with a Pro account or running it on your local machine.
    </p>
</div>
<div style="text-align: center;">
    <p style="text-align: center; font-weight: bold;">
        注意：每次切换输入样式时，相应的模型将被重新加载，可能需要一些时间。请耐心等待。
    </p>
    <p style="text-align: left; font-size: 1.1em;">
        欢迎使用 <strong>Cobra</strong> 演示。请按照以下步骤探索我们模型的能力：
    </p>
</div>
<div style="text-align: left; margin: 0 auto;">
    <ol style="font-size: 1.1em;">
        <li>选择输入样式：线条+阴影或仅线条。</li>
        <li>上传您的图像：点击“上传”按钮选择您想要上色的图像。</li>
        <li>预处理图像：点击“预处理”按钮从您的图像中提取线稿。</li>
        <li>（可选）获取颜色值并添加颜色提示：上传一张图像到左侧区域，点击获取颜色值；然后，为右侧的线稿添加颜色提示。</li>
        <li>上传参考图像：上传多个参考图像以帮助引导上色过程。</li>
        <li>（可选）设置推理参数：根据需要调整推理设置。</li>
        <li>运行：点击 <b>上色</b> 按钮开始处理。</li>
    </ol>
    <p>
        ⏱️ <b>ZeroGPU时间限制</b>：Hugging Face ZeroGPU 的推理时间限制为 180 秒。您可能需要使用免费帐户登录以使用此演示。大采样步骤可能会导致超时（GPU 中止）。在这种情况下，请考虑使用专业帐户登录或在本地计算机上运行。
    </p>
</div>
    """
)

    hint_mask = gr.State()
    hint_color = gr.State()
    query_image_origin = gr.State()
    resolution = gr.State()
    extracted_image_ori = gr.State()
    style = gr.State()
    
    with gr.Column():
        gr.Markdown("<h2 style='text-align: center;'>Load Model</h2>")
        with gr.Row():
            model_name = gr.Textbox(label="Model Name", value=None)
            with gr.Column():
                style = gr.Dropdown(label="Model List", choices=["line + shadow","line"], value="line + shadow")
                change_ckpt_button = gr.Button("Load Model")
                change_ckpt_button.click(change_ckpt, inputs=[style], outputs=[model_name])

        gr.Markdown("<h2 style='text-align: center;'>Line Drawing Extraction</h2>")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Image to Colorize")
                extract_button = gr.Button("Preprocess (Decolorize)")
            extracted_image = gr.Image(type="pil", label="Decolorized Result")

        
        gr.Markdown("<h2 style='text-align: center;'>Color Selection 🎨 (Left) and Hint Placement 💡 (Right) - Click with Mouse 🖱️</h2>")

        with gr.Row():
            with gr.Column():
                get_color_img = gr.Image(label="Upload an image to extract colors", type="numpy")
                color_value_output = gr.Textbox(label="Color Value")
                color_value_state = gr.State()
                get_color_img.select(
                    get_color_value,
                    [get_color_img],
                    [color_value_output, color_value_state]
                )
            with gr.Column():
                hint_color = gr.Image(label="Line Drawing Image", type="pil")
                hint_color.select(
                    draw_square,
                    [hint_color, hint_mask, color_value_state],
                    [hint_color, hint_mask]
                )

        gr.Markdown("<h2 style='text-align: center;'>Retrieval and Colorization</h2>")
        with gr.Row():
            reference_images = gr.Files(label="Reference Images (Upload multiple)", file_count="multiple")
            with gr.Column():
                output_gallery = gr.Gallery(label="Colorization Results", type="pil")
                seed = gr.Slider(label="Random Seed", minimum=0, maximum=100000, value=0, step=1)
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=10, step=1)
                colorize_button = gr.Button("Colorize")
                top_k = gr.Slider(label="Top K (Total Reference Images: 4K) ", minimum=1, maximum=50, value=3, step=1)
                with gr.Row():
                    enable_postprocess = gr.Checkbox(label="Enable Post Processing (high-res reconstruction)", value=ENABLE_POSTPROCESS)
                    postprocess_scale = gr.Slider(label="Postprocess Scale (official: 1.5)", minimum=0.75, maximum=1.5, value=POSTPROCESS_SCALE, step=0.125)


    extract_button.click(
        extract_sketch_line_image, 
        inputs=[input_image, model_name], 
        outputs=[extracted_image, 
                    hint_color, 
                    hint_mask, 
                    query_image_origin, 
                    extracted_image_ori,
                    resolution
                    ]
    )
    colorize_button.click(
        colorize_image, 
        inputs=[extracted_image, reference_images, resolution, seed, num_inference_steps, top_k, hint_mask, hint_color, query_image_origin, extracted_image_ori, enable_postprocess, postprocess_scale],
        outputs=output_gallery
    )
    with gr.Column():
        gr.Markdown("### Quick Examples")
        gr.Examples(
            examples=examples,
            inputs=[input_image, reference_images, model_name, seed, num_inference_steps, top_k],
            label="Examples",
            examples_per_page=8,
        )


if __name__ == "__main__":
    demo.launch()