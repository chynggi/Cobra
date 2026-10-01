"""Official app.py `colorize_image` logic, copied verbatim with StageTimer probes inserted.

Only `with timer(...)` blocks were added; every statement inside is the official code so
the profiled numbers describe the official implementation. Model objects come from the
imported official `app` module.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from cobra_utils.utils import process_image, process_image_Q_varres, process_image_ref_varres


def colorize_image_profiled(app, timer, extracted_line, reference_images, resolution, seed, num_inference_steps, top_k,
                            hint_mask=None, hint_color=None, query_image_origin=None, extracted_image_ori=None,
                            postprocess_scale=1.5, run_postprocess=True):
    pipeline, MultiResNetModel = app.pipeline, app.MultiResNetModel
    image_processor, image_encoder = app.image_processor, app.image_encoder
    transform, weight_dtype = app.transform, app.weight_dtype

    with timer("retrieval.prep"):
        reference_images = app.process_multi_images(reference_images)
        app.fix_random_seeds(seed)
        tar_width, tar_height = resolution
        query_image_bw = extracted_line.resize((tar_width, tar_height))
        query_image = query_image_bw.convert('RGB')
        query_image_origin = query_image_origin.resize((tar_width, tar_height))
        # official: int(tar_width*1.5), int(tar_height*1.5)
        query_image_vae = extracted_image_ori.resize((int(tar_width*postprocess_scale), int(tar_height*postprocess_scale)))
        reference_images = [process_image(ref_image, tar_width, tar_height) for ref_image in reference_images]
        query_patches_pil = process_image_Q_varres(query_image_origin, tar_width, tar_height)
        reference_patches_pil = []
        for reference_image in reference_images:
            reference_patches_pil += process_image_ref_varres(reference_image, tar_width, tar_height)
    with torch.no_grad():
        with timer("retrieval.clip_query"):
            clip_img = image_processor(images=query_patches_pil, return_tensors="pt").pixel_values.to(image_encoder.device, dtype=image_encoder.dtype)
            query_embeddings = image_encoder(clip_img).image_embeds
        with timer("retrieval.clip_refs_preproc"):
            reference_patches_pil_gray = [rimg.convert('RGB').convert('RGB') for rimg in reference_patches_pil]
            clip_img = image_processor(images=reference_patches_pil_gray, return_tensors="pt").pixel_values.to(image_encoder.device, dtype=image_encoder.dtype)
        with timer("retrieval.clip_refs_encode"):
            reference_embeddings = image_encoder(clip_img).image_embeds
        with timer("retrieval.similarity_sort"):
            cosine_similarities = F.cosine_similarity(query_embeddings.unsqueeze(1), reference_embeddings.unsqueeze(0), dim=-1)
            len_ref = len(reference_patches_pil)
            sorted_indices = torch.argsort(cosine_similarities, descending=True, dim=1).tolist()
        with timer("retrieval.select_patches"):
            top_k_indices = [cur_sortlist[:top_k] for cur_sortlist in sorted_indices]
            available_ref_patches = [[],[],[],[]]
            for i in range(len(top_k_indices)):
                for j in range(top_k):
                    available_ref_patches[i].append(reference_patches_pil[top_k_indices[i][j]].resize((tar_width//2, tar_height//2)).convert('RGB'))
            flat_available_ref_patches = [item for sublist in available_ref_patches for item in sublist]

    with timer("retrieval.grid"):
        grid_N = int(np.ceil(np.sqrt(len(flat_available_ref_patches))))
        small_tar_width = tar_width//grid_N
        small_tar_height = tar_height//grid_N
        grid_img = Image.new('RGB', (grid_N*small_tar_width, grid_N*small_tar_height), 'black')
        for i in range(len(flat_available_ref_patches)):
            grid_img.paste(flat_available_ref_patches[i].resize((small_tar_width, small_tar_height)), (i%grid_N*small_tar_width, int(i/grid_N)*small_tar_height))
        draw = ImageDraw.Draw(grid_img)
        draw.text((0, 0), "Reference Images", fill='red', font_size=50)

    with timer("diffusion"):
        generator = torch.Generator(device='cuda').manual_seed(seed)
        hint_mask = hint_mask.resize((tar_width//8, tar_height//8)).convert('RGB')
        hint_color = hint_color.convert('RGB')
        colorized_image = pipeline(
                cond_input=query_image_bw.convert('RGB'),
                cond_refs=available_ref_patches,
                hint_mask=hint_mask,
                hint_color=hint_color,
                num_inference_steps=num_inference_steps,
                generator = generator,
            )[0][0]
    if not run_postprocess:
        return colorized_image, colorized_image, top_k_indices
    high_res_image = official_postprocess(app, timer, colorized_image, query_image_vae)
    with timer("empty_cache"):
        torch.cuda.empty_cache()
    return high_res_image, colorized_image, top_k_indices


def official_postprocess(app, timer, colorized_image, query_image_vae):
    """Official post-processing block (verbatim) with stage probes."""
    pipeline, MultiResNetModel = app.pipeline, app.MultiResNetModel
    transform, weight_dtype = app.transform, app.weight_dtype
    with torch.no_grad():
        with timer("A.resize"):
            up_img = colorized_image.resize(query_image_vae.size)
        with timer("B.test_low_color"):
            test_low_color = transform(up_img).unsqueeze(0).to('cuda', dtype=weight_dtype)
        with timer("C.query_image_vae_"):
            query_image_vae_ = transform(query_image_vae).unsqueeze(0).to('cuda', dtype=weight_dtype)
        with timer("D.vae_encode_color"):
            h_color, hidden_list_color = pipeline.vae._encode(test_low_color,return_dict = False, hidden_flag = True)
        with timer("E.vae_encode_bw"):
            h_bw, hidden_list_bw = pipeline.vae._encode(query_image_vae_, return_dict = False, hidden_flag = True)
        with timer("F.concat"):
            hidden_list_double = [torch.cat((hidden_list_color[hidden_idx], hidden_list_bw[hidden_idx]), dim = 1) for hidden_idx in range(len(hidden_list_color))]
        with timer("G.multires"):
            hidden_list = MultiResNetModel(hidden_list_double)
        with timer("H.vae_decode"):
            output = pipeline.vae._decode(h_color.sample(),return_dict = False, hidden_list = hidden_list)[0]
        with timer("I.to_pil"):
            output[output > 1] = 1
            output[output < -1] = -1
            high_res_image = Image.fromarray(((output[0] * 0.5 + 0.5).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
    return high_res_image
