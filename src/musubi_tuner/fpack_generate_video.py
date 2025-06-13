import argparse
import copy
import gc
import hashlib
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.profiler
from diskcache import Cache
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from musubi_tuner.dataset import image_video_dataset
from musubi_tuner.frame_pack import hunyuan
from musubi_tuner.frame_pack.clip_vision import hf_clip_vision_encode
from musubi_tuner.frame_pack.framepack_utils import (
    load_image_encoders,
    load_text_encoder1,
    load_text_encoder2,
    load_vae,
)
from musubi_tuner.frame_pack.hunyuan_video_packed import (
    HunyuanVideoTransformer3DModelPacked,
    create_packed_model_empty,
    load_packed_model,
)
from musubi_tuner.frame_pack.k_diffusion_hunyuan import sample_hunyuan
from musubi_tuner.frame_pack.utils import crop_or_pad_yield_mask, soft_append_bcthw
from musubi_tuner.hunyuan_model.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from musubi_tuner.hv_generate_video import (
    save_images_grid,
    save_videos_grid,
    synchronize_device,
)
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.networks import lora_framepack
from musubi_tuner.utils.device_utils import clean_memory_on_device

from musubi_tuner.wan_generate_video import merge_lora_weights

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GenerationSettings:
    def __init__(self, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None):
        self.device = device
        self.dit_weight_dtype = dit_weight_dtype  # not used currently because model may be optimized


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="Wan 2.1 inference script")

    # WAN arguments
    # parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory (Wan 2.1 official).")
    parser.add_argument(
        "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++", "vanilla"], help="The solver used to sample."
    )

    parser.add_argument("--dit", type=str, default=None, help="DiT directory or path")
    parser.add_argument("--vae", type=str, default=None, help="VAE directory or path")
    parser.add_argument("--text_encoder1", type=str, required=True, help="Text Encoder 1 directory or path")
    parser.add_argument("--text_encoder2", type=str, required=True, help="Text Encoder 2 directory or path")
    parser.add_argument("--image_encoder", type=str, required=True, help="Image Encoder directory or path")
    parser.add_argument("--f1", action="store_true", help="Use F1 sampling method")

    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns")
    parser.add_argument(
        "--save_merged_model",
        type=str,
        default=None,
        help="Save merged model to path. If specified, no inference will be performed.",
    )

    # inference
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="prompt for generation. If `;;;` is used, it will be split into sections. Example: `section_index:prompt` or "
        "`section_index:prompt;;;section_index:prompt;;;...`, section_index can be `0` or `-1` or `0-2`, `-1` means last section, `0-2` means from 0 to 2 (inclusive).",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="negative prompt for generation, default is empty string. should not change.",
    )
    parser.add_argument(
        "--custom_system_prompt",
        type=str,
        default=None,
        help="Custom system prompt for LLM. If specified, it will override the default system prompt. See hunyuan_model/text_encoder.py for the default system prompt.",
    )
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256], help="video size, height and width")
    parser.add_argument("--video_seconds", type=float, default=5.0, help="video length, default is 5.0 seconds")
    parser.add_argument(
        "--video_sections",
        type=int,
        default=None,
        help="number of video sections, Default is None (auto calculate from video seconds)",
    )
    parser.add_argument(
        "--one_frame_inference",
        type=str,
        default=None,
        help="one frame inference, default is None, comma separated values from 'no_2x', 'no_4x', 'no_post', 'control_indices' and 'target_index'.",
    )
    parser.add_argument(
        "--control_image_path", type=str, default=None, nargs="*", help="path to control (reference) image for one frame inference."
    )
    parser.add_argument(
        "--control_image_mask_path",
        type=str,
        default=None,
        nargs="*",
        help="path to control (reference) image mask for one frame inference.",
    )
    parser.add_argument("--fps", type=int, default=30, help="video fps, default is 30")
    parser.add_argument("--infer_steps", type=int, default=25, help="number of inference steps, default is 25")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    # parser.add_argument(
    #     "--cpu_noise", action="store_true", help="Use CPU to generate noise (compatible with ComfyUI). Default is False."
    # )
    parser.add_argument("--latent_window_size", type=int, default=9, help="latent window size, default is 9. should not change.")
    parser.add_argument(
        "--embedded_cfg_scale", type=float, default=10.0, help="Embeded CFG scale (distilled CFG Scale), default is 10.0"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for classifier free guidance. Default is 1.0 (no guidance), should not change.",
    )
    parser.add_argument("--guidance_rescale", type=float, default=0.0, help="CFG Re-scale, default is 0.0. Should not change.")
    # parser.add_argument("--video_path", type=str, default=None, help="path to video for video2video inference")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="path to image for image2video inference. If `;;;` is used, it will be used as section images. The notation is same as `--prompt`.",
    )
    parser.add_argument("--end_image_path", type=str, default=None, help="path to end image for image2video inference")
    parser.add_argument(
        "--latent_paddings",
        type=str,
        default=None,
        help="latent paddings for each section, comma separated values. default is None (FramePack default paddings)",
    )
    # parser.add_argument(
    #     "--control_path",
    #     type=str,
    #     default=None,
    #     help="path to control video for inference with controlnet. video file or directory with images",
    # )
    # parser.add_argument("--trim_tail_frames", type=int, default=0, help="trim tail N frames from the video before saving")

    # # Flow Matching
    # parser.add_argument(
    #     "--flow_shift",
    #     type=float,
    #     default=None,
    #     help="Shift factor for flow matching schedulers. Default depends on task.",
    # )

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8")
    # parser.add_argument("--fp8_fast", action="store_true", help="Enable fast FP8 arithmetic (RTX 4XXX+), only for fp8_scaled")
    parser.add_argument(
        "--rope_scaling_factor", type=float, default=0.5, help="RoPE scaling factor for high resolution (H/W), default is 0.5"
    )
    parser.add_argument(
        "--rope_scaling_timestep_threshold",
        type=int,
        default=None,
        help="RoPE scaling timestep threshold, default is None (disable), if set, RoPE scaling will be applied only for timesteps >= threshold, around 800 is good starting point",
    )

    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for Text Encoder 1 (LLM)")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "torch", "sageattn", "xformers", "sdpa"],  #  "flash2", "flash3",
        help="attention mode",
    )
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument(
        "--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256"
    )
    parser.add_argument("--bulk_decode", action="store_true", help="decode all frames at once")
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="number of blocks to swap in the model")
    parser.add_argument(
        "--output_type",
        type=str,
        default="video",
        choices=["video", "images", "latent", "both", "latent_images"],
        help="output type",
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")
    parser.add_argument("--lycoris", action="store_true", help="use lycoris for inference")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument(
        "--compile_args",
        nargs=4,
        metavar=("BACKEND", "MODE", "DYNAMIC", "FULLGRAPH"),
        default=["inductor", "max-autotune-no-cudagraphs", "False", "False"],
        help="Torch.compile settings",
    )

    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for models and data. Default is None (no cache).")
    
    # Profiling arguments
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiling")
    parser.add_argument("--profile_shapes", action="store_true", default=False, help="Record tensor shapes in profiling (default: False)")
    parser.add_argument("--profile_memory", action="store_true", default=True, help="Profile memory usage (default: True)")
    parser.add_argument("--profile_stack", action="store_true", default=True, help="Record stack traces (default: True)")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level (default: INFO)")

    args = parser.parse_args()

    return args


def parse_prompt_line(line: str) -> Dict[str, Any]:
    """Parse a prompt line into a dictionary of argument overrides

    Args:
        line: Prompt line with options

    Returns:
        Dict[str, Any]: Dictionary of argument overrides
    """
    # TODO common function with hv_train_network.line_to_prompt_dict
    parts = line.split(" --")
    prompt = parts[0].strip()

    # Create dictionary of overrides
    overrides = {"prompt": prompt}
    # Initialize control_image_path and control_image_mask_path as a list to accommodate multiple paths
    overrides["control_image_path"] = []
    overrides["control_image_mask_path"] = []

    for part in parts[1:]:
        if not part.strip():
            continue
        option_parts = part.split(" ", 1)
        option = option_parts[0].strip()
        value = option_parts[1].strip() if len(option_parts) > 1 else ""

        # Map options to argument names
        if option == "w":
            overrides["video_size_width"] = int(value)
        elif option == "h":
            overrides["video_size_height"] = int(value)
        elif option == "f":
            overrides["video_seconds"] = float(value)
        elif option == "d":
            overrides["seed"] = int(value)
        elif option == "s":
            overrides["infer_steps"] = int(value)
        elif option == "g" or option == "l":
            overrides["guidance_scale"] = float(value)
        # elif option == "fs":
        #     overrides["flow_shift"] = float(value)
        elif option == "i":
            overrides["image_path"] = value
        # elif option == "im":
        #     overrides["image_mask_path"] = value
        # elif option == "cn":
        #     overrides["control_path"] = value
        elif option == "n":
            overrides["negative_prompt"] = value
        elif option == "vs":  # video_sections
            overrides["video_sections"] = int(value)
        elif option == "ei":  # end_image_path
            overrides["end_image_path"] = value
        elif option == "ci":  # control_image_path
            overrides["control_image_path"].append(value)
        elif option == "cim":  # control_image_mask_path
            overrides["control_image_mask_path"].append(value)
        elif option == "of":  # one_frame_inference
            overrides["one_frame_inference"] = value

    # If no control_image_path was provided, remove the empty list
    if not overrides["control_image_path"]:
        del overrides["control_image_path"]
    if not overrides["control_image_mask_path"]:
        del overrides["control_image_mask_path"]

    return overrides


def apply_overrides(args: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    """Apply overrides to args

    Args:
        args: Original arguments
        overrides: Dictionary of overrides

    Returns:
        argparse.Namespace: New arguments with overrides applied
    """
    args_copy = copy.deepcopy(args)

    for key, value in overrides.items():
        if key == "video_size_width":
            args_copy.video_size[1] = value
        elif key == "video_size_height":
            args_copy.video_size[0] = value
        else:
            setattr(args_copy, key, value)

    return args_copy


def check_inputs(args: argparse.Namespace) -> Tuple[int, int, int]:
    """Validate video size and length

    Args:
        args: command line arguments

    Returns:
        Tuple[int, int, float]: (height, width, video_seconds)
    """
    height = args.video_size[0]
    width = args.video_size[1]

    video_seconds = args.video_seconds
    if args.video_sections is not None:
        video_seconds = (args.video_sections * (args.latent_window_size * 4) + 1) / args.fps

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    return height, width, video_seconds


# region DiT model


def generate_optimized_model_filename(dit_path: str, lora_weight: list[str], lora_multiplier: list[float], fp8: bool, fp8_scaled: bool) -> str:
    key_parts = [
        f"dit:{os.path.basename(dit_path)}",
    ]    
    if lora_weight:
        for i, (lora_path, multiplier) in enumerate(zip(lora_weight, lora_multiplier)):
            key_parts.append(f"lora{i}:{os.path.basename(lora_path)}@{multiplier}")
    
    key_string = "_".join(key_parts)
    hash_digest = hashlib.sha256(key_string.encode()).hexdigest()[:8]
    
    base_name = os.path.splitext(os.path.basename(dit_path))[0]
    fp8_suffix = "_fp8s" if fp8_scaled else "_fp8" if fp8 else ""
    
    return f"{base_name}_{hash_digest}{fp8_suffix}.safetensors"


def load_dit_model(blocks_to_swap: int, fp8_scaled: bool, lora_weight: list[str], dit_path: str, attn_mode: str, rope_scaling_timestep_threshold: int, rope_scaling_factor: float, device: torch.device) -> HunyuanVideoTransformer3DModelPacked:
    """load DiT model

    Args:
        args: command line arguments
        device: device to use
        dit_dtype: data type for the model
        dit_weight_dtype: data type for the model weights. None for as-is

    Returns:
        HunyuanVideoTransformer3DModelPacked: DiT model
    """
    logger.debug(f"{blocks_to_swap=}, {fp8_scaled=}, {lora_weight=}, {dit_path=}, {attn_mode=}, {rope_scaling_timestep_threshold=}, {rope_scaling_factor=}, {device=}")
    loading_device = "cpu"
    if blocks_to_swap == 0 and not fp8_scaled and lora_weight is None:
        loading_device = device

    # do not fp8 optimize because we will merge LoRA weights
    model = load_packed_model(device, dit_path, attn_mode, loading_device)

    # apply RoPE scaling factor
    if rope_scaling_timestep_threshold is not None:
        logger.info(
            f"Applying RoPE scaling factor {rope_scaling_factor} for timesteps >= {rope_scaling_timestep_threshold}"
        )
        model.enable_rope_scaling(rope_scaling_timestep_threshold, rope_scaling_factor)
    return model


def optimize_model(
    model: HunyuanVideoTransformer3DModelPacked, 
    fp8_scaled: bool,
    blocks_to_swap: int,
    fp8: bool,
    device: torch.device
) -> None:
    """optimize the model (FP8 conversion, device move etc.)

    Args:
        model: dit model
        fp8_scaled: whether to use fp8 scaled optimization
        blocks_to_swap: number of blocks to swap
        fp8: whether to use fp8
        device: device to use
    """
    logger.debug(f"{fp8_scaled=}, {blocks_to_swap=}, {fp8=}, {device=}")
    if fp8_scaled:
        # load state dict as-is and optimize to fp8
        state_dict = model.state_dict()

        # if no blocks to swap, we can move the weights to GPU after optimization on GPU (omit redundant CPU->GPU copy)
        move_to_device = blocks_to_swap == 0  # if blocks_to_swap > 0, we will keep the model on CPU
        state_dict = model.fp8_optimization(state_dict, device, move_to_device, use_scaled_mm=False)  # args.fp8_fast)

        info = model.load_state_dict(state_dict, strict=True, assign=True)
        logger.info(f"Loaded FP8 optimized weights: {info}")

        if blocks_to_swap == 0:
            model.to(device)  # make sure all parameters are on the right device (e.g. RoPE etc.)
    else:
        # simple cast to dit_dtype
        target_dtype = None  # load as-is (dit_weight_dtype == dtype of the weights in state_dict)
        target_device = None

        if fp8:
            target_dtype = torch.float8_e4m3fn

        if blocks_to_swap == 0:
            logger.info(f"Move model to device: {device}")
            target_device = device

        if target_device is not None and target_dtype is not None:
            model.to(target_device, target_dtype)  # move and cast  at the same time. this reduces redundant copy operations

    # if args.compile:
    #     compile_backend, compile_mode, compile_dynamic, compile_fullgraph = args.compile_args
    #     logger.info(
    #         f"Torch Compiling[Backend: {compile_backend}; Mode: {compile_mode}; Dynamic: {compile_dynamic}; Fullgraph: {compile_fullgraph}]"
    #     )
    #     torch._dynamo.config.cache_size_limit = 32
    #     for i in range(len(model.blocks)):
    #         model.blocks[i] = torch.compile(
    #             model.blocks[i],
    #             backend=compile_backend,
    #             mode=compile_mode,
    #             dynamic=compile_dynamic.lower() in "true",
    #             fullgraph=compile_fullgraph.lower() in "true",
    #         )

    if blocks_to_swap > 0:
        logger.info(f"Enable swap {blocks_to_swap} blocks to CPU from device: {device}")
        model.enable_block_swap(blocks_to_swap, device, supports_backward=False)
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()
    else:
        # make sure the model is on the right device
        model.to(device)

    model.eval().requires_grad_(False)
    clean_memory_on_device(device)


def load_optimized_model(
    dit_path: str, 
    fp8_scaled: bool, 
    lora_weight: list[str], 
    lora_multiplier: list[float], 
    fp8: bool, 
    blocks_to_swap: int, 
    attn_mode: str, 
    rope_scaling_timestep_threshold: int, 
    rope_scaling_factor: float, 
    cache_dir: str | None,
    device: torch.device,
    include_patterns: Optional[list[str]],
    exclude_patterns: Optional[list[str]],
    lycoris: bool,
    save_merged_model: Optional[str],
    log_timing: Optional[Callable] = None
) -> Optional[HunyuanVideoTransformer3DModelPacked]:
    """load_optimized_dit_model_with_lora
    
    モデルのロード先は次のとおり。なお、実装では`block_to_swap`も考慮する。
    1. 最適化済みのモデルが存在する場合、GPUにロードする
    2. モデルがLoRAの適用やfp8量子化などを行わない場合、はじめからGPUにロードする
    3. モデルがLoRAの適用を行う場合、いったんCPUにロードしてLoRAのmergeを行う。その後、fp8最適化時にGPUにロードする。

    NOTE:
    - fp8_scaled をディスクにキャッシュすることはいったん諦めた。難しい。
    - LoRAのmerge時点でGPUに送っているようにも見えるのだが、ちょっとよく分からなかった...
    """
    if log_timing is None:
        log_timing = print

    cache_model_dir = os.path.join(cache_dir, "models") if cache_dir is not None else None

    cache_model_path = None
    if cache_model_dir is not None:
        filename = generate_optimized_model_filename(dit_path, lora_weight, lora_multiplier, fp8, fp8_scaled)
        cache_model_path = os.path.join(cache_model_dir, filename)
    
    if cache_model_path is not None and os.path.exists(cache_model_path):
        log_timing(f"load_optimized_model: {cache_model_path}")
        logger.info(f"Load optimized model from disk: {cache_model_path}")
        model = create_packed_model_empty(attn_mode)
        # ここで先に model.to_empty(device=device) すると、推論が動かなくなる。
        log_timing(f"load_file: {cache_model_path}")
        state_dict = load_file(cache_model_path, device=str(device))
        log_timing(f"apply_fp8_monkey_patch: {cache_model_path}")
        if fp8_scaled:
            apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=False)
        log_timing(f"load_state_dict: {cache_model_path}")
        model.load_state_dict(state_dict, strict=True, assign=True)        
        log_timing(f"to: {cache_model_path}")
        model.to(device)
        return model

    model = load_dit_model(blocks_to_swap, fp8_scaled, lora_weight, dit_path, attn_mode, rope_scaling_timestep_threshold, rope_scaling_factor, device)

    # merge LoRA weights
    if lora_weight is not None and len(lora_weight) > 0:
        # ugly hack to common merge_lora_weights function
        merge_lora_weights(
            lora_framepack,
            model,
            device,
            lora_weight,
            lora_multiplier,
            include_patterns,
            exclude_patterns,
            lycoris,
            save_merged_model,
            None  # converter
        )

        # if we only want to save the model, we can skip the rest
        if save_merged_model:
            return None

    # optimize model: fp8 conversion, block swap etc.
    optimize_model(model, fp8_scaled, blocks_to_swap, fp8, device)

    if cache_model_path is not None:
        logger.info(f"Saving optimized model to disk: {cache_model_path}")
        os.makedirs(os.path.dirname(cache_model_path), exist_ok=True)
        save_file(model.state_dict(), cache_model_path)
    return model


# endregion


def decode_latent(
    latent_window_size: int,
    total_latent_sections: int,
    bulk_decode: bool,
    vae: AutoencoderKLCausal3D,
    latent: torch.Tensor,
    device: torch.device,
    one_frame_inference_mode: bool = False,
) -> torch.Tensor:
    logger.info(f"Decoding video...")
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)  # add batch dimension

    vae.to(device)
    if not bulk_decode and not one_frame_inference_mode:
        latent_window_size = latent_window_size  # default is 9
        # total_latent_sections = (args.video_seconds * 30) / (latent_window_size * 4)
        # total_latent_sections = int(max(round(total_latent_sections), 1))
        num_frames = latent_window_size * 4 - 3

        latents_to_decode = []
        latent_frame_index = 0
        for i in range(total_latent_sections - 1, -1, -1):
            is_last_section = i == total_latent_sections - 1
            generated_latent_frames = (num_frames + 3) // 4 + (1 if is_last_section else 0)
            section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)

            section_latent = latent[:, :, latent_frame_index : latent_frame_index + section_latent_frames, :, :]
            if section_latent.shape[2] > 0:
                latents_to_decode.append(section_latent)

            latent_frame_index += generated_latent_frames

        latents_to_decode = latents_to_decode[::-1]  # reverse the order of latents to decode

        history_pixels = None
        for latent in tqdm(latents_to_decode):
            if history_pixels is None:
                history_pixels = hunyuan.vae_decode(latent, vae).cpu()
            else:
                overlapped_frames = latent_window_size * 4 - 3
                current_pixels = hunyuan.vae_decode(latent, vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
            clean_memory_on_device(device)
    else:
        # bulk decode
        logger.info(f"Bulk decoding or one frame inference")
        if not one_frame_inference_mode:
            history_pixels = hunyuan.vae_decode(latent, vae).cpu()  # normal
        else:
            # one frame inference
            history_pixels = [hunyuan.vae_decode(latent[:, :, i : i + 1, :, :], vae).cpu() for i in range(latent.shape[2])]
            history_pixels = torch.cat(history_pixels, dim=2)

    vae.to("cpu")

    logger.info(f"Decoded. Pixel shape {history_pixels.shape}")
    return history_pixels[0]  # remove batch dimension


def log_data_types(data: Any, name: str = "", prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> None:
    if current_depth > max_depth:
        logger.debug(f"{prefix}{name}: <max_depth_reached>")
        return
    
    if isinstance(data, dict):
        if current_depth == 0:
            logger.debug(f"=== Debug: {name} data types ===")
        
        for key, value in data.items():
            new_name = f"{name}[{key}]" if name else str(key)
            log_data_types(value, new_name, prefix, max_depth, current_depth + 1)
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            new_name = f"{name}[{i}]" if name else str(i)
            log_data_types(item, new_name, prefix, max_depth, current_depth + 1)
    else:
        type_info = f"{type(data)}"
        if hasattr(data, 'dtype'):
            type_info += f" with dtype {data.dtype}"
        if hasattr(data, 'shape'):
            type_info += f" shape {data.shape}"
        logger.debug(f"{prefix}{name}: {type_info}")


def parse_section_strings(input_string: str) -> dict[int, str]:
    "define parsing function"
    section_strings = {}
    if ";;;" in input_string:
        split_section_strings = input_string.split(";;;")
        for section_str in split_section_strings:
            if ":" not in section_str:
                start = end = 0
                section_str = section_str.strip()
            else:
                index_str, section_str = section_str.split(":", 1)
                index_str = index_str.strip()
                section_str = section_str.strip()

                m = re.match(r"^(-?\d+)(-\d+)?$", index_str)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2)[1:]) if m.group(2) is not None else start
                else:
                    start = end = 0
                    section_str = section_str.strip()
            for i in range(start, end + 1):
                section_strings[i] = section_str
    else:
        section_strings[0] = input_string

    # assert 0 in section_prompts, "Section prompts must contain section 0"
    if 0 not in section_strings:
        # use smallest section index. prefer positive index over negative index
        # if all section indices are negative, use the smallest negative index
        indices = list(section_strings.keys())
        if all(i < 0 for i in indices):
            section_index = min(indices)
        else:
            section_index = min(i for i in indices if i >= 0)
        section_strings[0] = section_strings[section_index]
    return section_strings


def preprocess_image(image_path: str, width, height) -> Tuple[torch.Tensor, np.ndarray, Optional[Image.Image]]:
    "prepare image"
    image_file = Image.open(image_path)
    if image_file.mode == "RGBA":
        alpha = image_file.split()[-1]
    else:
        alpha = None
    image = image_file.convert("RGB")

    image_np = np.array(image)  # PIL to numpy, HWC

    image_np = image_video_dataset.resize_image_to_bucket(image_np, (width, height))
    image_tensor = torch.from_numpy(image_np).float() / 127.5 - 1.0  # -1 to 1.0, HWC
    image_tensor = image_tensor.permute(2, 0, 1)[None, :, None]  # HWC -> CHW -> NCFHW, N=1, C=3, F=1
    return image_tensor, image_np, alpha


def encode_prompts(text_encoder1_path, fp8_llm, text_encoder2_path, device, prompt, negative_prompt, custom_system_prompt, guidance_scale) -> tuple[dict, dict]:
    section_prompts = parse_section_strings(prompt)
    n_prompt = negative_prompt if negative_prompt else ""

    # load text encoder
    tokenizer1, text_encoder1 = load_text_encoder1(text_encoder1_path, fp8_llm, device)
    tokenizer2, text_encoder2 = load_text_encoder2(text_encoder2_path)
    text_encoder2.to(device)

    logger.info(f"Encoding prompt")
    llama_vecs = {}
    llama_attention_masks = {}
    clip_l_poolers = {}
    with torch.autocast(device_type=device.type, dtype=text_encoder1.dtype), torch.no_grad():
        for index, prompt in section_prompts.items():
            llama_vec, clip_l_pooler = hunyuan.encode_prompt_conds(
                prompt, text_encoder1, text_encoder2, tokenizer1, tokenizer2, custom_system_prompt=custom_system_prompt
            )
            llama_vec = llama_vec.cpu()
            clip_l_pooler = clip_l_pooler.cpu()

            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)

            llama_vecs[index] = llama_vec
            llama_attention_masks[index] = llama_attention_mask
            clip_l_poolers[index] = clip_l_pooler

    if guidance_scale == 1.0:
        llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vecs[0]), torch.zeros_like(clip_l_poolers[0])
    else:
        with torch.autocast(device_type=device.type, dtype=text_encoder1.dtype), torch.no_grad():
            llama_vec_n, clip_l_pooler_n = hunyuan.encode_prompt_conds(
                n_prompt, text_encoder1, text_encoder2, tokenizer1, tokenizer2, custom_system_prompt=args.custom_system_prompt
            )
            llama_vec_n = llama_vec_n.cpu()
            clip_l_pooler_n = clip_l_pooler_n.cpu()

    llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

    # free text encoder and clean memory
    del tokenizer1, text_encoder1, tokenizer2, text_encoder2  # do not free shared models
    clean_memory_on_device(device)

    # prepare model input arguments
    arg_c = {}
    arg_null = {}
    for index in llama_vecs.keys():
        llama_vec = llama_vecs[index]
        llama_attention_mask = llama_attention_masks[index]
        clip_l_pooler = clip_l_poolers[index]
        arg_c_i = {
            "llama_vec": llama_vec,
            "llama_attention_mask": llama_attention_mask,
            "clip_l_pooler": clip_l_pooler,
            "prompt": section_prompts[index],  # for debugging
        }
        arg_c[index] = arg_c_i

    arg_null = {
        "llama_vec": llama_vec_n,
        "llama_attention_mask": llama_attention_mask_n,
        "clip_l_pooler": clip_l_pooler_n,
    }

    return arg_c, arg_null


def prepare_i2v_inputs(
    args: argparse.Namespace,
    device: torch.device,
    vae: AutoencoderKLCausal3D
) -> tuple:
    """Prepare inputs for I2V

    Args:
        args: command line arguments
        config: model configuration
        device: device to use
        vae: VAE model, used for image encoding
        shared_models: dictionary containing pre-loaded models

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
            (noise, context, context_null, y, (arg_c, arg_null))
    """

    height, width, video_seconds = check_inputs(args)

    section_image_paths = parse_section_strings(args.image_path)

    section_images = {}
    for index, image_path in section_image_paths.items():
        img_tensor, img_np, _ = preprocess_image(image_path, width, height)
        section_images[index] = (img_tensor, img_np)

    # check end image
    if args.end_image_path is not None:
        end_image_tensor, _, _ = preprocess_image(args.end_image_path, width, height)
    else:
        end_image_tensor = None

    # check end images
    if args.control_image_path is not None and len(args.control_image_path) > 0:
        control_image_tensors = []
        control_mask_images = []
        for ctrl_image_path in args.control_image_path:
            control_image_tensor, _, control_mask = preprocess_image(ctrl_image_path, width, height)
            control_image_tensors.append(control_image_tensor)
            control_mask_images.append(control_mask)
    else:
        control_image_tensors = None
        control_mask_images = None

    encoded_prompts = None
    if args.cache_dir is not None:
        encoded_prompts = Cache(os.path.join(args.cache_dir, "encoded_prompts"))
        logger.info(f"{len(encoded_prompts)=}")

    encode_prompts_decorated = encoded_prompts.memoize()(encode_prompts) if encoded_prompts is not None else encode_prompts # NOTE: Initialized cache has count: 0 and it is Falsy!
    arg_c, arg_null = encode_prompts_decorated(args.text_encoder1, args.fp8_llm, args.text_encoder2, device, args.prompt, args.negative_prompt, args.custom_system_prompt, args.guidance_scale)

    # Debug logging for arg_c and arg_null data types
    log_data_types(arg_c, "arg_c")
    log_data_types(arg_null, "arg_null")

    # region: load image encoder
    feature_extractor, image_encoder = load_image_encoders(args)
    image_encoder.to(device)

    # encode image with image encoder

    section_image_encoder_last_hidden_states = {}
    for index, (img_tensor, img_np) in section_images.items():
        with torch.no_grad():
            image_encoder_output = hf_clip_vision_encode(img_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state.cpu()
        section_image_encoder_last_hidden_states[index] = image_encoder_last_hidden_state

    # free image encoder and clean memory
    image_encoder.to("cpu")
    del image_encoder, feature_extractor
    clean_memory_on_device(device)
    # endregion

    # region: VAE encoding
    logger.info(f"Encoding image to latent space")
    vae.to(device)

    section_start_latents = {}
    for index, (img_tensor, img_np) in section_images.items():
        start_latent = hunyuan.vae_encode(img_tensor, vae).cpu()
        section_start_latents[index] = start_latent

    end_latent = hunyuan.vae_encode(end_image_tensor, vae).cpu() if end_image_tensor is not None else None

    control_latents = None
    if control_image_tensors is not None:
        control_latents = []
        for ctrl_image_tensor in control_image_tensors:
            control_latent = hunyuan.vae_encode(ctrl_image_tensor, vae).cpu()
            control_latents.append(control_latent)

    vae.to("cpu")  # move VAE to CPU to save memory
    clean_memory_on_device(device)
    # endregion

    arg_c_img = {}
    for index in section_images.keys():
        image_encoder_last_hidden_state = section_image_encoder_last_hidden_states[index]
        start_latent = section_start_latents[index]
        arg_c_img_i = {
            "image_encoder_last_hidden_state": image_encoder_last_hidden_state,
            "start_latent": start_latent,
            "image_path": section_image_paths[index],
        }
        arg_c_img[index] = arg_c_img_i

    return height, width, video_seconds, arg_c, arg_null, arg_c_img, end_latent, control_latents, control_mask_images


# def setup_scheduler(args: argparse.Namespace, config, device: torch.device) -> Tuple[Any, torch.Tensor]:
#     """setup scheduler for sampling

#     Args:
#         args: command line arguments
#         config: model configuration
#         device: device to use

#     Returns:
#         Tuple[Any, torch.Tensor]: (scheduler, timesteps)
#     """
#     if args.sample_solver == "unipc":
#         scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=config.num_train_timesteps, shift=1, use_dynamic_shifting=False)
#         scheduler.set_timesteps(args.infer_steps, device=device, shift=args.flow_shift)
#         timesteps = scheduler.timesteps
#     elif args.sample_solver == "dpm++":
#         scheduler = FlowDPMSolverMultistepScheduler(
#             num_train_timesteps=config.num_train_timesteps, shift=1, use_dynamic_shifting=False
#         )
#         sampling_sigmas = get_sampling_sigmas(args.infer_steps, args.flow_shift)
#         timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sampling_sigmas)
#     elif args.sample_solver == "vanilla":
#         scheduler = FlowMatchDiscreteScheduler(num_train_timesteps=config.num_train_timesteps, shift=args.flow_shift)
#         scheduler.set_timesteps(args.infer_steps, device=device)
#         timesteps = scheduler.timesteps

#         # FlowMatchDiscreteScheduler does not support generator argument in step method
#         org_step = scheduler.step

#         def step_wrapper(
#             model_output: torch.Tensor,
#             timestep: Union[int, torch.Tensor],
#             sample: torch.Tensor,
#             return_dict: bool = True,
#             generator=None,
#         ):
#             return org_step(model_output, timestep, sample, return_dict=return_dict)

#         scheduler.step = step_wrapper
#     else:
#         raise NotImplementedError("Unsupported solver.")

#     return scheduler, timesteps


def convert_lora_for_framepack(lora_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Check the format of the LoRA file
    keys = list(lora_sd.keys())
    if keys[0].startswith("lora_unet_"):
        # logging.info(f"Musubi Tuner LoRA detected")
        pass

    else:
        transformer_prefixes = ["diffusion_model", "transformer"]  # to ignore Text Encoder modules
        lora_suffix = None
        prefix = None
        for key in keys:
            if lora_suffix is None and "lora_A" in key:
                lora_suffix = "lora_A"
            if prefix is None:
                pfx = key.split(".")[0]
                if pfx in transformer_prefixes:
                    prefix = pfx
            if lora_suffix is not None and prefix is not None:
                break

        if lora_suffix == "lora_A" and prefix is not None:
            logging.info(f"Diffusion-pipe (?) LoRA detected, converting to the default LoRA format")
            lora_sd = convert_lora_from_diffusion_pipe_or_something(lora_sd, "lora_unet_")

        else:
            logging.info(f"LoRA file format not recognized. Using it as-is.")

    # Check LoRA is for FramePack or for HunyuanVideo
    is_hunyuan = False
    for key in lora_sd.keys():
        if "double_blocks" in key or "single_blocks" in key:
            is_hunyuan = True
            break
    if is_hunyuan:
        logging.info("HunyuanVideo LoRA detected, converting to FramePack format")
        lora_sd = convert_hunyuan_to_framepack(lora_sd)

    return lora_sd


def convert_lora_from_diffusion_pipe_or_something(lora_sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """
    Convert LoRA weights to the format used by the diffusion pipeline to Musubi Tuner.
    Copy from Musubi Tuner repo.
    """
    # convert from diffusers(?) to default LoRA
    # Diffusers format: {"diffusion_model.module.name.lora_A.weight": weight, "diffusion_model.module.name.lora_B.weight": weight, ...}
    # default LoRA format: {"prefix_module_name.lora_down.weight": weight, "prefix_module_name.lora_up.weight": weight, ...}

    # note: Diffusers has no alpha, so alpha is set to rank
    new_weights_sd = {}
    lora_dims = {}
    for key, weight in lora_sd.items():
        diffusers_prefix, key_body = key.split(".", 1)
        if diffusers_prefix != "diffusion_model" and diffusers_prefix != "transformer":
            print(f"unexpected key: {key} in diffusers format")
            continue

        new_key = f"{prefix}{key_body}".replace(".", "_").replace("_lora_A_", ".lora_down.").replace("_lora_B_", ".lora_up.")
        new_weights_sd[new_key] = weight

        lora_name = new_key.split(".")[0]  # before first dot
        if lora_name not in lora_dims and "lora_down" in new_key:
            lora_dims[lora_name] = weight.shape[0]

    # add alpha with rank
    for lora_name, dim in lora_dims.items():
        new_weights_sd[f"{lora_name}.alpha"] = torch.tensor(dim)

    return new_weights_sd


def convert_hunyuan_to_framepack(lora_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert HunyuanVideo LoRA weights to FramePack format.
    """
    new_lora_sd = {}
    for key, weight in lora_sd.items():
        if "double_blocks" in key:
            key = key.replace("double_blocks", "transformer_blocks")
            key = key.replace("img_mod_linear", "norm1_linear")
            key = key.replace("img_attn_qkv", "attn_to_QKV")  # split later
            key = key.replace("img_attn_proj", "attn_to_out_0")
            key = key.replace("img_mlp_fc1", "ff_net_0_proj")
            key = key.replace("img_mlp_fc2", "ff_net_2")
            key = key.replace("txt_mod_linear", "norm1_context_linear")
            key = key.replace("txt_attn_qkv", "attn_add_QKV_proj")  # split later
            key = key.replace("txt_attn_proj", "attn_to_add_out")
            key = key.replace("txt_mlp_fc1", "ff_context_net_0_proj")
            key = key.replace("txt_mlp_fc2", "ff_context_net_2")
        elif "single_blocks" in key:
            key = key.replace("single_blocks", "single_transformer_blocks")
            key = key.replace("linear1", "attn_to_QKVM")  # split later
            key = key.replace("linear2", "proj_out")
            key = key.replace("modulation_linear", "norm_linear")
        else:
            print(f"Unsupported module name: {key}, only double_blocks and single_blocks are supported")
            continue

        if "QKVM" in key:
            # split QKVM into Q, K, V, M
            key_q = key.replace("QKVM", "q")
            key_k = key.replace("QKVM", "k")
            key_v = key.replace("QKVM", "v")
            key_m = key.replace("attn_to_QKVM", "proj_mlp")
            if "_down" in key or "alpha" in key:
                # copy QKVM weight or alpha to Q, K, V, M
                assert "alpha" in key or weight.size(1) == 3072, f"QKVM weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
                new_lora_sd[key_m] = weight
            elif "_up" in key:
                # split QKVM weight into Q, K, V, M
                assert weight.size(0) == 21504, f"QKVM weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 : 3072 * 3]
                new_lora_sd[key_m] = weight[3072 * 3 :]  # 21504 - 3072 * 3 = 12288
            else:
                print(f"Unsupported module name: {key}")
                continue
        elif "QKV" in key:
            # split QKV into Q, K, V
            key_q = key.replace("QKV", "q")
            key_k = key.replace("QKV", "k")
            key_v = key.replace("QKV", "v")
            if "_down" in key or "alpha" in key:
                # copy QKV weight or alpha to Q, K, V
                assert "alpha" in key or weight.size(1) == 3072, f"QKV weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
            elif "_up" in key:
                # split QKV weight into Q, K, V
                assert weight.size(0) == 3072 * 3, f"QKV weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 :]
            else:
                print(f"Unsupported module name: {key}")
                continue
        else:
            # no split needed
            new_lora_sd[key] = weight

    return new_lora_sd


def generate(
    args: argparse.Namespace, gen_settings: GenerationSettings, prof: Optional[torch.profiler.profile] = None, log_timing=None
) -> Optional[tuple[AutoencoderKLCausal3D, torch.Tensor]]:
    """main function for generation

    Args:
        args: command line arguments
        shared_models: dictionary containing pre-loaded models
        log_timing: timing log function

    Returns:
        tuple: (AutoencoderKLCausal3D model (vae), torch.Tensor generated latent)
    """
    device, _dit_weight_dtype = (gen_settings.device, gen_settings.dit_weight_dtype)
    
    if log_timing:
        log_timing("Starting generation function")

    # prepare seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed  # set seed to args for saving

    vae = load_vae(args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, device)
    height, width, video_seconds, context, context_null, context_img, end_latent, control_latents, control_mask_images = (
        prepare_i2v_inputs(args, device, vae)
    )

    model = load_optimized_model(
        dit_path=args.dit,
        fp8_scaled=args.fp8_scaled,
        lora_weight=args.lora_weight,
        lora_multiplier=args.lora_multiplier,
        fp8=args.fp8,
        blocks_to_swap=args.blocks_to_swap,
        attn_mode=args.attn_mode,
        rope_scaling_timestep_threshold=args.rope_scaling_timestep_threshold,
        rope_scaling_factor=args.rope_scaling_factor,
        cache_dir=args.cache_dir,
        device=device,
        include_patterns=args.include_patterns,
        exclude_patterns=args.exclude_patterns,
        lycoris=args.lycoris,
        save_merged_model=args.save_merged_model,
        log_timing=log_timing,
    )
    # if we only want to save the model, we can skip the rest
    if model is None:
        return None

    # sampling
    latent_window_size = args.latent_window_size  # default is 9
    # ex: (5s * 30fps) / (9 * 4) = 4.16 -> 4 sections, 60s -> 1800 / 36 = 50 sections
    total_latent_sections = (video_seconds * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    # set random generator
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)
    num_frames = latent_window_size * 4 - 3

    logger.info(
        f"Video size: {height}x{width}@{video_seconds} (HxW@seconds), fps: {args.fps}, num sections: {total_latent_sections}, "
        f"infer_steps: {args.infer_steps}, frames per generation: {num_frames}"
    )

    # video generation ######
    one_frame_inference = None
    if args.one_frame_inference is not None:
        one_frame_inference = set()
        for mode in args.one_frame_inference.split(","):
            one_frame_inference.add(mode.strip())

    if one_frame_inference is not None:
        real_history_latents = generate_with_one_frame_inference(
            args,
            model,
            context,
            context_null,
            context_img,
            control_latents,
            control_mask_images,
            latent_window_size,
            height,
            width,
            device,
            seed_g,
            one_frame_inference,
        )
    else:
        real_history_latents = inference(
            model=model,
            context=context,
            context_null=context_null,
            context_img=context_img,
            end_latent=end_latent,
            height=height,
            width=width,
            device=device,
            seed_g=seed_g,
            latent_window_size=latent_window_size,
            total_latent_sections=total_latent_sections,
            f1_mode=args.f1,
            one_frame_inference=one_frame_inference,
            num_frames=num_frames,
            sample_solver=args.sample_solver,
            guidance_scale=args.guidance_scale,
            embedded_cfg_scale=args.embedded_cfg_scale,
            guidance_rescale=args.guidance_rescale,
            infer_steps=args.infer_steps,
            latent_paddings=args.latent_paddings,
            log_timing=log_timing,
        )

    del model  # free memory
    synchronize_device(device)

    # wait for 5 seconds until block swap is done
    if args.blocks_to_swap > 0:
        logger.info("Waiting for 5 seconds to finish block swap")
        time.sleep(5)

    gc.collect()
    clean_memory_on_device(device)

    return vae, real_history_latents


def inference(
    model: HunyuanVideoTransformer3DModelPacked,
    context: Dict[int, Dict[str, torch.Tensor]],
    context_null: Dict[str, torch.Tensor],
    context_img: Dict[int, Dict[str, torch.Tensor]],
    end_latent: Optional[torch.Tensor],
    height: int,
    width: int,
    device: torch.device,
    seed_g: torch.Generator,
    latent_window_size: int,
    total_latent_sections: int,
    f1_mode: bool,
    one_frame_inference: Optional[set[str]],
    num_frames: int,
    sample_solver: str,
    guidance_scale: float,
    embedded_cfg_scale: float,
    guidance_rescale: float,
    infer_steps: int,
    latent_paddings: Optional[str],
    log_timing: Optional[Callable] = None,
) -> torch.Tensor:
    # prepare history latents
    history_latents = torch.zeros((1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32)
    if end_latent is not None and not f1_mode:
        logger.info(f"Use end image(s): {end_latent}")
        history_latents[:, :, :1] = end_latent.to(history_latents)

    # prepare clean latents and indices
    if not f1_mode:
        # Inverted Anti-drifting
        total_generated_latent_frames = 0
        latent_paddings_list = list(reversed(range(total_latent_sections)))

        if total_latent_sections > 4 and one_frame_inference is None:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            # 4 sections: 3, 2, 1, 0. 50 sections: 3, 2, 2, ... 2, 1, 0
            latent_paddings_list = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        if latent_paddings is not None:
            # parse user defined latent paddings
            user_latent_paddings = [int(x) for x in latent_paddings.split(",")]
            if len(user_latent_paddings) < total_latent_sections:
                print(
                    f"User defined latent paddings length {len(user_latent_paddings)} does not match total sections {total_latent_sections}."
                )
                print(f"Use default paddings instead for unspecified sections.")
                latent_paddings_list[: len(user_latent_paddings)] = user_latent_paddings
            elif len(user_latent_paddings) > total_latent_sections:
                print(
                    f"User defined latent paddings length {len(user_latent_paddings)} is greater than total sections {total_latent_sections}."
                )
                print(f"Use only first {total_latent_sections} paddings instead.")
                latent_paddings_list = user_latent_paddings[:total_latent_sections]
            else:
                latent_paddings_list = user_latent_paddings
    else:
        start_latent = context_img[0]["start_latent"]
        history_latents = torch.cat([history_latents, start_latent], dim=2)
        total_generated_latent_frames = 1  # a bit hacky, but we employ the same logic as in official code
        latent_paddings_list = [0] * total_latent_sections  # dummy paddings for F1 mode

    for loop_index in range(total_latent_sections):
        latent_padding = latent_paddings_list[loop_index]

        if not f1_mode:
            # Inverted Anti-drifting
            section_index_reverse = loop_index  # 0, 1, 2, 3
            section_index = total_latent_sections - 1 - section_index_reverse  # 3, 2, 1, 0
            section_index_from_last = -(section_index_reverse + 1)  # -1, -2, -3, -4

            is_last_section = section_index == 0
            is_first_section = section_index_reverse == 0
            latent_padding_size = latent_padding * latent_window_size

            logger.info(f"latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}")
        else:
            section_index = loop_index  # 0, 1, 2, 3
            section_index_from_last = section_index - total_latent_sections  # -4, -3, -2, -1
            is_last_section = loop_index == total_latent_sections - 1
            is_first_section = loop_index == 0
            latent_padding_size = 0  # dummy padding for F1 mode

        # select start latent
        if section_index_from_last in context_img:
            image_index = section_index_from_last
        elif section_index in context_img:
            image_index = section_index
        else:
            image_index = 0

        start_latent = context_img[image_index]["start_latent"]
        image_path = context_img[image_index]["image_path"]
        if image_index != 0:  # use section image other than section 0
            logger.info(
                f"Apply experimental section image, latent_padding_size = {latent_padding_size}, image_path = {image_path}"
            )

        if not f1_mode:
            # Inverted Anti-drifting
            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            (
                clean_latent_indices_pre,
                blank_indices,
                latent_indices,
                clean_latent_indices_post,
                clean_latent_2x_indices,
                clean_latent_4x_indices,
            ) = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)

            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, : 1 + 2 + 16, :, :].split(
                [1, 2, 16], dim=2
            )
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

        else:
            # F1 mode
            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
            (
                clean_latent_indices_start,
                clean_latent_4x_indices,
                clean_latent_2x_indices,
                clean_latent_1x_indices,
                latent_indices,
            ) = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]) :, :, :].split(
                [16, 2, 1], dim=2
            )
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

        # if use_teacache:
        #     transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
        # else:
        #     transformer.initialize_teacache(enable_teacache=False)

        # prepare conditioning inputs
        if section_index_from_last in context:
            prompt_index = section_index_from_last
        elif section_index in context:
            prompt_index = section_index
        else:
            prompt_index = 0

        context_for_index = context[prompt_index]
        # if args.section_prompts is not None:
        logger.info(f"Section {section_index}: {context_for_index['prompt']}")
        if log_timing:
            log_timing(f"Starting generation for section {section_index}")

        llama_vec = context_for_index["llama_vec"].to(device, dtype=torch.bfloat16)
        llama_attention_mask = context_for_index["llama_attention_mask"].to(device)
        clip_l_pooler = context_for_index["clip_l_pooler"].to(device, dtype=torch.bfloat16)

        image_encoder_last_hidden_state = context_img[image_index]["image_encoder_last_hidden_state"].to(
            device, dtype=torch.bfloat16
        )

        llama_vec_n = context_null["llama_vec"].to(device, dtype=torch.bfloat16)
        llama_attention_mask_n = context_null["llama_attention_mask"].to(device)
        clip_l_pooler_n = context_null["clip_l_pooler"].to(device, dtype=torch.bfloat16)

        generated_latents = sample_hunyuan(
            transformer=model,
            sampler=sample_solver,
            width=width,
            height=height,
            frames=num_frames,
            real_guidance_scale=guidance_scale,
            distilled_guidance_scale=embedded_cfg_scale,
            guidance_rescale=guidance_rescale,
            # shift=3.0,
            num_inference_steps=infer_steps,
            generator=seed_g,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_attention_mask,
            prompt_poolers=clip_l_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_attention_mask_n,
            negative_prompt_poolers=clip_l_pooler_n,
            device=device,
            dtype=torch.bfloat16,
            image_embeddings=image_encoder_last_hidden_state,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
        )

        # concatenate generated latents
        total_generated_latent_frames += int(generated_latents.shape[2])
        if not f1_mode:
            # Inverted Anti-drifting: prepend generated latents to history latents
            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
                total_generated_latent_frames += 1

            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
        else:
            # F1 mode: append generated latents to history latents
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

        logger.info(f"Generated. Latent shape {real_history_latents.shape}")
        if log_timing:
            log_timing(f"Completed generation for section {section_index}")
        
        # # TODO support saving intermediate video
        # clean_memory_on_device(device)
        # vae.to(device)
        # if history_pixels is None:
        #     history_pixels = hunyuan.vae_decode(real_history_latents, vae).cpu()
        # else:
        #     section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
        #     overlapped_frames = latent_window_size * 4 - 3
        #     current_pixels = hunyuan.vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
        #     history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
        # vae.to("cpu")
        # # if not is_last_section:
        # #     # save intermediate video
        # #     save_video(history_pixels[0], args, total_generated_latent_frames)
        # print(f"Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}")

        # prof.step()

    return real_history_latents

def generate_with_one_frame_inference(
    args: argparse.Namespace,
    model: HunyuanVideoTransformer3DModelPacked,
    context: Dict[int, Dict[str, torch.Tensor]],
    context_null: Dict[str, torch.Tensor],
    context_img: Dict[int, Dict[str, torch.Tensor]],
    control_latents: Optional[List[torch.Tensor]],
    control_mask_images: Optional[List[Optional[Image.Image]]],
    latent_window_size: int,
    height: int,
    width: int,
    device: torch.device,
    seed_g: torch.Generator,
    one_frame_inference: set[str],
) -> torch.Tensor:
    # one frame inference
    sample_num_frames = 1
    latent_indices = torch.zeros((1, 1), dtype=torch.int64)  # 1x1 latent index for target image
    latent_indices[:, 0] = latent_window_size  # last of latent_window

    def get_latent_mask(mask_image: Image.Image) -> torch.Tensor:
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
        mask_image = mask_image.resize((width // 8, height // 8), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.Resampling.LANCZOS)
        mask_image = np.array(mask_image)  # PIL to numpy, HWC
        mask_image = torch.from_numpy(mask_image).float() / 255.0  # 0 to 1.0, HWC
        mask_image = mask_image.squeeze(-1)  # HWC -> HW
        mask_image = mask_image.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # HW -> 111HW (BCFHW)
        mask_image = mask_image.to(torch.float32)
        return mask_image

    if control_latents is None or len(control_latents) == 0:
        logger.info(f"No control images provided for one frame inference. Use zero latents for control images.")
        control_latents = [torch.zeros(1, 16, 1, height // 8, width // 8, dtype=torch.float32)]

    if "no_post" not in one_frame_inference:
        # add zero latents as clean latents post
        control_latents.append(torch.zeros((1, 16, 1, height // 8, width // 8), dtype=torch.float32))
        logger.info(f"Add zero latents as clean latents post for one frame inference.")

    # kisekaeichi and 1f-mc: both are using control images, but indices are different
    clean_latents = torch.cat(control_latents, dim=2)  # (1, 16, num_control_images, H//8, W//8)
    clean_latent_indices = torch.zeros((1, len(control_latents)), dtype=torch.int64)
    if "no_post" not in one_frame_inference:
        clean_latent_indices[:, -1] = 1 + latent_window_size  # default index for clean latents post

    for i in range(len(control_latents)):
        mask_image = None
        if args.control_image_mask_path is not None and i < len(args.control_image_mask_path):
            mask_image = get_latent_mask(Image.open(args.control_image_mask_path[i]))
            logger.info(
                f"Apply mask for clean latents 1x for {i + 1}: {args.control_image_mask_path[i]}, shape: {mask_image.shape}"
            )
        elif control_mask_images is not None and i < len(control_mask_images) and control_mask_images[i] is not None:
            mask_image = get_latent_mask(control_mask_images[i])
            logger.info(f"Apply mask for clean latents 1x for {i + 1} with alpha channel: {mask_image.shape}")
        if mask_image is not None:
            clean_latents[:, :, i : i + 1, :, :] = clean_latents[:, :, i : i + 1, :, :] * mask_image

    for one_frame_param in one_frame_inference:
        if one_frame_param.startswith("target_index="):
            target_index = int(one_frame_param.split("=")[1])
            latent_indices[:, 0] = target_index
            logger.info(f"Set index for target: {target_index}")
        elif one_frame_param.startswith("control_index="):
            control_indices = one_frame_param.split("=")[1].split(";")
            i = 0
            while i < len(control_indices) and i < clean_latent_indices.shape[1]:
                control_index = int(control_indices[i])
                clean_latent_indices[:, i] = control_index
                i += 1
            logger.info(f"Set index for clean latent 1x: {control_indices}")

    # "default" option does nothing, so we can skip it
    if "default" in one_frame_inference:
        pass

    if "no_2x" in one_frame_inference:
        clean_latents_2x = None
        clean_latent_2x_indices = None
        logger.info(f"No clean_latents_2x")
    else:
        clean_latents_2x = torch.zeros((1, 16, 2, height // 8, width // 8), dtype=torch.float32)
        index = 1 + latent_window_size + 1
        clean_latent_2x_indices = torch.arange(index, index + 2).unsqueeze(0)  #  2

    if "no_4x" in one_frame_inference:
        clean_latents_4x = None
        clean_latent_4x_indices = None
        logger.info(f"No clean_latents_4x")
    else:
        clean_latents_4x = torch.zeros((1, 16, 16, height // 8, width // 8), dtype=torch.float32)
        index = 1 + latent_window_size + 1 + 2
        clean_latent_4x_indices = torch.arange(index, index + 16).unsqueeze(0)  #  16

    logger.info(
        f"One frame inference. clean_latent: {clean_latents.shape} latent_indices: {latent_indices}, clean_latent_indices: {clean_latent_indices}, num_frames: {sample_num_frames}"
    )

    # prepare conditioning inputs
    prompt_index = 0
    image_index = 0

    context_for_index = context[prompt_index]
    logger.info(f"Prompt: {context_for_index['prompt']}")

    llama_vec = context_for_index["llama_vec"].to(device, dtype=torch.bfloat16)
    llama_attention_mask = context_for_index["llama_attention_mask"].to(device)
    clip_l_pooler = context_for_index["clip_l_pooler"].to(device, dtype=torch.bfloat16)

    image_encoder_last_hidden_state = context_img[image_index]["image_encoder_last_hidden_state"].to(device, dtype=torch.bfloat16)

    llama_vec_n = context_null["llama_vec"].to(device, dtype=torch.bfloat16)
    llama_attention_mask_n = context_null["llama_attention_mask"].to(device)
    clip_l_pooler_n = context_null["clip_l_pooler"].to(device, dtype=torch.bfloat16)

    generated_latents = sample_hunyuan(
        transformer=model,
        sampler=args.sample_solver,
        width=width,
        height=height,
        frames=1,
        real_guidance_scale=args.guidance_scale,
        distilled_guidance_scale=args.embedded_cfg_scale,
        guidance_rescale=args.guidance_rescale,
        # shift=3.0,
        num_inference_steps=args.infer_steps,
        generator=seed_g,
        prompt_embeds=llama_vec,
        prompt_embeds_mask=llama_attention_mask,
        prompt_poolers=clip_l_pooler,
        negative_prompt_embeds=llama_vec_n,
        negative_prompt_embeds_mask=llama_attention_mask_n,
        negative_prompt_poolers=clip_l_pooler_n,
        device=device,
        dtype=torch.bfloat16,
        image_embeddings=image_encoder_last_hidden_state,
        latent_indices=latent_indices,
        clean_latents=clean_latents,
        clean_latent_indices=clean_latent_indices,
        clean_latents_2x=clean_latents_2x,
        clean_latent_2x_indices=clean_latent_2x_indices,
        clean_latents_4x=clean_latents_4x,
        clean_latent_4x_indices=clean_latent_4x_indices,
    )

    real_history_latents = generated_latents.to(clean_latents)
    return real_history_latents


def save_latent(latent: torch.Tensor, args: argparse.Namespace, height: int, width: int) -> str:
    """Save latent to file

    Args:
        latent: Latent tensor
        args: command line arguments
        height: height of frame
        width: width of frame

    Returns:
        str: Path to saved latent file
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    seed = args.seed
    video_seconds = args.video_seconds
    latent_path = f"{save_path}/{time_flag}_{seed}_latent.safetensors"

    if args.no_metadata:
        metadata = None
    else:
        # Convert args to dictionary and select relevant fields
        args_dict = vars(args)
        metadata_fields = [
            "prompt", "negative_prompt", "infer_steps", "guidance_scale",
            "latent_window_size", "embedded_cfg_scale", "guidance_rescale",
            "sample_solver", "fps", "image_path", "end_image_path",
            "control_image_path", "control_image_mask_path", "one_frame_inference",
            "f1", "rope_scaling_factor", "rope_scaling_timestep_threshold"
        ]
        
        metadata = {}
        # Add args fields
        for field in metadata_fields:
            if field in args_dict and args_dict[field] is not None:
                metadata[field] = str(args_dict[field])
        
        # Add computed values
        metadata["seeds"] = str(seed)
        metadata["height"] = str(height)
        metadata["width"] = str(width)
        metadata["video_seconds"] = str(video_seconds)

    sd = {"latent": latent.contiguous()}
    save_file(sd, latent_path, metadata=metadata)
    logger.info(f"Latent saved to: {latent_path}")

    return latent_path


def save_video(
    video: torch.Tensor, args: argparse.Namespace, original_base_name: Optional[str] = None, latent_frames: Optional[int] = None
) -> str:
    """Save video to file

    Args:
        video: Video tensor
        args: command line arguments
        original_base_name: Original base name (if latents are loaded from files)

    Returns:
        str: Path to saved video file
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    latent_frames_str = "" if latent_frames is None else f"_{latent_frames}"
    video_path = f"{save_path}/{time_flag}_{seed}{original_name}{latent_frames_str}.mp4"

    video = video.unsqueeze(0)
    save_videos_grid(video, video_path, fps=args.fps, rescale=True)
    logger.info(f"Video saved to: {video_path}")

    return video_path


def save_images(sample: torch.Tensor, args: argparse.Namespace, original_base_name: Optional[str] = None) -> str:
    """Save images to directory

    Args:
        sample: Video tensor
        args: command line arguments
        original_base_name: Original base name (if latents are loaded from files)

    Returns:
        str: Path to saved images directory
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    image_name = f"{time_flag}_{seed}{original_name}"
    sample = sample.unsqueeze(0)
    one_frame_mode = args.one_frame_inference is not None
    save_images_grid(sample, save_path, image_name, rescale=True, create_subdir=not one_frame_mode)
    logger.info(f"Sample images saved to: {save_path}/{image_name}")

    return f"{save_path}/{image_name}"


def save_output(
    args: argparse.Namespace,
    vae: AutoencoderKLCausal3D,
    latent: torch.Tensor,
    device: torch.device,
    original_base_names: Optional[List[str]] = None,
    log_timing=None,
) -> None:
    """save output

    Args:
        args: command line arguments
        vae: VAE model
        latent: latent tensor
        device: device to use
        original_base_names: original base names (if latents are loaded from files)
    """
    height, width = latent.shape[-2], latent.shape[-1]  # BCTHW
    height *= 8
    width *= 8
    # print(f"Saving output. Latent shape {latent.shape}; pixel shape {height}x{width}")
    if args.output_type == "latent" or args.output_type == "both" or args.output_type == "latent_images":
        # save latent
        save_latent(latent, args, height, width)
    if args.output_type == "latent":
        return

    if log_timing:
        log_timing("Starting video decoding")
    total_latent_sections = (args.video_seconds * 30) / (args.latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    video = decode_latent(
        args.latent_window_size, total_latent_sections, args.bulk_decode, vae, latent, device, args.one_frame_inference is not None
    )
    if log_timing:
        log_timing("Completed video decoding")

    if args.output_type == "video" or args.output_type == "both":
        # save video
        original_name = "" if original_base_names is None else f"_{original_base_names[0]}"
        save_video(video, args, original_name)

    elif args.output_type == "images" or args.output_type == "latent_images":
        # save images
        original_name = "" if original_base_names is None else f"_{original_base_names[0]}"
        save_images(video, args, original_name)


def preprocess_prompts_for_batch(prompt_lines: List[str], base_args: argparse.Namespace) -> List[Dict]:
    """Process multiple prompts for batch mode

    Args:
        prompt_lines: List of prompt lines
        base_args: Base command line arguments

    Returns:
        List[Dict]: List of prompt data dictionaries
    """
    prompts_data = []

    for line in prompt_lines:
        line = line.strip()
        if not line or line.startswith("#"):  # Skip empty lines and comments
            continue

        # Parse prompt line and create override dictionary
        prompt_data = parse_prompt_line(line)
        logger.info(f"Parsed prompt data: {prompt_data}")
        prompts_data.append(prompt_data)

    return prompts_data


def load_shared_models(args: argparse.Namespace) -> Dict:
    """Load shared models for batch processing or interactive mode.
    Models are loaded to CPU to save memory.

    Args:
        args: Base command line arguments

    Returns:
        Dict: Dictionary of shared models
    """
    shared_models = {}
    tokenizer1, text_encoder1 = load_text_encoder1(args.text_encoder1, args.fp8_llm, "cpu")
    tokenizer2, text_encoder2 = load_text_encoder2(args.text_encoder2)
    feature_extractor, image_encoder = load_image_encoders(args)
    vae = load_vae(args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, "cpu")
    shared_models["tokenizer1"] = tokenizer1
    shared_models["text_encoder1"] = text_encoder1
    shared_models["tokenizer2"] = tokenizer2
    shared_models["text_encoder2"] = text_encoder2
    shared_models["feature_extractor"] = feature_extractor
    shared_models["image_encoder"] = image_encoder
    shared_models["vae"] = vae

    return shared_models


def get_generation_settings(args: argparse.Namespace) -> GenerationSettings:
    device = torch.device(args.device)

    dit_weight_dtype = None  # default
    if args.fp8_scaled:
        dit_weight_dtype = None  # various precision weights, so don't cast to specific dtype
    elif args.fp8:
        dit_weight_dtype = torch.float8_e4m3fn

    logger.info(f"Using device: {device}, DiT weight weight precision: {dit_weight_dtype}")

    gen_settings = GenerationSettings(device=device, dit_weight_dtype=dit_weight_dtype)
    return gen_settings


def main(prof, log_dir):
    # Parse arguments
    args = parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, force=True)  # force=True to override existing configuration
    
    timing_log_path = f"{log_dir}/timing.log"
    start_time = time.time()
    
    def log_timing(message):
        elapsed = time.time() - start_time
        with open(timing_log_path, 'a') as f:
            f.write(f"[T+{elapsed:.2f}s] {message}\n")
        logger.info(f"[T+{elapsed:.2f}s] {message}")
    
    log_timing("Process started")
    
    # Check if latents are provided
    latents_mode = args.latent_path is not None and len(args.latent_path) > 0

    # Set device
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    args.device = device

    prof.start()
    logger.info(f"Profiling config: shapes={args.profile_shapes}, memory={args.profile_memory}, stack={args.profile_stack}")

    if latents_mode:
        # Original latent decode mode
        original_base_names = []
        latents_list = []
        seeds = []

        # assert len(args.latent_path) == 1, "Only one latent path is supported for now"

        for latent_path in args.latent_path:
            original_base_names.append(os.path.splitext(os.path.basename(latent_path))[0])
            seed = 0

            if os.path.splitext(latent_path)[1] != ".safetensors":
                latents = torch.load(latent_path, map_location="cpu")
            else:
                latents = load_file(latent_path)["latent"]
                with safe_open(latent_path, framework="pt") as f:
                    metadata = f.metadata()
                if metadata is None:
                    metadata = {}
                logger.info(f"Loaded metadata: {metadata}")

                if "seeds" in metadata:
                    seed = int(metadata["seeds"])
                if "height" in metadata and "width" in metadata:
                    height = int(metadata["height"])
                    width = int(metadata["width"])
                    args.video_size = [height, width]
                if "video_seconds" in metadata:
                    args.video_seconds = float(metadata["video_seconds"])

            seeds.append(seed)
            logger.info(f"Loaded latent from {latent_path}. Shape: {latents.shape}")

            if latents.ndim == 5:  # [BCTHW]
                latents = latents.squeeze(0)  # [CTHW]

            latents_list.append(latents)

        # latent = torch.stack(latents_list, dim=0)  # [N, ...], must be same shape

        for i, latent in enumerate(latents_list):
            args.seed = seeds[i]

            vae = load_vae(args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, device)
            
            save_output(args, vae, latent, device, original_base_names, log_timing=log_timing)

    else:
        # Single prompt mode (original behavior)

        # Generate latent
        gen_settings = get_generation_settings(args)
        
        result = generate(args, gen_settings, prof=prof, log_timing=log_timing)
        # print(f"Generated latent shape: {latent.shape}")
        if result is None:
            # Model was saved, no further processing needed
            return
        
        vae, latent = result

        # Save latent and video
        save_output(args, vae, latent[0], device, None, log_timing)

    log_timing("Process completed")
    logger.info("Done!")
    logger.info(f"Timing log saved to: {timing_log_path}")


if __name__ == "__main__":
    args = parse_args()
    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    log_dir = f"{args.save_path}/profiles/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        # schedule=torch.profiler.schedule(
        #     wait=1,
        #     warmup=1,
        #     active=5,
        #     repeat=1
        # ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=args.profile_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.profile_stack
    ) as prof:
        prof.add_metadata_json("args", json.dumps(vars(args), default=str))
        main(prof, log_dir)
