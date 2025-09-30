import os
import torch
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from torch import nn
import math
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate.logging import get_logger
from typing import List, Optional, Tuple, Union
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid

import os
from pathlib import Path
import cv2
import wandb
import hydra
import torch
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace
from torch import autocast
import json
from collections import defaultdict
import copy
import warnings
import math
import torch.nn as nn
import lpips
import torch
import torch.nn.functional as F
import os
import torch
import numpy as np
from PIL import Image
import imageio

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import logging 
import datasets

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from transformers.utils import ContextManagers
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import List, Optional, Tuple, Union
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from safetensors.torch import save_file

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler
)
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.image_processor import VaeImageProcessor
from noposplat.misc.cam_utils import camera_normalization

# Configure beartype and jaxtyping.
with install_import_hook(
    ("noposplat",),
    ("beartype", "beartype"),
):
    from noposplat.config import load_typed_root_config
    from noposplat.dataset.data_module import DataModule

from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace
from noposplat.model.decoder import get_decoder
from noposplat.model.encoder import get_encoder
from noposplat.loss import get_losses
from noposplat.misc.wandb_tools import update_checkpoint_path
from noposplat.nopoModel import NoPoSplat_model
from noposplat.dataset.data_module import DataModule

from diffusers.utils import export_to_video, load_image, load_video


from safetensors.torch import load_file
from PIL import Image
from torchvision.transforms import ToPILImage

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
from torchvision.transforms import ToTensor
from safetensors import safe_open

# Configure beartype and jaxtyping.
with install_import_hook(
    ("noposplat",),
    ("beartype", "beartype"),
):
    from noposplat.config import load_typed_root_config
    from noposplat.dataset.data_module import DataModule

from omegaconf import DictConfig, OmegaConf

from noposplat.model.decoder import get_decoder
from noposplat.model.encoder import get_encoder
from noposplat.loss import get_losses
from noposplat.misc.wandb_tools import update_checkpoint_path
from noposplat.nopoModel import NoPoSplat_model

import os
import glob
import shutil

from PIL import Image
from safetensors import safe_open
from torchvision.transforms import ToPILImage


image_processor = VaeImageProcessor()

logger = get_logger(__name__, log_level="INFO")

def save_checkpoint(state, path, accelerator, transformer, max_to_keep=None):
    """
    Save model checkpoint, including model weights and training state.
    """
    if accelerator.is_main_process:
        checkpoints = glob.glob(os.path.join(path, "checkpoint-*"))
        if max_to_keep and len(checkpoints) >= max_to_keep:
            oldest_checkpoint = min(checkpoints, key=os.path.getctime)
            shutil.rmtree(oldest_checkpoint)
            print(f"Removed oldest checkpoint: {oldest_checkpoint}")

        save_path = os.path.join(path, f"checkpoint-{state['global_step']}")
        os.makedirs(save_path, exist_ok=True)

        unwrapped_transformer = accelerator.unwrap_model(transformer)

        unwrapped_transformer.save_pretrained(
            save_path,
            safe_serialization=True,
            max_shard_size="20GB",
            variant=None
        )

        accelerator.save_state(save_path)
        torch.save(state, os.path.join(save_path, "step_state.pt"))

        logger.info(f"Saved state to {save_path}")

def check_rank(rank_arg: int) -> None:
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != rank_arg:
        rank_arg = env_local_rank

def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin

def compute_prompt_embeds(
    text_encoder,
    tokenizer,
    prompt,
    device=None,
    dtype=None,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
):  
    """
    Accept raw text input, and generate embedding tensor
    Return:
        [batch_size * num_videos_per_prompt, seq_len, embed_dim]。
    """
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids 
    batch_size = text_input_ids.size(0)

    with torch.no_grad(): 
        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, embed_dim = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, embed_dim)

    return prompt_embeds


def load_nopo_model(cfg_dict: DictConfig):
    cfg = cfg_dict
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)

    noposplat = NoPoSplat_model(
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        encoder=encoder,
        encoder_visualizer=encoder_visualizer,
        decoder=decoder,
        losses=get_losses(cfg.loss)
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    noposplat.load_state_dict(checkpoint["state_dict"], strict=False)

    return noposplat

def build_transform(resize_h: int, resize_w: int, num_frames: int) -> transforms.Compose:
    """
    For input BFCHW tensor in [0,1]
    1. Resize to H W
    2. Normalize to [-1, 1]
    """
    return transforms.Compose([
        
        transforms.Lambda(lambda x: x.view(-1, *x.shape[2:])),  # [B, F, C, H, W] → [B*F, C, H, W]
        transforms.Resize((resize_h, resize_w)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Lambda(lambda x: x.view(x.shape[0] // num_frames, num_frames, *x.shape[1:])),
    ])

def custom_init_layers(model: torch.nn.Module) -> None:
    with torch.no_grad():
        pre_proj = model.patch_embed.pre_proj
        pre_proj.weight.zero_()
        
        print(f"Shape of pre_proj.weight: {pre_proj.weight.shape}")
        
        # First min(32, out_ch, in_ch) channel is identity matrix
        for i in range(min(32, pre_proj.out_channels, pre_proj.in_channels)):
            pre_proj.weight[i, i, 0, 0] = 1.0

        # Zero out the 33rd output channel if it exists
        if pre_proj.out_channels >= 33:
            pre_proj.weight[32, :, 0, 0] = 0.0
        
        print("After initialization:")
        print("First few elements of weight (should be identity in the first 32 channels):")
        print(pre_proj.weight[0:5, 0:5, 0, 0])

        if pre_proj.out_channels >= 33:
            print(f"Channel 33 (should be all zero): {pre_proj.weight[32, :, 0, 0]}")

        print(f"Max value in pre_proj.weight: {torch.max(pre_proj.weight)}")


def allocate_to_device(model: torch.nn.Module, device: torch.device) -> None:
    """solve meta tensor problem"""
    try:
        model.to_empty(device=device)
    except RuntimeError as e:
        raise RuntimeError(f"Device allocate failed: {str(e)}") from e


def load_shard_files(model_dir: str, shard_total: int) -> list:
    """Load sliced safetensor"""
    shard_pattern = os.path.join(
        model_dir, 
        "transformer", 
        "diffusion_pytorch_model-{shard:05d}-of-{total:05d}.safetensors"
    )
    
    shard_files = []
    for shard_idx in range(1, shard_total + 1):
        shard_path = shard_pattern.format(shard=shard_idx, total=shard_total)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Loss weight file: {shard_path}")
        shard_files.append(shard_path)
    
    return shard_files

def merge_shards(shard_files: list) -> dict:
    """Merge sliced weight"""
    merged_state_dict = {}
    for shard_path in shard_files:
        with safe_open(shard_path, framework="pt") as f:
            merged_state_dict.update({k: f.get_tensor(k) for k in f.keys()})
    return merged_state_dict

def load_and_validate(
    model: torch.nn.Module, 
    merged_state_dict: dict,
    device: torch.device
) -> dict:
    """Load weight and valid opration"""
    current_keys = set(model.state_dict().keys())
    pretrained_keys = set(merged_state_dict.keys())
    diff_report = {
        "added_params": list(current_keys - pretrained_keys),
        "missing_params": list(pretrained_keys - current_keys)
    }
    
    filtered_state_dict = { # k is in pretrained weight, should be 1025
        k: v.to(device) 
        for k, v in merged_state_dict.items()
        if k in current_keys and k != "patch_embed.pre_proj.weight"
    }
    
    load_info = model.load_state_dict(filtered_state_dict, strict=False)
    
    validation_report = {
        "total_params": len(current_keys),
        "successfully_loaded": len(filtered_state_dict),
        "missing_keys": load_info.missing_keys,
        "unexpected_keys": load_info.unexpected_keys,
        "param_diff": diff_report
    }
    return validation_report

def validate_custom_init(transformer: torch.nn.Module) -> None:
    pre_proj = transformer.patch_embed.pre_proj
    weight = pre_proj.weight.detach().cpu().float()
    
    print("Initial weight after custom initialization:\n", weight[0:5, 0:5, 0, 0])
    
    out_ch, in_ch, _, _ = weight.shape
    identity_valid = True
    check_n = min(32, out_ch, in_ch)
    for i in range(check_n):
        expected = torch.zeros(in_ch)
        expected[i] = 1.0
        actual = weight[i, :, 0, 0]
        
        if not torch.allclose(actual, expected, atol=1e-4):
            identity_valid = False
            print(f"Identity check failed at channel {i}. Expected: {expected}, but got: {actual}")
            break

    zero_valid = True
    if out_ch >= 33:
        actual_33 = weight[32, :, 0, 0]
        expected_33 = torch.zeros(in_ch)
        if not torch.allclose(actual_33, expected_33, atol=1e-4):
            zero_valid = False
            print(f"Zero check failed for 33rd channel. Expected: {expected_33}, but got: {actual_33}")
    
    print("\n=== Validation for pre_proj Initial ===")
    print(f"Identity matrix check (first {check_n} channels): {identity_valid}")
    if out_ch >= 33:
        print(f"Zero-out check for 33rd output channel: {zero_valid}")
    print(f"Weight shape: {tuple(weight.shape)}")


def load_modified_transformer(
    args, 
    accelerator,
    shard_total: int = 3
) -> Tuple[torch.nn.Module, dict]:
    from cogvideox.transformer.cogvideox_transformer_3d_pre_proj import CogVideoXTransformer3DModel

    assert hasattr(args, 'pretrained_model_name_or_path'), "args must include pretrained_model_name_or_path"
    
    # step1: create empty model
    config = CogVideoXTransformer3DModel.load_config(
        args.pretrained_model_name_or_path,
        subfolder="transformer"
    )
    transformer = CogVideoXTransformer3DModel.from_config(config)
    
    allocate_to_device(transformer, accelerator.device)
    
    shard_files = load_shard_files(args.pretrained_model_name_or_path, shard_total)
    merged_state_dict = merge_shards(shard_files)
    
    load_report = load_and_validate(
        transformer, 
        merged_state_dict, 
        accelerator.device
    )
    
    return transformer, load_report

def create_batch_faraway(batch: dict, min_dist_to_context = 30, max_num_clip=3) -> list:
    """
    Create batch faraway from the context
    """
    clip_list = []
    first_sub_seq = {
        "context": batch["context"],
        "scene": batch["scene"],
        "target": {key: value[:, :49] for key, value in batch["target"].items()}
    }
    clip_list.append(first_sub_seq)

    for i in range(1, max_num_clip):
        length = batch["target"]["image"].shape[1]
        # 随机选择一个子序列
        start_frame = np.random.randint(min_dist_to_context-1, 
                                        length - 49)
        end_frame = start_frame + 49
        

        if end_frame > length:
            end_frame = length
            start_frame = end_frame - 49
            sub_seq = {
                "context": batch["context"],
                "scene": batch["scene"],
                "target": {key: value[:, start_frame:end_frame] for key, value in batch["target"].items()}
            }
            clip_list.append(sub_seq)
            break
        else:
            sub_seq = {
                "context": batch["context"],
                "scene": batch["scene"],
                "target": {key: value[:, start_frame:end_frame] for key, value in batch["target"].items()}
            }
            clip_list.append(sub_seq)

    return clip_list

def morphological_dilation(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    assert kernel_size % 2 == 1, "Kernel size must be odd"
    
    dilated = F.max_pool2d(
        mask, 
        kernel_size=kernel_size, 
        padding=kernel_size//2,
        stride=1,     
        ceil_mode=True
    )
    
    return dilated.to(torch.int32).byte()

def generate_mask(depth_tensor: torch.Tensor, 
                  threshold: float = 1e-5, 
                  use_morphology: bool = False, 
                  kernel_size: int = 3,
                  return_pil: bool = False) -> Union[list, torch.Tensor]:
    """Generate mask from depth"""
    assert depth_tensor.dim() == 4, "Input should be [B,F,H,W]"
    
    mask = (depth_tensor <= threshold).float()  # [1, F, H, W]
    
    if use_morphology:
        mask = morphological_dilation(mask, kernel_size) # [1, F, H, W]
    
    mask_int = (mask * 255).to(torch.int32).byte().squeeze(0).cpu()  # [F, H, W]
    
    if return_pil:
        to_pil = ToPILImage(mode='L')
        return [to_pil(frame) for frame in mask_int]
    else:
        return mask_int.unsqueeze(0)
    
def process_mask(
    mask: torch.Tensor,     
    interpolate_mode: str = "bilinear",
    target_size: tuple = (480, 720),
    downsample_factor: int = 8,
    frame_start: int = 0,
    frame_interval: int = 4,
    num_frames: int = 13,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert interpolate_mode in ["nearest", "bilinear", "bicubic"], f"Unsupport: {interpolate_mode}"
    assert mask.ndim == 5, "Mask should be [B, F, C, H, W]"
    # assert mask.shape[1] == 49
    
    batch, frame, channel, hight, width = mask.shape
    mask_float = mask.float() / 255.0  # [B, F, C, H, W]
    mask_reshaped = mask_float.view(batch*frame, channel, hight, width)  # [B*F, C, H, W]
    resized = F.interpolate(
        mask_reshaped,
        size=target_size,
        mode=interpolate_mode,
        align_corners=False if interpolate_mode in ["bilinear", "bicubic"] else None
    )  # [B*F, C, target_H, target_W]
    
    resized = resized.view(batch, frame, channel, *target_size)  # [B, F, C, 480, 720]
    downsampler = torch.nn.AvgPool2d(
        kernel_size=downsample_factor,
        stride=downsample_factor
    )
    
    lowres = downsampler(resized.view(-1, *target_size))  # [B*F*C, 60, 90]

    lowres_h, lowres_w = target_size[0]//downsample_factor, target_size[1]//downsample_factor
    lowres = lowres.view(batch, frame, channel, lowres_h, lowres_w)  # [B, F, C, 60, 90]
    
    frame_indices = [frame_start + i*frame_interval for i in range(num_frames)]
    frame_indices = [idx for idx in frame_indices if idx < frame]
    
    sampled = lowres[:, frame_indices, :, :, :].to(dtype)  # [B, num_frames, C, 60, 90]
    
    return sampled


def calculate_psnr(img1, img2):
    img1 = ToTensor()(img1)
    img2 = ToTensor()(img2)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr_value = 10 * torch.log10((1.0 ** 2) / mse)
    return psnr_value.item()

def calculate_ssim(img1, img2):
    img1 = ToTensor()(img1)
    img2 = ToTensor()(img2)
    img1 = img1.permute(1, 2, 0).numpy()  # to HWC
    img2 = img2.permute(1, 2, 0).numpy()
    
    if img1.max() > 1.0 or img2.max() > 1.0:
        raise ValueError("Image data range must be in [0, 1]. Please normalize the images before passing them to SSIM calculation.")
    ssim_value = compare_ssim(img1, img2, data_range=1.0, multichannel=True, channel_axis=2)
    return ssim_value

def convert_depth_to_grayscale(depth_tensor: torch.Tensor) -> list:
    assert depth_tensor.dim() == 4, "Input should be a tensor with shape [B,F,H,W]"
    
    frames = depth_tensor.squeeze(0).cpu()  # [F, H, W]
    
    min_val = frames.min()
    max_val = frames.max()
    normalized = (frames - min_val) / (max_val - min_val)
    
    normalized_uint8 = (normalized * 255).byte()  # [F, H, W]
    
    to_pil = ToPILImage(mode='L')
    return [to_pil(frame) for frame in normalized_uint8]

def convert_color_to_PIL_list_0_1(tensor: torch.tensor) -> list:
    
    """receive a [-1, 1] tensor on cpu with shape BFCHW , return a list of PIL image
    """
    if tensor.shape[0] != 1:
        raise ValueError(f"Expact batch size to be 1, but got {tensor.shape[0]}")
    
    tensor = tensor.numpy()
    pil_list = []
    to_pil_image = ToPILImage()
    for i in range(tensor.shape[1]):
        image = tensor[0, i, :, :, :] # CHW
        image = torch.from_numpy(image).float()
        image = (image * 255).to(torch.uint8)
        pil_image = to_pil_image(image)
        pil_list.append(pil_image)
    return pil_list

def load_components(
    args: dict,
    modified_transformer: torch.nn.Module,
    dtype: torch.dtype = torch.bfloat16
) -> dict:
    components = {}
    
    components["tokenizer"] = T5Tokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "tokenizer")
    )
    
    components["text_encoder"] = T5EncoderModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "text_encoder"),
        torch_dtype=dtype
    )
    
    components["vae"] = AutoencoderKLCogVideoX.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "vae"),
        torch_dtype=dtype
    )
    
    components["transformer"] = modified_transformer.to(dtype=dtype)
    
    components["scheduler"] = CogVideoXDPMScheduler.from_config(
        args.scheduler_config_path, 
        timestep_spacing="trailing"
    )
    
    return components

def load_modified_transformer_for_inference(
    trained_weight_path,
    args: dict,
    dtype: torch.dtype = torch.bfloat16
):
    """
    Load modified transformer for inference
    """
    from cogvideox.transformer.cogvideox_transformer_3d_pre_proj import CogVideoXTransformer3DModel

    config = CogVideoXTransformer3DModel.load_config(
        args.pretrained_model_name_or_path,
        subfolder="transformer"
    )
    transformer = CogVideoXTransformer3DModel.from_config(config)

    if trained_weight_path is None:
        shard_files = load_shard_files(args.pretrained_model_name_or_path, 3)
        merged_state_dict = merge_shards(shard_files)
        _ = load_and_validate(
                            transformer, 
                            merged_state_dict, 
                            'cuda:0'
                            )
    else:
        state_dict = {}
        with safe_open(trained_weight_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # tensor = f.get_tensor(key)
                # print(f"{key}: {tensor.shape}")
                state_dict[key] = f.get_tensor(key)
            

        try:
            transformer.load_state_dict(state_dict, strict=True)
            print(f"loading weights from {trained_weight_path}")
        except Exception as e:
            missing_keys, unexpected_keys = e.args[0], e.args[1]
            error_msg = f"Faild to load weight:\n"
            error_msg += f"Missing keys: {len(missing_keys)}\n"
            error_msg += f"Unexpected keys: {len(unexpected_keys)}\n"
            error_msg += "Model structure is not matching weight file"
            raise RuntimeError(error_msg) from None

    transformer.eval()
    for param in transformer.parameters():
        param.requires_grad_(False) 

    transformer = transformer.to(dtype=dtype)

    return transformer

class DynamicClipGenerator:
    def __init__(self, total_batch, 
                 initial_context_indices, 
                 clip_length=49, 
                 min_context_interval=5, 
                 max_clips=5, 
                 device="cuda"):
        self.total_batch = total_batch
        self.total_frames = total_batch["target"]["image"].shape[1]
        self.clip_length = clip_length
        self.current_pos = 0
        self.context_stack = [initial_context_indices] # remember each clip's context index
        self.device = device
        self.clip_stack = [] # store generated clips
        self.max_clips = max_clips
        
        self.lpips_model = lpips.LPIPS(net='alex').cuda().eval()
        self.create_first_clip(clip_length)
    

    def create_first_clip(self, clip_length):
        # first 2 context frames index:
        context_indices = self.context_stack[0] 
        
        context_data = {
            key: self.total_batch["target"][key][:, context_indices] 
            for key in ["image", "extrinsics", "intrinsics", "near", "far", "index"]
        }

        batch = {
            "context": context_data,
            "scene": self.total_batch["scene"],
            "target": {key: value[:, :clip_length] for key, value in self.total_batch["target"].items()},
            "clip_range": (0, clip_length)
            }
        # normalize camera pose
        batch = apply_camera_normalization(batch, dynamic_scale=True)
        self.clip_stack.append(batch)
        print(f"----------- Clip [{0}] generated. Range: ({0}-{clip_length}), Padded: [{0}]")

    def generate_clip(self, prev_output, clip_num, ablation=False):
        remaining_frames = self.total_frames - self.current_pos
        if remaining_frames < 1:
            print(f"Stop Generation: Remaining Frame {remaining_frames} (Current Frame: {self.current_pos}, Total Frames: {self.total_frames})")
            return None

        # ============ Ablation Context Construction ============
        if ablation:
            context_dict = self._build_context_data(
                prev_output=prev_output,
                subseq_len=16,
                use_fixed_context=True
            )
        else:
            context_dict = self._build_context_data(
                prev_output=prev_output,
                subseq_len=16
            )

        # then based on fist context frames to select next batch 49 frames
        f_context_indexs = context_dict["index"][:, 0]
        f_context_indices = self.total_batch["target"]["index"][0] == f_context_indexs.item()

        start = torch.nonzero(f_context_indices, as_tuple=True)[0].item() # start idx on total seq
        
        raw_end = start + self.clip_length # index of end frame, so -1
        end = min(raw_end, self.total_frames)
        pad_frames = max(raw_end - self.total_frames, 0)
        
        
        target_data = {
            field: self.total_batch["target"][field][:, start:end] 
            for field in ["image", "extrinsics", "intrinsics", "near", "far", "index"]
        }
        if pad_frames > 0:
            target_data = self._pad_target_data(target_data, pad_frames)


        assert target_data["image"].shape[1] == self.clip_length, f"Target frame number not match: {target_data['image'].shape[1]} vs {self.clip_length}"

        clip_data = self._ensure_float32(
                {
                    "context": context_dict,
                    "target": target_data,
                    "scene": self.total_batch["scene"],
                    'clip_range':(int(start),int(end))
                }
            )

        # 应用归一化并更新指针
        clip_data = apply_camera_normalization(clip_data, dynamic_scale=True)
        self.current_pos = end # move pointer

        # 处理clip数据
        self.clip_stack.append(clip_data)
        print(f"----------- Clip [{clip_num + 1}] generated. Range: ({start}-{end}), Padded: [{pad_frames}]")


    def _ensure_float32(self, data):
        """递归将数据结构中所有张量转换为float32"""
        if isinstance(data, dict):
            return {k: self._ensure_float32(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._ensure_float32(v) for v in data)
        elif torch.is_tensor(data):
            return data.to(torch.float32) if data.dtype != torch.float32 else data
        else:
            return data  # 非张量数据保持原状
    
    def retrieve_clip(self, clip_num: int):
        """
        Input current clip idx, return the data
        """
        if not self.clip_stack or clip_num >= len(self.clip_stack):
            return None

        try:
            clip = copy.deepcopy(self.clip_stack[clip_num])
        except Exception as e:
            print(f"[ERROR] Deepcopy failed for clip {clip_num}: {e}")
            return None
        
        return clip
    
    def _pad_target_data(self, data_dict, pad_frames):
        """统一填充处理"""
        padded_data = {}
        for field, tensor in data_dict.items():
            # 获取最后一个有效帧
            last_frame = tensor[:, -1:]  # 保持维度 [B, 1, ...]
            
            # 生成填充内容
            padding = last_frame.repeat(1, pad_frames, *([1]*(tensor.dim()-2)))
            
            # 拼接数据
            padded_data[field] = torch.cat([tensor, padding], dim=1)
            
            # 类型和设备一致性
            padded_data[field] = padded_data[field].to(tensor.dtype).to(tensor.device)
        
        # 特殊处理index字段
        if 'index' in padded_data:
            last_idx = padded_data['index'][:, -1:]
            padded_data['index'] = torch.cat([
                padded_data['index'][:, :-pad_frames],
                last_idx.repeat(1, pad_frames)
            ], dim=1)
            
        return padded_data
    
    def _build_context_data(self, prev_output, subseq_len=16, use_fixed_context=False):
        """
        Select context frames from prev_output['color']:
        - If use_fixed_context=True, use the last frame and the 13th frame from the end;
        - Otherwise, select a pair of frames with 12-frame interval that has the smallest LPIPS score.

        Args:
            prev_output: Output from the previous clip, containing 'color' and 'clip_range'
            subseq_len: Length of the subsequence to consider
            use_fixed_context: Whether to fix context frames as -13 and -1 (relative indices)
        """
        # Color data of the previous clip, shape: [1, 49, 3, H, W]
        color_clip = prev_output['color']
        # Time range of the previous clip in the total sequence, e.g., (60, 109)
        clip_range = prev_output['clip_range']
        total_start, total_end = clip_range

        # Ensure the length of color_clip matches the clip range
        assert color_clip.shape[1] == (total_end - total_start), "clip_range does not match color shape"

        # Calculate start index of the subsequence within the clip
        subseq_start_idx = self.clip_length - subseq_len
        # Extract the subsequence from the clip, shape: [1, subseq_len, 3, H, W]
        candidate_clip = color_clip[:, subseq_start_idx:]
        # Remove batch dimension for frame selection, shape: [subseq_len, 3, H, W]
        candidate_frames = candidate_clip[0]

        if use_fixed_context:
            # Fixed context: 13th frame from the end and last frame of the subsequence
            i, j = subseq_len - 13, subseq_len - 1
            # Convert to global indices in the total sequence
            total_i = total_start + subseq_start_idx + i
            total_j = total_start + subseq_start_idx + j
        else:
            # LPIPS-based selection: find frame pairs with 12-frame interval
            valid_pairs = [(i, i + 12) for i in range(subseq_len - 12)]
            lpips_scores = []
            
            for i, j in valid_pairs:
                # Compute LPIPS distance between frame i and frame j
                f1 = candidate_frames[i].unsqueeze(0).to(self.device)
                f2 = candidate_frames[j].unsqueeze(0).to(self.device)
                distance = self.lpips_model(f1, f2).item()
                
                # Record global indices and corresponding LPIPS score
                total_i = total_start + subseq_start_idx + i
                total_j = total_start + subseq_start_idx + j
                lpips_scores.append((distance, (total_i, total_j)))
            
            # Select the pair with the smallest LPIPS score (most similar frames)
            total_i, total_j = min(lpips_scores, key=lambda x: x[0])[1]

        # Store the selected global indices in context stack
        self.context_stack.append([total_i, total_j])

        # Retrieve extrinsic/intrinsic data for the two context frames
        ctx_frame1 = self.get_context_frame_data(total_i, seq='target')
        ctx_frame2 = self.get_context_frame_data(total_j, seq='target')

        # Extract images from prev_output (convert global indices to local clip indices)
        local_i = total_i - total_start
        local_j = total_j - total_start
        ctx_frame1['image'] = self._align_tensor_shape(color_clip[:, local_i])
        ctx_frame2['image'] = self._align_tensor_shape(color_clip[:, local_j])

        def _safe_merge(field: str):
            """
            Safely merge two tensors from context frames.
            Checks dimension consistency before concatenation along dim=1.
            """
            tensor1 = ctx_frame1[field]
            tensor2 = ctx_frame2[field]
            assert tensor1.shape[0] == tensor2.shape[0], f"Dimension mismatch: {tensor1.shape} vs {tensor2.shape}"
            return torch.cat([tensor1, tensor2], dim=1)

        # Assemble final context dictionary with merged data
        context_dict = {
            'image': _safe_merge('image'),
            'extrinsics': _safe_merge('extrinsics'),
            'intrinsics': _safe_merge('intrinsics'),
            'near': _safe_merge('near'),
            'far': _safe_merge('far'),
            'index': _safe_merge('index')
        }
        return context_dict


    def _align_tensor_shape(self, tensor):
        return tensor.unsqueeze(1) if tensor.dim() == 4 else tensor

    def get_context_frame_data(self, indices: int, seq='target'):
        if isinstance(indices, int):
            indices = slice(indices, indices+1)
        
        context_frame = {
            "image": self.total_batch[seq]["image"][:, indices, ...],  # [B, 1, C, H, W]
            "extrinsics": self.total_batch[seq]["extrinsics"][:, indices, ...],  # [B, 1, 4, 4]
            "intrinsics": self.total_batch[seq]["intrinsics"][:, indices, ...],  # [B, 1, 3, 3]
            "near": self.total_batch[seq]["near"][:, indices, ...],  # [B, 1]
            "far": self.total_batch[seq]["far"][:, indices, ...],     # [B, 1]
            "index": self.total_batch[seq]["index"][:, indices, ...] # [B, 1]
        }
        
        assert context_frame["image"].dim() == 5, "Image应为5维: BFC(H)(W)"
        assert context_frame["extrinsics"].dim() == 4, "Extrinsics应为4维: BF44"
        return context_frame
    

def apply_camera_normalization(clip: dict, dynamic_scale=True) -> dict:
    normalized_clip = copy.deepcopy(clip)
    
    if dynamic_scale:
        context_extrinsics = normalized_clip["context"]["extrinsics"].squeeze(0)
        a = context_extrinsics[0, :3, 3]
        b = context_extrinsics[-1, :3, 3]
        scale = (a - b).norm()

        normalized_clip["target"]["extrinsics"] = normalized_clip["target"]["extrinsics"].clone()
        normalized_clip["context"]["extrinsics"] = normalized_clip["context"]["extrinsics"].clone()
        
        normalized_clip["target"]["extrinsics"][..., :3, 3] /= scale
        normalized_clip["context"]["extrinsics"][..., :3, 3] /= scale

        normalized_clip["context"]["near"] = (normalized_clip["context"]["near"] / scale).to(normalized_clip["context"]["near"].dtype)
        normalized_clip["context"]["far"] = (normalized_clip["context"]["far"] / scale).to(normalized_clip["context"]["far"].dtype)
        normalized_clip["target"]["near"] = (normalized_clip["target"]["near"] / scale).to(normalized_clip["target"]["near"].dtype)
        normalized_clip["target"]["far"] = (normalized_clip["target"]["far"] / scale).to(normalized_clip["target"]["far"].dtype)

    anchor_pose = normalized_clip["context"]["extrinsics"][:, 0]
    
    target_extrinsics = normalized_clip["target"]["extrinsics"].squeeze(0)
    context_extrinsics = normalized_clip["context"]["extrinsics"].squeeze(0)

    target_extrinsics = camera_normalization(anchor_pose, target_extrinsics).unsqueeze(0)
    context_extrinsics = camera_normalization(anchor_pose, context_extrinsics).unsqueeze(0)

    normalized_clip["target"]["extrinsics"] = target_extrinsics
    normalized_clip["context"]["extrinsics"] = context_extrinsics
    
    return normalized_clip

def pil_to_tensor(pil):
    np_array = np.array(pil).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_array).permute(2,0,1)          # [C,H,W]
    return tensor.unsqueeze(0).unsqueeze(0)              # [1,1,C,H,W]
    
def synthesize_final_video(all_outputs, scene_dir, total_frames, fps=8):
    """
    Generate final video and merged tensors simultaneously, supporting PSNR calculation.
    
    Args:
        all_outputs: List[Dict] - Contains PIL frames and normalized tensors for each clip
        scene_dir: Directory path to save the output video and tensors
        total_frames: int - Target total number of frames in the final sequence
        fps: int, optional - Frames per second of the output video, default is 8
    """
    # Initialize output paths
    scene_dir = Path(scene_dir)
    video_path = scene_dir / "cogvideo_157.mp4"
    cogvideo_tensor_path = scene_dir / "cogvideo_157.pt"
    noposplat_tensor_path = scene_dir / "noposplat_157.pt" 
    
    # Initialize storage structures
    frame_dict = {}          # Maps frame index to list of PIL images: {frame_idx: [pil_images]}
    cog_tensor_buffer = {}   # Buffers tensors for CogVideoX: {frame_idx: [tensors]}
    nopo_tensor_buffer = {}  # Buffers tensors for NoPoSplat: {frame_idx: [tensors]}
    h, w = 256, 256          # Assumed fixed resolution (as per original logic)

    # -------------------------------------------------------------------------
    # Step 1: Collect all frame data (PIL images + tensors) from each clip
    # -------------------------------------------------------------------------
    for clip in all_outputs:
        clip_start, clip_end = clip['clip_range']
        clip_frame_count = clip_end - clip_start

        # 1.1 Collect PIL frames (for video generation)
        for local_idx, pil_frame in enumerate(clip['pil_frames']):
            global_frame_idx = clip_start + local_idx
            # Skip if frame index exceeds total target frames
            if global_frame_idx >= total_frames:
                continue
            # Add frame to dictionary (initialize list if key doesn't exist)
            frame_dict.setdefault(global_frame_idx, []).append(pil_frame)

        # 1.2 Collect CogVideoX tensors (from 'color' key)
        # Assume 'color' shape: [Batch=1, Time, Channels, Height, Width]
        cog_clip_tensor = clip['color'].squeeze(0)  # Remove batch dim → [T, C, H, W]
        for local_t in range(cog_clip_tensor.shape[0]):
            global_t = clip_start + local_t
            if global_t >= total_frames:
                continue
            cog_tensor_buffer.setdefault(global_t, []).append(cog_clip_tensor[local_t])

        # 1.3 Collect NoPoSplat tensors (from 'input_color' key)
        nopo_clip_tensor = clip['input_color'].squeeze(0)  # Remove batch dim → [T, C, H, W]
        for local_t in range(nopo_clip_tensor.shape[0]):
            global_t = clip_start + local_t
            if global_t >= total_frames:
                continue
            nopo_tensor_buffer.setdefault(global_t, []).append(nopo_clip_tensor[local_t])

    # -------------------------------------------------------------------------
    # Step 2: Generate final video frames (with overlap blending)
    # -------------------------------------------------------------------------
    final_video_frames = []
    # Process frames in ascending order of global index
    for global_idx in sorted(frame_dict.keys()):
        frame_candidates = frame_dict[global_idx]
        
        # Use single frame if no overlap; blend if multiple overlapping frames
        if len(frame_candidates) == 1:
            final_video_frames.append(frame_candidates[0])
        else:
            # Blend overlapping frames using weighted average (via external function)
            blended_frame = blend_pil_frames(frame_candidates)
            final_video_frames.append(blended_frame)

    # Export merged frames to video
    export_to_video(final_video_frames, str(video_path), fps=fps)
    print(f"[✓] Video saved to: {video_path}, Total frames: {len(final_video_frames)}")

    # -------------------------------------------------------------------------
    # Helper: Process tensor buffer to generate merged tensor + count metadata
    # -------------------------------------------------------------------------
    def _process_tensor_buffer(buffer, save_path):
        """
        Helper function to merge tensors from buffer and save to disk.
        
        Args:
            buffer: Dict - Tensor buffer ({frame_idx: [tensors]})
            save_path: Path - Path to save the merged tensor
        Returns:
            merged_tensor: torch.Tensor - Merged tensor of shape [total_frames, C, H, W]
        """
        # Initialize merged tensor and count tracker (for averaging overlapping tensors)
        merged_tensor = torch.zeros((total_frames, 3, h, w), dtype=torch.float32)
        frame_count = torch.zeros((total_frames, 1, 1, 1))  # For weighted averaging

        # Merge tensors for each frame
        for frame_idx, tensor_list in buffer.items():
            # Stack all tensors for the current frame and compute average
            stacked_tensors = torch.stack(tensor_list, dim=0)  # Shape: [Num_Tensors, C, H, W]
            merged_tensor[frame_idx] = stacked_tensors.mean(dim=0)  # Average across overlapping tensors
            frame_count[frame_idx] = len(tensor_list)  # Record number of overlapping tensors

        # Save merged tensor with metadata
        torch.save({
            'tensor': merged_tensor,
            'counts': frame_count.squeeze(),  # Remove singleton dims for compactness
            'resolution': (h, w)
        }, str(save_path))
        
        return merged_tensor

    # -------------------------------------------------------------------------
    # Step 3: Process and save CogVideoX tensor
    # -------------------------------------------------------------------------
    merged_cog_tensor = _process_tensor_buffer(cog_tensor_buffer, cogvideo_tensor_path)
    print(f"[✓] CogVideoX output tensor saved to: {cogvideo_tensor_path}, Shape: {merged_cog_tensor.shape}")

    # -------------------------------------------------------------------------
    # Step 4: Process and save NoPoSplat tensor
    # -------------------------------------------------------------------------
    merged_nopo_tensor = _process_tensor_buffer(nopo_tensor_buffer, noposplat_tensor_path)
    print(f"[✓] NoPoSplat tensor saved to: {noposplat_tensor_path}, Shape: {merged_nopo_tensor.shape}")

    # for noposplat
    merged_tensor_nopo  = torch.zeros((total_frames, 3, h, w), dtype=torch.float32)
    counts = torch.zeros((total_frames, 1, 1, 1))  # 用于平均计数
    for idx, tensors in nopo_tensor_buffer.items():
        # 叠加所有来源的张量
        stacked = torch.stack(tensors, dim=0)  # [N, C, H, W]
        merged_tensor_nopo [idx] = stacked.mean(dim=0)
        counts[idx] = len(tensors)

    torch.save({
        'tensor': merged_tensor_nopo ,
        'counts': counts.squeeze(),
        'resolution': (h, w)
    }, str(noposplat_tensor_path))
    
    print(f"[✓] Noposplat tensor saved to: {noposplat_tensor_path}, shape: {merged_tensor_nopo .shape}")

def blend_pil_frames(frames):
    arrays = [np.array(f).astype(np.float32) for f in frames]
    blended = np.mean(arrays, axis=0).astype(np.uint8)
    return Image.fromarray(blended)

def move_batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    else:
        return batch
    
def process_single_clip_noposplat(batch: dict, clip_num: int, scene_dir: Path, noposplat):
    """
    Using Noposplat to process a single clip
    """
    context_image_list = convert_color_to_PIL_list_0_1(batch["context"]["image"].cpu())
    for i, img in enumerate(context_image_list):
        img.save(scene_dir / f"context_{clip_num}_{i}.png")

    print(f"target view indexs : {batch['target']['index']}, and the shape: {batch['target']['index'].shape}")
    print("-"*50)
    print(f"context view indexs : {batch['context']['index']}")

    # save gt video
    total_gt_video = batch["target"]["image"]  # BFCHW
    total_gt_video_list = convert_color_to_PIL_list_0_1(total_gt_video.cpu())
    gt_output_path = scene_dir / f"gt_{clip_num}.mp4"
    print(f"saving gt video with length: {len(total_gt_video_list)} to {gt_output_path}")
    export_to_video(total_gt_video_list, gt_output_path, fps=8)

    batch = move_batch_to_device(batch, "cuda")
    if batch['target']['index'].shape[0] != 1:
        raise ValueError(f"Expected batch_size is 1 but got {batch['target']['index'].shape[0]}")

    noposplat = noposplat.to("cuda")
    with torch.no_grad():
        with autocast(device_type="cuda",dtype=torch.float32):
            _, nopo_out = noposplat.test_step(batch)
        nopo_color = nopo_out.color.clip(0, 1)
        nopo_depth = nopo_out.depth
    noposplat = noposplat.to("cpu")

    mask_tensor = generate_mask(nopo_depth, threshold=1e-5, use_morphology=True, kernel_size=5)
    
    cogvideo_input = convert_color_to_PIL_list_0_1(nopo_color.cpu())
    path = scene_dir / f"cog_input_{clip_num}.mp4"
    print(f"saving cogvideo input with length: {len(cogvideo_input)} to {path}")
    export_to_video(cogvideo_input, path, fps=8)

    cogvideo_input = [nopo_color[0, i, :, :, :] for i in range(nopo_color.shape[1])]

    return cogvideo_input, mask_tensor, nopo_color

def get_clip_windows(tensor, clip_length=13, stride=10, num_clips=4):
    """
    从 shape [1, T, C, H, W] 的 global tensor 中提取多个 sliding window clip。

    返回 shape: [num_clips, clip_length, C, H, W]
    """
    clips = []
    for i in range(num_clips):
        start = i * stride
        end = start + clip_length
        clip = tensor[:, start:end]  # shape: [1, clip_length, C, H, W]
        clips.append(clip)
    return torch.cat(clips, dim=0)  # shape: [num_clips, clip_length, C, H, W]


class CausalLatentFuser:
    def __init__(self, total_clips=4, latent_frames=13, image_per_latent=4):
        self.latent_frames = latent_frames
        self.image_per_latent = image_per_latent
        self.total_clips = total_clips

    def fuse_batch(self, latents):
        """
        批量因果融合核心逻辑（输入维度 [B, F, C, H, W]）
        :param latents: 输入潜在张量 [B, F=13, C, H, W]
        :return: 融合后的潜在张量 [B, F=13, C, H, W]
        """
        B, F, C, H, W = latents.shape
        fused = latents.clone()
        
        # 动态混合权重（余弦退火
        blend_weights = torch.cos(torch.linspace(0, math.pi/2, 4))
        blend_weights = blend_weights.view(1, 4, 1, 1, 1).to(latents.device)
        
        for i in range(1, self.total_clips):
            # 提取重叠区域（维度调整为 [3, C, H, W]）
            src_feats = latents[i-1, -4:, ...]  # 取最后3帧
            tgt_feats = latents[i, :4, ...]  # 取第2、3、4帧（跳过第1帧）
                        
            # 时空混合（新增批次维度）
            blended = self._spatial_temporal_blend(
                src_feats.unsqueeze(0),  # [1, 3, C, H, W]
                tgt_feats.unsqueeze(0),  # [1, 3, C, H, W]
                blend_weights
            )  # [3, C, H, W]
            
            # 更新目标clip（保留前clip 70%信息）
            fused[i, :4, ...] = 0.7 * src_feats + 0.3 * blended
            
            # 逆向轻度混合（保持因果约束）
            fused[i-1, -4:, ...] = 0.7 * fused[i-1, -4:, ...] + 0.3 * blended

        return fused

    def _spatial_temporal_blend(self, src, tgt, weights):
        """三维时空混合（支持批量处理）
        :param src: 源特征 [B, 3, C, H, W]
        :param tgt: 目标特征 [B, 3, C, H, W]
        :param weights: 混合权重 [1,3,1,1,1]

        return: 融合后的特征 [3, C, H, W]
        """
        # 维度对齐广播
        blended = src * weights + tgt * (1 - weights)
        return blended.squeeze(0) # B3CHW -> 3CHW

class CausalX0Fuser:
    def __init__(self, total_clips=4, latent_frames=13):
        self.latent_frames = latent_frames
        self.total_clips = total_clips
        self.overlap = 3  # 重叠帧数
        
        # 初始化可学习参数
        self.blend_weights = nn.Parameter(torch.linspace(0.8, 0.2, self.overlap))
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1)
        )

    def fuse_x0(self, x0_batch: torch.Tensor, timestep: int, total_steps: int):
        """
        时空感知融合核心逻辑
        :param x0_batch: [B, F=13, C, H, W]
        :param timestep: 当前时间步
        :param total_steps: 总去噪步数
        """
        B, F, C, H, W = x0_batch.shape
        fused = x0_batch.clone()
        
        # 动态衰减因子（后期减少融合强度）
        decay_factor = 1.0 - (timestep / total_steps) ** 2
        
        for clip_idx in range(1, B):
            # 前Clip的重叠区域（最后3帧）
            prev_frames = x0_batch[clip_idx-1, -self.overlap:, ...]  # [3, C, H, W]
            
            # 当前Clip的重叠区域（前3帧）
            curr_frames = x0_batch[clip_idx, :self.overlap, ...]     # [3, C, H, W]
            
            # 空间注意力混合
            spatial_weights = self._compute_spatial_weights(prev_frames, curr_frames)  # [3, 1, H, W]
            
            # 时间混合权重（可学习参数 + 动态衰减）
            blend_weights = self.blend_weights.view(-1, 1, 1, 1) * decay_factor  # [3,1,1,1]
            
            # 执行融合
            blended = (prev_frames * blend_weights * spatial_weights + 
                      curr_frames * (1 - blend_weights * spatial_weights))
            
            # 更新当前Clip的前3帧
            fused[clip_idx, :self.overlap, ...] = blended
            
            # 逆向混合前Clip的后3帧（弱效应）
            fused[clip_idx-1, -self.overlap:, ...] = (
                0.95 * fused[clip_idx-1, -self.overlap:, ...] + 
                0.05 * blended
            )
        
        return fused

    def _compute_spatial_weights(self, prev: torch.Tensor, curr: torch.Tensor):
        """
        计算空间注意力权重
        :param prev: [3, C, H, W] 前Clip的帧
        :param curr: [3, C, H, W] 当前Clip的帧
        :return: [3, 1, H, W] 空间混合权重
        """
        batch_size = prev.shape[0]
        weights = []
        
        for t in range(batch_size):
            # 计算差异图
            diff = torch.abs(prev[t] - curr[t]).mean(dim=0, keepdim=True)  # [1, H, W]
            
            # 构建注意力输入
            attn_input = torch.cat([
                prev[t].mean(dim=0, keepdim=True),  # [1, H, W]
                curr[t].mean(dim=0, keepdim=True)   # [1, H, W]
            ], dim=0)  # [2, H, W]
            
            # 通过CNN生成权重
            attn_map = torch.sigmoid(self.spatial_attn(attn_input.unsqueeze(0)))  # [1, 1, H, W]
            weights.append(attn_map)
        
        return torch.stack(weights, dim=0).squeeze(1)  # [3, 1, H, W]
    


