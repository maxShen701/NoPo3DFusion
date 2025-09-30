import time
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import hydra
from jaxtyping import install_import_hook
from omegaconf import DictConfig

import torch
import numpy as np
from PIL import Image

from accelerate.logging import get_logger

from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import export_to_video

with install_import_hook(
    ("noposplat",),
    ("beartype", "beartype"),
):
    from noposplat.config import load_typed_root_config
    from noposplat.dataset.data_module import DataModule

from cogvideox.seq_diff_pipeline import CogVideoXImageToVideoPipeline


from utils import (
    check_rank,
    load_nopo_model,
    custom_init_layers,
    validate_custom_init,
    process_mask,
    load_components,
    load_modified_transformer_for_inference,
    convert_color_to_PIL_list_0_1,
    DynamicClipGenerator,
    synthesize_final_video,
    process_single_clip_noposplat
)

image_processor = VaeImageProcessor()


logger = get_logger(__name__, log_level="INFO")

def generate_video(
    args: dict,
    prompt: str,
    image: list,
    mask_feature: list,
    latents_dict: Optional[dict] = None,
    TR3D_model_path: Optional[str] = None,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 49,
    width: int = 720,
    height: int = 480,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42
    ):
    modi_transformer = load_modified_transformer_for_inference(TR3D_model_path,
                                                               args, dtype)
    if TR3D_model_path is None:
        validate_custom_init(modi_transformer)
        custom_init_layers(modi_transformer)
        validate_custom_init(modi_transformer)

    
    components = load_components(args, modi_transformer, dtype)
    pipe = CogVideoXImageToVideoPipeline(**components)

    # pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # pipe.to("cuda")
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    output = pipe(
        prompt=prompt,
        images=image,
        mask=mask_feature,
        height=height,
        width=width,
        # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
        num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
        num_frames=num_frames,  # Number of frames to generate
        num_inference_steps=num_inference_steps,  # Number of inference steps
        use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        # output_type = "latent"
    )
    # video_generate is a list of PIL images
    video_image = output.frames[0]
    latents_dict= output.latents_dict
    return video_image, latents_dict


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="main",
)
def main(cfg_dict: DictConfig):
    
    # Load typed root configuration
    cfg = load_typed_root_config(cfg_dict)
    args = cfg.cogvideox_cfg
    args = SimpleNamespace(**args)
    check_rank(args.local_rank)

    # Prepare the data module
    data_module = DataModule(
        dataset_cfgs=cfg.dataset,
        data_loader_cfg=cfg.data_loader,
        global_rank=args.local_rank
    )
    
    trans_model_path = args.transformer_weight_path
    # trans_model_path = None
    test_loader = data_module.test_dataloader()
    
    output_dir = cfg.test.output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    def process_single_clip_cogvideox(cogvideo_input: dict, 
                                      clip_range,
                                      mask_tensor: torch.Tensor, 
                                      clip_num: int, 
                                      scene_dir: Path,
                                      nopo_color: torch.Tensor):
        """
        处理单个clip
        """
        
        prompt = ""
        mask_feature = process_mask(
                    mask=mask_tensor.unsqueeze(2),
                    interpolate_mode="bilinear",
                    target_size=(480, 720),
                    downsample_factor=8,
                    frame_start=0,
                    frame_interval=4,
                    num_frames=13,
                    dtype=torch.bfloat16
                    )
        
        video_frame_list, _ = generate_video(
            args=args,
            prompt=prompt,
            image=cogvideo_input,
            TR3D_model_path = trans_model_path,
            mask_feature=mask_feature,
            num_frames=len(cogvideo_input)
        )

        def pil_list_to_batch_tensor(pil_list, normalize=True, device='cpu'):
            np_arrays = []
            for pil_img in pil_list:
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                np_arr = np.array(pil_img).astype(np.float32)  # [H, W, C]
                np_arr = np_arr.transpose(2, 0, 1)             # [C, H, W]
                if normalize:
                    np_arr /= 255.0                            
                np_arrays.append(np_arr)

            tensor = torch.stack([torch.from_numpy(arr) for arr in np_arrays], dim=0)  # [F, C, H, W]
            tensor = tensor.unsqueeze(0)                                               # [B, F, C, H, W]

            return tensor.to(device=device, dtype=torch.bfloat16)

        # resize the video_frame_list
        target_size = (256, 256)
        video_frame_list = [img.resize(target_size, Image.LANCZOS) for img in video_frame_list]
        cogvideo_output_path = scene_dir / f"cog_{clip_num}.mp4"
        export_to_video(video_frame_list, cogvideo_output_path, fps=8)

        tensor_data = pil_list_to_batch_tensor(
            video_frame_list, 
            normalize=True, 
            device="cuda"
        )
        
        return {
            'pil_frames': video_frame_list,
            'color': tensor_data,
            'input_color': nopo_color,
            'mask': mask_tensor,
            'clip_range': clip_range
        }


    for step, totalbatch in enumerate(test_loader):
        start_time = time.time()
        print(f"TOTAL target view indexs : {totalbatch['target']['index']}, and the shape: {totalbatch['target']['index'].shape}")
        print("-"*50)
        print(f"TOTAL context view indexs : {totalbatch['context']['index']}")

        (scene,) = totalbatch["scene"]
        scene_dir = output_dir / scene

        cog_output_path = scene_dir / "cogvideo_157.mp4"
        if cog_output_path.exists():
            print(f"cogvideo_157 output found at {cog_output_path}. Skipping scene {scene} with batch: {step}.")
            print("-+="*15)
            continue
        
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        total_gt_video = totalbatch["target"]["image"]  # BFCHW
        total_gt_video_list = convert_color_to_PIL_list_0_1(total_gt_video.cpu())
        gt_output_path = scene_dir / "gt_157.mp4"
        print(f"saving gt video with length: {len(total_gt_video_list)} to {gt_output_path}")
        export_to_video(total_gt_video_list, gt_output_path, fps=8)

        clip_generator = DynamicClipGenerator(
            total_batch=totalbatch,
            initial_context_indices=[0, 12],
            clip_length=49,
            min_context_interval=5,
            max_clips=6,
        ) 

        all_outputs = []
        for clip_idx in range(clip_generator.max_clips):
            current_clip = clip_generator.retrieve_clip(clip_idx)
            if current_clip is None:
                break
            # print(f"Clip {clip_idx} range: {current_clip['clip_range']}")
            noposplat = load_nopo_model(cfg)
            noposplat.requires_grad_(False)
            cogvideo_input, mask_tensor, nopo_color = process_single_clip_noposplat(
                current_clip,
                clip_idx,
                scene_dir,
                noposplat,
            )
            del noposplat
            torch.cuda.empty_cache()

            cogvideo_output = process_single_clip_cogvideox(
                cogvideo_input=cogvideo_input,
                clip_range=current_clip['clip_range'],
                mask_tensor=mask_tensor,
                clip_num=clip_idx,
                scene_dir=scene_dir,
                nopo_color=nopo_color
            )

            clip_generator.generate_clip(
                prev_output=cogvideo_output,
                clip_num=clip_idx,
                ablation=True
            )

            all_outputs.append(cogvideo_output)

        if args.synthesize_output:
            synthesize_final_video(all_outputs=all_outputs, 
                                   scene_dir=scene_dir, 
                                   total_frames=clip_generator.total_frames,
                                   fps=12)


        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"========== Batch {step} processing time: {elapsed_time:.2f} seconds =============")

if __name__ == "__main__":
    main()
