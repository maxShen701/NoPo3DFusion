import os
import logging
from types import SimpleNamespace

import wandb
import hydra
from jaxtyping import install_import_hook
from omegaconf import DictConfig

import torch
from torch import autocast

import datasets
import transformers
from transformers import (
    AutoTokenizer,
    T5EncoderModel,
    ContextManagers
)

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)
from diffusers.optimization import get_scheduler

with install_import_hook(
    ("noposplat",),
    ("beartype", "beartype"),
):
    from noposplat.config import load_typed_root_config
    from noposplat.dataset.data_module import DataModule

from utils import (
    save_checkpoint,
    move_batch_to_device,
    check_rank,
    prepare_rotary_positional_embeddings,
    compute_prompt_embeds,
    load_nopo_model,
    custom_init_layers,
    validate_custom_init,
    create_batch_faraway,
    generate_mask,
    process_mask,
    load_modified_transformer,
    build_transform
)


logger = get_logger(__name__, log_level="INFO")

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="main",
)
def infer(cfg_dict: DictConfig):
    # Load typed root configuration
    cfg = load_typed_root_config(cfg_dict)
    noposplat = load_nopo_model(cfg)
    args = cfg.cogvideox_cfg
    args = SimpleNamespace(**args)
    check_rank(args.local_rank)

    # Prepare the data module
    data_module = DataModule(
        dataset_cfgs=cfg.dataset,
        data_loader_cfg=cfg.data_loader,
        global_rank=args.local_rank
    )

    # Initial cogvideoX
    logging_dir = os.path.join(args.output_dir, args.logging_dir) 
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # log_with=args.report_to,
        project_config=accelerator_project_config,
        device_placement=True
    )
    set_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

     # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    print("--------load CogVideoXTransformer3DModel for 5B -------------")

    transformer, report = load_modified_transformer(args=args,
                                            accelerator=accelerator,
                                            shard_total=3)
    transformer = transformer.to(weight_dtype)

    if not args.resume_training:
        validate_custom_init(transformer)
        custom_init_layers(transformer)
        validate_custom_init(transformer)

    print("\nTransformer loading report: ")
    print(f"Successfully loaded: {report['successfully_loaded']}/{report['total_params']}")
    print(f"New added param: {len(report['param_diff']['added_params'])}")
    print(f"Lost param: {len(report['missing_keys'])}")
    print(f"Different param analysis: {report['param_diff']}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        vae = AutoencoderKLCogVideoX.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
    
    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    noposplat.requires_grad_(False)

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device)
    noposplat = noposplat.to(accelerator.device)

    transformer.train()
    # if accelerator.is_main_process:
    #     print("--------Set the transformer3d to trainable -------------")
    #     print(transformer)
    # https://huggingface.co/THUDM/CogVideoX-5b-I2V/blob/main/transformer/diffusion_pytorch_model.safetensors.index.json 
    trainable_modules = ['ff.net', 'to_q', 'to_v', 'proj_out', 'pre_proj', 'proj']
    # trainable_modules = ['pos_embedding', 'proj', 'ff.net', 'to_q', 'to_v', 'proj_out']
    trainable_modules_low_learning_rate = []

    for name, param in transformer.named_parameters():
        for trainable_module_name in trainable_modules + trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                param.requires_grad = True
                break
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing() 
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW


    trainable_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    trainable_params_optim = [
        {'params': [], 'lr': args.learning_rate},
        {'params': [], 'lr': args.learning_rate / 2},
    ]
    in_already = []
    for name, param in transformer.named_parameters():
        high_lr_flag = False
        if name in in_already:
            continue
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                in_already.append(name)
                high_lr_flag = True
                trainable_params_optim[0]['params'].append(param)
                if accelerator.is_main_process:
                    print(f"Set {name} to lr : {args.learning_rate}")
                break
        if high_lr_flag:
            continue
        for trainable_module_name in trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                in_already.append(name)
                trainable_params_optim[1]['params'].append(param)
                if accelerator.is_main_process:
                    print(f"Set {name} to lr : {args.learning_rate / 2}")
                break
    
    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            # weight_decay=args.adam_weight_decay,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, lr_scheduler = accelerator.prepare(
        transformer, optimizer, lr_scheduler
    )

    # Resume training from last checkpoint and logging
    if args.resume_training:
        output_dir = args.output_dir
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))

        if checkpoints:
            last_checkpoint = os.path.join(output_dir, checkpoints[-1])
            if accelerator.is_main_process:
                print("+="*30)
            accelerator.load_state(last_checkpoint)
            if accelerator.is_main_process:
                print("=+"*30)
            if accelerator.is_main_process:
                print(f"Last checkpoint path: {last_checkpoint}")
                logger.info(f"Last checkpoint path: {last_checkpoint}")
                print("********================= Training state restored ================********")
                logger.info("********================= Training state restored ================********")
                logger.info(f"***** Resume recording on wandb with ID: {args.wandb_resume_id} *****")

                wandb.init(
                    # set the wandb project where this run will be logged
                    project="cogvideoxI2V_fintune_re10k",
                    # track hyperparameters and run metadata
                    config={
                        "batch_size": cfg.data_loader.train.batch_size,
                        "epochs": args.num_train_epochs,
                        },
                    resume="allow",
                    id=args.wandb_resume_id
                )
        else:
            logger.warning("No checkpoints found.")

    train_dataloader = data_module.train_dataloader()

    if not args.resume_training and accelerator.is_main_process:
        logger.info("***** Running training *****")
        wandb.init(
                project="cogvideoxI2V_fintune_re10k",
                config={
                    "batch_size": cfg.data_loader.train.batch_size,
                    "epochs": args.num_train_epochs
                    }
                )
    logger.info(f"  Num Batches = {len(train_dataloader)}")
    logger.info(f"  Batch Size = {cfg.data_loader.train.batch_size}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data_loader.train.batch_size}")

    # Resume step info on main process
    if args.resume_training and accelerator.is_main_process:
        step_state = os.path.join(last_checkpoint, "step_state.pt")
        if os.path.exists(step_state):
            custom_state = torch.load(step_state, map_location="cpu")
            global_step = custom_state.get("global_step", 0)
            first_epoch = custom_state.get("epoch", 0)
            step = custom_state.get("step", 0)
            logger.info(f"Restored global_step: {global_step}, epoch: {first_epoch}, step: {step} from {step_state}")
            print(f"Restored global_step: {global_step}, epoch: {first_epoch}, step: {step}")
    else:
        logger.warning(f"Resume Training is {args.resume_training}!")
        global_step = 0
        first_epoch = 0
        step = 0

    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1) 
    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in transformer.parameters())
        trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable_params} ({trainable_params/total_params:.1%} of total)")
        
    for epoch in range(first_epoch, args.num_train_epochs): 
        for step, batch in enumerate(train_dataloader):
            if accelerator.is_main_process:
                print("&"*40)
                print("&"*40)
                print(f"target view indexs : {batch['target']['index']}, and the shape: {batch['target']['index'].shape}")
                print("-"*20)
                print(f"context view indexs : {batch['context']['index']}")
                print("&"*15)
            batch_list = create_batch_faraway(batch)
            for list_step, batch in enumerate(batch_list):
                # Noposplat stage
                if accelerator.is_main_process:
                    print("&"*15)
                    print(f"target view indexs : {batch['target']['index']}, and the shape: {batch['target']['index'].shape}")
                    print("-"*20)
                    print(f"context view indexs : {batch['context']['index']}")
                    print("&"*15)
                batch = move_batch_to_device(batch, accelerator.device)
                with torch.no_grad():
                    with autocast(device_type="cuda",dtype=torch.float32):
                        _, nopo_out = noposplat.test_step(batch)
                    # the first seq in batch_list is reversed on both gt and nopoout
                    nopo_color = nopo_out.color.clip(0, 1)
                    nopo_depth = nopo_out.depth
                    gt_video = batch["target"]["image"]  # BFCHW 

                    mask_tensor = generate_mask(nopo_depth, threshold=1e-5, use_morphology=True, kernel_size=5)
                    # print(f"mask tensor with shape: {mask_tensor.shape} with [min] {torch.min(mask_tensor)} and [max] {torch.max(mask_tensor)}")

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

                    # BFCHW, resize to 480*720
                    transform = build_transform(resize_h=480, resize_w=720, num_frames=gt_video.shape[1])
                    gt_video_resized = transform(gt_video)
                    nopo_video_resized = transform(nopo_color)

                    del gt_video
                    del nopo_color
                    torch.cuda.empty_cache()

                    # ready for VAE
                    gt_resized_video = gt_video_resized.permute(0, 2, 1, 3, 4) # BFCHW -> BCFHW
                    nopo_resized_video = nopo_video_resized.permute(0, 2, 1, 3, 4) # BFCHW -> BCFHW

                    gt_resized_video = gt_resized_video.to(device=accelerator.device,dtype=weight_dtype)
                    nopo_resized_video = nopo_resized_video.to(device=accelerator.device,dtype=weight_dtype)

                    gt_video, nopo_video, mask_feature = accelerator.prepare(gt_resized_video, nopo_resized_video, mask_feature)

                    gt_video_latents = vae.encode(gt_video).latent_dist.sample() * vae.config.scaling_factor

                    image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=gt_video.device)
                    image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=gt_video.dtype)
                    
                    input_noisy = nopo_video + torch.randn_like(nopo_video) * image_noise_sigma[:, None, None, None, None]
                    input_video_latents = vae.encode(input_noisy).latent_dist.sample() * vae.config.scaling_factor

                    mask_feature = mask_feature.permute(0, 2, 1, 3, 4).to(gt_video.device) # BF1HW -> B1FHW same to VAE output

                    is_warm_up = False
                    if step < args.warm_up_step and first_epoch == 0 and not args.resume_training:
                        is_warm_up = True
                        input_video_latents = gt_video_latents.clone()

                    input_video_latents = torch.cat([input_video_latents, mask_feature], dim=1) # concatenate on channel
                    # if accelerator.is_main_process:
                    #     print(f"input_video_latents's shape is : {input_video_latents.shape}, and warm_up: {is_warm_up}") 
                    
                    del gt_video
                    del input_noisy
                    torch.cuda.empty_cache()

                    video_latents = gt_video_latents.permute(0, 2, 1, 3, 4)
                    input_video_latents = input_video_latents.permute(0, 2, 1, 3, 4)

                dummy_prompt = ""
                prompt_embeds = compute_prompt_embeds(text_encoder=text_encoder, 
                                                    tokenizer=tokenizer,
                                                    prompt=dummy_prompt , 
                                                    device=accelerator.device, 
                                                    dtype=weight_dtype,
                                                    num_videos_per_prompt=gt_video_latents.shape[0])

        
                models_to_accumulate = [transformer]
                # print("----------start accelerator.accumulate --------------")
                with accelerator.accumulate(models_to_accumulate): 
                    video_latents = video_latents.to(dtype=weight_dtype)  # [B, F, C, H, W]
                    image_latents = input_video_latents.to(dtype=weight_dtype)  # [B, F, C, H, W]
                    batch_size, num_frames, num_channels, height, width = video_latents.shape 
                    
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, scheduler.config.num_train_timesteps, (batch_size,), device=video_latents.device
                    )
                    timesteps = timesteps.long()

                    # Sample noise that will be added to the latents
                    noise = torch.randn_like(video_latents)

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)
                    noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)

                    # Prepare rotary embeds
                    image_rotary_emb = ( 
                        prepare_rotary_positional_embeddings(
                            height=480,
                            width=720,
                            num_frames=num_frames,
                            vae_scale_factor_spatial=vae_scale_factor_spatial,
                            patch_size=model_config.patch_size,
                            attention_head_dim=model_config.attention_head_dim,
                            device=accelerator.device,
                        )
                        if model_config.use_rotary_positional_embeddings
                        else None
                    )
        
                    model_output = transformer(
                        hidden_states=noisy_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timesteps,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                    )[0]
                    model_pred = scheduler.get_velocity(model_output, noisy_video_latents, timesteps)

                    alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                    weights = 1 / (1 - alphas_cumprod)
                    while len(weights.shape) < len(model_pred.shape):
                        weights = weights.unsqueeze(-1)

                    target = video_latents

                    loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
                    loss = loss.mean()
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        params_to_clip = transformer.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    logs = {"loss": loss.detach().item(), 
                            "lr": lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step}

                    if accelerator.sync_gradients:
                        global_step += 1
                        if accelerator.is_main_process:
                            print(f"--------- Current session's step: {step}, epoch: {epoch}, loss: {loss.item()}, global_step: {global_step} ---------")

                    # accelerator.log(logs, step=global_step) 
                    if accelerator.is_main_process:
                        wandb.log(logs)

                    if accelerator.sync_gradients:
                        if global_step % args.checkpointing_steps == 0:
                            state_dict = {
                                "global_step": global_step,
                                "epoch": epoch,
                                "step": step
                            }
                            save_checkpoint(state_dict, 
                                            args.output_dir, 
                                            accelerator, 
                                            max_to_keep=args.max_to_keep)

    # Finish WandB logging on the main process
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    infer()
