#!/bin/bash
data_dir="re10k"
output_dir="outputs/test_seq_diff"
weight_path="pretrained/model.safetensors"

python test.py \
    +experiment=re10k \
    mode=test \
    wandb.name=re10k \
    checkpointing.load=noposplat/pretrained_weights/re10k.ckpt \
    dataset.re10k.roots=[${data_dir}] \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=noposplat/assets/evaluation_index_157.json \
    dataset.re10k.make_baseline_1=false\
    dataset.re10k.relative_pose=false\
    test.save_image=true \
    test.save_video=true \
    test.output_path="${output_dir}" \
    cogvideox_cfg.synthesize_output=true\
    cogvideox_cfg.transformer_weight_path="${weight_path}"

