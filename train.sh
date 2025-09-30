
accelerate launch --mixed_precision bf16 train.py \
  +experiment=re10k \
  mode=test \
  wandb.name=re10k \
  checkpointing.load=noposplat/pretrained_weights/re10k.ckpt \
  test.save_image=false \
  data_loader.train.batch_size=1 \
  cogvideox_cfg.output_dir="training_output" \
  cogvideox_cfg.resume_training=true \
  cogvideox_cfg.wandb_resume_id=xxx \
  cogvideox_cfg.max_to_keep=10 \
  cogvideox_cfg.checkpointing_steps=100