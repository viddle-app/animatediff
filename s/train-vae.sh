python train_vae_2.py \
  --mixed_precision="bf16" \
  --pretrained_model_name_or_path="/mnt/newdrive/models/v1-5" \
  --dataset_name="video_dataset" \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --report_to="wandb" \
  --output_dir "output_vae" \
  --seed 42 \
  --learning_rate "1e-5" \
  --scale_lr \
  --image_column "video_path" \
  --resolution 128 \
  --resume_from_checkpoint "latest"
