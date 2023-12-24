python train_lcm.py \
  --proportion_empty_prompts 0.1 \
  --max_grad_norm 1.0 \
  --pretrained_teacher_model /mnt/newdrive/models/v1-5 \
  --mixed_precision=fp16 \
  --resolution=256 \
  --lora_rank=64 \
  --learning_rate=1e-6 \
  --loss_type="huber" \
  --adam_weight_decay=0.0 \
  --max_train_steps=1000 \
  --max_train_samples=4000000 \
  --dataloader_num_workers=8 \
  --validation_steps=200 \
  --checkpointing_steps=200 --checkpoints_total_limit=10 \
  --train_batch_size=12 \
  --gradient_checkpointing --enable_xformers_memory_efficient_attention \
  --gradient_accumulation_steps=1 \
  --use_8bit_adam \
  --resume_from_checkpoint=latest \
  --report_to=wandb \
  --seed=453645634 


