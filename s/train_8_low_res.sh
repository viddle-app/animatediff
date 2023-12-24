python train_8_channel_low_res.py \
  --learning_rate "3e-7" \
  --scale_lr \
  --dataset_name "Ryan-sjtu/celebahq-caption" \
  --output_dir "output_restorer" \
  --validation_prompt "The mona lisa" \
  --seed 42 \
  --resolution 512 \
  --train_batch_size 12 \
  --num_train_epochs 10 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --checkpointing_steps 500 \
  --pretrained_model_name_or_path "/mnt/newdrive/models/v1-5" \
  --mixed_precision "no" \
  --image_column "image" \
  --caption_column "text" \
  --report_to "wandb" \
  --validation_steps 100 \
  --val_image_url './mona-lisa-1.jpg' \
  --conditioning_dropout_prob 0.5 \
  --enable_xformers_memory_efficient_attention

  # --dataset_config_name "2m_random_100k" \
