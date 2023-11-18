python train_unet.py \
  --learning_rate "1e-9" \
  --scale_lr \
  --dataset_name "laion_6plus" \
  --output_dir "output_dreamshaper_8_snr" \
  --validation_prompt "Close Up portrait of a woman turning in front of a lake artwork by Kawase Hasui" \
  --seed 42 \
  --resolution 512 \
  --train_batch_size 24 \
  --max_train_steps 800 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --checkpointing_steps 100 \
  --pretrained_model_name_or_path "Lykon/dreamshaper-8" \
  --input_perturbation 0.0 \
  --mixed_precision "bf16" \
  --image_column "URL" \
  --caption_column "TEXT" \
  --report_to "wandb" \
  --validation_steps 25 \
  --noise_offset 0.0 \
  --lr_scheduler "constant_with_warmup" \
  --lr_warmup_steps 10 \
  --snr_gamma 5.0 
  
  # --snr_scaling \
