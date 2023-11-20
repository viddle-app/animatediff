# originally copied from: 
from collections import OrderedDict
import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess
import glob

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

from src.models.unet import UNet3DConditionModel
from src.utils.util import save_videos_grid, zero_rank_print
from src.data.dataset_cached_latents_and_mode import WebVid10M
# from src.data.dataset import WebVid10M
from torchvision.transforms import ToTensor
from safetensors.torch import load_file
import bitsandbytes as bnb
# from src.pipelines.pipeline_animatediff_init_image_3 import StableDiffusionInitImagePipeline
from src.pipelines.pipeline_animatediff_init_image_4 import StableDiffusionInitImagePipeline

from itertools import repeat

def frame_diff(video_tensor):
    """
    Compute the frame difference for a video tensor.
    video_tensor should have shape (batch_size, channels, frames, height, width)
    """
    return video_tensor[:, :, 1:] - video_tensor[:, :, :-1]

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def video_diff_loss_6(original_video, generated_video, first_frame_weight=1.0):
    # diff both from the first frame
    original_diff = frame_diff(original_video)
    # now compute the frame difference of the generated
    generated_diff = frame_diff(generated_video)

    # add the first frame of the original and the generated
    # to the front of the original_diff and generated_diff
    original_diff = torch.cat([original_video[:, :, 0:1], original_diff], dim=2)
    generated_diff = torch.cat([generated_video[:, :, 0:1], generated_diff], dim=2)
    # take the difference between the two
    difference_difference = original_diff - generated_diff

    # multiple the first frame difference by the weight
    difference_difference[:, :, 0] = difference_difference[:, :, 0] * first_frame_weight

    # take the mse loss
    return (difference_difference**2)

def extract_motion_module(unet):
    mm_state_dict = OrderedDict()
    state_dict = unet.state_dict()
    state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    for key in state_dict:
        if "motion_module" in key:
            mm_state_dict[key] = state_dict[key]

    return mm_state_dict

def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
        
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank

def add_positional_encoding(x, dim):
    """
    Add positional encoding to the tensor.
    For the purpose of this demonstration, we'll use a simple sinusoidal encoding.
    You can replace this with any positional encoding of your choice.
    """
    print("x.shape", x.shape)
    pos = torch.arange(0, x.size(2), dtype=torch.float32, device=x.device)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim)).to(x.device)
    pos = pos.unsqueeze(1)
    div_term = div_term.unsqueeze(0)
    encoded = pos * div_term
    sin_enc = torch.sin(encoded)
    cos_enc = torch.cos(encoded[:, 1::2])
    encoding = torch.stack([sin_enc, cos_enc], dim=2).flatten(1)
    
    # Expanding dimensions to match the input shape and adding to the input tensor
    encoding = encoding.unsqueeze(0).unsqueeze(3).unsqueeze(4)
    return x + encoding

def main(
    image_finetune: bool,
    
    name: str,
    use_wandb: bool,
    launcher: str,
    
    output_dir: str,
    pretrained_model_path: str,

    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 32,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,
    cached_latents: bool = False,
    joint_train = False,
    motion_module_path=None,
    use_8bit_adam = False,
):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher)
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)
    
    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    dtype = torch.float32
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=dtype)
  
    tokenizer = pipeline.tokenizer  
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    fine_tuned_unet = "/mnt/newdrive/viddle-animatediff/output_8_channel/checkpoint-800/unet/"
    unet = UNet3DConditionModel.from_pretrained_2d(fine_tuned_unet, unet_additional_kwargs=unet_additional_kwargs)
    
    global_step = 0
    if not unet_additional_kwargs.motion_module_kwargs.zero_initialize:
        print("motion_module_path", motion_module_path)
        motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
        if "global_step" in motion_module_state_dict:
            global_step = int(motion_module_state_dict["global_step"])
            motion_module_state_dict = motion_module_state_dict["state_dict"]
        # update_pe(motion_module_state_dict)
        missing, unexpected = unet.load_state_dict(motion_module_state_dict, strict=False)
        
        print("unexpected", unexpected)
        # assert len(unexpected) == 0
        print("missing", len(missing))
        print("missing", missing)

        print(f"state_dict keys: {list(unet.config.keys())[:10]}")


    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        zero_rank_print(f"from checkpoint: {unet_checkpoint_path}")
        unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
        if "global_step" in unet_checkpoint_path: zero_rank_print(f"global_step: {unet_checkpoint_path['global_step']}")
        state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path
        print(f"state_dict keys: {list(state_dict.keys())[:10]}")
        state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
        m, u = unet.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {m[:10]}")
        # zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
        
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Set unet trainable parameters
    if joint_train:
        unet.requires_grad_(True)
    else:
        unet.requires_grad_(False)
        for name, param in unet.named_parameters():
            for trainable_module_name in trainable_modules:
                if trainable_module_name in name:
                    param.requires_grad = True
                    break

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if use_8bit_adam:
        optimizer_cls = bnb.optim.AdamW8bit
    else: 
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if is_main_process:
        zero_rank_print(f"trainable params number: {len(trainable_params)}")
        zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)


    # Get the training dataset
    train_dataset = WebVid10M(**train_data, is_image=image_finetune)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)



    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    if not image_finetune:
        validation_pipeline = StableDiffusionInitImagePipeline(
            unet=unet, 
            vae=vae, 
            tokenizer=tokenizer, 
            text_encoder=text_encoder, 
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to("cuda")
    else:
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path,
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
        )
    validation_pipeline.enable_vae_slicing()

    # DDP warpper
    unet.to(local_rank)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    actual_step = 0

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()
        
        for step, batch in enumerate(train_dataloader):
            if actual_step < global_step:
                print("skip step", actual_step)
                actual_step += 1
                continue

            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]
            
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                if not image_finetune:
                    if cached_latents:
                        frame_count = pixel_values.shape[1]
                        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w").to(device=local_rank)
                        # decode using the vae
                        with torch.no_grad():
                            pixel_values = pixel_values / 0.18125
                            pixel_values = vae.decode(pixel_values, return_dict=False)[0]
                            pixel_values = rearrange(pixel_values, "(b f) c h w -> b c f h w", f=frame_count).to(device="cpu", dtype=torch.float32)
                    else: 
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.gif", rescale=True)
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value / 2. + 0.5
                        torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.png")

            # Convert videos to latent space            
            latents = batch["pixel_values"].to(local_rank)
            mode_latents = batch["mode_values"].to(local_rank)
            if not image_finetune:
                latents = rearrange(latents, "b f c h w -> b c f h w")
                mode_latents = rearrange(mode_latents, "b f c h w -> b c f h w")
                video_length = latents.shape[2]
                print("video_length: ", video_length)

            print("latents.shape: ", latents.shape)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # repeat the mode_latents first frame video_length times to match the shape of noisy_latents
            first_frame = mode_latents[:, :, 0:1, :, :]

            # Repeat the first frame 16 times along the frame dimension
            image_latents = first_frame.repeat(1, 1, 16, 1, 1)
        
            # zero likes the latents without the first frame
            # zero_latents = torch.zeros_like(mode_latents[:, :, 1:, :, :])

            # image_latents = torch.cat([first_frame, zero_latents], dim=2)

            # Concatenate the encoded frames to the noisy_latents
            noisy_latents = torch.cat([noisy_latents, image_latents], dim=1)


            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]
                
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                model_pred_float = model_pred.float()
                target_float = target.float()
                # snr = compute_snr(noise_scheduler, timesteps)
                # snr_scale = 1 / torch.sqrt(snr)
                # loss = (snr_scale.reshape(-1, 1, 1, 1, 1) * 
                #         video_diff_loss_6(model_pred_float, 
                #                             target_float, 
                #                             first_frame_weight=1.0)).mean()
                # loss = video_diff_loss_6(model_pred_float, 
                #                            target_float, 
                #                            first_frame_weight=1.0).mean()
                loss = F.mse_loss(model_pred_float, target_float, reduction="mean")

            if mixed_precision_training:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Backpropagate
            if mixed_precision_training:
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            optimizer.zero_grad()
            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
                
            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataloader) - 1):
                save_path = os.path.join(output_dir, f"checkpoints")
                if joint_train:
                    motion_module = unet.state_dict()
                else:
                    motion_module = extract_motion_module(unet)
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": motion_module,
                }
                if step == len(train_dataloader) - 1:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                else:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                
            # Periodically validation
            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                samples = []
                
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)
                
                height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

                prompts = validation_data.prompts[:2] if global_step < 1000 and (not image_finetune) else validation_data.prompts

                for idx, prompt in enumerate(prompts):
                    if not image_finetune:
                        images = []
                        width = 256
                        height = 256
                        
                        background_frames_path = Path("/mnt/newdrive/viddle-animatediff/output/1c0c0f1e-73b4-417a-91c0-aaa7356e0cf0")
                        background_frames = glob.glob(os.path.join(background_frames_path, '*.png'))
                        background_frames = sorted(background_frames)
                        # get frame_count with a stride of 2
                        background_frames = background_frames[0]
                    
                        frame = Image.open(background_frames).resize((width, height))
                        
                        images = frame

                        sample = validation_pipeline(
                            prompt,
                            generator    = generator,
                            video_length = train_data.sample_n_frames,
                            image = images,
                            num_inference_steps = validation_data.get("num_inference_steps", 25),
                        )[0]

                        transform = ToTensor()
                        tensors = [transform(img) for img in sample]

                        # Stack these tensors together
                        sample = torch.stack(tensors, dim=0)
                        print("video", sample.shape)

                        sample = sample.permute(1, 0, 2, 3).unsqueeze(0)
                        output_path = f"{output_dir}/samples/sample-{global_step}/{idx}.gif"
                        save_videos_grid(sample, output_path)
                        samples.append(sample)
                        # Assuming 'images' is a list of PIL Image objects or numpy arrays
                        # and 'labels' is a list of corresponding labels or captions  
                        wandb.log({"validation_images": wandb.Image(output_path)})

                        
                    else:
                        sample = validation_pipeline(
                            prompt,
                            generator           = generator,
                            height              = height,
                            width               = width,
                            num_inference_steps = validation_data.get("num_inference_steps", 25),
                            guidance_scale      = validation_data.get("guidance_scale", 8.),
                        ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        samples.append(sample)
                
                if not image_finetune:
                    samples = torch.concat(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                    save_videos_grid(samples, save_path)
                    
                else:
                    samples = torch.stack(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.png"
                    torchvision.utils.save_image(samples, save_path, nrow=4)

                logging.info(f"Saved samples to {save_path}")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
        
            actual_step +=1
            
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb",    action="store_true")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)