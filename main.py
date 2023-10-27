import glob
import math
import os
from pathlib import Path
import random
import shutil
import sys
import uuid
import torch.nn.functional as F
# from src.pipelines.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from einops import rearrange

use_ufree = False
use_type = 'regular'
if use_type == 'overlapping':
  from src.pipelines.pipeline_animatediff_overlapping import AnimationPipeline
elif use_type == 'pix2pix_overlapping':
  from src.pipelines.pipeline_animatediff_overlapping_pix2pix import AnimationPipeline
elif use_type == 'conv':
  from src.pipelines.pipeline_animatediff_conv import AnimationPipeline
elif use_type == 'overlapping_noise_pred':
   from src.pipelines.pipeline_animatediff_overlapping_noise_pred import AnimationPipeline
elif use_type == 'overlapping_previous':
  from src.pipelines.pipeline_animatediff_overlapping_previous import AnimationPipeline
elif use_type == 'overlapping_previous_1':
  from src.pipelines.pipeline_animatediff_overlapping_previous_1 import AnimationPipeline
elif use_type == 'overlapping_previous_2':
  from src.pipelines.pipeline_animatediff_overlapping_previous_2 import AnimationPipeline
elif use_type == 'circular':
  from src.pipelines.pipeline_animatediff_circular import AnimationPipeline
elif use_type == 'overlapping_2':
  from src.pipelines.pipeline_animatediff_overlapping_2 import AnimationPipeline
elif use_type == 'overlapping_3':
  from src.pipelines.pipeline_animatediff_overlapping_3 import AnimationPipeline
elif use_type == 'overlapping_4':
  from src.pipelines.pipeline_animatediff_overlapping_4 import AnimationPipeline
elif use_type == 'reference':
  from src.pipelines.pipeline_animatediff_reference import AnimationPipeline
elif use_type == "init_image":
  from src.pipelines.pipeline_animatediff_init_image import AnimationPipeline
elif use_type == "ufree":
  from src.pipelines.pipeline_animatediff_ufree import AnimationPipeline
elif use_type == "pix2pix_2":
  from src.pipelines.pipeline_animatediff_pix2pix_2 import StableDiffusionInstructPix2PixPipeline
elif use_type == "init_image_2":
  from src.pipelines.pipeline_animatediff_init_image_2 import AnimationPipeline
else:
  from src.pipelines.pipeline_animatediff import AnimationPipeline
from src.pipelines.pipeline_animatediff_controlnet import StableDiffusionControlNetPipeline
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, DDIMScheduler, UniPCMultistepScheduler, StableDiffusionPipeline, HeunDiscreteScheduler
import torch
from transformers import CLIPTextModel, CLIPTokenizer
if use_type == "ufree" or use_ufree == True:
  from src.models.unet_ufree import UNet3DConditionModel
else:
  from src.models.unet import UNet3DConditionModel
import numpy as np
import cv2
from src.models.controlnet import ControlNetModel
from PIL import Image
from src.utils.image_utils import compute_spectrum, create_gif, create_mp4_from_images, save_spectrum_images, tensor_to_image_sequence
from diffusers.utils.torch_utils import randn_tensor
from torchvision.transforms import ToTensor
from safetensors.torch import load_file

image_render = True

if image_render:
   from src.pipelines.pipeline_stable_diffusion import StableDiffusionPipeline

print("use_type", use_type) 

class AttentionProcessorController:
    def __init__(self) -> None:
        self.attention_processors = []

    def add_attention_processor(self, attention_processor):
        self.attention_processors.append(attention_processor)

    def reset_last(self):
        for attention_processor in self.attention_processors:
            attention_processor.reset_last()

    def clear(self):
        for attention_processor in self.attention_processors:
            attention_processor.clear()

    def swap_next_to_last(self):
        for attention_processor in self.attention_processors:
            attention_processor.swap_next_to_last()

    def disable_cross_frame_attention(self):
        for attention_processor in self.attention_processors:
            attention_processor.disable_cross_frame_attention

    def enable_cross_frame_attention(self):
        for attention_processor in self.attention_processors:
            attention_processor.enable_cross_frame_attention

def make_progressive_latents(window_count, windows_length, height, width, 
                             alpha=0.5, dtype=torch.float16, generator=None,
                             device=torch.device("cpu")):
  # Assuming B, C, H, W, and alpha are given
  C, H, W = 4, height, width  # example values, you should replace with your own


  # Initialize the tensor for the first window
  epsilon_0 = randn_tensor((1, 4, 1, height, width), 
                               generator=generator, 
                              device=device, 
                              dtype=dtype)

  # Create a list to hold all frames, starting with the first frame
  windows = [epsilon_0]

  # Generate the rest of the frames
  for i in range(1, window_count*windows_length):  # now we are assuming the number of frames equals to batch size B
      # Generate independent noise for the current frame
      epsilon_ind = randn_tensor((1, 4, 1, height, width), 
                               generator=generator, 
                              device=device, 
                              dtype=dtype) / torch.sqrt(torch.tensor(1.0 + alpha**2))

      # Generate noise for the current frame based on the noise from the previous frame and the independent noise
      epsilon_i = (alpha / torch.sqrt(torch.tensor(1.0 + alpha**2))) * windows[i-1] + epsilon_ind

      # Add the current frame to the list of frames
      windows.append(epsilon_i)

  # Stack all frames along a new dimension (the batch dimension) to get a 5D tensor
  return torch.concat(windows, dim=2)

def set_upcast_softmax_to_false(obj):
    # If the object has the attribute 'upcast_softmax', set it to False
    if hasattr(obj, 'upcast_softmax'):
        obj.upcast_softmax = False

    # If the object is a list or a tuple, iterate over its elements
    if isinstance(obj, (list, tuple)):
        for item in obj:
            set_upcast_softmax_to_false(item)

    # If the object is a dictionary, iterate over its values
    elif isinstance(obj, dict):
        for item in obj.values():
            set_upcast_softmax_to_false(item)

    # If the object is a custom class object, iterate over its attributes
    elif hasattr(obj, '__dict__'):
        for attr in obj.__dict__.values():
            set_upcast_softmax_to_false(attr)

def make_preencoded_image_latents(pipe, init_latents, device, dtype, generator, timestep, noise_multiplier=1.0):
  shape = init_latents.shape
  print("shape", shape)
  # noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)'
  # noise = make_progressive_latents(shape[0], shape[2], shape[3], alpha=50)
  noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)  
  # get latents
  return pipe.scheduler.add_noise(init_latents, noise*noise_multiplier, timestep)

def get_timesteps(pipe, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = pipe.scheduler.timesteps[t_start * pipe.scheduler.order :]

    return timesteps, num_inference_steps - t_start

def encode_image_to_latent(pipe, image, generator, dtype):
    cuda_device = torch.device("cuda:0")
    image = image.to(device=cuda_device, dtype=dtype).unsqueeze(0)
    image = (2 * image - 1).clamp(-1, 1)

    # convert to a latent
    print("image", image.shape)
    the_latent = pipe.vae.encode(image).latent_dist.sample(generator).cpu()

    del image

    the_latent = pipe.vae.config.scaling_factor * the_latent

    return the_latent


def make_preencoded_latents(pipe,
                 num_inference_steps,
                 device,
                 dtype,
                 generator,
                 noise_image,
                 strength = 0.25,
                 noise_multiplier=1.0,
                 ):

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = get_timesteps(pipe,
                                                   num_inference_steps,
                                                   strength,
                                                   device)

    latent_timestep = timesteps[:1].repeat(noise_image.shape[0])
    print("latent_timestep", latent_timestep)

    print("noise_image", noise_image.shape)

    # noise_image=pipe.image_processor.preprocess(noise_image)
    return make_preencoded_image_latents(pipe=pipe,
                                init_latents=noise_image,
                                device=device,
                                dtype=dtype,
                                generator=generator,
                                timestep=latent_timestep,
                                noise_multiplier=noise_multiplier
                                ), timesteps, num_inference_steps



   


def upscale(pipeline, 
            video, 
            height, 
            width, 
            dtype, 
            seed, 
            frame_count, 
            num_inference_steps, 
            prompt, 
            negative_prompt, 
            guidance_scale,
            strength=0.25,):
    video = video.permute(1, 0, 2, 3)
        # upscale the video resolution 2x
    video = F.interpolate(video, size=(height, width), mode="bilinear")
    gen = torch.Generator().manual_seed(seed)
    
    encoded_frames = []
    for frame_index in range(video.shape[0]):
      frame = video[frame_index]
      with torch.no_grad():
        encoded_frames.append(encode_image_to_latent(pipeline, 
                                                  frame, 
                                                  generator=gen, 
                                                  dtype=dtype))
      torch.cuda.empty_cache()
    
    encoded_video = torch.concat(encoded_frames)
    print("encoded_video", encoded_video.shape)

    cpu_device = torch.device("cpu")

    latents, timesteps, _ = make_preencoded_latents(pipe=pipeline,
                            num_inference_steps=num_inference_steps,
                            device=cpu_device,
                            dtype=dtype,
                            generator=gen,
                            noise_image=encoded_video,
                            strength = 0.25,
                            noise_multiplier=1.0,
                            )
    latents = latents.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return pipeline(prompt=prompt, 
                negative_prompt=negative_prompt, 
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                  width=width,
                  height=height,
                  video_length=frame_count, 
                  latents=latents, 
                  timesteps=timesteps,
                  do_init_noise=False).videos[0]

def upscale_overlap(pipeline, 
            video, 
            height, 
            width, 
            dtype, 
            seed, 
            frame_count, 
            num_inference_steps, 
            prompt, 
            negative_prompt, 
            guidance_scale,
            strength=0.25,
            window_count=24,):
    video = video.permute(1, 0, 2, 3)
        # upscale the video resolution 2x
    video = F.interpolate(video, size=(height, width), mode="bilinear")
    gen = torch.Generator().manual_seed(seed)
    
    encoded_frames = []
    for frame_index in range(video.shape[0]):
      frame = video[frame_index]
      with torch.no_grad():
        encoded_frames.append(encode_image_to_latent(pipeline, 
                                                  frame, 
                                                  generator=gen, 
                                                  dtype=dtype))
      torch.cuda.empty_cache()
    
    encoded_video = torch.concat(encoded_frames)
    print("encoded_video", encoded_video.shape)

    cpu_device = torch.device("cpu")

    latents, timesteps, _ = make_preencoded_latents(pipe=pipeline,
                            num_inference_steps=num_inference_steps,
                            device=cpu_device,
                            dtype=dtype,
                            generator=gen,
                            noise_image=encoded_video,
                            strength = strength,
                            noise_multiplier=1.0,
                            )
    latents = latents.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return pipeline(prompt=prompt, 
                negative_prompt=negative_prompt, 
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                  width=width,
                  height=height,
                  video_length=frame_count, 
                  window_count=window_count,
                  latents=latents, 
                  timesteps=timesteps,
                  do_init_noise=False).videos[0]


def tensor_to_video(tensor, output_path, fps=30):
    """
    Convert a tensor of frames to a video.
    
    Parameters:
    - tensor: The tensor of frames with shape (num_frames, height, width, channels).
    - output_path: Path to save the output video.
    - fps: Frames per second for the output video.
    """
    tensor = tensor.cpu().permute(1, 2, 3, 0).numpy()

    # Get the shape details from the tensor
    num_frames, height, width, channels = tensor.shape
    print("num_frames", num_frames)
    print("height", height)
    print("width", width)
    print("channels", channels)
    # convert from 0 to 1 to 0 to 255
    tensor = tensor * 255

    # Ensure the tensor values are in the correct range [0, 255]
    tensor = np.clip(tensor, 0, 255).astype(np.uint8)

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = tensor[i]
        out.write(frame)

    out.release()

def compute_x_0(scheduler, timesteps, noisy_latents, prediction):
    
    if hasattr(scheduler, "sigmas"):
      # step_index = (scheduler.timesteps == timesteps).nonzero()[0].item()
      step_index = scheduler._step_index
      sigma = scheduler.sigmas[step_index]
      print("sigma", sigma)
      return noisy_latents - sigma * prediction
    else:

      alphas_cumprod = scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
      sqrt_alpha_prod = scheduler.alphas_cumprod[timesteps] ** 0.5
      sqrt_alpha_prod = sqrt_alpha_prod.flatten()
      while len(sqrt_alpha_prod.shape) < len(noisy_latents.shape):
          sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

      sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
      sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
      while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_latents.shape):
          sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

      sqrt_alpha_prod = sqrt_alpha_prod.to(device=noisy_latents.device)

      print("sqrt_alpha_prod", sqrt_alpha_prod)
      print("sqrt_one_minus_alpha_prod", sqrt_one_minus_alpha_prod)

      return (noisy_latents - prediction * sqrt_one_minus_alpha_prod ) / sqrt_alpha_prod  

def run(model,
        prompt="", 
        negative_prompt=None, 
        frame_count=24,
        num_inference_steps=20,
        guidance_scale=7.5,
        width=512,
        height=512,
        dtype=torch.float16,
        window_count=24,
        output_dir="output",
        use_single_file = True,
        lora_folder=None,
        lora_files=None,
        seed=None,
        last_n=21,
        unet_override=None,
        debug_latents=False,):
  scheduler_kwargs = {
   "num_train_timesteps": 1000,
   "beta_start": 0.00085,
   "beta_end": 0.012,
   "beta_schedule": "scaled_linear",
   # "clip_sample": False,
  }

  device = "cuda" if torch.cuda.is_available() else "cpu"

  unet_additional_kwargs = {
    "in_channels": 4,
    "unet_use_cross_frame_attention": False,
    "unet_use_temporal_attention": False,
    "use_motion_module": True,
    "motion_module_resolutions": [1, 2, 4, 8],
    "motion_module_mid_block": True,
    "motion_module_decoder_only": False,
    "motion_module_type": "Vanilla",
    "motion_module_kwargs": {
        "num_attention_heads": 8,
        "num_transformer_block": 1,
        "attention_block_types": ["Temporal_Self", "Temporal_Self"],
        "temporal_position_encoding": True,
        "temporal_position_encoding_max_len": 32,
        "temporal_attention_dim_div": 1,
        "zero_initialize" : False,
        "upcast_attention": True,

    },
  }

  

  if use_single_file:
    pipeline = StableDiffusionPipeline.from_single_file(model, torch_dtype=dtype)
    
    if lora_folder is not None:
      for lora_file in lora_files:
        pipeline.load_lora_weights(lora_folder, 
                                 weight_name=lora_file)
      
    tokenizer = pipeline.tokenizer  
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    if scheduler_kwargs is not None:
      scheduler = EulerAncestralDiscreteScheduler(**scheduler_kwargs)
    else:
      scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    if unet_override is not None:
      unet = UNet3DConditionModel.from_pretrained_2d(unet_override, 
                                                          subfolder="unet", 
                                                          unet_additional_kwargs=unet_additional_kwargs)

    else:
      unet = UNet3DConditionModel.from_unet2d(pipeline.unet, 
                                            unet_additional_kwargs=unet_additional_kwargs)

  else:
    tokenizer    = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer", torch_dtype=dtype)
    text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder", torch_dtype=dtype)
    vae          = AutoencoderKL.from_pretrained(model, subfolder="vae", torch_dtype=dtype)            
    unet         = UNet3DConditionModel.from_pretrained_2d(model, 
                                                          subfolder="unet", 
                                                          unet_additional_kwargs=unet_additional_kwargs)

  unet = unet.to(dtype=dtype) 

  use_controlnet = False
  # noise_scheduler=DDIMScheduler(**scheduler_kwargs)

  if (use_type == "overlapping_previous" or use_type == 'conv' or use_type == 'overlapping_previous_1' or use_type == 'overlapping_previous_2') and use_controlnet:
    # controlnet_path = Path("../models/ControlNet-v1-1/control_v11p_sd15_openpose.yaml")
    controlnet_path_0 = "lllyasviel/control_v11p_sd15_openpose"
    controlnet_path_1 = "lllyasviel/control_v11e_sd15_ip2p"
    # controlnet_path = "lllyasviel/control_v11f1e_sd15_tile"
    controlnet_0 = ControlNetModel.from_pretrained(controlnet_path_0, torch_dtype=dtype)
    controlnet_0 = controlnet_0.to(device=device)

    controlnet_1 = ControlNetModel.from_pretrained(controlnet_path_1, torch_dtype=dtype)
    controlnet_1 = controlnet_1.to(device=device)

    controlnet = MultiControlNetModel([controlnet_0, controlnet_1])
    print("controlnet", len(controlnet.nets))

    controlnet = controlnet_0

    print("before initialization")

    pipeline = AnimationPipeline(
      vae=vae, 
      text_encoder=text_encoder, 
      tokenizer=tokenizer, 
      unet=unet,
      scheduler=EulerAncestralDiscreteScheduler(**scheduler_kwargs),
      # scheduler=DDIMScheduler(pipe.scheduler.config),
      # scheduler=DDIMScheduler(**scheduler_kwargs),
      controlnet=controlnet,
    ).to(device)

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
  elif use_type == "pix2pix_2":
    pipeline = StableDiffusionInstructPix2PixPipeline(
      vae=vae, 
      text_encoder=text_encoder, 
      tokenizer=tokenizer, 
      unet=unet,
      scheduler=EulerAncestralDiscreteScheduler(**scheduler_kwargs),
      # scheduler=DDIMScheduler(**scheduler_kwargs),
      safety_checker=None,
      feature_extractor=None,
      requires_safety_checker=False,
    ).to(device)
  else:
      # if image_render:
         
      # else:
        pipeline = AnimationPipeline(
          vae=vae, 
          text_encoder=text_encoder, 
          tokenizer=tokenizer, 
          unet=unet,
          # scheduler=HeunDiscreteScheduler(**scheduler_kwargs),
          scheduler=scheduler,
          # scheduler=UniPCMultistepScheduler(**scheduler_kwargs),
        ).to(device)

       

  pipeline.enable_vae_slicing()
  pipeline.unet.eval()

      # set_upcast_softmax_to_false(pipeline)
      # pipeline.enable_sequential_cpu_offload()

  # motion_module_path = "motion-models/temporaldiff-v1-animatediff.ckpt"
  # motion_module_path = "models/mm-baseline-epoch-5.pth"
  # motion_module_path = "motion-models/mm-Stabilized_high.pth"
  # motion_module_path = "models/checkpoint.ckpt"
  # motion_module_path = "motion-models/overlapping-1e-6-1-20000-steps.ckpt"
  # motion_module_path = "motion-models/overlapping-5e-7-1-40000-steps.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-28T02-06-28/checkpoints/checkpoint.ckpt"
  # motion_module_path = "models/overlapping-1e-5-100-steps.pth"
  # motion_module_path = "models/overlapping-1e-5-3-20000-steps.pth"
  # motion_module_path = "models/overlapping-1e-5-2-100-steps.pth"
  # motion_module_path = "models/overlap-2-1000-steps.pth"
  # motion_module_path = "models/base-100-steps.pth"
  # motion_module_path = "models/overlap-epoch-1.pth"
  # motion_module_path = "models/mm-1000.pth"
  # motion_module_path = "models/motionModel_v03anime.ckpt"
  # motion_module_path = "models/mm_sd_v14.ckpt"
  # motion_module_path = "models/mm_sd_v15.ckpt"
  # motion_module_path = "motion-models/temporal-attn-5e-7-1-5000-steps.ckpt"
  # motion_module_path = "motion-models/temporal-attn-5e-7-2-15000-steps.ckpt"
  # motion_module_path = "motion-models/temporal-attn-pe-5e-7-4-50000-steps.ckpt"
  # motion_module_path = "motion-models/temporal-overlap-attn-5e-7-2-15000-steps.ckpt"
  # motion_module_path = "motion-models/temporal-attn-5e-7-3-40000-steps.ckpt"
  # motion_module_path = "motion-models/overlapping-attn-5e-7-1-5000-steps.ckpt"
  # motion_module_path = "motion-models/temporal-overlap-attn-5e-7-2-15000-steps.ckpt"
  motion_module_path = "motion-models/mm_sd_v15_v2.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-26T03-17-19/checkpoints/checkpoint-epoch-1.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-26T03-17-19/mm.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-26T08-56-44/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-26T09-38-19/checkpoints/mm.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-25T15-22-21/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-25T14-32-27/checkpoints/checkpoint.ckpt"
  # motion_module_path = "motion-models/temporal-attn-negative-pe-1e-6-1-5000-steps.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-26T12-50-02/checkpoints/mm.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-26T14-32-35/checkpoints/mm.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-26T16-52-17/checkpoints/mm.ckpt"
  # motion_module_path = '../ComfyUI/custom_nodes/ComfyUI-AnimateDiff/models/animatediffMotion_v15.ckpt'
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-28T09-41-25/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-28T10-57-52/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-28T11-39-21/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-28T19-12-07/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-03T01-59-48/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-04T08-42-20/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-02T15-45-27/checkpoints/checkpoint-epoch-1.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-04T15-22-48/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-04T16-00-16/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-04T16-33-35/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-04T16-57-01/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-04T22-58-15/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-05T07-04-34/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-05T20-09-05/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-06T19-42-02/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-06T21-43-24/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-06T23-24-45/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-07T00-35-40/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-07T02-19-36/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-07T15-59-37/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-07T20-44-16/checkpoints/checkpoint.ckpt"
  # diff no special unet
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-08T02-47-37/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-08T09-42-25/checkpoints/checkpoint.ckpt"
  # no diff loss
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-08T15-33-03/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-08T22-25-25/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-09T14-51-47/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-09T15-19-35/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-09T15-50-05/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-09T16-20-14/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-09T20-11-53/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-10T01-03-36/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-10T08-23-30/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-10T08-47-36/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-10T09-23-28/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-10T09-45-17/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-10T10-47-18/checkpoints/checkpoint-epoch-1.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-10T21-30-43/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-10T22-32-48/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-10T23-27-25/checkpoints/checkpoint-epoch-1.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-11T15-10-52/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-11T19-53-19/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-12T00-58-07/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-12T01-49-42/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-12T15-25-47/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-12T21-35-54/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-13T01-14-56/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-13T03-04-19/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-13T22-49-22/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-14T11-03-53/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-14T14-26-54/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-15T16-15-38/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-16T00-22-57/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-16T00-54-25/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-16T02-00-44/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-16T10-18-52/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-16T11-56-35/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-17T00-09-45/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-17T00-52-56/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-17T11-22-06/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-17T22-57-34/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-18T06-58-55/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-19T15-46-06/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-19T19-53-19/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-21T15-05-53/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-21T22-28-02/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-22T12-22-42/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-22T17-14-57/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-23T10-15-05/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-23T17-19-54/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-24T00-15-29/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-25T02-06-38/checkpoints/checkpoint.ckpt"
  # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-25T16-22-30/checkpoints/checkpoint.ckpt"
  motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
  missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
  # if "global_step" in motion_module_state_dict:
  #  raise Exception("global_step present. Not sure how to handle that.")
  print("unexpected", unexpected)
  # assert len(unexpected) == 0
  print("missing", len(missing))
  print("missing", missing)

  if seed is None:
    seed = random.randint(-sys.maxsize, sys.maxsize)

  # generators = [torch.Generator().manual_seed(seed) for _ in range(frame_count)]
  # 
  if use_type == 'regular':
    generators = torch.Generator(device=device).manual_seed(seed)
  else:
    generators = torch.Generator().manual_seed(seed)

  do_upscale = False
  use_img2img = False

  combined_callback_images = []
  cpu_device = torch.device("cpu")
  def callback(step, timestep, latents, prediction, x_0=None):
    latents = latents[:, :, 0:1]
    prediction = prediction[:, :, 0:1]
    
    timestep = timestep.to(device=cpu_device).long()
    # save the latents and compute the x_o and save that too
    if x_0 is not None:
      print("has x_0")
      x_0 = x_0[:, :, 0:1]
    else:
      x_0 = compute_x_0(pipeline.scheduler, 
                        timestep, 
                        latents, 
                        prediction).to(dtype=dtype)
    
    x_0 = rearrange(x_0, 'b c f h w -> (b f) c h w')
    x_0 = 1 / vae.config.scaling_factor * x_0
    
    x_0 = vae.decode(x_0, return_dict=False)[0]
    # convert to PIL image
    x_0 = (x_0 / 2 + 0.5).clamp(0, 1)

    # print("latents.std", latents.std())
    # print("latents.mean", latents.mean())
    # print("prediction.std", prediction.std())
    # print("prediction.mean", prediction.mean())
          

    latents = 1 / vae.config.scaling_factor * latents
    latents = rearrange(latents, 'b c f h w -> (b f) c h w')
    latents = vae.decode(latents, return_dict=False)[0]
    # convert to PIL image
    latents = (latents / 2 + 0.5).clamp(0, 1)
    
    prediction = 1 / vae.config.scaling_factor * prediction
    prediction = rearrange(prediction, 'b c f h w -> (b f) c h w')
    prediction = vae.decode(prediction, return_dict=False)[0]
    # convert to PIL image
    prediction = (prediction / 2 + 0.5).clamp(0, 1)

    # concat the tensors along the W dimension
    combined = torch.cat([latents, 
                          prediction, 
                          x_0, 
                          ], dim=3)

    combined_callback_images.append(combined)

  if not debug_latents:
     callback = None


  if use_type == 'overlapping' or use_type == 'overlapping_noise_pred' or use_type == 'overlapping_2' or use_type == 'overlapping_3' or use_type == 'overlapping_4' :
    video = pipeline(prompt=prompt, 
                  negative_prompt=negative_prompt, 
                  num_inference_steps=num_inference_steps,
                  guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    window_count=window_count,
                    video_length=frame_count,
                    generator=generators,
                    shift_count = 12
                    # wrap_around=True,
                    # alternate_direction=True,
                    ).videos[0]
    
    if do_upscale:
        video = upscale_overlap(pipeline,
              video,
              height*2,
              width*2,
              dtype,
              seed,
              frame_count,
              num_inference_steps,
              prompt,
              negative_prompt,
              guidance_scale,
              strength=0.25,
              window_count=window_count//4,
              )
  elif use_type == 'overlapping_previous' or use_type == 'conv' or use_type == 'overlapping_previous_1' or use_type == 'overlapping_previous_2':

      
      if use_controlnet:
        # load 16 frames from the directory
        # open_pose_path = Path("/mnt/newdrive/stable-diffusion-docker/output/dwpose")
        images_0 = []
        
        background_frames_path = Path("/mnt/newdrive/stable-diffusion-docker/output/dwpose")
        background_frames = glob.glob(os.path.join(background_frames_path, '*.png'))
        background_frames = sorted(background_frames)
        # get frame_count with a stride of 2
        background_frames = background_frames[::4]
        background_frames = background_frames[:frame_count]
        for frame_path in background_frames:
          frame = Image.open(frame_path).resize((width, height))
          images_0.append(frame)

        images_1 = []
        
        # background_frames_path = Path("/mnt/newdrive/stable-diffusion-docker/output/source_frames")
        background_frames_path = Path("/mnt/newdrive/stable-diffusion-docker/output/background_frames")
        background_frames = glob.glob(os.path.join(background_frames_path, '*.png'))
        background_frames = sorted(background_frames)
        # get frame_count with a stride of 2
        # background_frames = background_frames[::2]
        background_frames = background_frames[:frame_count]
        for frame_path in background_frames:
          frame = Image.open(frame_path).resize((width, height))
          images_1.append(frame)

        images = [images_0, images_1]

        images = images_0

        controlnet_conditioning_scale = 1.0
        guess_mode = False

        cpu_device = torch.device("cpu")

        # make a single frame of random noise and repeat it for the number of frames
        latent_generator = torch.Generator().manual_seed(seed)
        
        # latents = randn_tensor((1, 4, window_count, height // 8, width // 8), 
        #                       generator=latent_generator, 
        #                       device=cpu_device, 
        #                       dtype=dtype).repeat(1, 1, frame_count // window_count, 1, 1)

        if use_img2img:
          do_init_noise = False
          encoded_frames = []
            # load all the frames in the 
            # "/mnt/newdrive/stable-diffusion-docker/output/background_frames" folder
          background_frames_path = Path("/mnt/newdrive/stable-diffusion-docker/output/background_frames")
          background_frames = glob.glob(os.path.join(background_frames_path, '*.png'))
          background_frames = sorted(background_frames)
          background_frames = background_frames[:frame_count]
          for frame_path in background_frames:
          
            frame = Image.open(frame_path).resize((width, height))
            frame_tensor = pipeline.image_processor.preprocess(frame).to(device=device, dtype=dtype)
            print("frame_tensor", frame_tensor.shape)
            with torch.no_grad():
              encoded_frames.append(encode_image_to_latent(pipeline, 
                                                        frame_tensor.squeeze(0), 
                                                        generator=generators, 
                                                        dtype=dtype))
            torch.cuda.empty_cache()
          
          encoded_video = torch.concat(encoded_frames)
          print("encoded_video", encoded_video.shape)

          cpu_device = torch.device("cpu")

          latents, timesteps, _ = make_preencoded_latents(pipe=pipeline,
                                  num_inference_steps=num_inference_steps,
                                  device=cpu_device,
                                  dtype=dtype,
                                  generator=generators,
                                  noise_image=encoded_video,
                                  strength = 0.99,
                                  noise_multiplier=1.0,
                                  )
          latents = latents.unsqueeze(0).permute(0, 2, 1, 3, 4)
          print("latents", latents.shape)

        else:
          do_init_noise = True
          timesteps = None
          # latents = make_progressive_latents(frame_count // window_count,
          #                                  window_count, 
          #                                 height // 8, 
          #                                 width // 8, 
          #                                 alpha=0.0,
          #                                 dtype=dtype,
          #                                 generator=latent_generator,
          #                                 device=cpu_device,)

          # latents = randn_tensor((1, 4, 
          #                        window_count, height // 8, width // 8), generator=generators, dtype=dtype).repeat(1, 1, frame_count // window_count, 1, 1)
          latents = randn_tensor((1, 4, 
                                  frame_count, 
                                  height // 8, 
                                  width // 8), generator=generators, dtype=dtype)

        video = pipeline(prompt=prompt, 
              negative_prompt=negative_prompt, 
              num_inference_steps=num_inference_steps,
              guidance_scale=guidance_scale,
              width=width,
              height=height,
              window_count=window_count,
              video_length=frame_count,
              generator=generators,
              wrap_around=False,
              alternate_direction=False,
              guess_mode=guess_mode,
              image=images,
              controlnet_conditioning_scale=[controlnet_conditioning_scale, 1.0],
              min_offset = 3,
              max_offset = 5,
              latents=latents,
              offset_generator=latent_generator,
              do_init_noise=do_init_noise,
              timesteps=timesteps,
              ).videos[0]

      else:
        
        video = pipeline(prompt=prompt, 
              negative_prompt=negative_prompt, 
              num_inference_steps=num_inference_steps,
              guidance_scale=guidance_scale,
                width=width,
                height=height,
                window_count=window_count,
                video_length=frame_count,
                generator=generators,
                wrap_around=False,
                alternate_direction=False,
                min_offset = 3,
                max_offset = 5,
                callback=callback,
                # cpu_device=torch.device("cuda"),
                ).videos[0]
      
      if do_upscale:
        video = upscale_overlap(pipeline,
              video,
              height*2,
              width*2,
              dtype,
              seed,
              frame_count,
              num_inference_steps,
              prompt,
              negative_prompt,
              guidance_scale,
              strength=0.25,
              window_count=window_count//4,
              )
        

  elif use_type == 'pix2pix_overlapping':
    images = []
    
    # background_frames_path = Path("/mnt/newdrive/stable-diffusion-docker/output/dwpose")
    background_frames_path = Path("/mnt/newdrive/stable-diffusion-docker/output/background_frames")
    background_frames = glob.glob(os.path.join(background_frames_path, '*.png'))
    background_frames = sorted(background_frames)
    # get frame_count with a stride of 2
    # background_frames = background_frames[::2]
    background_frames = background_frames[:frame_count]
    for frame_path in background_frames:
       frame = Image.open(frame_path).resize((width, height))
       images.append(frame)




    video = pipeline(prompt=prompt, 
      negative_prompt=negative_prompt, 
      num_inference_steps=num_inference_steps,
      guidance_scale=guidance_scale,
        width=width,
        height=height,
        window_count=window_count,
        video_length=frame_count,
        generator=generators,
        wrap_around=True,
        alternate_direction=False,
        min_offset = 3,
        max_offset = 5,
        image_guidance_scale=0.9,
        pix2pix_image=images,
        # cpu_device=torch.device("cuda"),
        ).videos[0]

  elif use_type == "init_image":
    init_image = Image.open("mona-lisa-1.jpg")

    # convert the Image to a tensor
    generator = torch.Generator().manual_seed(seed)
    init_image = pipeline.image_processor.preprocess(init_image).to(device=device, dtype=dtype)
    print("init_image", init_image.shape)
    encoded_image = pipeline.vae.encode(init_image).latent_dist.sample(generator).cpu()
    encoded_image = pipeline.vae.config.scaling_factor * encoded_image

    encoded_video = encoded_image.repeat(frame_count, 1, 1, 1)
    print("encoded_video", encoded_video.shape)

    cpu_device = torch.device("cpu")

    latents, timesteps, _ = make_preencoded_latents(pipe=pipeline,
                            num_inference_steps=num_inference_steps,
                            device=cpu_device,
                            dtype=dtype,
                            generator=generator,
                            noise_image=encoded_video,
                            strength = 0.50,
                            noise_multiplier=1.0,
                            )
    print("time steps", len(timesteps))
    latents = latents.unsqueeze(0).permute(0, 2, 1, 3, 4)

    video = pipeline(prompt=prompt,
                      negative_prompt=negative_prompt,
                      num_inference_steps=num_inference_steps,
                      guidance_scale=guidance_scale,
                      width=width,
                      height=height,
                      video_length=frame_count,
                      init_image=init_image, 
                      latents=latents,
                      timesteps=timesteps,
                      do_init_noise=False,
                      ).videos[0]
  elif use_type == 'circular':
    # load a init image 
    
    # init_image = Image.open("0000.png")
    init_image = Image.open("byct.jpg")
    generator = torch.Generator().manual_seed(seed)
    init_image = pipeline.image_processor.preprocess(init_image).to(device=device, dtype=dtype)
    print("init_image", init_image.shape)
    encoded_image = pipeline.vae.encode(init_image).latent_dist.sample(generator).cpu()
    encoded_image = pipeline.vae.config.scaling_factor * encoded_image

    encoded_video = encoded_image.repeat(frame_count, 1, 1, 1)
    print("encoded_video", encoded_video.shape)

    cpu_device = torch.device("cpu")

    latents, timesteps, _ = make_preencoded_latents(pipe=pipeline,
                            num_inference_steps=num_inference_steps,
                            device=cpu_device,
                            dtype=dtype,
                            generator=generator,
                            noise_image=encoded_video,
                            strength = 0.65,
                            noise_multiplier=1.0,
                            )
    print("time steps", len(timesteps))
    latents = latents.unsqueeze(0).permute(0, 2, 1, 3, 4)

    
    video = pipeline(prompt=prompt, 
          negative_prompt=negative_prompt, 
          num_inference_steps=num_inference_steps,
          guidance_scale=guidance_scale,
            width=width,
            height=height,
            video_length=frame_count,
            start_image = init_image,
            latents=latents,
            timesteps=timesteps,
            do_init_noise=False).videos[0]
    
    if do_upscale:

      # save the video 
      # torch.save(video, "video.pt")
      # video = torch.load("video.pt")

      
      # upscale the video resolution 2x
      video = upscale(pipeline,
              video,
              height*2,
              width*2,
              dtype,
              seed,
              frame_count,
              num_inference_steps,
              prompt,
              negative_prompt,
              guidance_scale,
              strength=0.75,
              )
      
      # video = upscale(pipeline,
      #         video,
      #         height*4,
      #         width*4,
      #         dtype,
      #         seed,
      #         frame_count,
      #         num_inference_steps,
      #         prompt,
      #         negative_prompt,
      #         guidance_scale,
      #         strength=0.25,
      #         )


                        
    
  elif use_type == 'reference':
    video = pipeline(prompt=prompt, 
            negative_prompt=negative_prompt, 
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
              width=width,
              height=height,
              last_n=last_n,
              window_size=window_count,
              video_length=frame_count).videos[0]
  elif use_type == 'pix2pix_2':
    # resize the image to the correct size
    # image = Image.open("mona-lisa-1.jpg")
    # image = Image.open("biden.jpg")
    # image = Image.open("byct.jpg")
    # image = image.resize((width, height))

    images = []
    
    background_frames_path = Path("/mnt/newdrive/stable-diffusion-docker/output/source_frames")
    background_frames = glob.glob(os.path.join(background_frames_path, '*.png'))
    background_frames = sorted(background_frames)
    # get frame_count with a stride of 2
    background_frames = background_frames[::2]
    background_frames = background_frames[:frame_count]
    for frame_path in background_frames:
       frame = Image.open(frame_path).resize((width, height))
       images.append(frame)

    # make a 
    def f(x):
      return 2.0 - 4*torch.exp(-1 / (1 - (x - 1) ** 2))

    def sample_function(frame_count):
        x_values = torch.linspace(0, 2, frame_count)
        y_values = f(x_values)
        return y_values


    y_values_list = sample_function(frame_count).to(device=device, dtype=dtype)
    print("y_values_list", y_values_list)
    video = pipeline(prompt=prompt,
            # negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image=images,
            video_length=frame_count,
            image_guidance_scale=0.9,
            generator=generators,
            # image_guidance_scale=y_values_list,
              # width=width,
              # height=height,
              # window_size=window_count,
            
              )[0]
    # convert the list of PIL.Image images to a tensor

    # Convert each PIL.Image to a numpy array
    transform = ToTensor()
    tensors = [transform(img) for img in video]

# Stack these tensors together
    video = torch.stack(tensors, dim=0)
    print("video", video.shape)

    video = video.permute(1, 0, 2, 3)

  elif use_type == "init_image_2": 
    image = Image.open("mona-lisa-1.jpg")
    image = image.resize((width, height))

    image_weights = torch.linspace(1.0, 0.0, frame_count).to(device=device, dtype=dtype)
    # start from one and divide by two each time
    image_weights_list = []
    for i in range(frame_count):
      image_weights_list.append(1.0 / (2 ** i))
    image_weights = torch.tensor(image_weights_list).to(device=device, dtype=dtype)
    print("image_weights", image_weights.dtype)
    print("image_weights", image_weights)

    # image_weights = torch.ones(frame_count).to(device=device, dtype=dtype)

    video = pipeline(prompt=prompt, 
                  negative_prompt=negative_prompt, 
                  num_inference_steps=num_inference_steps,
                  guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator= generators,
                    video_length=frame_count,
                    image=image,
                    image_weights=image_weights,
                    noise_scheduler=noise_scheduler).videos[0]
  else:
    video = pipeline(prompt=prompt, 
                  negative_prompt=negative_prompt, 
                  num_inference_steps=num_inference_steps,
                  guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator= generators,
                    video_length=frame_count,
                    callback=callback).videos[0]
      


    # save the tensor 
    # torch.save(video, output_path + ".pt")
  print("video", video.shape)
  # tensor_to_video(video, output_path,                     fps=frame_count)
  # remove the images dir
  # check if the images dir exists
  if os.path.exists("images"): 
    shutil.rmtree("images")

  os.makedirs(output_dir, exist_ok=True)
  
  # if os.path.exists("spectrum-images"):
  #  shutil.rmtree("spectrum-images")

  os.makedirs("spectrum-images", exist_ok=True)

  filename = str(uuid.uuid4())
  output_path = os.path.join(output_dir, filename + ".gif")
  fps = 8
  images_folder = f"{output_dir}/{filename}"
  tensor_to_image_sequence(video, images_folder)
  images = glob.glob(f"{images_folder}/*.png")
  images.sort()
  create_gif(images, output_path, duration=1000/fps, loop=0)
  create_mp4_from_images(images_folder, output_path.replace(".gif", ".mp4"), fps=fps)
  spectrum = compute_spectrum(video)
  spectrum_uuid = str(uuid.uuid4())
  save_spectrum_images(spectrum, f"spectrum-images/{spectrum_uuid}")
  os.makedirs("latents", exist_ok=True)
  if len(combined_callback_images) > 0:
     combined = torch.cat(combined_callback_images, dim=2)
     # TODO save the images as GIFs and image sequences
     # output_path = f"{output_folder}/{step}.gif"
     # save_gif_from_tensor(combined, output_path)
     # save a folder of images
     image_folder_path = f"latents/{spectrum_uuid}"
     tensor_to_image_sequence(combined.permute(1, 0, 2, 3), image_folder_path)

  # TODO save the video spectrum

# prompt="neon glowing psychedelic man dancing, photography, award winning, gorgous, highly detailed",
# prompt="a retrowave painting of a tom cruise with a black beard, retrowave. city skyline far away, road, purple neon lights, low to the ground shot, (masterpiece,detailed,highres)"
# prompt="highly detail, Most Beautiful, extremely detailed book illustration of a woman fully clothed swaying, in the style of kay nielsen, dark white and light orange, apollinary vasnetsov, intricate embellishments, indonesian art, captivating gaze, by Kay Nielsen",

if __name__ == "__main__":
  # prompt = "photograph of a bald man laughing"
  # prompt = "photograph of a man scared"
  # prompt = "Make synthwave retrowave vaporware style, outside night time city skyline"
  # prompt = "turn her into princess leia, side buns hairstyle"
  # prompt = "make her open her mouth"
  prompt = "close up of a Girl swaying, red blouse, illustration by Wu guanzhong,China village,twojjbe trees in front of my chinese house,light orange, pink,white,blue ,8k"
  # prompt = "paint by frazetta, man dancing, mountain blue sky in background"
  # prompt = "1man dancing outside, clear blue sky sunny day, photography, award winning, highly detailed, bright, well lit"
  # prompt = "Cute Chunky anthropomorphic Siamese cat dressed in rags walking down a rural road, mindblowing illustration by Jean-Baptiste Monge + Emily Carr + Tsubasa Nakai"
  # prompt = "close up of A man walking, movie production, cinematic, photography, designed by daniel arsham, glowing white, futuristic, white liquid, highly detailed, 35mm"
  # prompt = "closeup of A woman dancing in front of a secret garden, upper body headshot, early renaissance paintings, Rogier van der Weyden paintings style"
  # prompt = "Close Up portrait of a woman turning in front of a lake artwork by Kawase Hasui"
  # prompt = "guy working at his computer"
  # prompt = "little boy plays with marbles"
  # prompt = "close up head shot of girl standing in a field of flowers, windy, long blonde hair in a blue dress smiling"
  # prompt = "RAW Photo, DSLR BREAK a young woman with bangs, (light smile:0.8), (smile:0.5), wearing relaxed shirt and trousers, causal clothes, (looking at viewer), focused, (modern and cozy office space), design agency office, spacious and open office, Scandinavian design space BREAK detailed, natural light"
  # prompt = "taken with iphone camera BREAK medium shot selfie of a pretty young woman BREAK (ombre:1.3) blonde pink BREAK film grain, medium quality"
  # prompt = "girl posing outside a bar on a rainy day, black clothes, street, neon sign, movie production still, high quality, neo-noir"
  # prompt = "cowboy shot, cyberpunk jacket, camera following woman walking and smoking down street on rainy night, led sign"
  # prompt = "A lego ninja bending down to pick a flower in the style of the lego movie. High quality render by arnold. Animal logic. 3D soft focus"
  prompt = "Glowing jellyfish, calm, slow hypnotic undulations, 35mm Nature photography, award winning"
  # prompt = "synthwave retrowave vaporware back of a delorean driving on highway, dmc rear grill, neon lights, palm trees and sunset in background, clear, high definition, sharp"
  # prompt = "dog walking in the flower, in the style of geometric graffiti"
  # prompt = "a doodle of a bear dancing, scribble, messy, stickfigure, badly drawn"
  # prompt = "make a painting by patrick nagel, gorgeous woman"
  # prompt = "make it oil painting, thick brush strokes, impasto, colorful"
  # prompt = "make it a charcoal sketch, minimal, simple line drawing, rough thick lines"
  # prompt = "woman laughing"
  # prompt = "close up of Embrodery Elijah Wood smiling in front of a embroidery landscape"
  # prompt = "tom cruise statue made of ice dancing empty black background"
  # prompt = "watermeloncarving, Terra Cotta Warriors,full body dancing"
  # prompt = "A doll walking in the forest jan svankmajer, brother quay style stop action animation"
  # model = Path("../models/dreamshaper-6")
  # model = Path("../models/deliberate_v2")
  # lora_file="Frazetta.safetensors"
  # lora_files=["retrowave_0.12.safetensors", "dmc12-000006.safetensors"]
  # lora
  # lora_files = ["doodle.safetensors"]
  # lora_files = ["kEmbroideryRev.safetensors"]
  # lora_files = ["made_of_ice.safetensors"]
  # lora_files = ["watermeloncarving-000004.safetensors"]
  lora_files=[]
  # TODO the multidiffusion successor probably has the answer for duration

  # lora_file=None
  # lora_file
  # model = "/mnt/newdrive/automatic1111/models/Stable-diffusion/dreamshaper_6BakedVae.safetensors"
  model = "/mnt/newdrive/automatic1111/models/Stable-diffusion/dreamshaper_8.safetensors"
  # model = "/mnt/newdrive/automatic1111/models/Stable-diffusion/epicrealism_pureEvolutionV5.safetensors"
  # model = "/mnt/newdrive/automatic1111/models/Stable-diffusion/instruct-pix2pix-00-22000.ckpt"
  # model = "/mnt/newdrive/automatic1111/models/Stable-diffusion/deliberate_v2.safetensors"
  # model = "/mnt/newdrive/automatic1111/models/Stable-diffusion/absolutereality_v181.safetensors"
  # model = "/mnt/newdrive/automatic1111/models/Stable-diffusion/neonskiesai_V10.safetensors"
  # model = "/mnt/newdrive/automatic1111/models/Stable-diffusion/synthwavepunk_v2.ckpt"
  # model = "/mnt/newdrive/models/miniSD"
  # model = "/mnt/newdrive/models/v1-5"
  # model = "/mnt/newdrive/viddle-animatediff/output_dreamshaper_8"
  run(model, 
      prompt=prompt,
      negative_prompt="compression artifacts, blurry, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, made by children, caricature, ugly, boring, sketch, lacklustre, repetitive, cropped, (long neck), body horror, out of frame, mutilated, tiled, frame, border",
      # negative_prompt="",
      height=512,
      width=512,
      frame_count=16, # 288,
      window_count=16,
      num_inference_steps=20,
      guidance_scale=7.0,
      last_n=23,
      seed=42,
      use_single_file=True,
      dtype=torch.float16,
      lora_folder="/mnt/newdrive/automatic1111/models/Lora",
      lora_files=lora_files,
      debug_latents=False,
      # unet_override="/mnt/newdrive/viddle-animatediff/output_dreamshaper_8/checkpoint-10000")
  )
