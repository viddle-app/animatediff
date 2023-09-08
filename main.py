import glob
import os
from pathlib import Path
import random
import sys
import uuid
use_type = 'overlapping'
if use_type == 'overlapping':
  from src.pipelines.pipeline_animatediff_overlapping import AnimationPipeline
elif use_type == 'overlapping_2':
  from src.pipelines.pipeline_animatediff_overlapping_2 import AnimationPipeline
elif use_type == 'reference':
  from src.pipelines.pipeline_animatediff_reference import AnimationPipeline
else:
  from src.pipelines.pipeline_animatediff import AnimationPipeline
from src.pipelines.pipeline_animatediff_controlnet import StableDiffusionControlNetPipeline
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, DDIMScheduler, StableDiffusionPipeline
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from src.models.unet import UNet3DConditionModel
import numpy as np
import cv2
from diffusers.models.controlnet import ControlNetModel
from PIL import Image
from src.utils.image_utils import create_gif, create_mp4_from_images, tensor_to_image_sequence

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

def run(model,
        prompt="", 
        negative_prompt="", 
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
        lora_file=None,
        seed=None):
  scheduler_kwargs = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "linear",
  }

  device = "cuda" if torch.cuda.is_available() else "cpu"

  unet_additional_kwargs = {
    "unet_use_cross_frame_attention": False,
    "unet_use_temporal_attention": False,
    "use_motion_module": True,
    "motion_module_resolutions": [1, 2, 4, 8],
    "motion_module_mid_block": False,
    "motion_module_decoder_only": False,
    "motion_module_type": "Vanilla",
    "motion_module_kwargs": {
        "num_attention_heads": 8,
        "num_transformer_block": 1,
        "attention_block_types": ["Temporal_Self", "Temporal_Self"],
        "temporal_position_encoding": True,
        "temporal_position_encoding_max_len": 24,
        "temporal_attention_dim_div": 1,
    },
  }

  

  if use_single_file:
    pipeline = StableDiffusionPipeline.from_single_file(model)

    if lora_folder is not None and lora_file is not None:
      pipeline.load_lora_weights(lora_folder, 
                                 weight_name=lora_file)
      
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
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

  if use_controlnet:
    # controlnet_path = Path("../models/ControlNet-v1-1/control_v11p_sd15_openpose.yaml")
    controlnet_path = "lllyasviel/control_v11p_sd15_openpose"
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)

    pipeline = StableDiffusionControlNetPipeline(
      vae=vae, 
      text_encoder=text_encoder, 
      tokenizer=tokenizer, 
      unet=unet,
      scheduler=EulerAncestralDiscreteScheduler(**scheduler_kwargs),
      controlnet=controlnet,
      safety_checker=None,
      feature_extractor=None,
    ).to(device)
  else:

      pipeline = AnimationPipeline(
        vae=vae, 
        text_encoder=text_encoder, 
        tokenizer=tokenizer, 
        unet=unet,
        # scheduler=DDIMScheduler(**scheduler_kwargs),
        scheduler=EulerAncestralDiscreteScheduler(**scheduler_kwargs),
      ).to(device)

  # motion_module_path = "models/mm-baseline-epoch-5.pth"
  motion_module_path = "models/mm-Stabilized_high.pth"
  # motion_module_path = "models/mm-1000.pth"
  # motion_module_path = "models/motionModel_v03anime.ckpt"
  # motion_module_path = "models/mm_sd_v14.ckpt"
  # motion_module_path = "models/mm_sd_v15.ckpt"
  # motion_module_path = '../ComfyUI/custom_nodes/ComfyUI-AnimateDiff/models/animatediffMotion_v15.ckpt'
  motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
  missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
  if "global_step" in motion_module_state_dict:
    raise Exception("global_step present. Not sure how to handle that.")
  print("unexpected", unexpected)
  # assert len(unexpected) == 0
  print("missing", len(missing))
  print("missing", missing)

  if seed is None:
    seed = random.randint(-sys.maxsize, sys.maxsize)

  # generators = [torch.Generator().manual_seed(seed) for _ in range(window_count)]
  generators = torch.Generator().manual_seed(seed)

  if use_controlnet: 
    # load 16 frames from the directory
    open_pose_path = Path("../diffusers-tests/test-data/openpose-2")

    # get the directory files and sort them using Glob
    png_files = glob.glob(os.path.join(open_pose_path, '*.png'))
    
    # Sort the png files
    sorted_files = sorted(png_files)
    
    # get the first 16 frames
    sorted_files = sorted_files[:frame_count]

    # load the images as PIL images
    images = [Image.open(file) for file in sorted_files]


  
  
    video = pipeline(prompt=prompt, 
                  negative_prompt=negative_prompt, 
                  num_inference_steps=num_inference_steps,
                  guidance_scale=guidance_scale,
                  width=width,
                  height=height,
                  video_length=frame_count,
                  image=images,
                  output_type="pt",
                  ).images.permute(1, 0, 2, 3)

  
  else:
    if use_type == 'overlapping' or use_type == 'overlapping_2':
      video = pipeline(prompt=prompt, 
                    negative_prompt=negative_prompt, 
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                      width=width,
                      height=height,
                            window_count=window_count,
                      video_length=frame_count,
                      generator=generators).videos[0]
    else:
      video = pipeline(prompt=prompt, 
                    negative_prompt=negative_prompt, 
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                      width=width,
                      height=height,
                      video_length=frame_count).videos[0]
  
   
    # save the tensor 
    # torch.save(video, output_path + ".pt")

  # tensor_to_video(video, output_path,                     fps=frame_count)
  os.makedirs(output_dir, exist_ok=True)
  filename = str(uuid.uuid4())
  output_path = os.path.join(output_dir, filename + ".gif")
  fps = 15
  tensor_to_image_sequence(video, "images")
  images = glob.glob("images/*.png")
  images.sort()
  create_gif(images, output_path, duration=1000/fps, loop=0)
  create_mp4_from_images("images", output_path.replace(".gif", ".mp4"), fps=fps)

# prompt="neon glowing psychedelic man dancing, photography, award winning, gorgous, highly detailed",
# prompt="a retrowave painting of a tom cruise with a black beard, retrowave. city skyline far away, road, purple neon lights, low to the ground shot, (masterpiece,detailed,highres)"
# prompt="highly detail, Most Beautiful, extremely detailed book illustration of a woman fully clothed swaying, in the style of kay nielsen, dark white and light orange, apollinary vasnetsov, intricate embellishments, indonesian art, captivating gaze, by Kay Nielsen",

if __name__ == "__main__":
  prompt = "close up Girl posing red dress upper body, illustration by Wu guanzhong,China village,twojjbe trees in front of my chinese house,light orange, pink,white,blue ,8k"
  # prompt = "frazetta, man dancing, ed harris wearing a suit, dancing, mountain blue sky in background"
  # prompt = "neon glowing psychedelic man dancing, photography, award winning, gorgous, highly detailed"
  # prompt = "Cute Chunky anthropomorphic Siamese cat dressed in rags walking down a rural road, mindblowing illustration by Jean-Baptiste Monge + Emily Carr + Tsubasa Nakai"
  # prompt = "A man standing in front of an event horizon, 16k HDR, hyper realistic, cinematic, photography, designed by daniel arsham, glowing white, futuristic, detailed white liquid, directed by stanley kubrick"
  # prompt = "closeup of A woman dancing in front of a secret garden, early renaissance paintings, Rogier van der Weyden paintings style"
  # prompt = "close up portrait of a woman in front of a lake artwork by Kawase Hasui"
  # model = Path("../models/dreamshaper-6")
  # model = Path("../models/deliberate_v2")
  # lora_file="Frazetta.safetensors"
  lora_file=None
  model = "/mnt/newdrive/automatic1111/models/Stable-diffusion/dreamshaper_8.safetensors"
  run(model, 
      prompt=prompt,
      negative_prompt="clone, cloned, bad anatomy, wrong anatomy, mutated hands and fingers, mutation, mutated, amputation, 3d render, lowres, signs, memes, labels, text, error, mutant, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, made by children, caricature, ugly, boring, sketch, lacklustre, repetitive, cropped, (long neck), body horror, out of frame, mutilated, tiled, frame, border",
      height=512,
      width=512,
      frame_count=15,
      window_count=15,
      num_inference_steps=20,
      guidance_scale=7.0,
      lora_folder="/mnt/newdrive/automatic1111/models/Lora",
      lora_file=lora_file)

