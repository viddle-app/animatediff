import glob
import os
from pathlib import Path
use_type = 'overlapping'
if use_type == 'overlapping':
  from src.pipelines.pipeline_animatediff_overlapping import AnimationPipeline
elif use_type == 'reference':
  from src.pipelines.pipeline_animatediff_reference import AnimationPipeline
else:
  from src.pipelines.pipeline_animatediff import AnimationPipeline
from src.pipelines.pipeline_animatediff_controlnet import StableDiffusionControlNetPipeline
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, DDIMScheduler
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from src.models.unet import UNet3DConditionModel
import numpy as np
import cv2
from diffusers.models.controlnet import ControlNetModel
from PIL import Image
from src.utils.image_utils import create_gif, tensor_to_image_sequence

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
        output_path="output.gif"):
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
  # motion_module_path = "models/mm-Stabilized_high.pth"
  # motion_module_path = "models/mm-1000.pth"
  # motion_module_path = "models/motionModel_v03anime.ckpt"
  # motion_module_path = "models/mm_sd_v14.ckpt"
  motion_module_path = "models/mm_sd_v15.ckpt"
  # motion_module_path = '../ComfyUI/custom_nodes/ComfyUI-AnimateDiff/models/animatediffMotion_v15.ckpt'
  motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
  missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
  if "global_step" in motion_module_state_dict:
    raise Exception("global_step present. Not sure how to handle that.")
  print("unexpected", unexpected)
  assert len(unexpected) == 0
  print("missing", len(missing))

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
    video = pipeline(prompt=prompt, 
                  negative_prompt=negative_prompt, 
                  num_inference_steps=num_inference_steps,
                  guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    video_length=frame_count).videos[0]
  
  
    # save the tensor 
    # torch.save(video, output_path + ".pt")

  # tensor_to_video(video, output_path, fps=frame_count)
  fps = 24
  tensor_to_image_sequence(video, "images")
  images = glob.glob("images/*.png")
  images.sort()
  create_gif(images, output_path, duration=1000/fps, loop=0)

# prompt="neon glowing psychedelic man dancing, photography, award winning, gorgous, highly detailed",

if __name__ == "__main__":
  run(Path("../models/dreamshaper-6"), 
      prompt="boris karloff",
      negative_prompt="ugly, blurry, wrong",
      height=512,
      width=512,
      frame_count=96,
      num_inference_steps=40)

