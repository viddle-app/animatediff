import glob
import math
import os
from pathlib import Path
import random
import sys
import uuid
use_type = 'overlapping_3'
if use_type == 'overlapping':
  from src.pipelines.pipeline_animatediff_overlapping import AnimationPipeline
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
else:
  from src.pipelines.pipeline_animatediff import AnimationPipeline
from src.pipelines.pipeline_animatediff_controlnet import StableDiffusionControlNetPipeline
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, DDIMScheduler, StableDiffusionPipeline
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from src.models.unet import UNet3DConditionModel
import numpy as np
import cv2
from src.models.controlnet import ControlNetModel
from PIL import Image
from src.utils.image_utils import create_gif, create_mp4_from_images, tensor_to_image_sequence

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
        lora_files=None,
        seed=None,
        last_n=21):
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
    pipeline = StableDiffusionPipeline.from_single_file(model, torch_dtype=dtype)
    
    if lora_folder is not None:
      for lora_file in lora_files:
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
    # controlnet_path = "lllyasviel/control_v11p_sd15_openpose"
    controlnet_path = "lllyasviel/control_v11f1e_sd15_tile"
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
        scheduler=DDIMScheduler(**scheduler_kwargs),
        # scheduler=EulerAncestralDiscreteScheduler(**scheduler_kwargs),
      ).to(device)

      # set_upcast_softmax_to_false(pipeline)
      # pipeline.enable_sequential_cpu_offload()

  # motion_module_path = "models/temporaldiff-v1-animatediff.ckpt"
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

  generators = [torch.Generator().manual_seed(seed) for _ in range(window_count)]
  # generators = torch.Generator().manual_seed(seed)

  if use_controlnet: 
    # load 16 frames from the directory
    # open_pose_path = Path("../diffusers-tests/test-data/openpose-2")

    # get the directory files and sort them using Glob
    # png_files = glob.glob(os.path.join(open_pose_path, '*.png'))
    
    # Sort the png files
    # sorted_files = sorted(png_files)
    
    # get the first 16 frames
    # sorted_files = sorted_files[:frame_count]

    # load the 0000.png and duplicate it 24 times
    # sorted_files = ["0000.png"] * frame_count
    sorted_files = ["byct.jpg"] * frame_count

    # load the images as PIL images
    images = [Image.open(file) for file in sorted_files]

    # create a list of values from 0 to 2pi with frame_count values
    
    x = np.linspace(-np.sqrt(2), np.sqrt(2), frame_count)
    print("x", x)
    
    k = 1

    # Compute the function values for each x
    controlnet_conditioning_scale = torch.tensor(1 - 1 * np.exp(-k * x**2))
    controlnet_conditioning_scale[0] = 1.0
    guess_mode = False
    
    video = pipeline(prompt=prompt, 
                  negative_prompt=negative_prompt, 
                  num_inference_steps=num_inference_steps,
                  guidance_scale=guidance_scale,
                  width=width,
                  height=height,
                  video_length=frame_count,
                  image=images,
                  controlnet_conditioning_scale=controlnet_conditioning_scale,
                  output_type="pt",
                  guess_mode=guess_mode,
                  ).images.permute(1, 0, 2, 3)

  
  else:
    if use_type == 'overlapping' or use_type == 'overlapping_2' or use_type == 'overlapping_3' or use_type == 'overlapping_4':
      video = pipeline(prompt=prompt, 
                    negative_prompt=negative_prompt, 
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                      width=width,
                      height=height,
                            window_count=window_count,
                      video_length=frame_count,
                      generator=generators).videos[0]
    elif use_type == 'circular':
      # load a init image 
      
      # init_image = Image.open("0000.png")
      init_image = None

      video = pipeline(prompt=prompt, 
            negative_prompt=negative_prompt, 
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
              width=width,
              height=height,
              video_length=frame_count,
              start_image = init_image).videos[0]
    
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
  # prompt = "photograph of a bald caucasian man listening"
  # prompt = "close up of a Girl's face swaying, red blouse, illustration by Wu guanzhong,China village,twojjbe trees in front of my chinese house,light orange, pink,white,blue ,8k"
  # prompt = "paint by frazetta, man dancing, mountain blue sky in background"
  # prompt = "neon glowing psychedelic man dancing, photography, award winning, gorgous, highly detailed"
  # prompt = "Cute Chunky anthropomorphic Siamese cat dressed in rags walking down a rural road, mindblowing illustration by Jean-Baptiste Monge + Emily Carr + Tsubasa Nakai"
  # prompt = "close up of A man walking in front of cityscape, movie production, cinematic, photography, designed by daniel arsham, glowing white, futuristic, white liquid, highly detailed, 35mm"
  # prompt = "closeup of A woman dancing in front of a secret garden, early renaissance paintings, Rogier van der Weyden paintings style"
  # prompt = "close up portrait of a woman in front of a lake artwork by Kawase Hasui"
  # prompt = "A lego ninja bending down to pick a flower in the style of the lego movie. High quality render by arnold. Animal logic. 3D soft focus"
  # prompt = "Glowing jellyfish, calm, slow hypnotic undulations, 35mm Nature photography, award winning"
  prompt = "synthwave retrowave vaporware back of a delorean driving on highway, dmc rear grill, neon lights, palm trees and sunset in background, nightcity"
  # prompt = "a doodle of a bear dancing, scribble, messy, stickfigure, badly drawn"
  # prompt = "close up of Embrodery Elijah Wood smiling in front of a embroidery landscape"
  # prompt = "photo of a tom cruise statue made of ice, model shoot, empty black background"
  # prompt = "watermeloncarving, Terra Cotta Warriors,full body dancing"
  # prompt = "A doll walking in the forest jan svankmajer, brother quay style stop action animation"
  # model = Path("../models/dreamshaper-6")
  # model = Path("../models/deliberate_v2")
  # lora_file="Frazetta.safetensors"
  lora_files=["NightCity.safetensors", "dmc12-000006.safetensors"]
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
  # model = "/mnt/newdrive/automatic1111/models/Stable-diffusion/deliberate_v2.safetensors"
  run(model, 
      prompt=prompt,
      negative_prompt="clone, cloned, bad anatomy, wrong anatomy, mutated hands and fingers, mutation, mutated, amputation, 3d render, lowres, signs, memes, labels, text, error, mutant, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, made by children, caricature, ugly, boring, sketch, lacklustre, repetitive, cropped, (long neck), body horror, out of frame, mutilated, tiled, frame, border",
      height=512,
      width=512,
      frame_count=48,
      window_count=24,
      num_inference_steps=50,
      guidance_scale=7.0,
      last_n=23,
      lora_folder="/mnt/newdrive/automatic1111/models/Lora",
      lora_files=lora_files)

