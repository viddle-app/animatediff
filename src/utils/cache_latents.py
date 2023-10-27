import os
from pathlib import Path
from diffusers import StableDiffusionPipeline
import torch
import cv2
from tqdm import tqdm

import cv2
import torch

import cv2
import torch

def load_video_as_tensor(video_path, target_height=256, target_width=256, num_frames=16, stride=4):
    """
    Load a video from the given path into a (B, C, H, W) tensor using OpenCV and PyTorch.
    
    Parameters:
    - video_path: str, path to the video file
    - target_height: int, desired height of the video frames
    - num_frames: int, the total number of frames to return
    - stride: int, the number of frames to skip between each sampled frame

    Returns:
    - PyTorch tensor of shape (B, C, H, W)
    """
    
    # Open video using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    counter = 0
    
    while len(frames) < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            break

        # If counter is 0, process the frame
        if counter == 0:
            # Resize the frame maintaining aspect ratio
            aspect_ratio = frame.shape[1] / frame.shape[0]
            resized_height = target_height
            resized_width = int(resized_height * aspect_ratio)
            resized_frame = cv2.resize(frame, (resized_width, resized_height))
            
            # Center crop the frame
            center_x = resized_width // 2
            center_y = resized_height // 2
            cropped_frame = resized_frame[
                center_y - target_height // 2 : center_y + target_height // 2,
                center_x - target_width // 2 : center_x + target_width // 2,
            ]
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            
            # Convert frame to tensor and normalize
            tensor_frame = torch.tensor(frame_rgb, dtype=torch.float32) / 255.0
            tensor_frame = 2.0 * tensor_frame - 1.0
            
            frames.append(tensor_frame)

        # Increment counter and reset if it reaches the stride value
        counter = (counter + 1) % stride
    
    # Close video capture
    cap.release()

    # Stack frames into tensor of shape (B, C, H, W)
    video_tensor = torch.stack(frames).permute(0, 3, 1, 2)

    return video_tensor




def cache_single_latent(vae, video_path, latent_cache_path,):
  # load the video as a (B, C, H, W) tensor
  pixel_values = load_video_as_tensor(video_path).to(device="cuda", dtype=torch.float32)

  # encode the video tensor using the vae
  latents = vae.encode(pixel_values).latent_dist.sample() * 0.18125
  # save the latents in the latent_cache_path
  torch.save(latents, latent_cache_path)

def cache_latents(video_folder, latent_cache_folder):
  # create a stable diffusion pipeline using the v1-5 model 
  # go through each folder and extract the 16 frames with stride 4
  # encode them and save them in the latent_cache_path

  pipeline = StableDiffusionPipeline.from_pretrained(Path("/mnt/newdrive/models/v1-5")).to("cuda")

  pipeline.enable_vae_slicing()
  
  # make the latent_cache_folder if it doesn't exist
  os.makedirs(latent_cache_folder, exist_ok=True)

  # get the list of all videos in the folder
  videos_in_folder = os.listdir(video_folder)

  # take the set difference of the videos in the folder and the videos in the latent_cache_folder
  videos_in_folder = set(videos_in_folder) - set(os.listdir(latent_cache_folder))

  for video_name in tqdm(videos_in_folder, desc="Caching latents"):
    video_path = os.path.join(video_folder, video_name)
    latent_cache_path = os.path.join(latent_cache_folder, video_name)
    
    cache_single_latent(pipeline.vae, video_path, latent_cache_path)

if __name__ == "__main__":
  video_folder = "data/videos"
  latent_cache_folder = "data/latent_cache"
  cache_latents(video_folder, latent_cache_folder)