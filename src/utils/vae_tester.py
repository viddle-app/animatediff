import os
import cv2
import torch
from tqdm import tqdm
from pathlib import Path
from einops import rearrange
from diffusers import StableDiffusionPipeline

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

# Function to decode latent tensors into video frames and save the video
def decode_and_save_video(vae, latents, save_path, video_length):
    latents = 1 / 0.18215 * latents
    
    video = []
    for frame_idx in tqdm(range(latents.shape[0]), desc="Decoding"):
        video.append(vae.decode(latents[frame_idx:frame_idx + 1]).sample)
    video = torch.cat(video)
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)

    video_np = video.cpu().numpy()[0].transpose(1, 2, 3, 0)
    video_np = (video_np * 255).astype('uint8')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 8.0, (video_np.shape[2], video_np.shape[1]))

    for i in range(video_np.shape[0]):
        out.write(cv2.cvtColor(video_np[i], cv2.COLOR_RGB2BGR))
    out.release()

# Main process function to load, encode, decode and save videos
def process_and_save_videos(video_folder, output_folder):
    pipeline = StableDiffusionPipeline.from_pretrained(Path("/mnt/newdrive/models/miniSD"), torch_dtype=torch.float32).to("cuda")
    pipeline.enable_vae_slicing()
    
    os.makedirs(output_folder, exist_ok=True)
    
    videos_in_folder = os.listdir(video_folder)

    for video_name in tqdm(videos_in_folder, desc="Processing Videos"):
        video_path = os.path.join(video_folder, video_name)
        output_video_path = os.path.join(output_folder, Path(video_name).stem + '_roundtripped.mp4')
        
        # Load and encode the video
        video_tensor = load_video_as_tensor(video_path).to("cuda")
        latents = pipeline.vae.encode(video_tensor).latent_dist.sample() * 0.18125
        
        # Decode and save the round-tripped video
        decode_and_save_video(pipeline.vae, latents, output_video_path, video_tensor.shape[0])

if __name__ == "__main__":
    with torch.no_grad():
      video_folder = "data/videos"
      output_folder = "data/miniSD_roundtripped"
      process_and_save_videos(video_folder, output_folder)
