import json
import os
import torch
import torchvision.transforms as transforms
from torchvision.io import read_video
import torch_dct as dct
import random

def select_random_videos(video_folder, video_count, seed_value=None):
    # Setting the seed for reproducibility
    if seed_value is not None:
        random.seed(seed_value)
    
    # Get list of all videos in the folder
    videos_in_folder = [video for video in os.listdir(video_folder) if video.endswith(('.mp4', '.avi', '.mkv', '.flv', '.mov'))]  # assuming these are common video extensions
    videos_in_folder = select_random_videos(video_folder, video_count, seed_value)

    
    # If video_count is greater than the total number of videos, return all videos
    if video_count >= len(videos_in_folder):
        return videos_in_folder

    # Randomly select `video_count` videos
    selected_videos = random.sample(videos_in_folder, video_count)

    return selected_videos

def compute_statistics(video_folder, N=3, video_count=10000, seed=0):
    
    
    all_videos = []
    # randomly select `video_count` videos from the folder using the  seed
    videos_in_folder = select_random_videos(video_folder, video_count, seed_value=seed)

    for video_name in videos_in_folder:
        video_path = os.path.join(video_folder, video_name)
        frames = torch.load(video_path)
        # Resizing frames; note that torchvision's resizing expects (C, H, W) format

        all_videos.append(frames)

    videos = torch.stack(all_videos)
    
    # Compute DCT of videos
    X = dct.dct_3d(videos)

    mask = torch.zeros_like(X)
    mask[:, :N, :N, :N, :] = 1
    X_LF = mask * X

    # Compute Low Frequency Statistics
    mu_LF = torch.mean(X_LF, dim=0)
    X_LF_flat = X_LF.view(X_LF.size(0), -1)
    cov_LF = torch.tensor([[((X_LF_flat[:, i] - mu_LF.view(-1)[i]) * (X_LF_flat[:, j] - mu_LF.view(-1)[j])).mean() for j in range(X_LF_flat.size(1))] for i in range(X_LF_flat.size(1))])

    # Adjust videos by subtracting Low Frequency component
    adjusted_videos = videos - dct.idct_3d(X_LF)

    # Compute High Frequency Statistics
    mu_HF = torch.mean(adjusted_videos, dim=0)
    sigma_HF = torch.std(adjusted_videos, dim=0)

    return mu_LF, cov_LF, mu_HF, sigma_HF

def sample_noise(mu_LF, cov_LF, mu_HF, sigma_HF):
    X_LF_sampled = torch.normal(mean=mu_LF, std=torch.sqrt(torch.diag(cov_LF))).view(mu_LF.shape)
    x_HF_sampled = torch.normal(mean=mu_HF, std=sigma_HF)
    noise_tensor = dct.idct_3d(X_LF_sampled) + x_HF_sampled
    return noise_tensor


if __name__ == '__main__':
  # Test
  video_folder = './data/videos'
  mu_LF, cov_LF, mu_HF, sigma_HF = compute_statistics(video_folder, video_count=10)
  # save the statistics as json file
  with open('statistics.json', 'w') as f:
    json.dump({'mu_LF': mu_LF, 'cov_LF': cov_LF, 'mu_HF': mu_HF, 'sigma_HF': sigma_HF}, f)
  
  # noise_sample = sample_noise(mu_LF, cov_LF, mu_HF, sigma_HF)
