import json
import os
import torch
import torchvision.transforms as transforms
from torchvision.io import read_video
import torch_dct as dct
import random
import torch.nn.functional as F
from einops import rearrange

def select_random_videos(video_folder, video_count, seed_value=None):
    if seed_value is not None:
        random.seed(seed_value)
    
    videos_in_folder = [video for video in os.listdir(video_folder)]

    if video_count >= len(videos_in_folder):
        return videos_in_folder

    selected_videos = random.sample(videos_in_folder, video_count)
    return selected_videos

def compute_statistics(video_folder, N=3, video_count=10000, seed=0):
    
    
    all_videos = []
    # randomly select `video_count` videos from the folder using the  seed
    videos_in_folder = select_random_videos(video_folder, video_count, seed_value=seed)

    for video_name in videos_in_folder:
        video_path = os.path.join(video_folder, video_name)
        frames = torch.load(video_path)
        frames = rearrange(frames, "f c h w -> c f h w")

        all_videos.append(frames)

    videos = torch.stack(all_videos).to("cuda")
    
    # Compute DCT of videos
    X = dct.dct_3d(videos)

    print("X shape: ", X.shape)

        # Assume X is the 3D DCT of videos with shape [batch_size, num_channels, depth, height, width]
    # Extract low-frequency components
    X_LF = X[:, :, :N, :N, :N]  # Shape: [batch_size, num_channels, N, N, N]

    # Compute Mean for low-frequency components
    mu_LF = torch.mean(X_LF, dim=0)  # Shape: [num_channels, N, N, N]

    # Flatten the low-frequency components for covariance computation
    X_LF_flat = X_LF.reshape(X_LF.size(0), -1)  # Shape: [batch_size, num_channels * N^3]

    # Center the data for covariance calculation
    X_LF_centered = X_LF_flat - mu_LF.view(-1)

    # Compute Covariance
    cov_LF = torch.matmul(X_LF_centered.T, X_LF_centered) / (X_LF_flat.size(0) - 1)

    # Compute high-frequency components
    X_HF = X.clone()  # Make a copy of the original DCT tensor
    X_HF[:, :, :N, :N, :N] = 0  # Zero out low-frequency components
    videos_HF = dct.idct_3d(X_HF)  # Inverse DCT to get back to video space

    # Compute Mean and Standard Deviation for high-frequency components
    mu_HF = torch.mean(videos_HF, dim=0)
    sigma_HF = torch.std(videos_HF, dim=0)
    print("mu_HF.shape", mu_HF.shape)
    print("sigma_HF.shape", sigma_HF.shape)
    print("cov_LF.shape", cov_LF.shape)
    print("mu_LF.shape", mu_LF.shape)

    return mu_LF, cov_LF, mu_HF, sigma_HF

def sample_noise(mu_LF, cov_LF, mu_HF, sigma_HF, num_channels=4, pad_size=(32, 32), num_frames=16):
    # Create a Multivariate Normal distribution
    mvn = torch.distributions.MultivariateNormal(mu_LF.view(-1), covariance_matrix=cov_LF)

    # Sample from the distribution and reshape to the original shape with the correct number of channels
    X_LF_sampled = mvn.sample().view(-1, num_channels, 3, 3, 3)

    # Pad the last two dimensions to match the size of the high-frequency component
    padding = (0, pad_size[1] - X_LF_sampled.size(-1), 0, pad_size[0] - X_LF_sampled.size(-2))
    X_LF_sampled_padded = F.pad(X_LF_sampled, padding, "constant", 0)

    # Pad the depth (frames) dimension to match the size of the high-frequency component
    X_LF_sampled_padded = F.pad(X_LF_sampled_padded, (0, 0, 0, 0, num_frames - X_LF_sampled_padded.size(2), 0))

    # Generate high-frequency noise
    x_HF_sampled = torch.normal(mean=mu_HF, std=sigma_HF)

    # Combine low and high-frequency noise
    print("X_LF_sampled_padded.shape", X_LF_sampled_padded.shape)
    print("x_HF_sampled.shape", x_HF_sampled.shape)
    noise_tensor = dct.idct_3d(X_LF_sampled_padded) + x_HF_sampled

    return noise_tensor

def produce_statistics():
  # Test
  video_folder = './data/latent_cache'
  mu_LF, cov_LF, mu_HF, sigma_HF = compute_statistics(video_folder, video_count=2500)
  # Convert tensors to lists before saving as JSON
  statistics = {
      'mu_LF': mu_LF.tolist(), 
      'cov_LF': cov_LF.tolist(), 
      'mu_HF': mu_HF.tolist(), 
      'sigma_HF': sigma_HF.tolist()
  }
  
  with open('statistics.json', 'w') as f:
      json.dump(statistics, f)

def make_noise():
  # load the statistics from the file
  with open('statistics.json', 'r') as f:
      statistics = json.load(f)
  mu_LF = torch.tensor(statistics['mu_LF']).to("cuda")
  cov_LF = torch.tensor(statistics['cov_LF']).to("cuda")
  mu_HF = torch.tensor(statistics['mu_HF']).to("cuda")
  sigma_HF = torch.tensor(statistics['sigma_HF']).to("cuda")
  
  noise_sample = sample_noise(mu_LF, cov_LF, mu_HF, sigma_HF)

  print("noise_sample.shape", noise_sample.shape)

  return noise_sample

if __name__ == '__main__':
  # produce_statistics()
  make_noise()
