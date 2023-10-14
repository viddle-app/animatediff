# originally copied from: 
import os
import math
from src.utils.util import save_power_spectrum_as_gif
from src.utils.image_utils import load_video_as_tensor, save_gif_from_tensor, tensor_to_image_sequence
from src.utils.valiation_utils import compute_2d_fft_batch
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

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
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from transformers import Adafactor
import torch.nn as nn


from src.models.unet import UNet3DConditionModel
# from src.pipelines.pipeline_animatediff import AnimationPipeline
# from src.pipelines.pipeline_animatediff_overlapping_previous_2 import AnimationPipeline
# from src.pipelines.pipeline_animatediff_overlapping_previous import AnimationPipeline
from src.pipelines.pipeline_animatediff import AnimationPipeline
from src.utils.util import save_videos_grid, zero_rank_print
# from src.data.dataset_overlapping import WebVid10M
from src.data.dataset import WebVid10M
from collections import OrderedDict

def loss_scaler(timestep, the_step):
    percent = the_step / 100000
    if the_step < 100000:
        return (1 - percent) * (1 / (1 + timestep)) + percent
    else:
        return 1.0

def validate_prediction(data,
                        noise_scheduler,
                        vae,
                        output_dir,
                        the_timestep,
                        latents, 
                        pixel_values, 
                        noisy_latents, 
                        model_pred, 
                        target, 
                        key, 
                        actual_step,
                        save_images,):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    sqrt_alpha_prod = noise_scheduler.alphas_cumprod[the_timestep] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(noisy_latents.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[the_timestep]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_latents.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    # print("latent.device", latents.device)
    # print("sqrt_alpha_prod.device", sqrt_alpha_prod.device)
    sqrt_alpha_prod = sqrt_alpha_prod.to(device=noisy_latents.device)
    # log the predicted and target videos
    # decode the latents of the predicted
    print("sqrt_alpha_prod: ", sqrt_alpha_prod)

    # noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    pred = (noisy_latents - model_pred * sqrt_one_minus_alpha_prod ) / sqrt_alpha_prod   
    # compute the video_frame_diff error
    
    # save the decoded_pred as a gif
    error_tensor = (target - model_pred)**2

    # compute the video_frame_diff error
    pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
    latent_diff = latents[:, :, 1:] - latents[:, :, :-1]

    model_pred_diff = model_pred[:, :, 1:] - model_pred[:, :, :-1]
    target_diff = target[:, :, 1:] - target[:, :, :-1]

    # compute the error of the diffs 
    diff_noise_error = (model_pred_diff - target_diff)**2
    print("pred_diff.shape: ", pred_diff.shape)
    print("latent_diff: ", latent_diff.shape)
    diff_error = (pred_diff - latent_diff)**2
    print("diff_error max", diff_error.max())
    print("diff_error min", diff_error.min())
    print("diff_error mean", diff_error.mean())
    print("diff_error std", diff_error.std())


    reconstruction_loss = error_tensor.mean()
    noisy_diff_loss = diff_noise_error.mean()
    diff_loss = loss_scaler(the_timestep, 0)*diff_error.mean()
    loss = 0.5 * reconstruction_loss + 0.5*(diff_loss + noisy_diff_loss)
    # TODO log the losses
    
    data[f"validation_reconstruction_loss_{the_timestep}"] = reconstruction_loss.item()
    data[f"validation_noisy_diff_loss_{the_timestep}"] = noisy_diff_loss.item()
    data[f"validation_diff_loss_{the_timestep}"] = diff_loss.item()
    data[f"validation_loss_{the_timestep}"] = loss.item()

    if save_images:
        # create a folder for debug outputs
        folder_path = f"{output_dir}/{key}"
        os.makedirs(folder_path, exist_ok=True)
        pred_latents = 1 / vae.config.scaling_factor * pred
        pred_latents = rearrange(pred_latents, "b c f h w -> (b f) c h w")
        pred = vae.decode(pred_latents, return_dict=False)[0]

        target_pixels = noisy_latents
        target_pixels = 1 / vae.config.scaling_factor * target_pixels
        target_pixels = rearrange(target_pixels, "b c f h w -> (b f) c h w")
        target_pixels = vae.decode(target_pixels, return_dict=False)[0]

        # decode the diff error
        print("diff_error.shape: ", diff_error.shape)
        diff_error = 1 / vae.config.scaling_factor * diff_error
        diff_error = rearrange(diff_error, "b c f h w -> (b f) c h w")
        diff_error = vae.decode(diff_error, return_dict=False)[0]
        
        # add black to the first frame 
        diff_error = torch.cat([torch.zeros_like(diff_error[0]).unsqueeze(0), diff_error], dim=0)
        
        # decode the diff noise error
        diff_noise_error = 1 / vae.config.scaling_factor * diff_noise_error
        diff_noise_error = rearrange(diff_noise_error, "b c f h w -> (b f) c h w")
        diff_noise_error = vae.decode(diff_noise_error, return_dict=False)[0]

        diff_noise_error = torch.cat([torch.zeros_like(diff_noise_error[0]).unsqueeze(0), diff_noise_error], dim=0)

        # decode the error tensor
        error_tensor = 1 / vae.config.scaling_factor * error_tensor
        error_tensor = rearrange(error_tensor, "b c f h w -> (b f) c h w")
        error_tensor = vae.decode(error_tensor, return_dict=False)[0]

        # compute the 2d fft log spectrum for the pixel values and the pred and the error

        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        target = (target / 2 + 0.5).clamp(0, 1)
        pred = (pred / 2 + 0.5).clamp(0, 1)
        diff_error_pixels = (diff_error / 2).clamp(0, 1)
        diff_noise_error_pixels = (diff_noise_error / 2).clamp(0, 1)
        error_tensor_pixels = (error_tensor / 2).clamp(0, 1)
        target_pixels = (target_pixels / 2 + 0.5).clamp(0, 1)

        # concat the tensors along the W dimension
        combined = torch.cat([pixel_values, 
                            target_pixels, 
                            pred, 
                            error_tensor_pixels, 
                            diff_error_pixels,
                            diff_noise_error_pixels,
                            ], dim=3)

        output_path = f"{folder_path}/{actual_step}_{the_timestep}.gif"
        save_gif_from_tensor(combined, output_path)
        # save a folder of images
        image_folder_path = f"{folder_path}/{actual_step}_{the_timestep}"
        tensor_to_image_sequence(combined.permute(1, 0, 2, 3), image_folder_path)

        wandb.log({key: wandb.Image(output_path)})







    
def load_loss_data(file_path):
    """
    Load loss data from a CSV file into a dictionary.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - dict: A dictionary where keys are steps and values are losses.
    """
    # Read the CSV file using pandas
    data = pd.read_csv(file_path)

    # Ensure 'step' is integer type for proper indexing
    data['step'] = data['step'].astype(int)

    # Create a dictionary where the key is the 'step' and the value is the 'loss'
    loss_dict = dict(zip(data['step'], data['loss']))
    
    return loss_dict

def extract_motion_module(unet):
    mm_state_dict = OrderedDict()
    state_dict = unet.state_dict()
    state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    for key in state_dict:
        if "motion_module" in key:
            mm_state_dict[key] = state_dict[key]

    return mm_state_dict

def append_loss_to_csv(step, loss, filename):
    # Check if the file exists. If not, create it and write headers
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            # Write header
            writer.writerow(['step', 'loss'])
        
        # Write data
        writer.writerow([step, loss])

class RunningAverages:
    def __init__(self):
        # Initialize a dictionary to hold the sum and count for each index
        self.data_dict = {}

    def update(self, index, value):
        # If the index is not in the dictionary, initialize it
        if index not in self.data_dict:
            self.data_dict[index] = {"sum": 0, "count": 0}
        # Update the sum and count for the index
        self.data_dict[index]["sum"] += value
        self.data_dict[index]["count"] += 1

    def graph_averages(self, plot_path):

         

        # Compute the running averages
        indexes = list(self.data_dict.keys())
        averages = [self.data_dict[i]["sum"] / self.data_dict[i]["count"] for i in indexes]

        # Convert indexes and averages to numpy arrays on CPU if they are torch tensors
        if torch.is_tensor(indexes):
            indexes = indexes.cpu().numpy()
        if torch.is_tensor(averages):
            averages = averages.cpu().numpy()

               # Plot the averages using a line graph
        plt.plot(indexes, averages, marker='o')
        plt.xlabel('Indexes')
        plt.ylabel('Averages')
        plt.title('Running Averages of Indexed Quantities')
        
        
        plt.savefig(plot_path)
        plt.close()

def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    print("launcher: ", launcher)
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

def entropy_of_normalized_frequency_spectrum(video_tensor):
    """
    video_tensor: torch.Tensor
        A tensor of shape (B, C, D, H, W)

    Returns:
        Tensor containing the entropy values for each channel, depth, and each batch.
    """
    spectrum = torch.fft.fftn(video_tensor, dim=(2, 3, 4))
    
    # Compute the spectral energy
    energy = torch.abs(spectrum)**2

    # Normalize to create a probability distribution
    prob_dist = energy / energy.sum(dim=(2,3,4), keepdim=True)

    # Compute the entropy: -sum(p * log(p))
    entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-10), dim=(2,3,4))
    
    return entropy


def l2_spectrum_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    # Ensure the inputs are of the same size
    pred = rearrange(pred, "b c f h w -> (b f) c h w")
    true = rearrange(true, "b c f h w -> (b f) c h w")

    assert pred.size() == true.size(), "Predicted and True tensors must have the same size"
    
    # Compute 2D FFT for both tensors along the H and W dimensions
    pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
    true_fft = torch.fft.fft2(true, dim=(-2, -1))
    
    # Compute the magnitude (L2 norm) of the complex numbers 
    pred_magnitude = torch.abs(pred_fft)
    true_magnitude = torch.abs(true_fft)
    
    # Compute L2 loss between the magnitudes
    loss = F.mse_loss(pred_magnitude, true_magnitude)
    
    return loss

def frame_diff(video_tensor):
    """
    Compute the frame difference for a video tensor.
    video_tensor should have shape (batch_size, channels, frames, height, width)
    """
    return video_tensor[:, :, 1:] - video_tensor[:, :, :-1]

def video_diff_loss(original_video, generated_video):
    """
    Compute the loss between the frame differences of the original and generated videos.
    Both videos should have shape (batch_size, num_frames, height, width, channels)
    """
    # Calculate frame differences
    original_diff = frame_diff(original_video)
    generated_diff = frame_diff(generated_video)

    # Calculate MSE loss between frame differences
    loss = F.mse_loss(original_diff, generated_diff, reduction='mean')
    return loss

def video_diff_loss_vec(original_video, generated_video):
    original_diff = frame_diff(original_video)
    generated_diff = frame_diff(generated_video)

    return (original_diff - generated_diff)**2

def compute_fft_and_update_state(input_tensor, state=None):
    # Step 1: Compute 1D FFT along the frequency dimension (F)
    rearranged = rearrange(input_tensor, "b c f h w -> c (b h w) f")

    # Step 1: Compute 1D FFT along the frequency dimension (now the last dimension)
    fft_result = torch.fft.fft(input_tensor)

    # Consider only the first 8 frequencies due to symmetry in real signals
    fft_result = fft_result[:, :, :8]

    # Step 2: Compute magnitude of the spectrum
    magnitude_spectrum = torch.abs(fft_result)

    if state is None:
        state = {}  # Ensure state is a dictionary if it's None
    if 'sum_magnitudes' not in state:
        state['sum_magnitudes'] = torch.zeros(8, device=input_tensor.device)
    if 'count' not in state:
        state['count'] = torch.zeros(8, device=input_tensor.device)

    # Update state: sum the magnitudes and increment the count for each frequency
    for freq_idx in range(8):
        freq_magnitude = magnitude_spectrum[:, :, freq_idx, :, :]
        state['sum_magnitudes'][freq_idx] += freq_magnitude.sum()
        state['count'][freq_idx] += freq_magnitude.numel()

    return state

def plot_self_ratio(spectrum_dict, plot_path):
    # Compute running average
    running_avg = spectrum_dict['sum_magnitudes'] / spectrum_dict['count']
    
    # Avoid division by zero by ensuring the DC component is not zero
    # dc_component = running_avg[0] + (running_avg[0] == 0).float() * 1e-10
    dc_component = running_avg[0]

    # Compute the ratio
    ratio = running_avg / dc_component
    
    # Convert to numpy for plotting
    ratio = ratio.cpu().numpy()
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar width
    width = 0.35  
    
    # Create bar positions
    freqs = torch.arange(8)
    
    # Create bars
    bars = ax.bar(freqs, ratio, width, label=None)
    
    # Labeling
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Ratio to DC Component')
    ax.set_title('Ratio of Each Frequency Magnitude to DC Component')
    ax.set_xticks(freqs)
    ax.set_xticklabels([str(f) for f in freqs])
    ax.legend()
    
    # Adding the text labels within each bar
    for bar, value in zip(bars, ratio):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, round(value, 5), ha='center', va='bottom')
    
    # Save plot
    plt.savefig(plot_path)
    plt.close(fig)  # Close the figure to free up memory

def plot_comparison(dict1, dict2, plot_path):
    # Compute running averages
    running_avg1 = dict1['sum_magnitudes'] / dict1['count']
    running_avg2 = dict2['sum_magnitudes'] / dict2['count']
    
    # Convert to numpy for plotting
    running_avg1 = running_avg1.cpu().numpy()
    running_avg2 = running_avg2.cpu().numpy()
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar width
    width = 0.35  
    
    # Create bar positions
    freqs = torch.arange(8)
    bar1_positions = freqs - width/2
    bar2_positions = freqs + width/2
    
    # Create bars
    ax.bar(bar1_positions, running_avg1, width, label='Predicted')
    ax.bar(bar2_positions, running_avg2, width, label='Target')
    
    # Labeling
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Average Magnitude')
    ax.set_title('Comparison of Average Magnitude per Frequency')
    ax.set_xticks(freqs)
    ax.set_xticklabels([str(f) for f in freqs])
    ax.legend()
    
    # Save plot
    plt.savefig(plot_path)
    plt.close(fig)

def get_lr(timestep, min_lr, max_lr, max_timestep=1000):
    return min_lr + (max_lr - min_lr) * (timestep / max_timestep)

def get_lr_2(timestep, min_lr, max_lr, k=6, max_timestep=1000):
    return min_lr + (max_lr - min_lr) * (1 - 1 / (1 + k * timestep / max_timestep))

def get_lr_3(timestep, min_lr, max_lr, max_timestep=1000):
    timestep_ratio = timestep / max_timestep
    lr_length = max_lr - min_lr
    lr_percent = 4 * (timestep_ratio - 0.5)**2
    return min_lr + lr_percent * lr_length

def get_lr_4(timestep, min_lr, max_lr, max_lr_2=5e-7, max_timestep=1000):
    timestep_ratio = timestep / max_timestep

    if timestep_ratio < 0.5:
        lr_length = max_lr - min_lr
        lr_percent = 4 * (timestep_ratio - 0.5)**2
        return min_lr + lr_percent * lr_length
    else:
        lr_length = max_lr_2 - min_lr
        lr_percent = 4 * (timestep_ratio - 0.5)**2
        return min_lr + lr_percent * lr_length
    
def get_lr_5(timestep, max_lr, min_lr, min_lr_2=1e-6, max_timestep=1000):
    timestep_ratio = timestep / max_timestep

    if timestep_ratio < 0.5:
        if min_lr > max_lr:
            min_lr, max_lr = max_lr, min_lr

        lr_length = max_lr - min_lr
        lr_percent = -4 * (timestep_ratio - 0.5)**2 + 1
        return min_lr + lr_percent * lr_length
    else:
        if min_lr_2 > max_lr:
            min_lr_2, max_lr = max_lr, min_lr_2

        lr_length = max_lr - min_lr_2
        lr_percent = -4 * (timestep_ratio - 0.5)**2 + 1
        return min_lr_2 + lr_percent * lr_length

def get_lr_6_b(timestep):
    if timestep < 100:
        return 8e-7
    elif timestep < 200:
        return 5e-7
    elif timestep < 300:
        return 3e-7
    elif timestep < 400:
        return 1e-7
    elif timestep < 500:
        return 5e-8
    elif timestep < 600:
        return 3e-8
    elif timestep < 700:
        return 8e-9
    elif timestep < 800:
        return 8e-10
    elif timestep < 850: 
        return 1e-10
    elif timestep < 900:
        return 8e-11
    elif timestep < 950:
        return 1e-11
    elif timestep < 975:        
        return 1e-12
    elif timestep < 985:
        return 1e-15
    elif timestep < 995:
        return 1e-17
    elif timestep < 998:
        return 1e-19
    else:
        return 1e-23 
        
def get_lr_6(timestep):
    return 1e-7

def cubic_resampling(x):
    """
    Calculate the expression (1 - (x/1000)^3) * 1000
    
    Parameters:
    x (float): Input to the function
    
    Returns:
    float: Result of the expression
    """
    return (1 - (x/1000)**3) * 1000

def inverse_cubic_resampling(x):
    """
    Calculate the inverse of the expression (1 - (x/1000)^3) * 1000
    
    Parameters:
    x (float): Input to the function
    
    Returns:
    float: Result of the expression
    """
    return 1000 * (1 - (1 - x/1000)**(1/3))

def compute_frequency_energies(input):
    # Assume input is (batch, channels, frames, height, width)
    # Compute the FFT along the frames dimension
    spectrum = torch.fft.fft(input, dim=2)
    # Compute the magnitude
    spectrum1d = torch.abs(spectrum)
    
    frequency_energies = []
    for i in range(spectrum1d.shape[2]//2 + 1):  # +1 to include the nyquist frequency
        # Extract the ith frequency band energy for all pixels
        img = spectrum1d[:, :, i, :, :]
        img = torch.log1p(img)
        # Summing energy across all pixels and channels
        summed_energy = img.sum(dim=[0, 1, 2, 3])  # summing over batch, channels, height, and width
        frequency_energies.append(summed_energy.item())  # Assuming you want to return python scalar
    
    return frequency_energies

slow_validation_steps = 100

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
):
    check_min_version("0.10.0.dev0")

    default_losses = load_loss_data("stats.csv")
    lr = learning_rate

    pred_fft_state = {}
    target_fft_state = {}

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
    noise_scheduler.set_timesteps(noise_scheduler.num_train_timesteps)

    if False:
        pipeline = StableDiffusionPipeline.from_single_file(pretrained_model_path, torch_dtype=dtype)
      
        tokenizer = pipeline.tokenizer  
        text_encoder = pipeline.text_encoder
        vae = pipeline.vae
        unet = UNet3DConditionModel.from_unet2d(pipeline.unet, 
                                                unet_additional_kwargs=unet_additional_kwargs)
    else: 
        tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", torch_dtype=dtype)
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", torch_dtype=dtype)
        vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", torch_dtype=dtype)

        if image_finetune:
            print("loading 2d unet")
            unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet", torch_dtype=dtype)
        else:      
            
            fine_tuned_unet = torch.load("/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-11T11-46-02/checkpoints/checkpoint-epoch-1.ckpt", map_location="cpu")
            unet = UNet3DConditionModel.from_unet2d(fine_tuned_unet, unet_additional_kwargs=unet_additional_kwargs)
            # unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, 
            #                                                subfolder="unet",
            #                                               unet_additional_kwargs=unet_additional_kwargs,)
        
    print(f"state_dict keys: {list(unet.config.keys())[:10]}")

    global_step = 0

    if image_finetune == False:
        # motion_module_path = "motion-models/mm-Stabilized_high.pth"
        # motion_module_path = "models/mm-1000.pth"
        # motion_module_path = "models/motionModel_v03anime.ckpt"
        # motion_module_path = "models/mm_sd_v14.ckpt"
        motion_module_path = "motion-models/mm_sd_v15_v2.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-07T15-59-37/checkpoints/checkpoint.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-28T19-12-07/checkpoints/checkpoint.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-28T14-37-24/checkpoints/checkpoint.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-28T15-46-14/checkpoints/checkpoint.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-28T10-57-52/checkpoints/checkpoint.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-09-28T11-39-21/checkpoints/checkpoint.ckpt"
        # motion_module_path = "motion-models/temporal-attn-pe-5e-7-4-50000-steps.ckpt"
        # motion_module_path = '../ComfyUI/custom_nodes/ComfyUI-AnimateDiff/models/animatediffMotion_v15.ckpt'
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-05T18-29-30/checkpoints/checkpoint.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-12T19-32-53/checkpoints/checkpoint.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-13T03-04-19/checkpoints/checkpoint.ckpt"
        

        motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
        missing, unexpected = unet.load_state_dict(motion_module_state_dict, strict=False)
        if "global_step" in motion_module_state_dict:
            raise Exception("global_step present. Not sure how to handle that.")
        # print("unexpected", unexpected)
        assert len(unexpected) == 0
        print("missing", len(missing))

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
    if image_finetune:
        unet.requires_grad_(True)
        for name, param in unet.named_parameters():
            param.requires_grad = True
            
    else:
        unet.requires_grad_(False)
        for name, param in unet.named_parameters():
            for trainable_module_name in trainable_modules:
                if trainable_module_name in name:
                    param.requires_grad = True
                    break

    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if True:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )
    elif False:
        optimizer = Adafactor(
            trainable_params,
            lr=learning_rate,  # Note: AdaFactor adjusts learning rates dynamically and doesn't require a fixed learning rate
            eps=(1e-30, 1e-3),  # Tuple of two eps values: regularizer term eps and factorization term eps
            clip_threshold=1.0,  # Gradient clipping threshold
            decay_rate=-0.8,  # Decay rate for the second moment. < 0 implies using the default AdaFactor schedule
            beta1=None,  # beta1 parameter: < 0 implies N/A
            weight_decay=0.0,  # Weight decay rate
            scale_parameter=True,  # Whether to scale the learning rate
            relative_step=False,  # Whether to compute the learning rate relative to the step
            warmup_init=False  # Whether to initialize the learning rate as 0 during warm-up
        )
    else: 
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=learning_rate,
            weight_decay=adam_weight_decay,  # Optional: if you want to use weight decay
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

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    if not image_finetune:
        validation_pipeline = AnimationPipeline(
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
        ).to("cuda")
    else:
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path,
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
        )
    validation_pipeline.enable_vae_slicing()

    # DDP warpper
    unet.to(local_rank)
    # unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    data_loader_steps = len(train_dataloader)
    print("data_loader_steps", data_loader_steps)
    num_update_steps_per_epoch = math.ceil(data_loader_steps / gradient_accumulation_steps)
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
    mean_grad = -1
    var_grad = -1
    actual_step = global_step
    
    step_index = 0
    direction = 1
    prev_grad = None
    angle = 0

    # optimizer_states = {}

    stdDevAccum = RunningAverages()

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()

        print(f"epoch: {epoch}, global_step: {global_step}, max_train_steps: {max_train_steps}")
        
        for step, batch in enumerate(train_dataloader):
            # unet.clear_last_encoder_hidden_states()
            print(f"step: {step}")


            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]
                
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.gif", rescale=True)
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value / 2. + 0.5
                        torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.png")
                    

                    

            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].to(local_rank)
            video_length = pixel_values.shape[1]
            print("video_length: ", video_length)

            
            if video_length == 16:
                overlapping_step = False
            elif video_length == 32:
                overlapping_step = True
            else:
                print("video_length: ", video_length)
                overlapping_step = False

            print("pixel_values.shape: ", pixel_values.shape)


            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()

                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            print("latents.shape", latents.shape)

            # keeping things simple for now
            # assert(bsz == 1)
            
            # set the number of inference steps

            # Sample a random timestep for each video
            # 
            if image_finetune:
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            else:
                # timesteps = torch.tensor([step_index]).repeat(bsz) 
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)

                # timesteps = cubic_resampling(timesteps.float())

                # clamp to the 0 and noise_scheduler.config.num_train_timesteps
                timesteps = torch.clamp(timesteps, 0, noise_scheduler.config.num_train_timesteps - 1)

                # if (step_index//100) in optimizer_states:
                #     optimizer.load_state_dict(optimizer_states[step_index // 100])



            # timesteps = torch.tensor([0])
            timesteps = timesteps.long()
            print("timesteps: ", timesteps)

            # get a random timestep for each video

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
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # I think some amount of time I need to add more noise
            # take a random int and mod it by 10
            # if it's 0, then add more noise

            # if step % 10 == 0:
            #    noisy_latents = noisy_latents + 0.1 * torch.randn_like(noisy_latents)

            def ddim_step(model_output: torch.FloatTensor,
                         timestep: int,
                         sample: torch.FloatTensor,
                         prev_timestep: int):
                eta = 0
                alpha_prod_t = noise_scheduler.alphas_cumprod[timestep]
                alpha_prod_t_prev = noise_scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

                beta_prod_t = 1 - alpha_prod_t

                # 3. compute predicted original sample from predicted noise also called
                # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                
                pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                pred_epsilon = model_output
          

                # 4. Clip or threshold "predicted x_0"
                if noise_scheduler.config.thresholding:
                    pred_original_sample = noise_scheduler._threshold_sample(pred_original_sample)
                elif noise_scheduler.config.clip_sample:
                    pred_original_sample = pred_original_sample.clamp(
                        -noise_scheduler.config.clip_sample_range, noise_scheduler.config.clip_sample_range
                    )

                # 5. compute variance: "sigma_t(η)" -> see formula (16)
                # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
                variance = noise_scheduler._get_variance(timestep, prev_timestep)
                std_dev_t = eta * variance ** (0.5)


                # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

                # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                return alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
            
            the_timestep = timesteps[0].item()

            # update the lr of the optimizer. I'm assuming no gradient accumulation here
            # lr = get_lr_5(the_timestep, 5e-7, 2e-7, 5e-9)

            # print("lr: ", lr)

            # for param_group in optimizer.param_groups:
            #    param_group['lr'] = lr

            # if the_timestep != 0 and video_length == 16:
            #    continue
            #elif the_timestep == 0 and video_length == 16 and step % 10 != 0:
            #    continue

            

            if overlapping_step and the_timestep > 2 and False:
                with torch.no_grad():
                    noisy_latents_1 = noisy_latents[:, :, :16].detach()
                    noisy_latents_2 = noisy_latents[:, :, 16:].detach()

                    model_pred_1 = unet(noisy_latents_1, timesteps, encoder_hidden_states).sample
                    unet.swap_next_to_last()
                    print("model_pred_1 mean and std", model_pred_1.mean(), model_pred_1.std())
                    model_pred_2 = unet(noisy_latents_2, timesteps, encoder_hidden_states).sample
                    unet.clear_last_encoder_hidden_states()
                    
                    print("model_pred_2 mean and std", model_pred_2.mean(), model_pred_2.std())

                    # how to randomly step
                    # just picking a random num inference steps won't work
                    # it might not have the current time step.
                    # I might want to start by picking a num inference steps
                    # then pick a random timestep from there 
                    # then step to the next timestep

                    min_timestep = max(1, the_timestep - 100)

                    previous_step = torch.randint(min_timestep, 
                                                   the_timestep - 1, 
                                                  (1,), 
                                                  device=latents.device)
                    previous_step = previous_step.long().item()
                    # previous_step = the_timestep - 1
                    # DDIM step 
                    noisy_latents_1 = ddim_step(model_pred_1, the_timestep, noisy_latents_1, previous_step)
                    noisy_latents_2 = ddim_step(model_pred_2, the_timestep, noisy_latents_2, previous_step)
                    # noisy_latents_1 = noise_scheduler.step(noisy_latents_1, the_timestep, model_pred_1).prev_sample
                    # noisy_latents_2 = noise_scheduler.step(noisy_latents_2, the_timestep, model_pred_2).prev_sample
                    

                    # take the last 8 frames from the first part and the first 8 frames from the second part
                    # noisy_latents = torch.cat([noisy_latents_1[:, :, 8:], noisy_latents_2[:, :, :8]], dim=2)

                    # take the middle 16 frames for the latents
                    latents = latents[:, :, 8:24]
                    
                    weights_0 = torch.tensor([1,1,1,1,1,1,1,1,7/8,6/8,5/8,4/8,3/8,2/8,1/8,0]).reshape(1, 1, -1, 1, 1).to(device=latents.device, dtype=latents.dtype)
                    weights_1 = torch.tensor([0,1/8,2/8,3/8,4/8,5/8,6/8,7/8,1,1,1,1,1,1,1,1]).reshape(1, 1, -1, 1, 1).to(device=latents.device, dtype=latents.dtype)

                    noisy_latents = noisy_latents_1 * weights_0 + noisy_latents_2 * weights_1

                    # print("timesteps before: ", timesteps)
                    # convert the previous_step to a tensor the same shape as timesteps
                    timesteps = torch.full_like(timesteps, previous_step)
                    the_timestep = previous_step

                    # print("timesteps after: ", timesteps)
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)
                    sqrt_alpha_prod = noise_scheduler.alphas_cumprod[the_timestep] ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                    while len(sqrt_alpha_prod.shape) < len(latents.shape):
                        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

                    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[the_timestep]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                    while len(sqrt_one_minus_alpha_prod.shape) < len(latents.shape):
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

                    # print("latent.device", latents.device)
                    # print("sqrt_alpha_prod.device", sqrt_alpha_prod.device)
                    sqrt_alpha_prod = sqrt_alpha_prod.to(device=latents.device)
                    recovered_noisy_samples = noisy_latents - sqrt_alpha_prod * latents 
                    # print("target before mean: ", target.mean())
                    # print("target before std: ", target.std())
                    target = recovered_noisy_samples  / sqrt_one_minus_alpha_prod


                noisy_latents = noisy_latents.clone().detach().requires_grad_(True) 
            else: 
                # slice the latents and target to 16 frames
                if not image_finetune:
                    noisy_latents = noisy_latents[:, :, :16]
                    target = target[:, :, :16]
            
            # print("target after mean: ", target.mean())
            # print("target after std: ", target.std())

                # clone the noisy_latents to have gradients

            # Predict the noise residual and compute loss
            # Mixed-precision training
            # unet.clear_last_encoder_hidden_states()

            # take diff power spectrum

            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                model_pred = unet(noisy_latents, the_timestep, encoder_hidden_states).sample
                
                print("model_pred.shape", model_pred.shape)
                
                model_pred_mean = model_pred.mean()
                model_pred_std = model_pred.std()

                target_mean = target.mean()
                target_std = target.std()

                ratio_of_std = model_pred_std / target_std
                diff_of_mean = model_pred_mean - target_mean
                
                print("model_pred_mean: ", model_pred_mean)
                print("model_pred_std: ", model_pred_std)
                print("target_mean: ", target_mean)
                print("target_std: ", target_std)
                print("ratio_of_std: ", ratio_of_std)
                print("diff_of_mean: ", diff_of_mean)

                # compute the per frame mean, std and ratios

                std_loss = F.mse_loss(model_pred_std, torch.tensor(1.0).to(model_pred_std.device).float(), reduction="mean") + F.mse_loss(diff_of_mean.float(), torch.tensor(0.0).to(diff_of_mean.device).float(), reduction="mean")
                mean_loss = F.mse_loss(model_pred_mean, torch.tensor(0.0).to(model_pred_std.device).float(), reduction="mean")
                normality_loss = std_loss + mean_loss

                reconstruction_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # actual_normalized_spectrum_entropy = entropy_of_normalized_frequency_spectrum(target)
                if not image_finetune:
                    predicted_normalized_spectrum_entropy = entropy_of_normalized_frequency_spectrum(model_pred)
                    predicted_frequency_energies = compute_frequency_energies(model_pred)
                    target_frequency_energies = compute_frequency_energies(target)
                else:
                    predicted_normalized_spectrum_entropy = torch.tensor(0.0).to(model_pred_std.device).float()



                sum_of_entropy_mag = predicted_normalized_spectrum_entropy.sum()
                print("sum_of_entropy_mag: ", sum_of_entropy_mag)

                # need to get the x_0
                
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)
                sqrt_alpha_prod = noise_scheduler.alphas_cumprod[the_timestep] ** 0.5
                sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                while len(sqrt_alpha_prod.shape) < len(latents.shape):
                    sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

                sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[the_timestep]) ** 0.5
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                while len(sqrt_one_minus_alpha_prod.shape) < len(latents.shape):
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

                # print("latent.device", latents.device)
                # print("sqrt_alpha_prod.device", sqrt_alpha_prod.device)
                sqrt_alpha_prod = sqrt_alpha_prod.to(device=latents.device)
                # log the predicted and target videos
                # decode the latents of the predicted
                print("sqrt_alpha_prod: ", sqrt_alpha_prod)

                # noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

                pred = (noisy_latents - model_pred * sqrt_one_minus_alpha_prod ) / sqrt_alpha_prod                

                the_frame_diff_loss_noise = video_diff_loss(model_pred, target)

                the_loss_scaler = loss_scaler(the_timestep, actual_step)
                print("the_loss_scaler", the_loss_scaler)
                the_frame_diff_loss = the_loss_scaler * F.mse_loss(pred.float(), latents.float(), reduction="mean")
                # the_frame_diff_loss_noise = nn.L1Loss()(pred.float(), latents.float())

                # the loss is the two losses dot producted
         
                # loss = normality_loss + the_frame_diff_loss
                if not image_finetune:
                    loss = 0.5 * reconstruction_loss + 0.5 * (the_frame_diff_loss + the_frame_diff_loss_noise)
                else:
                    loss = reconstruction_loss
                
                # loss = reconstruction_loss

                # print("entropy_loss: ", entropy_loss)
                # spectrum_loss = l2_spectrum_loss(model_pred, target)
                # loss = reconstruction_loss + spectrum_loss

        


                


            # unet.clear_last_encoder_hidden_states()

            grad_magnitudes = []

            # if step_index == noise_scheduler.num_train_timesteps - 1:
            #    direction = -100
            
            # elif step_index == 0:
            #    direction = 100
            
            step_index += 200

            if step_index > noise_scheduler.num_train_timesteps - 1:
                step_index += 1

            step_index = step_index % noise_scheduler.num_train_timesteps

            if actual_step % gradient_accumulation_steps == gradient_accumulation_steps - 1:
                # Backpropagate
                if mixed_precision_training:
                    scaler.scale(loss).backward()
                    """ >>> gradient clipping >>> """
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                    """ <<< gradient clipping <<< """
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    """ >>> gradient clipping >>> """
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                    """ <<< gradient clipping <<< """
                    optimizer.step()

                # Compute statistics
                if actual_step % 200 == 199:
                    for name, param in unet.named_parameters():
                        if param.grad is not None:
                            wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().detach().numpy())})
                    
                for param in unet.parameters():
                    if param.grad is not None:
                        grad_magnitudes.append(torch.norm(param.grad).item())
                
                if len(grad_magnitudes) > 0:
                    print("before grad mag sum")
                    mean_grad = sum(grad_magnitudes) / len(grad_magnitudes)
                    print("after grad mag sum")
                    var_grad = sum((x - mean_grad) ** 2 for x in grad_magnitudes) / len(grad_magnitudes)                    
                else:
                    mean_grad = -1
                    var_grad = -1


                optimizer.zero_grad()
                lr_scheduler.step()
                progress_bar.update(1)
                global_step += 1
                # optimizer_states[step_index//100] = optimizer.state_dict()


            ### <<<< Training <<<< ###
            actual_step += 1

            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log

   
                weight_sum = 0
                n_weights = 0
                weight_squared_sum = 0
                weights_max = 0
                weights_min = 10000000000


                for param in unet.parameters():                    
                    weight_sum += torch.sum(param)
                    weight_squared_sum += torch.sum(param * param)
                    weights_max = max(weights_max, torch.max(param).item())
                    weights_min = min(weights_max, torch.min(param).item())
                    n_weights += param.numel()

                mean_weight = weight_sum / n_weights
                var_weight = weight_squared_sum / n_weights - mean_weight * mean_weight


                # take the fft of the weights
                # take the fft of the gradients

                
                # mean_weight = sum(weights) / len(weights)
                # var_weight = sum((x - mean_weight) ** 2 for x in weights) / len(weights)

                # Log statistics to wandb

                # TODO fast validation
                # compare the per pixel total spectrum of
                # each frequency band
                # compute the spatial spectrum of each frame, predicted and target
                # create a histogram of the errors of the spectrum grouping by high and low frequencies

                # compute other metrics like the activations and 
                # attention heat maps

                if the_timestep < 100:
                    prefix = "0_99_"
                elif the_timestep < 200:
                    prefix = "100_199_"
                elif the_timestep < 300:
                    prefix = "200_299_"
                elif the_timestep < 400:
                    prefix = "300_399_"
                elif the_timestep < 500:
                    prefix = "400_499_"
                elif the_timestep < 600:
                    prefix = "500_599_"
                elif the_timestep < 700:
                    prefix = "600_699_"
                elif the_timestep < 800:
                    prefix = "700_799_"
                elif the_timestep < 900:
                    prefix = "800_899_"
                elif the_timestep < 1000:
                    prefix = "900_999_"

                data = {
                    f"{prefix}train_loss": loss.item(),
                    # f"{prefix}train_loss_ratio": loss_after.item() / loss.item(),
                    f"{prefix}reconstruction_loss": reconstruction_loss,
                    f"{prefix}predicted_normalized_spectrum_entropy": sum_of_entropy_mag.item(),
                    f"{prefix}the_frame_diff_loss": the_frame_diff_loss,
                    f"{prefix}the_frame_diff_loss_noise": the_frame_diff_loss_noise,
                    f"{prefix}normality_loss": normality_loss,
                    f"{prefix}mean_gradient": mean_grad, 
                    f"{prefix}max_gradient": max(grad_magnitudes) if len(grad_magnitudes) > 0 else -1,
                    f"{prefix}min_gradient": min(grad_magnitudes) if len(grad_magnitudes) > 0 else -1,
                    f"{prefix}variance_gradient": var_grad,
                    f"{prefix}mean_weights": mean_weight, 
                    f"{prefix}variance_weights": var_weight, 
                    f"{prefix}max_weights": weights_max,
                    f"{prefix}min_weights": weights_min,
                    f"{prefix}timestep": the_timestep,
                    f"{prefix}lr": lr,
                    }
                
                # for each predicted_frequency_energy and target_frequency_energy
                # add an entry to data

    
                csv_path = f"{output_dir}/stats.csv"
                append_loss_to_csv(actual_step, loss.item(), csv_path)

                if True:
                    # load mp4 from data/validation_videos/000352b9-5884-4f82-8153-7ef794979ee5.mp4
                    video_length = 16
                    pixel_values = load_video_as_tensor("data/validation_videos/000352b9-5884-4f82-8153-7ef794979ee5.mp4",
                                                        max_frames=video_length*4,).to("cuda")
                    print("pixel_values.shape: ", pixel_values.shape)

                    # assume (B, C, F, H, W)
                    # crop to 256 x 256
                    pixel_values = pixel_values[:, :, :video_length, :256, :256]
                    print("pixel_values.shape: ", pixel_values.shape)

                    
                    # encode it to latents
                    with torch.no_grad():
                        validate_timesteps = [999, 998, 997]

                        pixel_values = rearrange(pixel_values, "b c f h w -> (b f) c h w")
                        latents = vae.encode(pixel_values).latent_dist
                        latents = latents.sample()
                        latents = latents * 0.18215
                        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                        

                        for the_timestep in validate_timesteps:

                            timesteps = torch.tensor([the_timestep])

                            key = f"debug_output_{the_timestep}"

                            # create the target noise
                            target = torch.randn_like(latents)

                            # add noise to the latents
                            noisy_latents = noise_scheduler.add_noise(latents, target, timesteps)

                            # predict the noise
                            model_pred = unet(noisy_latents, the_timestep, encoder_hidden_states).sample

                            # call validate_prediction
                            validate_prediction(
                                data=data,
                                noise_scheduler=noise_scheduler,
                                vae=vae,
                                output_dir=output_dir,
                                the_timestep=the_timestep,
                                latents=latents,
                                pixel_values=pixel_values,
                                noisy_latents=noisy_latents,
                                model_pred=model_pred,
                                target=target,
                                key=key,
                                actual_step=actual_step,
                                save_images=actual_step % 100 == 1)
                
                print("data ", data)
                wandb.log(data, step=actual_step)



            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataloader) - 1) and global_step != 0:
                save_path = os.path.join(output_dir, f"checkpoints")
                if image_finetune:
                    trained_model = unet
                else:
                    trained_model = extract_motion_module(unet)
                
                if step == len(train_dataloader) - 1:
                    torch.save(trained_model, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                else:
                    torch.save(trained_model, os.path.join(save_path, f"checkpoint.ckpt"))
                
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                    
            # Periodically validation
            if is_main_process and (actual_step % validation_steps == 0 or actual_step in validation_steps_tuple) and step > 2:
                samples = []
                
                generator = torch.Generator(device=latents.device)
                # generator = torch.Generator(device="cpu")
                generator.manual_seed(global_seed)
                
                height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

                prompts = validation_data.prompts[:2] if global_step < 1000 and (not image_finetune) else validation_data.prompts

                for idx, prompt in enumerate(prompts):
                    if not image_finetune:
                        window_length = train_data.sample_n_frames
                        sample = validation_pipeline(
                            prompt,
                            generator    = generator,
                            video_length = train_data.sample_n_frames,
                            height       = 256,
                            width        = 256,
                            # window_count = window_length,
                            # wrap_around=True,
                            # alternate_direction=False,
                            # min_offset = window_length // 2,
                            # max_offset = window_length // 2,
                            # reference_attn = False,
                            **validation_data,
                        ).videos
                        output_path = f"{output_dir}/samples/sample-{global_step}/{idx}.gif"
                        save_videos_grid(sample, output_path)
                        fft_output_path = f"{output_dir}/samples/sample-{global_step}/fft-{idx}.gif"
                        save_power_spectrum_as_gif(sample, fft_output_path)
                        samples.append(sample)
                        wandb.log({"validation_images": wandb.Image(output_path)})
                        wandb.log({"validation_fft": wandb.Image(fft_output_path)})
                        
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
                
            logs = {"step_loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            
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