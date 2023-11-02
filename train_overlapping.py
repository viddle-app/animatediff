# originally copied from: 
import json
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
from diffusers import AutoencoderKL, DDIMScheduler, PNDMScheduler
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
from diffusers.models.attention_processor import Attention
from typing import Callable, Optional, Union



from src.models.unet import UNet3DConditionModel
# from src.pipelines.pipeline_animatediff import AnimationPipeline
# from src.pipelines.pipeline_animatediff_overlapping_previous_2 import AnimationPipeline
# from src.pipelines.pipeline_animatediff_overlapping_previous import AnimationPipeline
from src.pipelines.pipeline_animatediff import AnimationPipeline
from src.utils.util import save_videos_grid, zero_rank_print
# from src.data.dataset_overlapping import WebVid10M
from src.data.dataset_cached_latents import WebVid10M
from collections import OrderedDict
import xformers
import xformers.ops

class XFormersAttnProcessor_Scaled:
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        _, query_tokens, _ = hidden_states.shape
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        numerator = math.log(query_tokens) / math.log(query_tokens//4)
        attention_scale = math.sqrt(numerator / (attn.inner_dim // attn.heads))

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attention_scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states



def diff_from_first_frame(video_tensor):
    """
    Compute the difference between each frame and the first frame.
    video_tensor should have shape (batch_size, channels, frames, height, width)
    """
    # Get the first frame
    first_frame = video_tensor[:, :, 0:1]

    # Compute the difference between each frame and the first frame
    diff_from_first_frame = video_tensor[:, :, 1:] - first_frame
    # add the first frame to then front of the tensor
    diff_from_first_frame = torch.cat([first_frame, diff_from_first_frame], dim=2)
    return diff_from_first_frame


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
                        save_images,
                        log_images_to_wandb=True,
                        first_frame_weight=1.0,
                        ):
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
    
    # target_diff = diff_from_first_frame(target)[:, :, 1:]
    # model_pred_diff = diff_from_first_frame(model_pred)[:, :, 1:]

    target_diff = diff_from_first_frame(target)
    model_pred_diff = diff_from_first_frame(model_pred)
    
    diff_error = (target_diff - model_pred_diff)
    # multiple the first frame difference by the weight
    diff_error[:, :, 0] = diff_error[:, :, 0] * first_frame_weight
    diff_error = diff_error**2



    reconstruction_loss = error_tensor.mean()
    diff_loss = diff_error.mean()
    # loss = 0.5 * reconstruction_loss + 0.5*(diff_loss + noisy_diff_loss)
    snr = compute_snr(noise_scheduler, torch.tensor([the_timestep]).to(noisy_latents.device))
    snr_scale = 1/torch.sqrt(snr)
    loss = snr_scale * diff_loss
    # TODO log the losses
    
    data[f"validation_reconstruction_loss_{the_timestep}"] = reconstruction_loss.item()
    data[f"validation_noisy_diff_loss_{the_timestep}"] = diff_loss.item()
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

        # decode the diff noise error
        diff_error = 1 / vae.config.scaling_factor * diff_error
        diff_error = rearrange(diff_error, "b c f h w -> (b f) c h w")
        diff_error = vae.decode(diff_error, return_dict=False)[0]

        # add a frame of zeros
        # diff_error = torch.cat([torch.zeros_like(diff_error[0]).unsqueeze(0), diff_error], dim=0)

        target_diff_pixels = target_diff
        target_diff_pixels = 1 / vae.config.scaling_factor * target_diff_pixels
        target_diff_pixels = rearrange(target_diff_pixels, "b c f h w -> (b f) c h w")
        target_diff_pixels = vae.decode(target_diff_pixels, return_dict=False)[0]

        model_pred_diff_pixels = model_pred_diff
        model_pred_diff_pixels = 1 / vae.config.scaling_factor * model_pred_diff_pixels
        model_pred_diff_pixels = rearrange(model_pred_diff_pixels, "b c f h w -> (b f) c h w")
        model_pred_diff_pixels = vae.decode(model_pred_diff_pixels, return_dict=False)[0]


        # decode the error tensor
        error_tensor = 1 / vae.config.scaling_factor * error_tensor
        error_tensor = rearrange(error_tensor, "b c f h w -> (b f) c h w")
        error_tensor = vae.decode(error_tensor, return_dict=False)[0]

        # compute the 2d fft log spectrum for the pixel values and the pred and the error
        pred_diff = pred[1:,] - pred[:-1]
        # add a zero frame to start 
        pred_diff = torch.cat([torch.zeros_like(pred_diff[0]).unsqueeze(0), pred_diff], dim=0)
        pixel_diff = pixel_values[1:] - pixel_values[:-1]
        # add a zero frame to start
        pixel_diff = torch.cat([torch.zeros_like(pixel_diff[0]).unsqueeze(0), pixel_diff], dim=0)

        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        target = (target / 2 + 0.5).clamp(0, 1)
        pred = (pred / 2 + 0.5).clamp(0, 1)
        diff_error_pixels = (diff_error / 2 + 0.5).clamp(0, 1)
        error_tensor_pixels = (error_tensor / 2 + 0.5).clamp(0, 1)
        target_pixels = (target_pixels / 2 + 0.5).clamp(0, 1)
        pred_diff = (pred_diff / 2 + 0.5).clamp(0, 1)
        pixel_diff = (pixel_diff / 2 + 0.5).clamp(0, 1)

        # log scale the diff_error_pixels and the error_tensor_pixels
        diff_error_pixels = torch.log1p(diff_error_pixels)
        error_tensor_pixels = torch.log1p(error_tensor_pixels)

        # concat the tensors along the W dimension
        combined = torch.cat([pixel_values, 
                            target_pixels, 
                            pred, 
                            error_tensor_pixels, 
                            diff_error_pixels,
                            target_diff_pixels,
                            model_pred_diff_pixels,
                            ], dim=3)

        output_path = f"{folder_path}/{actual_step}_{the_timestep}.gif"
        save_gif_from_tensor(combined, output_path)
        # save a folder of images
        image_folder_path = f"{folder_path}/{actual_step}_{the_timestep}"
        tensor_to_image_sequence(combined.permute(1, 0, 2, 3), image_folder_path)

        if log_images_to_wandb:
            wandb.log({key: wandb.Image(output_path)}, step=actual_step)




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

def video_diff_loss_2(original_video, generated_video):
    # first compute the squared difference between the original and the generated
    squared_differences = torch.abs(original_video - generated_video)
    # drop the first frame
    squared_differences = squared_differences[:, :, 1:]

    # now compute the frame difference of the original
    original_diff = frame_diff(original_video)
    # now compute the frame difference of the generated
    generated_diff = frame_diff(generated_video)

    # take the difference between the two
    difference_difference = torch.abs(original_diff - generated_diff)

    # take the element wise multiplication of the squared_differences and the difference_difference
    return (squared_differences * difference_difference).mean()

def video_diff_loss_3(original_video, generated_video):
    # first compute the squared difference between the original and the generated
    squared_differences = (original_video - generated_video)**2
    # drop the first frame
    squared_differences = squared_differences[:, :, 1:]

    # now compute the frame difference of the original
    original_diff = frame_diff(original_video)
    # now compute the frame difference of the generated
    generated_diff = frame_diff(generated_video)

    # take the difference between the two
    difference_difference = (original_diff - generated_diff)**2
    # take the element wise multiplication of the squared_differences and the difference_difference
    return (squared_differences * difference_difference).mean()

def video_diff_loss_4(original_video, generated_video, first_frame_weight=1.0):
    # diff both from the first frame
    original_diff = diff_from_first_frame(original_video)
    # compute the 
    generated_diff = diff_from_first_frame(generated_video)

    # take the difference between the two
    difference_difference = original_diff - generated_diff

    # multiple the first frame difference by the weight
    difference_difference[:, :, 0] = difference_difference[:, :, 0] * first_frame_weight

    # take the mse loss
    return (difference_difference**2).mean()

def video_diff_loss_5(original_video, generated_video, first_frame_weight=1.0):
    # diff both from the first frame
    original_diff = diff_from_first_frame(original_video)[:, :, 1:]
    generated_diff = diff_from_first_frame(generated_video)[:, :, 1:]

    # take the difference between the two
    difference_difference = original_diff - generated_diff

    # take the mse loss
    return (difference_difference**2).mean()

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

def get_lr_6(timestep):
    if timestep < 100:
        return 5e-12
    elif timestep < 200:
        return 1e-11
    elif timestep < 300:
        return 5e-11
    elif timestep < 400:
        return 1e-10
    elif timestep < 500:
        return 5e-10
    elif timestep < 600:
        return 1e-9
    elif timestep < 700:
        return 5e-9
    elif timestep < 800:
        return 1e-9
    elif timestep < 900:
        return 5e-8
    else:
        return 1e-8
        


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
    learning_rate_reading_steps: int = None,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,
    first_frame_weight = 1.0,
    progressive_noise = None,
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
        # create the learning_rate file 
        with open(f"{output_dir}/learning_rate.txt", "w") as f:
            f.write(str(lr))

    # Load scheduler, tokenizer and models.
    dtype = torch.float32
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    # noise_scheduler = PNDMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
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
            
            # fine_tuned_unet = torch.load("/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-23T09-54-03/checkpoints/checkpoint-epoch-1.ckpt", map_location="cpu")
            # fine_tuned_unet = torch.load("/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-11T11-46-02/checkpoints/checkpoint-epoch-1.ckpt", map_location="cpu")
            # fine_tuned_unet = torch.load("/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-25T12-47-50/checkpoints/checkpoint-epoch-1.ckpt")
            # unet = UNet3DConditionModel.from_unet2d(fine_tuned_unet, unet_additional_kwargs=unet_additional_kwargs)
            unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, 
                                                     subfolder="unet",
                                                     unet_additional_kwargs=unet_additional_kwargs,)
        
    # unet.set_attn_processor(XFormersAttnProcessor_Scaled())
    print(f"state_dict keys: {list(unet.config.keys())[:10]}")

    # global_step = 0
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
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-16T14-07-53/checkpoints/checkpoint.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-17T11-22-06/checkpoints/checkpoint.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-17T02-08-28/checkpoints/checkpoint.ckpt"
        # otion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-19T14-17-48/checkpoints/checkpoint.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-18T06-58-55/checkpoints/checkpoint.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-19T19-53-19/checkpoints/checkpoint.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-20T16-12-28/checkpoints/checkpoint.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-21T15-05-53/checkpoints/checkpoint.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-26T05-53-31/checkpoints/checkpoint.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-27T14-21-27/checkpoints/checkpoint-epoch-2.ckpt"
        # motion_module_path = "/mnt/newdrive/viddle-animatediff/outputs/training-2023-10-28T01-33-45/checkpoints/checkpoint-epoch-9.ckpt"

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
                    print(name)
                    param.requires_grad = True

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
        step_rules="0.01:5,0.02:5,0.04:5,0.08:5,0.1:50,0.2:50,0.4:50,0.8:50,1.0:40000,0.8:2000,0.6:2000,0.4:2000,0.3"
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
            latents = batch["pixel_values"].to(local_rank)
            latents = rearrange(latents, "b f c h w -> b c f h w")
            video_length = latents.shape[2]
            print("video_length: ", video_length)

            print("latents.shape: ", latents.shape)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)

            if progressive_noise is not None:
                shape = noise.shape
                shape = (shape[0], shape[1], 1, shape[3], shape[4])
                shared_noise = torch.randn(shape, device=noise.device)
                noise = (1 - progressive_noise) * noise + shared_noise * progressive_noise

                std = torch.std(noise, dim=(0,1,3,4))
                noise = std.reshape(1,1,-1,1,1) * noise

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

                # timesteps = torch.tensor([999]).to(latents.device)

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
            
            

            # update the lr of the optimizer. I'm assuming no gradient accumulation here
            # lr = get_lr_6(the_timestep)

            # print("lr: ", lr)

            # for param_group in optimizer.param_groups:
            #  param_group['lr'] = lr

            # if the_timestep != 0 and video_length == 16:
            #    continue
            #elif the_timestep == 0 and video_length == 16 and step % 10 != 0:
            #    continue

        
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
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
                model_pred_float = model_pred.float()
                target_float = target.float()

                reconstruction_diff = (model_pred_float - target_float)**2
                reconstruction_loss = reconstruction_diff.mean()
            

                the_frame_diff_loss_noise = video_diff_loss(model_pred, target)
                
                # the_frame_diff_loss_noise = nn.L1Loss()(pred.float(), latents.float())
                snr = compute_snr(noise_scheduler, timesteps)
                snr_scale = 1 / torch.sqrt(snr)
                # the loss is the two losses dot producted
         
                # loss = normality_loss + the_frame_diff_loss
                if not image_finetune:
                    # pred = (noisy_latents - model_pred * sqrt_one_minus_alpha_prod ) / sqrt_alpha_prod 
                    # loss = (snr_scale.reshape(-1, 1, 1, 1, 1) * 
                    #        video_diff_loss_6(model_pred_float, 
                    #                         target_float, 
                    #                         first_frame_weight=first_frame_weight)).mean()
                    loss = video_diff_loss_6(model_pred_float, 
                                             target_float, 
                                             first_frame_weight=first_frame_weight).mean()
                    # loss = snr_scale * reconstruction_loss

                else:
                    the_loss = (model_pred - target) ** 2
                    the_loss = snr_scale.reshape(-1, 1, 1, 1) * the_loss
                    loss = the_loss.mean()
                

                


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

            if mixed_precision_training:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if actual_step % gradient_accumulation_steps == gradient_accumulation_steps - 1:
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

                # Compute statistics
                # if actual_step % 200 == 199:
                #    for name, param in unet.named_parameters():
                #        if param.grad is not None:
                #            wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().detach().numpy())})
                    
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
                if learning_rate_reading_steps is None:
                    lr_scheduler.step()
                progress_bar.update(1)
                global_step += 1
                # optimizer_states[step_index//100] = optimizer.state_dict()


            ### <<<< Training <<<< ###
            actual_step += 1

            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log

                data = {}
                for the_timestep in timesteps:

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

                    data[f"{prefix}train_loss"] = loss.item()
                    data[f"{prefix}reconstruction_loss"] = reconstruction_loss
                    data[f"{prefix}the_frame_diff_loss_noise"] = the_frame_diff_loss_noise
                    data[f"{prefix}mean_gradient"] = mean_grad
                    data[f"{prefix}variance_gradient"] = var_grad
                    
                
                data["loss"] = loss.item()
                data["reconstruction_loss"] = reconstruction_loss
                data["the_frame_diff_loss_noise"] = the_frame_diff_loss_noise
                data["mean_gradient"] = mean_grad
                data["variance_gradient"] = var_grad
                data["lr"] = optimizer.param_groups[0]["lr"]
                
                
                if len(grad_magnitudes) > 0:
                    data[f"{prefix}max_gradient"] = max(grad_magnitudes)
                    data[f"{prefix}min_gradient"] = min(grad_magnitudes)
                    data["max_gradient"] = max(grad_magnitudes)
                    data["min_gradient"] = min(grad_magnitudes)

                # for each predicted_frequency_energy and target_frequency_energy
                # add an entry to data

    
                csv_path = f"{output_dir}/stats.csv"
                append_loss_to_csv(actual_step, loss.item(), csv_path)

                if not image_finetune:
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
                        validate_timesteps = [999, 499, 59]

                        pixel_values = rearrange(pixel_values, "b c f h w -> (b f) c h w")
                        latents = vae.encode(pixel_values).latent_dist
                        latents = latents.sample()
                        latents = latents * 0.18215
                        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                        

                        for the_timestep in validate_timesteps:

                            timesteps = torch.tensor([the_timestep])

                            key = f"debug_output_{the_timestep}"
                            
                            generator = torch.Generator(device=latents.device)
                            # generator = torch.Generator(device="cpu")
                            generator.manual_seed(global_seed)

                            # create the target noise
                            if progressive_noise is not None:
                                shape = latents.shape
                                shape = (shape[0], shape[1], 1, shape[3], shape[4])
                                shared_noise = torch.randn(shape, generator=generator, device=latents.device)
                                target = torch.randn(latents.shape, generator=generator, device=latents.device)
                                # use the progress noise as weight and 
                                target = progressive_noise*shared_noise.repeat(1, 1, 16, 1, 1) + (1 - progressive_noise)*target
                                # compute the means and std for each frame, e.g. dim=2
                                
                                stds = target.std(dim=(0,1,3,4))
                                # divide to the stds to normalize
                                target = target / stds.reshape(1,1,-1,1,1)
                            else:
                                target = torch.randn(latents.shape, generator=generator, device=latents.device)

                            # add noise to the latents
                            noisy_latents = noise_scheduler.add_noise(latents, target, timesteps)

                            # predict the noise
                            with torch.no_grad():
                                prompt_ids = tokenizer(
                                    "Purple basketball, hardwood floor, bedroom, clothes on floor, hand flickering", 
                                    max_length=tokenizer.model_max_length, 
                                    padding="max_length", 
                                    truncation=True, 
                                    return_tensors="pt"
                                ).input_ids.to(latents.device)
                                encoder_hidden_states = text_encoder(prompt_ids)[0]
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
                                save_images=actual_step % 400 == 1,
                                log_images_to_wandb=True)
                
                wandb.log(data, step=actual_step)

            
            new_lr = False

            if learning_rate_reading_steps is not None:
                if global_step % learning_rate_reading_steps == 0:
                    # read the learning rate from the learning rate file
                    with open(f"{output_dir}/learning_rate.txt", "r") as f:
                        # read the lr as a dictionary of "timestep, lr" stored as json
                        lr_dict = json.load(f)

                        # compare against old lr
                        old_lr = optimizer.param_groups[0]["lr"]
                        print("read_lr: ", read_lr)
                        print("old_lr: ", old_lr)

                        new_lr = read_lr != optimizer.param_groups[0]["lr"]
                        print("new_lr: ", new_lr)

                        # set the new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = read_lr
                        if new_lr: 
                            # save the lr to the lr log file
                            with open(f"{output_dir}/lr_log.txt", "a") as f:
                                f.write(f"{global_step},{read_lr}\n")


            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataloader) - 1) and global_step != 0 or new_lr:
                save_path = os.path.join(output_dir, f"checkpoints")
                if image_finetune:
                    trained_model = unet
                else:
                    trained_model = extract_motion_module(unet)
                
                if step == len(train_dataloader) - 1:
                    torch.save(trained_model, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                elif new_lr:
                    torch.save(trained_model, os.path.join(save_path, f"checkpoint-step-{global_step}.ckpt"))
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