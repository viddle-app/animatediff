"""
TODO: fix training mixed precision -- issue with AdamW optimizer
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, DatasetDict
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from src.models.vae import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from decord import VideoReader
from einops import rearrange
import bitsandbytes as bnb
import io
from src.utils.util import save_videos_grid
from collections import OrderedDict

import lpips
from PIL import Image

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def log_validation_bk(test_dataloader, vae, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    vae_model = accelerator.unwrap_model(vae)
    images = []
    for _, sample in enumerate(test_dataloader):
        x = sample["pixel_values"].to(weight_dtype)
        print('x shape: ', x.shape)
        x = rearrange(x, "b c f h w -> b f c h w")
        reconstructions = vae_model(x).sample
        cpu_x = rearrange(sample["pixel_values"].cpu(), "b f c h w -> b c f h w")
        # select the first frame (dim 2) of the batch
        x_first_frame = cpu_x[:, :, 0, :, :]
        print("x_first_frame shape: ", x_first_frame.shape)
        reconstructions_first_frame = reconstructions[:, :, 0, :, :]
        images.append(
            torch.cat([x_first_frame, reconstructions_first_frame.cpu()], axis=0)
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "Original (left) / Reconstruction (right)", np_images, epoch
            )
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "Original (left) / Reconstruction (right)": [
                        wandb.Image(torchvision.utils.make_grid(image))
                        for _, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.gen_images}")

    del vae_model
    torch.cuda.empty_cache()

@torch.no_grad()
def log_validation(test_dataloader, vae, accelerator, weight_dtype, global_step, output_dir):
    logger.info("Running validation... ")


    vae_model = accelerator.unwrap_model(vae)
    images = []
    for idx, sample in enumerate(test_dataloader):
        x = sample["pixel_values"].to(weight_dtype)
        print('x shape: ', x.shape)
        x = rearrange(x, "b c f h w -> b f c h w")
        reconstructions = vae_model(x).sample
        cpu_x = rearrange(sample["pixel_values"].cpu(), "b f c h w -> b c f h w")

        # concat the reconstructions and the cpu_x next to each other along the w dimension

        concated = torch.cat([cpu_x, reconstructions.cpu()], dim=4)
        images.append(concated)

    images = torch.cat(images, dim=0)
    print("images shape: ", images.shape)

    output_path = f"{output_dir}/samples/sample-{global_step}/{idx}.gif"
    save_videos_grid(images, output_path, rescale=True)
    wandb.log({"validation_images": wandb.Image(output_path)})
        
    del vae_model
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a VAE training script."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vae-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=4,
        help="Number of images to remove from training set to be used as validation.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="vae-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--kl_scale",
        type=float,
        default=1e-6,
        help="Scaling factor for the Kullback-Leibler divergence penalty term.",
    )
    parser.add_argument(
        "--lpips_scale",
        type=float,
        default=1e-1,
        help="Scaling factor for the LPIPS metric",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args

def extract_motion_module(unet):
    mm_state_dict = OrderedDict()
    state_dict = unet.state_dict()
    state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    for key in state_dict:
        if "motion_module" in key:
            mm_state_dict[key] = state_dict[key]

    return mm_state_dict

def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load vae
    vae = AutoencoderKL.from_pretrained_2d(
        args.pretrained_model_name_or_path, 
        subfolder="vae",
        vae_additional_kwargs = {
            "use_motion_module" : True,
            "motion_module_resolutions"      : [ 1,2,4,8 ],
            "motion_module_type": "Vanilla",
            "use_inflated_groupnorm": False,
            "motion_module_kwargs": {
                "num_attention_heads"                : 8,
                "num_transformer_block"              : 1,
                "attention_block_types"              : [ "Temporal_Self", "Temporal_Self", ],
                "temporal_position_encoding"         : True,
                "temporal_position_encoding_max_len" : 32,
                "temporal_attention_dim_div"         : 1,
                "zero_initialize"                    : True,
                "upcast_attention"                   : False,
            } 
        # revision=args.revision
        }
    )
    vae.requires_grad_(False)
    
    # with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
    vae_params = []
    trainable_modules = ["motion_module"]
    for name, param in vae.named_parameters():
        
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                print(name)
                param.requires_grad = True
                vae_params.append(param)
        
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(vae_params, lr=args.learning_rate)
    # optimizer = bnb.optim.AdamW8bit(vae_params, lr=args.learning_rate)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        # dataset = load_dataset(
        #    args.dataset_name,
        #     args.dataset_config_name,
        #    cache_dir=args.cache_dir,
        #)
        dataset_dict = DatasetDict.load_from_disk(args.dataset_name)
        dataset = dataset_dict
        ## TODO just load the video data in image finetuneing mode
        ## then modify the vae to be 3d and load the video data
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )

    column_names = dataset["train"].column_names
    if args.image_column is None:
        image_column = column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )

    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomCrop(args.resolution),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess(examples):
        with torch.no_grad():
            image_filepaths = examples[image_column]
            print("image_filepaths: ", image_filepaths)
            valid_images = []
            sample_stride=4
            sample_n_frames=16

            for idx, image_filepath in enumerate(image_filepaths):
                video_reader = VideoReader(image_filepath)
                video_length = len(video_reader)
                
                clip_length = min(video_length, (sample_n_frames - 1) * sample_stride + 1)
                start_idx   = random.randint(0, video_length - clip_length)
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, sample_n_frames, dtype=int)
                pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
                del video_reader

                valid_images.append(pixel_values)
                    

            transformed_images = [train_transforms(image) for image in valid_images]
            print("transformed_images: ", transformed_images[0].shape)
            examples["pixel_values"] = transformed_images

            return examples

    with accelerator.main_process_first():
        # Split into train/test
        dataset = dataset["train"].train_test_split(test_size=args.test_samples)
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess)
        test_dataset = dataset["test"].with_transform(preprocess)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, collate_fn=collate_fn
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.num_train_epochs * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        vae,
        optimizer,
        train_dataloader,
        test_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        vae, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # ------------------------------ TRAIN ------------------------------ #
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num test samples = {len(test_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    global_step = 0
    first_epoch = 0

        # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)


    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    lpips_loss_fn = lpips.LPIPS(net="alex").to(accelerator.device)

    for epoch in range(first_epoch, args.num_train_epochs):
        vae.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(vae):
                target = batch["pixel_values"].to(weight_dtype)
                target = rearrange(target, "b f c h w -> b c f h w")
                print("target: ", target.shape)

                # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py
                posterior = vae.encode(target).latent_dist
                z = posterior.mode()
                pred = vae.decode(z).sample

                kl_loss = posterior.kl().mean()
                mse_loss = F.mse_loss(pred, target, reduction="mean")
                # rearrange for the lpips loss
                lips_pred = rearrange(pred, "b c f h w -> (b f) c h w")
                lips_target = rearrange(target, "b c f h w -> (b f) c h w")
                lpips_loss = lpips_loss_fn(lips_pred, lips_target).mean()

                loss = (
                    mse_loss + args.lpips_scale * lpips_loss + args.kl_scale * kl_loss
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                print("train_loss: ", train_loss)
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            else:
                print("not a sync_gradients step")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "mse": mse_loss.detach().item(),
                "lpips": lpips_loss.detach().item(),
                "kl": kl_loss.detach().item(),
            }
            accelerator.log(logs)
            progress_bar.set_postfix(**logs)

            if accelerator.is_main_process:
                if step % args.validation_steps == 0:
                    with torch.no_grad():
                        log_validation(test_dataloader, 
                                       vae, 
                                       accelerator, 
                                       weight_dtype, 
                                       step,
                                       output_dir=args.output_dir)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        vae.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()