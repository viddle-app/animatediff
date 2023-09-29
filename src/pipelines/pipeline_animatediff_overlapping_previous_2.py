# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py
# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
from typing import Callable, List, Optional, Union, Tuple, Dict, Any    
from dataclasses import dataclass

import PIL

import numpy as np
import torch
from tqdm import tqdm

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.loaders import FromSingleFileMixin
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput, randn_tensor, is_compiled_module

from einops import rearrange
from ..utils.partition_utils import partition_sliding, partition_sliding_2, partition_wrap_around_2, partitions, partitions_wrap_around
from diffusers.image_processor import VaeImageProcessor
from ..models.unet import UNet3DConditionModel
from ..models.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from ..models.attention import BasicTransformerBlock

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

class AnimationPipeline(DiffusionPipeline, FromSingleFileMixin):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        controlnet: Optional[Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel]] = None,

    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents, device):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")

        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            decoded = self.vae.decode(latents[frame_idx:frame_idx+1].to(device)).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            video.append(decoded.to("cpu"))
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, 
                        num_channels_latents, 
                        video_length, height, 
                        width, dtype, device, 
                        generator, window_size, 
                        latents=None, 
                        do_init_noise=True):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != video_length:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective video_length"
                f" size of {video_length}. Make sure the video_length size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        # check if the init_noise_sigma is a tensor
        if do_init_noise:
            if isinstance(self.scheduler.init_noise_sigma, torch.Tensor):
                latents = latents * self.scheduler.init_noise_sigma.to(device)
            else:
                latents = latents * self.scheduler.init_noise_sigma
        
        return latents

    def check_image(self, image, prompt, prompt_embeds, video_length):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_np = isinstance(image, np.ndarray)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size * video_length:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )


    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        window_count=24,
        wrap_around = True,
        min_offset = 3,
        max_offset = 5,
        offset_generator = None,
        alternate_direction = True,
        do_init_noise: bool = True,
        timesteps: Optional[torch.Tensor] = None,
        image: Optional[Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        num_images_per_prompt = 1,
        cpu_device = torch.device("cpu"),
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        reference_attn = True,
        **kwargs,

    ):
        print("window_count: ", window_count)
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        window_length = min(video_length, window_count)

        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            cpu_device,
            generator,
            window_length,
            latents,
            do_init_noise=do_init_noise,
        )
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        if offset_generator is None:
            offset_generator = generator
        
        controlnet = None
        if self.controlnet is not None:
            controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

            # align format for control guidance
            if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
                control_guidance_start = len(control_guidance_end) * [control_guidance_start]
            elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
                control_guidance_end = len(control_guidance_start) * [control_guidance_end]
            elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
                mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
                control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                    control_guidance_end
                ]
            if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)
            global_pool_conditions = (
                controlnet.config.global_pool_conditions
                if isinstance(controlnet, ControlNetModel)
                else controlnet.nets[0].config.global_pool_conditions
            )
            guess_mode = guess_mode or global_pool_conditions

        
            if isinstance(controlnet, ControlNetModel):
                image = self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=cpu_device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                height, width = image.shape[-2:]
                print("image: ", image.shape)
                image = rearrange(image, "(b f) c h w -> b c f h w", f=video_length)
            elif isinstance(controlnet, MultiControlNetModel):
                images = []

                for image_ in image:
                    image_ = self.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=cpu_device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )
                    image_ = rearrange(image_, "(b f) c h w -> b c f h w", f=video_length)
                    images.append(image_)

                image = images
                height, width = image[0].shape[-2:]
            else:
                assert False

            # 7.1 Create tensor stating which controlnets to keep
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        remainder = 1 if video_length % window_length != 0 else 0
        partition_count = video_length // window_length  + remainder
        min_offset = max(min_offset, 0)
        max_offset = min(max_offset, window_length)
        min_offset = min(min_offset, max_offset)
        max_offset = max(min_offset, max_offset)
        offset = min_offset
        first_window_size = window_length - offset

        MODE = "write"
        def hacked_basic_transformer_inner_forward(
                self,
                hidden_states: torch.FloatTensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                timestep: Optional[torch.LongTensor] = None,
                cross_attention_kwargs: Dict[str, Any] = None,
                class_labels: Optional[torch.LongTensor] = None,
                video_length: Optional[int] = None,
            ):
                if self.use_ada_layer_norm:
                    norm_hidden_states = self.norm1(hidden_states, timestep)
                elif self.use_ada_layer_norm_zero:
                    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                        hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                    )
                else:
                    norm_hidden_states = self.norm1(hidden_states)

                # 1. Self-Attention
                cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
                if self.only_cross_attention:
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                else:
                    if MODE == "write":
                        rearrange_norms = rearrange(norm_hidden_states, "(b f) d c -> b f d c", f=norm_hidden_states.shape[0] // 2)
                        # to_save = rearrange_norms[:, 0:1, :, :].detach().clone().repeat(1, first_window_size, 1, 1)
                        to_save = rearrange_norms[:, 0:1, :, :].detach().clone()
                        # rearrange_to_save = rearrange(to_save, "b f d c -> (b f) d c")
                        self.bank = to_save

                    
                    rearranged = rearrange(self.bank.repeat(1, norm_hidden_states.shape[0] // 2, 1, 1), "b f d c -> (b f) d c")

                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=torch.cat([norm_hidden_states, rearranged] , dim=1),
                        # attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )


                if self.use_ada_layer_norm_zero:
                    attn_output = gate_msa.unsqueeze(1) * attn_output
                hidden_states = attn_output + hidden_states

                if self.attn2 is not None:
                    norm_hidden_states = (
                        self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                    )

                    # 2. Cross-Attention
                    attn_output = self.attn2(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=encoder_attention_mask,
                        **cross_attention_kwargs,
                    )
                    hidden_states = attn_output + hidden_states

                # 3. Feed-forward
                norm_hidden_states = self.norm3(hidden_states)

                if self.use_ada_layer_norm_zero:
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

                ff_output = self.ff(norm_hidden_states)

                if self.use_ada_layer_norm_zero:
                    ff_output = gate_mlp.unsqueeze(1) * ff_output

                hidden_states = ff_output + hidden_states

                return hidden_states

        if reference_attn:
            attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]
            attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.bank = []

        # start_
        # wrong but close

        images_partition = None
        # Denoising loop
        print("window_length: ", window_length)
        print("video_length: ", video_length)
        print("min_offset: ", min_offset)
        print("max_offset: ", max_offset)
        latents_shape = latents.shape
        # latents_shape = (2,) + latents_shape[1:]
        count  = torch.zeros(latents_shape,
                             dtype=latents.dtype)
        values = torch.zeros(latents_shape,
                             dtype=latents.dtype)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps * partition_count) as progress_bar:
            for i, t in enumerate(timesteps):

                print("t: ", t)
                print("i: ", i)
                count.zero_()
                values.zero_()
                # if hasattr(self.unet, 'clear_last_encoder_hidden_states'):
                #     self.unet.clear_last_encoder_hidden_states()
                # 
                # if video_length == window_length:
                #     offset = 0
                # else:
                #     if min_offset == max_offset:
                #         offset = min_offset
                #     else:
                #         random_int_tensor = torch.randint(min_offset, max_offset, (1,), generator=offset_generator)
                #         offset = random_int_tensor.item()

                print("offset: ", offset)
                if wrap_around == True:
                    indices = partition_sliding_2(video_length, window_length, offset)
                else:
                    indices = partition_sliding_2(video_length, window_length, offset)
                print("indices: ", indices)
                if i % 2 == 1 and alternate_direction == True:
                    indices = reversed(indices)
                    self.unet.set_forward_direction(False)
                else:
                    self.unet.set_forward_direction(True)
                for p, partition_indices in enumerate(indices):
                    if p == 0:
                        MODE = "write"
                    else:
                        MODE = "read"

                    print("partition_indices: ", partition_indices)
                    # check if the partition_indices is a list or a tuple
                    if isinstance(partition_indices, tuple):
                        # make a list of the partition indices
                        start_interval = partition_indices[0]
                        end_interval = partition_indices[1]

                        # turn the start interval into a list of indices
                        start_indices = list(range(start_interval[0], start_interval[1]))
                        end_indices = list(range(end_interval[0], end_interval[1]))
                        # combine the lists
                        partition_indices_expanded = end_indices + start_indices 
                        print("partition_indices_expanded: ", partition_indices_expanded)

                        latent_partition = latents[:, :, partition_indices_expanded].to(device)
                    else:
                        partition_indices_expanded = list(range(partition_indices[0], partition_indices[1]))
                        latent_partition = latents[:, :, partition_indices[0]:partition_indices[1]].to(device)
                    print("latent_partition: ", latent_partition.shape)


                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latent_partition] * 2) if do_classifier_free_guidance else latent_partition
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if controlnet is not None:
                        current_window_length = len(partition_indices_expanded) if isinstance(partition_indices, tuple) else partition_indices[1] - partition_indices[0]
                        print("current_window_length: ", current_window_length)
                        if isinstance(partition_indices, tuple):
                            if isinstance(self.controlnet, MultiControlNetModel):
                                image_partition = []
                                for img in image:
                                    
                                    image_partition.append(img[:, :, partition_indices_expanded].to(device))
                            else:
                                images_partition = image[:, :, partition_indices_expanded].to(device)
                        else:
                            if isinstance(self.controlnet, MultiControlNetModel):
                                image_partition = []
                                for img in image:
                                    print("img: ", img.shape)
                                    image_partition.append(img[:, :, partition_indices[0]:partition_indices[1]].to(device))
                            else:
                                images_partition = image[:, :, partition_indices[0]:partition_indices[1]].to(device)
                        if isinstance(self.controlnet, MultiControlNetModel):
                            new_partition = []
                            for img in image_partition:
                                new_partition.append(rearrange(img, "b c f h w -> (b f) c h w"))
                            
                            images_partition = new_partition
                        else:
                            images_partition = rearrange(images_partition, "b c f h w -> (b f) c h w")
                        # controlnet(s) inference
                        if guess_mode and do_classifier_free_guidance:
                            # Infer ControlNet only for the conditional batch.
                            control_model_input = latent_partition
                            control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                            controlnet_prompt_embeds = text_embeddings.chunk(2)[1]
                        else:
                            control_model_input = latent_model_input
                            controlnet_prompt_embeds = text_embeddings

                        if isinstance(controlnet_keep[i], list):
                            cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                        else:
                            controlnet_cond_scale = controlnet_conditioning_scale
                            if isinstance(controlnet_cond_scale, list):
                                controlnet_cond_scale = controlnet_cond_scale[0]
                            cond_scale = controlnet_cond_scale * controlnet_keep[i]


                        control_model_input = rearrange(control_model_input, "b c f h w -> (b f) c h w")
                        # duplicate the controlnet_prompt_embeds video_length times
                        controlnet_prompt_embeds = controlnet_prompt_embeds.repeat_interleave(current_window_length, dim=0)

                        if isinstance(cond_scale, torch.Tensor):
                            cond_scale = cond_scale.to(device=device, dtype=controlnet.dtype)
                            cond_scale = torch.cat([cond_scale] * 2) if do_classifier_free_guidance and not guess_mode else cond_scale

                        print("control_model_input", control_model_input.shape, control_model_input.dtype)
                        # print("images_partition", image.shape)
                        print("controlnet_prompt_embeds", controlnet_prompt_embeds.shape)
                        print("cond_scale", cond_scale)
                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            control_model_input,
                            t,
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=images_partition,
                            conditioning_scale=cond_scale,
                            guess_mode=guess_mode,
                            return_dict=False,
                        )

                        samples_batch = batch_size * 2 if do_classifier_free_guidance and not guess_mode else batch_size

                        # reshape the controlnet samples
                        down_block_res_samples = [
                            rearrange(d, "(b f) c h w -> b c f h w", b=samples_batch, f=current_window_length) for d in down_block_res_samples
                        ]
                        mid_block_res_sample = rearrange(
                            mid_block_res_sample, "(b f) c h w -> b c f h w", b=samples_batch, f=current_window_length
                        )

                        if guess_mode and do_classifier_free_guidance:
                            # Infered ControlNet only for the conditional batch.
                            # To apply the output of ControlNet to both the unconditional and conditional batches,
                            # add 0 to the unconditional batch to keep it unchanged.
                            down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                            mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                    step_percentage = i / len(timesteps)
                    should_use_ufree = 0.7 < step_percentage
                    backbone_scale_1 = 1.2 if should_use_ufree else 1.0
                    backbone_scale_2 = 1.4 if should_use_ufree else 1.0
                    skip_scale_1 = 0.0 if should_use_ufree else 1.0
                    skip_scale_2 = 0.0 if should_use_ufree else 1.0
                    skip_scale_threshold_1 = 1
                    skip_scale_threshold_2 = 2
                    # predict the noise residual
                    if controlnet is not None:
                        noise_pred = self.unet(
                                        latent_model_input,
                                        t,
                                        encoder_hidden_states=text_embeddings,
                                        cross_attention_kwargs=cross_attention_kwargs,
                                        down_block_additional_residuals=down_block_res_samples,
                                        mid_block_additional_residual=mid_block_res_sample,
                                        return_dict=False,
                                        # backbone_scale_1=backbone_scale_1,
                                        # backbone_scale_2=backbone_scale_2,
                                        # skip_scale_1=skip_scale_1,
                                        # skip_scale_2=skip_scale_2,
                                        # skip_scale_threshold_1=skip_scale_threshold_1,
                                        # skip_scale_threshold_2=skip_scale_threshold_2,

                                    )[0]
                    else:
                        noise_pred = self.unet(latent_model_input, 
                                            t, 
                                            encoder_hidden_states=text_embeddings,
                                            # backbone_scale_1=backbone_scale_1,
                                            # backbone_scale_2=backbone_scale_2,
                                            # skip_scale_1=skip_scale_1,
                                            # skip_scale_2=skip_scale_2,
                                            # skip_scale_threshold_1=skip_scale_threshold_1,
                                            # skip_scale_threshold_2=skip_scale_threshold_2,
                                            ).sample.to(dtype=latents_dtype)
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    if isinstance(generator, list):
                        current_generators = []
                        for idx in partition_indices_expanded: 
                            current_generators.append(generator[idx])
                    else:
                        current_generators = generator
                    extra_step_kwargs = self.prepare_extra_step_kwargs(current_generators, eta)
                    latent_partition = self.scheduler.step(noise_pred, t, latent_partition, **extra_step_kwargs).prev_sample

                    # update the latents

                    weights = torch.tensor(partition_indices[2]).reshape(-1, 1, 1).repeat(1, latents_shape[3], latents_shape[4]).to(cpu_device)
                    values[:, :, partition_indices[0]:partition_indices[1]] += weights * latent_partition.to(cpu_device)

                    print("weights: ", weights.shape)
                    count[:, :, partition_indices[0]:partition_indices[1]] += weights

                    if images_partition is not None:
                        del images_partition
                        torch.cuda.empty_cache()

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latent_partition)

                    if hasattr(self.unet, 'swap_next_to_last'):
                        self.unet.swap_next_to_last()
            
                latents = torch.where(count > 0, values / count, latents)


        # Post-processing
        video = self.decode_latents(latents, device)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)