from diffusers.models.attention import Attention
from diffusers.models.attention_processor import LoRALinearLayer

import torch
from torch import nn
import torch.nn.functional as F

def rearrange_3(tensor, f):
    F, D, C = tensor.size()
    return torch.reshape(tensor, (F // f, f, D, C))


def rearrange_4(tensor):
    B, F, D, C = tensor.size()
    return torch.reshape(tensor, (B * F, D, C))

class RandomCrossFrameAttnProcessor:
    def __init__(self, 
            batch_size=2, 
            generator=None,
            last_values_count=8,):
        if not hasattr(F, "scaled_dot_product_attention"):
                raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")


        self.batch_size = batch_size
        self.last_keys = []
        self.last_values = []

        self.next_keys = []
        self.next_values = []

        self.call_index = 0

        self.last_values_index = 0
        self.last_values_count = last_values_count

        self.is_self_attn = False
        self.has_bank = True

        if generator is None:
            self.generator = torch.Generator()
            self.generator.manual_seed(0)
        else:
            self.generator = generator

        self.do_cross_frame_attention = True

    def disable_cross_frame_attention(self):
        self.do_cross_frame_attention = False

    def enable_cross_frame_attention(self):
        self.do_cross_frame_attention = True


    def reset_last(self):
        self.last_keys = []
        self.last_values = []

    def clear(self):
        self.last_keys = []
        self.last_values = []

        self.next_keys = []
        self.next_values = []


    def swap_next_to_last(self):
        self.call_index = 0

        self.last_keys = self.next_keys
        self.last_values = self.next_values
        self.next_keys = []
        self.next_values = []

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        self.is_self_attn = True if encoder_hidden_states is None else self.is_self_attn
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        if self.is_self_attn and self.do_cross_frame_attention:
            
            if self.has_bank:
                the_batch_size = self.batch_size 
            else: 
                the_batch_size = 1

            video_length = key.size()[0] // the_batch_size
            
            key = rearrange_3(key, video_length)
            value = rearrange_3(value, video_length)

            this_key = key[:, -self.last_values_count:]
            self.next_keys.append(this_key)
            self.next_values.append(value[:, -self.last_values_count:])

            if self.last_keys == []:
                random_offsets = (2 * torch.log(1 - torch.rand(video_length, generator=self.generator))).long()
                # random_offsets -= 1
                clamped_offsets = torch.clamp(random_offsets, min=-self.last_values_count, max=0)

                max_index = video_length
                shifted_indices = torch.clamp(clamped_offsets + torch.arange(video_length),
                     min=0, 
                     max=max_index)

                key = key[:, shifted_indices]
                value = value[:, shifted_indices]
            else:
                random_offsets = (2 * torch.log(1 - torch.rand(video_length, generator=self.generator))).long()
                # random_offsets -= 1
                actual_last_values_count = self.last_values[self.call_index].shape[1] 
                clamped_offsets = torch.clamp(random_offsets, min=-actual_last_values_count, max=0)

                max_index = video_length+actual_last_values_count-1
                shifted_indices = torch.clamp(clamped_offsets + torch.arange(video_length) + actual_last_values_count-1,
                     min=0, 
                     max=max_index)
                
                last_key = self.last_keys[self.call_index]                
                last_value = self.last_values[self.call_index]

                # try shifted all the features by one frame
                # and duplicated the last one and concating them

                

                # rearrange keys to have batch and frames in the 1st and 2nd dims respectively
                key = torch.cat([last_key, key], dim=1)
                key = key[:, shifted_indices]

                value = torch.cat([last_value, value], dim=1)[:, shifted_indices]

            key = rearrange_4(key)
            value = rearrange_4(value)
            
            self.call_index += 1

        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
class LoRAAttnProcessor2_0(nn.Module):
    r"""
    Processor for implementing the LoRA attention mechanism using PyTorch 2.0's memory-efficient scaled dot-product
    attention.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, rank=4, network_alpha=None):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank)

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0):
        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)

        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states