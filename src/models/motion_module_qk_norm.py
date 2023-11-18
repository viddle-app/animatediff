from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torchvision

from typing import Callable, Optional
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import FeedForward, Attention
import xformers
import xformers.ops

from einops import rearrange, repeat
import math
import sys


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def get_motion_module(
    in_channels,
    motion_module_type: str, 
    motion_module_kwargs: dict
):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(in_channels=in_channels, **motion_module_kwargs,)    
    else:
        raise ValueError



class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads                = 8,
        num_transformer_block              = 2,
        attention_block_types              =( "Temporal_Self", "Temporal_Self" ),
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
        temporal_attention_dim_div         = 1,
        zero_initialize                    = True,
        upcast_attention                   = False,
    ):
        super().__init__()
        
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
            upcast_attention=upcast_attention,
        )
        
        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, input_tensor, temb, encoder_hidden_states, attention_mask=None, anchor_frame_idx=None):
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask)

        output = hidden_states
        return output

    def set_save_attention_entropy(self, save_attention_entropy):
        self.temporal_transformer.set_save_attention_entropy(save_attention_entropy)

    def clear_attention_entropy(self):
        self.temporal_transformer.clear_attention_entropy()

    def get_attention_entropy(self):
        return self.temporal_transformer.get_attention_entropy()

    def get_entropies(self):
        return self.temporal_transformer.get_entropies()


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,

        num_layers,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),        
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)    

    def set_save_attention_entropy(self, save_attention_entropy):
        for block in self.transformer_blocks:
            block.set_save_attention_entropy(save_attention_entropy)

    def clear_attention_entropy(self):
        for block in self.transformer_blocks:
            block.clear_attention_entropy()

    def get_attention_entropy(self):
        attention_entropy = []
        min_attention_entropy = sys.float_info.max
        max_attention_entropy = sys.float_info.min

        for block in self.transformer_blocks:
            averages, min_attention, max_attention = block.get_attention_entropy()
            attention_entropy += averages
            min_attention_entropy = min(min_attention_entropy, min_attention)
            max_attention_entropy = max(max_attention_entropy, max_attention)
        return attention_entropy, min_attention_entropy, max_attention_entropy

    def get_entropies(self):
        all_entropies = []

        for block in self.transformer_blocks:
            all_entropies += block.get_entropies()

        return all_entropies

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length)
        
        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        
        return output


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
    ):
        super().__init__()

        attention_blocks = []
        norms = []
        
        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    cross_attention_dim=cross_attention_dim if block_name.endswith("_Cross") else None,
                    
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
        
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(dim))
            
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

    def set_save_attention_entropy(self, save_attention_entropy):
        for block in self.attention_blocks:
            block.set_save_attention_entropy(save_attention_entropy)

    def clear_attention_entropy(self):
        for block in self.attention_blocks:
            block.clear_attention_entropy()

    def get_attention_entropy(self):
        attention_entropy = []
        attention_entropy_min = sys.float_info.max
        attention_entropy_max = sys.float_info.min
        for block in self.attention_blocks:
            average, min_attention, max_attention = block.get_attention_entropy()
            attention_entropy.append(average)
            attention_entropy_min = min(attention_entropy_min, min_attention)
            attention_entropy_max = max(attention_entropy_max, max_attention)
        return attention_entropy, attention_entropy_min, attention_entropy_max

    def get_entropies(self):
        all_entropies = []
        for block in self.attention_blocks:
            all_entropies.append(block.get_entropies())
        return all_entropies

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                video_length=video_length,
            ) + hidden_states
            
        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
        
        output = hidden_states  
        return output


class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model, 
        dropout = 0., 
        max_len = 24
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def get_attention_scores(query, key, scale, upcast_attention, upcast_softmax, attention_mask=None):
    dtype = query.dtype

    # Handling upcasting for attention
    if upcast_attention:
        query = query.float()
        key = key.float()

    # Perform batch matrix-matrix product and scale
    attention_scores = torch.matmul(query, key.transpose(-1, -2)) * scale

    # Apply attention mask (if provided)
    if attention_mask is not None:
        # Ensure that attention_mask is broadcastable to the shape of attention_scores
        attention_scores += attention_mask

    # Handling upcasting for softmax
    if upcast_softmax:
        attention_scores = attention_scores.float()

    # Apply softmax to compute the attention probabilities
    attention_probs = attention_scores.softmax(dim=-1)

    # Convert back to the original dtype if necessary
    if upcast_softmax or upcast_attention:
        attention_probs = attention_probs.to(dtype)

    return attention_probs


class VersatileAttention(Attention):
    def __init__(
            self,
            attention_mode                     = None,
            cross_frame_attention_mode         = None,
            temporal_position_encoding         = False,
            temporal_position_encoding_max_len = 24,
            save_attention_entropy = False,            
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal"

        self.save_attention_entropy = save_attention_entropy
        self.attention_entropy = None
        self.min_attention_entropy = None
        self.max_attention_entropy = None

        init_amount = torch.log2(torch.tensor(16**2 - 16))
        self.learnable_scale = nn.Parameter(init_amount)

        # TODO need to store entropy losses
        # this is a tensor of per row average entropy - the max entropy
        # squared
        # then I collect all of these and take the mean assuming it is possible
        self.entropies = None

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["cross_attention_dim"] is not None
        
        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],
            dropout=0., 
            max_len=temporal_position_encoding_max_len
        ) if (temporal_position_encoding and attention_mode == "Temporal") else None

    def set_save_attention_entropy(self, save_attention_entropy):
        self.save_attention_entropy = save_attention_entropy

    def clear_attention_entropy(self):
        self.attention_entropy = None
        self.min_attention_entropy = None
        self.max_attention_entropy = None
    
    def get_entropies(self):
        return self.entropies

    def get_attention_entropy(self):
        return self.attention_entropy.item(), self.min_attention_entropy, self.max_attention_entropy

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def compute_attention_entropy(self, attention_matrix, eps=6.10e-05, batch_size=None, 
                                  ):
        
        if eps is None:
            A_clamped = attention_matrix
        else:
            A_clamped = torch.clamp(attention_matrix, eps, 1.0)

        # Compute the entropy for each row in each attention matrix
        # The result will have shape (B, T)
        self.entropies = -torch.sum(A_clamped * torch.log(A_clamped), dim=2)
        
        # Compute the average entropy for each matrix in the batch
        # The result will have shape (B,)
        avg_entropies_per_matrix = torch.mean(self.entropies, dim=1)

        self.entropies = rearrange(self.entropies, 
                                   "(b d) f -> b d f", 
                                   b=batch_size)
    

        # todo take the max and the min of the entropies

        # Compute the average of all the average entropies across the batch
        avg_entropy_across_batch = torch.mean(avg_entropies_per_matrix)
        min_entropy_across_batch = torch.min(avg_entropies_per_matrix).item()
        max_entropy_across_batch = torch.max(avg_entropies_per_matrix).item()

    
        return avg_entropy_across_batch, min_entropy_across_batch, max_entropy_across_batch

        
        
        

    def forward(self, 
                hidden_states, 
                encoder_hidden_states=None, 
                attention_mask=None, 
                video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape

        if self.attention_mode == "Temporal":
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            
            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(hidden_states)
            
            encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states
        else:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)

        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        # concat on the the encoder_hidden_states 
        # the first frames features duplicated 5 times
        if False:
            first_frame_states = encoder_hidden_states[:, 0:1, :]
            first_frame_states_repeated = first_frame_states.repeat_interleave(32, dim=1)
            encoder_hidden_states = torch.cat([encoder_hidden_states, first_frame_states_repeated], dim=1)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)


        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)
        # reshape back
        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)


        # update the scale to using
        # numerator = math.log(sequence_length) / math.log(sequence_length//4)
        # if not self.training:
        #    numerator = math.log(sequence_length) / math.log(sequence_length//4)
        #    numerator = 1
        #    self.scale = math.sqrt(numerator / (self.inner_dim // self.heads))

        # self.scale = self.learnable_scale


        if query.dtype == torch.float16:
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, op=None, scale=self.scale
            )

            if self.save_attention_entropy:
                attention_probs = get_attention_scores(query, key, self.learnable_scale,
                                                       self.upcast_attention, 
                                                       self.upcast_softmax,
                                                       attention_mask=attention_mask)
                avg, the_min, the_max = self.compute_attention_entropy(attention_probs,
                                                   batch_size=batch_size // video_length)
                self.attention_entropy = avg
                self.min_attention_entropy = the_min
                self.max_attention_entropy = the_max 

            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs =  get_attention_scores(query, key, self.learnable_scale,
                                                       self.upcast_attention, 
                                                       self.upcast_softmax,
                                                       attention_mask=attention_mask)

            if self.save_attention_entropy:
                avg, the_min, the_max = self.compute_attention_entropy(attention_probs,
                                                   batch_size=batch_size // video_length)
                self.attention_entropy = avg
                self.min_attention_entropy = the_min
                self.max_attention_entropy = the_max 

            hidden_states = torch.bmm(attention_probs, value)
        
        hidden_states = self.batch_to_head_dim(hidden_states)
        
        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if self.attention_mode == "Temporal":
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
