"""
DiffusionModelUNet with adaLN-Zero Timestep Conditioning

This module extends MONAI's DiffusionModelUNet to use Adaptive Layer Normalization
(adaLN-Zero) for timestep embedding, similar to the JiT/DiT architecture.

Key changes from original DiffusionModelUNet:
1. Replaces additive timestep injection with adaLN modulation (shift, scale, gate)
2. Uses RMSNorm instead of GroupNorm where appropriate
3. Zero-initializes the modulation output layers (adaLN-Zero)

References:
- DiT: https://github.com/facebookresearch/DiT
- JiT: Based on SiT and Lightning-DiT implementations
- MONAI GenerativeModels: https://github.com/Project-MONAI/GenerativeModels
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Pool
from monai.utils import ensure_tuple_rep


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    
    This is used for adaLN-Zero initialization, where the output layers of 
    modulation networks are zero-initialized to ensure stable training.
    At initialization, the network behaves as if the modulation is disabled,
    and the model gradually learns to use modulation during training.
    
    Reference: DiT (Scalable Diffusion Models with Transformers)
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Apply adaLN modulation: x * (1 + scale) + shift
    
    Args:
        x: Input tensor (B, C, H, W) or (B, L, C)
        shift: Shift parameter (B, C) or (B, C)
        scale: Scale parameter (B, C) or (B, C)
    
    Returns:
        Modulated tensor
    """
    if x.ndim == 4:  # (B, C, H, W)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
    elif x.ndim == 5:  # (B, C, H, W, D)
        return x * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
    else:  # (B, L, C)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More efficient than LayerNorm, used in LLaMA and modern transformers.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # Cast to float32 for numerical stability in RMS computation
        # This prevents underflow/overflow in the variance calculation
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class SpatialRMSNorm(nn.Module):
    """
    RMSNorm adapted for spatial features (B, C, H, W).
    Normalizes over spatial dimensions (H, W) for each channel.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) or (B, C, H, W, D)
        input_dtype = x.dtype
        # Cast to float32 for numerical stability
        x = x.to(torch.float32)
        
        if x.ndim == 4:
            # Compute RMS over spatial dimensions (H, W) for each channel
            variance = x.pow(2).mean(dim=[2, 3], keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return (self.weight.view(1, -1, 1, 1) * x).to(input_dtype)
        else:  # 5D
            # Compute RMS over spatial dimensions (H, W, D) for each channel
            variance = x.pow(2).mean(dim=[2, 3, 4], keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return (self.weight.view(1, -1, 1, 1, 1) * x).to(input_dtype)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Uses sinusoidal embedding followed by MLP.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class AdaLNResnetBlock(nn.Module):
    """
    Residual block with adaLN-Zero timestep conditioning.
    
    Instead of additive injection (h = h + temb), uses adaptive modulation:
    - shift, scale, gate = adaLN_modulation(temb)
    - h = norm(h) * (1 + scale) + shift
    - output = gate * output
    
    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        temb_channels: number of timestep embedding channels.
        out_channels: number of output channels.
        up: if True, performs upsampling.
        down: if True, performs downsampling.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        out_channels: int | None = None,
        up: bool = False,
        down: bool = False,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = in_channels
        self.emb_channels = temb_channels
        self.out_channels = out_channels or in_channels
        self.up = up
        self.down = down

        # First normalization and convolution
        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.nonlinearity = nn.SiLU()
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # Up/Downsampling
        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample(spatial_dims, in_channels, use_conv=False)
        elif down:
            self.downsample = Downsample(spatial_dims, in_channels, use_conv=False)

        # adaLN modulation for first norm (in_channels): shift1, scale1
        self.adaLN_modulation1 = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(temb_channels, 2 * in_channels, bias=True))
        )
        
        # adaLN modulation for second norm (out_channels): shift2, scale2, gate
        self.adaLN_modulation2 = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(temb_channels, 3 * self.out_channels, bias=True))
        )

        # Second normalization and convolution
        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.out_channels, eps=norm_eps, affine=True)
        self.conv2 = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Skip connection
        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # Get adaLN modulation parameters
        shift1, scale1 = self.adaLN_modulation1(emb).chunk(2, dim=-1)
        shift2, scale2, gate = self.adaLN_modulation2(emb).chunk(3, dim=-1)
        
        h = x
        
        # First block with adaLN modulation
        h = self.norm1(h)
        h = modulate(h, shift1, scale1)
        h = self.nonlinearity(h)

        # Up/Downsample
        if self.upsample is not None:
            # When batch size is large (>=64), ensure tensors are contiguous
            # to avoid memory fragmentation issues with upsampling operations
            if h.shape[0] >= 64:
                x = x.contiguous()
                h = h.contiguous()
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)

        # Second block with adaLN modulation  
        h = self.norm2(h)
        h = modulate(h, shift2, scale2)
        h = self.nonlinearity(h)
        h = self.conv2(h)
        
        # Apply gate to the residual path
        if self.spatial_dims == 2:
            h = gate[:, :, None, None] * h
        else:
            h = gate[:, :, None, None, None] * h

        return self.skip_connection(x) + h


class Downsample(nn.Module):
    """
    Downsampling layer.
    """
    def __init__(
        self, spatial_dims: int, num_channels: int, use_conv: bool, out_channels: int | None = None, padding: int = 1
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if use_conv:
            self.op = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=2,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            if self.num_channels != self.out_channels:
                raise ValueError("num_channels and out_channels must be equal when use_conv=False")
            self.op = Pool[Pool.AVG, spatial_dims](kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        del emb
        return self.op(x)


class Upsample(nn.Module):
    """
    Upsampling layer.
    """
    def __init__(
        self, spatial_dims: int, num_channels: int, use_conv: bool, out_channels: int | None = None, padding: int = 1
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            self.conv = None

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        del emb
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        if dtype == torch.bfloat16:
            x = x.to(dtype)

        if self.use_conv:
            x = self.conv(x)
        return x


class AdaLNAttentionBlock(nn.Module):
    """
    Attention block with adaLN modulation for timestep conditioning.
    Uses GroupNorm + adaLN modulation before attention.
    """
    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        temb_channels: int,
        num_head_channels: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels
        self.temb_channels = temb_channels

        # Validate head configuration
        if num_head_channels is not None:
            if num_channels % num_head_channels != 0:
                raise ValueError(f"num_channels ({num_channels}) must be divisible by num_head_channels ({num_head_channels})")
            self.num_heads = num_channels // num_head_channels
        else:
            self.num_heads = 1
        self.scale = 1 / math.sqrt(num_channels / self.num_heads)

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=norm_eps, affine=True)
        
        # adaLN modulation: shift, scale, gate
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(temb_channels, 3 * num_channels, bias=True))
        )

        self.to_q = nn.Linear(num_channels, num_channels)
        self.to_k = nn.Linear(num_channels, num_channels)
        self.to_v = nn.Linear(num_channels, num_channels)

        self.proj_attn = nn.Linear(num_channels, num_channels)

    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, dim // self.num_heads)
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size // self.num_heads, self.num_heads, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // self.num_heads, seq_len, dim * self.num_heads)
        return x

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        x = torch.bmm(attention_probs, value)
        return x

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        residual = x

        batch = channel = height = width = depth = -1
        if self.spatial_dims == 2:
            batch, channel, height, width = x.shape
        if self.spatial_dims == 3:
            batch, channel, height, width, depth = x.shape

        # Get adaLN modulation parameters
        shift, scale, gate = self.adaLN_modulation(emb).chunk(3, dim=-1)

        # Apply norm and modulation
        x = self.norm(x)
        x = modulate(x, shift, scale)

        if self.spatial_dims == 2:
            x = x.view(batch, channel, height * width).transpose(1, 2)
        if self.spatial_dims == 3:
            x = x.view(batch, channel, height * width * depth).transpose(1, 2)

        # Attention
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        x = self._attention(query, key, value)
        x = self.reshape_batch_dim_to_heads(x)
        x = self.proj_attn(x)

        if self.spatial_dims == 2:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width)
        if self.spatial_dims == 3:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width, depth)

        # Apply gate
        if self.spatial_dims == 2:
            x = gate[:, :, None, None] * x
        else:
            x = gate[:, :, None, None, None] * x

        return x + residual


class AdaLNDownBlock(nn.Module):
    """
    Down block with adaLN-Zero ResNet blocks.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        for i in range(num_res_blocks):
            in_ch = in_channels if i == 0 else out_channels
            resnets.append(
                AdaLNResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            if resblock_updown:
                self.downsampler = AdaLNResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = Downsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, list]:
        del context
        output_states = []

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class AdaLNAttnDownBlock(nn.Module):
    """
    Down block with adaLN-Zero ResNet blocks and attention.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
        num_head_channels: int = 1,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_ch = in_channels if i == 0 else out_channels
            resnets.append(
                AdaLNResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                AdaLNAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    temb_channels=temb_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            if resblock_updown:
                self.downsampler = AdaLNResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = Downsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, list]:
        del context
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, temb)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class AdaLNUpBlock(nn.Module):
    """
    Up block with adaLN-Zero ResNet blocks.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown
        resnets = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                AdaLNResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            if resblock_updown:
                self.upsampler = AdaLNResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                self.upsampler = Upsample(
                    spatial_dims=spatial_dims, num_channels=out_channels, use_conv=True, out_channels=out_channels
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list,
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states


class AdaLNAttnUpBlock(nn.Module):
    """
    Up block with adaLN-Zero ResNet blocks and attention.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                AdaLNResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                AdaLNAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    temb_channels=temb_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_upsample:
            if resblock_updown:
                self.upsampler = AdaLNResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                self.upsampler = Upsample(
                    spatial_dims=spatial_dims, num_channels=out_channels, use_conv=True, out_channels=out_channels
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list,
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, temb)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states


class AdaLNMidBlock(nn.Module):
    """
    Middle block with adaLN-Zero ResNet blocks and attention.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
    ) -> None:
        super().__init__()
        self.resnet_1 = AdaLNResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )
        self.attention = AdaLNAttentionBlock(
            spatial_dims=spatial_dims,
            num_channels=in_channels,
            temb_channels=temb_channels,
            num_head_channels=num_head_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )
        self.resnet_2 = AdaLNResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        del context
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attention(hidden_states, temb)
        hidden_states = self.resnet_2(hidden_states, temb)
        return hidden_states


class DiffusionModelUNetAdaLN(nn.Module):
    """
    DiffusionModelUNet with adaLN-Zero Timestep Conditioning.
    
    This model uses Adaptive Layer Normalization (adaLN-Zero) for timestep embedding
    injection, similar to DiT and JiT architectures. Instead of simple additive
    injection (h = h + temb), this uses learnable shift, scale, and gate parameters:
    
    - shift, scale, gate = adaLN_modulation(temb)
    - h = norm(h) * (1 + scale) + shift
    - output = gate * output
    
    The gate parameters are zero-initialized (adaLN-Zero) for stable training.
    
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks per level.
        num_channels: tuple of block output channels.
        attention_levels: list of levels to add attention.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        resblock_updown: if True use residual blocks for up/downsampling.
        num_head_channels: number of channels in each attention head.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,  # API compatibility - not used (use concatenation instead)
    ) -> None:
        super().__init__()
        
        # Note: with_conditioning parameter is kept for API compatibility but not used.
        # This model uses input concatenation for conditioning instead of cross-attention.
        _ = with_conditioning  # Explicitly mark as unused
        
        # Validate inputs
        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError("All num_channels must be divisible by norm_num_groups")

        if len(num_channels) != len(attention_levels):
            raise ValueError("num_channels and attention_levels must have same length")

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))

        if len(num_head_channels) != len(attention_levels):
            raise ValueError("num_head_channels must have same length as attention_levels")

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError("num_res_blocks must have same length as num_channels")

        self.in_channels = in_channels
        self.block_out_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels

        # Input convolution
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # Time embedding with adaLN style (similar to JiT)
        time_embed_dim = num_channels[0] * 4
        self.time_embed = TimestepEmbedder(time_embed_dim, frequency_embedding_size=256)

        # Down blocks
        self.down_blocks = nn.ModuleList([])
        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1

            if attention_levels[i]:
                down_block = AdaLNAttnDownBlock(
                    spatial_dims=spatial_dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_res_blocks=num_res_blocks[i],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    add_downsample=not is_final_block,
                    resblock_updown=resblock_updown,
                    num_head_channels=num_head_channels[i],
                )
            else:
                down_block = AdaLNDownBlock(
                    spatial_dims=spatial_dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_res_blocks=num_res_blocks[i],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    add_downsample=not is_final_block,
                    resblock_updown=resblock_updown,
                )

            self.down_blocks.append(down_block)

        # Middle block
        self.middle_block = AdaLNMidBlock(
            spatial_dims=spatial_dims,
            in_channels=num_channels[-1],
            temb_channels=time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_head_channels=num_head_channels[-1],
        )

        # Up blocks
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(num_channels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_head_channels = list(reversed(num_head_channels))
        output_channel = reversed_block_out_channels[0]
        
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(num_channels) - 1)]
            is_final_block = i == len(num_channels) - 1

            if reversed_attention_levels[i]:
                up_block = AdaLNAttnUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=input_channel,
                    prev_output_channel=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_res_blocks=reversed_num_res_blocks[i] + 1,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    add_upsample=not is_final_block,
                    resblock_updown=resblock_updown,
                    num_head_channels=reversed_num_head_channels[i],
                )
            else:
                up_block = AdaLNUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=input_channel,
                    prev_output_channel=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_res_blocks=reversed_num_res_blocks[i] + 1,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    add_upsample=not is_final_block,
                    resblock_updown=resblock_updown,
                )

            self.up_blocks.append(up_block)

        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels[0], eps=norm_eps, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[0],
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: input tensor (N, C, SpatialDims).
            timesteps: timestep tensor (N,).
            context: context tensor (not used, for API compatibility).
        
        Returns:
            Output tensor (N, out_channels, SpatialDims).
        """
        del context  # Not used in this implementation
        
        # 1. Time embedding
        emb = self.time_embed(timesteps)
        emb = emb.to(dtype=x.dtype)

        # 2. Initial convolution
        h = self.conv_in(x)

        # 3. Down
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=None)
            for residual in res_samples:
                down_block_res_samples.append(residual)

        # 4. Middle
        h = self.middle_block(hidden_states=h, temb=emb, context=None)

        # 5. Up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=None)

        # 6. Output
        h = self.out(h)

        return h
