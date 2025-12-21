# --------------------------------------------------------
# Custom DiffusionModelUNet implementation (Reference Only)
# 
# NOTE: This file is kept for reference purposes only.
# The actual implementation now uses MONAI's official DiffusionModelUNet
# from the monai-generative package.
#
# Installation: pip install monai-generative==0.2.3
# Import: from generative.networks.nets import DiffusionModelUNet
# Reference: https://github.com/Project-MONAI/GenerativeModels
# Example: https://github.com/milad1378yz/MOTFM
#
# Based on MONAI GenerativeModels
# Adapted for MR-to-CT synthesis with conditional inputs
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimestepEmbedding(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, embedding_dim, max_period=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        
        # MLP to process timestep embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
    
    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, timesteps):
        t_emb = self.timestep_embedding(timesteps, self.embedding_dim, self.max_period)
        t_emb = self.mlp(t_emb)
        return t_emb


class ResidualBlock(nn.Module):
    """
    Residual block with timestep conditioning.
    """
    def __init__(self, in_channels, out_channels, temb_channels, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add timestep embedding
        if temb is not None:
            temb_out = self.temb_proj(F.silu(temb))
            h = h + temb_out[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip_connection(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for U-Net.
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, h * w)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, b, num_heads, h*w, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        out = self.proj(out)
        
        return x + out


class ChannelBottleneck(nn.Module):
    """
    Channel bottleneck layer that reduces and then expands channel dimensions.
    This helps with computational efficiency and can act as a form of regularization.
    
    The layer performs: in_channels -> bottleneck_channels -> out_channels
    with optional normalization and activation between the projections.
    """
    def __init__(self, in_channels, out_channels, bottleneck_channels, use_norm=True, use_activation=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = bottleneck_channels
        
        # Projection down to bottleneck dimension
        self.proj_down = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        
        # Optional normalization and activation
        if use_norm:
            # Use min of bottleneck_channels and 32 for group norm to handle small channel counts
            num_groups = min(32, bottleneck_channels)
            # Ensure num_groups divides bottleneck_channels evenly
            while bottleneck_channels % num_groups != 0:
                num_groups -= 1
            self.norm = nn.GroupNorm(num_groups, bottleneck_channels)
        else:
            self.norm = nn.Identity()
        
        self.activation = nn.SiLU() if use_activation else nn.Identity()
        
        # Projection up to output dimension
        self.proj_up = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=True)
    
    def forward(self, x):
        h = self.proj_down(x)
        h = self.norm(h)
        h = self.activation(h)
        h = self.proj_up(h)
        return h


class Downsample(nn.Module):
    """
    Downsampling layer.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsampling layer.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x


class DiffusionModelUNet(nn.Module):
    """
    U-Net model for diffusion with timestep conditioning.
    Configured for MR-to-CT synthesis with 2-channel input and 1-channel output.
    
    Args:
        spatial_dims: Number of spatial dimensions (2 for 2D images).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_channels: Tuple of channel counts for each encoder/decoder level.
        attention_levels: Tuple of booleans indicating whether to use attention at each level.
        num_res_blocks: Number of residual blocks per level.
        num_head_channels: Number of channels per attention head.
        dropout: Dropout rate.
        bottleneck_channels: If provided, adds a channel bottleneck layer at the middle block.
            This reduces the channel dimension to bottleneck_channels before the middle 
            processing and expands back afterwards, which can improve efficiency and 
            act as regularization. Set to None to disable (default).
    """
    def __init__(
        self,
        spatial_dims=2,
        in_channels=2,
        out_channels=1,
        num_channels=(64, 128, 256, 512),
        attention_levels=(False, False, True, True),
        num_res_blocks=2,
        num_head_channels=32,
        dropout=0.0,
        bottleneck_channels=None,
    ):
        super().__init__()
        
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.attention_levels = attention_levels
        self.num_res_blocks = num_res_blocks
        self.bottleneck_channels = bottleneck_channels
        
        # Timestep embedding
        time_embed_dim = num_channels[0] * 4
        self.time_embed = TimestepEmbedding(time_embed_dim)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, num_channels[0], kernel_size=3, padding=1)
        
        # Encoder (downsampling path)
        # down_blocks is organized by level, with each level containing num_res_blocks sets of layers
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = num_channels[0]
        for i, ch_out in enumerate(num_channels):
            # Create a module list for all blocks at this level
            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                layers = nn.ModuleList([ResidualBlock(ch, ch_out, time_embed_dim, dropout)])
                ch = ch_out
                
                if attention_levels[i]:
                    num_heads = ch // num_head_channels
                    layers.append(AttentionBlock(ch, num_heads))
                
                level_blocks.append(layers)
            
            self.down_blocks.append(level_blocks)
            
            # Downsample (except for the last level)
            if i < len(num_channels) - 1:
                self.down_samples.append(Downsample(ch))
            else:
                self.down_samples.append(nn.Identity())
        
        # Channel bottleneck before middle block (optional)
        if bottleneck_channels is not None:
            self.bottleneck_down = ChannelBottleneck(
                in_channels=ch, 
                out_channels=bottleneck_channels, 
                bottleneck_channels=bottleneck_channels,
                use_norm=True,
                use_activation=True
            )
            middle_ch = bottleneck_channels
        else:
            self.bottleneck_down = None
            middle_ch = ch
        
        # Middle block
        self.mid_block1 = ResidualBlock(middle_ch, middle_ch, time_embed_dim, dropout)
        self.mid_attn = AttentionBlock(middle_ch, max(1, middle_ch // num_head_channels))
        self.mid_block2 = ResidualBlock(middle_ch, middle_ch, time_embed_dim, dropout)
        
        # Channel expansion after middle block (if bottleneck is used)
        if bottleneck_channels is not None:
            self.bottleneck_up = ChannelBottleneck(
                in_channels=middle_ch, 
                out_channels=ch, 
                bottleneck_channels=bottleneck_channels,
                use_norm=True,
                use_activation=True
            )
        else:
            self.bottleneck_up = None
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for i, ch_out in reversed(list(enumerate(num_channels))):
            for j in range(num_res_blocks + 1):
                # First block in each level gets skip connection from encoder
                # ch is current channels, ch_out is target channels for this level
                if j == 0:
                    ch_in = ch + ch_out  # skip connection adds encoder features
                else:
                    ch_in = ch_out  # subsequent blocks just process the features
                
                layers = [ResidualBlock(ch_in, ch_out, time_embed_dim, dropout)]
                ch = ch_out
                
                if attention_levels[i]:
                    num_heads = ch // num_head_channels
                    layers.append(AttentionBlock(ch, num_heads))
                
                self.up_blocks.append(nn.ModuleList(layers))
            
            # Upsample (except for the first level going backwards, which is the finest level)
            if i > 0:
                self.up_samples.append(Upsample(ch))
            else:
                self.up_samples.append(nn.Identity())
        
        # Output convolution
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, timesteps):
        """
        Args:
            x: Input tensor of shape (B, in_channels, H, W)
               For MR-to-CT: concatenation of [noisy_latent, mr_condition]
            timesteps: Timestep tensor of shape (B,)
        
        Returns:
            Output tensor of shape (B, out_channels, H, W)
            For MR-to-CT: predicted velocity field
        """
        # Timestep embedding
        temb = self.time_embed(timesteps)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Encoder
        hs = [h]  # Save initial features
        for level_idx, (level_blocks, downsample) in enumerate(zip(self.down_blocks, self.down_samples)):
            # Process all blocks at this level
            for blocks in level_blocks:
                for layer in blocks:
                    if isinstance(layer, ResidualBlock):
                        h = layer(h, temb)
                    else:
                        h = layer(h)
            # Save features BEFORE downsampling (for skip connections in decoder)
            # These features are at the same resolution as where the decoder will upsample back to
            hs.append(h)
            # Downsample to next level
            h = downsample(h)
        
        # Middle
        # Apply channel bottleneck down if enabled
        if self.bottleneck_down is not None:
            h = self.bottleneck_down(h)
        
        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)
        
        # Apply channel bottleneck up if enabled
        if self.bottleneck_up is not None:
            h = self.bottleneck_up(h)
        
        # Decoder
        # Process decoder levels in reverse order (from coarsest to finest resolution)
        # Skip connections are popped from hs in reverse order (finest to coarsest saved,
        # so popping gives coarsest to finest, which matches decoder progression)
        block_idx = 0
        for level_idx in range(len(self.num_channels)):
            # At the start of each level, apply skip connection
            # The skip features from encoder are at the same resolution as current decoder level
            if len(hs) > 0:
                skip = hs.pop()  # Pop in reverse order (coarsest to finest)
                h = torch.cat([h, skip], dim=1)
            
            # Process all blocks at this level
            for block_in_level in range(self.num_res_blocks + 1):
                blocks = self.up_blocks[block_idx]
                for layer in blocks:
                    if isinstance(layer, ResidualBlock):
                        h = layer(h, temb)
                    else:
                        h = layer(h)
                block_idx += 1
            
            # Upsample at the end of each level (except the last/finest level)
            h = self.up_samples[level_idx](h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h
