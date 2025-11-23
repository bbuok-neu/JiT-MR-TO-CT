# --------------------------------------------------------
# DiffusionModelUNet implementation
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
            nn.Linear(embedding_dim * 4, embedding_dim * 4),
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
    ):
        super().__init__()
        
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.attention_levels = attention_levels
        self.num_res_blocks = num_res_blocks
        
        # Timestep embedding
        time_embed_dim = num_channels[0] * 4
        self.time_embed = TimestepEmbedding(time_embed_dim)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, num_channels[0], kernel_size=3, padding=1)
        
        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = num_channels[0]
        for i, ch_out in enumerate(num_channels):
            for j in range(num_res_blocks):
                layers = [ResidualBlock(ch, ch_out, time_embed_dim, dropout)]
                ch = ch_out
                
                if attention_levels[i]:
                    num_heads = ch // num_head_channels
                    layers.append(AttentionBlock(ch, num_heads))
                
                self.down_blocks.append(nn.ModuleList(layers))
            
            # Downsample (except for the last level)
            if i < len(num_channels) - 1:
                self.down_samples.append(Downsample(ch))
            else:
                self.down_samples.append(nn.Identity())
        
        # Middle block
        self.mid_block1 = ResidualBlock(ch, ch, time_embed_dim, dropout)
        self.mid_attn = AttentionBlock(ch, ch // num_head_channels)
        self.mid_block2 = ResidualBlock(ch, ch, time_embed_dim, dropout)
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for i, ch_out in reversed(list(enumerate(num_channels))):
            for j in range(num_res_blocks + 1):
                # Account for skip connections
                ch_in = ch + (ch_out if j == 0 else 0)
                layers = [ResidualBlock(ch_in, ch_out, time_embed_dim, dropout)]
                ch = ch_out
                
                if attention_levels[i]:
                    num_heads = ch // num_head_channels
                    layers.append(AttentionBlock(ch, num_heads))
                
                self.up_blocks.append(nn.ModuleList(layers))
            
            # Upsample (except for the first level going backwards)
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
        hs = [h]
        for i, (blocks, downsample) in enumerate(zip(self.down_blocks, self.down_samples)):
            for layer in blocks:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, temb)
                else:
                    h = layer(h)
            hs.append(h)
            h = downsample(h)
        
        # Middle
        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)
        
        # Decoder
        for i, (blocks, upsample) in enumerate(zip(self.up_blocks, self.up_samples)):
            # Skip connection
            if i % (self.num_res_blocks + 1) == 0 and len(hs) > 0:
                h = torch.cat([h, hs.pop()], dim=1)
            
            for layer in blocks:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, temb)
                else:
                    h = layer(h)
            
            if (i + 1) % (self.num_res_blocks + 1) == 0:
                h = upsample(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h
