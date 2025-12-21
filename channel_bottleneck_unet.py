"""
Channel Bottleneck DiffusionModelUNet

This module implements a modified DiffusionModelUNet that adds a channel bottleneck layer
between down_blocks and mid_blocks. The bottleneck uses 1×1 convolutions to map
512 channels → 128 channels → 512 channels, mimicking the JiT model's ViT dimension
bottleneck design to force the network to perform low-dimensional manifold learning.

Reference: JiT model's BottleneckPatchEmbed design in model_jit.py
"""

import torch
import torch.nn as nn
from generative.networks.nets import DiffusionModelUNet


class ChannelBottleneck(nn.Module):
    """
    Channel bottleneck layer using 1×1 convolutions.
    Maps: in_channels → bottleneck_channels → in_channels
    
    Args:
        spatial_dims: Number of spatial dimensions (2 or 3).
        in_channels: Number of input/output channels.
        bottleneck_channels: Number of channels in the bottleneck.
    """
    
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        bottleneck_channels: int,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        
        # Choose the appropriate Conv layer based on spatial dimensions
        if spatial_dims == 2:
            conv_layer = nn.Conv2d
        elif spatial_dims == 3:
            conv_layer = nn.Conv3d
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        
        # 1×1 conv: in_channels → bottleneck_channels
        self.compress = conv_layer(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # 1×1 conv: bottleneck_channels → in_channels
        self.expand = conv_layer(
            in_channels=bottleneck_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the channel bottleneck.
        
        Args:
            x: Input tensor of shape (N, in_channels, *spatial_dims).
            
        Returns:
            Output tensor of the same shape as input.
        """
        x = self.compress(x)
        x = self.expand(x)
        return x


class ChannelBottleneckDiffusionModelUNet(DiffusionModelUNet):
    """
    Modified DiffusionModelUNet with a channel bottleneck layer between
    down_blocks and mid_blocks.
    
    This modification mimics the JiT model's ViT dimension bottleneck design
    (BottleneckPatchEmbed) to force the network to perform low-dimensional
    manifold learning.
    
    The bottleneck uses 1×1 convolutions to compress and expand channels:
    512 → 128 → 512 (default configuration)
    
    All other parameters and functionality remain identical to the original
    DiffusionModelUNet.
    
    Args:
        All args from DiffusionModelUNet, plus:
        bottleneck_channels: Number of channels in the bottleneck layer (default: 128).
    """
    
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int = 128,
        **kwargs
    ):
        # Initialize the parent DiffusionModelUNet
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        )
        
        # Store bottleneck configuration
        self.bottleneck_channels = bottleneck_channels
        
        # Get the number of channels at the deepest level (input to mid block)
        # This is the last channel count in num_channels (typically 512)
        mid_block_channels = self.block_out_channels[-1]
        
        # Create the channel bottleneck layer
        self.channel_bottleneck = ChannelBottleneck(
            spatial_dims=spatial_dims,
            in_channels=mid_block_channels,
            bottleneck_channels=bottleneck_channels
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with channel bottleneck between down_blocks and mid_blocks.
        
        Args:
            x: input tensor (N, C, SpatialDims).
            timesteps: timestep tensor (N,).
            context: context tensor (N, 1, ContextDim).
            class_labels: context tensor (N, ).
            down_block_additional_residuals: additional residual tensors for down blocks.
            mid_block_additional_residual: additional residual tensor for mid block.
            
        Returns:
            Output tensor of shape (N, out_channels, SpatialDims).
        """
        # Import the timestep embedding function from the parent module
        from generative.networks.nets.diffusion_model_unet import get_timestep_embedding
        
        # 1. time
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)
        
        # 2. class
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb = emb + class_emb
        
        # 3. initial convolution
        h = self.conv_in(x)
        
        # 4. down
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            for residual in res_samples:
                down_block_res_samples.append(residual)
        
        # Additional residual connections for ControlNets
        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples
        
        # *** CHANNEL BOTTLENECK: Apply between down_blocks and mid_blocks ***
        h = self.channel_bottleneck(h)
        
        # 5. mid
        h = self.middle_block(hidden_states=h, temb=emb, context=context)
        
        # Additional residual connections for ControlNets
        if mid_block_additional_residual is not None:
            h = h + mid_block_additional_residual
        
        # 6. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=context)
        
        # 7. output block
        h = self.out(h)
        
        return h
