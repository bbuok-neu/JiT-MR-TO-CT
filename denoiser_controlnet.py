"""
ControlNet-based Denoiser for Zero-Shot MR-to-CT Synthesis

Two-stage architecture:
1. Base Model: Unconditional DiffusionModelUNet trained on CT (in_channels=1)
2. ControlNet: Takes MIND features (in_channels = neigh_size^2-1 or 4) to guide generation

The ControlNet is initialized from Base Model weights (Encoder + MidBlock),
except conv_in which has different input channels and is randomly initialized.

Reference: https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/2d_controlnet/2d_controlnet.py
"""
import torch
import torch.nn as nn
from generative.networks.nets import DiffusionModelUNet, ControlNet

from mind import get_mind_channels


class ControlNetWrapper(nn.Module):
    """
    Wrapper that combines Base Model and ControlNet for inference.
    
    During forward pass:
    1. ControlNet processes MIND features and produces residual connections
    2. Base Model receives these residuals and generates the output
    """
    
    def __init__(self, base_model, controlnet):
        super().__init__()
        self.base_model = base_model
        self.controlnet = controlnet
    
    def forward(self, x, timesteps, controlnet_cond):
        """
        Forward pass combining ControlNet guidance with Base Model.
        
        Args:
            x: Noisy CT image (N, 1, H, W)
            timesteps: Diffusion timesteps (N,)
            controlnet_cond: MIND features (N, C, H, W) where C = neigh_size^2-1 or 4
        
        Returns:
            Predicted clean CT image
        """
        # Get ControlNet residuals
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            x=x,
            timesteps=timesteps,
            controlnet_cond=controlnet_cond
        )
        
        # Pass through Base Model with residual connections
        output = self.base_model(
            x=x,
            timesteps=timesteps,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample
        )
        
        return output


class Denoiser_ControlNet(nn.Module):
    """
    Two-stage denoiser using ControlNet for MIND-guided CT synthesis.
    
    Stage 1: Train Base Model (unconditional) on CT images
    Stage 2: Freeze Base Model, train ControlNet with MIND features
    """
    
    def __init__(self, args):
        super().__init__()
        
        # Get MIND parameters
        mind_neigh_size = getattr(args, 'mind_neigh_size', 7)
        mind_neigh4 = getattr(args, 'mind_neigh4', False)
        self.mind_channels = get_mind_channels(mind_neigh_size, mind_neigh4)
        
        self.stage = getattr(args, 'stage', 'stage1')
        self.img_size = args.img_size
        
        # Model hyperparameters (from diffusion_model_unet.py reference)
        model_config = {
            'spatial_dims': 2,
            'num_res_blocks': (2, 2, 2, 2),
            'num_channels': (64, 128, 256, 512),
            'attention_levels': (False, False, True, True),
            'norm_num_groups': 32,
            'num_head_channels': (64, 128, 256, 512),
            'with_conditioning': False,
            'resblock_updown': True,
        }
        
        # Check for flash attention availability
        use_flash_attention = False
        try:
            import xformers
            if torch.cuda.is_available():
                use_flash_attention = True
                print("Flash attention enabled")
        except ImportError:
            print("xformers not available, using standard attention")
        
        # Create Base Model (unconditional, in_channels=1)
        self.base_model = DiffusionModelUNet(
            in_channels=1,
            out_channels=1,
            use_flash_attention=use_flash_attention,
            **model_config
        )
        
        # Create ControlNet (MIND features, in_channels=48)
        # Only created for Stage 2
        self.controlnet = None
        if self.stage == 'stage2':
            # IMPORTANT: conditioning_embedding_num_channels controls downsampling
            # For image-space diffusion with full-resolution MIND features,
            # we use a single output channel to avoid spatial size mismatch.
            # The embedding will map 48 MIND channels -> 64 channels (num_channels[0])
            # without any spatial downsampling.
            self.controlnet = ControlNet(
                in_channels=1,  # Same as base model for noisy image
                conditioning_embedding_in_channels=self.mind_channels,  # MIND features (48)
                conditioning_embedding_num_channels=(64,),  # Single channel = no downsampling
                num_res_blocks=(2, 2, 2, 2),
                num_channels=(64, 128, 256, 512),
                attention_levels=(False, False, True, True),
                norm_num_groups=32,
                num_head_channels=(64, 128, 256, 512),
                spatial_dims=2,
                use_flash_attention=use_flash_attention,
                resblock_updown=True,  # Match base model
            )
            
            # Create wrapper for combined forward pass
            self.model = ControlNetWrapper(self.base_model, self.controlnet)
            
            # Freeze base model for ControlNet training
            for param in self.base_model.parameters():
                param.requires_grad = False
            
            print(f"ControlNet initialized with {self.mind_channels} MIND feature channels")
            print("Base model frozen for Stage 2 training")
        else:
            self.model = self.base_model
            print("Stage 1: Training unconditional Base Model")
        
        # Flow matching parameters
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        
        # EMA
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None
        
        # Sampling parameters
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params / 1e6:.2f}M")
    
    def load_base_model_weights(self, checkpoint_path):
        """
        Load pretrained Base Model weights.
        
        For Stage 2, loads all weights except conv_in (different input channels).
        
        Args:
            checkpoint_path: Path to Stage 1 checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Get current base model state dict
        base_model_state = self.base_model.state_dict()
        base_model_keys = set(base_model_state.keys())
        
        # Build new state dict with matching weights
        new_state_dict = {}
        loaded_keys = []
        skipped_keys = []
        
        for key, value in state_dict.items():
            # Try different key formats
            # Format 1: direct key (e.g., "down_blocks.0.resnets.0.conv1.weight")
            # Format 2: with 'net.' prefix (e.g., "net.down_blocks.0...")
            # Format 3: with 'base_model.' prefix (e.g., "base_model.down_blocks.0...")
            
            possible_keys = [
                key,
                key.replace('net.', ''),
                key.replace('base_model.', ''),
                key.replace('net.', '').replace('base_model.', ''),
            ]
            
            matched_key = None
            for pk in possible_keys:
                if pk in base_model_keys:
                    matched_key = pk
                    break
            
            if matched_key is not None:
                # Check if shapes match
                if value.shape == base_model_state[matched_key].shape:
                    new_state_dict[matched_key] = value
                    loaded_keys.append(matched_key)
                else:
                    skipped_keys.append((key, f"shape mismatch: {value.shape} vs {base_model_state[matched_key].shape}"))
            else:
                skipped_keys.append((key, "not in base model"))
        
        # Load the matched weights
        if new_state_dict:
            self.base_model.load_state_dict(new_state_dict, strict=False)
        
        print(f"Loaded {len(loaded_keys)} weight tensors to Base Model")
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} tensors (expected for conv_in mismatch)")
        
        return loaded_keys, skipped_keys
    
    def sample_t(self, n: int, device=None):
        """Sample timesteps from logit-normal distribution"""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)
    
    def forward(self, ct, mind_features=None):
        """
        Training forward pass.
        
        Stage 1: Unconditional training on CT
        Stage 2: Conditional training with MIND features
        
        Args:
            ct: Target CT image (N, 1, H, W)
            mind_features: MIND condition (N, C, H, W), None for Stage 1
        
        Returns:
            Loss value
        """
        t = self.sample_t(ct.size(0), device=ct.device).view(-1, *([1] * (ct.ndim - 1)))
        e = torch.randn_like(ct) * self.noise_scale
        
        # Flow matching: zt = t * x + (1 - t) * noise
        zt = t * ct + (1 - t) * e
        
        # True velocity
        v = (ct - zt) / (1 - t).clamp_min(self.t_eps)
        
        # Predict
        if self.stage == 'stage1' or mind_features is None:
            # Unconditional
            ct_pred = self.base_model(x=zt, timesteps=t.flatten())
        else:
            # ControlNet guided
            ct_pred = self.model(x=zt, timesteps=t.flatten(), controlnet_cond=mind_features)
        
        # Predicted velocity
        v_pred = (ct_pred - zt) / (1 - t).clamp_min(self.t_eps)
        
        # L2 loss
        loss = ((v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()
        
        return loss
    
    @torch.no_grad()
    def generate(self, mind_features=None, batch_size=1):
        """
        Generate CT images.
        
        Args:
            mind_features: MIND condition (N, C, H, W), None for unconditional
            batch_size: Number of samples (used if mind_features is None)
        
        Returns:
            Generated CT images (N, 1, H, W)
        """
        if mind_features is not None:
            device = mind_features.device
            bsz = mind_features.size(0)
        else:
            device = next(self.parameters()).device
            bsz = batch_size
        
        # Initialize with noise
        z = self.noise_scale * torch.randn(bsz, 1, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
        timesteps = timesteps.view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)
        
        stepper = self._euler_step if self.method == "euler" else self._heun_step
        
        # ODE integration
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, mind_features)
        
        # Final step
        z = self._euler_step(z, timesteps[-2], timesteps[-1], mind_features)
        
        return z
    
    @torch.no_grad()
    def _forward_sample(self, z, t, mind_features=None):
        """Forward pass during sampling"""
        if self.stage == 'stage1' or mind_features is None:
            ct_pred = self.base_model(x=z, timesteps=t.flatten())
        else:
            ct_pred = self.model(x=z, timesteps=t.flatten(), controlnet_cond=mind_features)
        
        v_pred = (ct_pred - z) / (1.0 - t).clamp_min(self.t_eps)
        return v_pred
    
    @torch.no_grad()
    def _euler_step(self, z, t, t_next, mind_features=None):
        v_pred = self._forward_sample(z, t, mind_features)
        return z + (t_next - t) * v_pred
    
    @torch.no_grad()
    def _heun_step(self, z, t, t_next, mind_features=None):
        v_pred_t = self._forward_sample(z, t, mind_features)
        z_euler = z + (t_next - t) * v_pred_t
        v_pred_next = self._forward_sample(z_euler, t_next, mind_features)
        v_pred = 0.5 * (v_pred_t + v_pred_next)
        return z + (t_next - t) * v_pred
    
    @torch.no_grad()
    def update_ema(self):
        """Update EMA parameters"""
        # Only update EMA for trainable parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        if self.ema_params1 is None:
            return
        
        for targ, src in zip(self.ema_params1, trainable_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, trainable_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
