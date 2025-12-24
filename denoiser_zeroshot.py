"""
Denoiser for Zero-Shot MR-to-CT Synthesis using MONAI's DiffusionModelUNet
Uses HOG (Histogram of Oriented Gradients) as cross-attention conditioning

Training: Use CT images with CT-HOG as condition
Inference: Use MR-HOG to guide CT synthesis (zero-shot domain transfer)

Key difference from original approach:
- Instead of concatenating MR with noisy CT (in_channels=2)
- We use cross-attention conditioning with HOG features (with_conditioning=True)
- HOG provides structural information that transfers between MR and CT modalities
"""
import importlib.util
import torch
import torch.nn as nn
from diffusion_model_unet import DiffusionModelUNet

# Check if xformers is available for flash attention
has_xformers = importlib.util.find_spec("xformers") is not None


class Denoiser_ZeroShot(nn.Module):
    """
    Zero-shot MR-to-CT synthesis using HOG-conditioned diffusion.
    
    Architecture:
    - Input: 1 channel (noisy CT during training, noise during inference)
    - Output: 1 channel (predicted CT)
    - Conditioning: HOG features via cross-attention
      - Shape: (B, 961, 36) for 256x256 images
      - Sequence length: 961 (31x31 blocks)
      - Feature dim: 36 (2x2 cells * 9 bins per block)
    """
    
    def __init__(self, args):
        super().__init__()
        
        # HOG feature dimensions for cross-attention
        # For 256x256 images with cell_size=8, block_size=2, num_bins=9
        # cross_attention_dim = block_size * block_size * num_bins = 2*2*9 = 36
        self.cross_attention_dim = getattr(args, 'cross_attention_dim', 36)
        
        # Check if flash attention can be used
        use_flash = has_xformers and torch.cuda.is_available()
        if not use_flash:
            print("Warning: Flash attention disabled (xformers not installed or CUDA not available)")
        
        # Use MONAI's DiffusionModelUNet with cross-attention conditioning
        # Input: 1 channel (noisy CT only, no concatenation)
        # Conditioning: HOG features via cross-attention mechanism
        self.net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,  # Only noisy CT (no concatenation with condition)
            out_channels=1,  # Predicted CT
            num_res_blocks=(2, 2, 2, 2),
            num_channels=(64, 128, 256, 512),
            attention_levels=(False, False, True, True),  # Cross-attention at higher resolution levels
            norm_num_groups=32,
            num_head_channels=(64, 128, 256, 512),
            with_conditioning=True,  # Enable cross-attention conditioning
            cross_attention_dim=self.cross_attention_dim,  # HOG feature dimension (36)
            resblock_updown=True,
            use_flash_attention=use_flash,  # Enable flash attention if available
        )
        
        self.img_size = args.img_size
        
        # Diffusion parameters
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        
        # EMA parameters
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None
        
        # Sampling parameters
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
    
    def sample_t(self, n: int, device=None):
        """Sample timesteps from logit-normal distribution."""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)
    
    def forward(self, ct, hog_features):
        """
        Training forward pass for zero-shot CT synthesis.
        
        Args:
            ct: Target CT image (N, 1, H, W)
            hog_features: HOG conditioning features (N, seq_len, feature_dim)
                         e.g., (N, 961, 36) for 256x256 images
        
        Returns:
            loss: L2 loss between predicted and true velocity
        """
        # Sample timesteps
        t = self.sample_t(ct.size(0), device=ct.device).view(-1, *([1] * (ct.ndim - 1)))
        
        # Sample noise
        e = torch.randn_like(ct) * self.noise_scale
        
        # Compute noisy CT: zt = t * ct + (1 - t) * noise
        zt = t * ct + (1 - t) * e
        
        # True velocity: v = (ct - zt) / (1 - t)
        v = (ct - zt) / (1 - t).clamp_min(self.t_eps)
        
        # Predict CT from noisy input with HOG conditioning
        # Uses cross-attention mechanism in the UNet
        ct_pred = self.net(x=zt, timesteps=t.flatten(), context=hog_features)
        
        # Predicted velocity: v_pred = (ct_pred - zt) / (1 - t)
        v_pred = (ct_pred - zt) / (1 - t).clamp_min(self.t_eps)
        
        # L2 loss on velocity
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()
        
        return loss
    
    @torch.no_grad()
    def generate(self, hog_features, batch_size=None):
        """
        Generate CT from HOG features using trained model.
        
        This is the zero-shot inference: use MR-derived HOG to guide CT synthesis.
        
        Args:
            hog_features: HOG conditioning features (N, seq_len, feature_dim)
                         Computed from MR images during inference
            batch_size: Optional batch size (inferred from hog_features if not provided)
        
        Returns:
            Generated CT image (N, 1, H, W)
        """
        device = hog_features.device
        bsz = batch_size if batch_size is not None else hog_features.size(0)
        
        # Initialize with noise
        z = self.noise_scale * torch.randn(bsz, 1, self.img_size, self.img_size, device=device)
        
        # Timestep schedule
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
        timesteps = timesteps.view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)
        
        # Select stepper
        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(f"Sampling method {self.method} not implemented")
        
        # ODE integration
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, hog_features)
        
        # Last step with Euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], hog_features)
        
        return z
    
    @torch.no_grad()
    def _forward_sample(self, z, t, hog_features):
        """
        Forward pass during sampling.
        
        Args:
            z: Current state (N, 1, H, W)
            t: Current timestep (N, 1, 1, 1)
            hog_features: HOG conditioning features (N, seq_len, feature_dim)
        
        Returns:
            Predicted velocity
        """
        # Predict CT with HOG conditioning
        ct_pred = self.net(x=z, timesteps=t.flatten(), context=hog_features)
        
        # Compute velocity
        v_pred = (ct_pred - z) / (1.0 - t).clamp_min(self.t_eps)
        
        return v_pred
    
    @torch.no_grad()
    def _euler_step(self, z, t, t_next, hog_features):
        """Euler step for ODE integration."""
        v_pred = self._forward_sample(z, t, hog_features)
        z_next = z + (t_next - t) * v_pred
        return z_next
    
    @torch.no_grad()
    def _heun_step(self, z, t, t_next, hog_features):
        """Heun step (2nd order) for ODE integration."""
        v_pred_t = self._forward_sample(z, t, hog_features)
        
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, hog_features)
        
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next
    
    @torch.no_grad()
    def update_ema(self):
        """Update exponential moving average of model parameters."""
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
