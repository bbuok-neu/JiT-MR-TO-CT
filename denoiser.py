import torch
import torch.nn as nn
from channel_bottleneck_unet import ChannelBottleneckDiffusionModelUNet


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        # Use ChannelBottleneckDiffusionModelUNet for conditional MR-to-CT synthesis
        # This is a modified MONAI DiffusionModelUNet with channel bottleneck layer
        # that mimics JiT's ViT dimension bottleneck for low-dimensional manifold learning.
        # Configure for 2-channel input (noisy_latent + condition) and 1-channel output
        # For MR-to-CT: both MR and CT are grayscale (1 channel each)
        self.condition_channels = getattr(args, 'condition_channels', 1)  # MR condition channels
        self.target_channels = getattr(args, 'target_channels', 1)  # CT target channels
        
        # Initialize ChannelBottleneckDiffusionModelUNet
        # Channel bottleneck: 512 → 128 → 512 between down_blocks and mid_blocks
        # Reference: https://github.com/Project-MONAI/GenerativeModels
        self.net = ChannelBottleneckDiffusionModelUNet(
            spatial_dims=2,
            in_channels=self.target_channels + self.condition_channels,  # noisy target + condition
            out_channels=self.target_channels,  # predicted target
            num_res_blocks=(2, 2, 2, 2),  # MONAI expects tuple/list, one value per level
            num_channels=(64, 128, 256, 512),
            attention_levels=(False, False, True, True),
            norm_num_groups=32,
            num_head_channels=(64, 128, 256, 512),  # MONAI expects tuple/list, typically matching num_channels
            with_conditioning=False,  # We use concatenation, not cross-attention conditioning
            resblock_updown=True,  # Include updown sampling in residual blocks
            bottleneck_channels=128,  # Channel bottleneck: 512 → 128 → 512
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        """
        Forward pass for conditional diffusion.
        For MR-to-CT synthesis:
        - x: target image (CT)
        - condition: conditioning image (MR)
        
        For current ImageNet setup (placeholder):
        - x: image (acts as both target and condition)
        - We use self-conditioning as a placeholder
        
        TODO: When actual MR-CT paired data is available, this method should be updated to:
        1. Accept an additional 'condition' parameter (MR image)
        2. Use that condition instead of self-conditioning
        3. Optionally support conditional dropout for classifier-free guidance
        """
        # Sample timestep
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale
        
        # Create noisy version of target
        z = t * x + (1 - t) * e
        
        # For MR-to-CT: concatenate noisy CT with MR condition
        # PLACEHOLDER: Currently using clean image as condition (self-conditioning)
        # This is a simplified placeholder until actual paired MR-CT data is integrated.
        # In actual MR-to-CT training, replace this with the actual MR image:
        #   condition = mr_image  # where mr_image is passed as an additional parameter
        condition = x  # Placeholder: use clean image as condition
        
        # Concatenate noisy target with condition
        z_cond = torch.cat([z, condition], dim=1)
        
        # Compute target velocity
        v = (x - z) / (1 - t).clamp_min(self.t_eps)
        
        # Predict with MONAI DiffusionModelUNet (takes concatenated input)
        # API: forward(x, timesteps, context=None)
        x_pred = self.net(x=z_cond, timesteps=t.flatten())
        
        # Compute predicted velocity
        # Note: UNet predicts the denoised output directly
        # We need to convert it to velocity: v_theta = (x_theta - z) / (1 - t)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        
        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()
        
        return loss

    @torch.no_grad()
    def generate(self, labels, condition=None):
        """
        Generate images using the diffusion model.
        
        Args:
            labels: Class labels (for compatibility, not used in conditional model)
            condition: Conditioning image (MR image for MR-to-CT synthesis)
                      If None, uses random noise as condition (placeholder)
        """
        device = labels.device
        bsz = labels.size(0)
        
        # Initialize noisy target
        z = self.noise_scale * torch.randn(bsz, self.target_channels, self.img_size, self.img_size, device=device)
        
        # Initialize condition (placeholder: random noise if not provided)
        if condition is None:
            condition = torch.randn(bsz, self.condition_channels, self.img_size, self.img_size, device=device)
        
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, condition)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], condition)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, condition):
        """
        Forward sampling step with conditioning.
        
        Args:
            z: noisy target
            t: timestep
            condition: conditioning image (MR for MR-to-CT)
        """
        # Concatenate noisy target with condition
        z_cond = torch.cat([z, condition], dim=1)
        
        # Predict with MONAI DiffusionModelUNet
        x_pred = self.net(x=z_cond, timesteps=t.flatten())
        v_pred = (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)
        
        return v_pred

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, condition):
        v_pred = self._forward_sample(z, t, condition)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, condition):
        v_pred_t = self._forward_sample(z, t, condition)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, condition)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
