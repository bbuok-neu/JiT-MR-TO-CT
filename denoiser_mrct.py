"""
Denoiser for Zero-Shot MR-to-CT Synthesis using MONAI's DiffusionModelUNet

Uses MIND (Modality Independent Neighbourhood Descriptor) features for conditioning:
- Training: Concatenate MIND(CT) features with noisy CT
- Inference: Use MIND(MR) features for zero-shot synthesis

Prediction: vθ = (xθ - zt)/(1-t), where input is concat(zt, mind_features)
"""
import torch
import torch.nn as nn
from generative.networks.nets import DiffusionModelUNet

from mind import get_mind_channels


class Denoiser_MRCT(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        
        # Get MIND parameters with defaults for backward compatibility
        mind_neigh_size = getattr(args, 'mind_neigh_size', 7)
        mind_neigh4 = getattr(args, 'mind_neigh4', False)
        
        # Calculate number of MIND feature channels
        mind_channels = get_mind_channels(mind_neigh_size, mind_neigh4)
        in_channels = 1 + mind_channels  # noisy CT (1ch) + MIND features
        
        # Use MONAI's DiffusionModelUNet for MR-to-CT synthesis
        # Input: in_channels = 1 (zt) + MIND channels (e.g., 48 for neigh_size=7)
        # Output: 1 channel (predicted CT)
        self.net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=in_channels,  # zt (1ch) + MIND features
            out_channels=1,  # predicted CT
            num_res_blocks=(2, 2, 2, 2),  # MONAI expects tuple/list, one value per level
            num_channels=(64, 128, 256, 512),
            attention_levels=(False, False, True, True),
            norm_num_groups=32,
            num_head_channels=(64, 128, 256, 512),  # Typically matching num_channels (MONAI best practice)
            with_conditioning=False,  # We use concatenation, not cross-attention conditioning
            resblock_updown=True,  # Include updown sampling in residual blocks
        )
        self.img_size = args.img_size
        self.mind_channels = mind_channels

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
        
        print(f"Denoiser initialized with {in_channels} input channels (1 + {mind_channels} MIND features)")

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, ct, mind_features):
        """
        Training forward pass for zero-shot MR-to-CT synthesis
        
        Args:
            ct: Target CT image (N, 1, H, W)
            mind_features: MIND feature condition (N, C, H, W) where C = neigh_size^2-1 or 4
                          During training, this is MIND(CT)
        
        Returns:
            loss: L2 loss between predicted and true velocity
        """
        t = self.sample_t(ct.size(0), device=ct.device).view(-1, *([1] * (ct.ndim - 1)))
        e = torch.randn_like(ct) * self.noise_scale

        # Compute noisy CT: zt = t * ct + (1 - t) * noise
        zt = t * ct + (1 - t) * e
        
        # True velocity: v = (ct - zt) / (1 - t)
        v = (ct - zt) / (1 - t).clamp_min(self.t_eps)

        # Concatenate zt and MIND features condition
        concat_input = torch.cat([zt, mind_features], dim=1)  # (N, 1+C, H, W)
        
        # Predict CT from concatenated input using MONAI UNet
        # API: forward(x, timesteps, context=None)
        ct_pred = self.net(x=concat_input, timesteps=t.flatten())
        
        # Predicted velocity: v_pred = (ct_pred - zt) / (1 - t)
        # Note: We compute velocity in the original CT space (using zt, not the concatenated input)
        # because the velocity formula v = (x - z) / (1-t) is defined for the data space
        v_pred = (ct_pred - zt) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def generate(self, mind_features):
        """
        Generate CT from MIND features using trained model
        
        For zero-shot MR-to-CT synthesis, pass MIND(MR) features.
        
        Args:
            mind_features: MIND feature condition (N, C, H, W)
                          For inference, this should be MIND(MR)
        
        Returns:
            Generated CT image (N, 1, H, W)
        """
        device = mind_features.device
        bsz = mind_features.size(0)
        
        # Initialize with noise
        z = self.noise_scale * torch.randn(bsz, 1, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ODE integration
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, mind_features)
        
        # Last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], mind_features)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, mind_features):
        """
        Forward pass during sampling
        
        Args:
            z: Current state (N, 1, H, W)
            t: Current timestep (N, 1, 1, 1)
            mind_features: MIND feature condition (N, C, H, W)
        
        Returns:
            Predicted velocity
        """
        # Concatenate z and MIND features condition
        concat_input = torch.cat([z, mind_features], dim=1)  # (N, 1+C, H, W)
        
        # Predict CT using MONAI DiffusionModelUNet
        ct_pred = self.net(x=concat_input, timesteps=t.flatten())
        
        # Compute velocity
        v_pred = (ct_pred - z) / (1.0 - t).clamp_min(self.t_eps)
        
        return v_pred

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, mind_features):
        v_pred = self._forward_sample(z, t, mind_features)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, mind_features):
        v_pred_t = self._forward_sample(z, t, mind_features)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, mind_features)

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

