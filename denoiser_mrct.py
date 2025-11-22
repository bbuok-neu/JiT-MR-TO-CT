"""
Denoiser for MR-to-CT Synthesis
Modified to concatenate MR with zt at each step
Prediction: vθ = (xθ - concat(zt, mr))/(1-t)
"""
import torch
import torch.nn as nn
from model_mrct import JiT_MRCT_models


class Denoiser_MRCT(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_MRCT_models[args.model](
            input_size=args.img_size,
            in_channels=2,  # zt + MR condition
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size

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

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, ct, mr):
        """
        Training forward pass for MR-to-CT synthesis
        
        Args:
            ct: Target CT image (N, 1, H, W)
            mr: Condition MR image (N, 1, H, W)
        
        Returns:
            loss: L2 loss between predicted and true velocity
        """
        t = self.sample_t(ct.size(0), device=ct.device).view(-1, *([1] * (ct.ndim - 1)))
        e = torch.randn_like(ct) * self.noise_scale

        # Compute noisy CT: zt = t * ct + (1 - t) * noise
        zt = t * ct + (1 - t) * e
        
        # True velocity: v = (ct - zt) / (1 - t)
        v = (ct - zt) / (1 - t).clamp_min(self.t_eps)

        # Concatenate zt and MR condition
        concat_input = torch.cat([zt, mr], dim=1)  # (N, 2, H, W)
        
        # Predict CT from concatenated input
        ct_pred = self.net(concat_input, t.flatten())
        
        # Predicted velocity: v_pred = (ct_pred - zt) / (1 - t)
        # Note: We use zt here, not concat(zt, mr), as per the formula
        v_pred = (ct_pred - zt) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def generate(self, mr):
        """
        Generate CT from MR image using trained model
        
        Args:
            mr: Condition MR image (N, 1, H, W)
        
        Returns:
            Generated CT image (N, 1, H, W)
        """
        device = mr.device
        bsz = mr.size(0)
        
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
            z = stepper(z, t, t_next, mr)
        
        # Last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], mr)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, mr):
        """
        Forward pass during sampling
        
        Args:
            z: Current state (N, 1, H, W)
            t: Current timestep (N, 1, 1, 1)
            mr: Condition MR image (N, 1, H, W)
        
        Returns:
            Predicted velocity
        """
        # Concatenate z and MR condition
        concat_input = torch.cat([z, mr], dim=1)  # (N, 2, H, W)
        
        # Predict CT
        ct_pred = self.net(concat_input, t.flatten())
        
        # Compute velocity
        v_pred = (ct_pred - z) / (1.0 - t).clamp_min(self.t_eps)
        
        return v_pred

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, mr):
        v_pred = self._forward_sample(z, t, mr)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, mr):
        v_pred_t = self._forward_sample(z, t, mr)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, mr)

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
