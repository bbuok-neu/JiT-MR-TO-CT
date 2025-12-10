"""
 Denoiser for MR-to-CT Synthesis
 Modified to concatenate MR with zt at each step
 Prediction: vθ = (xθ - concat(zt, mr))/(1-t)
"""
import os
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
            in_channels=6,  # 3-channel zt + 3-channel MR condition
            out_channels=3,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.use_pretrained = getattr(args, "use_pretrained", False)
        self.pretrained_path = getattr(args, "pretrained_path", "")

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

        if self.use_pretrained and self.pretrained_path:
            self._load_pretrained_weights(self.pretrained_path)

    def _to_three_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat single-channel inputs to three channels for pretrained compatibility."""
        if x.dim() == 4 and x.size(1) == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def _load_pretrained_weights(self, path: str):
        if not os.path.exists(path):
            print(f"Pretrained weight path {path} not found. Skipping pretrained loading.")
            return
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        processed_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
            if new_key.startswith("net."):
                new_key = new_key[len("net."):]
            processed_state_dict[new_key] = v

        patch_key = "x_embedder.proj1.weight"
        if patch_key in processed_state_dict:
            w = processed_state_dict[patch_key]
            if self.net.in_channels % w.shape[1] == 0:
                repeat_factor = self.net.in_channels // w.shape[1]
                # Expand pretrained patch embedding weights to match duplicated MR + zt channels
                processed_state_dict[patch_key] = w.repeat(1, repeat_factor, 1, 1)
            else:
                print(f"Pretrained patch embedding channels ({w.shape[1]}) do not divide target in_channels {self.net.in_channels}; skipping expansion.")

        missing, unexpected = self.net.load_state_dict(processed_state_dict, strict=False)
        print(f"Loaded pretrained weights from {path}")
        if missing:
            print(f"Missing keys (ignored): {missing}")
        if unexpected:
            print(f"Unexpected keys (ignored): {unexpected}")

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
            loss: L2 loss between predicted and true velocity for 3-channel CT prediction
        """
        ct = self._to_three_channels(ct)
        mr = self._to_three_channels(mr)

        t = self.sample_t(ct.size(0), device=ct.device).view(-1, *([1] * (ct.ndim - 1)))
        e = torch.randn_like(ct) * self.noise_scale

        # Compute noisy CT: zt = t * ct + (1 - t) * noise
        zt = t * ct + (1 - t) * e
        
        # True velocity: v = (ct - zt) / (1 - t)
        v = (ct - zt) / (1 - t).clamp_min(self.t_eps)

        # Concatenate zt and MR condition
        concat_input = torch.cat([zt, mr], dim=1)  # (N, 6, H, W)
        
        # Predict CT from concatenated input
        ct_pred = self.net(concat_input, t.flatten())
        
        # Predicted velocity: v_pred = (ct_pred - zt) / (1 - t)
        # Note: We compute velocity in the original CT space (using zt, not the concatenated input)
        # because the velocity formula v = (x - z) / (1-t) is defined for the data space
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
         mr: Condition MR image (N, 1, H, W) or (N, 3, H, W)
        
        Returns:
            Generated CT image (N, 3, H, W)
        """
        device = mr.device
        bsz = mr.size(0)
        
        # Initialize with noise
        mr = self._to_three_channels(mr)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
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
            z: Current state (N, 3, H, W)
            t: Current timestep (N, 1, 1, 1)
            mr: Condition MR image (N, 1, H, W) or (N, 3, H, W)
        
        Returns:
            Predicted velocity
        """
        mr = self._to_three_channels(mr)
        # Concatenate z and MR condition
        concat_input = torch.cat([z, mr], dim=1)  # (N, 6, H, W)
        
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
