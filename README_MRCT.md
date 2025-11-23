# MR-to-CT Synthesis with MONAI DiffusionModelUNet

This document describes how to use the MR-to-CT synthesis framework with MONAI's DiffusionModelUNet.

## Overview

This implementation uses MONAI's official `DiffusionModelUNet` for conditional image-to-image translation, specifically MR-to-CT synthesis. The framework is based on the `copilot/implement-mr-ct-synthesis` branch with the model replaced by MONAI's UNet.

## Dataset Structure

The paired MR-CT dataset should be organized as follows:

```
dataset/
├── mr/
│   ├── train/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── test/
│       ├── image001.jpg
│       └── ...
└── ct/
    ├── train/
    │   ├── image001.jpg  (paired with mr/train/image001.jpg)
    │   ├── image002.jpg
    │   └── ...
    └── test/
        ├── image001.jpg
        └── ...
```

**Requirements:**
- All images should be single-channel grayscale in JPG format
- MR and CT images must have matching filenames (paired)
- Images are automatically resized to `--img_size` (default 256×256)

## Installation

1. Install dependencies:
```bash
conda env create -f environment.yaml
conda activate jit
```

2. Verify MONAI is installed:
```bash
python -c "from generative.networks.nets import DiffusionModelUNet; print('MONAI installed successfully')"
```

## Training

### Basic Training

```bash
python main_mrct.py \
  --data_path /path/to/dataset \
  --img_size 256 \
  --batch_size 16 \
  --epochs 200 \
  --output_dir ./output_mrct
```

### Training with Augmentation

```bash
python main_mrct.py \
  --data_path /path/to/dataset \
  --img_size 256 \
  --batch_size 16 \
  --epochs 200 \
  --enable_augmentation \
  --enable_flip \
  --enable_elastic \
  --enable_bias_field \
  --enable_rician_noise \
  --output_dir ./output_mrct
```

### Distributed Training (Multi-GPU)

```bash
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  main_mrct.py \
  --data_path /path/to/dataset \
  --img_size 256 \
  --batch_size 16 \
  --epochs 200 \
  --output_dir ./output_mrct
```

## Evaluation

```bash
python main_mrct.py \
  --data_path /path/to/dataset \
  --img_size 256 \
  --evaluate_only \
  --resume ./output_mrct \
  --output_dir ./output_mrct
```

## Key Arguments

### Dataset Arguments
- `--data_path`: Path to dataset root directory (required)
- `--img_size`: Image size (default: 256)
- `--mr_mean`, `--mr_std`: Z-score normalization for MR (default: 0.5, 0.5)
- `--ct_mean`, `--ct_std`: Z-score normalization for CT (default: 0.5, 0.5)

### Training Arguments
- `--epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size per GPU (default: 16)
- `--lr`: Learning rate (default: computed from blr)
- `--blr`: Base learning rate (default: 5e-5)
- `--weight_decay`: Weight decay (default: 0.0)

### Augmentation Arguments

**Geometric (applied to both MR and CT):**
- `--enable_flip`: Random horizontal flip
- `--enable_elastic`: Elastic deformation
- `--rotation_degrees MIN MAX`: Rotation range in degrees (default: -15 15)
- `--zoom_range MIN MAX`: Zoom/scaling range (default: 0.9 1.1)

**MR-specific (intensity/artifact):**
- `--enable_bias_field`: Bias field simulation
- `--enable_motion_ghosting`: Motion ghosting
- `--enable_rician_noise`: Rician noise injection
- `--enable_gamma`: Gamma correction
- `--enable_cutout`: Random cutout/erasing

### Sampling Arguments
- `--sampling_method`: ODE sampling method (default: heun)
- `--num_sampling_steps`: Number of sampling steps (default: 50)

### Checkpointing
- `--output_dir`: Directory to save outputs (default: ./output_mrct)
- `--resume`: Checkpoint folder to resume from
- `--save_last_freq`: Frequency (epochs) to save checkpoints (default: 5)

## Model Architecture

The `denoiser_mrct.py` uses MONAI's DiffusionModelUNet:

```python
DiffusionModelUNet(
    spatial_dims=2,
    in_channels=2,  # zt (noisy CT) + MR condition
    out_channels=1,  # predicted CT
    num_res_blocks=(2, 2, 2, 2),  # 2 blocks per level
    num_channels=(64, 128, 256, 512),  # Channel progression
    attention_levels=(False, False, True, True),  # Attention at levels 3-4
    norm_num_groups=32,
    num_head_channels=(64, 128, 256, 512),  # Matches num_channels
    with_conditioning=False,  # Using concatenation instead
    resblock_updown=True,
)
```

**Key Features:**
- 4 resolution levels with skip connections
- Self-attention at deeper levels (256, 512 channels)
- Group normalization (32 groups)
- Timestep conditioning via sinusoidal embeddings
- ~38M parameters

## Velocity Prediction

The model predicts velocity in the flow ODE framework:

```
v_θ = (x_θ - z_t) / (1 - t)
```

Where:
- `x_θ`: Predicted clean CT
- `z_t`: Noisy CT at time t
- `t`: Diffusion timestep in [0, 1]

**Training:**
1. Sample timestep `t ~ Sigmoid(N(μ, σ))`
2. Create noisy CT: `z_t = t * CT + (1-t) * noise`
3. Input to UNet: `concat([z_t, MR])`
4. Predict clean CT: `CT_pred = UNet(concat([z_t, MR]), t)`
5. Compute velocity: `v_pred = (CT_pred - z_t) / (1-t)`
6. Loss: `||v - v_pred||²`

## Output Structure

```
output_mrct/
├── checkpoint-last.pth  # Latest checkpoint
├── checkpoint-100.pth   # Periodic checkpoints
├── events.out.tfevents.* # TensorBoard logs
└── samples/             # Generated samples during evaluation
```

## Monitoring with TensorBoard

```bash
tensorboard --logdir=./output_mrct
```

Metrics tracked:
- Training loss
- Learning rate
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MAE (Mean Absolute Error)

## Common Issues

### Out of Memory
- Reduce `--batch_size`
- Reduce `--img_size`
- Enable gradient checkpointing (if implemented)

### Dataset Loading Errors
- Verify directory structure matches requirements
- Check that MR and CT have same number of images
- Ensure image filenames match between MR and CT

### Slow Training
- Increase `--num_workers` for data loading
- Use distributed training with multiple GPUs
- Reduce `--eval_freq` to evaluate less frequently

## Comparison: ImageNet vs MR-CT

| Aspect | ImageNet (`main_jit.py`) | MR-CT (`main_mrct.py`) |
|--------|-------------------------|------------------------|
| Model | JiT ViT or MONAI UNet | MONAI UNet |
| Input | Single RGB image | Paired MR-CT images |
| Channels | 3 → 3 | 2 → 1 |
| Conditioning | Class labels | MR image |
| Dataset | ImageNet folders | Paired MR/CT folders |
| Augmentation | Standard vision | Medical imaging |

## References

- **Framework**: Based on `copilot/implement-mr-ct-synthesis` branch
- **Model**: MONAI GenerativeModels (https://github.com/Project-MONAI/GenerativeModels)
- **Base**: JiT paper "Back to Basics: Let Denoising Generative Models Denoise"

## Citation

If you use this code, please cite:

```bibtex
@article{li2025jit,
  title={Back to Basics: Let Denoising Generative Models Denoise},
  author={Li, Tianhong and He, Kaiming},
  journal={arXiv preprint arXiv:2511.13720},
  year={2025}
}
```

And MONAI:

```bibtex
@article{cardoso2022monai,
  title={MONAI: An open-source framework for deep learning in healthcare},
  author={Cardoso, M Jorge and others},
  journal={arXiv preprint arXiv:2211.02701},
  year={2022}
}
```
