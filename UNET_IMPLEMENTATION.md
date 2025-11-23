# DiffusionModelUNet for MR-to-CT Synthesis

This document describes the modifications made to support conditional image-to-image translation using a U-Net architecture, specifically designed for MR-to-CT synthesis.

## Architecture Changes

### From ViT to U-Net

The original JiT model used a Vision Transformer (ViT) architecture with class-conditional generation. This has been replaced with a **DiffusionModelUNet** architecture that supports image-conditional generation.

### Key Components

#### 1. DiffusionModelUNet (`unet.py`)

A U-Net architecture implementation with the following features:

- **Spatial dimensions**: 2D (for slice-based medical imaging)
- **Input channels**: 2 (noisy target + conditioning image)
- **Output channels**: 1 (predicted target)
- **Architecture**:
  - Encoder path with 4 levels: (64, 128, 256, 512) channels
  - Self-attention at deeper levels (last 2 levels)
  - Residual blocks with Group Normalization
  - Skip connections from encoder to decoder
  - Timestep conditioning via sinusoidal embeddings

**Configuration**:
```python
DiffusionModelUNet(
    spatial_dims=2,
    in_channels=2,  # noisy_CT (1ch) + MR_condition (1ch)
    out_channels=1,  # predicted_CT
    num_channels=(64, 128, 256, 512),
    attention_levels=(False, False, True, True),
    num_res_blocks=2,
    num_head_channels=32,
    dropout=0.0,
)
```

#### 2. Modified Denoiser (`denoiser.py`)

The `Denoiser` class has been updated to use the U-Net architecture:

**Key changes**:
- Uses `DiffusionModelUNet` instead of JiT ViT model
- Supports configurable `condition_channels` and `target_channels`
- Default: 1-channel condition (MR) + 1-channel target (CT)
- Forward pass concatenates condition with noisy target
- Sampling methods updated for image conditioning

## Conditional Diffusion Framework

### Training

During training, the model:
1. Samples a timestep `t`
2. Creates noisy version of target: `z = t * x + (1-t) * noise`
3. Concatenates with condition: `input = concat([z, condition], dim=1)`
4. Predicts velocity: `v_pred = (x_pred - z) / (1-t)`
5. Computes L2 loss against true velocity

### Velocity Prediction

The model predicts the velocity field `v_θ`:

```
v_θ = (x_θ - z_t) / (1 - t)
```

Where:
- `x_θ`: Predicted clean image
- `z_t`: Noisy input at time t
- `t`: Timestep in [0, 1]

### Inference

During generation:
1. Start with pure noise for the target
2. Use the conditioning image (MR) throughout the process
3. Iteratively denoise using ODE solver (Euler or Heun)
4. Concatenate condition at each step: `input = concat([z_t, condition], dim=1)`

## Usage

### For MR-to-CT Synthesis

The architecture is designed for medical image translation where:
- **Input**: MR scan (grayscale, 1 channel)
- **Output**: CT scan (grayscale, 1 channel)
- **Model input**: 2 channels (noisy CT + MR condition)
- **Model output**: 1 channel (predicted CT)

### Current Implementation Note

The current implementation includes a **placeholder** for actual MR-to-CT data:
- In `forward()`: Uses the clean image as its own condition (self-conditioning)
- This is temporary until proper paired MR-CT datasets are integrated

To use with real MR-CT data:
1. Modify the data loader to return paired (MR, CT) images
2. Update `engine_jit.py` training loop to pass both images
3. Replace `condition = x` in `forward()` with actual MR image

### Training Command (Future)

```bash
torchrun --nproc_per_node=8 main_jit.py \
  --model UNet \
  --img_size 256 \
  --condition_channels 1 \
  --target_channels 1 \
  --batch_size 128 \
  --epochs 600 \
  --data_path /path/to/mr-ct-dataset \
  --output_dir ./output_unet
```

## Model Parameters

The DiffusionModelUNet has significantly different parameter count compared to JiT:

- **JiT-B/16**: ~86M parameters (ViT-based)
- **DiffusionModelUNet**: ~38M parameters (U-Net with config above)

The U-Net is more parameter-efficient while being well-suited for dense prediction tasks like image-to-image translation.

## Technical Details

### Skip Connections

The U-Net uses skip connections to preserve spatial information:
- Features from encoder are concatenated with decoder features
- Helps maintain fine-grained details from the input
- Critical for medical imaging where small structures matter

### Attention Mechanism

Self-attention is applied at deeper levels:
- Levels 3 and 4 (256 and 512 channels)
- 32-dimensional head channels
- Helps capture long-range dependencies
- Important for understanding anatomical context

### Timestep Conditioning

Timestep information is injected via:
1. Sinusoidal positional encoding
2. MLP projection to feature dimension
3. Added to features in each residual block

This allows the network to adapt its behavior based on the noise level.

## Differences from Original JiT

| Aspect | Original JiT | Modified (U-Net) |
|--------|-------------|------------------|
| Architecture | Vision Transformer | U-Net |
| Conditioning | Class labels | Images (MR scans) |
| Input | Noisy image | Noisy image + condition |
| Spatial processing | Patch-based (16x16) | Convolutional |
| Attention | Global (all patches) | Selective (deep levels) |
| Skip connections | None | Yes (encoder-decoder) |
| Use case | Class-conditional generation | Image-to-image translation |

## Future Enhancements

1. **Data Pipeline**: Integrate paired MR-CT dataset loading
2. **Multi-scale**: Support different resolution levels
3. **3D Support**: Extend to volumetric (3D) medical images
4. **Classifier-free Guidance**: Adapt CFG for image conditioning
5. **Perceptual Loss**: Add perceptual/adversarial losses for better quality

## References

- Original JiT paper: "Back to Basics: Let Denoising Generative Models Denoise"
- MONAI Generative Models: https://github.com/Project-MONAI/GenerativeModels
- Diffusion Models Beat GANs on Image Synthesis (U-Net for diffusion)
