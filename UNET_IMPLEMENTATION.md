# DiffusionModelUNet for MR-to-CT Synthesis

This document describes the modifications made to support conditional image-to-image translation using a U-Net architecture from MONAI GenerativeModels, specifically designed for MR-to-CT synthesis.

## Architecture Changes

### From ViT to U-Net (MONAI)

The original JiT model used a Vision Transformer (ViT) architecture with class-conditional generation. This has been replaced with **MONAI's DiffusionModelUNet** architecture that supports image-conditional generation.

**Key Change**: We now use the official MONAI implementation instead of a custom UNet:
- **Package**: `monai-generative==0.2.3`
- **Import**: `from generative.networks.nets import DiffusionModelUNet`
- **Reference**: [MONAI GenerativeModels](https://github.com/Project-MONAI/GenerativeModels)
- **Example**: [MOTFM (Conditional Flow Matching)](https://github.com/milad1378yz/MOTFM)

### Installation

```bash
pip install monai-generative==0.2.3
```

Or add to `environment.yaml`:
```yaml
pip:
  - monai-generative==0.2.3
```

### Key Components

#### 1. MONAI DiffusionModelUNet (`denoiser.py`)

Using MONAI's official implementation with the following configuration:

**Configuration**:
```python
from generative.networks.nets import DiffusionModelUNet

model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=2,  # noisy_CT (1ch) + MR_condition (1ch)
    out_channels=1,  # predicted_CT
    num_res_blocks=(2, 2, 2, 2),  # One value per level
    num_channels=(64, 128, 256, 512),
    attention_levels=(False, False, True, True),
    norm_num_groups=32,
    num_head_channels=(32, 32, 32, 32),  # Must match num_channels length
    with_conditioning=False,  # We use concatenation, not cross-attention
    resblock_updown=True,
)
```

**Key Features**:
- **Spatial dimensions**: 2D (for slice-based medical imaging)
- **Input channels**: 2 (noisy target + conditioning image)
- **Output channels**: 1 (predicted target)
- **Architecture**:
  - 4 levels with (64, 128, 256, 512) channels
  - Self-attention at deeper levels (last 2 levels)
  - 2 residual blocks per level
  - Group Normalization with 32 groups
  - Skip connections from encoder to decoder
  - Timestep conditioning via sinusoidal embeddings

**API**:
```python
# Forward pass
output = model(x=input_tensor, timesteps=timestep_tensor, context=None)

# Parameters:
# - x: Input tensor (B, C, H, W) - concatenated [noisy_CT, MR_condition]
# - timesteps: Timestep tensor (B,) - diffusion timestep in [0, 1]
# - context: Optional cross-attention conditioning (not used in our case)
```

#### 2. Modified Denoiser (`denoiser.py`)

The `Denoiser` class has been updated to use MONAI's DiffusionModelUNet:

**Key changes**:
- Imports `DiffusionModelUNet` from `generative.networks.nets` (MONAI)
- Uses official MONAI implementation instead of custom UNet
- Supports configurable `condition_channels` and `target_channels`
- Default: 1-channel condition (MR) + 1-channel target (CT)
- Forward pass concatenates condition with noisy target
- Sampling methods updated for image conditioning
- API calls use explicit parameter names: `model(x=input, timesteps=t)`

**Note**: The custom UNet implementation is preserved in `unet_custom.py` for reference only.

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
3. **3D Support**: Extend to volumetric (3D) medical images using `spatial_dims=3`
4. **Classifier-free Guidance**: Adapt CFG for image conditioning
5. **Perceptual Loss**: Add perceptual/adversarial losses for better quality
6. **Cross-attention**: Explore MONAI's `with_conditioning=True` for alternative conditioning

## Why MONAI Instead of Custom Implementation?

### Advantages of Using MONAI

1. **Official Support**: Maintained by Project MONAI with regular updates
2. **Tested & Optimized**: Extensively tested in medical imaging applications
3. **Feature Rich**: Includes advanced features like flash attention, flexible conditioning
4. **Community**: Large community and extensive documentation
5. **Compatibility**: Works seamlessly with other MONAI components
6. **Performance**: Optimized implementation with better memory efficiency

### API Differences

The MONAI implementation has some parameter differences from our custom UNet:

| Parameter | Custom UNet | MONAI UNet | Notes |
|-----------|-------------|------------|-------|
| `num_res_blocks` | `int` | `tuple/list` | MONAI expects per-level values |
| `num_head_channels` | `int` | `tuple/list` | MONAI expects per-level values |
| `dropout` | `float` | Not used | MONAI has different dropout parameters |
| Forward API | `(x, t)` | `(x, timesteps, context)` | MONAI has explicit parameter names |
| Extra params | - | `norm_num_groups`, `resblock_updown`, etc. | MONAI has more configuration options |

### Custom UNet Reference

The original custom implementation is preserved in `unet_custom.py` for:
- Understanding the architecture
- Reference for those who prefer standalone implementation
- Educational purposes
- Debugging and comparison

## References

- Original JiT paper: "Back to Basics: Let Denoising Generative Models Denoise"
- MONAI Generative Models: https://github.com/Project-MONAI/GenerativeModels
- MONAI Documentation: https://docs.monai.io/
- MOTFM (Example Usage): https://github.com/milad1378yz/MOTFM
- Diffusion Models Beat GANs on Image Synthesis (U-Net for diffusion)
