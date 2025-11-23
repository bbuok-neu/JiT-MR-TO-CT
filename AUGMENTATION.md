# Data Augmentation for MR-to-CT Synthesis

This document describes the data augmentation features for medical image synthesis.

## Overview

The framework supports medical-specific data augmentation to improve model generalization and robustness. Augmentations are divided into two categories:

1. **Geometric transforms**: Applied to BOTH MR and CT to maintain spatial alignment
2. **Intensity/artifact transforms**: Applied ONLY to MR to simulate MRI-specific variations

## Supported Augmentations

### Geometric Transforms (Applied to Both MR and CT)

These transforms maintain spatial correspondence between MR-CT pairs:

#### 1. Random Rotation
- **Range**: ±10-15 degrees (configurable)
- **Purpose**: Handles variations in patient positioning
- **Default**: `-15` to `+15` degrees
- **Flag**: Enabled by default

#### 2. Random Horizontal Flip
- **Probability**: 50%
- **Purpose**: Augments dataset by creating mirror images
- **Note**: Appropriate for anatomical structures with bilateral symmetry
- **Flag**: `--enable_flip` (enabled by default)

#### 3. Elastic Deformation
- **Purpose**: Simulates variations in organ shapes across patients
- **Parameters**: 
  - Control points: 7 (default)
  - Max displacement: 7.5 voxels (default)
- **Note**: Only available with TorchIO
- **Flag**: `--enable_elastic` (enabled by default)

#### 4. Grid Distortion
- **Purpose**: Alternative to elastic deformation with grid-based warping
- **Parameters**:
  - Control points: 5 (default)
  - Max displacement: 10.0 voxels (default)
- **Note**: Only available with TorchIO
- **Flag**: `--enable_grid_distortion` (disabled by default)

#### 5. Random Zoom/Scaling
- **Range**: 0.9 to 1.1x (configurable)
- **Purpose**: Handles variations in patient size and FOV
- **Flag**: Enabled by default with `--zoom_range`

### MR-Only Transforms (Intensity/Artifact Simulation)

These transforms are applied ONLY to MR images to simulate MRI-specific variations:

#### 6. Bias Field Simulation
- **Purpose**: Simulates magnetic field inhomogeneity in MRI
- **Effect**: Creates smooth, spatially-varying intensity changes
- **Parameters**: Coefficient = 0.5 (default)
- **Note**: Forces model to rely on structure rather than absolute intensity
- **Flag**: `--enable_bias_field` (disabled by default)

#### 7. Motion Ghosting
- **Purpose**: Simulates patient motion during MRI acquisition
- **Effect**: Adds ghost artifacts along phase encoding direction
- **Parameters**: 
  - Number of ghosts: 2 (default)
  - Intensity: 0.5-1.0 (default)
- **Note**: Common MRI artifact from breathing or movement
- **Flag**: `--enable_motion_ghosting` (disabled by default)

#### 8. Rician Noise
- **Purpose**: Adds MRI-specific noise
- **Effect**: Rician-distributed noise (magnitude of complex Gaussian)
- **Parameters**: Std range: 0.0-0.05 (default)
- **Note**: More realistic than Gaussian noise for MRI
- **Flag**: `--enable_rician_noise` (disabled by default)

#### 9. Gamma Correction
- **Purpose**: Simulates different MRI sequence parameters
- **Effect**: Non-linear intensity adjustment (I_new = I_old^gamma)
- **Parameters**: Gamma range: 0.8-1.2 (default)
- **Note**: Models different T1/T2 weighting or scanner variations
- **Flag**: `--enable_gamma` (disabled by default)

#### 10. Random Cutout/Erasing
- **Purpose**: MAE-style masking for ViT feature learning
- **Effect**: Randomly erases rectangular regions
- **Parameters**:
  - Number of holes: 1-3 (default)
  - Hole size: 20-40 pixels (default)
- **Note**: Improves robustness to missing data
- **Flag**: `--enable_cutout` (disabled by default)

## Implementation

### Using TorchIO (Recommended)

TorchIO provides medical image-specific augmentations with proper handling of 3D/2D medical data.

**Installation**:
```bash
pip install torchio==0.19.6
```

or update conda environment:
```bash
conda env update -f environment.yaml
```

### Fallback to Basic Transforms

If TorchIO is not available, the framework automatically falls back to basic PyTorch/torchvision transforms:
- Rotation and flip using `torchvision.transforms`
- Elastic deformation is disabled (not available without TorchIO)
- Zoom using affine transforms

## Usage

### Command Line Arguments

**Basic usage with geometric augmentations:**
```bash
python main_mrct.py \
    --data_path /path/to/dataset \
    --enable_augmentation \              # Enable augmentation (default: enabled)
    --rotation_degrees -15 15 \          # Rotation range in degrees
    --enable_flip \                       # Enable random flip (default: enabled)
    --enable_elastic \                    # Enable elastic deformation (default: enabled)
    --zoom_range 0.9 1.1 \               # Zoom/scale range
    --use_torchio \                       # Use TorchIO (default: enabled if available)
    ... # other arguments
```

**Advanced usage with MR-only augmentations:**
```bash
python main_mrct.py \
    --data_path /path/to/dataset \
    --enable_augmentation \
    # Geometric (MR+CT)
    --rotation_degrees -15 15 \
    --enable_flip \
    --enable_elastic \
    --zoom_range 0.9 1.1 \
    --enable_grid_distortion \           # Optional: grid distortion
    # MR-only intensity/artifact
    --enable_bias_field \                # Bias field inhomogeneity
    --enable_motion_ghosting \           # Motion artifacts
    --enable_rician_noise \              # Rician noise
    --enable_gamma \                     # Gamma correction
    --enable_cutout \                    # Random cutout/erasing
    ... # other arguments
```

### Disabling Augmentation

To disable all augmentation:
```bash
python main_mrct.py --no_augmentation ...
```

### Custom Augmentation Settings

**Conservative (high-quality data):**
```bash
python main_mrct.py \
    --rotation_degrees -10 10 \          # Reduce rotation range
    --zoom_range 0.95 1.05 \             # Reduce zoom range
    --enable_flip \                       # Keep flip enabled
    ... # other arguments
```

**Aggressive (small dataset with MR-only augmentations):**
```bash
python main_mrct.py \
    --rotation_degrees -20 20 \
    --zoom_range 0.85 1.15 \
    --enable_flip \
    --enable_elastic \
    --enable_grid_distortion \
    --enable_bias_field \
    --enable_gamma \
    --enable_rician_noise \
    ... # other arguments
```

## Training Scripts

The provided training scripts include augmentation by default:

### Single GPU Training
```bash
./train_example.sh
```

Edit the script to customize augmentation parameters:
```bash
# Geometric augmentations (MR+CT)
ROTATION_DEGREES="-15 15"
ENABLE_FLIP="--enable_flip"
ENABLE_ELASTIC="--enable_elastic"
ZOOM_RANGE="0.9 1.1"

# MR-only augmentations (uncomment to enable)
# ENABLE_BIAS_FIELD="--enable_bias_field"
# ENABLE_MOTION_GHOSTING="--enable_motion_ghosting"
# ENABLE_RICIAN_NOISE="--enable_rician_noise"
# ENABLE_GAMMA="--enable_gamma"
# ENABLE_CUTOUT="--enable_cutout"
```

### Distributed Training
```bash
./train_distributed.sh
```

## Programmatic Usage

```python
from augmentations_mrct import get_medical_augmentation

# Create augmentation pipeline
aug = get_medical_augmentation(
    mode='train',
    rotation_degrees=(-15, 15),
    enable_flip=True,
    enable_elastic=True,
    zoom_range=(0.9, 1.1),
    use_torchio=True
)

# Apply to paired images (numpy arrays)
mr_augmented, ct_augmented = aug(mr, ct)
```

## Technical Details

### Critical Principle: Transform Separation

**⚠️ IMPORTANT**: All geometric transforms MUST be synchronized between MR and CT, while intensity/artifact transforms are applied ONLY to MR.

- **Geometric transforms** (rotation, flip, elastic, grid, zoom): Applied to BOTH MR and CT with same random seed
- **Intensity/artifact transforms** (bias field, motion, noise, gamma, cutout): Applied ONLY to MR

This separation is crucial because:
1. Spatial alignment must be preserved for paired training
2. MR-specific artifacts should not appear in CT
3. Model learns to be robust to MRI variations while maintaining CT correspondence

### Spatial Consistency

All geometric augmentations are applied **jointly** to MR and CT pairs using the same random parameters. This ensures:
- Perfect spatial alignment is maintained
- Geometric transformations are consistent
- The paired relationship is preserved

### Transform Pipeline

1. **Basic transforms** (cropping) → Applied first to both
2. **Geometric augmentations** (rotation, elastic, zoom, flip, grid) → Applied to both MR and CT jointly
3. **MR-only augmentations** (bias field, motion, noise, gamma) → Applied only to MR
4. **Cutout** (random erasing) → Applied only to MR as final step
5. **Normalization** (z-score) → Applied after all augmentations

### Performance Considerations

- Augmentations are applied on-the-fly during training
- TorchIO operations are optimized for medical images
- Elastic and grid deformation are computationally expensive (applied with lower probability)
- MR-only transforms add minimal overhead
- All transforms use GPU when available

## Best Practices

### Recommended Settings for Different Scenarios

**Conservative (High-quality data)**:
```bash
--rotation_degrees -10 10 \
--zoom_range 0.95 1.05 \
--enable_flip
# Disable elastic if images are already well-aligned
# Enable only gamma for mild intensity variation
--enable_gamma
```

**Moderate (Default with MR-only)**:
```bash
--rotation_degrees -15 15 \
--zoom_range 0.9 1.1 \
--enable_flip \
--enable_elastic \
--enable_bias_field \
--enable_gamma
```

**Aggressive (Small dataset, all augmentations)**:
```bash
--rotation_degrees -20 20 \
--zoom_range 0.85 1.15 \
--enable_flip \
--enable_elastic \
--enable_grid_distortion \
--enable_bias_field \
--enable_motion_ghosting \
--enable_rician_noise \
--enable_gamma \
--enable_cutout
```

### Anatomical Considerations

- **Brain**: Safe to use flip for sagittal/axial views; use bias field and motion ghosting
- **Abdomen**: Use flip cautiously (organs are asymmetric); bias field is very important
- **Spine**: Rotation should be limited (±5-10 degrees); cutout can help with vertebrae variations
- **Cardiac**: Motion ghosting is particularly relevant; avoid excessive elastic deformation

### Dataset Size

- **Large dataset (>10k pairs)**: Conservative geometric + mild MR-only (gamma, bias field)
- **Medium dataset (1k-10k pairs)**: Moderate geometric + MR-only augmentations (bias field, gamma, noise)
- **Small dataset (<1k pairs)**: Aggressive geometric + all MR-only augmentations

## Validation

Check if augmentation is working:
```python
from dataset_mrct import get_mrct_dataloaders

train_loader, test_loader = get_mrct_dataloaders(
    dataset_path='/path/to/data',
    enable_augmentation=True,
    ...
)

# The loader will print augmentation status:
# "Medical augmentation enabled:"
# "  - Rotation: (-15, 15)°"
# "  - Flip: True"
# "  - Elastic: True"
# "  - Zoom: (0.9, 1.1)"
# "  - Using: TorchIO"
```

## Troubleshooting

### TorchIO Not Available

If you see:
```
Warning: TorchIO not available. Install with: pip install torchio
Using: Basic transforms
```

Install TorchIO:
```bash
pip install torchio==0.19.6
```

### Elastic Deformation Disabled

Elastic deformation requires TorchIO. If using basic transforms, it will be automatically disabled.

### Memory Issues

If augmentation causes memory issues:
1. Disable elastic deformation: `--enable_elastic=False`
2. Reduce augmentation probability (edit `augmentations_mrct.py`)
3. Reduce batch size

## References

- **TorchIO**: Pérez-García et al., "TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning", 2021
- **Medical Image Augmentation**: Shorten & Khoshgoftaar, "A survey on Image Data Augmentation for Deep Learning", 2019

## See Also

- `augmentations_mrct.py` - Augmentation implementation
- `dataset_mrct.py` - Dataset loader with augmentation support
- `train_example.sh` - Training script with augmentation
- `QUICKSTART.md` - General usage guide
