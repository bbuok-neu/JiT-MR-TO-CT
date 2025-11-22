# Implementation Summary: MR-to-CT Medical Image Synthesis

## Overview

This implementation adapts the JiT (Just image Transformer) model for medical image synthesis, specifically converting MR (Magnetic Resonance) images to CT (Computed Tomography) images using paired training data.

## Key Modifications from Original JiT

### 1. Input Architecture Change

**Original JiT:**
- Input: 3-channel RGB images (clean or noisy)
- Conditioning: Class labels (1000 ImageNet classes)
- Formula: `v_θ = (x_θ - z_t) / (1 - t)`

**Modified for MR-to-CT:**
- Input: 2-channel concatenation `[z_t, MR]`
  - Channel 1: `z_t` (noisy CT at timestep t)
  - Channel 2: `MR` (conditioning MR image)
- Conditioning: Timestep only (no class labels)
- Formula: `v_θ = (CT_θ - z_t) / (1 - t)` where input is `concat(z_t, MR)`

### 2. Data Pipeline Changes

**Original:**
- ImageNet dataset with 1000 classes
- RGB images normalized to [-1, 1]
- Class-conditional generation

**Modified:**
- Paired MR-CT dataset
- Single-channel (grayscale) images
- Z-score normalization with preset mean/std
- Conditional generation based on input MR

### 3. Training Objective

**Training Process:**

1. Sample a paired (MR, CT) from dataset
2. Sample timestep: `t ~ logit-normal(μ=-0.8, σ=0.8)`
3. Create noisy CT: `z_t = t * CT + (1-t) * noise`
4. Concatenate: `input = concat(z_t, MR)`
5. Predict CT: `CT_pred = Model(input, t)`
6. Compute velocities:
   - True: `v = (CT - z_t) / (1 - t)`
   - Predicted: `v_pred = (CT_pred - z_t) / (1 - t)`
7. Loss: `L2(v, v_pred)`

### 4. Inference/Sampling

**Sampling Process:**

1. Start with noise: `z_0 ~ N(0, σ²)`
2. Given MR image, iterate from t=0 to t=1:
   - Concatenate: `input = concat(z_t, MR)`
   - Predict: `CT_pred = Model(input, t)`
   - Velocity: `v = (CT_pred - z_t) / (1 - t)`
   - Update: `z_{t+Δt} = z_t + Δt * v`
3. Output: Final CT image

Uses Heun or Euler ODE solver for integration.

## File Structure and Components

### Core Implementation Files

1. **dataset_mrct.py** (New)
   - `PairedMRCTDataset`: Loads paired MR-CT images
   - `get_mrct_dataloaders`: Creates train/test dataloaders
   - Features:
     - Single-channel grayscale loading
     - Z-score normalization
     - Paired data validation
     - Support for distributed training

2. **model_mrct.py** (New)
   - `JiT_MRCT`: Modified transformer model
   - Changes from original:
     - `in_channels=2` (for concatenated input)
     - `out_channels=1` (single-channel CT output)
     - No class embedding (time-only conditioning)
   - Model variants: B/16, B/32, L/16, L/32, H/16, H/32

3. **denoiser_mrct.py** (New)
   - `Denoiser_MRCT`: Wrapper for training and sampling
   - Key methods:
     - `forward()`: Training with concatenated input
     - `generate()`: Inference from MR to CT
     - `update_ema()`: Exponential moving average
   - Implements velocity-based diffusion

4. **engine_mrct.py** (New)
   - `train_one_epoch()`: Training loop for paired data
   - `evaluate()`: Evaluation with SSIM/PSNR metrics
   - Features:
     - Batch processing
     - Metric tracking
     - Image saving
     - EMA model switching

5. **metrics.py** (New)
   - `calculate_psnr()`: Peak Signal-to-Noise Ratio
   - `calculate_ssim()`: Structural Similarity Index
   - `evaluate_metrics()`: Combined evaluation
   - `MetricTracker`: Accumulate metrics over batches

6. **main_mrct.py** (New)
   - Main training script
   - Arguments:
     - Dataset paths
     - Model configuration
     - Normalization parameters
     - Training hyperparameters
   - Supports distributed training

### Utility and Helper Files

7. **calculate_stats.py** (New)
   - Calculate mean/std from training data
   - Essential for proper normalization

8. **test_mrct.py** (New)
   - Unit tests for all components
   - Creates dummy data for testing
   - Validates implementation

9. **validate_implementation.py** (New)
   - Syntax and structure validation
   - Checks all components are present
   - No dependencies required

10. **train_example.sh** (New)
    - Single-GPU training example
    - Documented parameters

11. **train_distributed.sh** (New)
    - Multi-GPU training example
    - Uses torchrun

12. **README_MRCT.md** (New)
    - Comprehensive documentation
    - Architecture details
    - Usage examples

13. **QUICKSTART.md** (New)
    - Step-by-step guide
    - Common troubleshooting

### Original Files (Preserved)

- `main_jit.py`: Original JiT training
- `model_jit.py`: Original JiT model
- `denoiser.py`: Original denoiser
- `engine_jit.py`: Original training engine
- `util/*`: Utility functions (used by both)

## Technical Details

### Model Architecture

```
Input: [z_t, MR] (B, 2, H, W)
  ↓
Patch Embedding (2-channel input)
  ↓
Add Positional Embeddings
  ↓
Transformer Blocks (with time conditioning)
  - Self-attention with RoPE
  - SwiGLU FFN
  - AdaLN modulation
  ↓
Final Layer
  ↓
Output: CT (B, 1, H, W)
```

### Normalization

**Z-score normalization:**
- `x_norm = (x - mean) / std`
- Applied to both MR and CT separately
- Parameters calculated from training data
- Essential for proper training

### Diffusion Process

**Velocity Formulation:**
- Instead of noise prediction, predicts velocity
- `v = dx/dt` where x is the data
- More stable for high-resolution images
- ODE: `dx/dt = v(x, t)`

**Timestep Distribution:**
- Logit-normal distribution: `t ~ sigmoid(N(μ, σ²))`
- Default: μ=-0.8, σ=0.8
- Focuses sampling on mid-range timesteps

### Training Details

**Optimizer:**
- AdamW with β=(0.9, 0.95)
- Base learning rate: 5e-5
- Scaled by batch size: `lr = blr * batch_size / 256`
- Constant schedule (no decay)

**EMA:**
- Two EMA models tracked
- Decay rates: 0.9999 and 0.9996
- EMA1 used for inference
- Improves generation quality

**Augmentation:**
- Center crop to target size
- Random horizontal flip
- No color augmentation (grayscale)

## Evaluation Metrics

### PSNR (Peak Signal-to-Noise Ratio)
- Range: 0 to ∞ dB (higher is better)
- Formula: `20 * log10(MAX) - 10 * log10(MSE)`
- Typical values: 25-35 dB

### SSIM (Structural Similarity Index)
- Range: 0 to 1 (higher is better)
- Considers structure, luminance, contrast
- More aligned with human perception
- Typical values: 0.85-0.95

## Usage Examples

### Basic Training
```bash
python main_mrct.py \
    --data_path /data/mrct \
    --model JiT-B/16 \
    --img_size 256 \
    --batch_size 16 \
    --epochs 200 \
    --mr_mean 0.5 --mr_std 0.5 \
    --ct_mean 0.5 --ct_std 0.5 \
    --online_eval --eval_freq 10 \
    --output_dir ./output
```

### Evaluation Only
```bash
python main_mrct.py \
    --data_path /data/mrct \
    --model JiT-B/16 \
    --img_size 256 \
    --mr_mean 0.5 --mr_std 0.5 \
    --ct_mean 0.5 --ct_std 0.5 \
    --resume ./output \
    --evaluate_only
```

## Advantages of This Approach

1. **Paired Supervision**: Uses ground truth CT for training
2. **Conditional Generation**: MR guides the generation process
3. **High Quality**: Transformer architecture captures fine details
4. **Flexible**: Works with different image sizes and resolutions
5. **Scalable**: Supports distributed training on multiple GPUs
6. **Medical Metrics**: Uses domain-appropriate evaluation (SSIM/PSNR)

## Limitations and Future Work

### Current Limitations
1. Requires paired MR-CT data (hard to obtain)
2. 2D slice-based (doesn't use 3D context)
3. Single-channel only (no multi-contrast MR)
4. Fixed normalization (must calculate beforehand)

### Potential Improvements
1. 3D volume processing
2. Multi-contrast MR input
3. Unpaired training (CycleGAN-style)
4. Adaptive normalization
5. Perceptual loss functions
6. Anatomical region-specific models

## Conclusion

This implementation successfully adapts JiT for medical image synthesis by:
- Modifying the input to accept conditioning images
- Implementing paired dataset loading with proper normalization
- Adapting the training objective for supervised learning
- Using medical-relevant evaluation metrics

The code is production-ready, well-documented, and tested.
