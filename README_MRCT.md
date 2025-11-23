# MR-to-CT Medical Image Synthesis using JiT

This implementation adapts the JiT (Just image Transformer) model for medical image synthesis, specifically for paired MR-to-CT translation.

## Overview

The key modifications for MR-to-CT synthesis:

1. **Conditional Input**: Model takes concatenated `[zt, MR]` as 2-channel input
2. **Single-Channel Output**: Generates single-channel CT images
3. **Velocity Prediction**: Modified formula `vθ = (CTθ - zt)/(1-t)` where input is `concat(zt, MR)`
4. **Z-Score Normalization**: Both MR and CT images are normalized using preset mean/std
5. **Paired Training**: Uses paired MR-CT data for supervised learning
6. **Medical Metrics**: Evaluates using SSIM and PSNR instead of FID/IS
7. **🆕 Medical Augmentation**: Optional augmentation with rotation, flip, elastic deformation, and zoom (see [AUGMENTATION.md](AUGMENTATION.md))

## Dataset Structure

Organize your dataset in the following structure:

```
dataset/
  ├── mr/
  │   ├── train/
  │   │   ├── 00001.jpg
  │   │   ├── 00002.jpg
  │   │   └── ...
  │   └── test/
  │       ├── 00001.jpg
  │       ├── 00002.jpg
  │       └── ...
  └── ct/
      ├── train/
      │   ├── 00001.jpg
      │   ├── 00002.jpg
      │   └── ...
      └── test/
          ├── 00001.jpg
          ├── 00002.jpg
          └── ...
```

**Important Notes:**
- All images must be in JPG/JPEG format (PNG also supported for preprocessing tools)
- MR and CT images must be paired (same filename in corresponding folders)
- Images are loaded as single-channel (grayscale)
- Images can be 2D slices from 3D volumes

## Installation

Use the existing conda environment:

```bash
conda env create -f environment.yaml
conda activate jit
```

The environment includes TorchIO for medical image augmentation. If not installed:
```bash
pip install torchio==0.19.6
```

## Training

### Basic Training Command

```bash
python main_mrct.py \
    --data_path /path/to/dataset \
    --model JiT-B/16 \
    --img_size 256 \
    --batch_size 16 \
    --epochs 200 \
    --warmup_epochs 5 \
    --blr 5e-5 \
    --mr_mean 0.5 --mr_std 0.5 \
    --ct_mean 0.5 --ct_std 0.5 \
    --output_dir ./output_mrct \
    --online_eval --eval_freq 10
```

### Training with Data Augmentation (Recommended)

```bash
python main_mrct.py \
    --data_path /path/to/dataset \
    --model JiT-B/16 \
    --img_size 256 \
    --batch_size 16 \
    --epochs 200 \
    --enable_augmentation \
    --rotation_degrees -15 15 \
    --enable_flip \
    --enable_elastic \
    --zoom_range 0.9 1.1 \
    --mr_mean 0.5 --mr_std 0.5 \
    --ct_mean 0.5 --ct_std 0.5 \
    --output_dir ./output_mrct \
    --online_eval --eval_freq 10
```

See [AUGMENTATION.md](AUGMENTATION.md) for detailed augmentation documentation.

### Distributed Training (Multi-GPU)

```bash
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    main_mrct.py \
    --data_path /path/to/dataset \
    --model JiT-B/16 \
    --img_size 256 \
    --batch_size 16 \
    --epochs 200 \
    --mr_mean 0.5 --mr_std 0.5 \
    --ct_mean 0.5 --ct_std 0.5 \
    --output_dir ./output_mrct \
    --online_eval --eval_freq 10
```

### Key Parameters

- `--data_path`: Path to dataset root directory
- `--model`: Model architecture (JiT-B/16, JiT-B/32, JiT-L/16, JiT-H/16)
- `--img_size`: Image size (256 or 512)
- `--batch_size`: Batch size per GPU
- `--mr_mean`, `--mr_std`: Z-score normalization parameters for MR images
- `--ct_mean`, `--ct_std`: Z-score normalization parameters for CT images
- `--online_eval`: Enable evaluation during training
- `--eval_freq`: Evaluation frequency in epochs

### Determining Normalization Parameters

Calculate mean and std from your training data:

```python
import numpy as np
from PIL import Image
import os

def calculate_stats(image_folder):
    values = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img = Image.open(os.path.join(image_folder, filename)).convert('L')
            values.append(np.array(img, dtype=np.float32) / 255.0)
    
    values = np.concatenate([v.flatten() for v in values])
    mean = np.mean(values)
    std = np.std(values)
    return mean, std

# Calculate for MR and CT separately
mr_mean, mr_std = calculate_stats('/path/to/dataset/mr/train')
ct_mean, ct_std = calculate_stats('/path/to/dataset/ct/train')

print(f"MR - Mean: {mr_mean:.4f}, Std: {mr_std:.4f}")
print(f"CT - Mean: {ct_mean:.4f}, Std: {ct_std:.4f}")
```

## Evaluation

### Evaluate a Trained Model

```bash
python main_mrct.py \
    --data_path /path/to/dataset \
    --model JiT-B/16 \
    --img_size 256 \
    --batch_size 16 \
    --mr_mean 0.5 --mr_std 0.5 \
    --ct_mean 0.5 --ct_std 0.5 \
    --output_dir ./output_mrct \
    --resume ./output_mrct \
    --evaluate_only
```

This will:
- Load the checkpoint from `--resume` directory
- Evaluate on the test set
- Calculate SSIM and PSNR metrics
- Save generated CT images in `output_dir/eval_images/`

## Model Architecture Details

### Input Processing

1. **Input**: Concatenated `[zt, MR]` (2 channels)
   - `zt`: Noisy CT at timestep t
   - `MR`: Conditioning MR image
   
2. **Patch Embedding**: 2-channel input → patches → embeddings

3. **Transformer Blocks**: Process embeddings with time conditioning

4. **Output**: Single-channel CT image

### Training Objective

At each training step:
1. Sample timestep: `t ~ logit-normal(μ=-0.8, σ=0.8)`
2. Add noise: `zt = t * CT + (1-t) * noise`
3. True velocity: `v = (CT - zt) / (1-t)`
4. Model input: `concat(zt, MR)`
5. Predict CT: `CT_pred = Model(concat(zt, MR), t)`
6. Predicted velocity: `v_pred = (CT_pred - zt) / (1-t)`
7. Loss: `L2(v, v_pred)`

### Sampling/Inference

During inference:
1. Start with noise: `z0 ~ N(0, σ)`
2. For t from 0 to 1:
   - Input: `concat(zt, MR)`
   - Predict: `CT_pred = Model(concat(zt, MR), t)`
   - Velocity: `v = (CT_pred - zt) / (1-t)`
   - Update: `z_{t+dt} = zt + dt * v`
3. Output final CT

## File Structure

```
JiT-MR-TO-CT/
├── main_mrct.py           # Main training script
├── model_mrct.py          # Modified JiT model for MR-CT
├── denoiser_mrct.py       # Denoiser wrapper for training
├── dataset_mrct.py        # Paired MR-CT dataset loader
├── engine_mrct.py         # Training and evaluation engine
├── metrics.py             # SSIM and PSNR metrics
├── util/                  # Utility functions (from original JiT)
├── README_MRCT.md         # This file
└── ...
```

## Expected Results

The model should achieve:
- PSNR: Typically 25-35 dB (higher is better)
- SSIM: Typically 0.85-0.95 (1.0 is perfect)

Results depend on:
- Dataset quality and size
- Model architecture (larger models = better results)
- Training duration
- Normalization parameters

## Tips for Best Results

1. **Data Quality**: Ensure MR-CT pairs are well-aligned
2. **Normalization**: Use appropriate mean/std from your data
3. **Image Size**: Start with 256x256, increase if GPU memory allows
4. **Training Time**: Train for at least 100-200 epochs
5. **Model Size**: Use JiT-L/16 or JiT-H/16 for best quality
6. **Sampling Steps**: Use 50-100 steps during inference for quality

## Troubleshooting

### Out of Memory
- Reduce `--batch_size`
- Use smaller model (JiT-B/16 instead of JiT-L/16)
- Reduce `--img_size`

### Poor Quality Results
- Increase training epochs
- Check normalization parameters
- Increase model size
- Verify data alignment

### Slow Training
- Enable mixed precision (already enabled with bfloat16)
- Use multiple GPUs with distributed training
- Increase batch size

## Citation

If you use this code, please cite the original JiT paper:

```bibtex
@article{li2025jit,
  title={Back to Basics: Let Denoising Generative Models Denoise},
  author={Li, Tianhong and He, Kaiming},
  journal={arXiv preprint arXiv:2511.13720},
  year={2025}
}
```

## License

This code inherits the license from the original JiT repository.
