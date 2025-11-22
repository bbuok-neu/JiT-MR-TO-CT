# Quick Start Guide for MR-to-CT Synthesis

This guide will help you get started with MR-to-CT medical image synthesis using JiT.

## Prerequisites

- NVIDIA GPU with CUDA support
- Conda or Miniconda installed
- Python 3.10+
- PyTorch 2.5.1+

## Step 1: Installation

```bash
# Clone the repository (if not already cloned)
git clone https://github.com/bbuok-neu/JiT-MR-TO-CT.git
cd JiT-MR-TO-CT

# Create and activate conda environment
conda env create -f environment.yaml
conda activate jit

# Verify installation
python validate_implementation.py
```

## Step 2: Prepare Your Dataset

Organize your MR-CT paired images in this structure:

```
your_dataset/
  ├── mr/
  │   ├── train/
  │   │   ├── 00001.jpg
  │   │   ├── 00002.jpg
  │   │   └── ...
  │   └── test/
  │       ├── 00001.jpg
  │       └── ...
  └── ct/
      ├── train/
      │   ├── 00001.jpg
      │   ├── 00002.jpg
      │   └── ...
      └── test/
          ├── 00001.jpg
          └── ...
```

**Important:**
- Images must be in JPG format
- MR and CT images must have matching filenames
- All images are loaded as grayscale (single channel)
- Images should be 2D slices

## Step 3: Calculate Normalization Parameters

```bash
python calculate_stats.py /path/to/your_dataset
```

This will output:
```
--mr_mean 0.485123 --mr_std 0.229456
--ct_mean 0.512345 --ct_std 0.198765
```

Save these values for training!

## Step 4: Train the Model

### Option A: Single GPU Training

```bash
python main_mrct.py \
    --data_path /path/to/your_dataset \
    --mr_mean 0.485123 --mr_std 0.229456 \
    --ct_mean 0.512345 --ct_std 0.198765 \
    --model JiT-B/16 \
    --img_size 256 \
    --batch_size 16 \
    --epochs 200 \
    --online_eval --eval_freq 10 \
    --output_dir ./output_mrct
```

### Option B: Multi-GPU Training (Recommended)

```bash
torchrun --nproc_per_node=4 main_mrct.py \
    --data_path /path/to/your_dataset \
    --mr_mean 0.485123 --mr_std 0.229456 \
    --ct_mean 0.512345 --ct_std 0.198765 \
    --model JiT-L/16 \
    --img_size 256 \
    --batch_size 8 \
    --epochs 200 \
    --online_eval --eval_freq 10 \
    --output_dir ./output_mrct
```

## Step 5: Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir ./output_mrct
```

Open your browser to `http://localhost:6006` to see:
- Training loss curves
- PSNR and SSIM metrics
- Learning rate schedule

## Step 6: Evaluate the Model

After training, evaluate on the test set:

```bash
python main_mrct.py \
    --data_path /path/to/your_dataset \
    --mr_mean 0.485123 --mr_std 0.229456 \
    --ct_mean 0.512345 --ct_std 0.198765 \
    --model JiT-L/16 \
    --img_size 256 \
    --output_dir ./output_mrct \
    --resume ./output_mrct \
    --evaluate_only
```

Generated images will be saved in `./output_mrct/eval_images/`

## Model Architecture Options

Choose based on your needs and GPU memory:

| Model | Parameters | GPU Memory | Quality | Training Time |
|-------|-----------|------------|---------|---------------|
| JiT-B/16 | ~86M | ~12GB | Good | Fast |
| JiT-B/32 | ~86M | ~6GB | Good | Fast |
| JiT-L/16 | ~300M | ~24GB | Better | Medium |
| JiT-H/16 | ~600M | ~40GB | Best | Slow |

For 256x256 images with batch size 16.

## Common Parameters Explained

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `--data_path` | Path to dataset | `/path/to/dataset` |
| `--model` | Model architecture | `JiT-B/16`, `JiT-L/16` |
| `--img_size` | Image size (square) | 256, 512 |
| `--batch_size` | Samples per GPU | 8-32 |
| `--epochs` | Training epochs | 100-400 |
| `--blr` | Base learning rate | 5e-5 (default) |
| `--mr_mean/std` | MR normalization | From calculate_stats.py |
| `--ct_mean/std` | CT normalization | From calculate_stats.py |
| `--num_sampling_steps` | ODE steps | 50-100 |
| `--online_eval` | Eval during training | Flag |
| `--eval_freq` | Eval frequency | 10-20 epochs |

## Expected Results

With proper training:
- **PSNR**: 25-35 dB (higher is better)
- **SSIM**: 0.85-0.95 (1.0 is perfect)

Results vary based on:
- Dataset quality
- Model size
- Training duration
- Image resolution

## Tips for Success

### 1. Data Quality
- Ensure MR-CT pairs are well-aligned
- Remove corrupt or misaligned images
- Consistent image quality across dataset

### 2. Training
- Start with smaller model (JiT-B/16) to verify setup
- Use smaller image size (256) initially
- Monitor loss and metrics regularly
- Train for at least 100 epochs

### 3. Inference
- Use more sampling steps (100) for best quality
- Experiment with different models
- Save intermediate results

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 8

# Or use smaller model
--model JiT-B/32

# Or reduce image size
--img_size 128
```

### Slow Training
```bash
# Use multiple GPUs
torchrun --nproc_per_node=4 main_mrct.py ...

# Reduce evaluation frequency
--eval_freq 20
```

### Poor Quality
- Check data alignment and quality
- Verify normalization parameters are correct
- Train longer (more epochs)
- Use larger model
- Increase image resolution

## Next Steps

1. **Experiment with parameters**: Try different models, learning rates, image sizes
2. **Data augmentation**: The pipeline includes random flips, could add more
3. **Fine-tuning**: Resume from checkpoint and train with different settings
4. **Inference optimization**: Reduce sampling steps for faster generation

## Testing Your Setup

Before training on real data, test with dummy data:

```bash
python test_mrct.py
```

This creates a small dummy dataset and tests all components.

## Getting Help

- Check `README_MRCT.md` for detailed documentation
- Review training logs in `output_dir/`
- Monitor TensorBoard for metrics
- Verify dataset structure matches requirements

## Citation

If you use this implementation, please cite:

```bibtex
@article{li2025jit,
  title={Back to Basics: Let Denoising Generative Models Denoise},
  author={Li, Tianhong and He, Kaiming},
  journal={arXiv preprint arXiv:2511.13720},
  year={2025}
}
```
