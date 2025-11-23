#!/bin/bash
# Example training script for MR-to-CT synthesis

# Configuration
DATA_PATH="/mnt/lihz/supervised-mr2ct/dataset"  # CHANGE THIS
OUTPUT_DIR="./output8_mrct_aug"
MODEL="JiT-B/8"
IMG_SIZE=256
BATCH_SIZE=16
EPOCHS=1000

# Normalization parameters (adjust based on your data)
# You should calculate these from your training data
MR_MEAN=0.1996
MR_STD=0.1619
CT_MEAN=0.238
CT_STD=0.1465

# Training hyperparameters
LEARNING_RATE=5e-5
WARMUP_EPOCHS=5
EVAL_FREQ=50

# Diffusion parameters
P_MEAN=-0.8
P_STD=0.8
NOISE_SCALE=1.0

# Sampling parameters
SAMPLING_METHOD="heun"
NUM_SAMPLING_STEPS=50

# Data augmentation parameters
ENABLE_AUGMENTATION="--enable_augmentation"  # Use --no_augmentation to disable

# Geometric augmentations (applied to both MR and CT)
ROTATION_DEGREES="-15 15"  # Min and max rotation in degrees
ENABLE_FLIP="--enable_flip"  # Random horizontal flip
ENABLE_ELASTIC="--enable_elastic"  # Elastic deformation
ZOOM_RANGE="0.9 1.1"  # Min and max zoom factor
#ENABLE_GRID_DISTORTION="--enable_grid_distortion"  # Uncomment to enable grid distortion

# MR-only augmentations (intensity/artifact transforms)
 ENABLE_BIAS_FIELD="--enable_bias_field"  # Uncomment to enable bias field simulation
 ENABLE_MOTION_GHOSTING="--enable_motion_ghosting"  # Uncomment to enable motion artifacts
 ENABLE_RICIAN_NOISE="--enable_rician_noise"  # Uncomment to enable Rician noise
 ENABLE_GAMMA="--enable_gamma"  # Uncomment to enable gamma correction
 ENABLE_CUTOUT="--enable_cutout"  # Uncomment to enable random cutout/erasing

echo "Starting MR-to-CT synthesis training..."
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL"

# Single GPU training
nohup python main_mrct.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --model $MODEL \
    --img_size $IMG_SIZE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --warmup_epochs $WARMUP_EPOCHS \
    --blr $LEARNING_RATE \
    --mr_mean $MR_MEAN --mr_std $MR_STD \
    --ct_mean $CT_MEAN --ct_std $CT_STD \
    --P_mean $P_MEAN --P_std $P_STD \
    --noise_scale $NOISE_SCALE \
    --sampling_method $SAMPLING_METHOD \
    --num_sampling_steps $NUM_SAMPLING_STEPS \
    $ENABLE_AUGMENTATION \
    --rotation_degrees $ROTATION_DEGREES \
    $ENABLE_FLIP \
    $ENABLE_ELASTIC \
    --zoom_range $ZOOM_RANGE \
    ${ENABLE_GRID_DISTORTION:-} \
    ${ENABLE_BIAS_FIELD:-} \
    ${ENABLE_MOTION_GHOSTING:-} \
    ${ENABLE_RICIAN_NOISE:-} \
    ${ENABLE_GAMMA:-} \
    ${ENABLE_CUTOUT:-} \
    --online_eval --eval_freq $EVAL_FREQ \
    --save_last_freq 50 \
    --log_freq 50 > output8_mrct_aug.log 2>&1 &

echo "Training completed!"
