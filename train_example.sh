#!/bin/bash
# Example training script for MR-to-CT synthesis

# Configuration
DATA_PATH="/path/to/your/dataset"  # CHANGE THIS
OUTPUT_DIR="./output_mrct"
MODEL="JiT-B/16"
IMG_SIZE=256
BATCH_SIZE=16
EPOCHS=200

# Normalization parameters (adjust based on your data)
# You should calculate these from your training data
MR_MEAN=0.5
MR_STD=0.5
CT_MEAN=0.5
CT_STD=0.5

# Training hyperparameters
LEARNING_RATE=5e-5
WARMUP_EPOCHS=5
EVAL_FREQ=10

# Diffusion parameters
P_MEAN=-0.8
P_STD=0.8
NOISE_SCALE=1.0

# Sampling parameters
SAMPLING_METHOD="heun"
NUM_SAMPLING_STEPS=50

echo "Starting MR-to-CT synthesis training..."
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL"

# Single GPU training
python main_mrct.py \
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
    --online_eval --eval_freq $EVAL_FREQ \
    --save_last_freq 5 \
    --log_freq 50

echo "Training completed!"
