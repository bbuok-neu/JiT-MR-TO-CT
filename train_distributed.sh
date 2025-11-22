#!/bin/bash
# Example script for multi-GPU distributed training

# Configuration
DATA_PATH="/path/to/your/dataset"  # CHANGE THIS
OUTPUT_DIR="./output_mrct"
MODEL="JiT-L/16"  # Larger model for better quality
IMG_SIZE=256
BATCH_SIZE=8  # Per GPU
EPOCHS=200
NUM_GPUS=4

# Normalization parameters
MR_MEAN=0.5
MR_STD=0.5
CT_MEAN=0.5
CT_STD=0.5

echo "Starting distributed MR-to-CT synthesis training on $NUM_GPUS GPUs..."

torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 \
    main_mrct.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --model $MODEL \
    --img_size $IMG_SIZE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --warmup_epochs 5 \
    --blr 5e-5 \
    --mr_mean $MR_MEAN --mr_std $MR_STD \
    --ct_mean $CT_MEAN --ct_std $CT_STD \
    --P_mean -0.8 --P_std 0.8 \
    --noise_scale 1.0 \
    --sampling_method heun \
    --num_sampling_steps 50 \
    --online_eval --eval_freq 10 \
    --save_last_freq 5 \
    --log_freq 50

echo "Training completed!"
