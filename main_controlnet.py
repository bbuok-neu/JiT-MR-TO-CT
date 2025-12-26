"""
Main Training Script for ControlNet-based Zero-Shot MR-to-CT Synthesis

Two-stage training:
- Stage 1: Train unconditional Base Model on CT (in_channels=1)
- Stage 2: Freeze Base Model, train ControlNet with MIND guidance

Usage:
    # Stage 1: Train Base Model
    python main_controlnet.py --data_path /path/to/data --stage stage1 --output_dir ./output_stage1

    # Stage 2: Train ControlNet (load Base Model weights)
    python main_controlnet.py --data_path /path/to/data --stage stage2 \
        --base_model_path ./output_stage1/checkpoint-last.pth --output_dir ./output_stage2
"""
import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
import copy

from engine_controlnet import train_one_epoch_controlnet, evaluate_controlnet
from denoiser_controlnet import Denoiser_ControlNet
from dataset_zeroshot import get_zeroshot_dataloaders


def get_args_parser():
    parser = argparse.ArgumentParser('ControlNet for Zero-Shot MR-to-CT Synthesis', add_help=False)

    # Architecture
    parser.add_argument('--model', default='ControlNet', type=str,
                        help='Model architecture')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    
    # Training stage
    parser.add_argument('--stage', default='stage1', type=str, choices=['stage1', 'stage2'],
                        help='Training stage: stage1 (base model) or stage2 (controlnet)')
    parser.add_argument('--base_model_path', default='', type=str,
                        help='Path to Stage 1 checkpoint for loading Base Model weights (Stage 2 only)')

    # Training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=5e-5,
                        help='Base learning rate')
    parser.add_argument('--min_lr', type=float, default=0.)
    parser.add_argument('--lr_schedule', type=str, default='constant')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--ema_decay1', type=float, default=0.9999)
    parser.add_argument('--ema_decay2', type=float, default=0.9996)
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)

    # Sampling
    parser.add_argument('--sampling_method', default='heun', type=str)
    parser.add_argument('--num_sampling_steps', default=50, type=int)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_only', action='store_true')

    # Dataset
    parser.add_argument('--data_path', required=True, type=str,
                        help='Path to dataset root directory')
    parser.add_argument('--ct_mean', type=float, default=0.5)
    parser.add_argument('--ct_std', type=float, default=0.5)
    parser.add_argument('--mr_mean', type=float, default=0.5)
    parser.add_argument('--mr_std', type=float, default=0.5)

    # MIND parameters
    parser.add_argument('--mind_patch_size', type=int, default=7)
    parser.add_argument('--mind_neigh_size', type=int, default=7,
                        help='MIND neighborhood size (output channels = neigh_size^2 - 1 = 48)')
    parser.add_argument('--mind_sigma', type=float, default=0.5)
    parser.add_argument('--mind_eps', type=float, default=1e-6)
    parser.add_argument('--mind_neigh4', action='store_true',
                        help='Use 4-connectivity (4 channels) instead of full neighborhood')

    # Intensity perturbation (Stage 2 only)
    parser.add_argument('--enable_perturbation', action='store_true',
                        help='Enable intensity perturbation for MIND robustness (Stage 2)')
    parser.set_defaults(enable_perturbation=True)
    parser.add_argument('--invert_prob', type=float, default=0.3,
                        help='Probability of random invert')
    parser.add_argument('--gamma_prob', type=float, default=0.3,
                        help='Probability of random gamma')
    parser.add_argument('--gamma_range', type=float, nargs=2, default=[0.5, 2.0],
                        help='Range for gamma values')
    parser.add_argument('--solarize_prob', type=float, default=0.3,
                        help='Probability of random solarize')
    parser.add_argument('--solarize_threshold', type=float, nargs=2, default=[0.3, 0.7],
                        help='Threshold range for solarization')
    parser.add_argument('--solarize_mode_a_prob', type=float, default=0.5,
                        help='Probability of solarize Mode A (invert) vs Mode B (zero)')

    # Checkpointing
    parser.add_argument('--output_dir', default='./output_controlnet')
    parser.add_argument('--resume', default='')
    parser.add_argument('--save_last_freq', type=int, default=5)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda')

    # Distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    global_rank = misc.get_rank()

    # TensorBoard logging
    if global_rank == 0 and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    print(f"Training Stage: {args.stage}")
    
    train_loader, test_loader = get_zeroshot_dataloaders(
        dataset_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ct_mean=args.ct_mean,
        ct_std=args.ct_std,
        mr_mean=args.mr_mean,
        mr_std=args.mr_std,
        img_size=args.img_size,
        distributed=args.distributed,
        # MIND
        mind_patch_size=args.mind_patch_size,
        mind_neigh_size=args.mind_neigh_size,
        mind_sigma=args.mind_sigma,
        mind_eps=args.mind_eps,
        mind_neigh4=args.mind_neigh4,
        # Stage
        stage=args.stage,
        # Perturbation
        enable_perturbation=args.enable_perturbation,
        invert_prob=args.invert_prob,
        gamma_prob=args.gamma_prob,
        gamma_range=tuple(args.gamma_range),
        solarize_prob=args.solarize_prob,
        solarize_threshold_range=tuple(args.solarize_threshold),
        solarize_mode_a_prob=args.solarize_mode_a_prob
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    model = Denoiser_ControlNet(args)

    # Load base model weights for Stage 2
    if args.stage == 'stage2' and args.base_model_path:
        print(f"Loading Base Model weights from: {args.base_model_path}")
        model.load_base_model_weights(args.base_model_path)

    model.to(device)

    # Calculate learning rate
    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"Base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"Actual lr: {args.lr:.2e}")
    print(f"Effective batch size: {eff_batch_size}")

    # DDP wrapper
    if args.distributed:
        # Only wrap trainable parameters
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=(args.stage == 'stage2')  # ControlNet may have unused base model params
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Optimizer - only for trainable parameters
    trainable_params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    print(f"Optimizer with {len(trainable_params)} trainable parameter groups")

    # Resume or initialize EMA
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model_without_ddp.load_state_dict(checkpoint['model'])
        
        # Load EMA
        if 'model_ema1' in checkpoint:
            ema_state_dict1 = checkpoint['model_ema1']
            ema_state_dict2 = checkpoint['model_ema2']
            model_without_ddp.ema_params1 = [ema_state_dict1[name].to(device) 
                                             for name, p in model_without_ddp.named_parameters() 
                                             if p.requires_grad]
            model_without_ddp.ema_params2 = [ema_state_dict2[name].to(device) 
                                             for name, p in model_without_ddp.named_parameters() 
                                             if p.requires_grad]
        
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"Resuming from epoch {args.start_epoch}")
        
        del checkpoint
    else:
        # Initialize EMA with trainable parameters
        model_without_ddp.ema_params1 = copy.deepcopy(trainable_params)
        model_without_ddp.ema_params2 = copy.deepcopy(trainable_params)
        print("Training from scratch")

    # Evaluate only mode
    if args.evaluate_only:
        print(f"Evaluating checkpoint at epoch {args.start_epoch}")
        with torch.no_grad():
            evaluate_controlnet(model_without_ddp, test_loader, args, args.start_epoch, log_writer)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_one_epoch_controlnet(
            model, model_without_ddp, train_loader, optimizer,
            device, epoch, log_writer, args
        )

        # Save checkpoint
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if epoch % 50 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Online evaluation
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate_controlnet(model_without_ddp, test_loader, args, epoch, log_writer)
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer:
            log_writer.flush()

    total_time = time.time() - start_time
    print(f'Training time: {datetime.timedelta(seconds=int(total_time))}')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
