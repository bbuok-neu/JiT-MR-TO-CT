"""
Main Training Script for Zero-Shot MR-to-CT Synthesis

Zero-shot approach:
- Training: Use only CT images with CT-derived HOG as conditioning
- Inference: Use MR images with MR-derived HOG for conditioning
- The model learns to reconstruct CT from HOG structural information
- At inference, MR-HOG provides similar structural guidance for CT synthesis
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
from engine_zeroshot import train_one_epoch_zeroshot, evaluate_zeroshot
from denoiser_zeroshot import Denoiser_ZeroShot
from dataset_zeroshot import get_zeroshot_train_dataloader, get_zeroshot_test_dataloader


def get_args_parser():
    parser = argparse.ArgumentParser('Zero-Shot MR-to-CT Synthesis with HOG Conditioning', add_help=False)

    # Architecture
    parser.add_argument('--model', default='UNet_ZeroShot', type=str, metavar='MODEL',
                        help='Model architecture (UNet with HOG cross-attention)')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    
    # HOG parameters
    parser.add_argument('--hog_cell_size', default=8, type=int,
                        help='HOG cell size in pixels')
    parser.add_argument('--hog_block_size', default=2, type=int,
                        help='HOG block size in cells')
    parser.add_argument('--hog_num_bins', default=9, type=int,
                        help='Number of HOG orientation bins')
    
    # HOG embedding parameters
    parser.add_argument('--hog_input_dim', default=36, type=int,
                        help='Raw HOG feature dimension (default: 2*2*9=36)')
    parser.add_argument('--hog_embed_dim', default=768, type=int,
                        help='HOG embedding dimension for cross-attention (default: 768)')
    parser.add_argument('--hog_grid_size', default=31, type=int,
                        help='HOG block grid size for positional encoding (default: 31 for 256x256)')
    parser.add_argument('--hog_embed_dropout', default=0.1, type=float,
                        help='Dropout rate for HOG embedding (default: 0.1)')
    
    # Cross-attention dimension (now uses hog_embed_dim, kept for backward compatibility)
    parser.add_argument('--cross_attention_dim', default=768, type=int,
                        help='Cross-attention dimension (default: 768, should match hog_embed_dim)')

    # Training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='First EMA decay rate')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='Second EMA decay rate')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE sampling method (euler or heun)')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true',
                        help='Perform online evaluation during training')
    parser.add_argument('--evaluate_only', action='store_true',
                        help='Only evaluate, do not train')

    # Dataset
    parser.add_argument('--data_path', required=True, type=str,
                        help='Path to the dataset root directory')
    parser.add_argument('--mr_mean', type=float, default=0.5,
                        help='Mean for MR z-score normalization')
    parser.add_argument('--mr_std', type=float, default=0.5,
                        help='Std for MR z-score normalization')
    parser.add_argument('--ct_mean', type=float, default=0.5,
                        help='Mean for CT z-score normalization')
    parser.add_argument('--ct_std', type=float, default=0.5,
                        help='Std for CT z-score normalization')

    # Checkpointing
    parser.add_argument('--output_dir', default='./output_zeroshot',
                        help='Directory to save outputs')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # Distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))
    
    # Compute cross_attention_dim from HOG params if not explicitly set
    computed_cross_attn_dim = args.hog_block_size * args.hog_block_size * args.hog_num_bins
    if args.cross_attention_dim != computed_cross_attn_dim:
        print(f"Warning: cross_attention_dim ({args.cross_attention_dim}) != "
              f"computed from HOG params ({computed_cross_attn_dim}). Using computed value.")
        args.cross_attention_dim = computed_cross_attn_dim

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Load training dataset (CT only with HOG)
    print(f"Loading CT-only dataset from: {args.data_path}")
    train_loader = get_zeroshot_train_dataloader(
        dataset_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ct_mean=args.ct_mean,
        ct_std=args.ct_std,
        img_size=args.img_size,
        distributed=args.distributed,
        enable_augmentation=False,  # No augmentation for now
        hog_cell_size=args.hog_cell_size,
        hog_block_size=args.hog_block_size,
        hog_num_bins=args.hog_num_bins
    )
    
    # Load test dataset (MR with HOG for zero-shot inference)
    print(f"Loading MR dataset for zero-shot inference from: {args.data_path}")
    test_loader = get_zeroshot_test_dataloader(
        dataset_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mr_mean=args.mr_mean,
        mr_std=args.mr_std,
        ct_mean=args.ct_mean,
        ct_std=args.ct_std,
        img_size=args.img_size,
        distributed=args.distributed,
        hog_cell_size=args.hog_cell_size,
        hog_block_size=args.hog_block_size,
        hog_num_bins=args.hog_num_bins
    )
    
    print(f"Training batches (CT only): {len(train_loader)}")
    print(f"Test batches (MR for zero-shot): {len(test_loader)}")

    # Configure torch compilation settings
    try:
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.optimize_ddp = False
    except AttributeError:
        pass

    # Create model
    model = Denoiser_ZeroShot(args)

    print("Model = Zero-Shot MR-to-CT with HOG conditioning")
    print(f"  Cross-attention dim: {args.cross_attention_dim}")
    print(f"  HOG params: cell_size={args.hog_cell_size}, block_size={args.hog_block_size}, num_bins={args.hog_num_bins}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Set up optimizer
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].to(device) for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].to(device) for name, _ in model_without_ddp.named_parameters()]
        print(f"✓ Loaded model weights from checkpoint")

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"✓ Loaded optimizer state")
            print(f"✓ Resuming training from epoch {args.start_epoch}")
        else:
            print("⚠ Warning: Checkpoint does not contain optimizer state or epoch information")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Evaluate only mode
    if args.evaluate_only:
        print("Evaluating checkpoint at epoch {}".format(args.start_epoch))
        with torch.no_grad():
            evaluate_zeroshot(model_without_ddp, test_loader, args, args.start_epoch, log_writer=log_writer)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    print(f"Training will run from epoch {args.start_epoch} to {args.epochs - 1}")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_one_epoch_zeroshot(model, model_without_ddp, train_loader, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically
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

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate_zeroshot(model_without_ddp, test_loader, args, epoch, log_writer=log_writer)
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
