"""
Main Training Script for MR-to-CT Synthesis using JiT
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
from engine_mrct import train_one_epoch, evaluate
from denoiser_mrct import Denoiser_MRCT
from dataset_mrct import get_mrct_dataloaders


def get_args_parser():
    parser = argparse.ArgumentParser('JiT for MR-to-CT Synthesis', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train (JiT-B/16, JiT-B/32, JiT-L/16, etc.)')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')
    parser.add_argument('--use_pretrained', action='store_true',
                        help='Use ImageNet pretrained weights for initialization')
    parser.add_argument('--pretrained_path', type=str, default='',
                        help='Path to pretrained checkpoint to load when using pretrained weights')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method (euler or heun)')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true',
                        help='Perform online evaluation during training')
    parser.add_argument('--evaluate_only', action='store_true',
                        help='Only evaluate, do not train')

    # dataset
    parser.add_argument('--data_path', required=True, type=str,
                        help='Path to the MR-CT dataset root directory')
    parser.add_argument('--mr_mean', type=float, default=0.5,
                        help='Mean for MR z-score normalization')
    parser.add_argument('--mr_std', type=float, default=0.5,
                        help='Std for MR z-score normalization')
    parser.add_argument('--ct_mean', type=float, default=0.5,
                        help='Mean for CT z-score normalization')
    parser.add_argument('--ct_std', type=float, default=0.5,
                        help='Std for CT z-score normalization')

    # data augmentation
    parser.add_argument('--enable_augmentation', action='store_true',
                        help='Enable medical image augmentation for training')
    parser.add_argument('--no_augmentation', action='store_false', dest='enable_augmentation',
                        help='Disable medical image augmentation')
    parser.set_defaults(enable_augmentation=True)
    parser.add_argument('--use_torchio', action='store_true',
                        help='Use TorchIO for augmentation (if available)')
    parser.set_defaults(use_torchio=True)
    
    # Geometric augmentations (applied to both MR and CT)
    parser.add_argument('--rotation_degrees', type=float, nargs=2, default=[-15, 15],
                        help='Range for random rotation in degrees (min max)')
    parser.add_argument('--enable_flip', action='store_true',
                        help='Enable random horizontal flip')
    parser.set_defaults(enable_flip=True)
    parser.add_argument('--enable_elastic', action='store_true',
                        help='Enable elastic deformation')
    parser.set_defaults(enable_elastic=True)
    parser.add_argument('--zoom_range', type=float, nargs=2, default=[0.9, 1.1],
                        help='Range for random zoom/scaling (min max)')
    parser.add_argument('--enable_grid_distortion', action='store_true',
                        help='Enable grid distortion (alternative to elastic)')
    
    # MR-only augmentations (intensity/artifact transforms)
    parser.add_argument('--enable_bias_field', action='store_true',
                        help='Enable bias field simulation (MR only)')
    parser.add_argument('--enable_motion_ghosting', action='store_true',
                        help='Enable motion ghosting simulation (MR only)')
    parser.add_argument('--enable_rician_noise', action='store_true',
                        help='Enable Rician noise injection (MR only)')
    parser.add_argument('--enable_gamma', action='store_true',
                        help='Enable gamma correction (MR only)')
    parser.add_argument('--enable_cutout', action='store_true',
                        help='Enable random cutout/erasing (MR only)')

    # checkpointing
    parser.add_argument('--output_dir', default='./output_mrct',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
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

    # Load MR-CT dataset
    print(f"Loading MR-CT dataset from: {args.data_path}")
    train_loader, test_loader = get_mrct_dataloaders(
        dataset_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mr_mean=args.mr_mean,
        mr_std=args.mr_std,
        ct_mean=args.ct_mean,
        ct_std=args.ct_std,
        img_size=args.img_size,
        distributed=args.distributed,
        enable_augmentation=args.enable_augmentation,
        use_torchio=args.use_torchio,
        # Geometric augmentations
        rotation_degrees=tuple(args.rotation_degrees),
        enable_flip=args.enable_flip,
        enable_elastic=args.enable_elastic,
        zoom_range=tuple(args.zoom_range),
        enable_grid_distortion=args.enable_grid_distortion,
        # MR-only augmentations
        enable_bias_field=args.enable_bias_field,
        enable_motion_ghosting=args.enable_motion_ghosting,
        enable_rician_noise=args.enable_rician_noise,
        enable_gamma=args.enable_gamma,
        enable_cutout=args.enable_cutout
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Configure torch compilation settings (may not be available in all versions)
    try:
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.optimize_ddp = False
    except AttributeError:
        # torch._dynamo not available in this PyTorch version
        pass

    # Create denoiser
    model = Denoiser_MRCT(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Set up optimizer with weight decay adjustment for bias and norm layers
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
        # Device-agnostic EMA parameter loading
        model_without_ddp.ema_params1 = [ema_state_dict1[name].to(device) for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].to(device) for name, _ in model_without_ddp.named_parameters()]
        print(f"✓ Loaded model weights from checkpoint")

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"✓ Loaded optimizer state")
            print(f"✓ Resuming training from epoch {args.start_epoch} (checkpoint was at epoch {checkpoint['epoch']})")
        else:
            print("⚠ Warning: Checkpoint does not contain optimizer state or epoch information")
            print("⚠ Training will start from epoch 0 with reinitialized optimizer")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Evaluate only mode
    if args.evaluate_only:
        print("Evaluating checkpoint at epoch {}".format(args.start_epoch))
        with torch.no_grad():
            evaluate(model_without_ddp, test_loader, args, args.start_epoch, log_writer=log_writer)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    print(f"Training will run from epoch {args.start_epoch} to {args.epochs - 1}")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, train_loader, optimizer, device, epoch, log_writer=log_writer, args=args)

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
                evaluate(model_without_ddp, test_loader, args, epoch, log_writer=log_writer)
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
