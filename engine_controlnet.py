"""
Training and Evaluation Engine for ControlNet-based Zero-Shot MR-to-CT Synthesis

Two-stage training:
- Stage 1: Train unconditional Base Model on CT only
- Stage 2: Freeze Base Model, train ControlNet with MIND guidance
"""
import math
import sys
import os

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import copy
from metrics import evaluate_metrics, MetricTracker


def train_one_epoch_controlnet(model, model_without_ddp, data_loader, optimizer, 
                                device, epoch, log_writer=None, args=None):
    """
    Train one epoch for ControlNet-based synthesis.
    
    Stage 1: Dataloader returns (ct, None) - unconditional training
    Stage 2: Dataloader returns (ct, mind_features) - conditional training
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}] Stage: {args.stage}'
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (ct, mind_features) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Per iteration lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Move to device
        ct = ct.to(device, non_blocking=True)
        if mind_features is not None:
            mind_features = mind_features.to(device, non_blocking=True)

        # Device-agnostic autocast
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            loss = model(ct, mind_features)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Device-agnostic synchronization
        if device.type == 'cuda':
            torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


@torch.no_grad()
def evaluate_controlnet(model_without_ddp, test_loader, args, epoch, log_writer=None, save_images=True):
    """
    Evaluate zero-shot MR-to-CT synthesis on test set.
    
    Test dataloader returns (ct, mind_features) where:
    - mind_features: MIND(MR) for zero-shot inference
    - ct: Ground truth CT
    """
    model_without_ddp.eval()
    local_rank = misc.get_rank()
    
    # Get trainable parameters for EMA
    trainable_params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    
    # Switch to EMA params
    original_params = [p.data.clone() for p in trainable_params]
    
    if model_without_ddp.ema_params1 is not None:
        for param, ema_param in zip(trainable_params, model_without_ddp.ema_params1):
            param.data.copy_(ema_param.data)
        print("Switched to EMA for evaluation")
    
    # Metric tracker
    metric_tracker = MetricTracker()
    
    # Create output directory
    save_folder = None
    if save_images and local_rank == 0:
        save_folder = os.path.join(args.output_dir, "eval_images", f"epoch_{epoch}")
        os.makedirs(save_folder, exist_ok=True)
        print(f"Saving evaluation images to: {save_folder}")
    
    img_idx = 0
    device_type = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    
    for batch_idx, (ct_true, mind_features) in enumerate(test_loader):
        ct_true = ct_true.to(args.device, non_blocking=True)
        if mind_features is not None:
            mind_features = mind_features.to(args.device, non_blocking=True)
        
        # Generate CT from MIND(MR) features
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            ct_pred = model_without_ddp.generate(mind_features)
        
        # Denormalize
        ct_pred_denorm = ct_pred * args.ct_std + args.ct_mean
        ct_true_denorm = ct_true * args.ct_std + args.ct_mean
        
        # Clamp to [0, 1]
        ct_pred_denorm = torch.clamp(ct_pred_denorm, 0.0, 1.0)
        ct_true_denorm = torch.clamp(ct_true_denorm, 0.0, 1.0)
        
        # Calculate metrics
        metrics = evaluate_metrics(ct_pred_denorm, ct_true_denorm)
        metric_tracker.update(metrics['psnr'], metrics['ssim'], batch_size=ct_true.size(0))
        
        # Save images
        if save_images and local_rank == 0 and save_folder is not None:
            for b_id in range(ct_true.size(0)):
                ct_true_img = ct_true_denorm[b_id, 0].cpu().numpy()
                ct_pred_img = ct_pred_denorm[b_id, 0].cpu().numpy()
                
                ct_true_img = np.clip(ct_true_img, 0, 1) * 255
                ct_pred_img = np.clip(ct_pred_img, 0, 1) * 255
                
                cv2.imwrite(
                    os.path.join(save_folder, f'{str(img_idx).zfill(5)}_ct_true.png'),
                    ct_true_img.astype(np.uint8)
                )
                cv2.imwrite(
                    os.path.join(save_folder, f'{str(img_idx).zfill(5)}_ct_pred.png'),
                    ct_pred_img.astype(np.uint8)
                )
                
                img_idx += 1
        
        if batch_idx % 10 == 0:
            print(f"Evaluation batch {batch_idx}/{len(test_loader)}, "
                  f"PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}")
    
    # Get average metrics
    avg_metrics = metric_tracker.get_average()
    
    # Log metrics
    if log_writer is not None and local_rank == 0:
        log_writer.add_scalar('eval/psnr', avg_metrics['psnr'], epoch)
        log_writer.add_scalar('eval/ssim', avg_metrics['ssim'], epoch)
    
    print(f"Evaluation Results - Epoch {epoch}:")
    print(f"  Average PSNR: {avg_metrics['psnr']:.4f} dB")
    print(f"  Average SSIM: {avg_metrics['ssim']:.4f}")
    
    # Restore original params
    for param, orig_param in zip(trainable_params, original_params):
        param.data.copy_(orig_param)
    print("Restored from EMA")
    
    return avg_metrics
