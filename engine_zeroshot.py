"""
Training and Evaluation Engine for Zero-Shot MR-to-CT Synthesis
Training: CT images only with HOG conditioning
Inference: MR images with HOG conditioning (zero-shot transfer)
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


def train_one_epoch_zeroshot(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    """
    Train one epoch for zero-shot CT synthesis using HOG conditioning.
    
    Training uses only CT images with CT-derived HOG as conditioning.
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (ct, hog_features) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Per iteration lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Move to device
        ct = ct.to(device, non_blocking=True)
        hog_features = hog_features.to(device, non_blocking=True)

        # Device-agnostic autocast
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            # Forward pass: CT + HOG (from CT) during training
            loss = model(ct, hog_features)

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
def evaluate_zeroshot(model_without_ddp, test_loader, args, epoch, log_writer=None, save_images=True):
    """
    Evaluate zero-shot MR-to-CT synthesis on test set.
    
    Uses MR images with MR-derived HOG for conditioning.
    Compares generated CT with ground truth CT.
    """
    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    
    # Switch to EMA params
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to EMA for evaluation")
    model_without_ddp.load_state_dict(ema_state_dict)
    
    # Metric tracker
    metric_tracker = MetricTracker()
    
    # Create output directory for saving images
    if save_images and misc.get_rank() == 0:
        save_folder = os.path.join(
            args.output_dir,
            "eval_images",
            f"epoch_{epoch}"
        )
        os.makedirs(save_folder, exist_ok=True)
        print(f"Saving evaluation images to: {save_folder}")
    else:
        save_folder = None
    
    img_idx = 0
    
    # Device-agnostic autocast
    device_type = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    
    for batch_idx, (mr, hog_features, ct_true) in enumerate(test_loader):
        mr = mr.to(args.device, non_blocking=True)
        hog_features = hog_features.to(args.device, non_blocking=True)
        
        # Check if CT ground truth is available
        has_ct = ct_true is not None
        if has_ct:
            # Handle the case where ct_true might be a list of tensors (some None)
            if isinstance(ct_true, torch.Tensor):
                ct_true = ct_true.to(args.device, non_blocking=True)
            else:
                has_ct = False
        
        # Generate CT from MR-derived HOG (zero-shot transfer)
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            ct_pred = model_without_ddp.generate(hog_features)
        
        # Denormalize for metrics calculation if ground truth is available
        if has_ct:
            ct_pred_denorm = ct_pred * args.ct_std + args.ct_mean
            ct_true_denorm = ct_true * args.ct_std + args.ct_mean
            
            # Clamp to [0, 1] for metrics
            ct_pred_denorm = torch.clamp(ct_pred_denorm, 0.0, 1.0)
            ct_true_denorm = torch.clamp(ct_true_denorm, 0.0, 1.0)
            
            # Calculate metrics for batch
            metrics = evaluate_metrics(ct_pred_denorm, ct_true_denorm)
            metric_tracker.update(metrics['psnr'], metrics['ssim'], batch_size=mr.size(0))
        
        # Save images (only from rank 0)
        if save_images and local_rank == 0 and save_folder is not None:
            for b_id in range(mr.size(0)):
                # Convert MR to numpy and save
                mr_img = (mr[b_id, 0].cpu().numpy() * args.mr_std + args.mr_mean)
                mr_img = np.clip(mr_img, 0, 1) * 255
                
                # Convert predicted CT to numpy and save
                ct_pred_img = (ct_pred[b_id, 0].cpu().float().numpy() * args.ct_std + args.ct_mean)
                ct_pred_img = np.clip(ct_pred_img, 0, 1) * 255
                
                # Save MR and predicted CT
                cv2.imwrite(
                    os.path.join(save_folder, f'{str(img_idx).zfill(5)}_mr.png'),
                    mr_img.astype(np.uint8)
                )
                cv2.imwrite(
                    os.path.join(save_folder, f'{str(img_idx).zfill(5)}_ct_pred.png'),
                    ct_pred_img.astype(np.uint8)
                )
                
                # Save ground truth CT if available
                if has_ct:
                    ct_true_img = ct_true_denorm[b_id, 0].cpu().numpy() * 255
                    cv2.imwrite(
                        os.path.join(save_folder, f'{str(img_idx).zfill(5)}_ct_true.png'),
                        ct_true_img.astype(np.uint8)
                    )
                
                img_idx += 1
        
        if batch_idx % 10 == 0:
            if has_ct:
                print(f"Evaluation batch {batch_idx}/{len(test_loader)}, "
                      f"Current PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}")
            else:
                print(f"Evaluation batch {batch_idx}/{len(test_loader)} (no CT ground truth)")
    
    # Get average metrics
    avg_metrics = metric_tracker.get_average()
    
    # Log metrics
    if log_writer is not None and misc.get_rank() == 0:
        log_writer.add_scalar('eval/psnr', avg_metrics['psnr'], epoch)
        log_writer.add_scalar('eval/ssim', avg_metrics['ssim'], epoch)
    
    print(f"Zero-Shot Evaluation Results - Epoch {epoch}:")
    print(f"  Average PSNR: {avg_metrics['psnr']:.4f} dB")
    print(f"  Average SSIM: {avg_metrics['ssim']:.4f}")
    
    # Switch back from EMA
    print("Switch back from EMA")
    model_without_ddp.load_state_dict(model_state_dict)
    
    return avg_metrics
