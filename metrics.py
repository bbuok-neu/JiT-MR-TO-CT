"""
Evaluation Metrics for Medical Image Synthesis
Implements SSIM and PSNR for MR-to-CT evaluation
"""
import torch
import torch.nn.functional as F
import math


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images
    
    Args:
        img1: First image tensor (N, C, H, W) or (C, H, W)
        img2: Second image tensor (N, C, H, W) or (C, H, W)
        max_val: Maximum possible pixel value (default: 1.0 for normalized images)
    
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_val) - 10 * torch.log10(mse)
    return psnr.item()


def calculate_ssim(img1, img2, window_size=11, max_val=1.0, size_average=True):
    """
    Calculate Structural Similarity Index (SSIM) between two images
    
    Args:
        img1: First image tensor (N, C, H, W)
        img2: Second image tensor (N, C, H, W)
        window_size: Size of the Gaussian window (default: 11)
        max_val: Maximum possible pixel value (default: 1.0 for normalized images)
        size_average: If True, return average SSIM; if False, return SSIM map
    
    Returns:
        SSIM value (scalar or map)
    """
    # Constants for stability
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # Create Gaussian window
    window = _create_window(window_size, img1.size(1)).to(img1.device)
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _create_window(window_size, num_channels):
    """
    Create a Gaussian window for SSIM calculation
    
    Args:
        window_size: Size of the window
        num_channels: Number of channels in the image
    
    Returns:
        Gaussian window tensor
    """
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([
            math.exp(-(x - window_size//2)**2 / float(2*sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(num_channels, 1, window_size, window_size).contiguous()
    return window


def evaluate_metrics(pred_ct, true_ct, denormalize_fn=None):
    """
    Evaluate PSNR and SSIM metrics for predicted and true CT images
    
    Args:
        pred_ct: Predicted CT image tensor (N, 1, H, W)
        true_ct: Ground truth CT image tensor (N, 1, H, W)
        denormalize_fn: Optional function to denormalize images before computing metrics
    
    Returns:
        Dictionary with 'psnr' and 'ssim' values
    """
    # Denormalize if needed
    if denormalize_fn is not None:
        pred_ct = denormalize_fn(pred_ct)
        true_ct = denormalize_fn(true_ct)
    
    # Ensure values are in valid range [0, 1]
    pred_ct = torch.clamp(pred_ct, 0.0, 1.0)
    true_ct = torch.clamp(true_ct, 0.0, 1.0)
    
    # Calculate metrics
    psnr = calculate_psnr(pred_ct, true_ct, max_val=1.0)
    ssim = calculate_ssim(pred_ct, true_ct, max_val=1.0)
    
    return {
        'psnr': psnr,
        'ssim': ssim
    }


class MetricTracker:
    """
    Track metrics over multiple batches
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.count = 0
    
    def update(self, psnr, ssim, batch_size=1):
        self.psnr_sum += psnr * batch_size
        self.ssim_sum += ssim * batch_size
        self.count += batch_size
    
    def get_average(self):
        if self.count == 0:
            return {'psnr': 0.0, 'ssim': 0.0}
        return {
            'psnr': self.psnr_sum / self.count,
            'ssim': self.ssim_sum / self.count
        }
