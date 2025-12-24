"""
HOG (Histogram of Oriented Gradients) Feature Extraction Utilities
Uses HOG-PyTorch for GPU-accelerated HOG computation
https://github.com/Simon-Bertrand/HOG-PyTorch
"""
import torch
import torch.nn as nn
from torch_hog import HOG


class HOGExtractor(nn.Module):
    """
    HOG feature extractor wrapper for cross-attention conditioning.
    
    For 256x256 images with standard HOG parameters:
    - Pixels per cell: 8x8
    - Cells per block: 2x2
    - Orientations: 9 (0°-180°)
    
    This produces:
    - Cells: 256/8 = 32 per dimension (32x32 = 1024 cells)
    - Blocks: 32-2+1 = 31 per dimension (31x31 = 961 blocks with stride=1)
    - Features per block: 2x2x9 = 36
    - Output shape: (B, 961, 36) for cross-attention conditioning
    """
    
    def __init__(self, cell_size=8, block_size=2, num_bins=9):
        """
        Args:
            cell_size: Size of each cell in pixels (default: 8)
            block_size: Number of cells per block (default: 2)
            num_bins: Number of orientation bins (default: 9)
        """
        super().__init__()
        self.cell_size = cell_size
        self.block_size = block_size
        self.num_bins = num_bins
        
        # Initialize HOG module from torch_hog
        self.hog = HOG(
            cellSize=cell_size,
            blockSize=block_size,
            nPhaseBins=num_bins,
            kernel='finite',  # Use finite difference for gradient computation
            normalization='L2',  # L2 block normalization
            accumulate='bilinear'  # Bilinear interpolation for bin accumulation
        )
    
    @property
    def feature_dim(self):
        """Returns the feature dimension per block (cross_attention_dim)."""
        return self.block_size * self.block_size * self.num_bins  # 2*2*9 = 36
    
    def get_sequence_length(self, img_size):
        """
        Calculate the sequence length (number of blocks) for a given image size.
        
        Args:
            img_size: Image height/width (assumes square images)
        
        Returns:
            Number of blocks (sequence length)
        """
        cells_per_dim = img_size // self.cell_size
        blocks_per_dim = cells_per_dim - self.block_size + 1
        return blocks_per_dim * blocks_per_dim  # e.g., 31*31 = 961 for 256x256
    
    def forward(self, x):
        """
        Extract HOG features from input images.
        
        Args:
            x: Input image tensor of shape (B, 1, H, W)
               Values should be in range [0, 1] or normalized
        
        Returns:
            HOG features of shape (B, seq_len, feature_dim)
            e.g., (B, 961, 36) for 256x256 images
        """
        # Ensure input is in correct format for HOG computation
        # HOG expects values in a reasonable range, so we may need to denormalize
        # For normalized images, we typically need to rescale to [0, 1] range
        
        # Compute HOG features
        # Output shape: (B, C, num_blocks_h, num_blocks_w, block_h, block_w, num_bins)
        hog_features = self.hog(x)
        
        # Get dimensions
        B = hog_features.shape[0]
        
        # Remove channel dimension for single-channel images
        # Shape: (B, num_blocks_h, num_blocks_w, block_h, block_w, num_bins)
        hog_features = hog_features.squeeze(1)
        
        # Reshape to (B, seq_len, feature_dim) for cross-attention
        # seq_len = num_blocks_h * num_blocks_w
        # feature_dim = block_h * block_w * num_bins
        seq_len = hog_features.shape[1] * hog_features.shape[2]
        feature_dim = hog_features.shape[3] * hog_features.shape[4] * hog_features.shape[5]
        
        hog_features = hog_features.reshape(B, seq_len, feature_dim)
        
        return hog_features


def compute_hog_features(image, hog_extractor, mean=0.5, std=0.5):
    """
    Compute HOG features from a normalized image.
    
    Args:
        image: Input image tensor of shape (1, H, W), normalized with z-score
        hog_extractor: HOGExtractor instance
        mean: Mean used for z-score normalization
        std: Std used for z-score normalization
    
    Returns:
        HOG features of shape (seq_len, feature_dim)
    """
    # Denormalize image to [0, 1] range for HOG computation
    # z-score: x_norm = (x - mean) / std  =>  x = x_norm * std + mean
    image_denorm = image * std + mean
    image_denorm = torch.clamp(image_denorm, 0.0, 1.0)
    
    # Add batch dimension if needed
    if image_denorm.dim() == 3:
        image_denorm = image_denorm.unsqueeze(0)
    
    # Compute HOG features
    with torch.no_grad():
        hog_features = hog_extractor(image_denorm)
    
    # Remove batch dimension
    hog_features = hog_features.squeeze(0)
    
    return hog_features
