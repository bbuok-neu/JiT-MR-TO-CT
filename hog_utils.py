"""
HOG (Histogram of Oriented Gradients) Feature Extraction Utilities
Uses HOG-PyTorch for GPU-accelerated HOG computation
https://github.com/Simon-Bertrand/HOG-PyTorch
"""
import math
import torch
import torch.nn as nn
from torch_hog import HOG


class PositionalEncoding2D(nn.Module):
    """
    2D Sinusoidal Positional Encoding for HOG block features.
    
    Since HOG features come from a 2D grid of blocks (e.g., 31x31 = 961 blocks),
    we use 2D positional encoding to preserve spatial relationships.
    
    This helps the model understand the spatial arrangement of HOG blocks
    and improves cross-attention conditioning.
    """
    
    def __init__(self, embed_dim, grid_size=31, max_grid_size=64):
        """
        Args:
            embed_dim: Embedding dimension (must be divisible by 4 for proper 2D sin/cos encoding)
            grid_size: Size of the HOG block grid (e.g., 31 for 256x256 images with cell=8, block=2)
            max_grid_size: Maximum grid size for position encoding buffer
        """
        super().__init__()
        
        if embed_dim % 4 != 0:
            raise ValueError(f"embed_dim must be divisible by 4 for 2D positional encoding, got {embed_dim}")
        
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        
        # Each dimension (x, y) gets half the embedding dimension
        # Each half is further split into sin/cos pairs
        dim_per_axis = embed_dim // 2
        
        # Create position encoding buffer
        pe = torch.zeros(max_grid_size, max_grid_size, embed_dim)
        
        # Compute 2D sinusoidal positions
        # k iterates through pairs (0,1), (2,3), etc. within each axis's half
        for i in range(max_grid_size):
            for j in range(max_grid_size):
                for k in range(0, dim_per_axis, 2):
                    div_term = math.exp(k * (-math.log(10000.0) / dim_per_axis))
                    # X position encoding (first half of embedding)
                    pe[i, j, k] = math.sin(i * div_term)
                    pe[i, j, k + 1] = math.cos(i * div_term)
                    # Y position encoding (second half of embedding)
                    pe[i, j, dim_per_axis + k] = math.sin(j * div_term)
                    pe[i, j, dim_per_axis + k + 1] = math.cos(j * div_term)
        
        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)
    
    def forward(self, x, grid_h=None, grid_w=None):
        """
        Add positional encoding to input features.
        
        Args:
            x: Input tensor of shape (B, seq_len, embed_dim)
            grid_h: Height of the grid (default: self.grid_size)
            grid_w: Width of the grid (default: self.grid_size)
        
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        grid_h = grid_h or self.grid_size
        grid_w = grid_w or self.grid_size
        
        # Get positional encoding for the grid
        pe = self.pe[:grid_h, :grid_w, :].reshape(-1, self.embed_dim)  # (seq_len, embed_dim)
        
        # Add to input (broadcast over batch dimension)
        return x + pe.unsqueeze(0)


class HOGEmbedding(nn.Module):
    """
    HOG Feature Embedding Module with Projection and Positional Encoding.
    
    Projects raw HOG features (36-dim) to a higher dimension (e.g., 768)
    and adds 2D positional encoding to preserve spatial information.
    
    This improves the capacity of HOG features for cross-attention conditioning
    and helps the model understand the spatial arrangement of HOG blocks.
    """
    
    def __init__(self, input_dim=36, output_dim=768, grid_size=31, dropout=0.1):
        """
        Args:
            input_dim: Input HOG feature dimension (default: 36 = 2*2*9)
            output_dim: Output embedding dimension (default: 768)
            grid_size: Size of the HOG block grid (default: 31 for 256x256 images)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        
        # Feature projection: 36 -> 768
        # Following the user's suggested architecture
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 2D Positional encoding for spatial information
        self.pos_encoding = PositionalEncoding2D(
            embed_dim=output_dim,
            grid_size=grid_size
        )
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hog_features, grid_h=None, grid_w=None):
        """
        Project HOG features and add positional encoding.
        
        Args:
            hog_features: Raw HOG features of shape (B, seq_len, input_dim)
                         e.g., (B, 961, 36) for 256x256 images
            grid_h: Height of the HOG block grid (optional)
            grid_w: Width of the HOG block grid (optional)
        
        Returns:
            Embedded HOG features of shape (B, seq_len, output_dim)
            e.g., (B, 961, 768)
        """
        # Project features: (B, 961, 36) -> (B, 961, 768)
        x = self.projection(hog_features)
        
        # Add 2D positional encoding
        x = self.pos_encoding(x, grid_h, grid_w)
        
        # Apply dropout
        x = self.dropout(x)
        
        return x


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
