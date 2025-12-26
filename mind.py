"""
MIND (Modality Independent Neighbourhood Descriptor) Module
Adapted from https://github.com/bbuok-neu/Pytorch-MIND

MIND is a modality-independent feature descriptor that captures local structural
information in medical images. The key property is that MIND(CT) ≈ MIND(MR) for
aligned image pairs, enabling zero-shot cross-modality synthesis.
"""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class MINDModule(nn.Module):
    """
    Compute MIND (Modality Independent Neighbourhood Descriptor) features.
    
    The output has (neigh_size * neigh_size - 1) channels when neigh4=False,
    or 4 channels when neigh4=True.
    
    Note: The output size is reduced by (patch_size + neigh_size - 2) / 2 on each side.
    Use compute_mind_with_padding() function to get MIND features with original size.
    
    Args:
        patch_size: Size of the Gaussian smoothing patch
        neigh_size: Size of the neighborhood (determines number of output channels)
        sigma: Standard deviation for Gaussian kernel
        eps: Small value for numerical stability
        image_size0: Height of input image
        image_size1: Width of input image
        neigh4: If True, use only 4-connectivity (4 channels output)
                If False, use full neighborhood (neigh_size^2 - 1 channels output)
    """
    def __init__(self, patch_size, neigh_size, sigma, eps, image_size0, image_size1, neigh4=False):
        super(MINDModule, self).__init__()
        self.patch_size = patch_size
        self.neigh_size = neigh_size
        self.sigma = sigma
        self.eps = eps
        self.image_size0 = image_size0
        self.image_size1 = image_size1
        self.neigh4 = neigh4
        
        # Pre-compute Gaussian kernel
        self.register_buffer('gaussian_weight', 
                            torch.from_numpy(self._gaussian_kernel(sigma, patch_size)))

    def forward(self, image):
        """
        Compute MIND features for input image.
        
        Args:
            image: Input image tensor (N, 1, H, W)
        
        Returns:
            MIND feature tensor (N, C, H', W') where:
            - C = 4 if neigh4=True, else neigh_size^2 - 1
            - H' = H - (patch_size + neigh_size - 2)
            - W' = W - (patch_size + neigh_size - 2)
        """
        reduce_size = int((self.patch_size + self.neigh_size - 2) / 2)
        
        # Compute crop boundaries once to avoid duplication
        crop_h_start = reduce_size
        crop_h_end = self.image_size0 - reduce_size
        crop_w_start = reduce_size
        crop_w_end = self.image_size1 - reduce_size

        # Estimate local variance of each pixel
        Vimg = torch.add(self._Dp(image, -1, 0), self._Dp(image, 1, 0))
        Vimg = torch.add(Vimg, self._Dp(image, 0, -1))
        Vimg = torch.add(Vimg, self._Dp(image, 0, 1))
        Vimg = torch.div(Vimg, 4) + self.eps

        output = None

        if not self.neigh4:
            # Full neighborhood (neigh_size^2 - 1 channels)
            half_neigh = self.neigh_size // 2
            shift_range = range(-half_neigh, half_neigh + 1)
            for xshift in shift_range:
                for yshift in shift_range:
                    if (xshift, yshift) == (0, 0):
                        continue
                    MIND_tmp = torch.exp(-self._Dp(image, xshift, yshift) / Vimg)
                    tmp = MIND_tmp[:, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
                    if output is None:
                        output = tmp
                    else:
                        output = torch.cat([output, tmp], 1)
        else:
            # 4-connectivity (4 channels)
            MIND_tmp = torch.exp(-self._Dp(image, -1, 0) / Vimg)
            output = MIND_tmp[:, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
            for xshift, yshift in [(1, 0), (0, -1), (0, 1)]:
                MIND_tmp = torch.exp(-self._Dp(image, xshift, yshift) / Vimg)
                tmp = MIND_tmp[:, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
                output = torch.cat([output, tmp], 1)

        # Normalization
        input_max, _ = torch.max(output, dim=1, keepdim=True)
        output = torch.div(output, input_max.clamp(min=self.eps))

        return output

    def _Dp(self, image, xshift, yshift):
        """Compute patch distance for given shift."""
        shift_image = self._torch_image_translate(image, xshift, yshift)
        diff = image - shift_image
        diff_square = diff * diff
        res = F.conv2d(diff_square, weight=self.gaussian_weight, stride=1, 
                      padding=self.patch_size // 2)
        return res

    @staticmethod
    def _gaussian_kernel(sigma, sz):
        """Generate Gaussian kernel."""
        xpos_vec = np.arange(sz)
        ypos_vec = np.arange(sz)
        output = np.ones([1, 1, sz, sz], dtype=np.float32)
        midpos = sz // 2
        for xpos in xpos_vec:
            for ypos in ypos_vec:
                output[:, :, xpos, ypos] = np.exp(
                    -((xpos - midpos) ** 2 + (ypos - midpos) ** 2) / (2 * sigma ** 2)
                ) / (2 * np.pi * sigma ** 2)
        return output

    @staticmethod
    def _torch_image_translate(input_, tx, ty, interpolation='nearest'):
        """Translate image by (tx, ty) pixels.
        
        Note: Requires image dimensions > 1 to avoid division by zero.
        """
        # Validate minimum image size to prevent division by zero
        h, w = input_.size()[2], input_.size()[3]
        if h <= 1 or w <= 1:
            raise ValueError(f"Image dimensions must be > 1, got {h}x{w}")
        
        translation_matrix = torch.zeros([input_.size(0), 3, 3], 
                                        dtype=input_.dtype, device=input_.device)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 0, 2] = -2 * tx / (h - 1)
        translation_matrix[:, 1, 2] = -2 * ty / (w - 1)
        translation_matrix[:, 2, 2] = 1.0
        grid = F.affine_grid(translation_matrix[:, 0:2, :], input_.size(), align_corners=True)
        wrp = F.grid_sample(input_, grid, mode=interpolation, align_corners=True)
        return wrp


def compute_mind_with_padding(image, patch_size=7, neigh_size=7, sigma=0.5, eps=1e-6, neigh4=False):
    """
    Compute MIND features and pad back to original image size.
    
    This is a convenience function that handles the size reduction from MIND
    computation by padding the output back to the original input size.
    
    Args:
        image: Input image tensor (N, 1, H, W)
        patch_size: Size of Gaussian smoothing patch (default: 7)
        neigh_size: Size of neighborhood (default: 7)
        sigma: Gaussian kernel sigma (default: 0.5)
        eps: Numerical stability epsilon (default: 1e-6)
        neigh4: Use 4-connectivity instead of full neighborhood (default: False)
    
    Returns:
        MIND features tensor (N, C, H, W) with same spatial size as input
        where C = 4 if neigh4 else neigh_size^2 - 1
    """
    N, _, H, W = image.shape
    
    # Create MIND module
    mind_module = MINDModule(
        patch_size=patch_size,
        neigh_size=neigh_size,
        sigma=sigma,
        eps=eps,
        image_size0=H,
        image_size1=W,
        neigh4=neigh4
    ).to(image.device)
    
    # Compute MIND features
    with torch.no_grad():
        mind_features = mind_module(image)
    
    # Calculate padding needed
    reduce_size = int((patch_size + neigh_size - 2) / 2)
    
    # Pad back to original size
    mind_features = F.pad(mind_features, (reduce_size, reduce_size, reduce_size, reduce_size), 
                         mode='reflect')
    
    return mind_features


def get_mind_channels(neigh_size=7, neigh4=False):
    """
    Calculate the number of MIND feature channels.
    
    Args:
        neigh_size: Size of neighborhood
        neigh4: Whether to use 4-connectivity
    
    Returns:
        Number of MIND feature channels
    """
    if neigh4:
        return 4
    else:
        return neigh_size * neigh_size - 1
