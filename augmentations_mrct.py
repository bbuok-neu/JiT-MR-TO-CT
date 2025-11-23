"""
Medical Image Augmentation for MR-CT Synthesis
Implements spatial augmentations using TorchIO for paired medical images
"""

import torch
import numpy as np
from typing import Tuple, Optional

try:
    import torchio as tio
    TORCHIO_AVAILABLE = True
except ImportError:
    TORCHIO_AVAILABLE = False
    print("Warning: TorchIO not available. Install with: pip install torchio")


class PairedMedicalAugmentation:
    """
    Augmentation pipeline for paired MR-CT images using TorchIO.
    Applies the same spatial transforms to both MR and CT images to maintain alignment.
    """
    
    def __init__(
        self,
        rotation_degrees: Tuple[float, float] = (-15, 15),
        enable_flip: bool = True,
        enable_elastic: bool = True,
        elastic_num_control_points: int = 7,
        elastic_max_displacement: float = 7.5,
        zoom_range: Tuple[float, float] = (0.9, 1.1),
        augmentation_probability: float = 0.8,
        use_torchio: bool = True
    ):
        """
        Initialize medical image augmentation pipeline.
        
        Args:
            rotation_degrees: Range for random rotation in degrees (min, max)
            enable_flip: Whether to enable random horizontal flip
            enable_elastic: Whether to enable elastic deformation
            elastic_num_control_points: Number of control points for elastic transform
            elastic_max_displacement: Maximum displacement in voxels for elastic transform
            zoom_range: Range for random zoom/scaling (min, max)
            augmentation_probability: Probability of applying augmentations
            use_torchio: Whether to use TorchIO (if available) or fallback to basic transforms
        """
        self.rotation_degrees = rotation_degrees
        self.enable_flip = enable_flip
        self.enable_elastic = enable_elastic
        self.elastic_num_control_points = elastic_num_control_points
        self.elastic_max_displacement = elastic_max_displacement
        self.zoom_range = zoom_range
        self.augmentation_probability = augmentation_probability
        self.use_torchio = use_torchio and TORCHIO_AVAILABLE
        
        if self.use_torchio:
            self._build_torchio_transform()
        else:
            if use_torchio:
                print("TorchIO requested but not available. Using basic augmentations.")
            self._build_basic_transform()
    
    def _build_torchio_transform(self):
        """Build TorchIO augmentation pipeline."""
        transforms_list = []
        
        # Random rotation
        if self.rotation_degrees[0] != 0 or self.rotation_degrees[1] != 0:
            transforms_list.append(
                tio.RandomAffine(
                    scales=0,  # No scaling in affine
                    degrees=(*self.rotation_degrees, 0, 0, 0, 0),  # Only rotate around z-axis (2D rotation)
                    translation=0,
                    p=self.augmentation_probability
                )
            )
        
        # Random flip (horizontal)
        if self.enable_flip:
            transforms_list.append(
                tio.RandomFlip(
                    axes=('LR',),  # Left-Right flip
                    flip_probability=0.5
                )
            )
        
        # Elastic deformation
        if self.enable_elastic:
            transforms_list.append(
                tio.RandomElasticDeformation(
                    num_control_points=self.elastic_num_control_points,
                    max_displacement=self.elastic_max_displacement,
                    p=self.augmentation_probability * 0.5  # Apply less frequently
                )
            )
        
        # Random zoom/scaling
        if self.zoom_range[0] != 1.0 or self.zoom_range[1] != 1.0:
            transforms_list.append(
                tio.RandomAffine(
                    scales=(*self.zoom_range, *self.zoom_range, 1),  # Scale x and y, not z
                    degrees=0,
                    translation=0,
                    p=self.augmentation_probability
                )
            )
        
        self.transform = tio.Compose(transforms_list) if transforms_list else None
    
    def _build_basic_transform(self):
        """Build basic augmentation pipeline without TorchIO."""
        from torchvision import transforms as T
        
        transforms_list = []
        
        # Random horizontal flip
        if self.enable_flip:
            transforms_list.append(T.RandomHorizontalFlip(p=0.5))
        
        # Random rotation (basic)
        if self.rotation_degrees[0] != 0 or self.rotation_degrees[1] != 0:
            transforms_list.append(
                T.RandomRotation(
                    degrees=self.rotation_degrees,
                    interpolation=T.InterpolationMode.BILINEAR
                )
            )
        
        # Random affine for zoom (basic approximation)
        if self.zoom_range[0] != 1.0 or self.zoom_range[1] != 1.0:
            transforms_list.append(
                T.RandomAffine(
                    degrees=0,
                    scale=self.zoom_range,
                    interpolation=T.InterpolationMode.BILINEAR
                )
            )
        
        self.transform = T.Compose(transforms_list) if transforms_list else None
    
    def __call__(self, mr: np.ndarray, ct: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation to paired MR-CT images.
        
        Args:
            mr: MR image as numpy array (H, W)
            ct: CT image as numpy array (H, W)
        
        Returns:
            Tuple of (augmented_mr, augmented_ct) as torch tensors (1, H, W)
        """
        if self.transform is None:
            # No augmentation, just convert to tensor
            mr_tensor = torch.from_numpy(mr).unsqueeze(0).float()
            ct_tensor = torch.from_numpy(ct).unsqueeze(0).float()
            return mr_tensor, ct_tensor
        
        if self.use_torchio:
            return self._apply_torchio(mr, ct)
        else:
            return self._apply_basic(mr, ct)
    
    def _apply_torchio(self, mr: np.ndarray, ct: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply TorchIO transformations to paired images."""
        # Convert to torch tensors and add batch and channel dimensions
        # TorchIO expects (C, H, W, D) format, we'll use D=1 for 2D slices
        mr_tensor = torch.from_numpy(mr).unsqueeze(0).unsqueeze(-1).float()  # (1, H, W, 1)
        ct_tensor = torch.from_numpy(ct).unsqueeze(0).unsqueeze(-1).float()  # (1, H, W, 1)
        
        # Create a subject with both images
        subject = tio.Subject(
            mr=tio.ScalarImage(tensor=mr_tensor),
            ct=tio.ScalarImage(tensor=ct_tensor)
        )
        
        # Apply transforms
        transformed = self.transform(subject)
        
        # Extract tensors and remove the depth dimension
        mr_out = transformed['mr'].data.squeeze(-1)  # (1, H, W)
        ct_out = transformed['ct'].data.squeeze(-1)  # (1, H, W)
        
        return mr_out, ct_out
    
    def _apply_basic(self, mr: np.ndarray, ct: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply basic transformations to paired images."""
        from PIL import Image
        
        # Scale to 0-255 range for PIL
        mr_uint8 = (mr if mr.max() > 1.0 else mr * 255).astype(np.uint8)
        ct_uint8 = (ct if ct.max() > 1.0 else ct * 255).astype(np.uint8)
        
        # Stack for joint transformation
        stacked = np.stack([mr_uint8, ct_uint8], axis=-1)
        stacked_pil = Image.fromarray(stacked.astype(np.uint8))
        
        # Apply transforms
        stacked_transformed = self.transform(stacked_pil)
        
        # Convert back to tensor and split channels
        if isinstance(stacked_transformed, torch.Tensor):
            # Already a tensor from torchvision transforms
            if stacked_transformed.shape[0] == 2:
                mr_out = stacked_transformed[0:1]  # (1, H, W)
                ct_out = stacked_transformed[1:2]  # (1, H, W)
            else:
                # Handle unexpected format
                mr_out = stacked_transformed[..., 0:1].permute(2, 0, 1)
                ct_out = stacked_transformed[..., 1:2].permute(2, 0, 1)
        else:
            # Convert PIL image to tensor
            from torchvision.transforms.functional import pil_to_tensor
            tensor = pil_to_tensor(stacked_transformed).float()
            mr_out = tensor[0:1]
            ct_out = tensor[1:2]
        
        # Ensure values are in [0, 255] range
        if mr_out.max() <= 1.0:
            mr_out = mr_out * 255.0
        if ct_out.max() <= 1.0:
            ct_out = ct_out * 255.0
        
        return mr_out, ct_out


def get_medical_augmentation(
    mode: str = 'train',
    rotation_degrees: Tuple[float, float] = (-15, 15),
    enable_flip: bool = True,
    enable_elastic: bool = True,
    zoom_range: Tuple[float, float] = (0.9, 1.1),
    use_torchio: bool = True
) -> Optional[PairedMedicalAugmentation]:
    """
    Get medical image augmentation pipeline.
    
    Args:
        mode: 'train' or 'test'. Returns None for test mode.
        rotation_degrees: Range for random rotation in degrees
        enable_flip: Whether to enable random horizontal flip
        enable_elastic: Whether to enable elastic deformation
        zoom_range: Range for random zoom/scaling
        use_torchio: Whether to use TorchIO (if available)
    
    Returns:
        PairedMedicalAugmentation instance for training, None for testing
    """
    if mode == 'test':
        return None
    
    return PairedMedicalAugmentation(
        rotation_degrees=rotation_degrees,
        enable_flip=enable_flip,
        enable_elastic=enable_elastic,
        zoom_range=zoom_range,
        use_torchio=use_torchio
    )
