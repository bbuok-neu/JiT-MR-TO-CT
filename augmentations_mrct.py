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
    Applies intensity/artifact transforms only to MR images.
    """
    
    def __init__(
        self,
        # Geometric transforms (applied to both MR and CT)
        rotation_degrees: Tuple[float, float] = (-15, 15),
        enable_flip: bool = True,
        enable_elastic: bool = True,
        elastic_num_control_points: int = 5,
        elastic_max_displacement: float = (5,5,0),
        zoom_range: Tuple[float, float] = (0.9, 1.1),
        enable_grid_distortion: bool = False,
        grid_num_control_points: int = 5,
        grid_max_displacement: float = 10.0,
        # MR-only intensity/artifact transforms
        enable_bias_field: bool = False,
        bias_coefficients: float = 0.5,
        enable_motion_ghosting: bool = False,
        motion_num_transforms: int = 2,
        motion_intensity: Tuple[float, float] = (0.5, 1.0),
        enable_rician_noise: bool = False,
        rician_std: Tuple[float, float] = (0.0, 0.05),
        enable_gamma: bool = False,
        gamma_range: Tuple[float, float] = (0.8, 1.2),
        enable_cutout: bool = False,
        cutout_num_holes: Tuple[int, int] = (1, 3),
        cutout_max_size: Tuple[int, int] = (20, 40),
        # General settings
        augmentation_probability: float = 0.5,
        use_torchio: bool = True
    ):
        """
        Initialize medical image augmentation pipeline.
        
        Args:
            Geometric transforms (applied to both MR and CT):
                rotation_degrees: Range for random rotation in degrees (min, max)
                enable_flip: Whether to enable random horizontal flip
                enable_elastic: Whether to enable elastic deformation
                elastic_num_control_points: Number of control points for elastic transform
                elastic_max_displacement: Maximum displacement in voxels for elastic transform
                zoom_range: Range for random zoom/scaling (min, max)
                enable_grid_distortion: Whether to enable grid distortion
                grid_num_control_points: Control points for grid distortion
                grid_max_displacement: Max displacement for grid distortion
            
            MR-only intensity/artifact transforms:
                enable_bias_field: Whether to simulate bias field inhomogeneity (MR only)
                bias_coefficients: Coefficient for bias field strength
                enable_motion_ghosting: Whether to simulate motion artifacts (MR only)
                motion_num_transforms: Number of motion transforms to apply
                motion_intensity: Intensity range for motion ghosting
                enable_rician_noise: Whether to add Rician noise (MR only)
                rician_std: Standard deviation range for Rician noise
                enable_gamma: Whether to apply gamma correction (MR only)
                gamma_range: Range for gamma values
                enable_cutout: Whether to apply random cutout/erasing (MR only)
                cutout_num_holes: Range for number of cutout holes
                cutout_max_size: Range for cutout hole size
            
            General:
                augmentation_probability: Probability of applying augmentations
                use_torchio: Whether to use TorchIO (if available) or fallback to basic transforms
        """
        # Geometric transforms
        self.rotation_degrees = rotation_degrees
        self.enable_flip = enable_flip
        self.enable_elastic = enable_elastic
        self.elastic_num_control_points = elastic_num_control_points
        self.elastic_max_displacement = elastic_max_displacement
        self.zoom_range = zoom_range
        self.enable_grid_distortion = enable_grid_distortion
        self.grid_num_control_points = grid_num_control_points
        self.grid_max_displacement = grid_max_displacement
        
        # MR-only transforms
        self.enable_bias_field = enable_bias_field
        self.bias_coefficients = bias_coefficients
        self.enable_motion_ghosting = enable_motion_ghosting
        self.motion_num_transforms = motion_num_transforms
        self.motion_intensity = motion_intensity
        self.enable_rician_noise = enable_rician_noise
        self.rician_std = rician_std
        self.enable_gamma = enable_gamma
        self.gamma_range = gamma_range
        self.enable_cutout = enable_cutout
        self.cutout_num_holes = cutout_num_holes
        self.cutout_max_size = cutout_max_size
        
        # General
        self.augmentation_probability = augmentation_probability
        self.use_torchio = use_torchio and TORCHIO_AVAILABLE
        
        if self.use_torchio:
            self._build_torchio_transforms()
        else:
            if use_torchio:
                print("TorchIO requested but not available. Using basic augmentations.")
            self._build_basic_transform()
    
    def _build_torchio_transforms(self):
        """Build TorchIO augmentation pipelines for geometric and MR-only transforms."""
        # Geometric transforms (applied to both MR and CT)
        geometric_transforms = []
        
        # Random rotation
        if self.rotation_degrees[0] != 0 or self.rotation_degrees[1] != 0:
            geometric_transforms.append(
                tio.RandomAffine(
                    scales=0,  # No scaling in affine
                    degrees=(*self.rotation_degrees, 0, 0, 0, 0),  # Only rotate around z-axis (2D rotation)
                    translation=0,
                    p=self.augmentation_probability
                )
            )
        
        # Random flip (horizontal)
        if self.enable_flip:
            geometric_transforms.append(
                tio.RandomFlip(
                    axes=('LR',),  # Left-Right flip
                    flip_probability=0.5
                )
            )
        
        # Elastic deformation
        if self.enable_elastic:
            geometric_transforms.append(
                tio.RandomElasticDeformation(
                    num_control_points=self.elastic_num_control_points,
                    max_displacement=self.elastic_max_displacement,
                    p=self.augmentation_probability * 0.5  # Apply less frequently
                )
            )
        
        # Grid distortion (alternative to elastic)
        if self.enable_grid_distortion:
            geometric_transforms.append(
                tio.RandomElasticDeformation(
                    num_control_points=self.grid_num_control_points,
                    max_displacement=self.grid_max_displacement,
                    p=self.augmentation_probability * 0.3  # Apply even less frequently
                )
            )
        
        # Random zoom/scaling
        if self.zoom_range[0] != 1.0 or self.zoom_range[1] != 1.0:
            geometric_transforms.append(
                tio.RandomAffine(
                    scales=(*self.zoom_range, *self.zoom_range, *(1,1)),  # Scale x and y, not z
                    degrees=0,
                    translation=0,
                    p=self.augmentation_probability
                )
            )
        
        self.geometric_transform = tio.Compose(geometric_transforms) if geometric_transforms else None
        
        # MR-only intensity/artifact transforms
        mr_only_transforms = []
        
        # Bias field simulation
        if self.enable_bias_field:
            mr_only_transforms.append(
                tio.RandomBiasField(
                    coefficients=self.bias_coefficients,
                    p=self.augmentation_probability * 0.6
                )
            )
        
        # Motion ghosting
        if self.enable_motion_ghosting:
            mr_only_transforms.append(
                tio.RandomGhosting(
                    num_ghosts=self.motion_num_transforms,
                    intensity=self.motion_intensity,
                    p=self.augmentation_probability * 0.4
                )
            )
        
        # Rician noise (MRI-specific)
        if self.enable_rician_noise:
            mr_only_transforms.append(
                tio.RandomNoise(
                    std=self.rician_std,
                    p=self.augmentation_probability * 0.5
                )
            )
        
        # Gamma correction
        if self.enable_gamma:
            mr_only_transforms.append(
                tio.RandomGamma(
                    log_gamma=self.gamma_range,
                    p=self.augmentation_probability * 0.6
                )
            )
        
        self.mr_only_transform = tio.Compose(mr_only_transforms) if mr_only_transforms else None
    
    def _build_basic_transform(self):
        """Build basic augmentation pipeline without TorchIO (geometric only)."""
        from torchvision import transforms as T
        
        # Only geometric transforms for basic mode
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
        
        self.geometric_transform = T.Compose(transforms_list) if transforms_list else None
        self.mr_only_transform = None  # No advanced transforms in basic mode
    
    def _apply_cutout(self, mr_tensor: torch.Tensor) -> torch.Tensor:
        """Apply random cutout/erasing to MR image only."""
        if not self.enable_cutout or np.random.rand() > self.augmentation_probability * 0.4:
            return mr_tensor
        
        # Get image dimensions
        _, H, W = mr_tensor.shape
        
        # Random number of holes
        num_holes = np.random.randint(self.cutout_num_holes[0], self.cutout_num_holes[1] + 1)
        
        # Clone tensor to avoid modifying original
        mr_cutout = mr_tensor.clone()
        
        for _ in range(num_holes):
            # Random hole size
            h = np.random.randint(self.cutout_max_size[0], self.cutout_max_size[1])
            w = np.random.randint(self.cutout_max_size[0], self.cutout_max_size[1])
            
            # Random position
            y = np.random.randint(0, max(1, H - h))
            x = np.random.randint(0, max(1, W - w))
            
            # Fill with zeros (or random noise)
            if np.random.rand() > 0.5:
                mr_cutout[:, y:y+h, x:x+w] = 0
            else:
                mr_cutout[:, y:y+h, x:x+w] = torch.randn_like(mr_cutout[:, y:y+h, x:x+w]) * 0.1
        
        return mr_cutout
    
    def __call__(self, mr: np.ndarray, ct: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation to paired MR-CT images.
        
        Args:
            mr: MR image as numpy array (H, W)
            ct: CT image as numpy array (H, W)
        
        Returns:
            Tuple of (augmented_mr, augmented_ct) as torch tensors (1, H, W)
        """
        if self.geometric_transform is None and self.mr_only_transform is None:
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
        
        # Apply geometric transforms to both MR and CT
        if self.geometric_transform is not None:
            # Create a subject with both images for joint geometric transforms
            subject = tio.Subject(
                mr=tio.ScalarImage(tensor=mr_tensor),
                ct=tio.ScalarImage(tensor=ct_tensor)
            )
            
            # Apply geometric transforms
            transformed = self.geometric_transform(subject)
            
            # Extract tensors
            mr_tensor = transformed['mr'].data
            ct_tensor = transformed['ct'].data
        
        # Apply MR-only intensity/artifact transforms
        if self.mr_only_transform is not None:
            # Create subject with only MR for intensity transforms
            mr_subject = tio.Subject(
                mr=tio.ScalarImage(tensor=mr_tensor)
            )
            
            # Apply MR-only transforms
            mr_transformed = self.mr_only_transform(mr_subject)
            mr_tensor = mr_transformed['mr'].data
        
        # Remove the depth dimension and apply cutout
        mr_out = mr_tensor.squeeze(-1)  # (1, H, W)
        ct_out = ct_tensor.squeeze(-1)  # (1, H, W)
        
        # Apply cutout to MR only
        mr_out = self._apply_cutout(mr_out)
        
        return mr_out, ct_out
    
    def _apply_basic(self, mr: np.ndarray, ct: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply basic transformations to paired images (geometric only)."""
        from PIL import Image
        
        # Scale to 0-255 range for PIL
        mr_uint8 = (mr if mr.max() > 1.0 else mr * 255).astype(np.uint8)
        ct_uint8 = (ct if ct.max() > 1.0 else ct * 255).astype(np.uint8)
        
        if self.geometric_transform is not None:
            # Stack for joint transformation
            stacked = np.stack([mr_uint8, ct_uint8], axis=-1)
            stacked_pil = Image.fromarray(stacked.astype(np.uint8))
            
            # Apply transforms
            stacked_transformed = self.geometric_transform(stacked_pil)
            
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
        else:
            # No transforms, just convert to tensor
            mr_out = torch.from_numpy(mr_uint8).unsqueeze(0).float()
            ct_out = torch.from_numpy(ct_uint8).unsqueeze(0).float()
        
        # Apply cutout to MR only (basic version)
        mr_out = self._apply_cutout(mr_out)
        
        return mr_out, ct_out


def get_medical_augmentation(
    mode: str = 'train',
    # Geometric transforms
    rotation_degrees: Tuple[float, float] = (-15, 15),
    enable_flip: bool = True,
    enable_elastic: bool = True,
    zoom_range: Tuple[float, float] = (0.9, 1.1),
    enable_grid_distortion: bool = False,
    # MR-only transforms
    enable_bias_field: bool = False,
    enable_motion_ghosting: bool = False,
    enable_rician_noise: bool = False,
    enable_gamma: bool = False,
    enable_cutout: bool = False,
    # General
    use_torchio: bool = True
) -> Optional[PairedMedicalAugmentation]:
    """
    Get medical image augmentation pipeline.
    
    Args:
        mode: 'train' or 'test'. Returns None for test mode.
        Geometric transforms (applied to both MR and CT):
            rotation_degrees: Range for random rotation in degrees
            enable_flip: Whether to enable random horizontal flip
            enable_elastic: Whether to enable elastic deformation
            zoom_range: Range for random zoom/scaling
            enable_grid_distortion: Whether to enable grid distortion
        MR-only transforms:
            enable_bias_field: Whether to simulate bias field
            enable_motion_ghosting: Whether to simulate motion artifacts
            enable_rician_noise: Whether to add Rician noise
            enable_gamma: Whether to apply gamma correction
            enable_cutout: Whether to apply random cutout/erasing
        General:
            use_torchio: Whether to use TorchIO (if available)
    
    Returns:
        PairedMedicalAugmentation instance for training, None for testing
    """
    if mode == 'test':
        return None
    
    return PairedMedicalAugmentation(
        # Geometric transforms
        rotation_degrees=rotation_degrees,
        enable_flip=enable_flip,
        enable_elastic=enable_elastic,
        zoom_range=zoom_range,
        enable_grid_distortion=enable_grid_distortion,
        # MR-only transforms
        enable_bias_field=enable_bias_field,
        enable_motion_ghosting=enable_motion_ghosting,
        enable_rician_noise=enable_rician_noise,
        enable_gamma=enable_gamma,
        enable_cutout=enable_cutout,
        # General
        use_torchio=use_torchio
    )
