"""
Paired MR-CT Dataset Loader for Medical Image Synthesis with MIND Features

For zero-shot MR-to-CT synthesis using MIND (Modality Independent Neighbourhood Descriptor):
- Training: Uses MIND(CT) as condition concatenated with noisy CT
- Inference: Uses MIND(MR) as condition for zero-shot synthesis

Loads paired MR and CT images from directory structure:
dataset/
  mr/
    train/
    test/
  ct/
    train/
    test/
"""

import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import numpy as np

from mind import compute_mind_with_padding, get_mind_channels


class PairedMRCTDataset(Dataset):
    """
    Dataset for paired MR-CT medical images with MIND features.
    
    For training: Returns (MIND(CT), CT) - uses CT's MIND as condition
    For testing: Returns (MIND(MR), CT) - uses MR's MIND for zero-shot inference
    
    All images are loaded as single-channel grayscale JPG.
    Z-score normalization is applied with preset mean and std.
    """
    
    def __init__(self, root_dir, split='train', mr_mean=0.0, mr_std=1.0, 
                 ct_mean=0.0, ct_std=1.0, transform=None, augmentation=None,
                 mind_patch_size=7, mind_neigh_size=7, mind_sigma=0.5, 
                 mind_eps=1e-6, mind_neigh4=False):
        """
        Args:
            root_dir: Root directory containing 'mr' and 'ct' folders
            split: 'train' or 'test'
            mr_mean: Mean for MR z-score normalization
            mr_std: Std for MR z-score normalization
            ct_mean: Mean for CT z-score normalization
            ct_std: Std for CT z-score normalization
            transform: Optional basic transform to apply to images (e.g., cropping)
            augmentation: Optional medical augmentation (PairedMedicalAugmentation instance)
            mind_patch_size: MIND Gaussian patch size
            mind_neigh_size: MIND neighborhood size
            mind_sigma: MIND Gaussian kernel sigma
            mind_eps: MIND numerical stability epsilon
            mind_neigh4: Use 4-connectivity for MIND
        """
        self.root_dir = root_dir
        self.split = split
        self.mr_mean = mr_mean
        self.mr_std = mr_std
        self.ct_mean = ct_mean
        self.ct_std = ct_std
        self.transform = transform
        self.augmentation = augmentation
        
        # MIND parameters
        self.mind_patch_size = mind_patch_size
        self.mind_neigh_size = mind_neigh_size
        self.mind_sigma = mind_sigma
        self.mind_eps = mind_eps
        self.mind_neigh4 = mind_neigh4
        
        # Build paths to MR and CT directories
        self.mr_dir = os.path.join(root_dir, 'mr', split)
        self.ct_dir = os.path.join(root_dir, 'ct', split)
        
        # Verify directories exist
        if not os.path.exists(self.mr_dir):
            raise ValueError(f"MR directory not found: {self.mr_dir}")
        if not os.path.exists(self.ct_dir):
            raise ValueError(f"CT directory not found: {self.ct_dir}")
        
        # Get list of image files (assuming paired images have same names)
        self.mr_files = sorted([f for f in os.listdir(self.mr_dir) 
                               if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')])
        self.ct_files = sorted([f for f in os.listdir(self.ct_dir) 
                               if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')])
        
        # Verify that MR and CT have same number of files
        if len(self.mr_files) != len(self.ct_files):
            raise ValueError(f"Number of MR images ({len(self.mr_files)}) does not match "
                           f"CT images ({len(self.ct_files)})")
        
        mind_channels = get_mind_channels(mind_neigh_size, mind_neigh4)
        print(f"Loaded {len(self.mr_files)} paired MR-CT images from {split} split")
        print(f"MIND features: {mind_channels} channels (patch={mind_patch_size}, neigh={mind_neigh_size}, neigh4={mind_neigh4})")
    
    def __len__(self):
        return len(self.mr_files)
    
    def _compute_mind(self, image_tensor):
        """
        Compute MIND features for an image tensor.
        
        Args:
            image_tensor: Input image (1, H, W)
        
        Returns:
            MIND features (C, H, W) padded to original size
        """
        # Add batch dimension
        image_batch = image_tensor.unsqueeze(0)  # (1, 1, H, W)
        
        # Compute MIND with padding
        mind_features = compute_mind_with_padding(
            image_batch,
            patch_size=self.mind_patch_size,
            neigh_size=self.mind_neigh_size,
            sigma=self.mind_sigma,
            eps=self.mind_eps,
            neigh4=self.mind_neigh4
        )
        
        # Remove batch dimension
        return mind_features.squeeze(0)  # (C, H, W)
    
    def __getitem__(self, idx):
        """
        Returns:
            For training: (mind_ct, ct) where mind_ct is MIND(CT) features
            For testing: (mind_mr, ct) where mind_mr is MIND(MR) features for zero-shot inference
            
            mind: MIND features tensor (C, H, W) where C = neigh_size^2-1 or 4
            ct: CT image tensor (1, H, W) normalized with z-score
        """
        # Load MR image
        mr_path = os.path.join(self.mr_dir, self.mr_files[idx])
        mr_image = Image.open(mr_path).convert('L')  # Convert to grayscale
        mr = np.array(mr_image, dtype=np.float32)
        
        # Load CT image
        ct_path = os.path.join(self.ct_dir, self.ct_files[idx])
        ct_image = Image.open(ct_path).convert('L')  # Convert to grayscale
        ct = np.array(ct_image, dtype=np.float32)
        
        # Verify shapes match
        if mr.shape != ct.shape:
            raise ValueError(f"MR shape {mr.shape} does not match CT shape {ct.shape} "
                           f"for file {self.mr_files[idx]}")
        
        # Apply basic transforms if provided (e.g., cropping)
        # These are applied before medical augmentation
        if self.transform is not None:
            # Stack for joint transformation
            # Scale to 0-255 range for PIL Image if needed
            if mr.max() <= 1.0:
                mr_uint8 = (mr * 255).astype(np.uint8)
                ct_uint8 = (ct * 255).astype(np.uint8)
            else:
                mr_uint8 = mr.astype(np.uint8)
                ct_uint8 = ct.astype(np.uint8)
            
            stacked = np.stack([mr_uint8, ct_uint8], axis=-1)
            stacked_pil = Image.fromarray(stacked.astype(np.uint8))
            stacked_transformed = self.transform(stacked_pil)
            
            # If transform returns tensor, split channels
            if isinstance(stacked_transformed, torch.Tensor):
                if stacked_transformed.shape[0] == 2:
                    mr = stacked_transformed[0].numpy()  # Remove channel dim for augmentation
                    ct = stacked_transformed[1].numpy()
                else:
                    # Handle case where transform outputs different format
                    mr = stacked_transformed[..., 0].numpy()
                    ct = stacked_transformed[..., 1].numpy()
            else:
                # If transform returns numpy array
                mr = stacked_transformed[..., 0]
                ct = stacked_transformed[..., 1]
        
        # Apply medical augmentation if provided (rotation, elastic, zoom, etc.)
        if self.augmentation is not None:
            mr, ct = self.augmentation(mr, ct)
        else:
            # No augmentation, just convert to tensor
            mr = torch.from_numpy(mr).unsqueeze(0).float()
            ct = torch.from_numpy(ct).unsqueeze(0).float()
        
        # Normalize to [0, 1] range first if values are in [0, 255]
        if mr.max() > 1.0:
            mr = mr / 255.0
        if ct.max() > 1.0:
            ct = ct / 255.0
        
        # Apply z-score normalization to CT
        ct_normalized = (ct - self.ct_mean) / self.ct_std
        
        # Compute MIND features:
        # - Training: use MIND(CT) as condition
        # - Testing: use MIND(MR) for zero-shot inference
        #
        # CRITICAL: MIND features are computed on [0,1] normalized images (NOT z-score)
        # because MIND is a structural descriptor that relies on local intensity differences.
        # Using [0,1] normalization ensures:
        # 1. Both CT and MR have the same intensity range for MIND computation
        # 2. MIND(CT) ≈ MIND(MR) property holds, enabling zero-shot cross-modality transfer
        # 3. Z-score normalization would introduce modality-specific mean/std bias
        if self.split == 'train':
            # Use CT for MIND features during training
            mind_features = self._compute_mind(ct)
        else:
            # Use MR for MIND features during testing (zero-shot)
            mind_features = self._compute_mind(mr)
        
        return mind_features, ct_normalized


def get_mrct_dataloaders(dataset_path, batch_size=16, num_workers=4, 
                         mr_mean=0.5, mr_std=0.5, ct_mean=0.5, ct_std=0.5,
                         img_size=256, distributed=False, 
                         enable_augmentation=True, use_torchio=True,
                         # Geometric augmentations
                         rotation_degrees=(-15, 15), enable_flip=True,
                         enable_elastic=True, zoom_range=(0.9, 1.1),
                         enable_grid_distortion=False,
                         # MR-only augmentations
                         enable_bias_field=False, enable_motion_ghosting=False,
                         enable_rician_noise=False, enable_gamma=False,
                         enable_cutout=False,
                         # MIND parameters
                         mind_patch_size=7, mind_neigh_size=7, mind_sigma=0.5,
                         mind_eps=1e-6, mind_neigh4=False):
    """
    Create dataloaders for MR-CT paired dataset with MIND features.
    
    For zero-shot MR-to-CT synthesis:
    - Training: Returns (MIND(CT), CT) pairs
    - Testing: Returns (MIND(MR), CT) pairs for zero-shot inference
    
    Args:
        dataset_path: Path to dataset root directory
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        mr_mean, mr_std: Z-score normalization params for MR
        ct_mean, ct_std: Z-score normalization params for CT
        img_size: Target image size (will be center cropped)
        distributed: Whether to use distributed training
        enable_augmentation: Whether to enable medical image augmentation for training
        use_torchio: Whether to use TorchIO for augmentation (if available)
        
        Geometric augmentations (applied to both MR and CT):
            rotation_degrees: Range for random rotation in degrees (min, max)
            enable_flip: Whether to enable random horizontal flip
            enable_elastic: Whether to enable elastic deformation
            zoom_range: Range for random zoom/scaling (min, max)
            enable_grid_distortion: Whether to enable grid distortion
        
        MR-only augmentations:
            enable_bias_field: Whether to simulate bias field inhomogeneity
            enable_motion_ghosting: Whether to simulate motion artifacts
            enable_rician_noise: Whether to add Rician noise
            enable_gamma: Whether to apply gamma correction
            enable_cutout: Whether to apply random cutout/erasing
        
        MIND parameters:
            mind_patch_size: MIND Gaussian patch size
            mind_neigh_size: MIND neighborhood size (output channels = neigh_size^2 - 1)
            mind_sigma: MIND Gaussian kernel sigma
            mind_eps: MIND numerical stability epsilon
            mind_neigh4: Use 4-connectivity (4 channels) instead of full neighborhood
    
    Returns:
        train_loader, test_loader
    """
    from torchvision import transforms
    from util.crop import center_crop_arr
    from augmentations_mrct import get_medical_augmentation
    
    # Define basic transforms for cropping (applied before augmentation)
    # Note: We no longer include RandomHorizontalFlip here as it's in medical augmentation
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
        transforms.PILToTensor()
    ])
    
    # Define transforms for testing (no augmentation)
    transform_test = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
        transforms.PILToTensor()
    ])
    
    # Get medical augmentation for training
    train_augmentation = None
    if enable_augmentation:
        train_augmentation = get_medical_augmentation(
            mode='train',
            # Geometric
            rotation_degrees=rotation_degrees,
            enable_flip=enable_flip,
            enable_elastic=enable_elastic,
            zoom_range=zoom_range,
            enable_grid_distortion=enable_grid_distortion,
            # MR-only
            enable_bias_field=enable_bias_field,
            enable_motion_ghosting=enable_motion_ghosting,
            enable_rician_noise=enable_rician_noise,
            enable_gamma=enable_gamma,
            enable_cutout=enable_cutout,
            # General
            use_torchio=use_torchio
        )
        if train_augmentation is not None:
            print(f"Medical augmentation enabled:")
            print(f"  Geometric (MR+CT): Rotation={rotation_degrees}°, Flip={enable_flip}, Elastic={enable_elastic}, Zoom={zoom_range}, Grid={enable_grid_distortion}")
            print(f"  MR-only: BiasField={enable_bias_field}, Motion={enable_motion_ghosting}, Noise={enable_rician_noise}, Gamma={enable_gamma}, Cutout={enable_cutout}")
            print(f"  Using: {'TorchIO' if use_torchio and train_augmentation.use_torchio else 'Basic transforms'}")
    
    # Create datasets
    train_dataset = PairedMRCTDataset(
        dataset_path, 
        split='train',
        mr_mean=mr_mean,
        mr_std=mr_std,
        ct_mean=ct_mean,
        ct_std=ct_std,
        transform=transform_train,
        augmentation=train_augmentation,
        mind_patch_size=mind_patch_size,
        mind_neigh_size=mind_neigh_size,
        mind_sigma=mind_sigma,
        mind_eps=mind_eps,
        mind_neigh4=mind_neigh4
    )
    
    test_dataset = PairedMRCTDataset(
        dataset_path,
        split='test',
        mr_mean=mr_mean,
        mr_std=mr_std,
        ct_mean=ct_mean,
        ct_std=ct_std,
        transform=transform_test,
        augmentation=None,  # No augmentation for test
        mind_patch_size=mind_patch_size,
        mind_neigh_size=mind_neigh_size,
        mind_sigma=mind_sigma,
        mind_eps=mind_eps,
        mind_neigh4=mind_neigh4
    )
    
    # Create dataloaders
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    return train_loader, test_loader
