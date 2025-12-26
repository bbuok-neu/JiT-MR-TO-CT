"""
Zero-Shot MR-to-CT Dataset with MIND Features and Intensity Perturbation

This dataset is designed for ControlNet-based zero-shot synthesis:
- Training: Returns (CT, MIND(perturbed_CT)) pairs
- Testing: Returns (CT, MIND(MR)) pairs for zero-shot inference

IntensityPerturbation augmentation is applied ONLY to the source CT used for 
MIND computation, NOT to the target Ground Truth CT. This trains the model
to be robust to intensity variations, improving zero-shot transfer to MR.
"""

import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import numpy as np
import random

from mind import compute_mind_with_padding, get_mind_channels


class IntensityPerturbation:
    """
    Custom intensity perturbation augmentation for MIND robustness training.
    
    Applied ONLY to source CT (for MIND calculation), NOT to target GT CT.
    This teaches the model that MIND features should be invariant to intensity changes.
    
    Includes:
    - RandomInvert: Invert intensities (1 - x)
    - RandomGamma: Nonlinear contrast adjustment
    - RandomSolarize: Two modes for thresholded intensity modification
    """
    
    def __init__(self, 
                 invert_prob=0.3,
                 gamma_prob=0.3,
                 gamma_range=(0.5, 2.0),
                 solarize_prob=0.3,
                 solarize_threshold_range=(0.3, 0.7),
                 solarize_mode_a_prob=0.5):
        """
        Args:
            invert_prob: Probability of applying intensity inversion
            gamma_prob: Probability of applying gamma correction
            gamma_range: Range for gamma values (min, max)
            solarize_prob: Probability of applying solarization
            solarize_threshold_range: Range for solarization threshold
            solarize_mode_a_prob: Probability of using Mode A (Invert) vs Mode B (Zero)
        """
        self.invert_prob = invert_prob
        self.gamma_prob = gamma_prob
        self.gamma_range = gamma_range
        self.solarize_prob = solarize_prob
        self.solarize_threshold_range = solarize_threshold_range
        self.solarize_mode_a_prob = solarize_mode_a_prob
    
    def random_invert(self, image):
        """Invert intensities: 1 - x"""
        return 1.0 - image
    
    def random_gamma(self, image):
        """Apply random gamma correction"""
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        # Gamma correction: x^gamma (assumes image is in [0, 1])
        return torch.pow(torch.clamp(image, min=1e-8), gamma)
    
    def random_solarize(self, image, use_mode_a=True):
        """
        Apply random solarization.
        
        Mode A (Invert): For pixels > threshold, value = max - x (classic solarize)
        Mode B (Zero): For pixels > threshold, value = 0 (simulate signal void)
        
        Args:
            image: Input image tensor (C, H, W) in [0, 1]
            use_mode_a: If True, use Mode A (Invert), else use Mode B (Zero)
        
        Returns:
            Solarized image
        """
        threshold = random.uniform(self.solarize_threshold_range[0], 
                                   self.solarize_threshold_range[1])
        max_val = image.max()
        
        mask = image > threshold
        
        if use_mode_a:
            # Mode A: Invert high values
            result = image.clone()
            result[mask] = max_val - image[mask]
        else:
            # Mode B: Zero high values (signal void)
            result = image.clone()
            result[mask] = 0.0
        
        return result
    
    def __call__(self, image):
        """
        Apply random intensity perturbations to the input image.
        
        Args:
            image: Input image tensor (1, H, W) or (H, W) in [0, 1]
        
        Returns:
            Perturbed image tensor with same shape
        """
        # Ensure tensor format
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        # Ensure 3D format (C, H, W)
        if image.dim() == 2:
            image = image.unsqueeze(0)
        
        result = image.clone()
        
        # Apply random invert
        if random.random() < self.invert_prob:
            result = self.random_invert(result)
        
        # Apply random gamma
        if random.random() < self.gamma_prob:
            result = self.random_gamma(result)
        
        # Apply random solarize
        if random.random() < self.solarize_prob:
            use_mode_a = random.random() < self.solarize_mode_a_prob
            result = self.random_solarize(result, use_mode_a)
        
        return result


class ZeroShotMRCTDataset(Dataset):
    """
    Dataset for ControlNet-based zero-shot MR-to-CT synthesis.
    
    Two-stage training strategy:
    - Stage 1 (Base Model): Train on CT only, dataset returns (CT, CT)
    - Stage 2 (ControlNet): Train with MIND guidance, returns (CT, MIND(perturbed_CT))
    
    For testing: Returns (CT, MIND(MR)) for zero-shot inference
    
    All images are loaded as single-channel grayscale.
    """
    
    def __init__(self, root_dir, split='train', 
                 ct_mean=0.0, ct_std=1.0,
                 mr_mean=0.0, mr_std=1.0,
                 transform=None,
                 # MIND parameters
                 mind_patch_size=7, mind_neigh_size=7, mind_sigma=0.5, 
                 mind_eps=1e-6, mind_neigh4=False,
                 # Training mode
                 stage='stage1',  # 'stage1' (base) or 'stage2' (controlnet)
                 # Intensity perturbation params (Stage 2 only)
                 enable_perturbation=True,
                 invert_prob=0.3,
                 gamma_prob=0.3,
                 gamma_range=(0.5, 2.0),
                 solarize_prob=0.3,
                 solarize_threshold_range=(0.3, 0.7),
                 solarize_mode_a_prob=0.5):
        """
        Args:
            root_dir: Root directory containing 'mr' and 'ct' folders
            split: 'train' or 'test'
            ct_mean, ct_std: Z-score normalization params for CT
            mr_mean, mr_std: Z-score normalization params for MR
            transform: Optional transform for cropping/resizing
            mind_*: MIND descriptor parameters
            stage: Training stage ('stage1' for base model, 'stage2' for controlnet)
            enable_perturbation: Enable intensity perturbation (Stage 2 only)
            *_prob: Probabilities for various perturbations
        """
        self.root_dir = root_dir
        self.split = split
        self.ct_mean = ct_mean
        self.ct_std = ct_std
        self.mr_mean = mr_mean
        self.mr_std = mr_std
        self.transform = transform
        self.stage = stage
        
        # MIND parameters
        self.mind_patch_size = mind_patch_size
        self.mind_neigh_size = mind_neigh_size
        self.mind_sigma = mind_sigma
        self.mind_eps = mind_eps
        self.mind_neigh4 = mind_neigh4
        
        # Intensity perturbation (Stage 2 only)
        self.perturbation = None
        if stage == 'stage2' and split == 'train' and enable_perturbation:
            self.perturbation = IntensityPerturbation(
                invert_prob=invert_prob,
                gamma_prob=gamma_prob,
                gamma_range=gamma_range,
                solarize_prob=solarize_prob,
                solarize_threshold_range=solarize_threshold_range,
                solarize_mode_a_prob=solarize_mode_a_prob
            )
        
        # Build paths
        self.ct_dir = os.path.join(root_dir, 'ct', split)
        self.mr_dir = os.path.join(root_dir, 'mr', split)
        
        # Verify CT directory exists
        if not os.path.exists(self.ct_dir):
            raise ValueError(f"CT directory not found: {self.ct_dir}")
        
        # Get CT files
        self.ct_files = sorted([f for f in os.listdir(self.ct_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # For Stage 2 testing, also need MR files (for zero-shot MIND(MR) evaluation)
        # Stage 1 test does not need MR files as it's unconditional
        self.mr_files = None
        if split == 'test' and stage == 'stage2':
            if not os.path.exists(self.mr_dir):
                raise ValueError(f"MR directory not found: {self.mr_dir}")
            self.mr_files = sorted([f for f in os.listdir(self.mr_dir) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if len(self.ct_files) != len(self.mr_files):
                raise ValueError(f"Number of CT ({len(self.ct_files)}) and MR ({len(self.mr_files)}) images must match")
        
        mind_channels = get_mind_channels(mind_neigh_size, mind_neigh4)
        print(f"[{split.upper()}] Loaded {len(self.ct_files)} images, Stage: {stage}")
        if stage == 'stage2':
            print(f"  MIND features: {mind_channels} channels")
            if self.perturbation:
                print(f"  Intensity perturbation: Invert={invert_prob:.1f}, Gamma={gamma_prob:.1f}, Solarize={solarize_prob:.1f}")
    
    def __len__(self):
        return len(self.ct_files)
    
    def _load_image(self, path):
        """Load image as grayscale tensor normalized to [0, 1]"""
        image = Image.open(path).convert('L')
        image = np.array(image, dtype=np.float32)
        
        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        # Ensure 3D (C, H, W)
        if image.dim() == 2:
            image = image.unsqueeze(0)
        
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        return image
    
    def _compute_mind(self, image_tensor):
        """Compute MIND features for an image tensor."""
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
        Returns depend on stage and split:
        
        Stage 1 (Base Model) - Train on CT only:
            Returns: (ct, None) - CT image, no condition
        
        Stage 2 (ControlNet) - Train with MIND:
            Train: Returns (ct, mind_features) where mind = MIND(perturbed_ct)
            Test:  Returns (ct, mind_features) where mind = MIND(mr)
        """
        # Load CT image
        ct_path = os.path.join(self.ct_dir, self.ct_files[idx])
        ct = self._load_image(ct_path)
        
        # Apply z-score normalization to CT (target)
        ct_normalized = (ct - self.ct_mean) / self.ct_std
        
        # Stage 1: Base model training - return only CT
        if self.stage == 'stage1':
            return ct_normalized, None
        
        # Stage 2: ControlNet training with MIND condition
        if self.split == 'train':
            # Training: Compute MIND from perturbed CT
            ct_for_mind = ct.clone()
            
            # Apply intensity perturbation to source CT (NOT to target GT)
            if self.perturbation is not None:
                ct_for_mind = self.perturbation(ct_for_mind)
            
            # Compute MIND features
            mind_features = self._compute_mind(ct_for_mind)
        else:
            # Testing: Compute MIND from MR (zero-shot)
            mr_path = os.path.join(self.mr_dir, self.mr_files[idx])
            mr = self._load_image(mr_path)
            mind_features = self._compute_mind(mr)
        
        return ct_normalized, mind_features


def get_zeroshot_dataloaders(dataset_path, batch_size=16, num_workers=4,
                             ct_mean=0.5, ct_std=0.5,
                             mr_mean=0.5, mr_std=0.5,
                             img_size=256, distributed=False,
                             # MIND parameters
                             mind_patch_size=7, mind_neigh_size=7, mind_sigma=0.5,
                             mind_eps=1e-6, mind_neigh4=False,
                             # Training stage
                             stage='stage1',
                             # Intensity perturbation
                             enable_perturbation=True,
                             invert_prob=0.3,
                             gamma_prob=0.3,
                             gamma_range=(0.5, 2.0),
                             solarize_prob=0.3,
                             solarize_threshold_range=(0.3, 0.7),
                             solarize_mode_a_prob=0.5):
    """
    Create dataloaders for zero-shot MR-to-CT synthesis.
    
    Args:
        dataset_path: Path to dataset root
        batch_size, num_workers: Dataloader params
        ct_mean, ct_std: CT normalization params
        mr_mean, mr_std: MR normalization params
        img_size: Target image size
        distributed: Use distributed sampler
        mind_*: MIND descriptor parameters
        stage: 'stage1' (base model) or 'stage2' (controlnet)
        *_prob: Intensity perturbation probabilities
    
    Returns:
        train_loader, test_loader
    """
    from torchvision import transforms
    import cv2 as cv2_resize
    
    # Resize transform instead of center crop to preserve full image content
    def resize_image(img, size):
        if isinstance(img, np.ndarray):
            # Use cv2 for high-quality resize
            return cv2_resize.resize(img, (size, size), interpolation=cv2_resize.INTER_LINEAR)
        return img
    
    transform = lambda img: resize_image(img, img_size)
    
    # Create datasets
    train_dataset = ZeroShotMRCTDataset(
        dataset_path,
        split='train',
        ct_mean=ct_mean,
        ct_std=ct_std,
        mr_mean=mr_mean,
        mr_std=mr_std,
        transform=transform,
        mind_patch_size=mind_patch_size,
        mind_neigh_size=mind_neigh_size,
        mind_sigma=mind_sigma,
        mind_eps=mind_eps,
        mind_neigh4=mind_neigh4,
        stage=stage,
        enable_perturbation=enable_perturbation,
        invert_prob=invert_prob,
        gamma_prob=gamma_prob,
        gamma_range=gamma_range,
        solarize_prob=solarize_prob,
        solarize_threshold_range=solarize_threshold_range,
        solarize_mode_a_prob=solarize_mode_a_prob
    )
    
    test_dataset = ZeroShotMRCTDataset(
        dataset_path,
        split='test',
        ct_mean=ct_mean,
        ct_std=ct_std,
        mr_mean=mr_mean,
        mr_std=mr_std,
        transform=transform,
        mind_patch_size=mind_patch_size,
        mind_neigh_size=mind_neigh_size,
        mind_sigma=mind_sigma,
        mind_eps=mind_eps,
        mind_neigh4=mind_neigh4,
        stage=stage,  # Pass the same stage; test dataset computes MIND from MR
        enable_perturbation=False  # No perturbation for test
    )
    
    # Create dataloaders
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle_train = True
    
    def collate_fn(batch):
        """Custom collate to handle None conditions in Stage 1"""
        cts = torch.stack([item[0] for item in batch])
        conds = [item[1] for item in batch]
        
        if conds[0] is None:
            return cts, None
        else:
            return cts, torch.stack(conds)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader
