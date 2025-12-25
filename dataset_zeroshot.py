"""
Zero-Shot MR-to-CT Dataset Loaders
Training: Only CT images with precomputed HOG features
Inference: Only MR images with precomputed HOG features

HOG features are computed in the dataset to avoid online computation overhead.
"""

import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import numpy as np
from hog_utils import HOGExtractor


class CTOnlyDataset(Dataset):
    """
    Dataset for zero-shot MR-to-CT training.
    Loads only CT images and computes HOG features for conditioning.
    
    Directory structure:
    dataset/
      ct/
        train/
        test/
    """
    
    def __init__(self, root_dir, split='train', ct_mean=0.5, ct_std=0.5, 
                 transform=None, augmentation=None, img_size=256,
                 hog_cell_size=8, hog_block_size=2, hog_num_bins=9):
        """
        Args:
            root_dir: Root directory containing 'ct' folder
            split: 'train' or 'test'
            ct_mean: Mean for CT z-score normalization
            ct_std: Std for CT z-score normalization
            transform: Optional basic transform to apply (e.g., cropping)
            augmentation: Optional augmentation (for training)
            img_size: Target image size
            hog_cell_size: HOG cell size in pixels
            hog_block_size: HOG block size in cells
            hog_num_bins: Number of HOG orientation bins
        """
        self.root_dir = root_dir
        self.split = split
        self.ct_mean = ct_mean
        self.ct_std = ct_std
        self.transform = transform
        self.augmentation = augmentation
        self.img_size = img_size
        
        # Initialize HOG extractor
        self.hog_extractor = HOGExtractor(
            cell_size=hog_cell_size,
            block_size=hog_block_size,
            num_bins=hog_num_bins
        )
        
        # Build path to CT directory
        self.ct_dir = os.path.join(root_dir, 'ct', split)
        
        # Verify directory exists
        if not os.path.exists(self.ct_dir):
            raise ValueError(f"CT directory not found: {self.ct_dir}")
        
        # Get list of image files
        self.ct_files = sorted([f for f in os.listdir(self.ct_dir) 
                               if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png')])
        
        print(f"Loaded {len(self.ct_files)} CT images from {split} split for zero-shot training")
        print(f"HOG config: cell_size={hog_cell_size}, block_size={hog_block_size}, num_bins={hog_num_bins}")
    
    def __len__(self):
        return len(self.ct_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            ct: CT image tensor (1, H, W) normalized with z-score
            hog: HOG features tensor (seq_len, feature_dim) for cross-attention
        """
        # Load CT image
        ct_path = os.path.join(self.ct_dir, self.ct_files[idx])
        ct_image = Image.open(ct_path).convert('L')  # Convert to grayscale
        ct = np.array(ct_image, dtype=np.float32)
        
        # Apply basic transforms if provided (e.g., cropping)
        if self.transform is not None:
            ct_uint8 = ct.astype(np.uint8) if ct.max() > 1.0 else (ct * 255).astype(np.uint8)
            ct_pil = Image.fromarray(ct_uint8, mode='L')
            ct_transformed = self.transform(ct_pil)
            if isinstance(ct_transformed, torch.Tensor):
                ct = ct_transformed.squeeze(0).numpy() if ct_transformed.dim() == 3 else ct_transformed.numpy()
            else:
                ct = np.array(ct_transformed, dtype=np.float32)
        
        # Apply augmentation if provided
        if self.augmentation is not None:
            ct = self.augmentation(ct)
            if not isinstance(ct, torch.Tensor):
                ct = torch.from_numpy(ct).unsqueeze(0).float()
        else:
            ct = torch.from_numpy(ct).unsqueeze(0).float()
        
        # Normalize to [0, 1] range first if values are in [0, 255]
        if ct.max() > 1.0:
            ct = ct / 255.0
        
        # Compute HOG features BEFORE z-score normalization (HOG needs [0,1] range)
        ct_for_hog = ct.clone()
        if ct_for_hog.dim() == 3:
            ct_for_hog = ct_for_hog.unsqueeze(0)  # Add batch dim
        with torch.no_grad():
            hog_features = self.hog_extractor(ct_for_hog)
            hog_features = hog_features.squeeze(0)  # Remove batch dim
        
        # Apply z-score normalization to CT
        ct = (ct - self.ct_mean) / self.ct_std
        
        return ct, hog_features


class MROnlyDataset(Dataset):
    """
    Dataset for zero-shot MR-to-CT inference.
    Loads only MR images and computes HOG features for conditioning.
    
    Directory structure:
    dataset/
      mr/
        test/
    """
    
    def __init__(self, root_dir, split='test', mr_mean=0.5, mr_std=0.5,
                 ct_mean=0.5, ct_std=0.5, transform=None, img_size=256,
                 hog_cell_size=8, hog_block_size=2, hog_num_bins=9):
        """
        Args:
            root_dir: Root directory containing 'mr' folder
            split: 'train' or 'test'
            mr_mean: Mean for MR z-score normalization
            mr_std: Std for MR z-score normalization
            ct_mean: Mean for CT normalization (used for ground truth if available)
            ct_std: Std for CT normalization (used for ground truth if available)
            transform: Optional basic transform to apply (e.g., cropping)
            img_size: Target image size
            hog_cell_size: HOG cell size in pixels
            hog_block_size: HOG block size in cells
            hog_num_bins: Number of HOG orientation bins
        """
        self.root_dir = root_dir
        self.split = split
        self.mr_mean = mr_mean
        self.mr_std = mr_std
        self.ct_mean = ct_mean
        self.ct_std = ct_std
        self.transform = transform
        self.img_size = img_size
        
        # Initialize HOG extractor
        self.hog_extractor = HOGExtractor(
            cell_size=hog_cell_size,
            block_size=hog_block_size,
            num_bins=hog_num_bins
        )
        
        # Build paths
        self.mr_dir = os.path.join(root_dir, 'mr', split)
        self.ct_dir = os.path.join(root_dir, 'ct', split)  # For ground truth comparison
        
        # Verify MR directory exists
        if not os.path.exists(self.mr_dir):
            raise ValueError(f"MR directory not found: {self.mr_dir}")
        
        # Check if CT directory exists (optional for ground truth)
        self.has_ct = os.path.exists(self.ct_dir)
        
        # Get list of image files
        self.mr_files = sorted([f for f in os.listdir(self.mr_dir) 
                               if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png')])
        
        if self.has_ct:
            self.ct_files = sorted([f for f in os.listdir(self.ct_dir) 
                                   if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png')])
        
        print(f"Loaded {len(self.mr_files)} MR images from {split} split for zero-shot inference")
        print(f"CT ground truth available: {self.has_ct}")
        print(f"HOG config: cell_size={hog_cell_size}, block_size={hog_block_size}, num_bins={hog_num_bins}")
    
    def __len__(self):
        return len(self.mr_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            mr: MR image tensor (1, H, W) normalized with z-score
            hog: HOG features tensor (seq_len, feature_dim) for cross-attention
            ct: CT ground truth tensor (1, H, W) if available, else None
        """
        # Load MR image
        mr_path = os.path.join(self.mr_dir, self.mr_files[idx])
        mr_image = Image.open(mr_path).convert('L')
        mr = np.array(mr_image, dtype=np.float32)
        
        # Apply basic transforms if provided
        if self.transform is not None:
            mr_uint8 = mr.astype(np.uint8) if mr.max() > 1.0 else (mr * 255).astype(np.uint8)
            mr_pil = Image.fromarray(mr_uint8, mode='L')
            mr_transformed = self.transform(mr_pil)
            if isinstance(mr_transformed, torch.Tensor):
                mr = mr_transformed.squeeze(0).numpy() if mr_transformed.dim() == 3 else mr_transformed.numpy()
            else:
                mr = np.array(mr_transformed, dtype=np.float32)
        
        # Convert to tensor
        mr = torch.from_numpy(mr).unsqueeze(0).float()
        
        # Normalize to [0, 1] range first if values are in [0, 255]
        if mr.max() > 1.0:
            mr = mr / 255.0
        
        # Compute HOG features BEFORE z-score normalization
        mr_for_hog = mr.clone()
        if mr_for_hog.dim() == 3:
            mr_for_hog = mr_for_hog.unsqueeze(0)
        with torch.no_grad():
            hog_features = self.hog_extractor(mr_for_hog)
            hog_features = hog_features.squeeze(0)
        
        # Apply z-score normalization to MR
        mr = (mr - self.mr_mean) / self.mr_std
        
        # Load CT ground truth if available
        ct = None
        if self.has_ct:
            ct_path = os.path.join(self.ct_dir, self.ct_files[idx])
            if os.path.exists(ct_path):
                ct_image = Image.open(ct_path).convert('L')
                ct = np.array(ct_image, dtype=np.float32)
                
                if self.transform is not None:
                    ct_uint8 = ct.astype(np.uint8) if ct.max() > 1.0 else (ct * 255).astype(np.uint8)
                    ct_pil = Image.fromarray(ct_uint8, mode='L')
                    ct_transformed = self.transform(ct_pil)
                    if isinstance(ct_transformed, torch.Tensor):
                        ct = ct_transformed.squeeze(0).numpy() if ct_transformed.dim() == 3 else ct_transformed.numpy()
                    else:
                        ct = np.array(ct_transformed, dtype=np.float32)
                
                ct = torch.from_numpy(ct).unsqueeze(0).float()
                if ct.max() > 1.0:
                    ct = ct / 255.0
                ct = (ct - self.ct_mean) / self.ct_std
        
        return mr, hog_features, ct


class PairedMRCTDataset(Dataset):
    """
    Dataset for MR-to-CT training with MR HOG conditioning.
    Loads paired MR and CT images, uses MR HOG features for conditioning.
    
    This is used to diagnose whether the HOG conditioning approach works
    when trained with the correct MR-CT pairs (non zero-shot).
    
    Directory structure:
    dataset/
      mr/
        train/
      ct/
        train/
    """
    
    def __init__(self, root_dir, split='train', mr_mean=0.5, mr_std=0.5,
                 ct_mean=0.5, ct_std=0.5, transform=None, augmentation=None, 
                 img_size=256, hog_cell_size=8, hog_block_size=2, hog_num_bins=9):
        """
        Args:
            root_dir: Root directory containing 'mr' and 'ct' folders
            split: 'train' or 'test'
            mr_mean: Mean for MR z-score normalization
            mr_std: Std for MR z-score normalization
            ct_mean: Mean for CT z-score normalization
            ct_std: Std for CT z-score normalization
            transform: Optional basic transform to apply (e.g., cropping)
            augmentation: Optional augmentation (for training)
            img_size: Target image size
            hog_cell_size: HOG cell size in pixels
            hog_block_size: HOG block size in cells
            hog_num_bins: Number of HOG orientation bins
        """
        self.root_dir = root_dir
        self.split = split
        self.mr_mean = mr_mean
        self.mr_std = mr_std
        self.ct_mean = ct_mean
        self.ct_std = ct_std
        self.transform = transform
        self.augmentation = augmentation
        self.img_size = img_size
        
        # Initialize HOG extractor
        self.hog_extractor = HOGExtractor(
            cell_size=hog_cell_size,
            block_size=hog_block_size,
            num_bins=hog_num_bins
        )
        
        # Build paths
        self.mr_dir = os.path.join(root_dir, 'mr', split)
        self.ct_dir = os.path.join(root_dir, 'ct', split)
        
        # Verify directories exist
        if not os.path.exists(self.mr_dir):
            raise ValueError(f"MR directory not found: {self.mr_dir}")
        if not os.path.exists(self.ct_dir):
            raise ValueError(f"CT directory not found: {self.ct_dir}")
        
        # Get list of image files
        self.mr_files = sorted([f for f in os.listdir(self.mr_dir) 
                               if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png')])
        self.ct_files = sorted([f for f in os.listdir(self.ct_dir) 
                               if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png')])
        
        # Ensure paired data
        if len(self.mr_files) != len(self.ct_files):
            print(f"Warning: MR ({len(self.mr_files)}) and CT ({len(self.ct_files)}) file counts don't match!")
        
        print(f"Loaded {len(self.mr_files)} paired MR-CT images from {split} split")
        print(f"Training mode: Using MR HOG features to reconstruct CT")
        print(f"HOG config: cell_size={hog_cell_size}, block_size={hog_block_size}, num_bins={hog_num_bins}")
    
    def __len__(self):
        return min(len(self.mr_files), len(self.ct_files))
    
    def __getitem__(self, idx):
        """
        Returns:
            ct: CT image tensor (1, H, W) normalized with z-score (target)
            hog: HOG features tensor (seq_len, feature_dim) computed from MR (condition)
        """
        # Load MR image (for HOG features)
        mr_path = os.path.join(self.mr_dir, self.mr_files[idx])
        mr_image = Image.open(mr_path).convert('L')
        mr = np.array(mr_image, dtype=np.float32)
        
        # Load CT image (target)
        ct_path = os.path.join(self.ct_dir, self.ct_files[idx])
        ct_image = Image.open(ct_path).convert('L')
        ct = np.array(ct_image, dtype=np.float32)
        
        # Apply basic transforms if provided
        if self.transform is not None:
            # Transform MR
            mr_uint8 = mr.astype(np.uint8) if mr.max() > 1.0 else (mr * 255).astype(np.uint8)
            mr_pil = Image.fromarray(mr_uint8, mode='L')
            mr_transformed = self.transform(mr_pil)
            if isinstance(mr_transformed, torch.Tensor):
                mr = mr_transformed.squeeze(0).numpy() if mr_transformed.dim() == 3 else mr_transformed.numpy()
            else:
                mr = np.array(mr_transformed, dtype=np.float32)
            
            # Transform CT
            ct_uint8 = ct.astype(np.uint8) if ct.max() > 1.0 else (ct * 255).astype(np.uint8)
            ct_pil = Image.fromarray(ct_uint8, mode='L')
            ct_transformed = self.transform(ct_pil)
            if isinstance(ct_transformed, torch.Tensor):
                ct = ct_transformed.squeeze(0).numpy() if ct_transformed.dim() == 3 else ct_transformed.numpy()
            else:
                ct = np.array(ct_transformed, dtype=np.float32)
        
        # Convert to tensors
        mr = torch.from_numpy(mr).unsqueeze(0).float()
        ct = torch.from_numpy(ct).unsqueeze(0).float()
        
        # Normalize to [0, 1] range first if values are in [0, 255]
        if mr.max() > 1.0:
            mr = mr / 255.0
        if ct.max() > 1.0:
            ct = ct / 255.0
        
        # Compute HOG features from MR image (BEFORE z-score normalization)
        mr_for_hog = mr.clone()
        if mr_for_hog.dim() == 3:
            mr_for_hog = mr_for_hog.unsqueeze(0)
        with torch.no_grad():
            hog_features = self.hog_extractor(mr_for_hog)
            hog_features = hog_features.squeeze(0)
        
        # Apply z-score normalization to CT (target)
        ct = (ct - self.ct_mean) / self.ct_std
        
        return ct, hog_features


def get_zeroshot_train_dataloader(dataset_path, batch_size=16, num_workers=4,
                                   ct_mean=0.5, ct_std=0.5, img_size=256,
                                   distributed=False, enable_augmentation=True,
                                   hog_cell_size=8, hog_block_size=2, hog_num_bins=9):
    """
    Create dataloader for zero-shot CT-only training.
    
    Args:
        dataset_path: Path to dataset root directory
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        ct_mean, ct_std: Z-score normalization params for CT
        img_size: Target image size
        distributed: Whether to use distributed training
        enable_augmentation: Whether to enable augmentation
        hog_*: HOG parameters
    
    Returns:
        train_loader
    """
    from torchvision import transforms
    from util.crop import center_crop_arr
    
    # Define transforms for cropping
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
        transforms.ToTensor()
    ])
    
    # Create dataset
    train_dataset = CTOnlyDataset(
        root_dir=dataset_path,
        split='train',
        ct_mean=ct_mean,
        ct_std=ct_std,
        transform=transform,
        augmentation=None,  # Augmentation disabled for zero-shot approach to preserve HOG consistency
        img_size=img_size,
        hog_cell_size=hog_cell_size,
        hog_block_size=hog_block_size,
        hog_num_bins=hog_num_bins
    )
    
    # Create dataloader
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
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
    
    return train_loader


def get_zeroshot_test_dataloader(dataset_path, batch_size=16, num_workers=4,
                                  mr_mean=0.5, mr_std=0.5, ct_mean=0.5, ct_std=0.5,
                                  img_size=256, distributed=False,
                                  hog_cell_size=8, hog_block_size=2, hog_num_bins=9):
    """
    Create dataloader for zero-shot MR-to-CT inference.
    
    Args:
        dataset_path: Path to dataset root directory
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        mr_mean, mr_std: Z-score normalization params for MR
        ct_mean, ct_std: Z-score normalization params for CT (for ground truth)
        img_size: Target image size
        distributed: Whether to use distributed inference
        hog_*: HOG parameters
    
    Returns:
        test_loader
    """
    from torchvision import transforms
    from util.crop import center_crop_arr
    
    # Define transforms for cropping
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
        transforms.ToTensor()
    ])
    
    # Create dataset
    test_dataset = MROnlyDataset(
        root_dir=dataset_path,
        split='test',
        mr_mean=mr_mean,
        mr_std=mr_std,
        ct_mean=ct_mean,
        ct_std=ct_std,
        transform=transform,
        img_size=img_size,
        hog_cell_size=hog_cell_size,
        hog_block_size=hog_block_size,
        hog_num_bins=hog_num_bins
    )
    
    # Create dataloader
    if distributed:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    return test_loader


def get_paired_mrct_train_dataloader(dataset_path, batch_size=16, num_workers=4,
                                      mr_mean=0.5, mr_std=0.5, ct_mean=0.5, ct_std=0.5,
                                      img_size=256, distributed=False,
                                      hog_cell_size=8, hog_block_size=2, hog_num_bins=9):
    """
    Create dataloader for paired MR-CT training with MR HOG conditioning.
    
    This is used to diagnose whether the HOG conditioning approach works
    when trained with the correct MR-CT pairs (non zero-shot).
    
    Args:
        dataset_path: Path to dataset root directory
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        mr_mean, mr_std: Z-score normalization params for MR
        ct_mean, ct_std: Z-score normalization params for CT
        img_size: Target image size
        distributed: Whether to use distributed training
        hog_*: HOG parameters
    
    Returns:
        train_loader
    """
    from torchvision import transforms
    from util.crop import center_crop_arr
    
    # Define transforms for cropping
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
        transforms.ToTensor()
    ])
    
    # Create dataset
    train_dataset = PairedMRCTDataset(
        root_dir=dataset_path,
        split='train',
        mr_mean=mr_mean,
        mr_std=mr_std,
        ct_mean=ct_mean,
        ct_std=ct_std,
        transform=transform,
        augmentation=None,
        img_size=img_size,
        hog_cell_size=hog_cell_size,
        hog_block_size=hog_block_size,
        hog_num_bins=hog_num_bins
    )
    
    # Create dataloader
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
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
    
    return train_loader
