"""
Calculate mean and standard deviation for MR and CT images
Run this on your training data to get normalization parameters
"""
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm


def calculate_stats(image_folder):
    """
    Calculate mean and std for images in a folder
    """
    print(f"Calculating statistics for: {image_folder}")
    
    # Get all image files
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print(f"Warning: No images found in {image_folder}")
        return None, None
    
    print(f"Found {len(image_files)} images")
    
    # Collect all pixel values
    all_values = []
    
    for filename in tqdm(image_files, desc="Loading images"):
        img_path = os.path.join(image_folder, filename)
        try:
            # Load as grayscale
            img = Image.open(img_path).convert('L')
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            all_values.append(img_array.flatten())
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    if len(all_values) == 0:
        print("Error: Could not load any images")
        return None, None
    
    # Concatenate all values
    all_values = np.concatenate(all_values)
    
    # Calculate statistics
    mean = np.mean(all_values)
    std = np.std(all_values)
    
    print(f"  Mean: {mean:.6f}")
    print(f"  Std:  {std:.6f}")
    print(f"  Min:  {np.min(all_values):.6f}")
    print(f"  Max:  {np.max(all_values):.6f}")
    
    return mean, std


def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_stats.py <dataset_path>")
        print("Example: python calculate_stats.py /path/to/dataset")
        print("\nDataset should have structure:")
        print("  dataset/")
        print("    mr/train/")
        print("    ct/train/")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    print("="*70)
    print("Calculating Normalization Statistics")
    print("="*70)
    
    # Calculate MR statistics
    mr_train_path = os.path.join(dataset_path, 'mr', 'train')
    if not os.path.exists(mr_train_path):
        print(f"Error: MR training folder not found: {mr_train_path}")
        sys.exit(1)
    
    print("\n1. MR Images")
    print("-" * 70)
    mr_mean, mr_std = calculate_stats(mr_train_path)
    
    if mr_mean is None:
        print("Failed to calculate MR statistics")
        sys.exit(1)
    
    # Calculate CT statistics
    ct_train_path = os.path.join(dataset_path, 'ct', 'train')
    if not os.path.exists(ct_train_path):
        print(f"Error: CT training folder not found: {ct_train_path}")
        sys.exit(1)
    
    print("\n2. CT Images")
    print("-" * 70)
    ct_mean, ct_std = calculate_stats(ct_train_path)
    
    if ct_mean is None:
        print("Failed to calculate CT statistics")
        sys.exit(1)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - Use these values in your training command:")
    print("="*70)
    print(f"--mr_mean {mr_mean:.6f} --mr_std {mr_std:.6f}")
    print(f"--ct_mean {ct_mean:.6f} --ct_std {ct_std:.6f}")
    print("\nFull training command example:")
    print(f"""
python main_mrct.py \\
    --data_path {dataset_path} \\
    --mr_mean {mr_mean:.6f} --mr_std {mr_std:.6f} \\
    --ct_mean {ct_mean:.6f} --ct_std {ct_std:.6f} \\
    --model JiT-B/16 \\
    --img_size 256 \\
    --batch_size 16 \\
    --epochs 200 \\
    --online_eval --eval_freq 10 \\
    --output_dir ./output_mrct
    """)
    
    print("="*70)


if __name__ == '__main__':
    main()
