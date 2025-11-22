"""
Test script to verify MR-CT synthesis implementation
Creates dummy data and tests the pipeline
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_mrct import PairedMRCTDataset, get_mrct_dataloaders
from model_mrct import JiT_MRCT_models
from denoiser_mrct import Denoiser_MRCT
from metrics import calculate_psnr, calculate_ssim, evaluate_metrics


def create_dummy_dataset(root_dir, num_train=10, num_test=3, img_size=256):
    """
    Create a dummy dataset for testing
    """
    print("Creating dummy dataset...")
    
    # Create directory structure
    for split in ['train', 'test']:
        os.makedirs(os.path.join(root_dir, 'mr', split), exist_ok=True)
        os.makedirs(os.path.join(root_dir, 'ct', split), exist_ok=True)
    
    # Create dummy images
    num_images = {'train': num_train, 'test': num_test}
    
    for split in ['train', 'test']:
        for i in range(num_images[split]):
            # Create dummy MR image (random grayscale)
            mr_img = np.random.randint(0, 256, (img_size, img_size), dtype=np.uint8)
            mr_pil = Image.fromarray(mr_img, mode='L')
            mr_pil.save(os.path.join(root_dir, 'mr', split, f'{str(i).zfill(5)}.jpg'))
            
            # Create dummy CT image (random grayscale, slightly different from MR)
            ct_img = np.random.randint(0, 256, (img_size, img_size), dtype=np.uint8)
            ct_pil = Image.fromarray(ct_img, mode='L')
            ct_pil.save(os.path.join(root_dir, 'ct', split, f'{str(i).zfill(5)}.jpg'))
    
    print(f"Created {num_train} training pairs and {num_test} test pairs")


def test_dataset_loading():
    """
    Test dataset loading
    """
    print("\n" + "="*50)
    print("TEST 1: Dataset Loading")
    print("="*50)
    
    # Create temporary dataset
    temp_dir = tempfile.mkdtemp()
    try:
        create_dummy_dataset(temp_dir, num_train=5, num_test=2, img_size=128)
        
        # Test dataset creation
        dataset = PairedMRCTDataset(
            root_dir=temp_dir,
            split='train',
            mr_mean=0.5,
            mr_std=0.5,
            ct_mean=0.5,
            ct_std=0.5
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Test loading a sample
        mr, ct = dataset[0]
        print(f"MR shape: {mr.shape}, CT shape: {ct.shape}")
        print(f"MR range: [{mr.min():.2f}, {mr.max():.2f}]")
        print(f"CT range: [{ct.min():.2f}, {ct.max():.2f}]")
        
        assert mr.shape == (1, 128, 128), f"Expected shape (1, 128, 128), got {mr.shape}"
        assert ct.shape == (1, 128, 128), f"Expected shape (1, 128, 128), got {ct.shape}"
        
        print("✓ Dataset loading test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading test failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_model_creation():
    """
    Test model creation and forward pass
    """
    print("\n" + "="*50)
    print("TEST 2: Model Creation and Forward Pass")
    print("="*50)
    
    try:
        # Create model
        model_fn = JiT_MRCT_models['JiT-B/16']
        model = model_fn(input_size=128, in_channels=2)
        
        print(f"Model created successfully")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {n_params / 1e6:.2f}M")
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 2, 128, 128)  # 2 channels: zt + MR
        t = torch.rand(batch_size)
        
        with torch.no_grad():
            output = model(x, t)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
        assert output.shape == (batch_size, 1, 128, 128), \
            f"Expected shape ({batch_size}, 1, 128, 128), got {output.shape}"
        
        print("✓ Model creation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_denoiser():
    """
    Test denoiser training and generation
    """
    print("\n" + "="*50)
    print("TEST 3: Denoiser Training and Generation")
    print("="*50)
    
    try:
        # Create dummy args
        class Args:
            model = 'JiT-B/16'
            img_size = 128
            attn_dropout = 0.0
            proj_dropout = 0.0
            P_mean = -0.8
            P_std = 0.8
            noise_scale = 1.0
            t_eps = 5e-2
            ema_decay1 = 0.9999
            ema_decay2 = 0.9996
            sampling_method = 'euler'
            num_sampling_steps = 10  # Reduced for testing
        
        args = Args()
        
        # Create denoiser
        denoiser = Denoiser_MRCT(args)
        print("Denoiser created successfully")
        
        # Initialize EMA
        import copy
        denoiser.ema_params1 = copy.deepcopy(list(denoiser.parameters()))
        denoiser.ema_params2 = copy.deepcopy(list(denoiser.parameters()))
        
        # Test training forward pass
        batch_size = 2
        ct = torch.randn(batch_size, 1, 128, 128)
        mr = torch.randn(batch_size, 1, 128, 128)
        
        loss = denoiser(ct, mr)
        print(f"Training loss: {loss.item():.4f}")
        
        # Test generation
        print("Testing generation (this may take a moment)...")
        with torch.no_grad():
            generated_ct = denoiser.generate(mr)
        
        print(f"Generated CT shape: {generated_ct.shape}")
        assert generated_ct.shape == (batch_size, 1, 128, 128), \
            f"Expected shape ({batch_size}, 1, 128, 128), got {generated_ct.shape}"
        
        print("✓ Denoiser test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Denoiser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """
    Test SSIM and PSNR metrics
    """
    print("\n" + "="*50)
    print("TEST 4: Metrics Calculation")
    print("="*50)
    
    try:
        # Create dummy images
        img1 = torch.rand(2, 1, 128, 128)
        img2 = img1 + 0.1 * torch.randn_like(img1)  # Add some noise
        
        # Test PSNR
        psnr = calculate_psnr(img1, img2)
        print(f"PSNR: {psnr:.2f} dB")
        
        # Test SSIM
        ssim = calculate_ssim(img1, img2)
        print(f"SSIM: {ssim:.4f}")
        
        # Test with identical images
        psnr_perfect = calculate_psnr(img1, img1)
        ssim_perfect = calculate_ssim(img1, img1)
        print(f"Perfect PSNR: {psnr_perfect:.2f} dB (should be inf)")
        print(f"Perfect SSIM: {ssim_perfect:.4f} (should be ~1.0)")
        
        assert ssim > 0 and ssim <= 1.0, f"SSIM should be in (0, 1], got {ssim}"
        assert ssim_perfect > 0.99, f"Perfect SSIM should be ~1.0, got {ssim_perfect}"
        
        print("✓ Metrics test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """
    Test dataloader creation and iteration
    """
    print("\n" + "="*50)
    print("TEST 5: DataLoader")
    print("="*50)
    
    temp_dir = tempfile.mkdtemp()
    try:
        create_dummy_dataset(temp_dir, num_train=10, num_test=3, img_size=128)
        
        # Create dataloaders
        train_loader, test_loader = get_mrct_dataloaders(
            dataset_path=temp_dir,
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            mr_mean=0.5,
            mr_std=0.5,
            ct_mean=0.5,
            ct_std=0.5,
            img_size=128,
            distributed=False
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test iteration
        for mr, ct in train_loader:
            print(f"Batch - MR: {mr.shape}, CT: {ct.shape}")
            break
        
        print("✓ DataLoader test passed!")
        return True
        
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """
    Run all tests
    """
    print("\n" + "="*70)
    print("MR-to-CT Synthesis Implementation Tests")
    print("="*70)
    
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Model Creation", test_model_creation),
        ("Denoiser", test_denoiser),
        ("Metrics", test_metrics),
        ("DataLoader", test_dataloader),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:30s} {status}")
    
    total_passed = sum(1 for _, r in results if r)
    total_tests = len(results)
    
    print("="*70)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n✓ All tests passed! Implementation is working correctly.")
        return 0
    else:
        print(f"\n✗ {total_tests - total_passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    exit(main())
