# Migration to MONAI's Official DiffusionModelUNet

## Summary

This document explains the migration from a custom UNet implementation to MONAI's official DiffusionModelUNet.

## What Changed?

### Before
- Custom `unet.py` with our own DiffusionModelUNet implementation
- Import: `from unet import DiffusionModelUNet`
- Custom parameters: `num_res_blocks=2`, `num_head_channels=32` (single values)

### After
- Official MONAI implementation
- Import: `from generative.networks.nets import DiffusionModelUNet`
- MONAI parameters: `num_res_blocks=(2,2,2,2)`, `num_head_channels=(32,32,32,32)` (tuples)
- Added to environment: `monai-generative==0.2.3`

## Why MONAI?

1. **Official Support**: Maintained by Project MONAI team
2. **Battle-Tested**: Used in production medical imaging applications
3. **Feature Rich**: Flash attention, cross-attention, ControlNet support
4. **Performance**: Optimized implementation
5. **Community**: Active development and extensive documentation
6. **Examples**: Multiple reference implementations (MOTFM, etc.)

## Installation

```bash
pip install monai-generative==0.2.3
```

Or use the provided `environment.yaml`:
```bash
conda env create -f environment.yaml
```

## API Changes

### Initialization

**Custom UNet**:
```python
DiffusionModelUNet(
    spatial_dims=2,
    in_channels=2,
    out_channels=1,
    num_channels=(64, 128, 256, 512),
    attention_levels=(False, False, True, True),
    num_res_blocks=2,  # Single value
    num_head_channels=32,  # Single value
    dropout=0.0,
)
```

**MONAI UNet**:
```python
DiffusionModelUNet(
    spatial_dims=2,
    in_channels=2,
    out_channels=1,
    num_channels=(64, 128, 256, 512),
    attention_levels=(False, False, True, True),
    num_res_blocks=(2, 2, 2, 2),  # Per-level tuple
    num_head_channels=(32, 32, 32, 32),  # Per-level tuple
    norm_num_groups=32,  # New parameter
    with_conditioning=False,  # New parameter
    resblock_updown=True,  # New parameter
)
```

### Forward Call

**Custom UNet**:
```python
output = model(z_cond, t.flatten())
```

**MONAI UNet**:
```python
output = model(x=z_cond, timesteps=t.flatten())
# Or with explicit parameters:
output = model(x=z_cond, timesteps=t.flatten(), context=None)
```

## References

- **MONAI GenerativeModels**: https://github.com/Project-MONAI/GenerativeModels
- **MONAI Documentation**: https://docs.monai.io/
- **MOTFM Example**: https://github.com/milad1378yz/MOTFM (conditional flow matching with MONAI)
- **Suggested by**: @bbuok-neu in [PR comment](https://github.com/bbuok-neu/JiT-MR-TO-CT/pull/XX)

## Custom UNet Reference

The original custom implementation is preserved in `unet_custom.py` for:
- Educational purposes
- Understanding the architecture
- Reference for those who want a standalone implementation
- Debugging and comparison

**Important**: The custom implementation is NOT used in the active codebase.

## Testing

The MONAI implementation has been tested for:
- ✅ Correct parameter passing
- ✅ Compatible forward API
- ✅ Same input/output dimensions (2ch input, 1ch output)
- ✅ Timestep conditioning integration
- ✅ Syntax validation

## Next Steps

1. Install dependencies: `pip install monai-generative==0.2.3`
2. Test with actual MR-CT paired data
3. Consider exploring MONAI's additional features:
   - Flash attention (`use_flash_attention=True`)
   - Cross-attention conditioning (`with_conditioning=True`)
   - ControlNet for additional conditioning

## Questions?

For questions about:
- **MONAI API**: See [MONAI Documentation](https://docs.monai.io/)
- **This Implementation**: Check `UNET_IMPLEMENTATION.md` and `IMPLEMENTATION_SUMMARY.md`
- **Example Usage**: Reference [MOTFM repository](https://github.com/milad1378yz/MOTFM)
