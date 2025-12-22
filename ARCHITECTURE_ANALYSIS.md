# Architecture Analysis: DiffusionModelUNet vs JiT (ViT)

This document provides a detailed technical analysis comparing the timestep embedding methods and attention normalization mechanisms between MONAI's **DiffusionModelUNet** and the original **JiT (Just Image Transformer)** architecture.

## Table of Contents

1. [Timestep Embedding Analysis](#timestep-embedding-analysis)
   - [DiffusionModelUNet Timestep Embedding](#diffusionmodelunet-timestep-embedding)
   - [JiT Timestep Embedding](#jit-timestep-embedding)
   - [Comparison](#timestep-embedding-comparison)
2. [Attention Normalization Analysis](#attention-normalization-analysis)
   - [UNet Attention Normalization (GroupNorm + LayerNorm)](#unet-attention-normalization)
   - [ViT Attention Normalization (RMSNorm)](#vit-attention-normalization)
   - [Comparison](#attention-normalization-comparison)
3. [Summary](#summary)
4. [Implications for MR-to-CT Synthesis](#implications-for-mr-to-ct-synthesis)

---

## Timestep Embedding Analysis

### DiffusionModelUNet Timestep Embedding

**Location**: `diffusion_model_unet.py`, lines 461-485, 1758-1762, 1887-1894

The DiffusionModelUNet uses a **two-stage sinusoidal timestep embedding** approach:

#### Stage 1: Sinusoidal Positional Encoding

```python
# diffusion_model_unet.py, get_timestep_embedding function (lines 461-485)
def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings following the implementation in 
    Ho et al. "Denoising Diffusion Probabilistic Models" (DDPM), NeurIPS 2020
    https://arxiv.org/abs/2006.11239
    """
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    freqs = torch.exp(exponent / half_dim)

    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # zero pad for odd embedding dimensions
    if embedding_dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

    return embedding
```

**Key Formula**:
```
freq_i = exp(-log(max_period) * i / half_dim), for i ∈ [0, half_dim)
embedding = [cos(t * freq_0), ..., cos(t * freq_k), sin(t * freq_0), ..., sin(t * freq_k)]
```

This creates a frequency-based encoding where:
- **Low-frequency components** capture the general noise level
- **High-frequency components** capture fine-grained timestep information

#### Stage 2: MLP Projection

```python
# diffusion_model_unet.py, DiffusionModelUNet.__init__ (lines 1758-1762)
time_embed_dim = num_channels[0] * 4  # e.g., 64 * 4 = 256
self.time_embed = nn.Sequential(
    nn.Linear(num_channels[0], time_embed_dim),  # 64 -> 256
    nn.SiLU(),                                    # Activation
    nn.Linear(time_embed_dim, time_embed_dim)    # 256 -> 256
)
```

The sinusoidal embedding is projected through a 2-layer MLP with SiLU activation:
1. `num_channels[0]` → `4 * num_channels[0]` (expansion)
2. SiLU non-linearity
3. `4 * num_channels[0]` → `4 * num_channels[0]` (refinement)

#### Stage 3: Injection into ResNet Blocks

```python
# diffusion_model_unet.py, ResnetBlock.forward (lines 686-690)
if self.spatial_dims == 2:
    temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None]
else:
    temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None, None]
h = h + temb  # Additive injection
```

**Timestep embedding is injected into each ResNet block via:**
1. SiLU activation on the timestep embedding
2. Linear projection to match feature channels
3. **Additive** combination with convolutional features (broadcast over spatial dimensions)

---

### JiT Timestep Embedding

**Location**: `model_jit.py`, lines 40-77, 192-202, 331-340

JiT uses a similar sinusoidal embedding with **Adaptive Layer Normalization (adaLN)** for injection:

#### Stage 1: Sinusoidal Positional Encoding

```python
# model_jit.py, TimestepEmbedder.timestep_embedding (lines 53-72)
@staticmethod
def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
```

**Key Formula** (same as UNet):
```
freq_i = exp(-log(max_period) * i / half_dim), for i ∈ [0, half_dim)
embedding = [cos(t * freq_0), ..., cos(t * freq_k), sin(t * freq_0), ..., sin(t * freq_k)]
```
The full embedding concatenates cosines for all frequencies, then sines for all frequencies.

#### Stage 2: MLP Projection

```python
# model_jit.py, TimestepEmbedder.__init__ (lines 44-51)
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
```

Similar 2-layer MLP:
1. `256` → `hidden_size` (projection to model dimension)
2. SiLU activation
3. `hidden_size` → `hidden_size` (refinement)

#### Stage 3: Combination with Class Embedding

```python
# model_jit.py, JiT.forward (lines 337-340)
t_emb = self.t_embedder(t)   # Timestep embedding
y_emb = self.y_embedder(y)   # Class/label embedding
c = t_emb + y_emb            # Combined conditioning signal
```

Timestep and class embeddings are **added together** to form a unified conditioning signal `c`.

#### Stage 4: Adaptive Layer Normalization (adaLN) Injection

```python
# model_jit.py, JiTBlock.forward (lines 197-202)
def forward(self, x, c, feat_rope=None):
    # Generate 6 modulation parameters from conditioning
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
        self.adaLN_modulation(c).chunk(6, dim=-1)
    
    # modulate() applies: x * (1 + scale) + shift to the normalized features
    # The full residual block: x + gate * sublayer(modulate(norm(x), shift, scale))
    x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
    x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
    return x
```

**adaLN modulation function**:
```python
# model_jit.py, modulate function (lines 13-14)
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```

**JiT uses 6 parameters per block**:
1. `shift_msa`, `scale_msa`, `gate_msa` - for attention modulation
2. `shift_mlp`, `scale_mlp`, `gate_mlp` - for MLP modulation

The gate parameters provide **multiplicative control** over the residual connections.

---

### Timestep Embedding Comparison

| Aspect | DiffusionModelUNet | JiT (ViT) |
|--------|-------------------|-----------|
| **Base Encoding** | Sinusoidal (cos + sin) | Sinusoidal (cos + sin) |
| **Default Frequency Dim** | `num_channels[0]` (e.g., 64) | 256 (fixed) |
| **MLP Structure** | 2-layer with SiLU | 2-layer with SiLU |
| **Final Embedding Dim** | `4 * num_channels[0]` (e.g., 256) | `hidden_size` (e.g., 1024) |
| **Injection Method** | **Additive** to conv features | **Adaptive LN** (scale, shift, gate) |
| **Injection Location** | Each ResNet block | Each Transformer block |
| **Modulation Parameters** | 1 (bias only) | 6 (shift, scale, gate × 2) |
| **Class Conditioning** | Separate embedding, added | Combined with timestep |

**Key Differences**:

1. **Injection Mechanism**:
   - **UNet**: Simple additive injection (`h = h + temb`)
   - **JiT**: Adaptive modulation with scale, shift, and gate parameters

2. **Conditioning Expressivity**:
   - **UNet**: 1 parameter per channel (bias)
   - **JiT**: 6 parameters per block (more expressive but more parameters)

3. **Gating**:
   - **UNet**: No gating mechanism
   - **JiT**: Learnable gates that can **suppress or amplify** residual contributions

---

## Attention Normalization Analysis

### UNet Attention Normalization

The DiffusionModelUNet uses two types of normalization in its attention mechanisms:

#### 1. GroupNorm in SpatialTransformer and AttentionBlock

```python
# diffusion_model_unet.py, SpatialTransformer.__init__ (line 275)
self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)

# diffusion_model_unet.py, AttentionBlock.__init__ (line 377)
self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=norm_eps, affine=True)
```

**GroupNorm Formula**:
```
y = (x - μ_g) / √(σ²_g + ε) * γ + β
```
Where:
- `μ_g`, `σ²_g` are computed over groups of channels
- `γ`, `β` are learnable affine parameters
- Typically `num_groups=32`

#### 2. LayerNorm in BasicTransformerBlock

```python
# diffusion_model_unet.py, BasicTransformerBlock.__init__ (lines 221-223)
self.norm1 = nn.LayerNorm(num_channels)  # Before self-attention
self.norm2 = nn.LayerNorm(num_channels)  # Before cross-attention
self.norm3 = nn.LayerNorm(num_channels)  # Before FFN
```

**LayerNorm Formula**:
```
y = (x - μ) / √(σ² + ε) * γ + β
```
Where:
- `μ`, `σ²` are computed over the last dimension (all features)
- `γ`, `β` are learnable affine parameters

#### Application Order (Pre-LN)

```python
# diffusion_model_unet.py, BasicTransformerBlock.forward (lines 225-233)
def forward(self, x, context=None):
    # Pre-LN: Normalize BEFORE attention/FFN
    x = self.attn1(self.norm1(x)) + x           # Self-attention
    x = self.attn2(self.norm2(x), context) + x  # Cross-attention
    x = self.ff(self.norm3(x)) + x              # Feed-forward
    return x
```

**UNet uses Pre-LayerNorm (Pre-LN)** architecture where normalization is applied before the sublayer.

---

### ViT Attention Normalization

JiT uses **RMSNorm** (Root Mean Square Layer Normalization) throughout:

#### RMSNorm Implementation

```python
# util/model_util.py, RMSNorm class (lines 137-151)
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)
```

**RMSNorm Formula**:
```
y = x / √(mean(x²) + ε) * γ
```

Key characteristics:
- **No mean subtraction** (only variance normalization)
- **No bias term** (only scale)
- Simpler computation than LayerNorm

#### RMSNorm in JiTBlock

```python
# model_jit.py, JiTBlock.__init__ (lines 186-189)
self.norm1 = RMSNorm(hidden_size, eps=1e-6)  # Before attention
self.norm2 = RMSNorm(hidden_size, eps=1e-6)  # Before MLP
```

#### RMSNorm for QK Normalization

```python
# model_jit.py, Attention.__init__ (lines 113-114)
self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
```

JiT applies **additional QK normalization** on query and key vectors:

```python
# model_jit.py, Attention.forward (lines 126-127)
q = self.q_norm(q)
k = self.k_norm(k)
```

#### Combined with adaLN

```python
# model_jit.py, JiTBlock.forward (lines 199-201)
# adaLN modulation is applied AFTER RMSNorm
x = x + gate_msa.unsqueeze(1) * self.attn(
    modulate(self.norm1(x), shift_msa, scale_msa),  # RMSNorm -> modulate -> attention
    rope=feat_rope
)
```

---

### Attention Normalization Comparison

| Aspect | DiffusionModelUNet | JiT (ViT) |
|--------|-------------------|-----------|
| **Primary Norm Type** | LayerNorm | RMSNorm |
| **Spatial Norm Type** | GroupNorm | N/A |
| **Formula** | `(x-μ)/σ * γ + β` | `x/√mean(x²) * γ` |
| **Has Mean Subtraction** | ✅ Yes | ❌ No |
| **Has Bias Term** | ✅ Yes | ❌ No |
| **Parameters** | 2N (weight + bias) | N (weight only) |
| **QK Normalization** | ❌ No | ✅ Yes (RMSNorm) |
| **Normalization Location** | Pre-LN | Pre-LN + adaLN modulation |
| **Adaptive Modulation** | ❌ No | ✅ Yes (scale, shift, gate) |

### Why These Differences Matter

#### 1. RMSNorm Advantages (JiT)
- **Computational Efficiency**: No mean computation, simpler forward/backward pass
- **Memory Efficiency**: Fewer parameters (no bias)
- **Training Stability**: Works well with large transformers (used in LLaMA, etc.)

#### 2. LayerNorm Advantages (UNet)
- **Theoretical Soundness**: Zero-centered activations
- **Compatibility**: Standard choice for vision transformers
- **Bias Term**: Additional expressivity for shift

#### 3. GroupNorm Advantages (UNet)
- **Batch Independence**: Works with any batch size (crucial for diffusion)
- **Spatial Awareness**: Groups channels, preserving spatial structure
- **Medical Imaging**: Particularly effective for medical image analysis

#### 4. QK Normalization (JiT only)
- **Attention Stability**: Prevents attention logits from exploding
- **Scale Invariance**: Makes attention computation more robust
- **Training Dynamics**: Improves convergence in deep transformers

---

## Summary

### Timestep Embedding Summary

| Feature | DiffusionModelUNet | JiT |
|---------|-------------------|-----|
| Encoding | Sinusoidal | Sinusoidal |
| Injection | **Additive** | **adaLN (scale, shift, gate)** |
| Expressivity | Lower | Higher |
| Parameters/block | O(channels) | O(6 × hidden_size) |

### Normalization Summary

| Feature | DiffusionModelUNet | JiT |
|---------|-------------------|-----|
| Main Norm | LayerNorm | **RMSNorm** |
| Spatial Norm | **GroupNorm** | N/A |
| Mean Subtraction | Yes | **No** |
| Bias Term | Yes | **No** |
| QK Norm | No | **Yes** |
| adaLN Modulation | No | **Yes** |

---

## Implications for MR-to-CT Synthesis

### Recommended Approach

For **MR-to-CT medical image synthesis**, the DiffusionModelUNet architecture is preferred because:

1. **GroupNorm**: Works well with variable batch sizes common in medical imaging
2. **Skip Connections**: Preserve fine anatomical details from input MR
3. **Spatial Structure**: Convolutional architecture naturally handles spatial relationships
4. **Proven Performance**: Extensive use in medical imaging diffusion models

### Potential Enhancements

If training instability is observed, consider:
1. Adding QK normalization to the attention layers
2. Using RMSNorm instead of LayerNorm in transformer blocks
3. Implementing adaLN for more expressive timestep conditioning

### Code Reference

```python
# DiffusionModelUNet timestep embedding (diffusion_model_unet.py)
t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])  # Sinusoidal
emb = self.time_embed(t_emb)  # MLP projection

# JiT timestep embedding (model_jit.py)
t_freq = self.timestep_embedding(t, self.frequency_embedding_size)  # Sinusoidal
t_emb = self.mlp(t_freq)  # MLP projection
c = t_emb + y_emb  # Combined with class embedding
# Then modulated via adaLN in each block
```

---

## References

- **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- **DiT**: Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023)
- **JiT**: Based on SiT and Lightning-DiT implementations
- **RMSNorm**: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
- **adaLN**: From Film and other adaptive conditioning methods
- **MONAI**: Project MONAI Generative Models
