# Backbone Architectures Analysis

This document provides a detailed analysis of the DINOv2 and DINOv3 backbone architectures used in this project.

## DINOv2

**Source**: `thirdparty/dinov2` (Meta Research)
**Paper**: "DINOv2: Learning Robust Visual Features without Supervision"

### Architecture Overview
DINOv2 uses a standard Vision Transformer (ViT) architecture with some specific modifications for self-supervised learning.

*   **Patch Embedding**: Standard non-overlapping patch projection.
*   **Positional Embedding**: Learnable absolute positional embeddings. Interpolated (bicubic) when input resolution changes.
*   **CLS Token**: Yes, prepended to the sequence.
*   **Register Tokens**: Optional (usually 4), appended after CLS token. Used to capture "background" information and reduce artifacts in feature maps.
*   **Blocks**: Standard Transformer Encoder blocks.
    *   **Attention**: `MemEffAttention` (Memory Efficient Attention, likely xFormers based).
    *   **FFN**: 
        *   Standard `Mlp` (GELU) for smaller models.
        *   `SwiGLUFFNFused` (SwiGLU activation) for Giant models.
    *   **LayerScale**: Used to stabilize training (initialized with small values).
    *   **DropPath**: Stochastic depth regularization.
*   **Normalization**: `LayerNorm` (eps=1e-6).

### Model Variants

| Model Name | Patch Size | Depth | Embed Dim | Heads | FFN | Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **dinov2_vits14** | 14 | 12 | 384 | 6 | MLP | ~21M |
| **dinov2_vitb14** | 14 | 12 | 768 | 12 | MLP | ~86M |
| **dinov2_vitl14** | 14 | 24 | 1024 | 16 | MLP | ~300M |
| **dinov2_vitg14** | 14 | 40 | 1536 | 24 | SwiGLU | ~1.1B |

*Note: `_reg` variants have the same architecture but include 4 extra register tokens.*

---

## DINOv3

**Source**: `thirdparty/dinov3` (Meta Research)
**Paper**: "DINOv3" (Assuming upcoming or internal reference, based on codebase)

### Architecture Overview
DINOv3 builds upon ViT but introduces several modern architectural improvements, aligning more with LLM architectures (like LLaMA).

*   **Patch Embedding**: Standard (Patch size 16 is default here, vs 14 in DINOv2).
*   **Positional Embedding**: **RoPE (Rotary Positional Embeddings)**. This is a significant change from DINOv2's absolute embeddings. It allows for better generalization to different resolutions.
*   **CLS Token**: Yes.
*   **Storage Tokens**: Similar to register tokens (default 4).
*   **Blocks**: `SelfAttentionBlock`.
    *   **Attention**: Uses RoPE.
    *   **FFN**: `SwiGLUFFN` (SwiGLU) is standard or optional depending on model size.
    *   **LayerScale**: Yes.
*   **Normalization**: `RMSNorm` (Root Mean Square Norm) or `LayerNorm` (bf16 compatible).
*   **Bias**: QKV bias is True, but some large models might disable it.

### Model Variants

| Model Name | Patch Size | Depth | Embed Dim | Heads | FFN Ratio | FFN Type | Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **dinov3_vits16** | 16 | 12 | 384 | 6 | 4.0 | MLP | Small |
| **dinov3_vits16plus** | 16 | 12 | 384 | 6 | **6.0** | **SwiGLU** | Small+ |
| **dinov3_vitb16** | 16 | 12 | 768 | 12 | 4.0 | MLP | Base |
| **dinov3_vitl16** | 16 | 24 | 1024 | 16 | 4.0 | MLP | Large |
| **dinov3_vitl16plus** | 16 | 24 | 1024 | 16 | **6.0** | **SwiGLU** | Large+ |
| **dinov3_vith16plus** | 16 | 32 | 1280 | 20 | **6.0** | **SwiGLU** | Huge+ |
| **dinov3_vit7b16** | 16 | 40 | 4096 | 32 | 3.0 | **SwiGLU** | 7B |

### Key Differences (DINOv2 vs DINOv3)

1.  **Positional Encoding**: DINOv2 uses Absolute PE; DINOv3 uses **RoPE**.
2.  **Patch Size**: DINOv2 defaults to 14; DINOv3 defaults to 16.
3.  **FFN**: DINOv3 aggressively adopts **SwiGLU** and higher expansion ratios (e.g., 6.0 in "plus" models) compared to DINOv2's standard MLP (except Giant).
4.  **Normalization**: DINOv3 often uses `RMSNorm` or `LayerNorm` with bf16 epsilon.
5.  **Tokens**: DINOv3 explicitly names "Storage Tokens" (default 4), functionally similar to DINOv2's registers.

