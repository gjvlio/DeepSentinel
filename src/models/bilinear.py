"""
bilinear.py — Compact Bilinear Pooling (CBP) of multimodal feature vectors.

Mathematical Formulation (Pham & Pagh, 2013; Fukui et al., 2016):
Given multimodal feature representations:
    Z_at ∈ R^{1536}  (Acoustic-linguistic representation: Wav2Vec2 || BERT)
    Z_v  ∈ R^{768}   (Visual facial sequence representation: ViT + GRU)

The full cross-modal outer product:
    Z_outer = Z_at ⊗ Z_v ∈ R^{1536 × 768}  (1,179,648 dimensions)
is prohibitively high-dimensional and prone to severe overfitting.

Compact Bilinear Pooling approximates Z_outer in a lower D-dimensional sketch space
(default D = 8,192) via Count Sketch projections and FFT circular convolution:

1. Count Sketch Projection Ψ(x, h, s) ∈ R^D:
       Ψ(x, h, s)_j = ∑_{i: h(i) = j} s(i) · x_i
   where:
       h: {1, ..., d} → {0, ..., D-1} is a uniform random hash function.
       s: {1, ..., d} → {-1, +1}      is a Rademacher random sign function.

2. Circular Convolution via Convolution Theorem:
   By the Tensor Sketch theorem, the projection of the outer product equals the circular
   convolution of the individual sketch projections:
       Ψ(Z_at ⊗ Z_v) = Ψ_1(Z_at, h_1, s_1) * Ψ_2(Z_v, h_2, s_2)
   In the Discrete Fourier Transform (DFT) frequency domain:
       x_CBP = F^{-1}( F(Ψ_1) ⊙ F(Ψ_2) )
   where F is the Fast Fourier Transform (rfft), F^{-1} is the inverse FFT (irfft),
   and ⊙ denotes element-wise complex multiplication.

3. Signed Square-Root Power Normalization:
       y_k = sign(x_CBP,k) · sqrt(|x_CBP,k| + ε)
   Suppresses extreme frequency peaks and linearizes the inner product space.

4. L2 Vector Normalization:
       fused = y / ||y||_2 = y / sqrt(∑_{k=1}^D y_k^2 + ε)
   Guarantees ||fused||_2 = 1, stabilizing downstream MLP logit gradients.

Input:  Z_at (B, 1536), Z_v (B, 768)
Output: fused (B, D) — default (B, 8192), strictly L2-normalized
"""
from __future__ import annotations

import torch
import torch.nn as nn


def compute_count_sketch(
    x: torch.Tensor,
    h: torch.Tensor,
    s: torch.Tensor,
    output_dim: int,
) -> torch.Tensor:
    """
    Count Sketch projection Ψ(x, h, s) ∈ R^{B × output_dim}:
        Ψ(x)_j = ∑_{i: h(i)=j} s(i) · x_i

    Args:
        x: (B, d) input feature tensor
        h: (d,) hash indices mapping each input dimension to [0, output_dim - 1]
        s: (d,) Rademacher random signs {-1.0, +1.0}
        output_dim: dimension of projected sketch space (e.g. 8192)
    Returns:
        y: (B, output_dim) projected sketch tensor
    """
    B = x.shape[0]
    x_fp32 = x.float()
    s_fp32 = s.float()
    y = torch.zeros(B, output_dim, device=x.device, dtype=torch.float32)
    y.scatter_add_(1, h.unsqueeze(0).expand(B, -1), x_fp32 * s_fp32)
    return y


def compute_circular_convolution_fft(
    psi_1: torch.Tensor,
    psi_2: torch.Tensor,
    output_dim: int,
) -> torch.Tensor:
    """
    Circular Convolution of Count Sketches via Convolution Theorem:
        x_CBP = F^{-1}( F(Ψ_1) ⊙ F(Ψ_2) )

    Args:
        psi_1: (B, output_dim) sketch projection of modality 1
        psi_2: (B, output_dim) sketch projection of modality 2
        output_dim: target dimension (n for irfft)
    Returns:
        x_cbp: (B, output_dim) unnormalized circular convolution result
    """
    fft_1 = torch.fft.rfft(psi_1)
    fft_2 = torch.fft.rfft(psi_2)
    return torch.fft.irfft(fft_1 * fft_2, n=output_dim)


def compute_signed_sqrt_power_normalization(
    x: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Signed Square-Root Power Normalization (Fukui et al., 2016):
        y_k = sign(x_k) · sqrt(|x_k| + ε)

    Linearizes the inner product space and prevents extreme magnitude outliers.
    """
    return torch.sign(x) * torch.sqrt(torch.abs(x) + eps)


def compute_l2_vector_normalization(
    y: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    First-principles L2 vector normalization:
        fused = y / ||y||_2 = y / sqrt(∑_{k=1}^D y_k^2 + ε)
    """
    l2_norm = torch.sqrt(torch.sum(y ** 2, dim=-1, keepdim=True) + eps)
    return y / l2_norm


def compute_compact_bilinear_pooling(
    z_at: torch.Tensor,
    z_v: torch.Tensor,
    h_at: torch.Tensor,
    s_at: torch.Tensor,
    h_v: torch.Tensor,
    s_v: torch.Tensor,
    output_dim: int = 8192,
) -> torch.Tensor:
    """
    Complete Compact Bilinear Pooling calculation directly expressing the paper's formula:
        1. Count Sketch:    Ψ_at = Sketch(Z_at),  Ψ_v = Sketch(Z_v)
        2. Circular FFT:    x = F^{-1}( F(Ψ_at) ⊙ F(Ψ_v) )
        3. Power Norm:      y = sign(x) · sqrt(|x| + ε)
        4. L2 Norm:         fused = y / ||y||_2

    Args:
        z_at: (B, 1536) acoustic-linguistic embedding
        z_v:  (B, 768) visual facial embedding
        h_at, s_at: hash and sign projections for Z_at
        h_v, s_v:   hash and sign projections for Z_v
        output_dim: target sketch dimension (8192)
    Returns:
        fused: (B, output_dim) L2-normalized compact bilinear feature
    """
    orig_dtype = z_at.dtype

    # Step 1: Count Sketch projections
    psi_at = compute_count_sketch(z_at, h_at, s_at, output_dim)
    psi_v  = compute_count_sketch(z_v,  h_v,  s_v,  output_dim)

    # Step 2: Circular Convolution via Discrete Fourier Transform
    x_cbp = compute_circular_convolution_fft(psi_at, psi_v, output_dim)

    # Step 3: Signed Square-Root Power Normalization
    y = compute_signed_sqrt_power_normalization(x_cbp)

    # Step 4: First-principles L2 Normalization
    fused = compute_l2_vector_normalization(y)

    return fused.to(orig_dtype)


class CompactBilinearFusion(nn.Module):
    """
    Compact Bilinear Pooling via Count Sketch + FFT Convolution (Fukui et al., 2016).

    Approximates Z_at ⊗ Z_v in D = output_dim space without materializing
    the 1.18-million-dimensional outer product matrix.
    Count Sketch hash tables (h) and sign functions (s) are fixed random projections
    (not learned parameters) seeded deterministically.
    """

    def __init__(
        self,
        z_at_dim:   int = 1536,
        z_v_dim:    int = 768,
        output_dim: int = 8192,
        seed:       int = 42,
    ):
        super().__init__()
        self.output_dim = output_dim

        rng = torch.Generator()
        rng.manual_seed(seed)

        # Hash index mappings: h(i) ∈ {0, ..., output_dim - 1}
        # Rademacher signs:    s(i) ∈ {-1.0, +1.0}
        h_at = torch.randint(0, output_dim, (z_at_dim,), generator=rng)
        s_at = torch.randint(0, 2,          (z_at_dim,), generator=rng).float() * 2.0 - 1.0
        h_v  = torch.randint(0, output_dim, (z_v_dim,),  generator=rng)
        s_v  = torch.randint(0, 2,          (z_v_dim,),  generator=rng).float() * 2.0 - 1.0

        self.register_buffer("h_at", h_at)
        self.register_buffer("s_at", s_at)
        self.register_buffer("h_v",  h_v)
        self.register_buffer("s_v",  s_v)

    def _sketch(self, x: torch.Tensor, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Count Sketch projection delegating to standalone formula function."""
        return compute_count_sketch(x, h, s, self.output_dim)

    def forward(self, z_at: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        """Delegates directly to the explicit mathematical CBP pipeline."""
        return compute_compact_bilinear_pooling(
            z_at, z_v, self.h_at, self.s_at, self.h_v, self.s_v, self.output_dim
        )


# Backward-compatible alias
BilinearFusion = CompactBilinearFusion


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("   COMPACT BILINEAR POOLING (CBP) -- RAW MATHEMATICAL VERIFICATION")
    print("=" * 70)
    print("Pham & Pagh (2013) / Fukui et al. (2016) Multimodal Outer-Product Approximation\n")

    torch.manual_seed(42)
    B = 2
    d_at = 1536
    d_v = 768
    D = 8192

    print(f"[STEP 1] INPUT FEATURE SPACES & DIMENSIONAL REDUCTION:")
    print(f"  Acoustic-Text vector  Z_at : ({B}, {d_at}) [Wav2Vec2 768 + BERT 768]")
    print(f"  Visual Facial vector  Z_v  : ({B}, {d_v})  [ViT-B/16 + GRU 768]")
    full_outer = d_at * d_v
    print(f"  Full Outer Product Dim     : {d_at} x {d_v} = {full_outer:,} dimensions")
    print(f"  CBP Target Sketch Dim  (D) : {D:,} dimensions")
    print(f"  Compression Ratio          : {full_outer / D:.1f}:1 reduction (prevents overfitting)\n")

    z_at = torch.randn(B, d_at)
    z_v = torch.randn(B, d_v)

    # Initialize model
    cbp = CompactBilinearFusion(z_at_dim=d_at, z_v_dim=d_v, output_dim=D, seed=42)

    print(f"[STEP 2] COUNT SKETCH PROJECTION (Psi_j = sum s(i) * x_i):")
    psi_at = compute_count_sketch(z_at, cbp.h_at, cbp.s_at, D)
    psi_v  = compute_count_sketch(z_v, cbp.h_v, cbp.s_v, D)
    print(f"  Psi(Z_at) sketch shape : {tuple(psi_at.shape)} | Mean: {psi_at.mean().item():+.5f} | Std: {psi_at.std().item():.5f}")
    print(f"  Psi(Z_v)  sketch shape : {tuple(psi_v.shape)}  | Mean: {psi_v.mean().item():+.5f} | Std: {psi_v.std().item():.5f}\n")

    print(f"[STEP 3] CIRCULAR CONVOLUTION VIA CONVOLUTION THEOREM (FFT):")
    print(f"  x_CBP = F^{{-1}}( F(Psi_at) (.) F(Psi_v) )")
    x_cbp = compute_circular_convolution_fft(psi_at, psi_v, D)
    print(f"  x_CBP raw shape        : {tuple(x_cbp.shape)}")
    print(f"  x_CBP raw L2 norm      : {torch.norm(x_cbp, p=2, dim=-1).tolist()}\n")

    print(f"[STEP 4] SIGNED SQUARE-ROOT POWER NORMALIZATION:")
    print(f"  y_k = sign(x_k) * sqrt(|x_k| + eps)")
    y = compute_signed_sqrt_power_normalization(x_cbp, eps=1e-5)
    print(f"  y shape                : {tuple(y.shape)}")
    print(f"  y L2 norm before L2    : {torch.norm(y, p=2, dim=-1).tolist()}\n")

    print(f"[STEP 5] L2 VECTOR NORMALIZATION (UNIT HYPER-SPHERE PROJECTION):")
    print(f"  fused = y / sqrt(sum y_k^2 + eps)")
    fused = compute_l2_vector_normalization(y, eps=1e-12)
    final_norms = torch.norm(fused, p=2, dim=-1).tolist()
    print(f"  Fused CBP shape        : {tuple(fused.shape)}")
    print(f"  Final L2 Norms         : {[f'{n:.6f}' for n in final_norms]}")
    assert all(abs(n - 1.0) < 1e-4 for n in final_norms), "Norm must equal 1.0!"
    print(f"  --> VERIFIED: All output vectors reside strictly on the unit hypersphere ||fused||_2 = 1.000000")
    print("=" * 70 + "\n")



