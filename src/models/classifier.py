"""
classifier.py — Deepfake Detection Classifier MLP with Squeeze-and-Excitation (SE).

Mathematical Formulation (Paper Section 3, Stage 5):
Given the multimodal representation x ∈ R^{input_dim}
(e.g., x_299 = [fused_proj || fused_emo || Δ || P_sarc] ∈ R^{299}):

1. First Dense Layer + Normalization:
       h_1 = GELU(LayerNorm(W_1 · x + b_1)) ∈ R^{512}

2. Squeeze-and-Excitation (1D SE Channel Attention; Hu et al., 2018):
       s = σ( W_{se,2} · GELU(W_{se,1} · h_1) ) ∈ (0, 1)^{512}
       h_{1, SE} = h_1 ⊙ s
   Learns adaptive channel recalibration across affective and physical features.

3. Second Dense Layer + Regularization:
       h_2 = GELU(LayerNorm(W_2 · Dropout(h_{1, SE}) + b_2)) ∈ R^{128}

4. Output Scoring:
       logit = W_3 · h_2 + b_3 ∈ R^1
   At inference, calibrated P(fake) = σ(logit) = 1 / (1 + exp(-logit)).
"""
from __future__ import annotations

import torch
import torch.nn as nn



def apply_squeeze_and_excitation_1d(x: torch.Tensor, se_fc: nn.Sequential) -> torch.Tensor:
    """
    1D Channel Squeeze-and-Excitation gating (Hu et al., 2018):
        s = σ( W_2 · GELU(W_1 · x) ) ∈ (0, 1)^C
        x_SE = x ⊙ s
    """
    w = se_fc(x)
    return x * w


class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_squeeze_and_excitation_1d(x, self.fc)



class ClassifierMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 8198,
        hidden1: int = 512,
        hidden2: int = 128,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, hidden1)
        self.norm1 = nn.LayerNorm(hidden1)
        self.act1  = nn.GELU()
        self.se    = SqueezeExcitation1D(hidden1, reduction=4)
        self.drop  = nn.Dropout(dropout)
        self.fc2   = nn.Linear(hidden1, hidden2)
        self.norm2 = nn.LayerNorm(hidden2)
        self.act2  = nn.GELU()
        self.out   = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, input_dim). Returns logit (B, 1)."""
        h1 = self.act1(self.norm1(self.fc1(x)))
        h1_se = self.se(h1)
        h2 = self.act2(self.norm2(self.fc2(self.drop(h1_se))))
        return self.out(h2)
