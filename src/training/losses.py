"""
losses.py — Multi-task Loss for Deepfake Detection from First Principles.

Mathematical Formulation (Paper Section 3, Stage 6):
The total optimization objective minimizes the multi-task loss function:
    L_total = L_det(Focal)
            + λ_a * L_CE(audio_emotion)
            + λ_b * L_CE(visual_emotion)
            + λ_s * L_BCE(sarcasm)
            + λ_d * L_DANN(domain)
            + λ_m * L_margin(contrastive)

First-Principles Loss Formulations:
1. Numerically Stable Binary Cross-Entropy with Pos-Weight (w):
       BCE(z, y; w) = (1 - y)·z + (1 + (w - 1)·y)·[ max(-z, 0) + log(1 + exp(-|z|)) ]
   Derived from L = - [w·y·log σ(z) + (1 - y)·log(1 - σ(z))] reformulated via LogSumExp
   to prevent floating-point underflow and overflow.

2. Focal Loss for Class Imbalance (Lin et al., 2017):
       p = σ(z) = 1 / (1 + exp(-z))
       p_t = p·y + (1 - p)·(1 - y)
       α_t = w·y + (1 - y)
       focal_weight = α_t · (1 - p_t)^γ
       L_det = (1 / N) ∑ focal_weight_i · BCE(z_i, y_i)

3. Categorical Cross-Entropy with Target Masking:
       LSE(z_i) = max_k(z_{i,k}) + log( ∑_k exp(z_{i,k} - max_j(z_{i,j})) )
       NLL_i = LSE(z_i) - z_{i, y_i}
       L_CE = (1 / |V|) ∑_{i ∈ V} NLL_i

4. Supervised Contrastive Margin Separation (Separates Real and Fake Distributions):
       L_margin = max( 0, M - (s̄_fake - s̄_real) )
   where s̄_fake and s̄_real are batch mean detector logits for fake and real samples,
   and M = 1.5 is the separation margin parameter.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def compute_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """
    First-principles numerically stable Binary Cross-Entropy with Logits:
        BCE(z, y; w) = (1 - y)·z + (1 + (w - 1)·y) · [ max(-z, 0) + log(1 + exp(-|z|)) ]

    Avoids calling exp(z) directly when z > 0 to guarantee mathematical stability
    against float32 saturation.
    """
    max_neg = torch.clamp(-logits, min=0.0)
    log_term = torch.log1p(torch.exp(-torch.abs(logits)))
    if pos_weight is not None:
        pw = float(pos_weight) if not isinstance(pos_weight, torch.Tensor) else pos_weight
        return (1.0 - targets) * logits + (1.0 + (pw - 1.0) * targets) * (max_neg + log_term)
    return torch.clamp(logits, min=0.0) - logits * targets + log_term


def compute_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
    pos_weight: float | None = 1.0,
) -> torch.Tensor:
    """
    First-principles Focal Loss for binary detection under class imbalance (Lin et al., 2017):
        p = 1 / (1 + exp(-logits))
        p_t = p·y + (1 - p)·(1 - y)
        focal_weight = α_t · (1 - p_t)^γ
        FL = (1 / N) ∑ focal_weight · BCE(logits, targets)
    """
    p = 1.0 / (1.0 + torch.exp(-logits))
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    ce = compute_binary_cross_entropy_with_logits(logits, targets)
    focal_weight = (1.0 - p_t) ** gamma
    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        focal_weight = focal_weight * alpha_t
    if pos_weight is not None and pos_weight != 1.0:
        pw_weight = targets * float(pos_weight) + (1.0 - targets)
        focal_weight = focal_weight * pw_weight
    return (focal_weight * ce).mean()



def compute_masked_categorical_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    First-principles Categorical Cross-Entropy with Target Masking:
        LSE(z) = max(z) + log( ∑ exp(z - max(z)) )
        NLL = LSE(z) - z_y
        L = mean_{valid}(NLL)
    """
    mask = (targets != ignore_index)
    if not mask.any():
        return logits.new_zeros(1).squeeze()
    z_valid = logits[mask]
    y_valid = targets[mask]

    # Stable LogSumExp: LSE(z) = m + log( ∑ exp(z - m) )
    max_val = torch.max(z_valid, dim=-1, keepdim=True).values
    lse = max_val + torch.log(torch.sum(torch.exp(z_valid - max_val), dim=-1, keepdim=True))

    # Negative log-likelihood: NLL = LSE - z[target]
    nll = lse.squeeze(-1) - z_valid.gather(1, y_valid.unsqueeze(-1)).squeeze(-1)
    return nll.mean()


def compute_contrastive_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.5,
) -> torch.Tensor:
    """
    First-principles Supervised Contrastive Margin Loss:
        L_margin = max( 0, M - (s̄_fake - s̄_real) )
    """
    fake_mask = (labels == 1)
    real_mask = (labels == 0)
    if fake_mask.any() and real_mask.any():
        mean_fake = logits[fake_mask].mean()
        mean_real = logits[real_mask].mean()
        return torch.clamp(float(margin) - (mean_fake - mean_real), min=0.0)
    return logits.new_zeros(1).squeeze()


def compute_multitask_total_loss(
    l_det: torch.Tensor,
    l_emo_a: torch.Tensor,
    l_emo_b: torch.Tensor,
    l_sarc: torch.Tensor,
    l_domain: torch.Tensor,
    l_margin: torch.Tensor,
    lambda_a: float = 0.1,
    lambda_b: float = 0.1,
    lambda_sarcasm: float = 0.05,
    lambda_domain: float = 0.1,
    lambda_margin: float = 0.2,
) -> torch.Tensor:
    """
    Weighted multi-task optimization objective (Paper Section 3, Stage 6):
        L_total = L_det + λ_a·L_emo,A + λ_b·L_emo,B + λ_s·L_sarc + λ_d·L_domain + λ_m·L_margin
    """
    return (
        l_det
        + lambda_a * l_emo_a
        + lambda_b * l_emo_b
        + lambda_sarcasm * l_sarc
        + lambda_domain * l_domain
        + lambda_margin * l_margin
    )


@dataclass
class LossOutput:
    total: torch.Tensor
    bce: torch.Tensor
    emotion_a: torch.Tensor
    emotion_b: torch.Tensor
    sarcasm: torch.Tensor
    domain: torch.Tensor = torch.tensor(0.0)
    margin: torch.Tensor = torch.tensor(0.0)


class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss Module directly translating the thesis loss formulation.
    """
    def __init__(
        self,
        lambda_a: float = 0.1,
        lambda_b: float = 0.1,
        lambda_sarcasm: float = 0.05,
        lambda_domain: float = 0.1,
        lambda_margin: float = 0.2,
        margin: float = 1.5,
        pos_weight: float | None = 1.0,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda_a       = lambda_a
        self.lambda_b       = lambda_b
        self.lambda_sarcasm = lambda_sarcasm
        self.lambda_domain  = lambda_domain
        self.lambda_margin  = lambda_margin
        self.margin         = margin
        self._pw            = pos_weight
        self.use_focal      = use_focal
        self.focal_gamma    = focal_gamma

        # Retained as transparent fallback modules for backward compatibility
        self._bce           = nn.BCEWithLogitsLoss()
        self._ce            = nn.CrossEntropyLoss(ignore_index=-1)
        self._bce_sum       = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(
        self,
        fake_logit:           torch.Tensor,             # (B, 1)  raw fake score
        fake_label:           torch.Tensor,             # (B,)    0 or 1 or -1 (MUStARD)
        emotion_a_logits:     torch.Tensor,             # (B, 6)  audio emotion
        emotion_b_logits:     torch.Tensor,             # (B, 6)  visual emotion
        audio_emotion_label:  torch.Tensor,             # (B,)    0-5 or -1
        visual_emotion_label: torch.Tensor,             # (B,)    0-5 or -1
        sarcasm_logit:        torch.Tensor,             # (B, 1)  raw sarcasm score
        sarcasm_label:        torch.Tensor,             # (B,)    0, 1, or -1 (masked)
        domain_logits:        torch.Tensor | None = None,# (B, 5)  dataset domain logits (DANN)
        domain_label:         torch.Tensor | None = None,# (B,)    0-4 dataset domain index
    ) -> LossOutput:
        # Step 1: Detection Loss (Focal Loss or BCE) — mask MUStARD clips (fake_label=-1)
        fake_mask = fake_label != -1
        if fake_mask.any():
            logits = fake_logit.squeeze(1)[fake_mask]
            labels = fake_label[fake_mask].float()
            if self.use_focal:
                l_bce = compute_focal_loss(
                    logits, labels, gamma=self.focal_gamma, pos_weight=self._pw
                )
            elif self._pw is not None and self._pw != 1.0:
                l_bce = compute_binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=self._pw
                ).mean()
            else:
                l_bce = compute_binary_cross_entropy_with_logits(logits, labels).mean()
        else:
            l_bce = fake_logit.new_zeros(1).squeeze()

        # Step 2: Supervised Contrastive Margin Loss
        if self.lambda_margin > 0 and fake_mask.any():
            l_margin = compute_contrastive_margin_loss(
                fake_logit.squeeze(1)[fake_mask], fake_label[fake_mask], margin=self.margin
            )
        else:
            l_margin = fake_logit.new_zeros(1).squeeze()

        # Step 3: Auxiliary Emotion Categorical Cross Entropy Loss
        l_emo_a = compute_masked_categorical_cross_entropy(
            emotion_a_logits, audio_emotion_label, ignore_index=-1
        )
        l_emo_b = compute_masked_categorical_cross_entropy(
            emotion_b_logits, visual_emotion_label, ignore_index=-1
        )

        # Step 4: Sarcasm Binary Cross Entropy Loss (Mask non-MUStARD clips)
        sarc_mask = sarcasm_label != -1
        if sarc_mask.any():
            sarc_bce = compute_binary_cross_entropy_with_logits(
                sarcasm_logit.squeeze(1)[sarc_mask], sarcasm_label[sarc_mask].float()
            )
            l_sarc = sarc_bce.sum() / sarc_mask.sum().float()
        else:
            l_sarc = fake_logit.new_zeros(1).squeeze()

        # Step 5: Domain Adversarial Cross Entropy Loss (DANN)
        if domain_logits is not None and domain_label is not None and (domain_label != -1).any():
            l_domain = compute_masked_categorical_cross_entropy(
                domain_logits, domain_label, ignore_index=-1
            )
        else:
            l_domain = fake_logit.new_zeros(1).squeeze()

        # Step 6: Multi-Task Weighted Sum via direct mathematical formula
        total = compute_multitask_total_loss(
            l_det=l_bce,
            l_emo_a=l_emo_a,
            l_emo_b=l_emo_b,
            l_sarc=l_sarc,
            l_domain=l_domain,
            l_margin=l_margin,
            lambda_a=self.lambda_a,
            lambda_b=self.lambda_b,
            lambda_sarcasm=self.lambda_sarcasm,
            lambda_domain=self.lambda_domain,
            lambda_margin=self.lambda_margin,
        )

        return LossOutput(
            total=total,
            bce=l_bce,
            emotion_a=l_emo_a,
            emotion_b=l_emo_b,
            sarcasm=l_sarc,
            domain=l_domain,
            margin=l_margin,
        )


if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("   DEEPSENTINEL MULTI-TASK LOSS FUNCTIONS -- RAW FORMULA VERIFICATION")
    print("=" * 75)
    print("Paper Formulation: Joint Detection, Dual Emotion, Sarcasm, Domain & Margin\n")

    torch.manual_seed(42)
    B = 4

    # Synthetic batch data
    logits = torch.tensor([2.5, -2.1, 0.4, -0.2], requires_grad=True)  # Detection logits
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])                       # Ground truth fake/real

    # Step 1: Numerical BCE & Focal Loss
    bce_raw = compute_binary_cross_entropy_with_logits(logits, targets)
    fl_raw = compute_focal_loss(logits, targets, alpha=0.25, gamma=2.0)
    print("[STEP 1] DETECTION LOSS FORMULATIONS:")
    print(f"  Raw Binary Cross Entropy (BCE) per sample : {bce_raw.detach().numpy().round(5)}")
    print(f"  BCE Mean Loss                             : {bce_raw.mean().item():.6f}")
    print(f"  Focal Loss (alpha=0.25, gamma=2.0) Mean   : {fl_raw.item():.6f}")
    print(f"  --> Focal loss down-weights confident easy examples to focus on hard boundaries.\n")

    # Step 2: Masked Cross Entropy for Emotions
    emo_logits = torch.randn(B, 6)
    emo_targets = torch.tensor([2, 0, -1, 4])  # Note sample 3 has ignore_index=-1
    l_emo = compute_masked_categorical_cross_entropy(emo_logits, emo_targets, ignore_index=-1)
    print("[STEP 2] MASKED EMOTION CATEGORICAL CROSS ENTROPY:")
    print(f"  Targets with missing/unlabeled mask (-1) : {emo_targets.tolist()}")
    print(f"  Masked Emotion NLL Loss                  : {l_emo.item():.6f}\n")

    # Step 3: Contrastive Margin Loss
    fake_scores = torch.tensor([0.85, 0.92])
    real_scores = torch.tensor([0.15, 0.22])
    l_margin = compute_contrastive_margin_loss(fake_scores, real_scores, margin=1.5)
    print("[STEP 3] CONTRASTIVE MARGIN SEPARATION LOSS:")
    print(f"  Mean Fake Score: {fake_scores.mean().item():.3f} | Mean Real Score: {real_scores.mean().item():.3f}")
    print(f"  Separation Gap : {fake_scores.mean().item() - real_scores.mean().item():.3f}")
    print(f"  Margin Loss    : max(0, 1.5 - gap) = {l_margin.item():.6f}\n")

    # Step 4: Multi-Task Loss Composition
    criterion = MultiTaskLoss(
        lambda_a=0.3,
        lambda_b=0.3,
        lambda_sarcasm=0.2,
        lambda_domain=0.1,
        lambda_margin=0.05,
    )
    fake_logits_batch = logits.unsqueeze(1)
    targets_batch = targets
    emo_a_logits = torch.randn(B, 6)
    emo_b_logits = torch.randn(B, 6)
    sarc_logits = torch.randn(B, 1)
    sarc_labels = torch.tensor([1, -1, 0, -1])
    dom_logits = torch.randn(B, 5)
    dom_labels = torch.tensor([0, 1, 4, 2])

    loss_out = criterion(
        fake_logit=fake_logits_batch,
        fake_label=targets_batch,
        emotion_a_logits=emo_a_logits,
        emotion_b_logits=emo_b_logits,
        audio_emotion_label=emo_targets,
        visual_emotion_label=emo_targets,
        sarcasm_logit=sarc_logits,
        sarcasm_label=sarc_labels,
        domain_logits=dom_logits,
        domain_label=dom_labels,
    )

    print("[STEP 4] FULL MULTI-TASK TOTAL LOSS COMPOSITION:")
    print(f"  L_det     (Detection BCE)          : {loss_out.bce.item():.6f}")
    print(f"  L_emo_a   (Audio Emotion NLL)      : {loss_out.emotion_a.item():.6f} (lambda=0.30)")
    print(f"  L_emo_b   (Visual Emotion NLL)     : {loss_out.emotion_b.item():.6f} (lambda=0.30)")
    print(f"  L_sarc    (Sarcasm BCE)            : {loss_out.sarcasm.item():.6f} (lambda=0.20)")
    print(f"  L_domain  (Domain Adversarial NLL) : {loss_out.domain.item():.6f} (lambda=0.10)")
    print(f"  L_margin  (Cluster Separation)     : {loss_out.margin.item():.6f} (lambda=0.05)")
    print("  " + "-" * 60)
    print(f"  TOTAL MULTI-TASK LOSS              : {loss_out.total.item():.6f}")
    print("=" * 75 + "\n")


