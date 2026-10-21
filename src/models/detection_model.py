"""
detection_model.py — Full Deepfake Detection Module.

Two forward paths:
  forward(audio_values, input_ids, attention_mask, keyframe_pixels)
      — end-to-end, includes Wav2Vec2 + BERT + ViT backbones (Phase 2)
  forward_from_features(z_at, z_v)
      — bypasses backbones, uses precomputed feature vectors (Phase 1 / inference)

Output: DetectorOutput(logit, emotion_a, emotion_b, sarcasm)
  logit      — (B, 1) raw fake score; apply sigmoid for P(fake)
  emotion_a  — (B, 6) audio emotion logits
  emotion_b  — (B, 6) visual emotion logits
  sarcasm    — (B, 1) raw sarcasm score; apply sigmoid for P(sarcastic)
"""
from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .emotion_heads import EmotionHeadA, EmotionHeadB
from .bilinear import BilinearFusion
from .classifier import ClassifierMLP
from .sarcasm_head import SarcasmHead


def compute_emotion_probabilities(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Emotion Softmax probability distribution:
        P_k = exp(z_k - max(z)) / sum_{j=1}^C exp(z_j - max(z))

    Maps raw emotion logits to a valid categorical probability simplex P ∈ Δ^{C-1}.
    """
    max_val = torch.max(logits, dim=dim, keepdim=True).values
    exp_shifted = torch.exp(logits - max_val)
    return exp_shifted / torch.sum(exp_shifted, dim=dim, keepdim=True)


# Backwards-compatible alias
compute_softmax = compute_emotion_probabilities


def compute_detection_probability(logits: torch.Tensor) -> torch.Tensor:
    """
    Standard logistic sigmoid function:
        P(fake) = σ(z) = 1 / (1 + exp(-z))

    Maps unbounded detection logit z ∈ (-∞, +∞) to calibrated probability P ∈ (0, 1).
    """
    return 1.0 / (1.0 + torch.exp(-logits))


# Backwards-compatible alias
compute_sigmoid = compute_detection_probability


def compute_emotion_discrepancy(prob_a: torch.Tensor, prob_b: torch.Tensor) -> torch.Tensor:
    """
    Cross-modal emotion discrepancy vector (Paper Section 3, Stage 4):
        Δ_k = |P_{A, k} - P_{B, k}|,   k ∈ {1, ..., 6}

    Measures the absolute affective divergence between acoustic prosody (P_A)
    and visual facial Action Unit expression (P_B).
    """
    return torch.abs(prob_a - prob_b)


def compute_discrepancy_magnitude(delta: torch.Tensor) -> torch.Tensor:
    """
    Euclidean norm of the cross-modal discrepancy vector:
        ||Δ||_2 = sqrt( ∑_{k=1}^6 Δ_k^2 )
    """
    return torch.sqrt(torch.sum(delta ** 2, dim=-1))


def compute_affect_cooccurrence_matrix(prob_a: torch.Tensor, prob_b: torch.Tensor) -> torch.Tensor:
    """
    Affective joint co-occurrence outer product matrix (Paper Section 3, Stage 4):
        M_emo = P_A ⊗ P_B = P_A P_B^T ∈ R^{6 × 6}
        fused_emo = vec(M_emo) ∈ R^{36}

    Encodes fine-grained cross-modal emotional pairs (e.g., Happy face + Angry voice)
    which distinguishes specific deceptive configurations.
    """
    outer = torch.bmm(prob_a.unsqueeze(2), prob_b.unsqueeze(1))  # (B, 6, 6)
    return outer.view(prob_a.size(0), 36)                        # (B, 36)


def compute_bottleneck_projection(
    fused_cbp: torch.Tensor,
    proj_layer: nn.Linear,
    norm_layer: nn.LayerNorm,
) -> torch.Tensor:
    """
    Sub-symbolic CBP bottleneck projection (Paper Section 3, Stage 5):
        fused_proj = GELU(LayerNorm(W_proj · fused_CBP + b_proj)) ∈ R^{256}

    Compresses 8,192D sketch space into 256 dimensions to prevent dimensional
    asymmetry and gradient domination over affective features.
    """
    return F.gelu(norm_layer(proj_layer(fused_cbp)))


def assemble_hybrid_299d_feature(
    fused_proj: torch.Tensor,
    fused_emo: torch.Tensor,
    delta: torch.Tensor,
    sarcasm: torch.Tensor,
) -> torch.Tensor:
    """
    Assembly of the Multi-Scale Hybrid Bottleneck vector (Paper Section 3, Stage 5):
        x_299 = [ fused_proj (256D) || fused_emo (36D) || Δ (6D) || P_sarc (1D) ] ∈ R^{299}
    """
    return torch.cat([fused_proj, fused_emo, delta, sarcasm], dim=-1)



class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) for Domain-Adversarial Training (DANN; Ganin & Lempitsky, 2015).

    Mathematical Formulation:
        Forward pass (Identity operator):
            R_α(x) = x
        Backward pass (Gradient negation with dynamic schedule α):
            dL / dx = -α * (dL / dy)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL) for Domain-Adversarial Training (DANN).
    Forward pass: Identity mapping R_α(x) = x.
    Backward pass: Multiplies gradients by -alpha to encourage domain-invariant representations.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, alpha: float | None = None) -> torch.Tensor:
        a = alpha if alpha is not None else self.alpha
        return GradientReversalFunction.apply(x, a)



@dataclass
class DetectorOutput:
    logit: torch.Tensor             # (B, 1) — raw, no sigmoid
    emotion_a: torch.Tensor         # (B, 6) — audio emotion logits
    emotion_b: torch.Tensor         # (B, 6) — visual emotion logits
    sarcasm: torch.Tensor           # (B, 1) — raw sarcasm logit
    domain_logits: Optional[torch.Tensor] = None  # (B, 5) — domain classifier logits (DANN)


class DeepfakeDetector(nn.Module):
    """
    Full detection model.

    Backbone components (Wav2Vec2, BERT, ViT) are loaded lazily on first call to
    forward() so that forward_from_features() can be used without loading 1+ GB of
    pretrained weights during Phase 1 training.
    """

    Z_AT_DIM = 1536   # 768 (Wav2Vec) + 768 (BERT)
    Z_V_DIM  = 768    # ViT CLS token
    CBP_DIM  = 8192   # Compact Bilinear Pooling output dimension

    def __init__(
        self,
        wav2vec_model: str = "facebook/wav2vec2-base",
        bert_model:    str = "bert-base-uncased",
        vit_model:     str = "google/vit-base-patch16-224",
        n_emotions:    int = 6,
        cbp_dim:       int = 8192,
        dropout_heads: float = 0.3,
        dropout_cls:   float = 0.4,
        classifier_mode: str = "baseline",
        proj_dim:      Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if proj_dim is not None:
            cbp_dim = proj_dim
        self._wav2vec_name = wav2vec_model
        self._bert_name    = bert_model
        self._vit_name     = vit_model
        self.classifier_mode = classifier_mode

        # Detection components (always present)
        self.emotion_head_a  = EmotionHeadA(self.Z_AT_DIM, n_emotions, dropout_heads)
        self.emotion_head_b  = EmotionHeadB(self.Z_V_DIM,  n_emotions, dropout_heads)
        self.sarcasm_head    = SarcasmHead(self.Z_AT_DIM, dropout=dropout_heads)
        self.bilinear_fusion = BilinearFusion(self.Z_AT_DIM, self.Z_V_DIM, cbp_dim)

        if classifier_mode == "mismatch_only":
            fused_dim = n_emotions + 1
        elif classifier_mode == "emotion_bilinear":
            fused_dim = 36 + n_emotions + 1
        elif classifier_mode == "bottleneck":
            self.bilinear_proj = nn.Linear(cbp_dim, 256)
            self.proj_ln = nn.LayerNorm(256)
            fused_dim = 256 + 36 + n_emotions + 1
        elif classifier_mode == "high_dropout":
            self.high_dropout = nn.Dropout(0.85)
            fused_dim = cbp_dim + n_emotions + 1
        else:  # baseline
            fused_dim = cbp_dim + n_emotions + 1

        self.classifier = ClassifierMLP(fused_dim, dropout=dropout_cls)

        # Domain Adversarial Classifier (DANN / GRL) to neutralize dataset shortcuts
        self.grl = GradientReversalLayer(alpha=1.0)
        domain_in_dim = 256 if classifier_mode == "bottleneck" else (cbp_dim if classifier_mode in ("baseline", "high_dropout") else fused_dim)
        self.domain_classifier = nn.Sequential(
            nn.Linear(domain_in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5),  # 5 Source Domains: {0: CREMA-D, 1: MELD, 2: MOSEI, 3: MUStARD, 4: SYNTHETIC}
        )

        # Cross-Modal Attention between audio/text and visual keyframes
        self.cross_attn_at = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.cross_attn_v  = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.norm_at       = nn.LayerNorm(768)
        self.norm_v        = nn.LayerNorm(768)

        # Temporal GRU Aggregator over visual keyframe sequence
        self.vit_gru       = nn.GRU(input_size=768, hidden_size=768, num_layers=2, batch_first=True)

        # Backbones — loaded on demand
        self._wav2vec: Optional[nn.Module] = None
        self._bert:    Optional[nn.Module] = None
        self._vit:     Optional[nn.Module] = None
        self._backbones_loaded = False

    # ── Backbone management ────────────────────────────────────────────────────

    def load_backbones(self) -> None:
        """Instantiate Wav2Vec2, BERT, ViT. Safe to call multiple times."""
        if self._backbones_loaded:
            return
        device = next(self.classifier.parameters()).device
        from transformers import Wav2Vec2Model, BertModel, ViTModel
        self._wav2vec = Wav2Vec2Model.from_pretrained(self._wav2vec_name).to(device)
        self._wav2vec.gradient_checkpointing_disable()
        self._bert    = BertModel.from_pretrained(self._bert_name).to(device)
        self._vit     = ViTModel.from_pretrained(self._vit_name).to(device)
        self.wav2vec2 = self._wav2vec
        self.bert     = self._bert
        self.vit      = self._vit
        self._backbones_loaded = True

    def freeze_backbones(self) -> None:
        self.load_backbones()
        for m in (self._wav2vec, self._bert, self._vit):
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_backbones(self) -> None:
        if not self._backbones_loaded:
            raise RuntimeError("Call load_backbones() before unfreeze_backbones().")
        for m in (self._wav2vec, self._bert, self._vit):
            for p in m.parameters():
                p.requires_grad = True

    def unfreeze_top_layers(self, n_layers: int = 2) -> None:
        """Unfreeze only the top N transformer blocks of each backbone.
        Much lower VRAM than full unfreeze — gradients only for last N layers."""
        if not self._backbones_loaded:
            raise RuntimeError("Call load_backbones() before unfreeze_top_layers().")
        # Keep everything frozen first
        for m in (self._wav2vec, self._bert, self._vit):
            for p in m.parameters():
                p.requires_grad = False

        # Wav2Vec2: encoder.layers[-n_layers:]
        for layer in self._wav2vec.encoder.layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

        # BERT: encoder.layer[-n_layers:]
        for layer in self._bert.encoder.layer[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self._bert.pooler.parameters():
            p.requires_grad = True

        # ViT unfreezing (supports both self._vit.encoder.layer and self._vit.layers architectures)
        vit_layers = None
        vit_encoder = getattr(self._vit, "encoder", None)
        if vit_encoder is None and hasattr(self._vit, "vit"):
            vit_encoder = getattr(self._vit.vit, "encoder", None)
            
        if vit_encoder is not None and hasattr(vit_encoder, "layer"):
            vit_layers = vit_encoder.layer
        elif hasattr(self._vit, "layers"):
            vit_layers = self._vit.layers
            
        if vit_layers is None:
            raise AttributeError("Could not find ViT layers or encoder module. Check transformers library version.")
            
        for layer in vit_layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
                
        vit_layernorm = getattr(self._vit, "layernorm", None)
        if vit_layernorm is None and hasattr(self._vit, "vit"):
            vit_layernorm = getattr(self._vit.vit, "layernorm", None)
            
        if vit_layernorm is not None:
            for p in vit_layernorm.parameters():
                p.requires_grad = True

    def enable_gradient_checkpointing(self) -> None:
        """Trade compute for memory — recompute activations on backward pass."""
        if not self._backbones_loaded:
            raise RuntimeError("Call load_backbones() first.")
        self._wav2vec.gradient_checkpointing_enable()
        self._bert.gradient_checkpointing_enable()
        self._vit.gradient_checkpointing_enable()

    # ── Core detection logic ───────────────────────────────────────────────────

    def _detect(
        self,
        z_at: torch.Tensor,
        z_v: torch.Tensor,
        grl_alpha: float = 1.0,
        z_at_emo: Optional[torch.Tensor] = None,
        has_speech: bool = True,
    ) -> DetectorOutput:
        """
        Shared detection logic after feature extraction.

        Mathematical Formulations (Paper Section 3):
        1. Modality Emotion Distributions:
               P_A = Softmax(EmotionHeadA(Z_at)) ∈ Δ^5
               P_B = Softmax(EmotionHeadB(Z_v))  ∈ Δ^5
        2. Cross-Modal Emotion Discrepancy Vector:
               Δ_k = |P_{A, k} - P_{B, k}|,   k ∈ {1, ..., 6}
        3. Joint Affect Co-occurrence State:
               fused_emo = vec(P_A ⊗ P_B) = vec(P_A P_B^T) ∈ R^{36}
        4. Compact Bilinear Bottleneck Projection:
               fused_proj = GELU(LayerNorm(W_proj · fused_CBP)) ∈ R^{256}
        5. 299D Hybrid Classifier Representation:
               x_299 = [fused_proj (256D) || fused_emo (36D) || Δ (6D) || P_sarc (1D)] ∈ R^{299}
        6. Detection Prediction:
               logit = ClassifierMLP(x_299) ∈ R^1
        """
        emo_source = z_at_emo if z_at_emo is not None else z_at
        emo_a = self.emotion_head_a(emo_source)  # (B, 6) raw audio emotion logits
        emo_b = self.emotion_head_b(z_v)         # (B, 6) raw visual emotion logits
        sarc  = self.sarcasm_head(z_at)          # (B, 1) raw sarcasm logit

        fused = self.bilinear_fusion(z_at, z_v)  # (B, 8192) L2-normalized CBP feature

        # Step 1: Modality emotion probability distributions: P_k = exp(z_k) / sum_j exp(z_j)
        prob_b = compute_emotion_probabilities(emo_b, dim=-1)
        if not has_speech:
            B = z_at.size(0)
            # When speech is absent (silent/quiet video), vocal emotion is 100% neutral,
            # sarcasm is 0, and cross-modal incongruence Δ = 0.
            prob_a = torch.zeros(B, 6, device=z_at.device)
            prob_a[:, 0] = 1.0  # 100% neutral voice
            sarc = torch.zeros(B, 1, device=z_at.device)
            delta = torch.zeros(B, 6, device=z_at.device)
        else:
            prob_a = compute_emotion_probabilities(emo_a, dim=-1)
            # Step 2: Cross-Modal Emotion Discrepancy Vector: Δ = |P_A - P_B| ∈ [0, 1]^6
            delta = compute_emotion_discrepancy(prob_a, prob_b)

        if self.classifier_mode == "mismatch_only":
            # Ablation mode: classify strictly on discrepancy Δ and sarcasm score
            combined = torch.cat([delta, sarc], dim=-1)
            domain_in = combined
        elif self.classifier_mode == "emotion_bilinear":
            # Outer product state: M_emo = P_A ⊗ P_B ∈ R^{6 × 6} → vec(M) ∈ R^{36}
            fused_emo = compute_affect_cooccurrence_matrix(prob_a, prob_b)
            combined = torch.cat([fused_emo, delta, sarc], dim=-1)
            domain_in = combined
        elif self.classifier_mode == "bottleneck":
            # Stage 5 Master Model (299D Hybrid Architecture):
            # 1. Affective Co-Occurrence Matrix: vec(P_A ⊗ P_B) ∈ R^{36}
            fused_emo = compute_affect_cooccurrence_matrix(prob_a, prob_b)
            # 2. Sub-symbolic CBP bottleneck: fused_proj = GELU(LayerNorm(W_proj * fused)) ∈ R^{256}
            fused_proj = compute_bottleneck_projection(fused, self.bilinear_proj, self.proj_ln)
            # 3. 299D Hybrid concatenation: x_299 = [fused_proj || fused_emo || delta || sarc] ∈ R^{299}
            combined = assemble_hybrid_299d_feature(fused_proj, fused_emo, delta, sarc)
            domain_in = fused_proj
        elif self.classifier_mode == "high_dropout":
            fused_drop = self.high_dropout(fused)
            combined = torch.cat([fused_drop, delta, sarc], dim=-1)
            domain_in = fused
        else:  # baseline
            combined = torch.cat([fused, delta, sarc], dim=-1)
            domain_in = fused

        # Final classification MLP: logit ∈ R^1
        logit = self.classifier(combined)


        # Domain adversarial classifier through Gradient Reversal Layer (DANN GRL)
        rev_feat = self.grl(domain_in, alpha=grl_alpha)
        domain_logits = self.domain_classifier(rev_feat)      # (B, 5)

        return DetectorOutput(
            logit=logit,
            emotion_a=emo_a,
            emotion_b=emo_b,
            sarcasm=sarc,
            domain_logits=domain_logits,
        )

    # ── Phase 1 path (cached features) ────────────────────────────────────────

    def forward_from_features(
        self,
        z_at: torch.Tensor,
        z_v:  torch.Tensor,
        grl_alpha: float = 1.0,
        z_at_emo: Optional[torch.Tensor] = None,
        has_speech: bool = True,
    ) -> DetectorOutput:
        """
        Phase 1 forward pass - takes precomputed Z_at (B,1536) and Z_v (B,768) or (B,8,768).
        Does NOT require backbones to be loaded.
        """
        has_cross_attn = getattr(self, "_has_cross_attn", True)
        if not has_cross_attn or z_v.ndim == 2:
            z_v_vec = z_v if z_v.ndim == 2 else z_v.mean(dim=1)
            return self._detect(z_at, z_v_vec, grl_alpha=grl_alpha, z_at_emo=z_at_emo, has_speech=has_speech)

        w2v_emb = z_at[:, :768]
        bert_emb = z_at[:, 768:]
        if z_v.ndim == 3:
            z_v_seq = z_v                               # Genuine keyframe sequence (B, K, 768)
        else:
            raise ValueError(f"Unexpected z_v shape: {z_v.shape}")
        return self._forward_impl(w2v_emb, bert_emb, z_v_seq, grl_alpha=grl_alpha, z_at_emo=z_at_emo, has_speech=has_speech)

    # ── Phase 2 path (end-to-end) ─────────────────────────────────────────────

    def forward(
        self,
        audio_values:    torch.Tensor,            # (B, T_audio)
        input_ids:       torch.Tensor,            # (B, seq_len)
        attention_mask:  torch.Tensor,            # (B, seq_len)
        keyframe_pixels: torch.Tensor,            # (B, K, 3, 224, 224)
        grl_alpha:       float = 1.0,
        z_at_emo:        Optional[torch.Tensor] = None,
        has_speech:      bool = True,
    ) -> DetectorOutput:
        """
        Phase 2 end-to-end forward pass.
        Runs Wav2Vec2 + BERT for audio-text, ViT for visual, followed by Cross-Attention and GRU.
        Call load_backbones() once before using this path.
        """
        if not self._backbones_loaded:
            raise RuntimeError(
                "Backbones not loaded. Call model.load_backbones() first, "
                "or use forward_from_features() for Phase 1 training."
            )

        # Wav2Vec2 conv layers are not FP16-safe — run in float32 outside autocast
        with torch.amp.autocast("cuda", enabled=False):
            w2v_out = self._wav2vec(audio_values.float()).last_hidden_state  # (B, T', 768)
            w2v_emb = w2v_out.mean(dim=1).to(audio_values.dtype)            # (B, 768)

        # Text branch — BERT on ASR transcript tokens
        bert_out = self._bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_emb = bert_out.last_hidden_state[:, 0, :]            # (B, 768) CLS token

        # Visual branch — ViT on K keyframes per clip
        B, K, C, H, W = keyframe_pixels.shape
        frames = keyframe_pixels.view(B * K, C, H, W)
        vit_out = self._vit(pixel_values=frames).last_hidden_state[:, 0, :]  # (B*K, 768)
        z_v_seq = vit_out.view(B, K, 768)                        # (B, K, 768)

        return self._forward_impl(w2v_emb, bert_emb, z_v_seq, grl_alpha=grl_alpha, z_at_emo=z_at_emo, has_speech=has_speech)

    def _forward_impl(
        self,
        w2v_emb: torch.Tensor,
        bert_emb: torch.Tensor,
        z_v_seq: torch.Tensor,
        grl_alpha: float = 1.0,
        z_at_emo: Optional[torch.Tensor] = None,
        has_speech: bool = True,
    ) -> DetectorOutput:
        if not has_speech:
            # Genuine silence: visual sequence is purely visual, no cross-modal noise contamination
            gru_out, _ = self.vit_gru(z_v_seq)
            z_v = gru_out[:, -1, :]
            z_at_clean = torch.zeros(z_v.size(0), self.Z_AT_DIM, device=z_v.device)
            return self._detect(z_at_clean, z_v, grl_alpha=grl_alpha, z_at_emo=z_at_clean, has_speech=False)

        # Pure audio-text embedding before cross-attention visual contamination
        z_at_clean = torch.cat([w2v_emb, bert_emb], dim=-1)

        # 1. Multi-Head Cross-Modal Attention
        audio_text_seq = torch.stack([w2v_emb, bert_emb], dim=1)  # (B, 2, 768)

        # Visual queries audio/text
        z_v_attn, _ = self.cross_attn_v(query=z_v_seq, key=audio_text_seq, value=audio_text_seq)
        z_v_seq = self.norm_v(z_v_seq + z_v_attn)

        # Audio/text queries visual
        at_attn, _ = self.cross_attn_at(query=audio_text_seq, key=z_v_seq, value=z_v_seq)
        audio_text_seq = self.norm_at(audio_text_seq + at_attn)

        # Split back to acoustic and linguistic
        w2v_emb_fused = audio_text_seq[:, 0, :]
        bert_emb_fused = audio_text_seq[:, 1, :]
        z_at_fused = torch.cat([w2v_emb_fused, bert_emb_fused], dim=-1)            # (B, 1536)

        # 2. Temporal GRU Aggregation on Visual Sequence
        gru_out, _ = self.vit_gru(z_v_seq)                       # (B, K, 768)
        z_v = gru_out[:, -1, :]                                  # Take last hidden state (B, 768)

        # 3. Detect
        return self._detect(z_at_fused, z_v, grl_alpha=grl_alpha, z_at_emo=z_at_emo if z_at_emo is not None else z_at_clean, has_speech=has_speech)

    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False):
        """
        Custom load_state_dict to handle:
        - Auto-instantiation of backbones if state_dict has fine-tuned backbones.
        - Auto-adaptation between 299-D bottleneck and 8199-D baseline heads.
        - Bidirectional ViT layer naming remapping across transformers versions.
        """
        # Auto-instantiate backbones if checkpoint contains backbone weights
        if not self._backbones_loaded and any(k.startswith(("wav2vec2.", "bert.", "vit.", "_wav2vec.", "_bert.", "_vit.")) for k in state_dict.keys()):
            self.load_backbones()

        # Auto-detect bottleneck (299-D) vs baseline (8199-D) head to prevent size mismatch
        if "classifier.fc1.weight" in state_dict:
            ckpt_in_features = state_dict["classifier.fc1.weight"].shape[1]
            if ckpt_in_features == 299 and self.classifier_mode != "bottleneck":
                device = next(self.classifier.parameters()).device
                self.classifier_mode = "bottleneck"
                self.bilinear_proj = nn.Linear(8192, 256).to(device)
                self.proj_ln = nn.LayerNorm(256).to(device)
                self.classifier = ClassifierMLP(299, dropout=0.4).to(device)
                self.domain_classifier = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.LayerNorm(128),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 5),
                ).to(device)
            elif ckpt_in_features == 8199 and self.classifier_mode != "baseline":
                device = next(self.classifier.parameters()).device
                self.classifier_mode = "baseline"
                if hasattr(self, "bilinear_proj"):
                    del self.bilinear_proj
                if hasattr(self, "proj_ln"):
                    del self.proj_ln
                self.classifier = ClassifierMLP(8199, dropout=0.4).to(device)
                self.domain_classifier = nn.Sequential(
                    nn.Linear(8192, 128),
                    nn.LayerNorm(128),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 5),
                ).to(device)

        remapped_state = {}
        vit_has_encoder = False
        if hasattr(self, "_vit") and self._vit is not None:
            vit_has_encoder = hasattr(self._vit, "encoder") or (hasattr(self._vit, "vit") and hasattr(self._vit.vit, "encoder"))

        for k, v in state_dict.items():
            new_k = k
            for prefix in ["_vit.", "vit."]:
                if vit_has_encoder and new_k.startswith(prefix + "layers."):
                    parts = new_k.split(".")
                    i = parts[2]
                    rest = ".".join(parts[3:])
                    mapping = {
                        "attention.q_proj.weight": f"{prefix}encoder.layer.{i}.attention.attention.query.weight",
                        "attention.q_proj.bias":   f"{prefix}encoder.layer.{i}.attention.attention.query.bias",
                        "attention.k_proj.weight": f"{prefix}encoder.layer.{i}.attention.attention.key.weight",
                        "attention.k_proj.bias":   f"{prefix}encoder.layer.{i}.attention.attention.key.bias",
                        "attention.v_proj.weight": f"{prefix}encoder.layer.{i}.attention.attention.value.weight",
                        "attention.v_proj.bias":   f"{prefix}encoder.layer.{i}.attention.attention.value.bias",
                        "attention.o_proj.weight": f"{prefix}encoder.layer.{i}.attention.output.dense.weight",
                        "attention.o_proj.bias":   f"{prefix}encoder.layer.{i}.attention.output.dense.bias",
                        "layernorm_before.weight": f"{prefix}encoder.layer.{i}.layernorm_before.weight",
                        "layernorm_before.bias":   f"{prefix}encoder.layer.{i}.layernorm_before.bias",
                        "layernorm_after.weight":  f"{prefix}encoder.layer.{i}.layernorm_after.weight",
                        "layernorm_after.bias":    f"{prefix}encoder.layer.{i}.layernorm_after.bias",
                        "mlp.fc1.weight":          f"{prefix}encoder.layer.{i}.intermediate.dense.weight",
                        "mlp.fc1.bias":            f"{prefix}encoder.layer.{i}.intermediate.dense.bias",
                        "mlp.fc2.weight":          f"{prefix}encoder.layer.{i}.output.dense.weight",
                        "mlp.fc2.bias":            f"{prefix}encoder.layer.{i}.output.dense.bias",
                    }
                    if rest in mapping:
                        new_k = mapping[rest]
                elif (not vit_has_encoder) and new_k.startswith(prefix + "encoder.layer."):
                    parts = new_k.split(".")
                    i = parts[3]
                    rest = ".".join(parts[4:])
                    inv_mapping = {
                        "attention.attention.query.weight": f"{prefix}layers.{i}.attention.q_proj.weight",
                        "attention.attention.query.bias":   f"{prefix}layers.{i}.attention.q_proj.bias",
                        "attention.attention.key.weight":   f"{prefix}layers.{i}.attention.k_proj.weight",
                        "attention.attention.key.bias":     f"{prefix}layers.{i}.attention.k_proj.bias",
                        "attention.attention.value.weight": f"{prefix}layers.{i}.attention.v_proj.weight",
                        "attention.attention.value.bias":   f"{prefix}layers.{i}.attention.v_proj.bias",
                        "attention.output.dense.weight":    f"{prefix}layers.{i}.attention.o_proj.weight",
                        "attention.output.dense.bias":      f"{prefix}layers.{i}.attention.o_proj.bias",
                        "layernorm_before.weight":          f"{prefix}layers.{i}.layernorm_before.weight",
                        "layernorm_before.bias":            f"{prefix}layers.{i}.layernorm_before.bias",
                        "layernorm_after.weight":           f"{prefix}layers.{i}.layernorm_after.weight",
                        "layernorm_after.bias":             f"{prefix}layers.{i}.layernorm_after.bias",
                        "intermediate.dense.weight":        f"{prefix}layers.{i}.mlp.fc1.weight",
                        "intermediate.dense.bias":          f"{prefix}layers.{i}.mlp.fc1.bias",
                        "output.dense.weight":              f"{prefix}layers.{i}.mlp.fc2.weight",
                        "output.dense.bias":                f"{prefix}layers.{i}.mlp.fc2.bias",
                    }
                    if rest in inv_mapping:
                        new_k = inv_mapping[rest]
            remapped_state[new_k] = v

        try:
            return super().load_state_dict(remapped_state, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(remapped_state, strict=strict)

    # ── Convenience ───────────────────────────────────────────────────────────

    @staticmethod
    def p_fake(logit: torch.Tensor) -> torch.Tensor:
        """
        Convert raw logit → P(fake) ∈ [0, 1] via first-principles logistic sigmoid:
            P(fake) = 1 / (1 + exp(-logit))
        """
        return compute_sigmoid(logit)


if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("   DEEPSENTINEL MULTIMODAL BOTTLENECK & 299D FEATURE ASSEMBLY")
    print("=" * 75)
    print("Architecture: Audio-Text + Visual CBP + Affective Incongruence Bottleneck\n")

    torch.manual_seed(42)
    B = 2
    d_at = 1536
    d_v = 768

    model = DeepfakeDetector(classifier_mode="bottleneck")
    model.eval()

    z_at = torch.randn(B, d_at)
    z_v = torch.randn(B, d_v)

    with torch.no_grad():
        # Step 1: Forward heads
        emo_a = model.emotion_head_a(z_at)
        emo_b = model.emotion_head_b(z_v)
        sarc = model.sarcasm_head(z_at)

        prob_a = compute_emotion_probabilities(emo_a)
        prob_b = compute_emotion_probabilities(emo_b)

        print("[STEP 1] EMOTION PROBABILITY DISTRIBUTIONS (SOFTMAX P in Delta^5):")
        print(f"  P_A (Acoustic-Text) : {prob_a[0].numpy().round(4)} | Sum: {prob_a[0].sum().item():.6f}")
        print(f"  P_V (Visual Face)   : {prob_b[0].numpy().round(4)} | Sum: {prob_b[0].sum().item():.6f}\n")

        # Step 2: Discrepancy Vector and Norm
        delta = compute_emotion_discrepancy(prob_a, prob_b)
        delta_norm = compute_discrepancy_magnitude(delta)
        print("[STEP 2] AFFECTIVE INCONGRUENCE FORMULA DERIVATIONS:")
        print(f"  Delta = |P_A - P_V| : {delta[0].numpy().round(4)} in R^6")
        print(f"  ||Delta||_2 norm    : {delta_norm[0].item():.6f} = sqrt(sum Delta_k^2)\n")

        # Step 3: Affect Co-occurrence Outer Product
        fused_emo = compute_affect_cooccurrence_matrix(prob_a, prob_b)
        print("[STEP 3] AFFECT CO-OCCURRENCE KRONECKER OUTER PRODUCT:")
        print(f"  fused_emo = vec(P_A (x) P_V) : shape {tuple(fused_emo.shape)} in R^36")
        print(f"  Vector sum                   : {fused_emo[0].sum().item():.6f} (= 1.000000)\n")

        # Step 4: CBP and Bottleneck Projection
        fused_cbp = model.bilinear_fusion(z_at, z_v)
        fused_proj = compute_bottleneck_projection(fused_cbp, model.bilinear_proj, model.proj_ln)
        print("[STEP 4] CBP & BOTTLENECK PROJECTION:")
        print(f"  fused_cbp  : shape {tuple(fused_cbp.shape)} in R^8192 (Unit L2-norm)")
        print(f"  fused_proj : shape {tuple(fused_proj.shape)} in R^256 (GELU + LayerNorm)\n")

        # Step 5: Assembly of 299D Hybrid Representation
        x_299 = assemble_hybrid_299d_feature(fused_proj, fused_emo, delta, sarc)
        print("[STEP 5] 299D HYBRID FEATURE VECTOR ASSEMBLY (PAPER SECTION 3.4):")
        print(f"  Formula: x_299 = [ fused_proj (256) || fused_emo (36) || Delta (6) || sarc (1) ]")
        print(f"  Assembly: 256 + 36 + 6 + 1 = {256 + 36 + 6 + 1} dimensions")
        print(f"  Resulting Tensor Shape     : {tuple(x_299.shape)}\n")

        # Step 6: Classification and Sigmoid Probability
        out = model._detect(z_at, z_v)
        prob_fake = compute_detection_probability(out.logit)
        print("[STEP 6] RAW LOGIT & FIRST-PRINCIPLES SIGMOID DETECTION:")
        for i in range(B):
            print(f"  Sample {i+1}: Logit = {out.logit[i].item():+.4f} --> P(fake) = 1/(1 + exp(-z)) = {prob_fake[i].item():.6f} ({prob_fake[i].item()*100:.2f}%)")
    print("=" * 75 + "\n")


