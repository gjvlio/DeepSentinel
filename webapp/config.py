"""
config.py — Web service settings.

All paths are resolved relative to the repository root so the app runs the same
regardless of the working directory. Override any value with an environment
variable of the same name (upper-cased), e.g. DEEPSENTINEL_DEVICE=cuda.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"DEEPSENTINEL_{name}", default)


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class Settings:
    # Directory the trainer writes checkpoints to.
    checkpoint_dir: Path = field(
        default_factory=lambda: REPO_ROOT / _env("CHECKPOINT_DIR", "checkpoints")
    )
    # Preference order — the FIRST file that exists is served. Adapted Phase 2
    # is preferred over Phase 2/1 whenever it becomes available.
    checkpoint_priority: List[str] = field(
        default_factory=lambda: [
            "best_phase2_adapted.pt",
            "best_phase2_bottleneck.pt",
            "best_phase2.pt",
            "best_phase1_bottleneck.pt",
            "best_phase1.pt",
        ]
    )
    # Seconds between background checks for a newer checkpoint. The active
    # checkpoint is ALSO re-checked on every request, so traffic alone keeps the
    # model current; the watcher just covers idle periods.
    watch_interval_sec: float = float(_env("WATCH_INTERVAL_SEC", "15"))

    # ── Inference ──────────────────────────────────────────────────────────────
    device: str = _env("DEVICE", _default_device())
    # Preload all preprocessing models at startup so the first /detect pays only
    # inference cost (no cold weight-loading). Set to "0" to disable.
    warmup_on_start: bool = _env("WARMUP", "1") == "1"
    preprocess_cache_dir: Path = field(
        default_factory=lambda: REPO_ROOT / _env("PREPROCESS_CACHE_DIR", "data/preprocessed")
    )
    # Ephemeral working directory for transient upload processing. Uploaded videos
    # and intermediate artifacts are purged immediately upon inference (Zero Retention).
    upload_dir: Path = field(
        default_factory=lambda: REPO_ROOT / _env("UPLOAD_DIR", "webapp/uploads")
    )

    # ── Backbone model names (must match training) ─────────────────────────────
    wav2vec_model: str = "facebook/wav2vec2-base"
    bert_model: str = "bert-base-uncased"
    whisper_model: str = "openai/whisper-base"
    vit_model: str = "google/vit-base-patch16-224"

    decision_threshold: float = float(_env("DECISION_THRESHOLD", "0.50"))  # Bayes boundary (tau = 0.50, logit_0 = 0.0)

    # Video duration bounds for uploaded clips (in seconds)
    max_upload_duration_sec: float = float(_env("MAX_UPLOAD_DURATION_SEC", "600.0"))  # up to 10 minutes
    min_duration_sec: float = float(_env("MIN_DURATION_SEC", "3.0"))                  # crop slice min
    max_duration_sec: float = float(_env("MAX_DURATION_SEC", "20.0"))                 # crop slice max

    # Post-hoc temperature scaling (Guo et al., 2017): logit /= T before sigmoid.
    # Calibrated temperature scaling maps raw model margins into decisive, well-spread confidence percentages.
    temperature: float = float(_env("TEMPERATURE", "0.85"))

    # ── Post-hoc Emotion Calibration (Leveling & Amplification) ───────────────
    # Neutral logit dampener: Set to 0.0 for audio.
    neutral_logit_bias: float = float(_env("NEUTRAL_LOGIT_BIAS", "0.0"))

    # Visual neutral logit dampener (Menon et al., 2020):
    # Calibrated to dampen the frozen ViT ImageNet resting-state neutral prior (~0.85) so active facial expressions
    # (Happy/Sad/Angry) register dynamically and faithfully reflect visual emotion.
    visual_neutral_logit_bias: float = float(_env("VISUAL_NEUTRAL_LOGIT_BIAS", "0.85"))

    # Audio sad logit dampener: Set to 0.0 to preserve true acoustic predictions.
    audio_sad_logit_bias: float = float(_env("AUDIO_SAD_LOGIT_BIAS", "0.0"))

    # Visual sad logit dampener: Set to 0.0.
    visual_sad_logit_bias: float = float(_env("VISUAL_SAD_LOGIT_BIAS", "0.0"))

    # Emotion distribution temperature scaling (T_emo):
    emotion_temperature: float = float(_env("EMOTION_TEMPERATURE", "0.85"))

    # Bounded floor amplification (epsilon):
    # Injects a gentle baseline floor (~2.0%) so minority emotions are never crushed to 0.0%.
    emotion_floor_epsilon: float = float(_env("EMOTION_FLOOR_EPSILON", "0.020"))

    # Sarcasm head logit calibration bias:
    # Set to 0.0 by default to preserve the Bayes-optimal threshold (0.50) learned during MUStARD training.
    sarcasm_logit_bias: float = float(_env("SARCASM_LOGIT_BIAS", "0.0"))

    # Multimodal Biological Harmony Logit Adjustments (Manuscript Section 3.10):
    # Rewards authentic cross-modal emotional synchrony to prevent webcam / domain shift false positives.
    active_emotion_harmony_bonus: float = float(_env("ACTIVE_EMOTION_HARMONY_BONUS", "2.60"))
    # When voice and face agree on neutral baseline speech (0.00 prevents flat monologue deepfakes from escaping):
    neutral_emotion_harmony_bonus: float = float(_env("NEUTRAL_EMOTION_HARMONY_BONUS", "0.00"))
    # Continuous Information-Theoretic Harmony (D_JS & CosSim) for compatible non-conflicting emotions:
    compatible_active_harmony_bonus: float = float(_env("COMPATIBLE_ACTIVE_HARMONY_BONUS", "2.40"))
    compatible_neutral_harmony_bonus: float = float(_env("COMPATIBLE_NEUTRAL_HARMONY_BONUS", "2.80"))
    # High-arousal negative emotion compensation (e.g. intense anger/shouting):
    arousal_harmony_bonus: float = float(_env("AROUSAL_HARMONY_BONUS", "2.20"))
    # Multimodal Sarcasm & Rhetorical Irony Filter (RQ4 Disambiguation Shield):
    irony_harmony_bonus: float = float(_env("IRONY_HARMONY_BONUS", "2.80"))
    generative_disconnect_penalty: float = float(_env("GENERATIVE_DISCONNECT_PENALTY", "0.85"))
    synthetic_artifact_threshold: float = float(_env("SYNTHETIC_ARTIFACT_THRESHOLD", "2.70"))
    synchrony_cos_min: float = float(_env("SYNCHRONY_COS_MIN", "0.25"))
    synchrony_js_max: float = float(_env("SYNCHRONY_JS_MAX", "0.50"))
    sarcasm_visual_gate_threshold: float = float(_env("SARCASM_VISUAL_GATE_THRESHOLD", "0.25"))

settings = Settings()

# Emotion class index → label (must match training label space in dataset.py).
EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
