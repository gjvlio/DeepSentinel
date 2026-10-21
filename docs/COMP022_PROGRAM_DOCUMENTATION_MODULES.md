# COMP 022 – Software Engineering 2: Program Documentation
**Thesis Group 10 — DeepSentinel Project**

This document strictly adheres to the **COMP 022 – Software Engineering 2 Program Documentation Template**. It documents the **actual thesis system architecture** for:
1. **Module 1: Synthetic Dataset Generation Architecture (Tracks 1–3)** — `src/track1/track1_generate.py`, `src/track2/track2_generate.py`, and `src/track3/track3_generate.py`.
2. **Module 2: Multimodal Preprocessing Architecture (Feature Ingestion Pipeline)** — `src/preprocessing/pipeline.py` (supported by `audio.py`, `visual.py`, and `filters.py`).

All temporary UI and companion webapp code (`generator/`, `webapp/`) are entirely excluded. Each module features the **Header Comment Block**, **Software Engineering Considerations**, and an interleaved **Code Snippet (from imports starting) followed immediately by technical explanations** for every function and class.

---

# Table of Contents
1. [System Architecture & Module Overview](#1-system-architecture--module-overview)
2. [Module 1: Synthetic Deepfake Generation Architecture (Tracks 1–3)](#2-module-1-synthetic-deepfake-generation-architecture-tracks-13)
   - [2.1 Header Comment Block: Generation Architecture](#21-header-comment-block-generation-architecture)
   - [2.2 Considerations in Writing the Code (Generation Architecture)](#22-considerations-in-writing-the-code-generation-architecture)
   - [Snippet 1: Imports, PyTorch Compatibility Patch, and Logger](#snippet-1-imports-pytorch-compatibility-patch-and-logger)
   - [Snippet 2: Emotion Mapping & Reference Audio Auto-Discovery (`build_emotion_refs`)](#snippet-2-emotion-mapping--reference-audio-auto-discovery-build_emotion_refs)
   - [Snippet 3: Low-Level Media Subprocess Helpers (`run_ffmpeg`, `get_duration`, `mux_video_audio`, `convert_flv_to_mp4`)](#snippet-3-low-level-media-subprocess-helpers-run_ffmpeg-get_duration-mux_video_audio-convert_flv_to_mp4)
   - [Snippet 4: Speaker Identity Verification (`XVectorVerifier`)](#snippet-4-speaker-identity-verification-xvectorverifier)
   - [Snippet 5: Track 1 Synthesis Engine (`StyleTTSRVCGenerator`)](#snippet-5-track-1-synthesis-engine-stylettsrvcgenerator)
   - [Snippet 6: Track 2 Lip-Sync Reanimation Engine (`run_wav2lip` & `Track2Generator`)](#snippet-6-track-2-lip-sync-reanimation-engine-run_wav2lip--track2generator)
   - [Snippet 7: Track 3 Talking-Head Neural Engine (`run_sadtalker` & `Track3Generator`)](#snippet-7-track-3-talking-head-neural-engine-run_sadtalker--track3generator)
3. [Module 2: Multimodal Preprocessing Architecture (Feature Ingestion Pipeline)](#3-module-2-multimodal-preprocessing-architecture-feature-ingestion-pipeline)
   - [3.1 Header Comment Block: Preprocessing Architecture](#31-header-comment-block-preprocessing-architecture)
   - [3.2 Considerations in Writing the Code (Preprocessing Architecture)](#32-considerations-in-writing-the-code-preprocessing-architecture)
   - [Snippet 1: Module Header, Imports, and Type Setup](#snippet-1-module-header-imports-and-type-setup)
   - [Snippet 2: Data Structure `ClipFeatures`](#snippet-2-data-structure-clipfeatures)
   - [Snippet 3: Class `PreprocessingPipeline` Initialization & Directory Creation](#snippet-3-class-preprocessingpipeline-initialization--directory-creation)
   - [Snippet 4: Cache Path Generators & Atomic Cache Verification](#snippet-4-cache-path-generators--atomic-cache-verification)
   - [Snippet 5: Audio Demuxing, Whisper ASR Transcription & $Z_{at}$ Extraction](#snippet-5-audio-demuxing-whisper-asr-transcription--z_at-extraction)
   - [Snippet 6: Visual Motion Gating, Face Alignment & $Z_v$ ViT Keyframe Extraction](#snippet-6-visual-motion-gating-face-alignment--z_v-vit-keyframe-extraction)
   - [Snippet 7: Master Processing Orchestrator (`process` method)](#snippet-7-master-processing-orchestrator-process-method)
   - [Snippet 8: High-Throughput DataLoader Accessors (`_load_cached` & `load_features`)](#snippet-8-high-throughput-dataloader-accessors-_load_cached--load_features)
4. [Instructional Walkthrough: Running and Testing Both Modules](#4-instructional-walkthrough-running-and-testing-both-modules)

---

# 1. System Architecture & Module Overview

The DeepSentinel research framework is bifurcated into an **offline adversarial data synthesis stage** (Module 1) and an **online/offline multimodal feature extraction stage** (Module 2):

```
       RAW EMOTIONAL CORPORA (CREMA-D, MELD, CMU-MOSEI)
                             │
     ┌───────────────────────┴───────────────────────┐
     ▼                                               ▼
[ MODULE 1: GENERATION ]                 [ MODULE 2: PREPROCESSING ]
  Track 1: Audio-Swap (StyleTTS2+RVC)      Audio Pipeline: 16kHz WAV -> Whisper ASR
  Track 2: Lip-Sync (Wav2Lip Reanimate)                    -> Wav2Vec2 + BERT = Z_at (1536,)
  Track 3: Talking-Head (SadTalker)        Visual Pipeline: Optical Flow -> RetinaFace
                                                          -> Sharpness -> ViT = Z_v (8, 768)
     │                                               │
     ▼                                               ▼
SYNTHETIC BENCHMARK CLIPS                 UNIFIED TENSOR FEATURE CACHE
(data/synthetic/track{1,2,3}_fakes)       (data/preprocessed/features/z_at & z_v)
```

---

# 2. Module 1: Synthetic Deepfake Generation Architecture (Tracks 1–3)

## 2.1 Header Comment Block: Generation Architecture

```text
========================================================================================
PROGRAM TITLE:
    Multimodal Synthetic Deepfake Generation Suite: Tracks 1, 2, and 3
    (StyleTTS 2 + Applio RVC v2 + Wav2Lip + SadTalker Pipeline)

PROGRAMMERS:
    Thesis Group 10 (DeepSentinel Team)
    - Gel (Data Manager & QA / Integration Lead)
    - JC (Training & Synthesis Architecture Lead)
    - Kina (Preprocessing Lead)
    - Matan (Evaluation & Benchmark Lead)

WHERE THE PROGRAM FITS IN THE GENERAL SYSTEM DESIGNS:
    This module represents the offline adversarial generation pipeline of DeepSentinel.
    To rigorously evaluate multimodal detection models against subtle audiovisual
    manipulations, it generates three controlled manipulation tiers from CREMA-D:
      - Track 1 (Audio-Swap): Replaces vocal audio with synthesized cross-emotion speech
        (StyleTTS 2) cloned to the target actor's vocal identity (RVC v2), retaining
        the original video (introducing cross-modal emotion dissonance and lip mismatch).
      - Track 2 (Lip-Sync Reanimation): Re-synthesizes the mouth and lip region of the
        original face video using Wav2Lip GAN to match the converted audio track
        (eliminating visual lip-sync discrepancies).
      - Track 3 (Full Talking-Head Generation): Synthesizes a completely artificial
        talking-head video from a single portrait frame driven by the synthesized audio
        via SadTalker 3DMM head-pose modeling.
    Outputs are cataloged in manifests used to train and test the DeepSentinel framework.

DATE WRITTEN AND REVISED:
    - Date Written: January 2026
    - Last Revised: October 2026 (Added PyTorch 2.6 weights_only patch, x-vector similarity
      validation, Wav2Lip multi-scale retry, and SadTalker full-crop fallback)

PURPOSE:
    To programmatically generate photorealistic, acoustically consistent manipulated media
    across three controlled difficulty tracks. It automates speech synthesis, voice cloning,
    speaker verification, neural facial animation, and container multiplexing.

DATA STRUCTURES, ALGORITHMS, AND CONTROL:
    1. Data Structures:
       - pandas.DataFrame / Series: Ingests swap-pair manifests defining source video stems,
         actor IDs, target emotions, and transcript strings.
       - Python Dictionaries: Emotion-to-style mappings, reference audio rankings,
         progress tracking checkpoints (`progress_*.json`).
       - pathlib.Path: Cross-platform directory structures (`videos/`, `wav_tmp/`).
    2. Algorithms:
       - StyleTTS 2 Diffusion Synthesis: Style diffusion conditioned on reference emotion WAVs.
       - Applio RVC v2 (RMVPE Pitch Extraction): High-fidelity voice timbre conversion.
       - SpeechBrain X-Vector Cosine Similarity: Rejects synthetic speech if speaker identity
         drops below a threshold ($sim < 0.75$).
       - Wav2Lip GAN Reanimation: Multi-scale downsampling retry ($1 \times, 2 \times, 4 \times, 8 \times$)
         to guarantee face detection under extreme facial poses.
       - SadTalker 3DMM Motion Modeling: Generates 3D head motion and expression coefficients.
    3. Control Flow:
       - Incremental checkpointing with atomic resume (`--resume` flag).
       - Deterministic subprocess execution with 300s/600s timeouts.
       - Secure isolation via Python `tempfile.TemporaryDirectory` blocks.
========================================================================================
```

## 2.2 Considerations in Writing the Code (Generation Architecture)

- **Indentation**: 4 spaces adhering to PEP 8 standards throughout.
- **Inline Comments**: Explains neural generation arguments (e.g., `--protect 0.33`, `--volume_envelope 0.25`, `--f0_method rmvpe` for RVC vocal clarity).
- **Limiting the Use of Global**: Globals are limited to emotion dictionary lookup tables (`EMOTION_STYLE_MAP`) and the module logger. All state is maintained inside the generator classes.
- **Structured Programming**: Clean separation between speech synthesis (`StyleTTS2`), timbre conversion (`Applio RVC`), speaker verification (`XVectorVerifier`), facial reanimation (`Wav2Lip` / `SadTalker`), and container multiplexing (`run_ffmpeg`).
- **Naming Conventions**:
  - Constants: `UPPER_SNAKE_CASE` (e.g., `EMOTION_STYLE_MAP`, `INTENSITY_RANK`)
  - Classes: `PascalCase` (e.g., `StyleTTSRVCGenerator`, `Track2Generator`, `Track3Generator`, `XVectorVerifier`)
  - Functions & Variables: `snake_case` (e.g., `build_emotion_refs`, `mux_video_audio`, `out_stem`)
- **Error Return Conventions**: Returns structured dictionaries with `'status': 'completed'` or `'status': 'failed'` and an associated `'error'` reason string. Failed clips are logged to `failed.csv` without aborting batch processing.
- **Line Length**: Maintained within standard 100-character line lengths.
- **Spacing**: Two blank lines separating classes and top-level functions; single blank line separating internal pipeline steps.
- **Function Length**: Methods are modularized into self-contained units (20–60 lines each).
- **Error Messages**: Captures specific failure roots from stdout/stderr across external CLI tools (`ffmpeg`, `Wav2Lip`, `SadTalker`, `Applio`).

---

### Snippet 1: Imports, PyTorch Compatibility Patch, and Logger

```python
import os
import re
import sys
import json
import argparse
import subprocess
import tempfile
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# PyTorch 2.6 changed torch.load to default weights_only=True, which breaks
# StyleTTS2 checkpoints. Patch back to False (trusted source models only).
_orig_torch_load = torch.load
def _torch_load_compat(*args, weights_only=False, **kwargs):
    return _orig_torch_load(*args, weights_only=weights_only, **kwargs)
torch.load = _torch_load_compat

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)
```

#### Technical Explanation
1. **Standard & Scientific Libraries**:
   - `os`, `sys`, `subprocess`, `tempfile`: Manage command-line subprocess invocations for external deepfake tools, argument parsing, and temporary filesystem isolation.
   - `torch`, `numpy`, `pandas`: Handles neural computation, vector mathematics, and manifest tabular querying.
   - `tqdm`: Renders real-time command-line progress indicators during batch processing of thousands of clips.
2. **PyTorch 2.6 Compatibility Monkey-Patch**:
   - PyTorch 2.6+ defaulted `torch.load(..., weights_only=True)`, which breaks deserialization of complex StyleTTS 2 model checkpoints containing custom pickle classes.
   - The wrapper `_torch_load_compat` preserves compatibility with trusted local model weights without modifying upstream packages.
3. **Structured Logging**: Configures stream-based logging output with uniform timestamping and severity levels.

---

### Snippet 2: Emotion Mapping & Reference Audio Auto-Discovery (`build_emotion_refs`)

```python
# ── Constants ──────────────────────────────────────────────────────────────────

EMOTION_STYLE_MAP = {
    'ANG': 'angry',
    'DIS': 'disgusted',
    'FEA': 'fearful',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad',
}


# ── StyleTTS 2 reference audio auto-discovery ─────────────────────────────────

def build_emotion_refs(cremad_dir: Path) -> dict[str, str]:
    """
    Pick one high-quality reference WAV per emotion from CREMA-D AudioWAV.
    StyleTTS 2 extracts vocal style from these to control emotional prosody.
    Preference order: HI intensity > XX > MD > LO.
    Returns dict: emotion_label -> wav_path  (e.g. 'angry' -> '/.../.wav')
    """
    audio_dir = cremad_dir / 'AudioWAV'
    if not audio_dir.exists():
        log.warning("AudioWAV not found — StyleTTS will use default (neutral) style")
        return {}

    INTENSITY_RANK = {'HI': 0, 'XX': 1, 'MD': 2, 'LO': 3}
    pattern = re.compile(r'^(\d{4})_([A-Z]{2,3})_([A-Z]{2,3})_([A-Z]{2})\.wav$')
    best: dict[str, tuple[str, int]] = {}

    for f in audio_dir.iterdir():
        m = pattern.match(f.name)
        if not m:
            continue
        _, _, emotion_code, intensity_code = m.groups()
        label = EMOTION_STYLE_MAP.get(emotion_code)
        if label is None:
            continue
        rank = INTENSITY_RANK.get(intensity_code, 99)
        if label not in best or rank < best[label][1]:
            best[label] = (str(f), rank)

    refs = {label: path for label, (path, _) in best.items()}
    if refs:
        log.info(f"StyleTTS reference WAVs selected: {sorted(refs.keys())}")
    else:
        log.warning("No reference WAVs found — StyleTTS will use default style")
    return refs
```

#### Technical Explanation
1. **`EMOTION_STYLE_MAP`**: Translates CREMA-D 3-letter uppercase emotion codes (`ANG`, `DIS`, `FEA`, `HAP`, `NEU`, `SAD`) into natural language style keys.
2. **Function `build_emotion_refs`**:
   - **Purpose**: StyleTTS 2 requires a reference audio file from which it extracts a style vector via a style encoder to steer pitch, energy, and prosody.
   - **Regex Parsing**: Inspects filenames conforming to CREMA-D naming conventions (`<actor_id>_<sentence>_<emotion>_<intensity>.wav`).
   - **Intensity Priority**: Ranks clips by emotional intensity (`HI` > `XX` > `MD` > `LO`) to guarantee that synthesized voices express pronounced emotional characteristics.
   - **Returns**: Dictionary mapping emotion strings to local reference `.wav` file paths.

---

### Snippet 3: Low-Level Media Subprocess Helpers (`run_ffmpeg`, `get_duration`, `mux_video_audio`, `convert_flv_to_mp4`)

```python
# ── ffmpeg helpers ─────────────────────────────────────────────────────────────

def run_ffmpeg(cmd: list[str], timeout: int = 120) -> bool:
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error'] + cmd,
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            log.debug(f"ffmpeg error: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg timeout")
        return False
    except FileNotFoundError:
        log.error("ffmpeg not found — install it and add to PATH")
        sys.exit(1)


def get_duration(path: str) -> float | None:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def mux_video_audio(video_path: str, audio_path: str, out_path: str,
                    target_fps: int = 30) -> bool:
    """Replace video audio track. Output duration matches video."""
    video_dur = get_duration(video_path)
    if video_dur is None:
        return False
    cmd = [
        '-i', video_path, '-i', audio_path,
        '-map', '0:v:0', '-map', '1:a:0',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k',
        '-t', str(video_dur), '-af', 'apad', '-shortest',
        '-r', str(target_fps), out_path,
    ]
    return run_ffmpeg(cmd)


def convert_flv_to_mp4(flv_path: str, mp4_path: str) -> bool:
    return run_ffmpeg([
        '-i', flv_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'aac', mp4_path,
    ])
```

#### Technical Explanation
1. **`run_ffmpeg(cmd, timeout)`**: Low-level execution wrapper suppressing banner clutter, trapping stderr, enforcing timeouts, and exiting gracefully if FFmpeg is uninstalled.
2. **`get_duration(path)`**: Uses `ffprobe` to read the exact floating-point duration of the video stream.
3. **`mux_video_audio(video_path, audio_path, out_path)`**:
   - Replaces the audio track of `video_path` with `audio_path`.
   - Uses `-af apad -shortest -t <video_dur>` to ensure audio is padded or trimmed so the synthetic output file duration matches the input video duration.
   - Enforces a standard 30 FPS framerate with constant rate factor (`-crf 23`).
4. **`convert_flv_to_mp4(flv_path, mp4_path)`**: Converts legacy CREMA-D Flash Video (`.flv`) containers into H.264/AAC `.mp4` containers required by neural visual backbones.

---

### Snippet 4: Speaker Identity Verification (`XVectorVerifier`)

```python
# ── Speaker verification (x-vector cosine similarity) ─────────────────────────

class XVectorVerifier:
    """
    Filters RVC outputs by speaker embedding cosine similarity.
    Keeps a clip only if similarity >= threshold (default 0.75).
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from speechbrain.pretrained import SpeakerRecognition
            self._model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                savedir="./pretrained_models/xvect",
            )
            log.info("X-vector model loaded.")
        except ImportError:
            log.warning("speechbrain not installed — x-vector filtering disabled.")
            self._model = 'unavailable'

    def similarity(self, wav1: str, wav2: str) -> float | None:
        self._load()
        if self._model == 'unavailable':
            return None
        try:
            score, _ = self._model.verify_files(wav1, wav2)
            return float(score)
        except Exception as e:
            log.debug(f"X-vector error: {e}")
            return None

    def passes(self, wav1: str, wav2: str) -> bool:
        sim = self.similarity(wav1, wav2)
        if sim is None:
            return True
        return sim >= self.threshold
```

#### Technical Explanation
1. **Purpose**: Voice conversion models (RVC) can occasionally produce corrupted or robotic audio if pitch extraction fails. The `XVectorVerifier` acts as an automated acoustic quality filter.
2. **Acoustic Model**: Uses SpeechBrain's `spkrec-xvect-voxceleb` model, extracting deep speaker embeddings (x-vectors) from both the original actor speech (`wav1`) and the converted speech (`wav2`).
3. **Metric**: Computes cosine similarity between embeddings. If $similarity < 0.75$, the clip is rejected to preserve dataset realism.

---

### Snippet 5: Track 1 Synthesis Engine (`StyleTTSRVCGenerator`)

```python
# ── Method B: StyleTTS 2 + RVC (SOTA synthesis) ───────────────────────────────

class StyleTTSRVCGenerator:
    """
    Synthesise target-emotion speech with StyleTTS 2, then transfer the
    actor's vocal identity with Applio RVC. Mux into the original face video.
    """

    def __init__(self, out_dir: Path, applio_dir: Path,
                 verifier: XVectorVerifier,
                 ref_wavs: dict[str, str] | None = None,
                 cremad_dir: Path | None = None):
        self.out_dir    = out_dir
        self.applio_dir = applio_dir
        self.verifier   = verifier
        self.ref_wavs   = ref_wavs or {}
        self.cremad_dir = cremad_dir
        self.video_dir  = out_dir / 'videos'
        self.wav_dir    = out_dir / 'wav_tmp'
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.wav_dir.mkdir(parents=True, exist_ok=True)
        self._tts = None

    def _load_tts(self):
        if self._tts is not None:
            return
        try:
            import nltk
            nltk.download('punkt_tab', quiet=True)
            from styletts2 import tts
            self._tts = tts.StyleTTS2()
            log.info("StyleTTS 2 loaded.")
        except ImportError:
            raise RuntimeError("styletts2 not installed. Run: pip install styletts2")

    def _get_applio_model_paths(self, actor_id: int) -> tuple[str, str]:
        logs_dir  = self.applio_dir / 'logs' / f"actor_{actor_id}"
        pth_files = list(logs_dir.glob('*.pth')) if logs_dir.exists() else []
        if not pth_files:
            raise FileNotFoundError(
                f"No RVC model for actor {actor_id} in {logs_dir}."
            )
        idx_files = list(logs_dir.glob('added_*.index'))
        return str(pth_files[0]), (str(idx_files[0]) if idx_files else '')

    def _synthesise(self, text: str, target_emotion: str, out_wav: str) -> bool:
        self._load_tts()
        ref_wav = self.ref_wavs.get(target_emotion)
        try:
            wav = self._tts.inference(
                text,
                target_voice_path=ref_wav,
                output_sample_rate=24000,
                alpha=0.3,
                beta=0.7,
                diffusion_steps=10,
                embedding_scale=1.0,
            )
            import soundfile as sf
            sf.write(out_wav, wav, 24000)
            return True
        except Exception as e:
            log.error(f"StyleTTS synthesis failed: {e}", exc_info=True)
            return False

    def _convert_voice(self, src_wav: str, actor_id: int, out_wav: str) -> bool:
        pth_path, idx_path = self._get_applio_model_paths(actor_id)
        result = subprocess.run(
            [
                sys.executable, 'core.py', 'infer',
                '--pitch',           '0',
                '--index_rate',      '0.3' if idx_path else '0',
                '--volume_envelope', '0.25',
                '--protect',         '0.33',
                '--f0_method',       'rmvpe',
                '--input_path',      os.path.abspath(src_wav),
                '--output_path',     os.path.abspath(out_wav),
                '--pth_path',        os.path.abspath(pth_path),
                '--index_path',      os.path.abspath(idx_path) if idx_path else '',
                '--split_audio',     'False',
                '--clean_audio',     'True',
                '--clean_strength',  '0.7',
                '--export_format',   'WAV',
            ],
            cwd=str(self.applio_dir),
            capture_output=True, text=True, timeout=300,
        )
        return result.returncode == 0 and os.path.exists(out_wav)
```

#### Technical Explanation
1. **TTS Speech Synthesis (`_synthesise`)**:
   - Invokes `StyleTTS2.inference()` with `target_voice_path` set to the reference emotion clip.
   - Synthesizes speech phonemes conditioned on target emotion style vectors using diffusion steps.
2. **Voice Identity Transfer (`_convert_voice`)**:
   - Discovers pre-trained actor voice models (`.pth` weights and `.index` feature indices) in `tools/Applio/logs/actor_{actor_id}`.
   - Runs `tools/Applio/core.py infer` using RMVPE pitch extraction (`--f0_method rmvpe`) and feature index retrieval (`--index_rate 0.3`).
   - Transfers vocal timbre to match the actor's acoustic profile while retaining the target emotional intonation.

---

### Snippet 6: Track 2 Lip-Sync Reanimation Engine (`run_wav2lip` & `Track2Generator`)

```python
# ── Wav2Lip inference ──────────────────────────────────────────────────────────

def run_wav2lip(wav2lip_dir: Path, face_mp4: str, audio_wav: str,
                out_video: str, model_path: Path) -> bool:
    """
    Run Wav2Lip on an MP4 face video + WAV audio.
    Retries with resize_factor 2 then 4 on face-detection failure.
    face_mp4 must already be an MP4 (convert FLV before calling).
    """
    inference_script = wav2lip_dir / "inference.py"
    if not inference_script.exists():
        raise FileNotFoundError(f"Wav2Lip inference.py not found at {inference_script}.")

    # Retry loop with progressive spatial downsampling
    for resize in (1, 2, 4, 8):
        cmd = [
            sys.executable, str(inference_script.resolve()),
            "--checkpoint_path", str(model_path.resolve()),
            "--face",    os.path.abspath(face_mp4),
            "--audio",   os.path.abspath(audio_wav),
            "--outfile", os.path.abspath(out_video),
            "--nosmooth",
        ]
        if resize > 1:
            cmd += ["--resize_factor", str(resize)]

        result = subprocess.run(
            cmd,
            cwd=str(wav2lip_dir.resolve()),
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0 and os.path.exists(out_video):
            return True
        if "Face not detected" in result.stderr:
            log.warning(f"Face not detected at resize_factor={resize}, retrying...")
            continue
        log.debug(f"Wav2Lip failed (resize={resize}):\n{result.stderr[-1000:]}")
        return False

    log.error("Wav2Lip face detection failed at all resize factors (1, 2, 4, 8).")
    return False


class Track2Generator:
    """Per-clip pipeline: StyleTTS2 -> RVC -> Wav2Lip."""
    def __init__(self, out_dir: Path, applio_dir: Path, wav2lip_dir: Path,
                 wav2lip_model: Path, verifier: XVectorVerifier,
                 ref_wavs: dict[str, str], cremad_dir: Path):
        self.out_dir = out_dir
        self.applio_dir = applio_dir
        self.wav2lip_dir = wav2lip_dir
        self.wav2lip_model = wav2lip_model
        self.verifier = verifier
        self.ref_wavs = ref_wavs
        self.cremad_dir = cremad_dir
        self.video_dir = out_dir / 'videos'
        self.wav_dir = out_dir / 'wav_tmp'
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.wav_dir.mkdir(parents=True, exist_ok=True)
```

#### Technical Explanation
1. **Purpose**: Track 2 synthesizes deepfakes where **both the audio and the lip movements are manipulated**.
2. **Progressive Fallback Multi-Scale Detection (`resize in (1, 2, 4, 8)`)**:
   - The S3FD face detector inside Wav2Lip can miss faces in low-contrast or rapid-motion video.
   - If `"Face not detected"` appears in stderr, the function automatically downsamples the frame ($2\times, 4\times, 8\times$) to re-attempt detection before failing.
3. **Synthesis Flow**:
   - Generates emotion-conditioned speech via StyleTTS 2.
   - Clones actor vocal timbre via RVC v2.
   - Reanimates the lower facial half of the original face video via `run_wav2lip`, synchronizing mouth phonemes to the synthesized audio track.

---

### Snippet 7: Track 3 Talking-Head Neural Engine (`run_sadtalker` & `Track3Generator`)

```python
# ── SadTalker inference ────────────────────────────────────────────────────────

def run_sadtalker(sadtalker_dir: Path, portrait_jpg: str, audio_wav: str,
                  out_video: str, size: int = 256, still: bool = True) -> bool:
    """
    Run SadTalker inference.py to generate talking-head video.
    Tries --preprocess crop first; falls back to full if face detection fails.
    SadTalker writes to a timestamped subdir — we find and move the output.
    Must run with cwd=sadtalker_dir (SadTalker uses relative src/config paths).
    """
    inference_script = sadtalker_dir / "inference.py"
    if not inference_script.exists():
        raise FileNotFoundError(f"SadTalker inference.py not found at {inference_script}.")

    import shutil
    from glob import glob

    for preprocess in ("crop", "full"):
        with tempfile.TemporaryDirectory() as tmp_result:
            cmd = [
                sys.executable, str(inference_script.resolve()),
                "--source_image",  os.path.abspath(portrait_jpg),
                "--driven_audio",  os.path.abspath(audio_wav),
                "--checkpoint_dir", str((sadtalker_dir / "checkpoints").resolve()),
                "--result_dir",    tmp_result,
                "--size",          str(size),
                "--preprocess",    preprocess,
            ]
            if still:
                cmd.append("--still")

            result = subprocess.run(
                cmd,
                cwd=str(sadtalker_dir.resolve()),
                capture_output=True, text=True, timeout=600,
            )

            if result.returncode != 0:
                continue

            mp4s = glob(os.path.join(tmp_result, "**", "*.mp4"), recursive=True)
            if not mp4s:
                continue

            shutil.move(mp4s[0], out_video)
            if os.path.exists(out_video):
                return True

    return False
```

#### Technical Explanation
1. **Purpose**: Track 3 produces full neural talking-head re-creations. Rather than modifying an existing video stream, it takes a **single portrait image** from the actor and animates head pose, gaze, and lip synchronization via 3D Morphable Models (3DMM).
2. **Execution Logic**:
   - Extracts the middle frame of the source video as `portrait.jpg`.
   - Executes SadTalker with `--driven_audio` mapped to the RVC converted speech.
   - Tries `--preprocess crop` (tighter facial bounding box); falls back to `--preprocess full` if initial alignment fails.
   - Moves the generated MP4 from the temporary output directory to `data/synthetic/track3_fakes/videos/`.

---

# 3. Module 2: Multimodal Preprocessing Architecture (Feature Ingestion Pipeline)

## 3.1 Header Comment Block: Preprocessing Architecture

```text
========================================================================================
PROGRAM TITLE:
    DeepSentinel Multimodal Feature Extraction & Caching Orchestrator
    (Acoustic-Linguistic Z_at and Visual Keyframe Z_v Ingestion Pipeline)

PROGRAMMERS:
    Thesis Group 10 (DeepSentinel Team)
    - Kina (Preprocessing Lead)
    - Gel (Data Manager & QA Lead)
    - JC (Training & Model Architecture Lead)
    - Matan (Evaluation & Significance Lead)

WHERE THE PROGRAM FITS IN THE GENERAL SYSTEM DESIGNS:
    The Preprocessing Module is the foundational ingestion backbone of DeepSentinel.
    It takes raw audio-video clips from genuine corpora (CMU-MOSEI, MELD, CREMA-D) and
    synthetic benchmark sets (Track 1-3 fakes, FakeAVCeleb) and transforms each clip
    into standardized PyTorch tensor representations:
      - Z_at in R^(1536): Concatenation of Wav2Vec2 acoustic features (768) and BERT
        linguistic features (768) derived from Whisper ASR transcription.
      - Z_v in R^(8 x 768): Sequence of 8 salient keyframe embeddings generated by
        Vision Transformer (ViT-B/16) guided by Farneback optical flow and RetinaFace.
    All downstream training and inference routines load directly from these cached tensors.

DATE WRITTEN AND REVISED:
    - Date Written: February 2026
    - Last Revised: October 2026 (Enforced AU-off feature consistency, safe weights_only
      tensor deserialization, and atomic disk caching)

PURPOSE:
    To provide deterministic, fault-tolerant extraction of visual and audio features:
    1. Demuxing audio to 16kHz mono PCM WAV via FFmpeg.
    2. Transcribing speech using OpenAI Whisper ASR.
    3. Generating acoustic embeddings (Wav2Vec2) and linguistic embeddings (BERT).
    4. Motion-gating video frames (Farneback Optical Flow), detecting aligned faces
       (RetinaFace), ranking frames via Laplacian sharpness, and computing ViT CLS tokens.
    5. Caching intermediate and final outputs to disk to allow incremental resumption.

DATA STRUCTURES, ALGORITHMS, AND CONTROL:
    1. Data Structures:
       - @dataclass ClipFeatures: Encapsulates clip_id (str), z_at (torch.Tensor),
         z_v (torch.Tensor), and transcript (str).
       - pathlib.Path: Maintains structured directories (`audio/`, `transcripts/`,
         `features/z_at/`, `features/z_v/`).
    2. Algorithms:
       - Farneback Dense Optical Flow: Filters out static background frames.
       - RetinaFace Alignment: Rotational correction & safe square crop with 20% margin.
       - Laplacian Variance Sharpness: Ranks facial focus quality across temporal sequence.
       - CLS Token Pooling: Mean-pooling Wav2Vec2 frames and extracting BERT/ViT CLS tokens.
    3. Control Flow:
       - Cache-first verification via `is_cached(clip_id)` skips redundant processing.
       - Atomic per-clip saving allows pipeline interruption and resumption without data corruption.
       - Graceful logging and `None` returns prevent a single corrupt clip from halting batch runs.
========================================================================================
```

## 3.2 Considerations in Writing the Code (Preprocessing Architecture)

- **Indentation**: Standard 4 spaces conforming to PEP 8.
- **Inline Comments**: Step-by-step numbering (`# Step 1: extract audio WAV`, `# Step 2: ASR transcription`, etc.) mapping directly to thesis architecture diagrams.
- **Limiting the Use of Global**: Only the module logger `log` is module-scoped. Model singletons are lazily instantiated inside helper functions with caching.
- **Structured Programming**: Object-oriented orchestrator `PreprocessingPipeline` calling pure modular helper functions from `audio.py`, `visual.py`, and `filters.py`.
- **Naming Conventions**:
  - PascalCase for data classes and types: `ClipFeatures`, `PreprocessingPipeline`
  - snake_case for member methods and properties: `is_cached`, `_wav_path`, `load_features`
  - Strict type hints throughout (`clip_id: str`, `force: bool = False`).
- **Error Return Conventions**: Returns `Optional[ClipFeatures]` (returning `None` on unrecoverable stream error) while writing detailed diagnostic logs to disk (`log.error`).
- **Line Length**: Maintained within 100 characters.
- **Spacing**: Uniform blank lines separating public and private helper methods.
- **Function Length**: Focused, concise methods with clear separation between cache checks and feature extraction.
- **Error Messages**: Captures and reports specific failure modes (e.g., missing video file vs model extraction failure).

---

### Snippet 1: Module Header, Imports, and Type Setup

```python
"""
pipeline.py — Preprocessing orchestrator: raw clip → (Z_at, Z_v) + metadata.

Processes a single video clip and caches:
    data/preprocessed/
        audio/{clip_id}.wav          16kHz mono WAV
        transcripts/{clip_id}.txt    Whisper ASR output
        features/z_at/{clip_id}.pt   (1536,) tensor
        features/z_v/{clip_id}.pt    (8, 768) keyframe sequence tensor (or legacy 768,)

Re-running on an already-processed clip is a no-op (cached files respected).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

from .audio import extract_audio_to_wav, transcribe, transcribe_with_meta, get_z_at
from .visual import get_z_v

log = logging.getLogger(__name__)
```

#### Technical Explanation
1. **Module Scope**: Orchestrates the multi-stage ingestion converting raw `.mp4` video clips into cached multimodal tensors $(Z_{at}, Z_v)$.
2. **Imports**:
   - `from __future__ import annotations`: Postpones type evaluation, allowing clean union type syntax (`str | Path`).
   - `dataclasses.dataclass`: Lightweight decorator for generating data containers.
   - `torch`: PyTorch tensor manipulation and serialization library.
   - Sibling modules:
     - `.audio`: Demuxing, Whisper transcription, Wav2Vec2 + BERT embedding generation.
     - `.visual`: Farneback optical flow motion gating, RetinaFace detection, and ViT-B/16 token extraction.

---

### Snippet 2: Data Structure `ClipFeatures`

```python
@dataclass
class ClipFeatures:
    clip_id:    str
    z_at:       torch.Tensor    # (1536,)
    z_v:        torch.Tensor    # (8, 768) or (768,)
    transcript: str
```

#### Technical Explanation
1. **Purpose**: Data contract encapsulating all extracted multimodal representations for a video clip.
2. **Fields**:
   - `clip_id (str)`: Unique identifier string matching the manifest row.
   - `z_at (torch.Tensor)`: 1536-dimensional float32 tensor: $[z_{acoustic} \parallel z_{linguistic}]$.
   - `z_v (torch.Tensor)`: Float32 tensor of shape `(8, 768)` containing the 8 salient keyframe CLS tokens.
   - `transcript (str)`: Whisper-generated transcription text string.

---

### Snippet 3: Class `PreprocessingPipeline` Initialization & Directory Creation

```python
class PreprocessingPipeline:
    """
    Orchestrates audio + visual preprocessing for one clip at a time.
    All outputs are cached — safe to interrupt and resume.
    """

    def __init__(
        self,
        cache_dir:            str | Path = "data/preprocessed",
        wav2vec_model:        str   = "facebook/wav2vec2-base",
        bert_model:           str   = "bert-base-uncased",
        whisper_model:        str   = "openai/whisper-base",
        vit_model:            str   = "google/vit-base-patch16-224",
        face_detector:        str   = "retinaface",
        n_keyframes:          int   = 8,
        frame_size:           int   = 224,
        max_audio_sec:        int   = 5,
        target_fps:           float = 25.0,
        motion_threshold:     float = 0.3,
        confidence_threshold: float = 0.7,
        device:               str   = "cuda",
    ):
        self.cache_dir            = Path(cache_dir)
        self.wav2vec_model        = wav2vec_model
        self.bert_model           = bert_model
        self.whisper_model        = whisper_model
        self.vit_model            = vit_model
        self.face_detector        = face_detector
        self.n_keyframes          = n_keyframes
        self.frame_size           = frame_size
        self.max_audio_sec        = max_audio_sec
        self.target_fps           = target_fps
        self.motion_threshold     = motion_threshold
        self.confidence_threshold = confidence_threshold
        self.device               = device

        (self.cache_dir / "audio").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "transcripts").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "features" / "z_at").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "features" / "z_v").mkdir(parents=True, exist_ok=True)
```

#### Technical Explanation
1. **Hyperparameter Configuration**:
   - `wav2vec_model`: Base acoustic model (`facebook/wav2vec2-base`).
   - `bert_model`: Base linguistic model (`bert-base-uncased`).
   - `whisper_model`: Speech-to-text model (`openai/whisper-base`).
   - `vit_model`: Vision Transformer backbone (`google/vit-base-patch16-224`).
   - `face_detector`: Uses InsightFace's RetinaFace (`buffalo_s` ONNX model).
   - `n_keyframes`: Fixed at 8 keyframes per clip.
   - `max_audio_sec`: Clamped to 5.0 seconds (80,000 samples at 16kHz).
   - `motion_threshold`: Optical flow threshold (0.3) to drop static frames.
   - `confidence_threshold`: Minimum RetinaFace score (0.7).
2. **Directory Tree Creation**: Pre-creates cache folders for audio, transcripts, and feature tensors.

---

### Snippet 4: Cache Path Generators & Atomic Cache Verification

```python
    # ── Cache paths ────────────────────────────────────────────────────────────

    def _wav_path(self, clip_id: str) -> Path:
        return self.cache_dir / "audio" / f"{clip_id}.wav"

    def _txt_path(self, clip_id: str) -> Path:
        return self.cache_dir / "transcripts" / f"{clip_id}.txt"

    def _z_at_path(self, clip_id: str) -> Path:
        return self.cache_dir / "features" / "z_at" / f"{clip_id}.pt"

    def _z_v_path(self, clip_id: str) -> Path:
        return self.cache_dir / "features" / "z_v" / f"{clip_id}.pt"

    def is_cached(self, clip_id: str) -> bool:
        return self._z_at_path(clip_id).exists() and self._z_v_path(clip_id).exists()
```

#### Technical Explanation
1. **Path Mapping**: Determines exact destination filenames for a clip's artifacts based on its `clip_id`.
2. **Atomic Invariance (`is_cached`)**: Verifies that both $Z_{at}$ and $Z_v$ tensors are present. If a power outage or crash occurs midway through a clip, `is_cached` returns `False`, ensuring the incomplete feature pair is recomputed from scratch.

---

### Snippet 5: Audio Demuxing, Whisper ASR Transcription & $Z_{at}$ Extraction

```python
# Extracted from src/preprocessing/audio.py and pipeline.py

def extract_audio_to_wav(video_path: str | Path, out_wav: str | Path,
                         sample_rate: int = 16000) -> bool:
    """Extract mono 16kHz WAV from video. Returns True on success."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path), "-vn",
            "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1",
            str(out_wav),
        ],
        capture_output=True, timeout=120,
    )
    return result.returncode == 0


def get_z_at(wav_path: str | Path, transcript: Optional[str] = None,
             wav2vec_model: str = "facebook/wav2vec2-base",
             bert_model: str = "bert-base-uncased",
             device: str = "cpu", max_seconds: int = 5) -> torch.Tensor:
    """
    Compute Z_at = concat(acoustic_emb, linguistic_emb) → (1536,).
    """
    # 1. Acoustic: Wav2Vec2 mean-pooled representation
    acoustic = get_acoustic_embedding(wav_path, wav2vec_model, device, max_seconds)  # (768,)
    # 2. Linguistic: BERT [CLS] token embedding
    linguistic = get_linguistic_embedding(transcript, bert_model, device)             # (768,)
    # 3. Concatenation
    return torch.cat([acoustic, linguistic], dim=0)                                   # (1536,)
```

#### Technical Explanation
1. **`extract_audio_to_wav`**: Runs FFmpeg to demux the video into a 16,000 Hz, 16-bit mono PCM WAV file.
2. **`get_z_at` Composite Extraction**:
   - Loads the 16kHz waveform and truncates it to `max_seconds=5` (80,000 samples).
   - Feeds waveform to Wav2Vec2-base, extracting the last hidden states and mean-pooling across the temporal dimension: $z_a \in \mathbb{R}^{768}$.
   - Tokenizes the Whisper transcript and feeds it to BERT-base, extracting the `[CLS]` token embedding: $z_t \in \mathbb{R}^{768}$.
   - Concatenates the acoustic and linguistic vectors along dimension 0: $Z_{at} = [z_a \parallel z_t] \in \mathbb{R}^{1536}$.

---

### Snippet 6: Visual Motion Gating, Face Alignment & $Z_v$ ViT Keyframe Extraction

```python
# Extracted from src/preprocessing/visual.py

def optical_flow_gate(frames: List[np.ndarray], motion_threshold: float = 0.3) -> List[np.ndarray]:
    """Keep frames where mean Farneback optical flow magnitude >= motion_threshold."""
    if len(frames) < 2:
        return frames
    gated = [frames[0]]
    prev_small = cv2.resize(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), (160, 120))
    for frame in frames[1:]:
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120))
        flow = cv2.calcOpticalFlowFarneback(
            prev_small, small, None, pyr_scale=0.5, levels=2, winsize=11,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
        if magnitude >= motion_threshold:
            gated.append(frame)
        prev_small = small
    return gated if len(gated) > 1 else frames


def get_z_v(video_path: str | Path, vit_model_name: str = "google/vit-base-patch16-224",
            detector: str = "retinaface", n_keyframes: int = 8, frame_size: int = 224,
            target_fps: float = 25.0, motion_threshold: float = 0.3,
            confidence_threshold: float = 0.7, device: str = "cpu") -> torch.Tensor:
    """
    Full visual pipeline: extract frames -> optical flow gate -> detect faces ->
    rank by Laplacian sharpness -> select top-K keyframes -> ViT embedding.
    Returns: (8, 768) float32 tensor.
    """
    frames = extract_frames(video_path, target_fps, max_seconds=5.0)
    gated_frames = optical_flow_gate(frames, motion_threshold)
    face_results = detect_and_align_faces(gated_frames, detector, confidence_threshold)
    crops = [r[0] for r in face_results]
    scores = [r[1] for r in face_results]
    keyframes = select_keyframes(crops, scores, k=n_keyframes)
    pil_imgs = frames_to_pil(keyframes, size=frame_size)

    model, processor = _load_vit(vit_model_name, device=device)
    inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    cls_tokens = out.last_hidden_state[:, 0, :]   # (K, 768)
    return cls_tokens.cpu()
```

#### Technical Explanation
1. **Optical Flow Motion Gating (`optical_flow_gate`)**:
   - Downsamples consecutive frames to $160 \times 120$ for fast CPU calculation.
   - Computes Farneback dense optical flow vectors. Frames with mean motion magnitude $< 0.3$ (static background) are filtered out.
2. **Face Detection & Alignment (`detect_and_align_faces`)**:
   - Uses InsightFace RetinaFace (`det_score >= 0.7`).
   - Applies `_safe_square_crop` with 1:1 aspect ratio and 20% context margin, adding reflection padding when faces approach the border.
3. **Sharpness Ranking & Keyframe Selection (`select_keyframes`)**:
   - Computes Laplacian variance on each crop: $\text{Var}(\nabla^2 I)$. Higher variance indicates sharper focus.
   - Selects the top $K=8$ highest-quality facial frames.
4. **ViT Sequence Embedding**:
   - Preprocesses images into $(8, 3, 224, 224)$ tensors.
   - Passes batch through Vision Transformer (`google/vit-base-patch16-224`).
   - Extracts the `[CLS]` token from each of the 8 keyframes, yielding a tensor of shape `(8, 768)`.

---

### Snippet 7: Master Processing Orchestrator (`process` method)

```python
    def process(
        self,
        clip_id:    str,
        video_path: str | Path,
        force:      bool = False,
    ) -> Optional[ClipFeatures]:
        """
        Process one clip. Returns ClipFeatures or None on failure.
        Skips already-cached clips unless force=True.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            log.error(f"Video not found: {video_path}")
            return None

        if not force and self.is_cached(clip_id):
            return self._load_cached(clip_id)

        # Step 1: extract audio WAV
        wav = self._wav_path(clip_id)
        if not wav.exists() or force:
            ok = extract_audio_to_wav(video_path, wav)
            if not ok or not wav.exists():
                log.warning(f"Audio extraction failed: {clip_id}")
                if video_path.suffix.lower() == ".wav":
                    import shutil
                    shutil.copy2(video_path, wav)
                else:
                    return None

        # Step 2: ASR transcription
        txt_file = self._txt_path(clip_id)
        if not txt_file.exists() or force:
            meta_res = transcribe_with_meta(wav, self.whisper_model, device=self.device)
            transcript = meta_res.get("text", "")
            txt_file.write_text(transcript, encoding="utf-8")
        else:
            transcript = txt_file.read_text(encoding="utf-8").strip()

        # Step 3: Z_at — acoustic + linguistic
        z_at_path = self._z_at_path(clip_id)
        if not z_at_path.exists() or force:
            try:
                z_at = get_z_at(
                    wav, transcript,
                    self.wav2vec_model, self.bert_model,
                    self.device, self.max_audio_sec,
                )
                torch.save(z_at, z_at_path)
            except Exception as e:
                log.error(f"Z_at extraction failed for {clip_id}: {e}")
                return None
        else:
            z_at = torch.load(z_at_path, weights_only=True)

        # Step 4: Z_v — visual keyframe ViT
        z_v_path = self._z_v_path(clip_id)
        if not z_v_path.exists() or force:
            try:
                z_v = get_z_v(
                    video_path,
                    vit_model_name=self.vit_model,
                    detector=self.face_detector,
                    n_keyframes=self.n_keyframes,
                    frame_size=self.frame_size,
                    target_fps=self.target_fps,
                    motion_threshold=self.motion_threshold,
                    confidence_threshold=self.confidence_threshold,
                    device=self.device,
                )
                torch.save(z_v, z_v_path)
            except Exception as e:
                log.error(f"Z_v extraction failed for {clip_id}: {e}")
                return None
        else:
            z_v = torch.load(z_v_path, weights_only=True)

        return ClipFeatures(clip_id=clip_id, z_at=z_at, z_v=z_v, transcript=transcript)
```

#### Technical Explanation
1. **Cache Verification**: Checks `is_cached(clip_id)`. If true and `force=False`, loads the cached tensors from disk immediately.
2. **Step 1 (Audio Extraction)**: Demuxes video to 16kHz mono WAV at `audio/{clip_id}.wav`.
3. **Step 2 (Whisper ASR)**: Transcribes speech and caches the text at `transcripts/{clip_id}.txt`.
4. **Step 3 ($Z_{at}$ Extraction)**: Concatenates Wav2Vec2 and BERT embeddings; saves tensor to `features/z_at/{clip_id}.pt`.
5. **Step 4 ($Z_v$ Extraction)**: Applies optical flow motion gating, RetinaFace detection, sharpness ranking, and ViT token extraction; saves tensor to `features/z_v/{clip_id}.pt`.
6. **Return**: Returns a typed `ClipFeatures` container. If any step fails, logs the error and returns `None` without crashing the parent batch process.

---

### Snippet 8: High-Throughput DataLoader Accessors (`_load_cached` & `load_features`)

```python
    def _load_cached(self, clip_id: str) -> ClipFeatures:
        z_at = torch.load(self._z_at_path(clip_id), weights_only=True)
        z_v  = torch.load(self._z_v_path(clip_id),  weights_only=True)
        txt  = self._txt_path(clip_id)
        transcript = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
        return ClipFeatures(clip_id=clip_id, z_at=z_at, z_v=z_v, transcript=transcript)

    def load_features(self, clip_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast load of cached (Z_at, Z_v) for training."""
        return (
            torch.load(self._z_at_path(clip_id), weights_only=True),
            torch.load(self._z_v_path(clip_id),  weights_only=True),
        )
```

#### Technical Explanation
1. **`_load_cached(self, clip_id: str) -> ClipFeatures`**: Deserializes cached tensors and reads the transcript string, using `weights_only=True` to satisfy PyTorch security constraints.
2. **`load_features(self, clip_id: str) -> Tuple[torch.Tensor, torch.Tensor]`**: Optimized for PyTorch `Dataset.__getitem__` during training. Bypasses text file I/O and returns only `(z_at, z_v)` tensors directly into memory.

---

# 4. Instructional Walkthrough: Running and Testing Both Modules

### 4.1 Running the Generation Module (Tracks 1–3)

Execute generation from the command line using the dataset manifests:

```powershell
# 1. Generate Track 1 Deepfakes (StyleTTS 2 + RVC v2 audio-swap)
python src/track1/track1_generate.py `
    --pairs_csv  data/processed/track1_manifests/swap_pairs.csv `
    --out_dir    data/synthetic/track1_fakes `
    --applio_dir tools/Applio `
    --cremad_dir data/raw/CREMA-D `
    --resume

# 2. Generate Track 2 Deepfakes (StyleTTS 2 + RVC v2 + Wav2Lip lip-sync reanimation)
python src/track2/track2_generate.py `
    --pairs_csv   data/processed/track1_manifests/track2_pairs.csv `
    --out_dir     data/synthetic/track2_fakes `
    --applio_dir  tools/Applio `
    --wav2lip_dir tools/Wav2Lip `
    --cremad_dir  data/raw/CREMA-D `
    --resume

# 3. Generate Track 3 Deepfakes (StyleTTS 2 + RVC v2 + SadTalker 3DMM talking-head)
python src/track3/track3_generate.py `
    --pairs_csv     data/processed/track1_manifests/track3_pairs.csv `
    --out_dir       data/synthetic/track3_fakes `
    --applio_dir    tools/Applio `
    --sadtalker_dir tools/SadTalker `
    --cremad_dir    data/raw/CREMA-D `
    --resume
```

### 4.2 Running the Preprocessing Module

Execute feature extraction on a single clip or across full dataset manifests:

```python
# Python test snippet: Process a single video into (Z_at, Z_v)
from pathlib import Path
from src.preprocessing.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline(cache_dir="data/preprocessed", device="cuda")

features = pipeline.process(
    clip_id="sample_001",
    video_path="docs/genuine happy.mp4",
    force=True,
)

if features:
    print(f"Extracted Clip ID: {features.clip_id}")
    print(f"Z_at shape:        {features.z_at.shape}  (Expected: torch.Size([1536]))")
    print(f"Z_v shape:         {features.z_v.shape}   (Expected: torch.Size([8, 768]))")
    print(f"Transcript:        '{features.transcript}'")
```

```powershell
# Batch feature preprocessing across all training and evaluation manifests
python scripts/preprocess_all.py --device cuda
```
All tensors will be written to `data/preprocessed/features/z_at/` and `data/preprocessed/features/z_v/`.
