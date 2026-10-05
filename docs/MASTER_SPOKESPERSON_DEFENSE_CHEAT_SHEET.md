# DeepSentinel: Master Spokesperson & Defense Leader Cheat Sheet
## *The Plain-English, Analogy-Driven Code & Theory Guide*

> **Official Thesis Title:** *A Multimodal Deepfake Detection Framework Leveraging Bilinear Pooling and Emotion Mismatch*  
> **Institution:** Polytechnic University of the Philippines — Department of Computer Science (BSCS 2026)  
> **Authors:** Cabral, Shikina Y. | Caparas, John Christian B. | Exconde, Matan John B. | Rivera, Geuel John D.  
> **Purpose of this Cheat Sheet:** Designed for the leader and team to master the entire system by heart. Translates complex neural networks and mathematical equations into intuitive, real-world analogies that anyone—from a non-technical panelist or grandparent to a seasoned AI professor—can instantly understand, while mapping every claim to its exact line of source code.

---

# Table of Contents
1. [The 60-Second "Grandmother Pitch" & Master Analogies](#1-the-60-second-grandmother-pitch--master-analogies)
2. [The 7-Stage Pipeline: Analogy + Code + Math](#2-the-7-stage-pipeline-analogy--code--math)
   - [Stage 1: Ingestion & The Silent Gatekeeper](#stage-1-ingestion--the-silent-gatekeeper)
   - [Stage 2: The Tri-Modal Senses (Ears, Brain, Eyes)](#stage-2-the-tri-modal-senses-ears-brain-eyes)
   - [Stage 3: Compact Bilinear Pooling (The Kitchen Blender & Compressor)](#stage-3-compact-bilinear-pooling-the-kitchen-blender--compressor)
   - [Stage 4: Emotion & Sarcasm Heads (The Lie Detectors)](#stage-4-emotion--sarcasm-heads-the-lie-detectors)
   - [Stage 5: The 299D Hybrid Bottleneck (The Courtroom Dossier)](#stage-5-the-299d-hybrid-bottleneck-the-courtroom-dossier)
   - [Stage 6: Multi-Task Loss & Margin (The Strict Drill Sergeant)](#stage-6-multi-task-loss--margin-the-strict-drill-sergeant)
   - [Stage 7: Forensic Calibration & The Harmony Prior (The Humane Judge)](#stage-7-forensic-calibration--the-harmony-prior-the-humane-judge)
3. [The 4 Historical Turning Points: "Why We Built It This Way"](#3-the-4-historical-turning-points-why-we-built-it-this-way)
4. [Dataset Inventory & The "Celebrity Shield" Analogy](#4-dataset-inventory--the-celebrity-shield-analogy)
5. [The SOTA Benchmarks & DeLong Test in Plain English](#5-the-sota-benchmarks--delong-test-in-plain-english)
6. [Top 20 Panel Questions: Layman Analogy + Code Defense Scripts](#6-top-20-panel-questions-layman-analogy--code-defense-scripts)
7. [Master Codebase Navigation Cheat Table](#7-master-codebase-navigation-cheat-table)

---

# 1. The 60-Second "Grandmother Pitch" & Master Analogies

### The Core Problem: The Disappearing Brushstrokes
> **Analogy:** Imagine an art forgery. Old forged paintings had obvious mistakes—the paint was smudged, the canvas had weird seams, or the colors looked digital. Traditional deepfake detectors were like magnifying glasses looking for those blurry brushstrokes.  
> But modern AI (like Sora or MuseTalk) is now so advanced that it paints with **zero brushstroke errors**. The picture looks 100% physically perfect. If you only look at pixel quality, you get tricked every single time.

### Our Solution: The Neurobiological Harmony
> **Analogy:** DeepSentinel doesn't look at the brushstrokes. It looks at the **acting and emotional harmony**.  
> In real humans, when you are genuinely terrified, your voice trembles in a specific pitch, your brain chooses distressed words, and your facial muscles (eyebrows pulling up, eyes widening) activate together automatically. It is a biological reflex wired into the human nervous system.  
> But deepfake creators build synthetic videos in **separate, disconnected factories**:
> - Factory 1 clones a voice (TTS / RVC).
> - Factory 2 animates the lips (Wav2Lip).
> - Factory 3 swaps or animates the face (SadTalker / FSGAN).  
> None of these factories talk to each other about genuine human emotion. As a result, the deepfake might have a furious screaming voice coming out of a smiling or emotionless face.  
> **DeepSentinel catches the lie by detecting this cross-modal emotional mismatch ($\boldsymbol{\Delta}$).**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           THE MASTER ANALOGY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  TRADITIONAL DETECTORS: "Is there glue on the edges of the face mask?"      │
│  (Fails when modern AI makes seamless masks)                                │
│                                                                             │
│  DEEPSENTINEL (OURS):   "Why is the person telling a tragedy with a cheerful│
│                          voice while their eyes are completely dead?"       │
│  (Catches deepfakes regardless of how high-definition the pixels are)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 2. The 7-Stage Pipeline: Analogy + Code + Math

```
  Raw Video (.mp4)
       │
  [Stage 1] ──▶ Ingestion & Gatekeeping ──▶ Audio Energy Check (has_speech: bool)
       │
  [Stage 2] ──▶ Tri-Modal Feature Extraction:
                ├── Audio (16kHz Mono) ────────▶ Wav2Vec 2.0 (768D) ──┐
                ├── Audio (Whisper ASR) ───────▶ BERT-Uncased (768D) ──┼─▶ Z_at (1536D) ──┐
                └── Video (RetinaFace 8 Frames) ─▶ ViT-B/16 (768D) ────┼─▶ Z_v  (768D)  ──┼─▶ [Stage 3] CBP (8192D) ──▶ fused_proj (256D)
                                                                       │                  │
  [Stage 4] ───────────────────────────────────────────────────────────┼─▶ EmotionHeadA ─▶ P_A (6D) ──┬─▶ Δ = |P_A - P_B| (6D)
                                                                       ├─▶ EmotionHeadB ─▶ P_B (6D) ──┼─▶ fused_emo = P_A ⊗ P_B (36D)
                                                                       └─▶ SarcasmHead  ──────────────┴─▶ P_sarc (1D)
                                                                                                                │
  [Stage 5] ──────────────────────────────────────────────────────────────────────────────────┐                 │
                                                                                               ▼                 │
                                                                             ┌───────────────────────────────────┴─┐
                                                                             │  299D Hybrid Bottleneck Classifier  │
                                                                             │    256D + 36D + 6D + 1D = 299D      │
                                                                             └──────────────────┬──────────────────┘
                                                                                                │
  [Stage 6] ──▶ Multi-Task Loss: Focal (γ=2) + Margin (m=1.5) + DANN GRL                       ▼
                                                                                   Raw Logit (-∞ to +∞)
                                                                                                │
  [Stage 7] ──▶ Forensic Evidence Engine: D_JS Synchrony + Biological Harmony Prior             ▼
                                                                                   Calibrated Verdict P(fake) ∈ [0, 1]
```

---

### Stage 1: Ingestion & The Silent Gatekeeper
* **Analogy:** The Bouncer at the Door. If a visitor comes in and stays completely quiet, the bouncer tells the lie detector: *"He hasn't spoken a word; don't analyze his breathing noise as a speech confession."*
* **Where in code:** [`webapp/input_validator.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/input_validator.py#L50-L130)
* **How it works:**
  1. Opens video using OpenCV (`cv2.VideoCapture`).
  2. Enforces an interactive 3.0s–20.0s timeline trimming window.
  3. Loads 16 kHz audio via `soundfile.read(dtype='float32')` (avoids Windows C++ DLL crashes in `torchaudio.load()`).
  4. Calculates Root Mean Square (RMS) energy:
     $$E_{\text{RMS}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$$
     If $E_{\text{RMS}} < \text{threshold}$, sets `has_speech = False`.

---

### Stage 2: The Tri-Modal Senses (Ears, Brain, Eyes)
* **Analogy:** A three-person detective team:
  - Detective 1 (**Ears / Wav2Vec 2.0**) listens to pitch, tremor, and voice speed.
  - Detective 2 (**Brain / Whisper + BERT**) transcribes the words and understands context.
  - Detective 3 (**Eyes / RetinaFace + ViT**) watches the facial micro-expressions.
* **Where in code:** [`src/preprocessing/audio.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/audio.py) and [`src/preprocessing/visual.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/visual.py)
* **How it works:**
  1. **Acoustic:** `facebook/wav2vec2-base` extracts 768D vector $Z_{\text{audio}}$.
  2. **Linguistic:** `openai/whisper-base` transcribes speech $\to$ `bert-base-uncased` encodes `[CLS]` token $\to Z_{\text{text}} \in \mathbb{R}^{768}$.
     $$Z_{at} = [Z_{\text{audio}} \,\|\, Z_{\text{text}}] \in \mathbb{R}^{1536}$$
  3. **Visual Face & Landmark Detection:** InsightFace RetinaFace (`det_500m.onnx`) finds 5 landmarks with a 1:1 square crop ($+20\%$ margin).
  4. **Keyframe Sharpness Ranking:** Calculates Laplacian variance:
     $$S = \text{Var}\left(\nabla^2 I\right)$$
     Filters out motion blur and ranks the **top 8 sharpest, most expressive keyframes**.
  5. **ViT & Temporal Dynamics:** `google/vit-base-patch16-224` extracts 8 CLS tokens $(B, 8, 768) \to$ processed by a 2-layer Recurrent GRU ([`detection_model.py#L404`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L404)) $\to Z_v \in \mathbb{R}^{768}$.

---

### Stage 3: Compact Bilinear Pooling (The Kitchen Blender & Compressor)
* **Analogy:** Mixing all ingredients together without letting the recipe book explode.  
  If you cross-multiply every audio feature (1,536) with every visual feature (768), you get **1,179,648 combinations**. That is over 1.1 million numbers for a single video! A computer will crash or memorize noise.  
  Compact Bilinear Pooling is like a **smart compression algorithm** (Count Sketch + FFT) that preserves all the rich cross-modal interactions while shrinking 1.18 million numbers down to **8,192 numbers**, then projects them cleanly into **256 numbers**.
* **Where in code:** [`src/models/bilinear.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/bilinear.py#L24-L87)
* **Mathematical Formula:**
  $$\psi(Z_{at}, Z_v) = \text{FFT}^{-1}\Big(\text{FFT}(\text{CountSketch}(Z_{at})) \odot \text{FFT}(\text{CountSketch}(Z_v))\Big) \in \mathbb{R}^{8192}$$
  $$\mathbf{y} = \text{sign}(\mathbf{x})\sqrt{|\mathbf{x}| + \epsilon}, \quad \mathbf{fused} = \frac{\mathbf{y}}{\|\mathbf{y}\|_2} \xrightarrow{\text{Linear}(8192 \to 256) + \text{LayerNorm} + \text{GELU}} \mathbf{fused\_proj} \in \mathbb{R}^{256}$$

---

### Stage 4: Emotion & Sarcasm Heads (The Lie Detectors)
* **Analogy:** Taking two separate witness statements and checking where they contradict.
* **Where in code:** [`src/models/emotion_heads.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/emotion_heads.py) & [`src/models/sarcasm_head.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/sarcasm_head.py)
* **How it works:**
  1. **Emotion Head A (Audio-Text):** $\text{Linear}(1536 \to 256 \to 6) \implies P_A \in \mathbb{R}^6$.
  2. **Emotion Head B (Visual):** $\text{Linear}(768 \to 256 \to 6) \implies P_B \in \mathbb{R}^6$.
  3. **Ekman 6 Emotion Classes:** `[0: Neutral, 1: Happy, 2: Sad, 3: Angry, 4: Fear, 5: Disgust]`.
  4. **The Incongruence Vector ($\boldsymbol{\Delta}$):**
     $$\boldsymbol{\Delta} = |P_A - P_B| = \Big[|\text{Neu}_A - \text{Neu}_B|,\, |\text{Hap}_A - \text{Hap}_B|,\, \dots,\, |\text{Dis}_A - \text{Dis}_B|\Big] \in \mathbb{R}^6$$
  5. **Affective Co-occurrence ($\mathbf{fused\_emo}$):** Outer product $P_A \otimes P_B \in \mathbb{R}^{36}$ (e.g., probability of Angry Voice + Happy Face).
  6. **Sarcasm Head ($P_{\text{sarc}}$):** $\text{Linear}(1536 \to 256 \to 1) \implies P_{\text{sarc}} \in \mathbb{R}^1$ (prevents genuine sarcasm from being flagged as a deepfake).

---

### Stage 5: The 299D Hybrid Bottleneck (The Courtroom Dossier)
* **Analogy:** Assembling the final evidence folder for the trial judge:
  - 256 pages of rich physical video-audio blending evidence ($\mathbf{fused\_proj}$).
  - 36 pages of joint emotional matrix data ($\mathbf{fused\_emo}$).
  - 6 pages of exact per-emotion discrepancies ($\boldsymbol{\Delta}$).
  - 1 page stating if the person was just being sarcastic ($P_{\text{sarc}}$).
  - **Total Dossier Thickness = $256 + 36 + 6 + 1 = 299\text{ Dimensions}$**.
* **Where in code:** [`src/models/classifier.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/classifier.py) and [`src/models/detection_model.py#L278`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L278)
* **Classification Pipeline:**
  $$\mathbf{x}_{299} \xrightarrow{\text{LayerNorm}} \text{Linear}(299 \to 512) \xrightarrow{\text{GELU}} \text{SE}_{1\text{D}}(512) \xrightarrow{\text{Drop}(0.4)} \text{Linear}(512 \to 128) \xrightarrow{\text{GELU}} \text{Linear}(128 \to 1) \implies \text{Raw Logit}$$
  *(Features 1D Squeeze-and-Excitation to give higher attention to the most decisive evidence lines).*

---

### Stage 6: Multi-Task Loss & Margin (The Strict Drill Sergeant)
* **Analogy:** Training a multi-athlete Olympian. The coach doesn't just grade whether the athlete won the race; they grade their running form, breathing control, and swimming technique all at once.
* **Where in code:** [`src/training/losses.py`](file:///d:/Documents/Programming/Thesis_G10/src/training/losses.py#L60-L120)
* **Multi-Task Loss Formula:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}}(\text{Focal}) + \lambda_a \mathcal{L}_{\text{CE}}^{(A)} + \lambda_b \mathcal{L}_{\text{CE}}^{(B)} + \lambda_{\text{sarc}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dom}} \mathcal{L}_{\text{DANN}} + \lambda_{\text{margin}} \mathcal{L}_{\text{margin}}$$
* **Supervised Margin Loss ($m=1.5, \lambda=0.2$):**
  $$\mathcal{L}_{\text{margin}} = \max\left(0,\, 1.5 - (\bar{s}_{\text{fake}} - \bar{s}_{\text{real}})\right)$$
  Forces the model to push Real scores down to $\sim 0.10$ and Fake scores up to $\sim 0.90$, banning ambiguous $0.50$ scores.
* **Domain Adversarial Training (DANN GRL):** Multiplies gradients by $-\alpha$ to force the network to ignore studio lighting differences between TV clips and webcam videos.

---

### Stage 7: Forensic Calibration & The Harmony Prior (The Humane Judge)
* **Analogy:** Contextual courtroom common sense. If a webcam sensor has yellow lighting or low resolution, an uncalibrated machine might get confused. But if a human is clearly and naturally laughing with both their voice and face in perfect harmony, common sense tells us they are genuine.
* **Where in code:** [`webapp/model_service.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py#L330-L460)
* **How it works:**
  1. **Jensen-Shannon Divergence ($D_{\text{JS}}$):** Measures continuous mathematical synchrony between voice ($P_A$) and face ($P_B$).
  2. **Multimodal Biological Harmony Prior:** When active emotions match and $D_{\text{JS}} \le 0.07$, applies a **$-2.70$ logit bonus**, preventing webcam lighting noise from causing false alarms on authentic human beings.
  3. **Asymmetric Active Sharpening:** Active emotions peak at $60\%\text{–}70\%$ ($T=0.65$), while neutral softening ($T=1.15$) prevents "Neutral" from dominating subtle human expressions.
  4. **8-State Forensic Matrix:** Outputs clear findings like `STATE_REAL_HARMONY`, `STATE_FAKE_EMOTION_DESYNC`, or `STATE_REAL_DEADPAN_IRONY`.

---

# 3. The 4 Historical Turning Points: "Why We Built It This Way"

When the panel asks: *"Why did you make this architectural decision?"*, use this chronological defense story:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      THE 4 TURNING POINTS IN DEVELOPMENT                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. THE 1.18M DIMENSION CRASH ──▶ Solution: Compact Bilinear Pooling (CBP)   │
│ 2. THE SILENT VIDEO CATASTROPHE ──▶ Solution: Speech Energy Gating (RMS)    │
│ 3. THE SARCASM FALSE ALARM ────▶ Solution: Multi-Task Sarcasm Head (MUStARD)│
│ 4. THE WEBCAM "BANO" BIAS ─────▶ Solution: Biological Harmony Prior & D_JS  │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Trial 1: The Outer Product Explosion**  
   * *Problem:* Multiplying 1536D and 768D gave 1,179,648 dimensions, causing CUDA Out-of-Memory crashes and severe overfitting.  
   * *Solution:* Implemented Count Sketch with FFT circular convolution in [`src/models/bilinear.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/bilinear.py), compressing features to 8,192D and projecting to 256D.
2. **Trial 2: The Silent Video Catastrophe (78% Fake on Silence)**  
   * *Problem:* When a real human sat silently in front of their webcam, ambient microphone fan hiss was fed into Wav2Vec, causing emotion heads to hallucinate 92.8% Fear and 81.6% Sarcasm!  
   * *Solution:* Added `has_speech` flag in [`webapp/input_validator.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/input_validator.py). When silent, vocal emotion is mathematically grounded to 100% Neutral ($P_A=[1,0,0,0,0,0], \boldsymbol{\Delta}=\mathbf{0}$).
3. **Trial 3: The Sarcasm False Alarm**  
   * *Problem:* In deadpan humor (e.g. Chandler Bing from Friends), people deliberately say sad/angry things with a straight face. The detector thought this natural irony was a deepfake.  
   * *Solution:* Built a dedicated Sarcasm Head in [`src/models/sarcasm_head.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/sarcasm_head.py) trained on the MUStARD dataset and added $P_{\text{sarc}}$ directly into the 299D bottleneck.
4. **Trial 4: The Real Webcam Calibration Skew (54% Real turning 71% Fake)**  
   * *Problem:* Training on broadcast TV clips (MELD) caused webcam footage to have a slight domain offset. Overly aggressive temperature scaling doubled the logit, turning genuine humans into 71% Fakes.  
   * *Solution:* Implemented the **Biological Harmony Prior** and **Asymmetric Sharpening** in [`webapp/model_service.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py), granting logit bonuses when vocal and facial emotions biologically align.

---

# 4. Dataset Inventory & The "Celebrity Shield" Analogy

Total Core Training Turnover Pool: **17,741 verified multimodal clips** partitioned with **0% speaker overlap**:

| Dataset / Track | Real or Fake | Role in Pipeline | Verified Clips |
| :--- | :--- | :--- | :---: |
| **Track 1** (StyleTTS2 + RVC) | Fake (CREMA-D) | Audio swap (cloned voice + original lips) | 1,452 |
| **Track 2** (Wav2Lip) | Fake (CREMA-D) | Audio swap + AI lip synchronization | 2,267 |
| **Track 3** (SadTalker) | Fake (CREMA-D) | Full face re-animation from a single portrait | 3,722 |
| **MELD Real** | Real (TV Dialogue) | Authentic emotional conversational benchmark | 3,334 |
| **CMU-MOSEI** | Real (YouTube) | Diverse in-the-wild natural video speeches | 6,276 |
| **MUStARD** | Real (Sarcasm) | Multimodal sarcasm & rhetorical irony | 690 |
| **TOTAL DATASET POOL** | **80/10/10 Split** | **Triple-stratified speaker disjoint** | **17,741** |

### The "Celebrity Shield" Analogy (External Evaluation on FakeAVCeleb)
> **Analogy:** Imagine preparing a student for a final history exam. You give them practice questions on *European History (Celebrity Group A)*. But for the final exam, you test them exclusively on *Asian History (Celebrity Group B)* with questions they have never seen before.  
> We partitioned the FakeAVCeleb benchmark ($N=700$) such that **Celebrity Set A $\cap$ Celebrity Set B = $\emptyset$ (Zero Overlap)**. The detector was tested on completely unseen humans, proving it learned generalizable human emotion, not individual faces.

---

# 5. The SOTA Benchmarks & DeLong Test in Plain English

### Benchmark Results on Balanced FakeAVCeleb Parity ($N=700$)

| Architecture | Detection Basis | Accuracy | Fake Recall | F1-Score | MCC | AUC-ROC [95% CI] | DeLong vs DeepSentinel |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepSentinel (Ours)** | **Affect-Bilinear Multi-Head** | **82.14%** | **87.14%** | **0.8299** | **+0.6461** | **0.9020** [0.877–0.924] | **Reference Baseline** |
| **AceNet** | Cross-Attention Multi-Modal | 64.00% | 52.00% | 0.5909 | +0.2884 | 0.6425 [0.600–0.682] | $p < 0.001$ ($Z = 9.87$) |
| **MesoNet-4** | Spatial Texture CNN | 52.00% | 48.57% | 0.5030 | +0.0401 | 0.5389 [0.495–0.583] | $p < 0.001$ ($Z = 12.61$) |
| **LipForensics** | Spatiotemporal Lip Sync | 52.00% | 50.00% | 0.5102 | +0.0400 | 0.5132 [0.469–0.553] | $p < 0.001$ ($Z = 13.44$) |
| **XceptionNet** | Deep Spatial CNN | 50.57% | 51.14% | 0.5085 | +0.0114 | 0.5002 [0.458–0.542] | $p < 0.001$ ($Z = 13.98$) |
| **ResNet-AV** | Feature Concatenation | 46.14% | 45.14% | 0.4560 | -0.0772 | 0.4629 [0.419–0.506] | $p < 0.001$ ($Z = 15.22$) |

### The "Two Doctors" Analogy for DeLong's Significance Test
* **What is a DeLong Test?**  
  > **Analogy:** If Doctor A diagnoses 82 out of 100 patients correctly, and Doctor B diagnoses 64 out of 100, is Doctor A truly smarter, or did they just get lucky with easier patients?  
  > The **DeLong test** is a strict statistical formula that compares both doctors on the *exact same 700 patients side-by-side*.
* **Our DeLong Result:**  
  DeepSentinel's **$+25.95\%$ AUC lead over AceNet** yields $Z=9.87$ and **$p < 0.001$**. In statistics, a $p$-value below $0.001$ means there is less than a 1-in-1,000 chance this occurred by luck. The superiority is mathematically proven.

---

# 6. Top 20 Panel Questions: Layman Analogy + Code Defense Scripts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TOP PANEL INQUIRIES QUICK-FIRE SCRIPT                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Q1: "What is your central thesis contribution in one sentence?"
> **Spoken Answer:**  
> *"Traditional detectors look for pixel flaws which disappear with modern generative AI; DeepSentinel instead detects synthetic media by measuring the biological and affective incongruence between facial expressions and speech prosody."*

#### Q2: "Why use Compact Bilinear Pooling instead of simple feature concatenation?"
> **Layman Analogy:** Concatenation is like putting salt and pepper in separate bowls side by side. Bilinear pooling is like grinding them together into every bite.  
> **Code Defense:** *"Concatenation only captures linear relationships. CBP captures full multiplicative outer-product interactions ($Z_{at} \otimes Z_v$). We use Count Sketch FFT convolution in [`src/models/bilinear.py:L78-L86`](file:///d:/Documents/Programming/Thesis_G10/src/models/bilinear.py#L78-L86) to compress 1.18 million dimensions down to 8,192D with signed-sqrt and L2 normalization."*

#### Q3: "Where does the number 299 in your bottleneck classifier come from?"
> **Code Defense:** *"In [`src/models/detection_model.py:L278`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L278), the classifier input is $\mathbf{x}_{299} = [\mathbf{fused\_proj}_{256\text{D}} \,\|\, \mathbf{fused\_emo}_{36\text{D}} \,\|\, \boldsymbol{\Delta}_{6\text{D}} \,\|\, P_{\text{sarc}}_{1\text{D}}]$. Exactly $256 + 36 + 6 + 1 = 299$ dimensions."*

#### Q4: "How do you handle silent videos without hallucinating emotions?"
> **Layman Analogy:** If the suspect is silent, you don't interpret their room fan noise as a confession.  
> **Code Defense:** *"In [`webapp/input_validator.py:L72`](file:///d:/Documents/Programming/Thesis_G10/webapp/input_validator.py#L72), audio RMS energy sets `has_speech`. In [`detection_model.py:L254-L262`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L254-L262), if `not has_speech`, cross-attention is bypassed, voice emotion is set to 100% Neutral, sarcasm is 0, and $\boldsymbol{\Delta}=\mathbf{0}$."*

#### Q5: "How does your model distinguish sarcasm from a deepfake mismatch?"
> **Layman Analogy:** In sarcasm, a person intentionally speaks with a monotone or contradictory tone to make a joke. It is natural human irony, not a malicious computer glitch.  
> **Code Defense:** *"We implemented a dedicated Sarcasm Head in [`src/models/sarcasm_head.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/sarcasm_head.py) trained on the MUStARD dataset. $P_{\text{sarc}}$ is explicitly concatenated into the 299D classifier so the network learns to differentiate sarcasm from synthesis."*

#### Q6: "Why did you use 8 keyframes instead of analyzing all 300 frames in a video?"
> **Layman Analogy:** If you take 300 photos of someone sitting in a chair, 290 of them are identical duplicates. You only need the 8 clearest photos where their face is actively moving.  
> **Code Defense:** *"In [`src/preprocessing/visual.py:L130-L210`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/visual.py#L130-L210), we rank frames by Laplacian variance sharpness ($S=\text{Var}(\nabla^2 I)$) and motion gating ($>0.30$). Selecting 8 keyframes reduces computational load by $97\%$ while preserving peak facial Action Units."*

#### Q7: "What prevents data leakage between your training and test datasets?"
> **Code Defense:** *"In [`docs/FEW_SHOT_ADAPTATION_AND_DEFENSE_STRATEGY.md`](file:///d:/Documents/Programming/Thesis_G10/docs/FEW_SHOT_ADAPTATION_AND_DEFENSE_STRATEGY.md), we enforce the Pre-Sampling Identity Shield: $\text{Celebrity Set A} \cap \text{Celebrity Set B} = \emptyset$. Checkpoint `best_phase2_adapted.pt` was adapted on Set A; the 700 evaluation clips come strictly from Set B ($0\%$ identity overlap)."*

#### Q8: "What loss functions did you use during training?"
> **Code Defense:** *"In [`src/training/losses.py:L60-L100`](file:///d:/Documents/Programming/Thesis_G10/src/training/losses.py#L60-L100), we formulated Multi-Task Loss: Focal Loss ($\gamma=2, \text{pos\_weight}=1.3835$), Supervised Margin Loss ($m=1.5, \lambda=0.2$), Emotion Cross-Entropy, Sarcasm BCE, and DANN Domain loss."*

#### Q9: "Why did you add Supervised Margin Loss ($m=1.5$)?"
> **Layman Analogy:** If a student scores 51% on a test, you're not sure if they really know the material. Margin loss forces the model to separate grades decisively—A's ($>90\%$) for fakes and F's ($<10\%$) for reals.  
> **Code Defense:** *"Margin loss $\max(0, 1.5 - (\bar{s}_{\text{fake}} - \bar{s}_{\text{real}}))$ prevents logit compression around 0.50, forcing a confident gap between classes."*

#### Q10: "What is DANN and why is it in your model?"
> **Layman Analogy:** Removing the yellow studio tint so the model doesn't think 'yellow lighting' means 'deepfake'.  
> **Code Defense:** *"Domain-Adversarial Neural Network with Gradient Reversal Layer in [`detection_model.py:L31-L56`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L31-L56). It forces the bottleneck to learn domain-invariant features across all 5 dataset sources."*

#### Q11: "What is the Biological Harmony Prior in your web application?"
> **Code Defense:** *"Implemented in [`webapp/model_service.py:L446-L455`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py#L446-L455). In genuine human discourse, active emotion concordance ($D_{\text{JS}} \le 0.07$) is proof of authenticity. We apply a $-2.70$ logit bonus to prevent consumer webcam noise from triggering false alarms."*

#### Q12: "How does the web application stream real-time telemetry to the user?"
> **Code Defense:** *"In [`webapp/main.py:L200-L245`](file:///d:/Documents/Programming/Thesis_G10/webapp/main.py#L200-L245), we implement Server-Sent Events (SSE) streaming at `/analyze/stream`. It streams RetinaFace bounding boxes, 5 landmarks, and live Whisper transcripts directly to [`webapp/static/js/app.js`](file:///d:/Documents/Programming/Thesis_G10/webapp/static/js/app.js)."*

#### Q13: "What are the 6 basic emotions used in the model?"
> **Code Defense:** *"Ekman's 6 basic universal emotions: Neutral (0), Happy (1), Sad (2), Angry (3), Fear (4), and Disgust (5), declared in [`src/models/emotion_heads.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/emotion_heads.py)."*

#### Q14: "Why does the model achieve 98.5% on compound attacks (fsgan-wav2lip)?"
> **Code Defense:** *"Compound attacks tamper with both the face and the audio, creating extreme affective dissonance ($\boldsymbol{\Delta} \gg 0$). DeepSentinel's multi-head architecture catches this immediately."*

#### Q15: "Why did MesoNet-4 and XceptionNet score near 50% on FakeAVCeleb?"
> **Code Defense:** *"MesoNet-4 and XceptionNet rely exclusively on spatial pixel artifacts. Because FakeAVCeleb uses high-quality neural blending, pixel artifacts are minimal, causing conventional CNNs to collapse to random guessing ($50.0\%\text{–}52.0\%$ AUC)."*

#### Q16: "What is the role of the 1D Squeeze-and-Excitation layer in the classifier?"
> **Code Defense:** *"In [`src/models/classifier.py:L15-L28`](file:///d:/Documents/Programming/Thesis_G10/src/models/classifier.py#L15-L28), `SqueezeExcitation1D` applies self-attention across the 512 channels, learning to dynamically boost discrepancy signals ($\boldsymbol{\Delta}$) when visual features are ambiguous."*

#### Q17: "How is the model protected against out-of-distribution audio crashes?"
> **Code Defense:** *"In [`src/preprocessing/audio.py:L40-L60`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/audio.py#L40-L60), we use `soundfile.read` with fallback resampling and float32 conversion, bypassing Windows `torchaudio.load()` TorchCodec DLL failures."*

#### Q18: "What is Jensen-Shannon Divergence and why use it over argmax label comparison?"
> **Layman Analogy:** Instead of only looking at the #1 winning emotion, you compare the full emotional profile (e.g., 60% happy + 20% surprised vs 55% happy + 25% surprised).  
> **Code Defense:** *"In [`webapp/model_service.py:L413-L420`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py#L413-L420), $D_{\text{JS}}(P_A \parallel P_B)$ measures smooth information-theoretic divergence, avoiding brittle discrete label flips."*

#### Q19: "What is Phase 1 vs Phase 2 training?"
> **Code Defense:** *"Phase 1 (`scripts/colab_stage1.py`) trains the 299D bottleneck on frozen precomputed features across 14,193 clips. Phase 2 (`scripts/train_adaptation.py`) performs few-shot domain adaptation by unfreezing top-4 Wav2Vec2 and ViT layers on 300 adaptation clips."*

#### Q20: "Can DeepSentinel detect deepfakes where someone cloned both the face and voice perfectly?"
> **Code Defense:** *"Yes. Even if both face and voice are cloned, independent generative models cannot replicate the spontaneous millisecond-level neuromuscular coupling between vocal pitch inflection and micro-expression Action Units (Ekman & Friesen, 1969)."*

---

# 7. Master Codebase Navigation Cheat Table

| What the Panel Asks About | Exactly Where It Is in the Code | Line Numbers |
| :--- | :--- | :--- |
| **Input Validation & Silence Gating** | [`webapp/input_validator.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/input_validator.py) | `L50–L130` (`validate_video`, `validate_speech_presence`) |
| **Audio Loading & Resampling** | [`src/preprocessing/audio.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/audio.py) | `L40–L70` (`load_audio_waveform`) |
| **Wav2Vec 2.0 Feature Extraction** | [`src/preprocessing/audio.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/audio.py) | `L75–L95` (`extract_audio_features`) |
| **Whisper ASR & BERT Embedding** | [`src/preprocessing/audio.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/audio.py) | `L100–L165` (`transcribe_speech`, `extract_text_features`) |
| **RetinaFace 5-Point Landmarks** | [`src/preprocessing/visual.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/visual.py) | `L45–L120` (`detect_faces_retinaface`, `crop_face_square`) |
| **Laplacian Sharpness Keyframes** | [`src/preprocessing/visual.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/visual.py) | `L130–L210` (`select_keyframes`, `_rank_frames`) |
| **ViT-Base/16 Visual Extraction** | [`src/preprocessing/visual.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/visual.py) | `L220–L280` (`extract_visual_features`) |
| **Temporal GRU Dynamics** | [`src/models/detection_model.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py) | `L404` (`self.vit_gru = nn.GRU(768, 768, 2)`) |
| **Compact Bilinear Pooling (CBP)** | [`src/models/bilinear.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/bilinear.py) | `L24–L87` (`CompactBilinearFusion`, `_sketch`, FFT) |
| **Sub-Symbolic Projection (256D)** | [`src/models/detection_model.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py) | `L277` (`fused_proj = F.gelu(self.proj_ln(...))`) |
| **Emotion Heads A & B** | [`src/models/emotion_heads.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/emotion_heads.py) | `L15–L55` (`EmotionHeadA`, `EmotionHeadB`) |
| **Emotion Incongruence ($\boldsymbol{\Delta}$)** | [`src/models/detection_model.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py) | `L264` (`delta = torch.abs(prob_a - prob_b)`) |
| **Affect Co-occurrence (36D)** | [`src/models/detection_model.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py) | `L275–L276` (`outer = torch.bmm(...)`) |
| **Sarcasm Head & Gating** | [`src/models/sarcasm_head.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/sarcasm_head.py) | `L10–L25` (`SarcasmHead`) |
| **299D Concatenation Assembly** | [`src/models/detection_model.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py) | `L278` (`combined = torch.cat([fused_proj, ...])`) |
| **Classifier MLP & 1D SE Layer** | [`src/models/classifier.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/classifier.py) | `L15–L55` (`SqueezeExcitation1D`, `ClassifierMLP`) |
| **Domain Adversarial GRL (DANN)** | [`src/models/detection_model.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py) | `L31–L56` (`GradientReversalLayer`) |
| **Multi-Task & Margin Loss** | [`src/training/losses.py`](file:///d:/Documents/Programming/Thesis_G10/src/training/losses.py) | `L60–L120` (`MultiTaskLoss`, Margin $m=1.5$) |
| **DeLong Significance Test** | [`src/evaluation/significance.py`](file:///d:/Documents/Programming/Thesis_G10/src/evaluation/significance.py) | `L18–L150` (`delong_test`, `_fast_delong`) |
| **SOTA Benchmark Visualizer** | [`scripts/plot_sota_comparisons.py`](file:///d:/Documents/Programming/Thesis_G10/scripts/plot_sota_comparisons.py) | `L50–L250` (ROC curves, comparative bar charts) |
| **Biological Harmony Prior** | [`webapp/model_service.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py) | `L446–L455` (`harmony_bonus`) |
| **Jensen-Shannon Synchrony $D_{\text{JS}}$** | [`webapp/model_service.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py) | `L413–L420` (`d_js = 0.5 * (kl_a + kl_b)`) |
| **Asymmetric Active Sharpening** | [`webapp/model_service.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py) | `L357–L366` (`_calibrate_emotion_probs`, $T=0.65$) |
| **Live SSE Telemetry Stream HUD** | [`webapp/main.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/main.py) | `L200–L245` (`/analyze/stream`, bounding box & landmarks) |
| **Dominant Emotion Banner Card** | [`webapp/static/js/app.js`](file:///d:/Documents/Programming/Thesis_G10/webapp/static/js/app.js) | `L120–L180` (`EMO_COLORS`, `dominant-card`) |

---
*Created for Thesis G10 Oral Defense & Code Review Sessions (BSCS 2026).*
