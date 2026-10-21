# DeepSentinel: Master Pipeline Defense & Inspection Cheat Sheet

> **Document Type:** Oral Defense & Professor Inspection Cheat Sheet  
> **Repository Root:** `d:\Documents\Programming\Thesis_G10`  
> **Python Environment:** `.venv\Scripts\python.exe` (Python 3.11)  
> **Core Status:** 100% De-blackboxed, First-Principles Mathematics, Zero Third-Party Evaluation Wrappers  

---

## 1. Quick Terminal Commands Summary

All commands are run from the **project root directory** (`d:\Documents\Programming\Thesis_G10`) with the virtual environment activated:

```powershell
# 1. Activate Environment
.venv\Scripts\Activate.ps1
```

### A. Terminal Inspection of Raw Formulas & Results
| Command | Module / Purpose | What the Professor Sees |
| :--- | :--- | :--- |
| `python scripts/inspection/inspect_pipeline.py --part evaluation`<br>*(or `python -m src.evaluation.metrics`)* | **Evaluation Metrics** | Raw Contingency ($TP=305, TN=270, FP=80, FN=45, N=700$), step-by-step arithmetic (82.14% Acc, 79.22% Prec, 87.14% Rec, 77.14% Spec, 0.8299 F1, 0.9020 AUC), and per-pipeline breakdown. |
| `python scripts/inspection/inspect_pipeline.py --part cbp`<br>*(or `python -m src.models.bilinear`)* | **Compact Bilinear Pooling** | Count Sketch hashing ($1.18\text{M} \to 8,192$), circular convolution via FFT, signed sqrt power norm, and unit L2-norm verification ($\lVert\mathbf{fused}\rVert_2 = 1.000000$). |
| `python scripts/inspection/inspect_pipeline.py --part model`<br>*(or `python -m src.models.detection_model`)* | **Model Dissection (299D)** | Dual Softmax ($P_A, P_B \in \Delta^5$), Incongruence $\boldsymbol{\Delta} = \|P_A - P_B\|$, Kronecker co-occurrence ($36\text{D}$), bottleneck projection ($256\text{D}$), 299D assembly ($256+36+6+1$), and logistic Sigmoid. |
| `python scripts/inspection/inspect_pipeline.py --part losses`<br>*(or `python -m src.training.losses`)* | **Multi-Task Losses** | Focal Loss, Masked Categorical Cross-Entropy (`ignore_index=-1`), Contrastive Margin separation, and weighted total loss composition. |
| `python scripts/inspection/inspect_pipeline.py --part significance`<br>*(or `python -m src.evaluation.significance`)* | **Significance Testing** | Sun & Xu (2014) paired DeLong test on AUC vs ACE-Net ($p < 0.0001$), Efron paired bootstrap ($N=2,000$ resamples) with 95% Confidence Intervals. |
| `python scripts/inspection/inspect_pipeline.py --part all` | **Full Pipeline Audit** | Runs all 5 modules sequentially in ~10 seconds. |

### B. Live Training Flow Demonstration (From Scratch)
| Command | Mode | Description |
| :--- | :--- | :--- |
| `python scripts/inspection/demo_training_flow.py` | **Full Cycle (8 clips)** | Runs an end-to-end training cycle on 8 demo clips (4 Real + 4 Fake) in ~4 seconds: Ingestion $\to$ CBP $\to$ Discrepancy $\to$ 299D Assembly $\to$ Multi-Task Loss $\to$ Backpropagation $\to$ Optimizer Update $\to$ Validation step. |
| `python scripts/inspection/demo_training_flow.py --stage forward` | **Forward Pass Only** | Ingestion through 299D bottleneck assembly and Sigmoid probabilities. |
| `python scripts/inspection/demo_training_flow.py --stage loss` | **Loss Computation Only** | Displays individual loss components and weighted sum arithmetic. |
| `python scripts/inspection/demo_training_flow.py --stage backward` | **Gradient Flow Audit** | Runs `loss.backward()` and prints parameter counts + gradient norms for all 7 modules (proves zero vanishing gradients). |
| `python scripts/inspection/demo_training_flow.py --stage validation` | **Validation Only** | Evaluates validation clips and extracts instant contingency matrix and accuracy. |
| `python scripts/inspection/demo_training_flow.py --clips 16` | **Extended Demo** | Scales demo run to 16 clips (8 Real + 8 Fake). |

---

## 2. Where the Pipeline Originated

### A. Pretrained Feature Backbones
1. **Acoustic Backbone:** `facebook/wav2vec2-base-960h`
   - Pretrained on 960 hours of LibriSpeech. Extracts 768-dimensional temporal acoustic representations from raw 16kHz audio waveforms.
2. **Linguistic Backbone:** `google-bert/bert-base-uncased`
   - 12-layer transformer pretrained on BookCorpus and English Wikipedia. Extracts 768-dimensional contextual linguistic embeddings from speech transcripts.
   - **Combined Acoustic-Text Representation:** $\mathbf{Z}_{at} = [\mathbf{z}_{\text{wav2vec}} \,\|\, \mathbf{z}_{\text{bert}}] \in \mathbb{R}^{1536}$.
3. **Visual Facial Backbone:** `google/vit-base-patch16-224`
   - Vision Transformer pretrained on ImageNet-21k. Extracts 768-dimensional visual feature vectors from cropped facial keyframes, aggregated temporally via a bidirectional GRU: $\mathbf{Z}_v \in \mathbb{R}^{768}$.

### B. Datasets & Ground-Truth Origins
1. **Synthetic Deepfake Manipulation Corpora (FakeAVCeleb & Tracks):**
   - **Video Manipulations:** Faceswap, FSGAN (facial reenactment).
   - **Audio Manipulations:** Real-Time Voice Cloning (RTVC), StyleTTS.
   - **Audio-Visual Lip-Sync Manipulations:** Wav2Lip, SadTalker.
   - **Synthetic Pipelines Tested:** Faceswap, Faceswap-Wav2Lip, FSGAN, FSGAN-Wav2Lip, Wav2Lip, RTVC.
2. **Authentic Emotional Ground-Truth Corpora:**
   - **CREMA-D:** 7,442 clips across 91 actors labeled with 6 basic emotions (*Anger, Disgust, Fear, Happy, Neutral, Sad*).
   - **MELD:** Multi-party television dialogue across 1,433 conversations with multimodal emotion annotations.
   - **CMU-MOSEI:** 23,453 multimodal video segments annotated for emotional sentiment.
3. **Sarcasm Detection Ground-Truth:**
   - **MUStARD:** Multimodal Sarcasm Detection Dataset derived from sitcom dialogues (*Friends, The Big Bang Theory*).
4. **Official Balanced Evaluation Benchmark ($N=700$):**
   - Stored at: `data/eval_results/annotated_nonbinary_full.csv`
   - Perfectly balanced: **350 Authentic Clips** vs **350 Synthetic Clips** across all 5 generation pipelines.

---

## 3. Model Dissection & Mathematical Architecture

### Stage 1: Compact Bilinear Pooling (Pham & Pagh, 2013; Fukui et al., 2016)
- **Full Cross-Modal Outer Product:** $\mathbf{Z}_{\text{outer}} = \mathbf{Z}_{at} \otimes \mathbf{Z}_v \in \mathbb{R}^{1536 \times 768} = 1,179,648\text{D}$ (intractable, causes severe overfitting).
- **Count Sketch Projection:** Project $\mathbf{x} \in \mathbb{R}^d$ into $\mathbb{R}^D$ ($D=8,192$) using random hash $h(i) \in [0, D-1]$ and Rademacher sign $s(i) \in \{-1, +1\}$:
  $$\Psi(\mathbf{x}, h, s)_j = \sum_{i: h(i)=j} s(i) x_i$$
- **FFT Circular Convolution (Convolution Theorem):**
  $$\mathbf{x}_{\text{CBP}} = \mathcal{F}^{-1}\left( \mathcal{F}(\Psi_1) \odot \mathcal{F}(\Psi_2) \right)$$
- **Signed Square-Root Power Normalization:**
  $$y_k = \text{sign}(x_k) \sqrt{|x_k| + \epsilon}$$
- **L2 Vector Normalization:**
  $$\mathbf{fused} = \frac{\mathbf{y}}{\sqrt{\sum_{k=1}^D y_k^2 + \epsilon}} \implies \lVert\mathbf{fused}\rVert_2 = 1.000000$$

### Stage 2: Affective Incongruence & Bottleneck Assembly
- **Emotion Probabilities (Softmax on $\mathbb{R}^6$):**
  $$P_k = \frac{\exp(z_k - \max \mathbf{z})}{\sum_j \exp(z_j - \max \mathbf{z})}, \quad \sum_{k=1}^6 P_k = 1.000000$$
- **Affective Discrepancy Vector & Magnitude:**
  $$\boldsymbol{\Delta} = |P_A - P_V| \in \mathbb{R}^6, \quad \lVert\boldsymbol{\Delta}\rVert_2 = \sqrt{\sum_{k=1}^6 \Delta_k^2}$$
- **Affect Co-occurrence Kronecker Outer Product:**
  $$\mathbf{fused\_emo} = \text{vec}(P_A \otimes P_V) \in \mathbb{R}^{36}, \quad \sum_{i=1}^{36} \mathbf{fused\_emo}_i = 1.000000$$
- **Bottleneck Projection:**
  $$\mathbf{fused\_proj} = \text{GELU}(\text{LayerNorm}(\mathbf{W}_{\text{proj}}\mathbf{fused} + \mathbf{b})) \in \mathbb{R}^{256}$$
- **299D Hybrid Representation Assembly:**
  $$\mathbf{x}_{299} = [\mathbf{fused\_proj}\,(256) \,\|\, \mathbf{fused\_emo}\,(36) \,\|\, \boldsymbol{\Delta}\,(6) \,\|\, s_{\text{sarc}}\,(1)] \in \mathbb{R}^{299}$$
- **Classifier MLP & Logistic Sigmoid Detection:**
  $$P(\text{fake}) = \frac{1}{1 + \exp(-\text{logit})}$$
- **DANN Domain Adversarial Classifier:**
  - Gradient Reversal Layer with schedule: $\alpha(p) = \frac{2}{1 + \exp(-10p)} - 1$ across 5 source domains.

### Stage 3: Multi-Task Loss Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + 0.30\mathcal{L}_{\text{emoA}} + 0.30\mathcal{L}_{\text{emoB}} + 0.20\mathcal{L}_{\text{sarc}} + 0.10\mathcal{L}_{\text{domain}} + 0.05\mathcal{L}_{\text{margin}}$$
- **Focal Loss:** $\mathcal{L}_{\text{FL}} = \frac{1}{N}\sum \alpha_t (1 - p_t)^\gamma \text{BCE}(\text{logit}, y)$
- **Masked Categorical Cross-Entropy:** Numerically stable LogSumExp NLL with `ignore_index=-1`.
- **Contrastive Margin Loss:** $\max(0, 1.5 - (\bar{s}_{\text{fake}} - \bar{s}_{\text{real}}))$.

---

## 4. Final Model Official Evaluation Results

### A. Raw Contingency Matrix ($N=700$)
- **True Positives (TP):** **305** (Synthetic deepfake correctly detected)
- **True Negatives (TN):** **270** (Authentic media correctly accepted)
- **False Positives (FP):** **80** (Authentic media misflagged as fake)
- **False Negatives (FN):** **45** (Synthetic deepfake missed as real)
- **Total Evaluated Clips (N):** **700** $= 305 + 270 + 80 + 45$

### B. Step-by-Step Metric Derivations
$$\text{Accuracy} = \frac{TP + TN}{N} = \frac{305 + 270}{700} = \frac{575}{700} = \mathbf{82.14\%}$$

$$\text{Precision (PPV)} = \frac{TP}{TP + FP} = \frac{305}{305 + 80} = \frac{305}{385} = \mathbf{79.22\%}$$

$$\text{Recall (Sensitivity / TPR)} = \frac{TP}{TP + FN} = \frac{305}{305 + 45} = \frac{305}{350} = \mathbf{87.14\%}$$

$$\text{Specificity (TNR)} = \frac{TN}{TN + FP} = \frac{270}{270 + 80} = \frac{270}{350} = \mathbf{77.14\%}$$

$$\text{Balanced Accuracy} = \frac{\text{Recall} + \text{Specificity}}{2} = \frac{87.14\% + 77.14\%}{2} = \mathbf{82.14\%}$$

$$\text{F1-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times 0.792208 \times 0.871429}{0.792208 + 0.871429} = \mathbf{0.8299}$$

$$\text{Matthews Correlation Coefficient (MCC)} = \frac{305\times 270 - 80\times 45}{\sqrt{385 \times 350 \times 350 \times 315}} = \mathbf{+0.6461}$$

$$\text{Negative Predictive Value (NPV)} = \frac{TN}{TN + FN} = \frac{270}{270 + 45} = \mathbf{85.71\%}$$

$$\text{False Positive Rate (FPR)} = \frac{FP}{FP + TN} = \frac{80}{80 + 270} = \mathbf{22.86\%}$$

$$\text{False Negative Rate (FNR)} = \frac{FN}{FN + TP} = \frac{45}{45 + 305} = \mathbf{12.86\%}$$

$$\text{ROC-AUC} = \text{Mann-Whitney U Rank Integral} = \mathbf{0.9020}$$

### C. Per-Pipeline Breakdown
| Generation Pipeline | Clip Count ($n$) | Accuracy | F1-Score | Recall | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **faceswap** | 13 | **84.62%** | 0.9167 | 84.62% | 0.00% |
| **faceswap-wav2lip** | 58 | **98.28%** | 0.9913 | 98.28% | 0.00% |
| **fsgan** | 40 | **62.50%** | 0.7692 | 62.50% | 0.00% |
| **fsgan-wav2lip** | 69 | **98.55%** | 0.9927 | 98.55% | 0.00% |
| **real (Authentic)** | 350 | **77.14%** | 0.0000 | 0.00% | 77.14% |
| **rtvc** | 5 | **60.00%** | 0.7500 | 60.00% | 0.00% |
| **wav2lip** | 165 | **85.45%** | 0.9216 | 85.45% | 0.00% |

### D. Statistical Significance vs Baselines (Paired on Identical 700 Clips)
- **Sun & Xu (2014) Paired DeLong's Test on AUC:**
  - $\text{AUC}_{\text{DeepSentinel}} = \mathbf{0.9018}$ vs $\text{AUC}_{\text{ACE-Net}} = \mathbf{0.6425}$
  - Difference: $+0.2593$ | $z = 11.2672$ | $p = 0.0000$ ($p < 0.001 \implies$ **Statistically Significant**)
- **Efron Paired Bootstrap Scorecard ($N=2,000$ iterations):**
  - **Accuracy:** DeepSentinel $82.57\%$ vs ACE-Net $64.00\%$ $\to$ Diff: **$+18.57\%$** $[14.00\%, 22.86\%]$, $p < 0.0001^*$
  - **Recall:** DeepSentinel $84.00\%$ vs ACE-Net $52.00\%$ $\to$ Diff: **$+32.00\%$** $[26.06\%, 38.44\%]$, $p < 0.0001^*$
  - **F1-Score:** DeepSentinel $82.82\%$ vs ACE-Net $59.09\%$ $\to$ Diff: **$+23.73\%$** $[18.85\%, 28.87\%]$, $p < 0.0001^*$
  - **ROC-AUC:** DeepSentinel $90.18\%$ vs ACE-Net $64.25\%$ $\to$ Diff: **$+25.93\%$** $[21.62\%, 30.56\%]$, $p < 0.0001^*$

---

## 5. Repository Script Directory Layout

All scripts have been organized into functional subdirectories under `scripts/`:

```text
scripts/
├── inspection/               <-- DEFENSE & DEMO SCRIPTS (RUN THESE FOR THE PROFESSOR)
│   ├── __init__.py
│   ├── inspect_pipeline.py   # Master inspection CLI (--part evaluation/cbp/model/losses/significance/all)
│   └── demo_training_flow.py # End-to-end training loop demonstration (8 clips, 7 stages)
│
├── evaluation/               <-- BENCHMARKING & METRIC EVALUATION
│   ├── __init__.py
│   ├── evaluate.py
│   ├── evaluate_all_models.py
│   ├── evaluate_fakeavceleb.py
│   ├── evaluate_internal_test.py
│   ├── evaluate_sarcasm.py
│   ├── eval_acenet_baseline.py
│   ├── eval_public_baselines.py
│   ├── compare_frameworks.py
│   ├── baseline_logreg.py
│   ├── export_comparative_benchmark_reference.py
│   └── plot_sota_comparisons.py
│
├── training/                 <-- MODEL TRAINING & OPTIMIZATION
│   ├── __init__.py
│   ├── train.py
│   ├── train_adaptation.py
│   ├── train_full.py
│   ├── print_training_summary.py
│   ├── colab_run_stage.py
│   ├── colab_stage1.py
│   └── colab_stage2.py
│
├── preprocessing/            <-- DATASET EXTRACTION & MANIFEST GENERATION
│   ├── __init__.py
│   ├── preprocess_all.py
│   ├── build_fakeavceleb_manifest.py
│   ├── build_preprocessed_manifest.py
│   ├── create_dataset_splits.py
│   ├── segment_cmumosei.py
│   ├── generate_track4_musetalk.py
│   └── setup_musetalk.py
│
└── verification/             <-- SPECIFICATION & INFERENCE CHECKING
    ├── __init__.py
    ├── verify_cache_spec.py
    └── verify_inference_pipeline.py
```

*(Note: Clean proxies remain in `scripts/evaluate.py`, `scripts/train.py`, `scripts/preprocess_all.py`, and `scripts/inspect_pipeline.py` so legacy commands continue to execute seamlessly).*
