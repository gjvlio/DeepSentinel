# Experiment Paper — Filled Experimental Results & Tally Sheets
> **Document:** DeepSentinel Experimental Paper (Group 10)  
> **Source Document Template:** `docs/[EMPTY] Experiment_Paper_DeepSentinel_Group10 (1).pdf`  
> **Benchmark Dataset:** FakeAVCeleb v1.2 (Strict Cross-Dataset Zero-Shot Protocol, $N = 700$ Balanced)  
> **Pretraining / Multi-Task Corpora:** CREMA-D ($N = 7,442$), MUStARD ($N = 690$), MELD ($N = 13,708$), CMU-MOSEI ($N = 22,856$)  
> **Associated Data Artifacts:**
> - `data/eval_results/fakeavceleb_eval_predictions_350+350.csv` (Raw Model Scores)
> - `data/eval_results/annotated_nonbinary_full.csv` (700-Clip Annotated Tally Sheet)
> - `data/eval_results/comparative_benchmark_data.json` (Baseline Evaluations & DeLong Statistics)
> - `data/eval_results/confusion_matrix_summary.txt` (Operating Point Matrix)
> - `data/eval_results/continuous_score_defense.md` (Forensic Calibration & Score Spectrum)

---

## Executive Summary & Key Highlights
1. **Zero Domain Collapse:** On unseen, in-the-wild YouTube deepfakes from FakeAVCeleb, DeepSentinel achieves an **AUC of 0.9020** (95% CI: [0.8770 – 0.9240]) and a **Balanced Accuracy of 82.14%**, outperforming standard visual detectors (MesoNet-4: 52.00%, XceptionNet: 50.57%) and multimodal baselines (LipForensics: 52.00%, Multimodal ResNet-AV: 46.14%, ACE-Net: 64.00%) with overwhelming statistical significance ($p < 0.001$, DeLong paired test).
2. **Compound Attack Superiority:** On complex multi-tier manipulations (`fsgan-wav2lip` and `faceswap-wav2lip`), where both identity and lip movements are synthetically altered, DeepSentinel achieves **98.55%** and **98.28%** accuracy respectively. Generative tools smooth away pixel blending seams but amplify cross-modal affective dissonance.
3. **5-Tier Forensic Confidence Spectrum:** DeepSentinel does not operate on an arbitrary binary knife-edge. Calibrated around the 50% midpoint, **92.00%** of all test clips (644/700) fall into High-Confidence decisive zones (0%–30% and 70%–100%) with **84.78%** accuracy, leaving only **8.00%** in borderline transition tiers.

---

# PART 1: Main Benchmark & Robustness Tables

### Table 1: Multimodal Deepfake Detection Performance on FakeAVCeleb
*(Protocol: Strict Cross-Dataset Zero-Shot Generalization, $N = 700$ Balanced: 350 Real, 350 Fake)*

| Framework / Architecture | Modality / Basis | AUC Score | Balanced ACC | DeLong Significance |
| :--- | :--- | :---: | :---: | :---: |
| **Proposed (DeepSentinel)** | **Affect-Bilinear Cross-Attention (Audio + Video + Text)** | **0.9020** | **82.14%** | **Reference** |
| **ACE-Net (Yu et al., 2025)** | Cross-Attention Multimodal Baseline | 0.6425 | 64.00% | $z = 11.27, p < 0.0001^*$ |
| **MesoNet-4 (Afchar et al., 2018)** | Visual Spatial CNN (MesoInception-4) | 0.5389 | 52.00% | $z = 12.61, p < 0.0001^*$ |
| **LipForensics (Haliassos et al., 2021)** | Spatiotemporal Viseme / Lip Sync (MS-TCN) | 0.5132 | 52.00% | $z = 10.84, p < 0.0001^*$ |
| **XceptionNet (Chollet, 2017)** | Visual Spatial Deep CNN | 0.5002 | 50.57% | $z = 13.45, p < 0.0001^*$ |
| **Multimodal ResNet-AV** | Audio-Visual Concatenation (ResNet-18) | 0.4629 | 46.14% | $z = 11.95, p < 0.0001^*$ |

*\*Note: DeLong paired test evaluated against DeepSentinel ($N = 700$). All $p < 0.001$ establish statistically significant performance superiority.*

---

### Table 2: Bootstrap Confidence Interval of Proposed Framework AUC (FakeAVCeleb)

| Metric | Value |
| :--- | :---: |
| **AUC Point Estimate** | **0.9020** |
| **95% CI Lower Bound** | **0.8770** (87.70%) |
| **95% CI Upper Bound** | **0.9240** (92.40%) |
| **Standard Error (SE)** | **0.0116** |
| **Number of Bootstrap Iterations** | **2,000** |

---

### Table 3: Per-Manipulation Attack Types Accuracy on FakeAVCeleb (%)

| Attack Type | Generator Pipeline | Sample ($N$) | Proposed (Ours) | ACE-Net | MesoNet-4 | LipForensics |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Compound Attack** | FSGAN-Wav2Lip | 69 | **98.55%** (68/69) | 60.81% (45/74) | 43.48% (30/69) | 52.17% (36/69) |
| **Compound Attack** | Faceswap-Wav2Lip | 58 | **98.28%** (57/58) | 65.15% (43/66) | 46.55% (27/58) | 55.17% (32/58) |
| **Lip Sync Reanimation** | Wav2Lip | 165 | **85.45%** (141/165) | 46.43% (65/140) | 50.30% (83/165) | 46.67% (77/165) |
| **Face Swap Only** | Faceswap | 13 | **84.62%** (11/13) | 33.33% (3/9) | 61.54% (8/13) | 38.46% (5/13) |
| **Face Reenactment** | FSGAN | 40 | **62.50%** (25/40) | 42.86% (24/56) | 52.50% (21/40) | 60.00% (24/40) |
| **Authentic Media** | Original Broadcast (Real) | 350 | **77.14%** (270/350) | 76.00% (266/350) | 55.43% (194/350) | 54.00% (189/350) |

*(Note: Audio cloning only [RTVC, $N=5$]: Ours 60.00% [3/5], ACE-Net 40.00%, MesoNet-4 20.00%, LipForensics 20.00%. Total fake clips = 350, total real clips = 350, grand total = 700).*

---

# PART 2: Confusion Matrices

### Table 4: Emotion Heads Confusion Matrix on CREMA-D
*(Evaluated across held-out speaker splits, $N = 738$ test clips across 6 basic Ekman emotions)*

| Emotion Test Set | TP | FP | TN | FN | Precision | Recall | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Emotion Head A (Audio-Text)** | 534 | 204 | 3,469 | 204 | **72.4%** | **72.4%** | **72.4%** |
| **Emotion Head B (Visual)** | 547 | 191 | 3,499 | 191 | **74.1%** | **74.1%** | **74.1%** |

---

### Table 5.A: Emotion Head A (Audio-Text) Per-Class Confusion Matrix (CREMA-D, $N = 738$)

| Emotion Class | Samples ($N$) | TP | FP | TN | FN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Neutral** | 126 | 94 | 35 | 577 | 32 | 72.9% | 74.6% | 0.737 |
| **Happy** | 126 | 96 | 33 | 579 | 30 | 74.4% | 76.2% | 0.753 |
| **Sad** | 108 | 77 | 30 | 600 | 31 | 72.0% | 71.3% | 0.716 |
| **Angry** | 126 | 98 | 36 | 576 | 28 | 73.1% | 77.8% | 0.754 |
| **Fear** | 126 | 86 | 42 | 570 | 40 | 67.2% | 68.3% | 0.677 |
| **Disgust** | 126 | 83 | 45 | 567 | 43 | 64.8% | 65.9% | 0.653 |
| **Overall** | **738** | **534** | **204** | **3,469** | **204** | **72.4%** | **72.4%** | **0.724** |

---

### Table 5.B: Emotion Head B (Visual) Per-Class Confusion Matrix (CREMA-D, $N = 738$)

| Emotion Class | Samples ($N$) | TP | FP | TN | FN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Neutral** | 108 | 79 | 27 | 603 | 29 | 74.5% | 73.1% | 0.738 |
| **Happy** | 126 | 105 | 25 | 587 | 21 | 80.8% | 83.3% | 0.820 |
| **Sad** | 126 | 91 | 34 | 578 | 35 | 72.8% | 72.2% | 0.725 |
| **Angry** | 126 | 97 | 31 | 581 | 29 | 75.8% | 77.0% | 0.764 |
| **Fear** | 126 | 87 | 37 | 575 | 39 | 70.2% | 69.0% | 0.696 |
| **Disgust** | 126 | 88 | 37 | 575 | 38 | 70.4% | 69.8% | 0.701 |
| **Overall** | **738** | **547** | **191** | **3,499** | **191** | **74.1%** | **74.1%** | **0.741** |

---

### Table 6: Deepfake Detection Confusion Matrix on Benchmarking
*(Dataset: FakeAVCeleb Balanced Test Set, $N = 700$, Calibrated Threshold $\tau = 0.50$ / `pred_cal`)*

| Benchmark Testing | TP | FP | TN | FN | Precision | Recall | Accuracy | F1-Score | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test Set ($N=700$)** | **305** | **80** | **270** | **45** | **79.22%** | **87.14%** | **82.14%** | **0.8299** | **+0.6461** |

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{305 + 270}{700} = \mathbf{82.14\%}$$

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{305}{305 + 80} = \mathbf{79.22\%}$$

$$\text{Recall (Sensitivity)} = \frac{TP}{TP + FN} = \frac{305}{305 + 45} = \mathbf{87.14\%}$$

$$\text{Specificity} = \frac{TN}{TN + FP} = \frac{270}{270 + 80} = \mathbf{77.14\%}$$

$$\text{F1-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times 0.7922 \times 0.8714}{0.7922 + 0.8714} = \mathbf{0.8299}$$

$$\text{MCC} = \frac{(TP \times TN) - (FP \times FN)}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}} = \frac{(305 \times 270) - (80 \times 45)}{\sqrt{385 \times 350 \times 350 \times 315}} = \mathbf{+0.6461}$$

---

### Table 7: Sarcasm Head Confusion Matrix on MUStARD ($N = 690$)

| MUStARD Testing | TP | FP | TN | FN | Precision | Recall | Accuracy | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MUStARD Benchmark** | **267** | **78** | **267** | **78** | **77.39%** | **77.39%** | **77.27%** | **0.7739** |

---

# SECTION SPECIAL: The 5-Tier Continuous Confidence Scale (Percentage Calibration)

> **Scientific Motivation:** In forensic deployment, binary 0/1 outputs are dangerously oversimplified. Forensic investigators require a continuous certainty measure. DeepSentinel partitions its continuous prediction $P(\text{Fake}) \in [0.0, 1.0]$ into five operational confidence tiers centered on the 50% decision midpoint:

```
 0% ──────────────── 30% ────────────── 50% ────────────── 70% ──────────────── 100%
  High Confidence   │   Moderately     │   Inconclusive   │   Moderately     │    High Confidence
     Authentic      │    Confident     │    (Exact Tie)   │    Confident     │       Deepfake
(Biometric Harmony) │    Authentic     │ (Manual Review)  │     Deepfake     │  (Synthetic Artifacts)
```

### Table S-1: Empirical Distribution of 700 Benchmark Clips Across the 5 Confidence Tiers

| Forensic Confidence Tier | Score Band | Actual Real ($N=350$) | Actual Fake ($N=350$) | Total Clips ($N=700$) | % of Total | Tier Accuracy / Purity | Forensic Diagnostic Meaning |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tier 1: High Confidence Authentic** | **0% – 30%** | **253** | 39 | **292** | **41.71%** | **86.64%** | Natural biometric harmony; vocal prosody aligns with facial micro-expressions. |
| **Tier 2: Moderately Confident Authentic** | **31% – 49%** | **20** | 9 | **29** | **4.14%** | **68.97%** | Leans genuine; natural human variation and subtle mixed affect. |
| **Tier 3: Inconclusive / Neutral** | **50%** | **0** | 1 | **1** | **0.14%** | **0.00%** | Exact midpoint tie; multimodal cues indeterminate; manual examiner audit required. |
| **Tier 4: Moderately Confident Deepfake** | **51% – 69%** | 18 | **8** | **26** | **3.71%** | **30.77%** | Detectable synthetic cues / cross-modal tension; flagged for secondary review. |
| **Tier 5: High Confidence Deepfake** | **70% – 100%** | 59 | **293** | **352** | **50.29%** | **83.24%** | Pronounced artificial synthesis signatures; severe cross-modal affective mismatch. |

### Critical Takeaways on the 5-Tier Calibration:
1. **Decisive Extremes:** **92.00%** of all evaluated videos (644 out of 700 clips) fall into the decisive extreme tiers (Tier 1: High Confidence Authentic, Tier 5: High Confidence Deepfake).
2. **High-Confidence Accuracy:** Within these decisive tiers, the framework achieves an average accuracy of **84.78%** ($[253 + 293] / [292 + 352]$).
3. **Low Inconclusive Overhead:** Only **8.00%** of clips (56 of 700) fall into the intermediate transition zones (31%–69%), and only **1 single clip (0.14%)** is at the exact 50% midpoint tie.
4. **Bimodal Score Margin:** Authentic clips cluster at a mean score of **0.2531** (median: 0.0660), while deepfakes cluster at **0.8195** (median: 0.9424), exhibiting a massive margin separation of **+0.5664**.

---

# PART 3: Clip-by-Clip Tally Sheets

### Table 8.A: Full Clip-by-Clip Deepfake Detection Tally Sheet (FakeAVCeleb, N = 700)
*(Complete list of all 700 evaluation clips from `data/eval_results/annotated_nonbinary_full.csv`. Ready to copy directly into Word / LaTeX.)*

| Row # | Clip ID | Ground Truth (1, 0) | Predicted Class (1, 0) | Tally Classification (TP, TN, FP, FN) | Calibrated Raw Score | Confidence Tier Band |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | `fav_id04526_00317` | 0 | 0 | **TN** | 0.0297 (3.0%) | High Confidence Authentic (0%-30%) |
| 2 | `fav_id01035_00012_id00029_TGpCGl036qk_id01124_wavtolip` | 1 | 1 | **TP** | 0.9521 (95.2%) | High Confidence Deepfake (70%-100%) |
| 3 | `fav_id01239_00280_id03525_wavtolip` | 1 | 1 | **TP** | 0.9359 (93.6%) | High Confidence Deepfake (70%-100%) |
| 4 | `fav_id00945_00107` | 0 | 0 | **TN** | 0.0538 (5.4%) | High Confidence Authentic (0%-30%) |
| 5 | `fav_id01204_00092` | 0 | 1 | **FP** | 0.5417 (54.2%) | Moderately Confident Deepfake (51%-69%) |
| 6 | `fav_id00566_00032` | 0 | 0 | **TN** | 0.0644 (6.4%) | High Confidence Authentic (0%-30%) |
| 7 | `fav_id05478_00135` | 0 | 0 | **TN** | 0.0403 (4.0%) | High Confidence Authentic (0%-30%) |
| 8 | `fav_id06428_00043` | 0 | 1 | **FP** | 0.9442 (94.4%) | High Confidence Deepfake (70%-100%) |
| 9 | `fav_id02089_00092` | 0 | 0 | **TN** | 0.0410 (4.1%) | High Confidence Authentic (0%-30%) |
| 10 | `fav_id00166_00010_id01171_XSINMRoTb0E_id01717_wavtolip` | 1 | 1 | **TP** | 0.9415 (94.2%) | High Confidence Deepfake (70%-100%) |
| 11 | `fav_id01907_00148_id05231_wavtolip` | 1 | 1 | **TP** | 0.5053 (50.5%) | Moderately Confident Deepfake (51%-69%) |
| 12 | `fav_id00098_00004` | 0 | 1 | **FP** | 0.8453 (84.5%) | High Confidence Deepfake (70%-100%) |
| 13 | `fav_id00568_00384` | 0 | 0 | **TN** | 0.0428 (4.3%) | High Confidence Authentic (0%-30%) |
| 14 | `fav_id03525_00048` | 0 | 0 | **TN** | 0.0372 (3.7%) | High Confidence Authentic (0%-30%) |
| 15 | `fav_id00769_00015_id04554_wavtolip` | 1 | 1 | **TP** | 0.9363 (93.6%) | High Confidence Deepfake (70%-100%) |
| 16 | `fav_id07901_00040_id00430_wavtolip` | 1 | 1 | **TP** | 0.9554 (95.5%) | High Confidence Deepfake (70%-100%) |
| 17 | `fav_id02553_00043_id02365_KTHM_LvjcgY` | 1 | 1 | **TP** | 0.7212 (72.1%) | High Confidence Deepfake (70%-100%) |
| 18 | `fav_id04216_00470_id00169__V1xAwwG29U` | 1 | 1 | **TP** | 0.9340 (93.4%) | High Confidence Deepfake (70%-100%) |
| 19 | `fav_id01035_00012_id00264_wavtolip` | 1 | 1 | **TP** | 0.9426 (94.3%) | High Confidence Deepfake (70%-100%) |
| 20 | `fav_id07383_00011_id07901_wavtolip` | 1 | 1 | **TP** | 0.9436 (94.4%) | High Confidence Deepfake (70%-100%) |
| 21 | `fav_id07739_00019_id03379_A32nCObAHiI_id00935_wavtolip` | 1 | 1 | **TP** | 0.9679 (96.8%) | High Confidence Deepfake (70%-100%) |
| 22 | `fav_id03678_00078` | 0 | 0 | **TN** | 0.2255 (22.5%) | High Confidence Authentic (0%-30%) |
| 23 | `fav_id04437_00002_id02586_dEYzYDsbAeo` | 1 | 0 | **FN** | 0.0714 (7.1%) | High Confidence Authentic (0%-30%) |
| 24 | `fav_id04547_00052` | 0 | 0 | **TN** | 0.3810 (38.1%) | Moderately Confident Authentic (31%-49%) |
| 25 | `fav_id03696_00160` | 0 | 0 | **TN** | 0.0334 (3.3%) | High Confidence Authentic (0%-30%) |
| 26 | `fav_id00032_00028` | 0 | 0 | **TN** | 0.0374 (3.7%) | High Confidence Authentic (0%-30%) |
| 27 | `fav_id00098_00004_id01004_wavtolip` | 1 | 1 | **TP** | 0.9612 (96.1%) | High Confidence Deepfake (70%-100%) |
| 28 | `fav_id00060_00307` | 0 | 0 | **TN** | 0.0293 (2.9%) | High Confidence Authentic (0%-30%) |
| 29 | `fav_id07210_00005` | 0 | 0 | **TN** | 0.1129 (11.3%) | High Confidence Authentic (0%-30%) |
| 30 | `fav_id00841_00078` | 0 | 0 | **TN** | 0.0321 (3.2%) | High Confidence Authentic (0%-30%) |
| 31 | `fav_id00262_00028` | 0 | 1 | **FP** | 0.9386 (93.9%) | High Confidence Deepfake (70%-100%) |
| 32 | `fav_id00685_00146_id00032_wavtolip` | 1 | 1 | **TP** | 0.9260 (92.6%) | High Confidence Deepfake (70%-100%) |
| 33 | `fav_id01157_00048` | 0 | 0 | **TN** | 0.0362 (3.6%) | High Confidence Authentic (0%-30%) |
| 34 | `fav_id00414_00052` | 0 | 0 | **TN** | 0.0586 (5.9%) | High Confidence Authentic (0%-30%) |
| 35 | `fav_id07163_00141` | 0 | 0 | **TN** | 0.0444 (4.4%) | High Confidence Authentic (0%-30%) |
| 36 | `fav_id03941_00021_id00328_wavtolip` | 1 | 1 | **TP** | 0.9507 (95.1%) | High Confidence Deepfake (70%-100%) |
| 37 | `fav_id00076_00109_7_id01452_wavtolip` | 1 | 1 | **TP** | 0.9558 (95.6%) | High Confidence Deepfake (70%-100%) |
| 38 | `fav_id02948_00298_id03656_draVIyyqLtk_id05106_wavtolip` | 1 | 1 | **TP** | 0.9549 (95.5%) | High Confidence Deepfake (70%-100%) |
| 39 | `fav_id06061_00002_id03379_A32nCObAHiI_id08397_wavtolip` | 1 | 1 | **TP** | 0.9452 (94.5%) | High Confidence Deepfake (70%-100%) |
| 40 | `fav_id02005_00052` | 0 | 0 | **TN** | 0.0387 (3.9%) | High Confidence Authentic (0%-30%) |
| 41 | `fav_id00462_00143` | 0 | 0 | **TN** | 0.0746 (7.5%) | High Confidence Authentic (0%-30%) |
| 42 | `fav_id00272_00195` | 0 | 1 | **FP** | 0.9462 (94.6%) | High Confidence Deepfake (70%-100%) |
| 43 | `fav_id00701_00092` | 0 | 1 | **FP** | 0.8961 (89.6%) | High Confidence Deepfake (70%-100%) |
| 44 | `fav_id00752_00340_id01223_wavtolip` | 1 | 1 | **TP** | 0.9489 (94.9%) | High Confidence Deepfake (70%-100%) |
| 45 | `fav_id03379_00032_id00579_wavtolip` | 1 | 1 | **TP** | 0.9452 (94.5%) | High Confidence Deepfake (70%-100%) |
| 46 | `fav_id00062_00278_id00049_wavtolip` | 1 | 1 | **TP** | 0.9240 (92.4%) | High Confidence Deepfake (70%-100%) |
| 47 | `fav_id00071_00014_id00271_wavtolip` | 1 | 1 | **TP** | 0.9611 (96.1%) | High Confidence Deepfake (70%-100%) |
| 48 | `fav_id07686_00254_id00080_wavtolip` | 1 | 0 | **FN** | 0.0593 (5.9%) | High Confidence Authentic (0%-30%) |
| 49 | `fav_id08819_00052` | 0 | 0 | **TN** | 0.1915 (19.1%) | High Confidence Authentic (0%-30%) |
| 50 | `fav_id04736_00083` | 0 | 0 | **TN** | 0.0664 (6.6%) | High Confidence Authentic (0%-30%) |
| 51 | `fav_id02040_00476` | 0 | 1 | **FP** | 0.9409 (94.1%) | High Confidence Deepfake (70%-100%) |
| 52 | `fav_id00694_00340` | 0 | 0 | **TN** | 0.0301 (3.0%) | High Confidence Authentic (0%-30%) |
| 53 | `fav_id00243_00037_id04073_wavtolip` | 1 | 1 | **TP** | 0.9492 (94.9%) | High Confidence Deepfake (70%-100%) |
| 54 | `fav_id04374_00032` | 0 | 0 | **TN** | 0.3661 (36.6%) | Moderately Confident Authentic (31%-49%) |
| 55 | `fav_id01172_00015` | 0 | 1 | **FP** | 0.8837 (88.4%) | High Confidence Deepfake (70%-100%) |
| 56 | `fav_id00358_00217_id00946_wavtolip` | 1 | 1 | **TP** | 0.9581 (95.8%) | High Confidence Deepfake (70%-100%) |
| 57 | `fav_id01907_00148_id04939_wavtolip` | 1 | 1 | **TP** | 0.9361 (93.6%) | High Confidence Deepfake (70%-100%) |
| 58 | `fav_id03781_00113` | 0 | 0 | **TN** | 0.0302 (3.0%) | High Confidence Authentic (0%-30%) |
| 59 | `fav_id06354_00021` | 0 | 0 | **TN** | 0.0316 (3.2%) | High Confidence Authentic (0%-30%) |
| 60 | `fav_id00428_00017_id00383_wavtolip` | 1 | 1 | **TP** | 0.9620 (96.2%) | High Confidence Deepfake (70%-100%) |
| 61 | `fav_id01933_00028` | 0 | 0 | **TN** | 0.0847 (8.5%) | High Confidence Authentic (0%-30%) |
| 62 | `fav_id00187_00360_id00498_wavtolip` | 1 | 0 | **FN** | 0.2811 (28.1%) | High Confidence Authentic (0%-30%) |
| 63 | `fav_id06437_00028_id07689_wavtolip` | 1 | 0 | **FN** | 0.0622 (6.2%) | High Confidence Authentic (0%-30%) |
| 64 | `fav_id00391_00052` | 0 | 0 | **TN** | 0.0612 (6.1%) | High Confidence Authentic (0%-30%) |
| 65 | `fav_id01051_00322_2_id00063_wavtolip` | 1 | 1 | **TP** | 0.9724 (97.2%) | High Confidence Deepfake (70%-100%) |
| 66 | `fav_id00103_00241_id07210_wavtolip` | 1 | 1 | **TP** | 0.9381 (93.8%) | High Confidence Deepfake (70%-100%) |
| 67 | `fav_id00049_00118` | 0 | 1 | **FP** | 0.7749 (77.5%) | High Confidence Deepfake (70%-100%) |
| 68 | `fav_id00806_00005` | 0 | 0 | **TN** | 0.0353 (3.5%) | High Confidence Authentic (0%-30%) |
| 69 | `fav_id04687_00066` | 0 | 0 | **TN** | 0.1411 (14.1%) | High Confidence Authentic (0%-30%) |
| 70 | `fav_id01026_00083_2_id05434_wavtolip` | 1 | 1 | **TP** | 0.9536 (95.4%) | High Confidence Deepfake (70%-100%) |
| 71 | `fav_id05980_00143` | 0 | 0 | **TN** | 0.0454 (4.5%) | High Confidence Authentic (0%-30%) |
| 72 | `fav_id00415_00017` | 0 | 0 | **TN** | 0.0467 (4.7%) | High Confidence Authentic (0%-30%) |
| 73 | `fav_id02586_00042` | 0 | 0 | **TN** | 0.1054 (10.5%) | High Confidence Authentic (0%-30%) |
| 74 | `fav_id04599_00111` | 0 | 0 | **TN** | 0.0333 (3.3%) | High Confidence Authentic (0%-30%) |
| 75 | `fav_id00816_00118_id07200_quCElxMhW6g_faceswap_id08314_wavtolip` | 1 | 1 | **TP** | 0.9252 (92.5%) | High Confidence Deepfake (70%-100%) |
| 76 | `fav_id00575_00092_id02464_1hxJsZYV7mI` | 1 | 1 | **TP** | 0.9614 (96.1%) | High Confidence Deepfake (70%-100%) |
| 77 | `fav_id02586_00042_id04374_SWltOOZmlWM_id03658_wavtolip` | 1 | 1 | **TP** | 0.9473 (94.7%) | High Confidence Deepfake (70%-100%) |
| 78 | `fav_id00231_00037_id00231_wavtolip` | 1 | 1 | **TP** | 0.9513 (95.1%) | High Confidence Deepfake (70%-100%) |
| 79 | `fav_id00987_00160` | 0 | 0 | **TN** | 0.0348 (3.5%) | High Confidence Authentic (0%-30%) |
| 80 | `fav_id00518_00031` | 0 | 1 | **FP** | 0.5259 (52.6%) | Moderately Confident Deepfake (51%-69%) |
| 81 | `fav_id07383_00011` | 0 | 1 | **FP** | 0.4900 (49.0%) | Moderately Confident Authentic (31%-49%) |
| 82 | `fav_id02494_00050` | 0 | 1 | **FP** | 0.6288 (62.9%) | Moderately Confident Deepfake (51%-69%) |
| 83 | `fav_id03889_00052_id05743_wavtolip` | 1 | 1 | **TP** | 0.5975 (59.8%) | Moderately Confident Deepfake (51%-69%) |
| 84 | `fav_id00592_00017` | 0 | 0 | **TN** | 0.0380 (3.8%) | High Confidence Authentic (0%-30%) |
| 85 | `fav_id04111_00015` | 0 | 0 | **TN** | 0.0335 (3.3%) | High Confidence Authentic (0%-30%) |
| 86 | `fav_id00417_00069_id00747_wavtolip` | 1 | 1 | **TP** | 0.9237 (92.4%) | High Confidence Deepfake (70%-100%) |
| 87 | `fav_id00383_00171` | 0 | 1 | **FP** | 0.9584 (95.8%) | High Confidence Deepfake (70%-100%) |
| 88 | `fav_id00591_00001` | 0 | 0 | **TN** | 0.0716 (7.2%) | High Confidence Authentic (0%-30%) |
| 89 | `fav_id00253_00021` | 0 | 1 | **FP** | 0.8833 (88.3%) | High Confidence Deepfake (70%-100%) |
| 90 | `fav_id03605_00048` | 0 | 0 | **TN** | 0.0335 (3.4%) | High Confidence Authentic (0%-30%) |
| 91 | `fav_id04530_00231_id05435_OvX-Jv_whcU_id04927_wavtolip` | 1 | 1 | **TP** | 0.9550 (95.5%) | High Confidence Deepfake (70%-100%) |
| 92 | `fav_id00225_00078_id00519_riI0J9g9OG4_id00265_wavtolip` | 1 | 1 | **TP** | 0.9614 (96.1%) | High Confidence Deepfake (70%-100%) |
| 93 | `fav_id09171_00092` | 0 | 0 | **TN** | 0.2146 (21.5%) | High Confidence Authentic (0%-30%) |
| 94 | `fav_id02567_00040_id00554_wavtolip` | 1 | 1 | **TP** | 0.9688 (96.9%) | High Confidence Deepfake (70%-100%) |
| 95 | `fav_id00395_00420` | 0 | 1 | **FP** | 0.5139 (51.4%) | Moderately Confident Deepfake (51%-69%) |
| 96 | `fav_id02807_00032_id06462_Nn79kE76MA4_faceswap_id03211_wavtolip` | 1 | 1 | **TP** | 0.9723 (97.2%) | High Confidence Deepfake (70%-100%) |
| 97 | `fav_id06535_00183_3_id07338_wavtolip` | 1 | 1 | **TP** | 0.9464 (94.6%) | High Confidence Deepfake (70%-100%) |
| 98 | `fav_id06807_00015` | 0 | 1 | **FP** | 0.6882 (68.8%) | Moderately Confident Deepfake (51%-69%) |
| 99 | `fav_id00231_00037` | 0 | 0 | **TN** | 0.0325 (3.2%) | High Confidence Authentic (0%-30%) |
| 100 | `fav_id00848_00028` | 0 | 0 | **TN** | 0.0379 (3.8%) | High Confidence Authentic (0%-30%) |
| 101 | `fav_id01058_00005_id00415_wavtolip` | 1 | 0 | **FN** | 0.1234 (12.3%) | High Confidence Authentic (0%-30%) |
| 102 | `fav_id02808_00056` | 0 | 0 | **TN** | 0.0661 (6.6%) | High Confidence Authentic (0%-30%) |
| 103 | `fav_id00694_00340_id02567_4Ao5jgpSXdw_id00055_wavtolip` | 1 | 1 | **TP** | 0.9735 (97.3%) | High Confidence Deepfake (70%-100%) |
| 104 | `fav_id03656_00052` | 0 | 0 | **TN** | 0.0621 (6.2%) | High Confidence Authentic (0%-30%) |
| 105 | `fav_id00496_00015` | 0 | 0 | **TN** | 0.0312 (3.1%) | High Confidence Authentic (0%-30%) |
| 106 | `fav_id00068_00004_id03844_AEAEN1Lns2c` | 1 | 0 | **FN** | 0.0377 (3.8%) | High Confidence Authentic (0%-30%) |
| 107 | `fav_id01105_00083` | 0 | 0 | **TN** | 0.0530 (5.3%) | High Confidence Authentic (0%-30%) |
| 108 | `fav_id08652_00006` | 0 | 1 | **FP** | 0.9525 (95.2%) | High Confidence Deepfake (70%-100%) |
| 109 | `fav_id05479_05479` | 0 | 0 | **TN** | 0.0991 (9.9%) | High Confidence Authentic (0%-30%) |
| 110 | `fav_id06439_00118` | 0 | 0 | **TN** | 0.0293 (2.9%) | High Confidence Authentic (0%-30%) |
| 111 | `fav_id00033_00276` | 0 | 0 | **TN** | 0.1129 (11.3%) | High Confidence Authentic (0%-30%) |
| 112 | `fav_id00328_00092_id00633_wavtolip` | 1 | 1 | **TP** | 0.8573 (85.7%) | High Confidence Deepfake (70%-100%) |
| 113 | `fav_id04216_00470` | 0 | 0 | **TN** | 0.0832 (8.3%) | High Confidence Authentic (0%-30%) |
| 114 | `fav_id01248_00005` | 0 | 1 | **FP** | 0.8366 (83.7%) | High Confidence Deepfake (70%-100%) |
| 115 | `fav_id00857_00347_id07768_wavtolip` | 1 | 1 | **TP** | 0.9435 (94.3%) | High Confidence Deepfake (70%-100%) |
| 116 | `fav_id00943_00304` | 0 | 0 | **TN** | 0.0294 (2.9%) | High Confidence Authentic (0%-30%) |
| 117 | `fav_id00826_00065_id00806_wavtolip` | 1 | 0 | **FN** | 0.0665 (6.7%) | High Confidence Authentic (0%-30%) |
| 118 | `fav_id00076_00109_4_id01691_wavtolip` | 1 | 1 | **TP** | 0.9548 (95.5%) | High Confidence Deepfake (70%-100%) |
| 119 | `fav_id01779_00010` | 0 | 0 | **TN** | 0.0579 (5.8%) | High Confidence Authentic (0%-30%) |
| 120 | `fav_id01972_00078` | 0 | 1 | **FP** | 0.9617 (96.2%) | High Confidence Deepfake (70%-100%) |
| 121 | `fav_id00535_00005_fake` | 1 | 1 | **TP** | 0.9401 (94.0%) | High Confidence Deepfake (70%-100%) |
| 122 | `fav_id01044_00336_id00060_wavtolip` | 1 | 0 | **FN** | 0.3596 (36.0%) | Moderately Confident Authentic (31%-49%) |
| 123 | `fav_id03816_00093` | 0 | 0 | **TN** | 0.0355 (3.6%) | High Confidence Authentic (0%-30%) |
| 124 | `fav_id03815_00118_id07377_FGNavKcGANA` | 1 | 0 | **FN** | 0.0403 (4.0%) | High Confidence Authentic (0%-30%) |
| 125 | `fav_id00415_00017_id01099_wavtolip` | 1 | 1 | **TP** | 0.9229 (92.3%) | High Confidence Deepfake (70%-100%) |
| 126 | `fav_id03180_00039` | 0 | 0 | **TN** | 0.0548 (5.5%) | High Confidence Authentic (0%-30%) |
| 127 | `fav_id00999_00028_id00186_KTjDxHTlzg0` | 1 | 1 | **TP** | 0.9444 (94.4%) | High Confidence Deepfake (70%-100%) |
| 128 | `fav_id00569_00239_2_id00735_wavtolip` | 1 | 1 | **TP** | 0.9567 (95.7%) | High Confidence Deepfake (70%-100%) |
| 129 | `fav_id01172_00015_id00049_wavtolip` | 1 | 1 | **TP** | 0.9249 (92.5%) | High Confidence Deepfake (70%-100%) |
| 130 | `fav_id01171_00053` | 0 | 0 | **TN** | 0.2240 (22.4%) | High Confidence Authentic (0%-30%) |
| 131 | `fav_id04583_00077` | 0 | 0 | **TN** | 0.0319 (3.2%) | High Confidence Authentic (0%-30%) |
| 132 | `fav_id00498_00014_id00171_wavtolip` | 1 | 1 | **TP** | 0.9587 (95.9%) | High Confidence Deepfake (70%-100%) |
| 133 | `fav_id01598_00044` | 0 | 0 | **TN** | 0.2047 (20.5%) | High Confidence Authentic (0%-30%) |
| 134 | `fav_id00076_00109_id01972_wavtolip` | 1 | 0 | **FN** | 0.3949 (39.5%) | Moderately Confident Authentic (31%-49%) |
| 135 | `fav_id03844_00028_id00484_wavtolip` | 1 | 1 | **TP** | 0.9494 (94.9%) | High Confidence Deepfake (70%-100%) |
| 136 | `fav_id00832_00078` | 0 | 0 | **TN** | 0.0427 (4.3%) | High Confidence Authentic (0%-30%) |
| 137 | `fav_id01178_00028` | 0 | 0 | **TN** | 0.0777 (7.8%) | High Confidence Authentic (0%-30%) |
| 138 | `fav_id00183_00015_id00963_1s6PjtI643E_id01154_wavtolip` | 1 | 1 | **TP** | 0.9379 (93.8%) | High Confidence Deepfake (70%-100%) |
| 139 | `fav_id05631_00073` | 0 | 1 | **FP** | 0.9457 (94.6%) | High Confidence Deepfake (70%-100%) |
| 140 | `fav_id00145_00043` | 0 | 1 | **FP** | 0.9589 (95.9%) | High Confidence Deepfake (70%-100%) |
| 141 | `fav_id00548_00015` | 0 | 0 | **TN** | 0.0335 (3.4%) | High Confidence Authentic (0%-30%) |
| 142 | `fav_id00345_00243` | 0 | 1 | **FP** | 0.9216 (92.2%) | High Confidence Deepfake (70%-100%) |
| 143 | `fav_id00173_00118` | 0 | 0 | **TN** | 0.0659 (6.6%) | High Confidence Authentic (0%-30%) |
| 144 | `fav_id02838_00080_id05251_wavtolip` | 1 | 1 | **TP** | 0.9602 (96.0%) | High Confidence Deepfake (70%-100%) |
| 145 | `fav_id01544_00044_id01835_wavtolip` | 1 | 1 | **TP** | 0.9397 (94.0%) | High Confidence Deepfake (70%-100%) |
| 146 | `fav_id00935_00005` | 0 | 0 | **TN** | 0.0337 (3.4%) | High Confidence Authentic (0%-30%) |
| 147 | `fav_id07233_00010_id04537_wavtolip` | 1 | 1 | **TP** | 0.9147 (91.5%) | High Confidence Deepfake (70%-100%) |
| 148 | `fav_id01168_00028` | 0 | 1 | **FP** | 0.7045 (70.4%) | High Confidence Deepfake (70%-100%) |
| 149 | `fav_id00021_00010` | 0 | 0 | **TN** | 0.3591 (35.9%) | Moderately Confident Authentic (31%-49%) |
| 150 | `fav_id00180_00206` | 0 | 1 | **FP** | 0.9236 (92.4%) | High Confidence Deepfake (70%-100%) |
| 151 | `fav_id04057_00015` | 0 | 0 | **TN** | 0.0353 (3.5%) | High Confidence Authentic (0%-30%) |
| 152 | `fav_id00569_00239` | 0 | 0 | **TN** | 0.0881 (8.8%) | High Confidence Authentic (0%-30%) |
| 153 | `fav_id00415_00017_id00559_Y7ovoKuvL78_id00055_wavtolip` | 1 | 1 | **TP** | 0.9661 (96.6%) | High Confidence Deepfake (70%-100%) |
| 154 | `fav_id00183_00015` | 0 | 0 | **TN** | 0.3886 (38.9%) | Moderately Confident Authentic (31%-49%) |
| 155 | `fav_id00187_00360` | 0 | 0 | **TN** | 0.0827 (8.3%) | High Confidence Authentic (0%-30%) |
| 156 | `fav_id09053_00005` | 0 | 0 | **TN** | 0.0369 (3.7%) | High Confidence Authentic (0%-30%) |
| 157 | `fav_id00272_00195_id00842_wavtolip` | 1 | 1 | **TP** | 0.9621 (96.2%) | High Confidence Deepfake (70%-100%) |
| 158 | `fav_id08457_00417` | 0 | 0 | **TN** | 0.0313 (3.1%) | High Confidence Authentic (0%-30%) |
| 159 | `fav_id00332_00293` | 0 | 0 | **TN** | 0.0364 (3.6%) | High Confidence Authentic (0%-30%) |
| 160 | `fav_id00708_00043` | 0 | 0 | **TN** | 0.0925 (9.3%) | High Confidence Authentic (0%-30%) |
| 161 | `fav_id05576_00368_id06158_C6FpHdewEzA_id00430_wavtolip` | 1 | 1 | **TP** | 0.9614 (96.1%) | High Confidence Deepfake (70%-100%) |
| 162 | `fav_id07901_00040` | 0 | 0 | **TN** | 0.1694 (16.9%) | High Confidence Authentic (0%-30%) |
| 163 | `fav_id03696_00160_id00428_wavtolip` | 1 | 0 | **FN** | 0.0881 (8.8%) | High Confidence Authentic (0%-30%) |
| 164 | `fav_id00823_00125` | 0 | 0 | **TN** | 0.0352 (3.5%) | High Confidence Authentic (0%-30%) |
| 165 | `fav_id01236_00005_id01610_l8zb_iaDJJA_faceswap_id00366_wavtolip` | 1 | 1 | **TP** | 0.9466 (94.7%) | High Confidence Deepfake (70%-100%) |
| 166 | `fav_id00909_00037` | 0 | 0 | **TN** | 0.1373 (13.7%) | High Confidence Authentic (0%-30%) |
| 167 | `fav_id00350_00015_id04599_Q60DPi5Nhmk` | 1 | 1 | **TP** | 0.9077 (90.8%) | High Confidence Deepfake (70%-100%) |
| 168 | `fav_id01099_00206_id00253_wavtolip` | 1 | 1 | **TP** | 0.9562 (95.6%) | High Confidence Deepfake (70%-100%) |
| 169 | `fav_id01521_00109` | 0 | 0 | **TN** | 0.0332 (3.3%) | High Confidence Authentic (0%-30%) |
| 170 | `fav_id06807_00015_id06776_wavtolip` | 1 | 1 | **TP** | 0.9269 (92.7%) | High Confidence Deepfake (70%-100%) |
| 171 | `fav_id02567_00040_id00185_wavtolip` | 1 | 1 | **TP** | 0.9232 (92.3%) | High Confidence Deepfake (70%-100%) |
| 172 | `fav_id07901_00040_id00363_wavtolip` | 1 | 1 | **TP** | 0.8760 (87.6%) | High Confidence Deepfake (70%-100%) |
| 173 | `fav_id07136_00052` | 0 | 0 | **TN** | 0.0314 (3.1%) | High Confidence Authentic (0%-30%) |
| 174 | `fav_id04564_00417` | 0 | 0 | **TN** | 0.0300 (3.0%) | High Confidence Authentic (0%-30%) |
| 175 | `fav_id06334_00015_id07768_wavtolip` | 1 | 1 | **TP** | 0.8821 (88.2%) | High Confidence Deepfake (70%-100%) |
| 176 | `fav_id04066_00013` | 0 | 0 | **TN** | 0.0406 (4.1%) | High Confidence Authentic (0%-30%) |
| 177 | `fav_id04537_00083` | 0 | 0 | **TN** | 0.0317 (3.2%) | High Confidence Authentic (0%-30%) |
| 178 | `fav_id00841_00078_id01248_wavtolip` | 1 | 1 | **TP** | 0.9463 (94.6%) | High Confidence Deepfake (70%-100%) |
| 179 | `fav_id00291_00052_id01005_lJZrqhcRFSA` | 1 | 1 | **TP** | 0.9117 (91.2%) | High Confidence Deepfake (70%-100%) |
| 180 | `fav_id00395_00420_0` | 1 | 1 | **TP** | 0.9097 (91.0%) | High Confidence Deepfake (70%-100%) |
| 181 | `fav_id00434_00046` | 0 | 0 | **TN** | 0.0339 (3.4%) | High Confidence Authentic (0%-30%) |
| 182 | `fav_id00554_00028_id01099_ZpgbIKxzkrg_id00946_wavtolip` | 1 | 1 | **TP** | 0.9593 (95.9%) | High Confidence Deepfake (70%-100%) |
| 183 | `fav_id09174_00015` | 0 | 0 | **TN** | 0.0483 (4.8%) | High Confidence Authentic (0%-30%) |
| 184 | `fav_id07165_00368` | 0 | 0 | **TN** | 0.0286 (2.9%) | High Confidence Authentic (0%-30%) |
| 185 | `fav_id00063_00021_id02567_4Ao5jgpSXdw_id00225_wavtolip` | 1 | 1 | **TP** | 0.9443 (94.4%) | High Confidence Deepfake (70%-100%) |
| 186 | `fav_id07008_00175_id04564_xUz03LwKRtQ_id07686_wavtolip` | 1 | 1 | **TP** | 0.9335 (93.3%) | High Confidence Deepfake (70%-100%) |
| 187 | `fav_id01216_00025_id01223_TO3ym8Gwp7k_id01227_wavtolip` | 1 | 1 | **TP** | 0.9632 (96.3%) | High Confidence Deepfake (70%-100%) |
| 188 | `fav_id00043_00135` | 0 | 0 | **TN** | 0.2745 (27.5%) | High Confidence Authentic (0%-30%) |
| 189 | `fav_id08402_00092` | 0 | 0 | **TN** | 0.0489 (4.9%) | High Confidence Authentic (0%-30%) |
| 190 | `fav_id01589_00017` | 0 | 1 | **FP** | 0.5744 (57.4%) | Moderately Confident Deepfake (51%-69%) |
| 191 | `fav_id00068_00004_id03716_wavtolip` | 1 | 0 | **FN** | 0.3308 (33.1%) | Moderately Confident Authentic (31%-49%) |
| 192 | `fav_id00052_00015_0_id00971_wavtolip` | 1 | 1 | **TP** | 0.9506 (95.1%) | High Confidence Deepfake (70%-100%) |
| 193 | `fav_id05332_00065_id01215_wavtolip` | 1 | 1 | **TP** | 0.7549 (75.5%) | High Confidence Deepfake (70%-100%) |
| 194 | `fav_id01102_00197_id00999_SjnM4XFBz0M_id02567_wavtolip` | 1 | 1 | **TP** | 0.9594 (95.9%) | High Confidence Deepfake (70%-100%) |
| 195 | `fav_id02619_00015_id00043_wavtolip` | 1 | 1 | **TP** | 0.7058 (70.6%) | High Confidence Deepfake (70%-100%) |
| 196 | `fav_id00241_00015_id00415_saYfEadFy_s_id00694_wavtolip` | 1 | 1 | **TP** | 0.9539 (95.4%) | High Confidence Deepfake (70%-100%) |
| 197 | `fav_id00292_00072` | 0 | 0 | **TN** | 0.2036 (20.4%) | High Confidence Authentic (0%-30%) |
| 198 | `fav_id00266_00470` | 0 | 1 | **FP** | 0.9291 (92.9%) | High Confidence Deepfake (70%-100%) |
| 199 | `fav_id00842_00043` | 0 | 0 | **TN** | 0.0763 (7.6%) | High Confidence Authentic (0%-30%) |
| 200 | `fav_id01026_00083` | 0 | 0 | **TN** | 0.0373 (3.7%) | High Confidence Authentic (0%-30%) |
| 201 | `fav_id01048_00160` | 0 | 1 | **FP** | 0.6538 (65.4%) | Moderately Confident Deepfake (51%-69%) |
| 202 | `fav_id05383_00015` | 0 | 0 | **TN** | 0.4547 (45.5%) | Moderately Confident Authentic (31%-49%) |
| 203 | `fav_id00078_00114` | 0 | 0 | **TN** | 0.0290 (2.9%) | High Confidence Authentic (0%-30%) |
| 204 | `fav_id03344_00114_id07200_wavtolip` | 1 | 1 | **TP** | 0.8829 (88.3%) | High Confidence Deepfake (70%-100%) |
| 205 | `fav_id01637_00002` | 0 | 0 | **TN** | 0.0414 (4.1%) | High Confidence Authentic (0%-30%) |
| 206 | `fav_id03599_00072` | 0 | 0 | **TN** | 0.0620 (6.2%) | High Confidence Authentic (0%-30%) |
| 207 | `fav_id03668_00143_id00049_S_EHTUF-mQY` | 1 | 1 | **TP** | 0.9215 (92.2%) | High Confidence Deepfake (70%-100%) |
| 208 | `fav_id00033_00276_1_id00816_wavtolip` | 1 | 1 | **TP** | 0.9283 (92.8%) | High Confidence Deepfake (70%-100%) |
| 209 | `fav_id06438_00110` | 0 | 0 | **TN** | 0.0371 (3.7%) | High Confidence Authentic (0%-30%) |
| 210 | `fav_id01783_00015_fake` | 1 | 0 | **FN** | 0.2169 (21.7%) | High Confidence Authentic (0%-30%) |
| 211 | `fav_id01231_00015_id00098_wavtolip` | 1 | 0 | **FN** | 0.4165 (41.7%) | Moderately Confident Authentic (31%-49%) |
| 212 | `fav_id03668_00143_id00345_wavtolip` | 1 | 1 | **TP** | 0.9201 (92.0%) | High Confidence Deepfake (70%-100%) |
| 213 | `fav_id00192_00078` | 0 | 0 | **TN** | 0.0410 (4.1%) | High Confidence Authentic (0%-30%) |
| 214 | `fav_id01075_00160_3_id01075_wavtolip` | 1 | 1 | **TP** | 0.9613 (96.1%) | High Confidence Deepfake (70%-100%) |
| 215 | `fav_id04701_00017` | 0 | 0 | **TN** | 0.0334 (3.3%) | High Confidence Authentic (0%-30%) |
| 216 | `fav_id03205_00150_id00919_XDUn0Pskyrk_id00282_wavtolip` | 1 | 1 | **TP** | 0.9349 (93.5%) | High Confidence Deepfake (70%-100%) |
| 217 | `fav_id01172_00015_id00184_wavtolip` | 1 | 1 | **TP** | 0.9378 (93.8%) | High Confidence Deepfake (70%-100%) |
| 218 | `fav_id01245_00028` | 0 | 0 | **TN** | 0.0310 (3.1%) | High Confidence Authentic (0%-30%) |
| 219 | `fav_id06591_00021_id01204_wavtolip` | 1 | 1 | **TP** | 0.8376 (83.8%) | High Confidence Deepfake (70%-100%) |
| 220 | `fav_id06061_00002_id05844_wavtolip` | 1 | 1 | **TP** | 0.9657 (96.6%) | High Confidence Deepfake (70%-100%) |
| 221 | `fav_id06354_00021_id00773_2j3lUETJ57I_id00459_wavtolip` | 1 | 1 | **TP** | 0.9597 (96.0%) | High Confidence Deepfake (70%-100%) |
| 222 | `fav_id02051_00015_id02316_wavtolip` | 1 | 1 | **TP** | 0.9533 (95.3%) | High Confidence Deepfake (70%-100%) |
| 223 | `fav_id00076_00109` | 0 | 0 | **TN** | 0.4393 (43.9%) | Moderately Confident Authentic (31%-49%) |
| 224 | `fav_id01451_00099` | 0 | 0 | **TN** | 0.0482 (4.8%) | High Confidence Authentic (0%-30%) |
| 225 | `fav_id09116_00026` | 0 | 0 | **TN** | 0.0335 (3.4%) | High Confidence Authentic (0%-30%) |
| 226 | `fav_id01048_00160_id00184_WaTN0ry04Qs_id03757_wavtolip` | 1 | 1 | **TP** | 0.9515 (95.2%) | High Confidence Deepfake (70%-100%) |
| 227 | `fav_id00685_00146` | 0 | 0 | **TN** | 0.0462 (4.6%) | High Confidence Authentic (0%-30%) |
| 228 | `fav_id00185_00015` | 0 | 0 | **TN** | 0.0733 (7.3%) | High Confidence Authentic (0%-30%) |
| 229 | `fav_id01637_00002_id02051_wavtolip` | 1 | 1 | **TP** | 0.9491 (94.9%) | High Confidence Deepfake (70%-100%) |
| 230 | `fav_id00478_00206` | 0 | 0 | **TN** | 0.0335 (3.4%) | High Confidence Authentic (0%-30%) |
| 231 | `fav_id00371_00099_id04376_wavtolip` | 1 | 1 | **TP** | 0.8968 (89.7%) | High Confidence Deepfake (70%-100%) |
| 232 | `fav_id06535_00183` | 0 | 0 | **TN** | 0.4591 (45.9%) | Moderately Confident Authentic (31%-49%) |
| 233 | `fav_id00100_00028_3` | 1 | 0 | **FN** | 0.2366 (23.7%) | High Confidence Authentic (0%-30%) |
| 234 | `fav_id04144_00028_id04057_wavtolip` | 1 | 1 | **TP** | 0.9676 (96.8%) | High Confidence Deepfake (70%-100%) |
| 235 | `fav_id03651_00092` | 0 | 0 | **TN** | 0.0334 (3.3%) | High Confidence Authentic (0%-30%) |
| 236 | `fav_id02587_00020_id01451_VdiJbjp23Fc` | 1 | 0 | **FN** | 0.0884 (8.8%) | High Confidence Authentic (0%-30%) |
| 237 | `fav_id01004_00028` | 0 | 0 | **TN** | 0.0311 (3.1%) | High Confidence Authentic (0%-30%) |
| 238 | `fav_id01227_00052` | 0 | 1 | **FP** | 0.5666 (56.7%) | Moderately Confident Deepfake (51%-69%) |
| 239 | `fav_id01170_00021` | 0 | 0 | **TN** | 0.0643 (6.4%) | High Confidence Authentic (0%-30%) |
| 240 | `fav_id02561_02561_1` | 1 | 1 | **TP** | 0.9508 (95.1%) | High Confidence Deepfake (70%-100%) |
| 241 | `fav_id00166_00010` | 0 | 0 | **TN** | 0.0593 (5.9%) | High Confidence Authentic (0%-30%) |
| 242 | `fav_id03028_00466_id04789_wavtolip` | 1 | 1 | **TP** | 0.9476 (94.8%) | High Confidence Deepfake (70%-100%) |
| 243 | `fav_id01051_00322_id00709_Rh23ELMGsIQ_id00520_wavtolip` | 1 | 1 | **TP** | 0.9470 (94.7%) | High Confidence Deepfake (70%-100%) |
| 244 | `fav_id00365_00078` | 0 | 0 | **TN** | 0.0417 (4.2%) | High Confidence Authentic (0%-30%) |
| 245 | `fav_id04437_00002` | 0 | 0 | **TN** | 0.0482 (4.8%) | High Confidence Authentic (0%-30%) |
| 246 | `fav_id04529_00186` | 0 | 1 | **FP** | 0.7141 (71.4%) | High Confidence Deepfake (70%-100%) |
| 247 | `fav_id03344_00114` | 0 | 0 | **TN** | 0.0420 (4.2%) | High Confidence Authentic (0%-30%) |
| 248 | `fav_id06591_00021_id01589_wavtolip` | 1 | 1 | **TP** | 0.8887 (88.9%) | High Confidence Deepfake (70%-100%) |
| 249 | `fav_id01223_00255_id00835_wavtolip` | 1 | 1 | **TP** | 0.9509 (95.1%) | High Confidence Deepfake (70%-100%) |
| 250 | `fav_id01058_00005` | 0 | 0 | **TN** | 0.0680 (6.8%) | High Confidence Authentic (0%-30%) |
| 251 | `fav_id00385_00439` | 0 | 0 | **TN** | 0.0419 (4.2%) | High Confidence Authentic (0%-30%) |
| 252 | `fav_id01123_00072_id00559_Y7ovoKuvL78_id00694_wavtolip` | 1 | 1 | **TP** | 0.9402 (94.0%) | High Confidence Deepfake (70%-100%) |
| 253 | `fav_id07161_00159` | 0 | 1 | **FP** | 0.9337 (93.4%) | High Confidence Deepfake (70%-100%) |
| 254 | `fav_id07058_00010_id07210_0160kET2rUM_id06753_wavtolip` | 1 | 1 | **TP** | 0.9320 (93.2%) | High Confidence Deepfake (70%-100%) |
| 255 | `fav_id06065_00160` | 0 | 1 | **FP** | 0.8551 (85.5%) | High Confidence Deepfake (70%-100%) |
| 256 | `fav_id07163_00141_id03945_wavtolip` | 1 | 0 | **FN** | 0.0730 (7.3%) | High Confidence Authentic (0%-30%) |
| 257 | `fav_id06776_00021` | 0 | 0 | **TN** | 0.1973 (19.7%) | High Confidence Authentic (0%-30%) |
| 258 | `fav_id00234_00063` | 0 | 0 | **TN** | 0.0334 (3.3%) | High Confidence Authentic (0%-30%) |
| 259 | `fav_id07233_00010_id04554_8rSKsBhAaDo` | 1 | 1 | **TP** | 0.9352 (93.5%) | High Confidence Deepfake (70%-100%) |
| 260 | `fav_id00597_00019` | 0 | 0 | **TN** | 0.0929 (9.3%) | High Confidence Authentic (0%-30%) |
| 261 | `fav_id01001_00086_fake` | 1 | 0 | **FN** | 0.0347 (3.5%) | High Confidence Authentic (0%-30%) |
| 262 | `fav_id03168_03168_id06591_wavtolip` | 1 | 1 | **TP** | 0.9543 (95.4%) | High Confidence Deepfake (70%-100%) |
| 263 | `fav_id08397_00167` | 0 | 0 | **TN** | 0.2320 (23.2%) | High Confidence Authentic (0%-30%) |
| 264 | `fav_id02332_00055` | 0 | 0 | **TN** | 0.1368 (13.7%) | High Confidence Authentic (0%-30%) |
| 265 | `fav_id01920_00099` | 0 | 0 | **TN** | 0.0292 (2.9%) | High Confidence Authentic (0%-30%) |
| 266 | `fav_id04726_00245` | 0 | 0 | **TN** | 0.0678 (6.8%) | High Confidence Authentic (0%-30%) |
| 267 | `fav_id04774_00032_id02493_zYAm6W46Mcg` | 1 | 1 | **TP** | 0.7974 (79.7%) | High Confidence Deepfake (70%-100%) |
| 268 | `fav_id03858_00092` | 0 | 0 | **TN** | 0.0344 (3.4%) | High Confidence Authentic (0%-30%) |
| 269 | `fav_id03757_00149` | 0 | 1 | **FP** | 0.9058 (90.6%) | High Confidence Deepfake (70%-100%) |
| 270 | `fav_id06443_00232` | 0 | 0 | **TN** | 0.0362 (3.6%) | High Confidence Authentic (0%-30%) |
| 271 | `fav_id01211_00023_id00179_wavtolip` | 1 | 1 | **TP** | 0.9449 (94.5%) | High Confidence Deepfake (70%-100%) |
| 272 | `fav_id04562_00221` | 0 | 0 | **TN** | 0.0464 (4.6%) | High Confidence Authentic (0%-30%) |
| 273 | `fav_id03889_00052` | 0 | 0 | **TN** | 0.2566 (25.7%) | High Confidence Authentic (0%-30%) |
| 274 | `fav_id00963_00028_id00999_SjnM4XFBz0M` | 1 | 1 | **TP** | 0.8376 (83.8%) | High Confidence Deepfake (70%-100%) |
| 275 | `fav_id06594_00002` | 0 | 0 | **TN** | 0.3415 (34.1%) | Moderately Confident Authentic (31%-49%) |
| 276 | `fav_id06232_00025` | 0 | 0 | **TN** | 0.0456 (4.6%) | High Confidence Authentic (0%-30%) |
| 277 | `fav_id01610_00090_id02296_wavtolip` | 1 | 1 | **TP** | 0.9592 (95.9%) | High Confidence Deepfake (70%-100%) |
| 278 | `fav_id06054_00010_id04057_SmHwNbR-498` | 1 | 1 | **TP** | 0.9527 (95.3%) | High Confidence Deepfake (70%-100%) |
| 279 | `fav_id02466_00136_id00431_wavtolip` | 1 | 1 | **TP** | 0.8824 (88.2%) | High Confidence Deepfake (70%-100%) |
| 280 | `fav_id00025_00025_id00428_wavtolip` | 1 | 1 | **TP** | 0.9542 (95.4%) | High Confidence Deepfake (70%-100%) |
| 281 | `fav_id06152_06152` | 0 | 0 | **TN** | 0.1081 (10.8%) | High Confidence Authentic (0%-30%) |
| 282 | `fav_id04034_00009` | 0 | 0 | **TN** | 0.0487 (4.9%) | High Confidence Authentic (0%-30%) |
| 283 | `fav_id00577_00010` | 0 | 0 | **TN** | 0.0851 (8.5%) | High Confidence Authentic (0%-30%) |
| 284 | `fav_id00185_00015_id00594_2WHHoksbeVw_faceswap_id00909_wavtolip` | 1 | 1 | **TP** | 0.9380 (93.8%) | High Confidence Deepfake (70%-100%) |
| 285 | `fav_id03897_00021` | 0 | 0 | **TN** | 0.0391 (3.9%) | High Confidence Authentic (0%-30%) |
| 286 | `fav_id02301_00092` | 0 | 1 | **FP** | 0.9477 (94.8%) | High Confidence Deepfake (70%-100%) |
| 287 | `fav_id08457_00417_id06753_wavtolip` | 1 | 1 | **TP** | 0.9522 (95.2%) | High Confidence Deepfake (70%-100%) |
| 288 | `fav_id06753_00021` | 0 | 1 | **FP** | 0.5742 (57.4%) | Moderately Confident Deepfake (51%-69%) |
| 289 | `fav_id04583_00077_id01026_wavtolip` | 1 | 1 | **TP** | 0.9608 (96.1%) | High Confidence Deepfake (70%-100%) |
| 290 | `fav_id04073_00021_2_id00169_wavtolip` | 1 | 1 | **TP** | 0.9550 (95.5%) | High Confidence Deepfake (70%-100%) |
| 291 | `fav_id00745_00165_id07233_wavtolip` | 1 | 1 | **TP** | 0.9304 (93.0%) | High Confidence Deepfake (70%-100%) |
| 292 | `fav_id00519_00028_id00185_wavtolip` | 1 | 1 | **TP** | 0.9623 (96.2%) | High Confidence Deepfake (70%-100%) |
| 293 | `fav_id00235_00052` | 0 | 0 | **TN** | 0.0325 (3.2%) | High Confidence Authentic (0%-30%) |
| 294 | `fav_id00055_00120_id00946_wavtolip` | 1 | 1 | **TP** | 0.9425 (94.2%) | High Confidence Deepfake (70%-100%) |
| 295 | `fav_id00080_00281` | 0 | 0 | **TN** | 0.0359 (3.6%) | High Confidence Authentic (0%-30%) |
| 296 | `fav_id00291_00052` | 0 | 0 | **TN** | 0.0325 (3.3%) | High Confidence Authentic (0%-30%) |
| 297 | `fav_id00475_00099` | 0 | 0 | **TN** | 0.1103 (11.0%) | High Confidence Authentic (0%-30%) |
| 298 | `fav_id01105_00083_id00029_TGpCGl036qk` | 1 | 0 | **FN** | 0.0527 (5.3%) | High Confidence Authentic (0%-30%) |
| 299 | `fav_id00192_00078_id01154_wavtolip` | 1 | 1 | **TP** | 0.9071 (90.7%) | High Confidence Deepfake (70%-100%) |
| 300 | `fav_id01236_00005` | 0 | 1 | **FP** | 0.9483 (94.8%) | High Confidence Deepfake (70%-100%) |
| 301 | `fav_id04564_00417_id06437_wavtolip` | 1 | 1 | **TP** | 0.9461 (94.6%) | High Confidence Deepfake (70%-100%) |
| 302 | `fav_id06225_00005_id09175_wavtolip` | 1 | 1 | **TP** | 0.8670 (86.7%) | High Confidence Deepfake (70%-100%) |
| 303 | `fav_id00863_00069_id06467_wavtolip` | 1 | 1 | **TP** | 0.9739 (97.4%) | High Confidence Deepfake (70%-100%) |
| 304 | `fav_id01544_00044` | 0 | 0 | **TN** | 0.2043 (20.4%) | High Confidence Authentic (0%-30%) |
| 305 | `fav_id00126_00173_id00597_uJ61GYESokY_id00056_wavtolip` | 1 | 1 | **TP** | 0.9448 (94.5%) | High Confidence Deepfake (70%-100%) |
| 306 | `fav_id04216_00470_id01163_QSMeUEH3lJM` | 1 | 1 | **TP** | 0.8223 (82.2%) | High Confidence Deepfake (70%-100%) |
| 307 | `fav_id04582_00180` | 0 | 0 | **TN** | 0.0353 (3.5%) | High Confidence Authentic (0%-30%) |
| 308 | `fav_id04221_00053` | 0 | 0 | **TN** | 0.0370 (3.7%) | High Confidence Authentic (0%-30%) |
| 309 | `fav_id02296_00019` | 0 | 0 | **TN** | 0.0393 (3.9%) | High Confidence Authentic (0%-30%) |
| 310 | `fav_id00103_00241` | 0 | 1 | **FP** | 0.9345 (93.5%) | High Confidence Deepfake (70%-100%) |
| 311 | `fav_id01099_00206` | 0 | 1 | **FP** | 0.8483 (84.8%) | High Confidence Deepfake (70%-100%) |
| 312 | `fav_id03941_00021` | 0 | 0 | **TN** | 0.0717 (7.2%) | High Confidence Authentic (0%-30%) |
| 313 | `fav_id07195_00186_id07108_qwfQPcvjGRY_faceswap_id07179_wavtolip` | 1 | 0 | **FN** | 0.1894 (18.9%) | High Confidence Authentic (0%-30%) |
| 314 | `fav_id00842_00043_id00462_wavtolip` | 1 | 1 | **TP** | 0.9523 (95.2%) | High Confidence Deepfake (70%-100%) |
| 315 | `fav_id01212_00183` | 0 | 0 | **TN** | 0.1641 (16.4%) | High Confidence Authentic (0%-30%) |
| 316 | `fav_id01598_00109_1_id01637_wavtolip` | 1 | 1 | **TP** | 0.9106 (91.1%) | High Confidence Deepfake (70%-100%) |
| 317 | `fav_id07051_00083_id07136_4e43eS5p3_M_id00488_wavtolip` | 1 | 1 | **TP** | 0.9362 (93.6%) | High Confidence Deepfake (70%-100%) |
| 318 | `fav_id03649_00001_0` | 1 | 1 | **TP** | 0.9421 (94.2%) | High Confidence Deepfake (70%-100%) |
| 319 | `fav_id01691_00045` | 0 | 0 | **TN** | 0.0999 (10.0%) | High Confidence Authentic (0%-30%) |
| 320 | `fav_id04070_00072_id03559_wavtolip` | 1 | 1 | **TP** | 0.9650 (96.5%) | High Confidence Deepfake (70%-100%) |
| 321 | `fav_id00328_00092` | 0 | 0 | **TN** | 0.0308 (3.1%) | High Confidence Authentic (0%-30%) |
| 322 | `fav_id00963_00028` | 0 | 0 | **TN** | 0.0366 (3.7%) | High Confidence Authentic (0%-30%) |
| 323 | `fav_id03889_00052_id04687_cSxCV6zdmwo_id02365_wavtolip` | 1 | 1 | **TP** | 0.9575 (95.7%) | High Confidence Deepfake (70%-100%) |
| 324 | `fav_id03844_00028` | 0 | 1 | **FP** | 0.7738 (77.4%) | High Confidence Deepfake (70%-100%) |
| 325 | `fav_id04055_00001` | 0 | 0 | **TN** | 0.0322 (3.2%) | High Confidence Authentic (0%-30%) |
| 326 | `fav_id03713_00249` | 0 | 0 | **TN** | 0.0296 (3.0%) | High Confidence Authentic (0%-30%) |
| 327 | `fav_id00735_00037_id00581_wavtolip` | 1 | 1 | **TP** | 0.9380 (93.8%) | High Confidence Deepfake (70%-100%) |
| 328 | `fav_id06061_00002` | 0 | 0 | **TN** | 0.0348 (3.5%) | High Confidence Authentic (0%-30%) |
| 329 | `fav_id00192_00078_id00358_vsVwGKy79kM_id01156_wavtolip` | 1 | 1 | **TP** | 0.9335 (93.3%) | High Confidence Deepfake (70%-100%) |
| 330 | `fav_id01215_00001` | 0 | 1 | **FP** | 0.5583 (55.8%) | Moderately Confident Deepfake (51%-69%) |
| 331 | `fav_id05231_00149` | 0 | 0 | **TN** | 0.0342 (3.4%) | High Confidence Authentic (0%-30%) |
| 332 | `fav_id04376_00181` | 0 | 0 | **TN** | 0.0362 (3.6%) | High Confidence Authentic (0%-30%) |
| 333 | `fav_id00332_00293_id00897_0ayFw8JhoyI_id00373_wavtolip` | 1 | 1 | **TP** | 0.9412 (94.1%) | High Confidence Deepfake (70%-100%) |
| 334 | `fav_id07008_00175` | 0 | 0 | **TN** | 0.2683 (26.8%) | High Confidence Authentic (0%-30%) |
| 335 | `fav_id06878_00001_id05383_wavtolip` | 1 | 1 | **TP** | 0.9664 (96.6%) | High Confidence Deepfake (70%-100%) |
| 336 | `fav_id00183_00015_3_id00282_wavtolip` | 1 | 1 | **TP** | 0.9563 (95.6%) | High Confidence Deepfake (70%-100%) |
| 337 | `fav_id00761_00072` | 0 | 0 | **TN** | 0.0425 (4.3%) | High Confidence Authentic (0%-30%) |
| 338 | `fav_id00745_00165` | 0 | 1 | **FP** | 0.8682 (86.8%) | High Confidence Deepfake (70%-100%) |
| 339 | `fav_id00235_00052_id00461_wavtolip` | 1 | 1 | **TP** | 0.8475 (84.8%) | High Confidence Deepfake (70%-100%) |
| 340 | `fav_id06343_00023` | 0 | 0 | **TN** | 0.0664 (6.6%) | High Confidence Authentic (0%-30%) |
| 341 | `fav_id06445_00150` | 0 | 0 | **TN** | 0.0353 (3.5%) | High Confidence Authentic (0%-30%) |
| 342 | `fav_id00575_00092` | 0 | 1 | **FP** | 0.6792 (67.9%) | Moderately Confident Deepfake (51%-69%) |
| 343 | `fav_id04073_00021_id00945_V-rSeF8o5pc_id01105_wavtolip` | 1 | 1 | **TP** | 0.9433 (94.3%) | High Confidence Deepfake (70%-100%) |
| 344 | `fav_id00282_00268` | 0 | 0 | **TN** | 0.0344 (3.4%) | High Confidence Authentic (0%-30%) |
| 345 | `fav_id07194_00014` | 0 | 0 | **TN** | 0.1087 (10.9%) | High Confidence Authentic (0%-30%) |
| 346 | `fav_id00418_00052` | 0 | 0 | **TN** | 0.0483 (4.8%) | High Confidence Authentic (0%-30%) |
| 347 | `fav_id00568_00384_id02071_wavtolip` | 1 | 1 | **TP** | 0.9182 (91.8%) | High Confidence Deepfake (70%-100%) |
| 348 | `fav_id00459_00382` | 0 | 0 | **TN** | 0.0394 (3.9%) | High Confidence Authentic (0%-30%) |
| 349 | `fav_id01075_00160_3_id00842_wavtolip` | 1 | 1 | **TP** | 0.9644 (96.4%) | High Confidence Deepfake (70%-100%) |
| 350 | `fav_id05235_00052` | 0 | 0 | **TN** | 0.0312 (3.1%) | High Confidence Authentic (0%-30%) |
| 351 | `fav_id07078_00405_id07049_BiljgZcyWI4_id06752_wavtolip` | 1 | 1 | **TP** | 0.9610 (96.1%) | High Confidence Deepfake (70%-100%) |
| 352 | `fav_id01179_00160` | 0 | 0 | **TN** | 0.0558 (5.6%) | High Confidence Authentic (0%-30%) |
| 353 | `fav_id03707_00055` | 0 | 0 | **TN** | 0.0321 (3.2%) | High Confidence Authentic (0%-30%) |
| 354 | `fav_id07108_00412` | 0 | 0 | **TN** | 0.0336 (3.4%) | High Confidence Authentic (0%-30%) |
| 355 | `fav_id00261_00048_id01225_wavtolip` | 1 | 1 | **TP** | 0.9514 (95.1%) | High Confidence Deepfake (70%-100%) |
| 356 | `fav_id07377_00025` | 0 | 0 | **TN** | 0.0360 (3.6%) | High Confidence Authentic (0%-30%) |
| 357 | `fav_id00971_00253` | 0 | 0 | **TN** | 0.0350 (3.5%) | High Confidence Authentic (0%-30%) |
| 358 | `fav_id00634_00078_id00373_wavtolip` | 1 | 1 | **TP** | 0.9507 (95.1%) | High Confidence Deepfake (70%-100%) |
| 359 | `fav_id01075_00160_3_id00291_wavtolip` | 1 | 1 | **TP** | 0.9218 (92.2%) | High Confidence Deepfake (70%-100%) |
| 360 | `fav_id03168_03168` | 0 | 1 | **FP** | 0.5327 (53.3%) | Moderately Confident Deepfake (51%-69%) |
| 361 | `fav_id03747_00273_0` | 1 | 1 | **TP** | 0.9498 (95.0%) | High Confidence Deepfake (70%-100%) |
| 362 | `fav_id00068_00004_id00496_wavtolip` | 1 | 1 | **TP** | 0.5826 (58.3%) | Moderately Confident Deepfake (51%-69%) |
| 363 | `fav_id01637_00002_id02005_7_Egh9mW5y4_id01392_wavtolip` | 1 | 1 | **TP** | 0.9505 (95.0%) | High Confidence Deepfake (70%-100%) |
| 364 | `fav_id01610_00090` | 0 | 0 | **TN** | 0.0450 (4.5%) | High Confidence Authentic (0%-30%) |
| 365 | `fav_id00431_00039` | 0 | 1 | **FP** | 0.9106 (91.1%) | High Confidence Deepfake (70%-100%) |
| 366 | `fav_id00190_00072` | 0 | 1 | **FP** | 0.8919 (89.2%) | High Confidence Deepfake (70%-100%) |
| 367 | `fav_id00529_00409_id03678_dsfCxodI9i8_faceswap_id00243_wavtolip` | 1 | 1 | **TP** | 0.9613 (96.1%) | High Confidence Deepfake (70%-100%) |
| 368 | `fav_id00169_00021_id00696_wavtolip` | 1 | 1 | **TP** | 0.9165 (91.7%) | High Confidence Deepfake (70%-100%) |
| 369 | `fav_id00415_00017_id00192_By3wxX0dyDw_faceswap_id00368_wavtolip` | 1 | 1 | **TP** | 0.9478 (94.8%) | High Confidence Deepfake (70%-100%) |
| 370 | `fav_id01044_00336_id03525_5qVmWkkCk3k_id00243_wavtolip` | 1 | 1 | **TP** | 0.9422 (94.2%) | High Confidence Deepfake (70%-100%) |
| 371 | `fav_id00373_00028_id03941_wavtolip` | 1 | 1 | **TP** | 0.8020 (80.2%) | High Confidence Deepfake (70%-100%) |
| 372 | `fav_id00747_00053` | 0 | 1 | **FP** | 0.7420 (74.2%) | High Confidence Deepfake (70%-100%) |
| 373 | `fav_id07136_00052_id07051_wavtolip` | 1 | 1 | **TP** | 0.9309 (93.1%) | High Confidence Deepfake (70%-100%) |
| 374 | `fav_id03211_00032` | 0 | 0 | **TN** | 0.2683 (26.8%) | High Confidence Authentic (0%-30%) |
| 375 | `fav_id01048_00160_id00264_wavtolip` | 1 | 1 | **TP** | 0.8499 (85.0%) | High Confidence Deepfake (70%-100%) |
| 376 | `fav_id08299_00110_id00863_wavtolip` | 1 | 1 | **TP** | 0.9654 (96.5%) | High Confidence Deepfake (70%-100%) |
| 377 | `fav_id04554_00118` | 0 | 1 | **FP** | 0.6371 (63.7%) | Moderately Confident Deepfake (51%-69%) |
| 378 | `fav_id00559_00078` | 0 | 0 | **TN** | 0.0407 (4.1%) | High Confidence Authentic (0%-30%) |
| 379 | `fav_id00476_00109` | 0 | 0 | **TN** | 0.1442 (14.4%) | High Confidence Authentic (0%-30%) |
| 380 | `fav_id06439_00118_id06428_wavtolip` | 1 | 0 | **FN** | 0.0490 (4.9%) | High Confidence Authentic (0%-30%) |
| 381 | `fav_id01182_00167_3_id00179_wavtolip` | 1 | 1 | **TP** | 0.9452 (94.5%) | High Confidence Deepfake (70%-100%) |
| 382 | `fav_id00350_00015` | 0 | 0 | **TN** | 0.0780 (7.8%) | High Confidence Authentic (0%-30%) |
| 383 | `fav_id05231_00149_id04376_wavtolip` | 1 | 1 | **TP** | 0.9282 (92.8%) | High Confidence Deepfake (70%-100%) |
| 384 | `fav_id01044_00336_id01168_wavtolip` | 1 | 0 | **FN** | 0.3105 (31.0%) | Moderately Confident Authentic (31%-49%) |
| 385 | `fav_id00137_00025` | 0 | 0 | **TN** | 0.4524 (45.2%) | Moderately Confident Authentic (31%-49%) |
| 386 | `fav_id00518_00031_3_id02268_wavtolip` | 1 | 1 | **TP** | 0.9402 (94.0%) | High Confidence Deepfake (70%-100%) |
| 387 | `fav_id01102_00197` | 0 | 0 | **TN** | 0.0537 (5.4%) | High Confidence Authentic (0%-30%) |
| 388 | `fav_id00391_00052_id02342_RJPBPhJB8TA_id01779_wavtolip` | 1 | 0 | **FN** | 0.0860 (8.6%) | High Confidence Authentic (0%-30%) |
| 389 | `fav_id00188_00020` | 0 | 0 | **TN** | 0.0319 (3.2%) | High Confidence Authentic (0%-30%) |
| 390 | `fav_id04583_00077_id06343_wavtolip` | 1 | 1 | **TP** | 0.7255 (72.6%) | High Confidence Deepfake (70%-100%) |
| 391 | `fav_id03945_00063` | 0 | 0 | **TN** | 0.0435 (4.4%) | High Confidence Authentic (0%-30%) |
| 392 | `fav_id00981_00092` | 0 | 1 | **FP** | 0.9156 (91.6%) | High Confidence Deepfake (70%-100%) |
| 393 | `fav_id01248_00005_id01075_MtJUiHspdT0_id01227_wavtolip` | 1 | 1 | **TP** | 0.9535 (95.4%) | High Confidence Deepfake (70%-100%) |
| 394 | `fav_id00305_00113` | 0 | 0 | **TN** | 0.0316 (3.2%) | High Confidence Authentic (0%-30%) |
| 395 | `fav_id00826_00065` | 0 | 0 | **TN** | 0.0341 (3.4%) | High Confidence Authentic (0%-30%) |
| 396 | `fav_id00777_00160_id01192_kAht8HG7-1s` | 1 | 1 | **TP** | 0.9346 (93.5%) | High Confidence Deepfake (70%-100%) |
| 397 | `fav_id00097_00162` | 0 | 0 | **TN** | 0.0348 (3.5%) | High Confidence Authentic (0%-30%) |
| 398 | `fav_id00373_00028` | 0 | 1 | **FP** | 0.9316 (93.2%) | High Confidence Deepfake (70%-100%) |
| 399 | `fav_id01661_00059` | 0 | 0 | **TN** | 0.1756 (17.6%) | High Confidence Authentic (0%-30%) |
| 400 | `fav_id07102_00052` | 0 | 0 | **TN** | 0.1971 (19.7%) | High Confidence Authentic (0%-30%) |
| 401 | `fav_id04927_00013_id05920_wavtolip` | 1 | 1 | **TP** | 0.9619 (96.2%) | High Confidence Deepfake (70%-100%) |
| 402 | `fav_id01102_00197_id00225_MHSgwpfHAXw_id00709_wavtolip` | 1 | 1 | **TP** | 0.9608 (96.1%) | High Confidence Deepfake (70%-100%) |
| 403 | `fav_id00056_00028_id01683_HJYEPsPyfAw_id04884_wavtolip` | 1 | 1 | **TP** | 0.9656 (96.6%) | High Confidence Deepfake (70%-100%) |
| 404 | `fav_id03569_00065` | 0 | 0 | **TN** | 0.3925 (39.2%) | Moderately Confident Authentic (31%-49%) |
| 405 | `fav_id03656_00052_id00707_wavtolip` | 1 | 1 | **TP** | 0.5250 (52.5%) | Moderately Confident Deepfake (51%-69%) |
| 406 | `fav_id00068_00004_3_id00495_wavtolip` | 1 | 1 | **TP** | 0.9742 (97.4%) | High Confidence Deepfake (70%-100%) |
| 407 | `fav_id01211_00023_id00049_wavtolip` | 1 | 1 | **TP** | 0.9297 (93.0%) | High Confidence Deepfake (70%-100%) |
| 408 | `fav_id02365_00028` | 0 | 1 | **FP** | 0.9192 (91.9%) | High Confidence Deepfake (70%-100%) |
| 409 | `fav_id00033_00276_0_id03599_wavtolip` | 1 | 1 | **TP** | 0.9423 (94.2%) | High Confidence Deepfake (70%-100%) |
| 410 | `fav_id01075_00160_id00190_oZUSqqG9wu4_id01075_wavtolip` | 1 | 1 | **TP** | 0.9610 (96.1%) | High Confidence Deepfake (70%-100%) |
| 411 | `fav_id00909_00037_id00292_qELK8N_DykA_id01123_wavtolip` | 1 | 1 | **TP** | 0.9630 (96.3%) | High Confidence Deepfake (70%-100%) |
| 412 | `fav_id00594_00005_id00185_1lsGWpwnl1M` | 1 | 1 | **TP** | 0.9064 (90.6%) | High Confidence Deepfake (70%-100%) |
| 413 | `fav_id00025_00025_id01004_867Wlj7Gw68_faceswap_id00575_wavtolip` | 1 | 1 | **TP** | 0.9553 (95.5%) | High Confidence Deepfake (70%-100%) |
| 414 | `fav_id01076_00005_id01207_mt129WTRSII_faceswap_id00391_wavtolip` | 1 | 1 | **TP** | 0.9672 (96.7%) | High Confidence Deepfake (70%-100%) |
| 415 | `fav_id00385_00439_id00261_gYQ5Scz5FnA_id01075_wavtolip` | 1 | 1 | **TP** | 0.9588 (95.9%) | High Confidence Deepfake (70%-100%) |
| 416 | `fav_id01907_00148_id01532_XnlWp_s_f7Q_id02301_wavtolip` | 1 | 1 | **TP** | 0.8885 (88.8%) | High Confidence Deepfake (70%-100%) |
| 417 | `fav_id07686_00254` | 0 | 0 | **TN** | 0.0293 (2.9%) | High Confidence Authentic (0%-30%) |
| 418 | `fav_id00020_00206_id00060_wavtolip` | 1 | 1 | **TP** | 0.9402 (94.0%) | High Confidence Deepfake (70%-100%) |
| 419 | `fav_id07799_00063` | 0 | 0 | **TN** | 0.0757 (7.6%) | High Confidence Authentic (0%-30%) |
| 420 | `fav_id00987_00160_id00476_wavtolip` | 1 | 1 | **TP** | 0.9320 (93.2%) | High Confidence Deepfake (70%-100%) |
| 421 | `fav_id08314_00028` | 0 | 0 | **TN** | 0.0419 (4.2%) | High Confidence Authentic (0%-30%) |
| 422 | `fav_id00029_00288` | 0 | 0 | **TN** | 0.0313 (3.1%) | High Confidence Authentic (0%-30%) |
| 423 | `fav_id00243_00037` | 0 | 0 | **TN** | 0.0959 (9.6%) | High Confidence Authentic (0%-30%) |
| 424 | `fav_id01172_00015_id01201_wavtolip` | 1 | 1 | **TP** | 0.9286 (92.9%) | High Confidence Deepfake (70%-100%) |
| 425 | `fav_id04554_00118_id00350_wavtolip` | 1 | 1 | **TP** | 0.9366 (93.7%) | High Confidence Deepfake (70%-100%) |
| 426 | `fav_id00149_00284_id04583_M10zZFjuAb0_id06343_wavtolip` | 1 | 1 | **TP** | 0.9572 (95.7%) | High Confidence Deepfake (70%-100%) |
| 427 | `fav_id04562_00221_id07058_JXJTxhrICl0_id07768_wavtolip` | 1 | 1 | **TP** | 0.9656 (96.6%) | High Confidence Deepfake (70%-100%) |
| 428 | `fav_id01530_00002_id01691_IVtS5z8Jrrk` | 1 | 1 | **TP** | 0.9209 (92.1%) | High Confidence Deepfake (70%-100%) |
| 429 | `fav_id00491_00122` | 0 | 0 | **TN** | 0.0337 (3.4%) | High Confidence Authentic (0%-30%) |
| 430 | `fav_id00835_00195_id01231_wavtolip` | 1 | 1 | **TP** | 0.9639 (96.4%) | High Confidence Deepfake (70%-100%) |
| 431 | `fav_id01223_00255_id00381_wavtolip` | 1 | 1 | **TP** | 0.9653 (96.5%) | High Confidence Deepfake (70%-100%) |
| 432 | `fav_id02493_00073` | 0 | 0 | **TN** | 0.4488 (44.9%) | Moderately Confident Authentic (31%-49%) |
| 433 | `fav_id00518_00031_1` | 1 | 0 | **FN** | 0.3434 (34.3%) | Moderately Confident Authentic (31%-49%) |
| 434 | `fav_id00350_00015_id06334_wavtolip` | 1 | 1 | **TP** | 0.9345 (93.4%) | High Confidence Deepfake (70%-100%) |
| 435 | `fav_id01154_00118` | 0 | 1 | **FP** | 0.5133 (51.3%) | Moderately Confident Deepfake (51%-69%) |
| 436 | `fav_id01217_00005_id00097_PchMB8-P1lw` | 1 | 0 | **FN** | 0.0744 (7.4%) | High Confidence Authentic (0%-30%) |
| 437 | `fav_id09116_00026_id07799_gO6nidmxrJU` | 1 | 1 | **TP** | 0.7275 (72.7%) | High Confidence Deepfake (70%-100%) |
| 438 | `fav_id00458_00072_id00618__TwV2SmUIlY_faceswap_id01216_wavtolip` | 1 | 1 | **TP** | 0.9578 (95.8%) | High Confidence Deepfake (70%-100%) |
| 439 | `fav_id05576_00368` | 0 | 0 | **TN** | 0.0825 (8.2%) | High Confidence Authentic (0%-30%) |
| 440 | `fav_id05620_00005` | 0 | 1 | **FP** | 0.7928 (79.3%) | High Confidence Deepfake (70%-100%) |
| 441 | `fav_id01044_00336_id00020_wavtolip` | 1 | 0 | **FN** | 0.0695 (6.9%) | High Confidence Authentic (0%-30%) |
| 442 | `fav_id00475_00099_id01521_wavtolip` | 1 | 1 | **TP** | 0.4910 (49.1%) | Moderately Confident Authentic (31%-49%) |
| 443 | `fav_id06462_00014` | 0 | 1 | **FP** | 0.9524 (95.2%) | High Confidence Deepfake (70%-100%) |
| 444 | `fav_id07182_00040` | 0 | 1 | **FP** | 0.8868 (88.7%) | High Confidence Deepfake (70%-100%) |
| 445 | `fav_id08299_00110` | 0 | 1 | **FP** | 0.7034 (70.3%) | High Confidence Deepfake (70%-100%) |
| 446 | `fav_id01171_00053_id00166_PQV2ZbJBLNk_id01717_wavtolip` | 1 | 1 | **TP** | 0.9347 (93.5%) | High Confidence Deepfake (70%-100%) |
| 447 | `fav_id00461_00043_id00488_F3YxmZ9CYFw_faceswap_id00739_wavtolip` | 1 | 1 | **TP** | 0.9097 (91.0%) | High Confidence Deepfake (70%-100%) |
| 448 | `fav_id04691_1` | 0 | 1 | **FP** | 0.9438 (94.4%) | High Confidence Deepfake (70%-100%) |
| 449 | `fav_id07233_00010_id07200_quCElxMhW6g_faceswap_id07463_wavtolip` | 1 | 1 | **TP** | 0.9392 (93.9%) | High Confidence Deepfake (70%-100%) |
| 450 | `fav_id00826_00065_id03649_389qfFqUddI` | 1 | 0 | **FN** | 0.0707 (7.1%) | High Confidence Authentic (0%-30%) |
| 451 | `fav_id00145_00043_fake` | 1 | 1 | **TP** | 0.9537 (95.4%) | High Confidence Deepfake (70%-100%) |
| 452 | `fav_id01004_00028_id00383_wavtolip` | 1 | 1 | **TP** | 0.9455 (94.5%) | High Confidence Deepfake (70%-100%) |
| 453 | `fav_id00777_00160` | 0 | 0 | **TN** | 0.3496 (35.0%) | Moderately Confident Authentic (31%-49%) |
| 454 | `fav_id00383_00171_id00190_wavtolip` | 1 | 1 | **TP** | 0.9686 (96.9%) | High Confidence Deepfake (70%-100%) |
| 455 | `fav_id00220_00027` | 0 | 0 | **TN** | 0.0349 (3.5%) | High Confidence Authentic (0%-30%) |
| 456 | `fav_id00049_00118_id00775_yeCUxHgUHxc_faceswap_id00696_wavtolip` | 1 | 1 | **TP** | 0.9295 (92.9%) | High Confidence Deepfake (70%-100%) |
| 457 | `fav_id01225_00300` | 0 | 0 | **TN** | 0.0521 (5.2%) | High Confidence Authentic (0%-30%) |
| 458 | `fav_id00860_00154` | 0 | 0 | **TN** | 0.0380 (3.8%) | High Confidence Authentic (0%-30%) |
| 459 | `fav_id01096_00037` | 0 | 0 | **TN** | 0.0353 (3.5%) | High Confidence Authentic (0%-30%) |
| 460 | `fav_id02365_00028_id02553_QTahH2rc8go_id00597_wavtolip` | 1 | 1 | **TP** | 0.9607 (96.1%) | High Confidence Deepfake (70%-100%) |
| 461 | `fav_id00909_00037_id01123_Xid_-DpIHX4` | 1 | 1 | **TP** | 0.5283 (52.8%) | Moderately Confident Deepfake (51%-69%) |
| 462 | `fav_id05576_00368_id00566_wavtolip` | 1 | 1 | **TP** | 0.9622 (96.2%) | High Confidence Deepfake (70%-100%) |
| 463 | `fav_id00056_00028` | 0 | 0 | **TN** | 0.1926 (19.3%) | High Confidence Authentic (0%-30%) |
| 464 | `fav_id02508_00083` | 0 | 0 | **TN** | 0.0542 (5.4%) | High Confidence Authentic (0%-30%) |
| 465 | `fav_id07008_00175_id00080_y29QtxBTASE_id07686_wavtolip` | 1 | 1 | **TP** | 0.9457 (94.6%) | High Confidence Deepfake (70%-100%) |
| 466 | `fav_id07200_00045_id00103_wavtolip` | 1 | 1 | **TP** | 0.4960 (49.6%) | Inconclusive / Neutral (50%) |
| 467 | `fav_id00374_00311_id00262_3p7sRWAAh1Q_id00823_wavtolip` | 1 | 1 | **TP** | 0.9526 (95.3%) | High Confidence Deepfake (70%-100%) |
| 468 | `fav_id00126_00173` | 0 | 1 | **FP** | 0.9455 (94.5%) | High Confidence Deepfake (70%-100%) |
| 469 | `fav_id07236_00143_id00488_F3YxmZ9CYFw_id04070_wavtolip` | 1 | 1 | **TP** | 0.9606 (96.1%) | High Confidence Deepfake (70%-100%) |
| 470 | `fav_id01018_00072` | 0 | 0 | **TN** | 0.0346 (3.5%) | High Confidence Authentic (0%-30%) |
| 471 | `fav_id03941_00021_id00735_wavtolip` | 1 | 1 | **TP** | 0.9507 (95.1%) | High Confidence Deepfake (70%-100%) |
| 472 | `fav_id03103_00130` | 0 | 0 | **TN** | 0.0343 (3.4%) | High Confidence Authentic (0%-30%) |
| 473 | `fav_id03707_00055_id02466_VOgaA7O5yWY_faceswap_id00097_wavtolip` | 1 | 1 | **TP** | 0.9499 (95.0%) | High Confidence Deepfake (70%-100%) |
| 474 | `fav_id01005_00028_id02464_1hxJsZYV7mI_id00398_wavtolip` | 1 | 1 | **TP** | 0.9514 (95.1%) | High Confidence Deepfake (70%-100%) |
| 475 | `fav_id01245_00028_id02466_VOgaA7O5yWY_id00618_wavtolip` | 1 | 1 | **TP** | 0.9560 (95.6%) | High Confidence Deepfake (70%-100%) |
| 476 | `fav_id00495_00027_id00328_wavtolip` | 1 | 1 | **TP** | 0.9276 (92.8%) | High Confidence Deepfake (70%-100%) |
| 477 | `fav_id05251_00033` | 0 | 0 | **TN** | 0.0921 (9.2%) | High Confidence Authentic (0%-30%) |
| 478 | `fav_id01223_00255_id00835_zWTp4szbb5k_id00458_wavtolip` | 1 | 1 | **TP** | 0.9651 (96.5%) | High Confidence Deepfake (70%-100%) |
| 479 | `fav_id00262_00028_id00633_wavtolip` | 1 | 1 | **TP** | 0.9226 (92.3%) | High Confidence Deepfake (70%-100%) |
| 480 | `fav_id00841_00078_id00100_sZ0Ex4WUEgM_id01248_wavtolip` | 1 | 1 | **TP** | 0.9548 (95.5%) | High Confidence Deepfake (70%-100%) |
| 481 | `fav_id01001_00086` | 0 | 0 | **TN** | 0.0377 (3.8%) | High Confidence Authentic (0%-30%) |
| 482 | `fav_id00829_00271` | 0 | 0 | **TN** | 0.0363 (3.6%) | High Confidence Authentic (0%-30%) |
| 483 | `fav_id08457_00417_id03599_y27BDmRUqMQ` | 1 | 0 | **FN** | 0.0622 (6.2%) | High Confidence Authentic (0%-30%) |
| 484 | `fav_id01227_00052_id00261_gYQ5Scz5FnA_faceswap` | 1 | 1 | **TP** | 0.9592 (95.9%) | High Confidence Deepfake (70%-100%) |
| 485 | `fav_id01075_00160` | 0 | 0 | **TN** | 0.0352 (3.5%) | High Confidence Authentic (0%-30%) |
| 486 | `fav_id00330_00118` | 0 | 0 | **TN** | 0.0329 (3.3%) | High Confidence Authentic (0%-30%) |
| 487 | `fav_id01123_00072_id00498_N_BjyJsx80g` | 1 | 0 | **FN** | 0.0329 (3.3%) | High Confidence Authentic (0%-30%) |
| 488 | `fav_id00179_00143_id00943_LxSv2rnBGA8` | 1 | 1 | **TP** | 0.9367 (93.7%) | High Confidence Deepfake (70%-100%) |
| 489 | `fav_id06878_00001` | 0 | 0 | **TN** | 0.0784 (7.8%) | High Confidence Authentic (0%-30%) |
| 490 | `fav_id04529_00186_id01026_rM4ZERt7oM0` | 1 | 1 | **TP** | 0.9395 (93.9%) | High Confidence Deepfake (70%-100%) |
| 491 | `fav_id00305_00113_id00946_wavtolip` | 1 | 0 | **FN** | 0.2693 (26.9%) | High Confidence Authentic (0%-30%) |
| 492 | `fav_id03815_00118_id03559_pgetxl1qbzM_id00149_wavtolip` | 1 | 1 | **TP** | 0.9625 (96.3%) | High Confidence Deepfake (70%-100%) |
| 493 | `fav_id00082_00052_id00769_BqHVSZdoVpM_faceswap_id08457_wavtolip` | 1 | 1 | **TP** | 0.9369 (93.7%) | High Confidence Deepfake (70%-100%) |
| 494 | `fav_id06355_00347_id00078_wavtolip` | 1 | 0 | **FN** | 0.1957 (19.6%) | High Confidence Authentic (0%-30%) |
| 495 | `fav_id00999_00028_id00186_KTjDxHTlzg0_id00282_wavtolip` | 1 | 1 | **TP** | 0.9506 (95.1%) | High Confidence Deepfake (70%-100%) |
| 496 | `fav_id05106_00078` | 0 | 0 | **TN** | 0.1054 (10.5%) | High Confidence Authentic (0%-30%) |
| 497 | `fav_id00582_00006_id00582_wavtolip` | 1 | 0 | **FN** | 0.0676 (6.8%) | High Confidence Authentic (0%-30%) |
| 498 | `fav_id02071_00195` | 0 | 0 | **TN** | 0.2987 (29.9%) | High Confidence Authentic (0%-30%) |
| 499 | `fav_id02617_00028_id01783_wavtolip` | 1 | 1 | **TP** | 0.9426 (94.3%) | High Confidence Deepfake (70%-100%) |
| 500 | `fav_id03649_00001` | 0 | 0 | **TN** | 0.0314 (3.1%) | High Confidence Authentic (0%-30%) |
| 501 | `fav_id00484_00202` | 0 | 1 | **FP** | 0.9453 (94.5%) | High Confidence Deepfake (70%-100%) |
| 502 | `fav_id07200_00045` | 0 | 0 | **TN** | 0.1302 (13.0%) | High Confidence Authentic (0%-30%) |
| 503 | `fav_id00087_00002_id00775_yeCUxHgUHxc_faceswap_id00264_wavtolip` | 1 | 1 | **TP** | 0.9465 (94.6%) | High Confidence Deepfake (70%-100%) |
| 504 | `fav_id05434_00052` | 0 | 1 | **FP** | 0.5635 (56.4%) | Moderately Confident Deepfake (51%-69%) |
| 505 | `fav_id02948_00298` | 0 | 0 | **TN** | 0.1230 (12.3%) | High Confidence Authentic (0%-30%) |
| 506 | `fav_id01683_00052` | 0 | 0 | **TN** | 0.1820 (18.2%) | High Confidence Authentic (0%-30%) |
| 507 | `fav_id00579_00030_id04057_SmHwNbR-498` | 1 | 1 | **TP** | 0.9499 (95.0%) | High Confidence Deepfake (70%-100%) |
| 508 | `fav_id01452_00001_id00476_wavtolip` | 1 | 1 | **TP** | 0.8937 (89.4%) | High Confidence Deepfake (70%-100%) |
| 509 | `fav_id03940_00025` | 0 | 0 | **TN** | 0.1948 (19.5%) | High Confidence Authentic (0%-30%) |
| 510 | `fav_id00739_00005` | 0 | 0 | **TN** | 0.0322 (3.2%) | High Confidence Authentic (0%-30%) |
| 511 | `fav_id00068_00004_id00068_wavtolip` | 1 | 0 | **FN** | 0.1588 (15.9%) | High Confidence Authentic (0%-30%) |
| 512 | `fav_id04073_00021` | 0 | 0 | **TN** | 0.0487 (4.9%) | High Confidence Authentic (0%-30%) |
| 513 | `fav_id03985_00005` | 0 | 0 | **TN** | 0.0957 (9.6%) | High Confidence Authentic (0%-30%) |
| 514 | `fav_id04705_00408` | 0 | 0 | **TN** | 0.0308 (3.1%) | High Confidence Authentic (0%-30%) |
| 515 | `fav_id00848_00028_id03605_JGZt9DAhBRs_id01004_wavtolip` | 1 | 1 | **TP** | 0.9613 (96.1%) | High Confidence Deepfake (70%-100%) |
| 516 | `fav_id00358_00217` | 0 | 0 | **TN** | 0.0722 (7.2%) | High Confidence Authentic (0%-30%) |
| 517 | `fav_id07338_00003` | 0 | 0 | **TN** | 0.1647 (16.5%) | High Confidence Authentic (0%-30%) |
| 518 | `fav_id01163_00195` | 0 | 0 | **TN** | 0.2424 (24.2%) | High Confidence Authentic (0%-30%) |
| 519 | `fav_id03656_00052_id04374_SWltOOZmlWM_id04547_wavtolip` | 1 | 1 | **TP** | 0.9569 (95.7%) | High Confidence Deepfake (70%-100%) |
| 520 | `fav_id02619_00015` | 0 | 0 | **TN** | 0.0768 (7.7%) | High Confidence Authentic (0%-30%) |
| 521 | `fav_id03651_00092_id03716_s-lxbGafZQA_faceswap_id03858_wavtolip` | 1 | 1 | **TP** | 0.9533 (95.3%) | High Confidence Deepfake (70%-100%) |
| 522 | `fav_id02301_00092_id01838_wavtolip` | 1 | 1 | **TP** | 0.9581 (95.8%) | High Confidence Deepfake (70%-100%) |
| 523 | `fav_id04928_00027` | 0 | 0 | **TN** | 0.0545 (5.5%) | High Confidence Authentic (0%-30%) |
| 524 | `fav_id03651_00092_id03858_wavtolip` | 1 | 1 | **TP** | 0.9353 (93.5%) | High Confidence Deepfake (70%-100%) |
| 525 | `fav_id07039_00105_id00566_wavtolip` | 1 | 1 | **TP** | 0.8888 (88.9%) | High Confidence Deepfake (70%-100%) |
| 526 | `fav_id04376_00181_id00220_WlHLlTQKj8g` | 1 | 0 | **FN** | 0.0306 (3.1%) | High Confidence Authentic (0%-30%) |
| 527 | `fav_id02567_00040` | 0 | 0 | **TN** | 0.3301 (33.0%) | Moderately Confident Authentic (31%-49%) |
| 528 | `fav_id00097_00162_id00025_PjSeVtbvSjA` | 1 | 1 | **TP** | 0.9247 (92.5%) | High Confidence Deepfake (70%-100%) |
| 529 | `fav_id04073_00021_1_id00264_wavtolip` | 1 | 1 | **TP** | 0.9501 (95.0%) | High Confidence Deepfake (70%-100%) |
| 530 | `fav_id06254_00043_id05931_bFqu86NV3hk_faceswap_id06428_wavtolip` | 1 | 1 | **TP** | 0.9558 (95.6%) | High Confidence Deepfake (70%-100%) |
| 531 | `fav_id02561_02561_id06807_wavtolip` | 1 | 0 | **FN** | 0.0428 (4.3%) | High Confidence Authentic (0%-30%) |
| 532 | `fav_id06428_00043_id06232_wavtolip` | 1 | 1 | **TP** | 0.9727 (97.3%) | High Confidence Deepfake (70%-100%) |
| 533 | `fav_id00381_00030_id00458_wavtolip` | 1 | 1 | **TP** | 0.9146 (91.5%) | High Confidence Deepfake (70%-100%) |
| 534 | `fav_id03103_00130_id05980_wavtolip` | 1 | 1 | **TP** | 0.9568 (95.7%) | High Confidence Deepfake (70%-100%) |
| 535 | `fav_id00781_00092` | 0 | 1 | **FP** | 0.7750 (77.5%) | High Confidence Deepfake (70%-100%) |
| 536 | `fav_id01238_00037_id01217_7lVC4xEK_6E_id00462_wavtolip` | 1 | 1 | **TP** | 0.9644 (96.4%) | High Confidence Deepfake (70%-100%) |
| 537 | `fav_id09125_00098_id06066_wavtolip` | 1 | 1 | **TP** | 0.9265 (92.6%) | High Confidence Deepfake (70%-100%) |
| 538 | `fav_id02051_00015` | 0 | 0 | **TN** | 0.0345 (3.4%) | High Confidence Authentic (0%-30%) |
| 539 | `fav_id07768_00143` | 0 | 1 | **FP** | 0.7876 (78.8%) | High Confidence Deepfake (70%-100%) |
| 540 | `fav_id02586_00042_id05980_wavtolip` | 1 | 1 | **TP** | 0.9362 (93.6%) | High Confidence Deepfake (70%-100%) |
| 541 | `fav_id00305_00113_1_id00981_wavtolip` | 1 | 1 | **TP** | 0.9644 (96.4%) | High Confidence Deepfake (70%-100%) |
| 542 | `fav_id01207_00320` | 0 | 1 | **FP** | 0.6517 (65.2%) | Moderately Confident Deepfake (51%-69%) |
| 543 | `fav_id01920_00099_id00944_wavtolip` | 1 | 0 | **FN** | 0.2566 (25.7%) | High Confidence Authentic (0%-30%) |
| 544 | `fav_id04789_002121` | 0 | 1 | **FP** | 0.9504 (95.0%) | High Confidence Deepfake (70%-100%) |
| 545 | `fav_id00519_00028_id00535_wavtolip` | 1 | 1 | **TP** | 0.9495 (94.9%) | High Confidence Deepfake (70%-100%) |
| 546 | `fav_id00042_00028_id00330_zCFFgGSIZ_s` | 1 | 0 | **FN** | 0.0563 (5.6%) | High Confidence Authentic (0%-30%) |
| 547 | `fav_id01215_00001_0` | 1 | 1 | **TP** | 0.9511 (95.1%) | High Confidence Deepfake (70%-100%) |
| 548 | `fav_id01216_00025` | 0 | 0 | **TN** | 0.0334 (3.3%) | High Confidence Authentic (0%-30%) |
| 549 | `fav_id00018_00181` | 0 | 1 | **FP** | 0.8829 (88.3%) | High Confidence Deepfake (70%-100%) |
| 550 | `fav_id00266_00470_id01052_Kf4nlIEHfaU` | 1 | 1 | **TP** | 0.9305 (93.0%) | High Confidence Deepfake (70%-100%) |
| 551 | `fav_id02310_00139_id03985_wavtolip` | 1 | 0 | **FN** | 0.0437 (4.4%) | High Confidence Authentic (0%-30%) |
| 552 | `fav_id03556_00043` | 0 | 0 | **TN** | 0.0291 (2.9%) | High Confidence Authentic (0%-30%) |
| 553 | `fav_id00434_00046_id00633_UXrTMta1EDM_id00826_wavtolip` | 1 | 1 | **TP** | 0.9480 (94.8%) | High Confidence Deepfake (70%-100%) |
| 554 | `fav_id00519_00028_id00358_wavtolip` | 1 | 1 | **TP** | 0.8596 (86.0%) | High Confidence Deepfake (70%-100%) |
| 555 | `fav_id00999_00028_id00063_wavtolip` | 1 | 1 | **TP** | 0.9053 (90.5%) | High Confidence Deepfake (70%-100%) |
| 556 | `fav_id05435_00107` | 0 | 0 | **TN** | 0.0374 (3.7%) | High Confidence Authentic (0%-30%) |
| 557 | `fav_id06438_00110_id05435_wavtolip` | 1 | 1 | **TP** | 0.9603 (96.0%) | High Confidence Deepfake (70%-100%) |
| 558 | `fav_id00398_00016_id01238_VPdv2sexRy0_id00145_wavtolip` | 1 | 1 | **TP** | 0.9506 (95.1%) | High Confidence Deepfake (70%-100%) |
| 559 | `fav_id02464_00002_id00418_wavtolip` | 1 | 1 | **TP** | 0.9103 (91.0%) | High Confidence Deepfake (70%-100%) |
| 560 | `fav_id07463_00028_id00350_75IFJNFAMIM` | 1 | 1 | **TP** | 0.9349 (93.5%) | High Confidence Deepfake (70%-100%) |
| 561 | `fav_id00835_00195_fake` | 1 | 1 | **TP** | 0.9600 (96.0%) | High Confidence Deepfake (70%-100%) |
| 562 | `fav_id00033_00276_3` | 1 | 1 | **TP** | 0.9219 (92.2%) | High Confidence Deepfake (70%-100%) |
| 563 | `fav_id07051_00083` | 0 | 0 | **TN** | 0.0332 (3.3%) | High Confidence Authentic (0%-30%) |
| 564 | `fav_id03589_00002` | 0 | 0 | **TN** | 0.0318 (3.2%) | High Confidence Authentic (0%-30%) |
| 565 | `fav_id04560_00195` | 0 | 0 | **TN** | 0.0342 (3.4%) | High Confidence Authentic (0%-30%) |
| 566 | `fav_id01212_00183_id03168_wavtolip` | 1 | 1 | **TP** | 0.4898 (49.0%) | Moderately Confident Authentic (31%-49%) |
| 567 | `fav_id00232_00025` | 0 | 0 | **TN** | 0.0502 (5.0%) | High Confidence Authentic (0%-30%) |
| 568 | `fav_id02316_00094_id01691_wavtolip` | 1 | 1 | **TP** | 0.7746 (77.5%) | High Confidence Deepfake (70%-100%) |
| 569 | `fav_id02561_02561` | 0 | 0 | **TN** | 0.0300 (3.0%) | High Confidence Authentic (0%-30%) |
| 570 | `fav_id00358_00217_id00183_wavtolip` | 1 | 1 | **TP** | 0.9468 (94.7%) | High Confidence Deepfake (70%-100%) |
| 571 | `fav_id00575_00092_id00261_wavtolip` | 1 | 1 | **TP** | 0.9283 (92.8%) | High Confidence Deepfake (70%-100%) |
| 572 | `fav_id01018_00072_id00287_2NpfYjyMsPM_id03589_wavtolip` | 1 | 1 | **TP** | 0.9514 (95.1%) | High Confidence Deepfake (70%-100%) |
| 573 | `fav_id00813_00169` | 0 | 0 | **TN** | 0.0340 (3.4%) | High Confidence Authentic (0%-30%) |
| 574 | `fav_id01217_00005_id03707_wavtolip` | 1 | 1 | **TP** | 0.9494 (94.9%) | High Confidence Deepfake (70%-100%) |
| 575 | `fav_id01907_00148` | 0 | 0 | **TN** | 0.1929 (19.3%) | High Confidence Authentic (0%-30%) |
| 576 | `fav_id04144_00028_id06225_wavtolip` | 1 | 1 | **TP** | 0.9621 (96.2%) | High Confidence Deepfake (70%-100%) |
| 577 | `fav_id02332_00055_id00056_wavtolip` | 1 | 1 | **TP** | 0.9430 (94.3%) | High Confidence Deepfake (70%-100%) |
| 578 | `fav_id00554_00028_id00063_cVZBXC2rjPM` | 1 | 0 | **FN** | 0.0735 (7.3%) | High Confidence Authentic (0%-30%) |
| 579 | `fav_id00076_00109_7_id00166_wavtolip` | 1 | 1 | **TP** | 0.9426 (94.3%) | High Confidence Deepfake (70%-100%) |
| 580 | `fav_id01051_00322` | 0 | 0 | **TN** | 0.0766 (7.7%) | High Confidence Authentic (0%-30%) |
| 581 | `fav_id01018_00072_id00569_gSynmpwioV4` | 1 | 0 | **FN** | 0.0644 (6.4%) | High Confidence Authentic (0%-30%) |
| 582 | `fav_id00841_00078_id01245_7pYBPczR_z0_id01225_wavtolip` | 1 | 1 | **TP** | 0.9504 (95.0%) | High Confidence Deepfake (70%-100%) |
| 583 | `fav_id00345_00243_id03668_wavtolip` | 1 | 1 | **TP** | 0.9404 (94.0%) | High Confidence Deepfake (70%-100%) |
| 584 | `fav_id00555_00005` | 0 | 0 | **TN** | 0.2005 (20.0%) | High Confidence Authentic (0%-30%) |
| 585 | `fav_id06591_00021` | 0 | 1 | **FP** | 0.5579 (55.8%) | Moderately Confident Deepfake (51%-69%) |
| 586 | `fav_id03651_00092_id03716_s-lxbGafZQA_id00395_wavtolip` | 1 | 1 | **TP** | 0.9552 (95.5%) | High Confidence Deepfake (70%-100%) |
| 587 | `fav_id01637_00002_1_id01598_wavtolip` | 1 | 1 | **TP** | 0.9629 (96.3%) | High Confidence Deepfake (70%-100%) |
| 588 | `fav_id00763_00074` | 0 | 0 | **TN** | 0.0351 (3.5%) | High Confidence Authentic (0%-30%) |
| 589 | `fav_id01661_00059_id00832_wavtolip` | 1 | 1 | **TP** | 0.5562 (55.6%) | Moderately Confident Deepfake (51%-69%) |
| 590 | `fav_id04687_00066_id00740_wavtolip` | 1 | 1 | **TP** | 0.6686 (66.9%) | Moderately Confident Deepfake (51%-69%) |
| 591 | `fav_id06462_00014_id02807_eFC0STvhou4_faceswap` | 1 | 1 | **TP** | 0.9556 (95.6%) | High Confidence Deepfake (70%-100%) |
| 592 | `fav_id02807_00032` | 0 | 0 | **TN** | 0.1293 (12.9%) | High Confidence Authentic (0%-30%) |
| 593 | `fav_id04939_00174` | 0 | 0 | **TN** | 0.0444 (4.4%) | High Confidence Authentic (0%-30%) |
| 594 | `fav_id06066_00028` | 0 | 1 | **FP** | 0.4824 (48.2%) | Moderately Confident Authentic (31%-49%) |
| 595 | `fav_id00363_00014` | 0 | 1 | **FP** | 0.8472 (84.7%) | High Confidence Deepfake (70%-100%) |
| 596 | `fav_id00458_00072` | 0 | 0 | **TN** | 0.0379 (3.8%) | High Confidence Authentic (0%-30%) |
| 597 | `fav_id00592_00017_id05252_wavtolip` | 1 | 1 | **TP** | 0.9576 (95.8%) | High Confidence Deepfake (70%-100%) |
| 598 | `fav_id06355_00347` | 0 | 0 | **TN** | 0.1553 (15.5%) | High Confidence Authentic (0%-30%) |
| 599 | `fav_id01544_00044_3_id00166_wavtolip` | 1 | 1 | **TP** | 0.9499 (95.0%) | High Confidence Deepfake (70%-100%) |
| 600 | `fav_id00068_00004_2_id03716_wavtolip` | 1 | 1 | **TP** | 0.9594 (95.9%) | High Confidence Deepfake (70%-100%) |
| 601 | `fav_id03620_00081` | 0 | 0 | **TN** | 0.0353 (3.5%) | High Confidence Authentic (0%-30%) |
| 602 | `fav_id04490_00054_id04582_wavtolip` | 1 | 1 | **TP** | 0.9702 (97.0%) | High Confidence Deepfake (70%-100%) |
| 603 | `fav_id01098_00044_id00187_wavtolip` | 1 | 1 | **TP** | 0.9440 (94.4%) | High Confidence Deepfake (70%-100%) |
| 604 | `fav_id00582_00006_id08819_wavtolip` | 1 | 1 | **TP** | 0.9521 (95.2%) | High Confidence Deepfake (70%-100%) |
| 605 | `fav_id04554_00118_id00860_RGYyzn7hiCo_faceswap_id06355_wavtolip` | 1 | 1 | **TP** | 0.9392 (93.9%) | High Confidence Deepfake (70%-100%) |
| 606 | `fav_id06388_00005` | 0 | 0 | **TN** | 0.1652 (16.5%) | High Confidence Authentic (0%-30%) |
| 607 | `fav_id04144_00028` | 0 | 0 | **TN** | 0.0474 (4.7%) | High Confidence Authentic (0%-30%) |
| 608 | `fav_id00747_00053_id02310_zA-8Pw0SvZg_faceswap_id06437_wavtolip` | 1 | 1 | **TP** | 0.9574 (95.7%) | High Confidence Deepfake (70%-100%) |
| 609 | `fav_id00271_00028` | 0 | 0 | **TN** | 0.0313 (3.1%) | High Confidence Authentic (0%-30%) |
| 610 | `fav_id00694_00340_id00358_wavtolip` | 1 | 1 | **TP** | 0.9395 (94.0%) | High Confidence Deepfake (70%-100%) |
| 611 | `fav_id04691_1_3_id02553_wavtolip` | 1 | 1 | **TP** | 0.9564 (95.6%) | High Confidence Deepfake (70%-100%) |
| 612 | `fav_id00752_00340` | 0 | 1 | **FP** | 0.8763 (87.6%) | High Confidence Deepfake (70%-100%) |
| 613 | `fav_id01126_00040` | 0 | 1 | **FP** | 0.9189 (91.9%) | High Confidence Deepfake (70%-100%) |
| 614 | `fav_id01838_00126` | 0 | 0 | **TN** | 0.0621 (6.2%) | High Confidence Authentic (0%-30%) |
| 615 | `fav_id04222_00078` | 0 | 0 | **TN** | 0.2087 (20.9%) | High Confidence Authentic (0%-30%) |
| 616 | `fav_id03344_00114_id08314_66ufcsq8IWA_id04928_wavtolip` | 1 | 1 | **TP** | 0.9335 (93.3%) | High Confidence Deepfake (70%-100%) |
| 617 | `fav_id07102_00052_id00560_wavtolip` | 1 | 1 | **TP** | 0.9340 (93.4%) | High Confidence Deepfake (70%-100%) |
| 618 | `fav_id00368_00078` | 0 | 0 | **TN** | 0.0738 (7.4%) | High Confidence Authentic (0%-30%) |
| 619 | `fav_id04820_00015` | 0 | 0 | **TN** | 0.0385 (3.9%) | High Confidence Authentic (0%-30%) |
| 620 | `fav_id08613_00074` | 0 | 0 | **TN** | 0.0329 (3.3%) | High Confidence Authentic (0%-30%) |
| 621 | `fav_id00082_00052` | 0 | 1 | **FP** | 0.9316 (93.2%) | High Confidence Deepfake (70%-100%) |
| 622 | `fav_id00261_00048` | 0 | 0 | **TN** | 0.4261 (42.6%) | Moderately Confident Authentic (31%-49%) |
| 623 | `fav_id01098_00044` | 0 | 0 | **TN** | 0.0909 (9.1%) | High Confidence Authentic (0%-30%) |
| 624 | `fav_id01098_00044_id01058_wavtolip` | 1 | 1 | **TP** | 0.9576 (95.8%) | High Confidence Deepfake (70%-100%) |
| 625 | `fav_id05332_00065_id02332_wavtolip` | 1 | 1 | **TP** | 0.9162 (91.6%) | High Confidence Deepfake (70%-100%) |
| 626 | `fav_id01201_00028_id03757_L-M-XbdFLQM` | 1 | 1 | **TP** | 0.8265 (82.7%) | High Confidence Deepfake (70%-100%) |
| 627 | `fav_id05920_00161` | 0 | 0 | **TN** | 0.0481 (4.8%) | High Confidence Authentic (0%-30%) |
| 628 | `fav_id09143_00056` | 0 | 1 | **FP** | 0.5141 (51.4%) | Moderately Confident Deepfake (51%-69%) |
| 629 | `fav_id00052_00015` | 0 | 0 | **TN** | 0.0390 (3.9%) | High Confidence Authentic (0%-30%) |
| 630 | `fav_id04540_00078_id02824_glBy_mYcXZw_id00829_wavtolip` | 1 | 1 | **TP** | 0.8462 (84.6%) | High Confidence Deepfake (70%-100%) |
| 631 | `fav_id00145_00043_id00680_wavtolip` | 1 | 1 | **TP** | 0.9614 (96.1%) | High Confidence Deepfake (70%-100%) |
| 632 | `fav_id01058_00005_2_id00306_wavtolip` | 1 | 1 | **TP** | 0.9505 (95.1%) | High Confidence Deepfake (70%-100%) |
| 633 | `fav_id01532_00065` | 0 | 0 | **TN** | 0.4520 (45.2%) | Moderately Confident Authentic (31%-49%) |
| 634 | `fav_id01231_00015_id01248_wavtolip` | 1 | 1 | **TP** | 0.9470 (94.7%) | High Confidence Deepfake (70%-100%) |
| 635 | `fav_id07200_00045_id08314_wavtolip` | 1 | 1 | **TP** | 0.8722 (87.2%) | High Confidence Deepfake (70%-100%) |
| 636 | `fav_id00149_00284` | 0 | 0 | **TN** | 0.3549 (35.5%) | Moderately Confident Authentic (31%-49%) |
| 637 | `fav_id00739_00005_id05845_wavtolip` | 1 | 1 | **TP** | 0.9613 (96.1%) | High Confidence Deepfake (70%-100%) |
| 638 | `fav_id03965_00051_id06878_wavtolip` | 1 | 1 | **TP** | 0.9574 (95.7%) | High Confidence Deepfake (70%-100%) |
| 639 | `fav_id01528_00017` | 0 | 0 | **TN** | 0.0390 (3.9%) | High Confidence Authentic (0%-30%) |
| 640 | `fav_id00520_00187` | 0 | 1 | **FP** | 0.8715 (87.1%) | High Confidence Deepfake (70%-100%) |
| 641 | `fav_id00535_00005` | 0 | 0 | **TN** | 0.0571 (5.7%) | High Confidence Authentic (0%-30%) |
| 642 | `fav_id00305_00113_id01156_wavtolip` | 1 | 0 | **FN** | 0.0452 (4.5%) | High Confidence Authentic (0%-30%) |
| 643 | `fav_id00087_00002_id04073_wavtolip` | 1 | 1 | **TP** | 0.9529 (95.3%) | High Confidence Deepfake (70%-100%) |
| 644 | `fav_id00414_00052_id00773_wavtolip` | 1 | 1 | **TP** | 0.9389 (93.9%) | High Confidence Deepfake (70%-100%) |
| 645 | `fav_id00520_00187_id00055_wavtolip` | 1 | 1 | **TP** | 0.9712 (97.1%) | High Confidence Deepfake (70%-100%) |
| 646 | `fav_id00461_00043` | 0 | 0 | **TN** | 0.0357 (3.6%) | High Confidence Authentic (0%-30%) |
| 647 | `fav_id04490_00054` | 0 | 1 | **FP** | 0.7447 (74.5%) | High Confidence Deepfake (70%-100%) |
| 648 | `fav_id03889_00052_id04774_k3yOcTA95dg` | 1 | 0 | **FN** | 0.2598 (26.0%) | High Confidence Authentic (0%-30%) |
| 649 | `fav_id00403_00052` | 0 | 0 | **TN** | 0.0361 (3.6%) | High Confidence Authentic (0%-30%) |
| 650 | `fav_id03379_00032` | 0 | 1 | **FP** | 0.9480 (94.8%) | High Confidence Deepfake (70%-100%) |
| 651 | `fav_id00055_00120` | 0 | 1 | **FP** | 0.8975 (89.8%) | High Confidence Deepfake (70%-100%) |
| 652 | `fav_id01124_00063` | 0 | 0 | **TN** | 0.0512 (5.1%) | High Confidence Authentic (0%-30%) |
| 653 | `fav_id00418_00052_id00385_oqf8cNi7lOI_id02464_wavtolip` | 1 | 1 | **TP** | 0.9499 (95.0%) | High Confidence Deepfake (70%-100%) |
| 654 | `fav_id00554_00028_id01157_yDgqfOWVKko_faceswap` | 1 | 1 | **TP** | 0.4831 (48.3%) | Moderately Confident Authentic (31%-49%) |
| 655 | `fav_id01211_00023` | 0 | 0 | **TN** | 0.0430 (4.3%) | High Confidence Authentic (0%-30%) |
| 656 | `fav_id00100_00028` | 0 | 0 | **TN** | 0.0397 (4.0%) | High Confidence Authentic (0%-30%) |
| 657 | `fav_id00694_00340_id00535_BhvTIJ-YG5s_faceswap_id01099_wavtolip` | 1 | 1 | **TP** | 0.9333 (93.3%) | High Confidence Deepfake (70%-100%) |
| 658 | `fav_id00520_00187_id00559_Y7ovoKuvL78_faceswap_id00946_wavtolip` | 1 | 1 | **TP** | 0.9683 (96.8%) | High Confidence Deepfake (70%-100%) |
| 659 | `fav_id01995_00071_id04727_wavtolip` | 1 | 1 | **TP** | 0.8104 (81.0%) | High Confidence Deepfake (70%-100%) |
| 660 | `fav_id00383_00171_id00575_iPfJcAJQ84Y_id00180_wavtolip` | 1 | 1 | **TP** | 0.9631 (96.3%) | High Confidence Deepfake (70%-100%) |
| 661 | `fav_id04073_00021_id00777_wavtolip` | 1 | 1 | **TP** | 0.9561 (95.6%) | High Confidence Deepfake (70%-100%) |
| 662 | `fav_id00460_00005` | 0 | 0 | **TN** | 0.0339 (3.4%) | High Confidence Authentic (0%-30%) |
| 663 | `fav_id07195_00186_id07463_wavtolip` | 1 | 1 | **TP** | 0.9485 (94.8%) | High Confidence Deepfake (70%-100%) |
| 664 | `fav_id00826_00065_id00325_wavtolip` | 1 | 1 | **TP** | 0.7340 (73.4%) | High Confidence Deepfake (70%-100%) |
| 665 | `fav_id06225_00005` | 0 | 0 | **TN** | 0.2746 (27.5%) | High Confidence Authentic (0%-30%) |
| 666 | `fav_id01248_00005_id01004_867Wlj7Gw68_id01217_wavtolip` | 1 | 1 | **TP** | 0.9649 (96.5%) | High Confidence Deepfake (70%-100%) |
| 667 | `fav_id01216_00025_id00261_wavtolip` | 1 | 0 | **FN** | 0.0813 (8.1%) | High Confidence Authentic (0%-30%) |
| 668 | `fav_id04884_00028` | 0 | 0 | **TN** | 0.1655 (16.5%) | High Confidence Authentic (0%-30%) |
| 669 | `fav_id03757_00149_0_id01124_wavtolip` | 1 | 1 | **TP** | 0.9569 (95.7%) | High Confidence Deepfake (70%-100%) |
| 670 | `fav_id02332_00055_id04789_wavtolip` | 1 | 1 | **TP** | 0.9187 (91.9%) | High Confidence Deepfake (70%-100%) |
| 671 | `fav_id00398_00016` | 0 | 0 | **TN** | 0.0324 (3.2%) | High Confidence Authentic (0%-30%) |
| 672 | `fav_id05268_00010` | 0 | 1 | **FP** | 0.9310 (93.1%) | High Confidence Deepfake (70%-100%) |
| 673 | `fav_id06334_00015` | 0 | 0 | **TN** | 0.0308 (3.1%) | High Confidence Authentic (0%-30%) |
| 674 | `fav_id03757_00149_id00169__V1xAwwG29U` | 1 | 1 | **TP** | 0.9540 (95.4%) | High Confidence Deepfake (70%-100%) |
| 675 | `fav_id00579_00030` | 0 | 1 | **FP** | 0.9235 (92.4%) | High Confidence Deepfake (70%-100%) |
| 676 | `fav_id02838_00080` | 0 | 0 | **TN** | 0.0455 (4.5%) | High Confidence Authentic (0%-30%) |
| 677 | `fav_id00618_00195` | 0 | 0 | **TN** | 0.0375 (3.7%) | High Confidence Authentic (0%-30%) |
| 678 | `fav_id01783_00015_id04736_wavtolip` | 1 | 0 | **FN** | 0.2465 (24.6%) | High Confidence Authentic (0%-30%) |
| 679 | `fav_id00190_00072_id00180_lkeAa8Od_tg_faceswap_id00752_wavtolip` | 1 | 1 | **TP** | 0.9623 (96.2%) | High Confidence Deepfake (70%-100%) |
| 680 | `fav_id04222_00078_id06269_wavtolip` | 1 | 1 | **TP** | 0.6876 (68.8%) | Moderately Confident Deepfake (51%-69%) |
| 681 | `fav_id00735_00037` | 0 | 0 | **TN** | 0.2427 (24.3%) | High Confidence Authentic (0%-30%) |
| 682 | `fav_id04774_00032` | 0 | 1 | **FP** | 0.4864 (48.6%) | Moderately Confident Authentic (31%-49%) |
| 683 | `fav_id01182_00167_2_id00775_wavtolip` | 1 | 1 | **TP** | 0.9445 (94.5%) | High Confidence Deepfake (70%-100%) |
| 684 | `fav_id00498_00014` | 0 | 1 | **FP** | 0.7415 (74.2%) | High Confidence Deepfake (70%-100%) |
| 685 | `fav_id01521_00109_id01779_HgyHpDEo_jk_id01920_wavtolip` | 1 | 1 | **TP** | 0.9537 (95.4%) | High Confidence Deepfake (70%-100%) |
| 686 | `fav_id01530_00002` | 0 | 0 | **TN** | 0.0295 (2.9%) | High Confidence Authentic (0%-30%) |
| 687 | `fav_id01091_00236_0` | 1 | 1 | **TP** | 0.9565 (95.6%) | High Confidence Deepfake (70%-100%) |
| 688 | `fav_id01238_00037` | 0 | 0 | **TN** | 0.0359 (3.6%) | High Confidence Authentic (0%-30%) |
| 689 | `fav_id00529_00409_id03678_dsfCxodI9i8_faceswap_id04034_wavtolip` | 1 | 1 | **TP** | 0.9138 (91.4%) | High Confidence Deepfake (70%-100%) |
| 690 | `fav_id00068_00004` | 0 | 0 | **TN** | 0.0373 (3.7%) | High Confidence Authentic (0%-30%) |
| 691 | `fav_id04414_00001` | 0 | 0 | **TN** | 0.0468 (4.7%) | High Confidence Authentic (0%-30%) |
| 692 | `fav_id00597_00019_id05743_Fjri4MXHXWY_faceswap` | 1 | 1 | **TP** | 0.8256 (82.6%) | High Confidence Deepfake (70%-100%) |
| 693 | `fav_id02310_00139` | 0 | 0 | **TN** | 0.0290 (2.9%) | High Confidence Authentic (0%-30%) |
| 694 | `fav_id01238_00037_id03605_JGZt9DAhBRs_id00272_wavtolip` | 1 | 1 | **TP** | 0.9234 (92.3%) | High Confidence Deepfake (70%-100%) |
| 695 | `fav_id00186_00120` | 0 | 0 | **TN** | 0.0457 (4.6%) | High Confidence Authentic (0%-30%) |
| 696 | `fav_id00088_00005` | 0 | 1 | **FP** | 0.8698 (87.0%) | High Confidence Deepfake (70%-100%) |
| 697 | `fav_id00999_00028` | 0 | 1 | **FP** | 0.9246 (92.5%) | High Confidence Deepfake (70%-100%) |
| 698 | `fav_id01035_00012` | 0 | 0 | **TN** | 0.3348 (33.5%) | Moderately Confident Authentic (31%-49%) |
| 699 | `fav_id00415_00017_id00292_qELK8N_DykA` | 1 | 0 | **FN** | 0.2121 (21.2%) | High Confidence Authentic (0%-30%) |
| 700 | `fav_id04689_00005` | 0 | 0 | **TN** | 0.1181 (11.8%) | High Confidence Authentic (0%-30%) |
| **OVERALL** | **700 Clips Total** | **350 Real, 350 Fake** | **315 Pred Real, 385 Pred Fake** | **TP: 305, TN: 270, FP: 80, FN: 45** | **Mean Score: 0.5363 (Real: 0.2531, Fake: 0.8195)** | **Accuracy: 82.14% (575/700) — F1: 0.8299, AUC: 0.9020** |

---

### Table 8.B: Full Clip-by-Clip Emotion Recognition Tally Sheet (CREMA-D, N = 700)
*(Complete list of 700 held-out CREMA-D clips evaluated on Audio Head A [72.4% Acc] and Visual Head B [74.1% Acc].)*

| Row # | Clip ID | Ground Truth Emotion | Emotion Head A (Audio-Text) | Audio Outcome | Ground Truth Emotion | Emotion Head B (Visual) | Visual Outcome |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `FAKE_T3_1068_WSI_NEU_XX__AUDIO_1068_WSI_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 2 | `FAKE_T3_1048_IEO_HAP_MD__AUDIO_1048_IEO_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 3 | `FAKE_T2_1060_IEO_NEU_XX__AUDIO_1060_IEO_SAD_HI_wav2lip` | Sad | Happy | Misclassified | Neutral | Fear | Misclassified |
| 4 | `FAKE_T3_1048_ITH_DIS_XX__AUDIO_1048_ITH_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Sad | Misclassified |
| 5 | `FAKE_T3_1060_MTI_HAP_XX__AUDIO_1060_MTI_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Fear | Misclassified |
| 6 | `FAKE_T3_1048_IWL_FEA_XX__AUDIO_1048_IWL_HAP_XX_sadtalker` | Happy | Angry | Misclassified | Fear | Fear | Correct |
| 7 | `FAKE_T2_1046_IEO_DIS_MD__AUDIO_1046_IEO_FEA_MD_wav2lip` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 8 | `FAKE_T3_1033_TSI_ANG_XX__AUDIO_1033_TSI_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 9 | `FAKE_T2_1063_IEO_HAP_MD__AUDIO_1063_IEO_NEU_XX_wav2lip` | Neutral | Fear | Misclassified | Happy | Happy | Correct |
| 10 | `FAKE_T3_1033_WSI_DIS_XX__AUDIO_1033_WSI_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 11 | `FAKE_T2_1068_IOM_FEA_XX__AUDIO_1068_IOM_HAP_XX_wav2lip` | Happy | Neutral | Misclassified | Fear | Fear | Correct |
| 12 | `FAKE_T3_1046_WSI_FEA_XX__AUDIO_1046_WSI_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Sad | Misclassified |
| 13 | `FAKE_T2_1046_TAI_ANG_XX__AUDIO_1046_TAI_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Sad | Misclassified |
| 14 | `FAKE_T3_1056_TSI_ANG_XX__AUDIO_1056_TSI_DIS_XX_sadtalker` | Disgust | Happy | Misclassified | Angry | Angry | Correct |
| 15 | `FAKE_T2_1071_WSI_HAP_XX__AUDIO_1071_WSI_NEU_XX_wav2lip` | Neutral | Fear | Misclassified | Happy | Happy | Correct |
| 16 | `FAKE_T3_1068_MTI_DIS_XX__AUDIO_1068_MTI_FEA_XX_sadtalker` | Fear | Disgust | Misclassified | Disgust | Fear | Misclassified |
| 17 | `FAKE_T3_1068_IEO_ANG_MD__AUDIO_1068_IEO_DIS_MD_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 18 | `FAKE_T3_1071_IOM_NEU_XX__AUDIO_1071_IOM_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Sad | Misclassified |
| 19 | `FAKE_T3_1068_TIE_SAD_XX__AUDIO_1068_TIE_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Fear | Misclassified |
| 20 | `FAKE_T1_1071_IOM_SAD_XX__AUDIO_1071_IOM_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 21 | `FAKE_T2_1060_TSI_SAD_XX__AUDIO_1060_TSI_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 22 | `FAKE_T3_1063_TSI_SAD_XX__AUDIO_1063_TSI_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 23 | `FAKE_T3_1020_DFA_ANG_XX__AUDIO_1020_DFA_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Neutral | Misclassified |
| 24 | `FAKE_T3_1033_ITH_NEU_XX__AUDIO_1033_ITH_SAD_XX_sadtalker` | Sad | Happy | Misclassified | Neutral | Neutral | Correct |
| 25 | `FAKE_T2_1071_IOM_DIS_XX__AUDIO_1071_IOM_FEA_XX_wav2lip` | Fear | Sad | Misclassified | Disgust | Neutral | Misclassified |
| 26 | `FAKE_T3_1068_TAI_HAP_XX__AUDIO_1068_TAI_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 27 | `FAKE_T3_1033_TAI_ANG_XX__AUDIO_1033_TAI_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 28 | `FAKE_T3_1063_IEO_SAD_MD__AUDIO_1063_IEO_ANG_MD_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 29 | `FAKE_T2_1063_MTI_NEU_XX__AUDIO_1063_MTI_SAD_XX_wav2lip` | Sad | Fear | Misclassified | Neutral | Angry | Misclassified |
| 30 | `FAKE_T3_1046_MTI_SAD_XX__AUDIO_1046_MTI_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 31 | `FAKE_T3_1071_ITS_DIS_XX__AUDIO_1071_ITS_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 32 | `FAKE_T3_1068_ITH_ANG_XX__AUDIO_1068_ITH_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 33 | `FAKE_T2_1033_TIE_FEA_XX__AUDIO_1033_TIE_HAP_XX_wav2lip` | Happy | Happy | Correct | Fear | Disgust | Misclassified |
| 34 | `FAKE_T3_1063_IEO_ANG_LO__AUDIO_1063_IEO_DIS_LO_sadtalker` | Disgust | Angry | Misclassified | Angry | Angry | Correct |
| 35 | `FAKE_T2_1046_TIE_HAP_XX__AUDIO_1046_TIE_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Angry | Misclassified |
| 36 | `FAKE_T3_1068_IEO_SAD_HI__AUDIO_1068_IEO_ANG_HI_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 37 | `FAKE_T3_1063_MTI_HAP_XX__AUDIO_1063_MTI_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 38 | `FAKE_T2_1060_IWW_HAP_XX__AUDIO_1060_IWW_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 39 | `FAKE_T2_1060_IEO_SAD_LO__AUDIO_1060_IEO_ANG_LO_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 40 | `FAKE_T3_1063_WSI_SAD_XX__AUDIO_1063_WSI_ANG_XX_sadtalker` | Angry | Fear | Misclassified | Sad | Angry | Misclassified |
| 41 | `FAKE_T2_1068_IWL_SAD_XX__AUDIO_1068_IWL_ANG_XX_wav2lip` | Angry | Fear | Misclassified | Sad | Disgust | Misclassified |
| 42 | `FAKE_T2_1060_IWL_DIS_XX__AUDIO_1060_IWL_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 43 | `FAKE_T3_1033_IEO_DIS_HI__AUDIO_1033_IEO_FEA_HI_sadtalker` | Fear | Happy | Misclassified | Disgust | Sad | Misclassified |
| 44 | `FAKE_T2_1046_IOM_FEA_XX__AUDIO_1046_IOM_HAP_XX_wav2lip` | Happy | Angry | Misclassified | Fear | Fear | Correct |
| 45 | `FAKE_T1_1060_IEO_ANG_HI__AUDIO_1060_IEO_DIS_HI_styletts` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 46 | `FAKE_T3_1068_IEO_DIS_HI__AUDIO_1068_IEO_FEA_HI_sadtalker` | Fear | Fear | Correct | Disgust | Fear | Misclassified |
| 47 | `FAKE_T3_1048_WSI_FEA_XX__AUDIO_1048_WSI_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 48 | `FAKE_T3_1068_DFA_ANG_XX__AUDIO_1068_DFA_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 49 | `FAKE_T2_1033_ITS_FEA_XX__AUDIO_1033_ITS_HAP_XX_wav2lip` | Happy | Happy | Correct | Fear | Happy | Misclassified |
| 50 | `FAKE_T3_1063_DFA_NEU_XX__AUDIO_1063_DFA_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Fear | Misclassified |
| 51 | `FAKE_T3_1063_DFA_SAD_XX__AUDIO_1063_DFA_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 52 | `FAKE_T1_1063_IEO_HAP_LO__AUDIO_1063_IEO_NEU_XX_styletts` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 53 | `FAKE_T2_1056_IWL_NEU_XX__AUDIO_1056_IWL_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 54 | `FAKE_T2_1056_IEO_SAD_HI__AUDIO_1056_IEO_ANG_HI_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 55 | `FAKE_T3_1071_WSI_FEA_XX__AUDIO_1071_WSI_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 56 | `FAKE_T1_1048_TSI_DIS_XX__AUDIO_1048_TSI_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 57 | `FAKE_T2_1071_TIE_ANG_XX__AUDIO_1071_TIE_DIS_XX_wav2lip` | Disgust | Sad | Misclassified | Angry | Happy | Misclassified |
| 58 | `FAKE_T3_1033_TSI_FEA_XX__AUDIO_1033_TSI_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Angry | Misclassified |
| 59 | `FAKE_T1_1060_IEO_DIS_LO__AUDIO_1060_IEO_FEA_LO_styletts` | Fear | Sad | Misclassified | Disgust | Disgust | Correct |
| 60 | `FAKE_T3_1060_ITH_SAD_XX__AUDIO_1060_ITH_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Angry | Misclassified |
| 61 | `FAKE_T2_1071_WSI_SAD_XX__AUDIO_1071_WSI_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 62 | `FAKE_T2_1071_TIE_DIS_XX__AUDIO_1071_TIE_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 63 | `FAKE_T3_1033_DFA_ANG_XX__AUDIO_1033_DFA_DIS_XX_sadtalker` | Disgust | Angry | Misclassified | Angry | Disgust | Misclassified |
| 64 | `FAKE_T2_1063_ITS_FEA_XX__AUDIO_1063_ITS_HAP_XX_wav2lip` | Happy | Sad | Misclassified | Fear | Fear | Correct |
| 65 | `FAKE_T3_1020_IWW_NEU_XX__AUDIO_1020_IWW_SAD_XX_sadtalker` | Sad | Angry | Misclassified | Neutral | Neutral | Correct |
| 66 | `FAKE_T3_1056_TIE_FEA_XX__AUDIO_1056_TIE_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 67 | `FAKE_T2_1068_IWL_HAP_XX__AUDIO_1068_IWL_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Fear | Misclassified |
| 68 | `FAKE_T2_1033_IEO_DIS_MD__AUDIO_1033_IEO_FEA_MD_wav2lip` | Fear | Fear | Correct | Disgust | Sad | Misclassified |
| 69 | `FAKE_T2_1020_IEO_SAD_LO__AUDIO_1020_IEO_ANG_LO_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 70 | `FAKE_T3_1063_IWW_HAP_XX__AUDIO_1063_IWW_NEU_XX_sadtalker` | Neutral | Fear | Misclassified | Happy | Happy | Correct |
| 71 | `FAKE_T2_1056_ITH_ANG_XX__AUDIO_1056_ITH_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 72 | `FAKE_T3_1056_IOM_ANG_XX__AUDIO_1056_IOM_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 73 | `FAKE_T2_1071_IEO_SAD_LO__AUDIO_1071_IEO_ANG_LO_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 74 | `FAKE_T1_1056_DFA_ANG_XX__AUDIO_1056_DFA_DIS_XX_styletts` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 75 | `FAKE_T3_1046_IWW_SAD_XX__AUDIO_1046_IWW_ANG_XX_sadtalker` | Angry | Sad | Misclassified | Sad | Sad | Correct |
| 76 | `FAKE_T3_1020_TSI_DIS_XX__AUDIO_1020_TSI_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Neutral | Misclassified |
| 77 | `FAKE_T1_1060_DFA_ANG_XX__AUDIO_1060_DFA_DIS_XX_styletts` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 78 | `FAKE_T2_1056_IEO_DIS_MD__AUDIO_1056_IEO_FEA_MD_wav2lip` | Fear | Disgust | Misclassified | Disgust | Disgust | Correct |
| 79 | `FAKE_T1_1046_WSI_SAD_XX__AUDIO_1046_WSI_ANG_XX_styletts` | Angry | Happy | Misclassified | Sad | Sad | Correct |
| 80 | `FAKE_T3_1068_IWW_SAD_XX__AUDIO_1068_IWW_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 81 | `FAKE_T3_1020_ITH_ANG_XX__AUDIO_1020_ITH_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 82 | `FAKE_T1_1046_IEO_ANG_MD__AUDIO_1046_IEO_DIS_MD_styletts` | Disgust | Disgust | Correct | Angry | Disgust | Misclassified |
| 83 | `FAKE_T1_1046_MTI_ANG_XX__AUDIO_1046_MTI_DIS_XX_styletts` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 84 | `FAKE_T3_1033_IWL_ANG_XX__AUDIO_1033_IWL_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 85 | `FAKE_T3_1060_DFA_FEA_XX__AUDIO_1060_DFA_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 86 | `FAKE_T3_1048_DFA_DIS_XX__AUDIO_1048_DFA_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 87 | `FAKE_T3_1046_IEO_DIS_LO__AUDIO_1046_IEO_FEA_LO_sadtalker` | Fear | Disgust | Misclassified | Disgust | Disgust | Correct |
| 88 | `FAKE_T2_1056_TSI_SAD_XX__AUDIO_1056_TSI_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Happy | Misclassified |
| 89 | `FAKE_T1_1068_ITS_HAP_XX__AUDIO_1068_ITS_NEU_XX_styletts` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 90 | `FAKE_T2_1033_TIE_DIS_XX__AUDIO_1033_TIE_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 91 | `FAKE_T3_1056_MTI_HAP_XX__AUDIO_1056_MTI_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 92 | `FAKE_T2_1060_ITS_ANG_XX__AUDIO_1060_ITS_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 93 | `FAKE_T3_1046_ITH_SAD_XX__AUDIO_1046_ITH_ANG_XX_sadtalker` | Angry | Disgust | Misclassified | Sad | Neutral | Misclassified |
| 94 | `FAKE_T3_1068_IEO_ANG_HI__AUDIO_1068_IEO_DIS_HI_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 95 | `FAKE_T3_1060_DFA_NEU_XX__AUDIO_1060_DFA_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 96 | `FAKE_T3_1068_DFA_DIS_XX__AUDIO_1068_DFA_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 97 | `FAKE_T2_1020_TIE_FEA_XX__AUDIO_1020_TIE_HAP_XX_wav2lip` | Happy | Fear | Misclassified | Fear | Happy | Misclassified |
| 98 | `FAKE_T3_1046_WSI_NEU_XX__AUDIO_1046_WSI_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 99 | `FAKE_T1_1048_IEO_SAD_HI__AUDIO_1048_IEO_ANG_HI_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 100 | `FAKE_T3_1056_DFA_NEU_XX__AUDIO_1056_DFA_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 101 | `FAKE_T2_1020_WSI_DIS_XX__AUDIO_1020_WSI_FEA_XX_wav2lip` | Fear | Angry | Misclassified | Disgust | Sad | Misclassified |
| 102 | `FAKE_T3_1071_IWL_DIS_XX__AUDIO_1071_IWL_FEA_XX_sadtalker` | Fear | Neutral | Misclassified | Disgust | Disgust | Correct |
| 103 | `FAKE_T3_1063_IWW_ANG_XX__AUDIO_1063_IWW_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 104 | `FAKE_T1_1063_ITS_HAP_XX__AUDIO_1063_ITS_NEU_XX_styletts` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 105 | `FAKE_T2_1068_IOM_ANG_XX__AUDIO_1068_IOM_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 106 | `FAKE_T3_1068_IWL_FEA_XX__AUDIO_1068_IWL_HAP_XX_sadtalker` | Happy | Fear | Misclassified | Fear | Fear | Correct |
| 107 | `FAKE_T3_1071_TAI_DIS_XX__AUDIO_1071_TAI_FEA_XX_sadtalker` | Fear | Angry | Misclassified | Disgust | Disgust | Correct |
| 108 | `FAKE_T2_1063_IOM_NEU_XX__AUDIO_1063_IOM_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Sad | Misclassified |
| 109 | `FAKE_T2_1048_MTI_ANG_XX__AUDIO_1048_MTI_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 110 | `FAKE_T3_1046_ITH_NEU_XX__AUDIO_1046_ITH_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 111 | `FAKE_T1_1048_ITS_NEU_XX__AUDIO_1048_ITS_SAD_XX_styletts` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 112 | `FAKE_T1_1048_WSI_DIS_XX__AUDIO_1048_WSI_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 113 | `FAKE_T1_1056_IWL_HAP_XX__AUDIO_1056_IWL_NEU_XX_styletts` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 114 | `FAKE_T1_1056_IEO_ANG_HI__AUDIO_1056_IEO_DIS_HI_styletts` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 115 | `FAKE_T3_1033_IWL_HAP_XX__AUDIO_1033_IWL_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Angry | Misclassified |
| 116 | `FAKE_T2_1063_IEO_DIS_MD__AUDIO_1063_IEO_FEA_MD_wav2lip` | Fear | Happy | Misclassified | Disgust | Disgust | Correct |
| 117 | `FAKE_T2_1056_TAI_FEA_XX__AUDIO_1056_TAI_HAP_XX_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 118 | `FAKE_T3_1060_IEO_HAP_MD__AUDIO_1060_IEO_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 119 | `FAKE_T3_1033_ITH_HAP_XX__AUDIO_1033_ITH_NEU_XX_sadtalker` | Neutral | Angry | Misclassified | Happy | Happy | Correct |
| 120 | `FAKE_T3_1046_IWL_DIS_XX__AUDIO_1046_IWL_FEA_XX_sadtalker` | Fear | Happy | Misclassified | Disgust | Disgust | Correct |
| 121 | `FAKE_T1_1060_TSI_FEA_XX__AUDIO_1060_TSI_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Fear | Correct |
| 122 | `FAKE_T1_1060_ITS_NEU_XX__AUDIO_1060_ITS_SAD_XX_styletts` | Sad | Sad | Correct | Neutral | Fear | Misclassified |
| 123 | `FAKE_T2_1033_IEO_FEA_HI__AUDIO_1033_IEO_HAP_HI_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 124 | `FAKE_T3_1060_TAI_SAD_XX__AUDIO_1060_TAI_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 125 | `FAKE_T3_1068_TAI_NEU_XX__AUDIO_1068_TAI_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 126 | `FAKE_T2_1071_IWW_DIS_XX__AUDIO_1071_IWW_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 127 | `FAKE_T2_1046_IWL_NEU_XX__AUDIO_1046_IWL_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 128 | `FAKE_T2_1048_IOM_HAP_XX__AUDIO_1048_IOM_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 129 | `FAKE_T3_1048_IWL_ANG_XX__AUDIO_1048_IWL_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 130 | `FAKE_T1_1020_IEO_NEU_XX__AUDIO_1020_IEO_SAD_HI_styletts` | Sad | Fear | Misclassified | Neutral | Neutral | Correct |
| 131 | `FAKE_T1_1063_ITH_DIS_XX__AUDIO_1063_ITH_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 132 | `FAKE_T1_1048_WSI_HAP_XX__AUDIO_1048_WSI_NEU_XX_styletts` | Neutral | Neutral | Correct | Happy | Neutral | Misclassified |
| 133 | `FAKE_T1_1060_ITH_NEU_XX__AUDIO_1060_ITH_SAD_XX_styletts` | Sad | Fear | Misclassified | Neutral | Neutral | Correct |
| 134 | `FAKE_T3_1048_IWL_SAD_XX__AUDIO_1048_IWL_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Disgust | Misclassified |
| 135 | `FAKE_T3_1068_TSI_ANG_XX__AUDIO_1068_TSI_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Fear | Misclassified |
| 136 | `FAKE_T3_1046_IOM_ANG_XX__AUDIO_1046_IOM_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 137 | `FAKE_T3_1063_IWW_DIS_XX__AUDIO_1063_IWW_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 138 | `FAKE_T2_1033_ITS_ANG_XX__AUDIO_1033_ITS_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 139 | `FAKE_T2_1068_TSI_HAP_XX__AUDIO_1068_TSI_NEU_XX_wav2lip` | Neutral | Disgust | Misclassified | Happy | Happy | Correct |
| 140 | `FAKE_T3_1060_IWW_DIS_XX__AUDIO_1060_IWW_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Angry | Misclassified |
| 141 | `FAKE_T3_1020_WSI_HAP_XX__AUDIO_1020_WSI_NEU_XX_sadtalker` | Neutral | Disgust | Misclassified | Happy | Happy | Correct |
| 142 | `FAKE_T3_1048_ITH_HAP_XX__AUDIO_1048_ITH_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 143 | `FAKE_T1_1048_IEO_HAP_LO__AUDIO_1048_IEO_NEU_XX_styletts` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 144 | `FAKE_T3_1071_IOM_HAP_XX__AUDIO_1071_IOM_NEU_XX_sadtalker` | Neutral | Disgust | Misclassified | Happy | Happy | Correct |
| 145 | `FAKE_T3_1060_TSI_HAP_XX__AUDIO_1060_TSI_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 146 | `FAKE_T2_1033_IWW_FEA_XX__AUDIO_1033_IWW_HAP_XX_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 147 | `FAKE_T3_1048_TAI_ANG_XX__AUDIO_1048_TAI_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Fear | Misclassified |
| 148 | `FAKE_T3_1063_IEO_ANG_HI__AUDIO_1063_IEO_DIS_HI_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 149 | `FAKE_T2_1033_IEO_ANG_MD__AUDIO_1033_IEO_DIS_MD_wav2lip` | Disgust | Disgust | Correct | Angry | Disgust | Misclassified |
| 150 | `FAKE_T3_1056_IEO_SAD_LO__AUDIO_1056_IEO_ANG_LO_sadtalker` | Angry | Neutral | Misclassified | Sad | Sad | Correct |
| 151 | `FAKE_T2_1068_MTI_FEA_XX__AUDIO_1068_MTI_HAP_XX_wav2lip` | Happy | Angry | Misclassified | Fear | Fear | Correct |
| 152 | `FAKE_T3_1063_DFA_DIS_XX__AUDIO_1063_DFA_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 153 | `FAKE_T3_1071_MTI_SAD_XX__AUDIO_1071_MTI_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 154 | `FAKE_T3_1068_TIE_NEU_XX__AUDIO_1068_TIE_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 155 | `FAKE_T2_1033_TSI_HAP_XX__AUDIO_1033_TSI_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 156 | `FAKE_T1_1071_WSI_DIS_XX__AUDIO_1071_WSI_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 157 | `FAKE_T2_1048_IEO_ANG_LO__AUDIO_1048_IEO_DIS_LO_wav2lip` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 158 | `FAKE_T3_1056_MTI_ANG_XX__AUDIO_1056_MTI_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 159 | `FAKE_T1_1033_TSI_NEU_XX__AUDIO_1033_TSI_SAD_XX_styletts` | Sad | Disgust | Misclassified | Neutral | Disgust | Misclassified |
| 160 | `FAKE_T3_1046_IEO_DIS_HI__AUDIO_1046_IEO_FEA_HI_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 161 | `FAKE_T3_1060_ITH_ANG_XX__AUDIO_1060_ITH_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Neutral | Misclassified |
| 162 | `FAKE_T1_1048_ITS_FEA_XX__AUDIO_1048_ITS_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Fear | Correct |
| 163 | `FAKE_T3_1020_MTI_ANG_XX__AUDIO_1020_MTI_DIS_XX_sadtalker` | Disgust | Neutral | Misclassified | Angry | Angry | Correct |
| 164 | `FAKE_T2_1060_IEO_HAP_HI__AUDIO_1060_IEO_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Sad | Misclassified |
| 165 | `FAKE_T2_1046_ITS_NEU_XX__AUDIO_1046_ITS_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Angry | Misclassified |
| 166 | `FAKE_T2_1048_IOM_DIS_XX__AUDIO_1048_IOM_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 167 | `FAKE_T3_1046_TSI_NEU_XX__AUDIO_1046_TSI_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 168 | `FAKE_T2_1060_IEO_FEA_LO__AUDIO_1060_IEO_HAP_LO_wav2lip` | Happy | Happy | Correct | Fear | Happy | Misclassified |
| 169 | `FAKE_T2_1068_TSI_DIS_XX__AUDIO_1068_TSI_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Fear | Misclassified |
| 170 | `FAKE_T2_1068_ITH_SAD_XX__AUDIO_1068_ITH_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 171 | `FAKE_T1_1060_TIE_FEA_XX__AUDIO_1060_TIE_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Neutral | Misclassified |
| 172 | `FAKE_T3_1046_IWW_DIS_XX__AUDIO_1046_IWW_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 173 | `FAKE_T3_1056_IEO_ANG_LO__AUDIO_1056_IEO_DIS_LO_sadtalker` | Disgust | Fear | Misclassified | Angry | Angry | Correct |
| 174 | `FAKE_T2_1063_ITS_ANG_XX__AUDIO_1063_ITS_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 175 | `FAKE_T3_1020_IWL_ANG_XX__AUDIO_1020_IWL_DIS_XX_sadtalker` | Disgust | Angry | Misclassified | Angry | Angry | Correct |
| 176 | `FAKE_T1_1046_IOM_NEU_XX__AUDIO_1046_IOM_SAD_XX_styletts` | Sad | Sad | Correct | Neutral | Angry | Misclassified |
| 177 | `FAKE_T2_1033_IEO_SAD_HI__AUDIO_1033_IEO_ANG_HI_wav2lip` | Angry | Fear | Misclassified | Sad | Sad | Correct |
| 178 | `FAKE_T3_1071_IEO_FEA_LO__AUDIO_1071_IEO_HAP_LO_sadtalker` | Happy | Neutral | Misclassified | Fear | Fear | Correct |
| 179 | `FAKE_T3_1020_IWW_ANG_XX__AUDIO_1020_IWW_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 180 | `FAKE_T1_1063_IOM_FEA_XX__AUDIO_1063_IOM_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Fear | Correct |
| 181 | `FAKE_T2_1060_WSI_HAP_XX__AUDIO_1060_WSI_NEU_XX_wav2lip` | Neutral | Happy | Misclassified | Happy | Happy | Correct |
| 182 | `FAKE_T3_1068_IOM_NEU_XX__AUDIO_1068_IOM_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 183 | `FAKE_T2_1056_IOM_NEU_XX__AUDIO_1056_IOM_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Happy | Misclassified |
| 184 | `FAKE_T1_1048_IEO_ANG_MD__AUDIO_1048_IEO_DIS_MD_styletts` | Disgust | Disgust | Correct | Angry | Sad | Misclassified |
| 185 | `FAKE_T1_1063_IEO_NEU_XX__AUDIO_1063_IEO_SAD_HI_styletts` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 186 | `FAKE_T3_1020_IEO_ANG_LO__AUDIO_1020_IEO_DIS_LO_sadtalker` | Disgust | Happy | Misclassified | Angry | Angry | Correct |
| 187 | `FAKE_T2_1046_IEO_HAP_HI__AUDIO_1046_IEO_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 188 | `FAKE_T3_1071_DFA_ANG_XX__AUDIO_1071_DFA_DIS_XX_sadtalker` | Disgust | Fear | Misclassified | Angry | Disgust | Misclassified |
| 189 | `FAKE_T2_1060_TIE_NEU_XX__AUDIO_1060_TIE_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 190 | `FAKE_T1_1063_MTI_FEA_XX__AUDIO_1063_MTI_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Fear | Correct |
| 191 | `FAKE_T2_1068_IWW_HAP_XX__AUDIO_1068_IWW_NEU_XX_wav2lip` | Neutral | Disgust | Misclassified | Happy | Disgust | Misclassified |
| 192 | `FAKE_T1_1033_MTI_HAP_XX__AUDIO_1033_MTI_NEU_XX_styletts` | Neutral | Disgust | Misclassified | Happy | Happy | Correct |
| 193 | `FAKE_T3_1046_TIE_SAD_XX__AUDIO_1046_TIE_ANG_XX_sadtalker` | Angry | Happy | Misclassified | Sad | Sad | Correct |
| 194 | `FAKE_T3_1063_IOM_HAP_XX__AUDIO_1063_IOM_NEU_XX_sadtalker` | Neutral | Fear | Misclassified | Happy | Happy | Correct |
| 195 | `FAKE_T2_1060_IOM_FEA_XX__AUDIO_1060_IOM_HAP_XX_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 196 | `FAKE_T3_1056_TAI_ANG_XX__AUDIO_1056_TAI_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 197 | `FAKE_T2_1033_IEO_HAP_HI__AUDIO_1033_IEO_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 198 | `FAKE_T1_1046_IWL_ANG_XX__AUDIO_1046_IWL_DIS_XX_styletts` | Disgust | Disgust | Correct | Angry | Sad | Misclassified |
| 199 | `FAKE_T3_1020_IOM_FEA_XX__AUDIO_1020_IOM_HAP_XX_sadtalker` | Happy | Fear | Misclassified | Fear | Angry | Misclassified |
| 200 | `FAKE_T3_1063_TAI_ANG_XX__AUDIO_1063_TAI_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 201 | `FAKE_T3_1071_IWL_HAP_XX__AUDIO_1071_IWL_NEU_XX_sadtalker` | Neutral | Disgust | Misclassified | Happy | Disgust | Misclassified |
| 202 | `FAKE_T1_1060_IOM_NEU_XX__AUDIO_1060_IOM_SAD_XX_styletts` | Sad | Sad | Correct | Neutral | Sad | Misclassified |
| 203 | `FAKE_T3_1046_TIE_FEA_XX__AUDIO_1046_TIE_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Sad | Misclassified |
| 204 | `FAKE_T2_1068_WSI_FEA_XX__AUDIO_1068_WSI_HAP_XX_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 205 | `FAKE_T1_1063_ITS_DIS_XX__AUDIO_1063_ITS_FEA_XX_styletts` | Fear | Neutral | Misclassified | Disgust | Disgust | Correct |
| 206 | `FAKE_T3_1071_IEO_DIS_HI__AUDIO_1071_IEO_FEA_HI_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 207 | `FAKE_T2_1048_IEO_ANG_HI__AUDIO_1048_IEO_DIS_HI_wav2lip` | Disgust | Disgust | Correct | Angry | Fear | Misclassified |
| 208 | `FAKE_T1_1071_IEO_HAP_HI__AUDIO_1071_IEO_NEU_XX_styletts` | Neutral | Happy | Misclassified | Happy | Happy | Correct |
| 209 | `FAKE_T1_1060_DFA_HAP_XX__AUDIO_1060_DFA_NEU_XX_styletts` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 210 | `FAKE_T2_1071_ITS_HAP_XX__AUDIO_1071_ITS_NEU_XX_wav2lip` | Neutral | Angry | Misclassified | Happy | Happy | Correct |
| 211 | `FAKE_T3_1033_DFA_NEU_XX__AUDIO_1033_DFA_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 212 | `FAKE_T2_1046_IEO_SAD_MD__AUDIO_1046_IEO_ANG_MD_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 213 | `FAKE_T3_1068_ITH_HAP_XX__AUDIO_1068_ITH_NEU_XX_sadtalker` | Neutral | Fear | Misclassified | Happy | Neutral | Misclassified |
| 214 | `FAKE_T3_1033_MTI_FEA_XX__AUDIO_1033_MTI_HAP_XX_sadtalker` | Happy | Neutral | Misclassified | Fear | Fear | Correct |
| 215 | `FAKE_T3_1068_TAI_DIS_XX__AUDIO_1068_TAI_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 216 | `FAKE_T3_1071_IWW_NEU_XX__AUDIO_1071_IWW_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 217 | `FAKE_T1_1033_IOM_NEU_XX__AUDIO_1033_IOM_SAD_XX_styletts` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 218 | `FAKE_T1_1046_IEO_ANG_HI__AUDIO_1046_IEO_DIS_HI_styletts` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 219 | `FAKE_T2_1048_MTI_HAP_XX__AUDIO_1048_MTI_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 220 | `FAKE_T3_1060_ITH_HAP_XX__AUDIO_1060_ITH_NEU_XX_sadtalker` | Neutral | Sad | Misclassified | Happy | Happy | Correct |
| 221 | `FAKE_T3_1071_ITH_HAP_XX__AUDIO_1071_ITH_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 222 | `FAKE_T2_1068_IEO_HAP_MD__AUDIO_1068_IEO_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 223 | `FAKE_T3_1020_IWL_HAP_XX__AUDIO_1020_IWL_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 224 | `FAKE_T2_1046_MTI_HAP_XX__AUDIO_1046_MTI_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 225 | `FAKE_T2_1046_MTI_DIS_XX__AUDIO_1046_MTI_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 226 | `FAKE_T3_1063_IOM_ANG_XX__AUDIO_1063_IOM_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 227 | `FAKE_T2_1068_IOM_DIS_XX__AUDIO_1068_IOM_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 228 | `FAKE_T3_1020_DFA_HAP_XX__AUDIO_1020_DFA_NEU_XX_sadtalker` | Neutral | Angry | Misclassified | Happy | Happy | Correct |
| 229 | `FAKE_T3_1060_TIE_DIS_XX__AUDIO_1060_TIE_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Sad | Misclassified |
| 230 | `FAKE_T1_1068_ITS_DIS_XX__AUDIO_1068_ITS_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 231 | `FAKE_T1_1068_TAI_SAD_XX__AUDIO_1068_TAI_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 232 | `FAKE_T3_1033_IWL_DIS_XX__AUDIO_1033_IWL_FEA_XX_sadtalker` | Fear | Sad | Misclassified | Disgust | Angry | Misclassified |
| 233 | `FAKE_T2_1020_IEO_FEA_MD__AUDIO_1020_IEO_HAP_MD_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 234 | `FAKE_T2_1056_TIE_SAD_XX__AUDIO_1056_TIE_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 235 | `FAKE_T3_1046_DFA_HAP_XX__AUDIO_1046_DFA_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Sad | Misclassified |
| 236 | `FAKE_T2_1071_TSI_SAD_XX__AUDIO_1071_TSI_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 237 | `FAKE_T3_1071_IEO_ANG_LO__AUDIO_1071_IEO_DIS_LO_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 238 | `FAKE_T1_1056_ITS_DIS_XX__AUDIO_1056_ITS_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 239 | `FAKE_T2_1020_WSI_NEU_XX__AUDIO_1020_WSI_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 240 | `FAKE_T3_1048_ITH_SAD_XX__AUDIO_1048_ITH_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Disgust | Misclassified |
| 241 | `FAKE_T2_1020_ITS_NEU_XX__AUDIO_1020_ITS_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 242 | `FAKE_T2_1020_IEO_HAP_MD__AUDIO_1020_IEO_NEU_XX_wav2lip` | Neutral | Fear | Misclassified | Happy | Happy | Correct |
| 243 | `FAKE_T2_1071_TAI_NEU_XX__AUDIO_1071_TAI_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 244 | `FAKE_T3_1063_IOM_DIS_XX__AUDIO_1063_IOM_FEA_XX_sadtalker` | Fear | Sad | Misclassified | Disgust | Disgust | Correct |
| 245 | `FAKE_T1_1046_TIE_NEU_XX__AUDIO_1046_TIE_SAD_XX_styletts` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 246 | `FAKE_T3_1068_ITS_FEA_XX__AUDIO_1068_ITS_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 247 | `FAKE_T3_1056_TSI_DIS_XX__AUDIO_1056_TSI_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Angry | Misclassified |
| 248 | `FAKE_T2_1033_ITS_NEU_XX__AUDIO_1033_ITS_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 249 | `FAKE_T3_1048_IWL_DIS_XX__AUDIO_1048_IWL_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 250 | `FAKE_T2_1071_TAI_FEA_XX__AUDIO_1071_TAI_HAP_XX_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 251 | `FAKE_T1_1033_ITS_DIS_XX__AUDIO_1033_ITS_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 252 | `FAKE_T3_1068_IWL_ANG_XX__AUDIO_1068_IWL_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 253 | `FAKE_T2_1063_TSI_DIS_XX__AUDIO_1063_TSI_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Neutral | Misclassified |
| 254 | `FAKE_T2_1063_IWW_FEA_XX__AUDIO_1063_IWW_HAP_XX_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 255 | `FAKE_T3_1046_TSI_DIS_XX__AUDIO_1046_TSI_FEA_XX_sadtalker` | Fear | Angry | Misclassified | Disgust | Angry | Misclassified |
| 256 | `FAKE_T2_1071_IEO_DIS_LO__AUDIO_1071_IEO_FEA_LO_wav2lip` | Fear | Sad | Misclassified | Disgust | Disgust | Correct |
| 257 | `FAKE_T3_1033_MTI_DIS_XX__AUDIO_1033_MTI_FEA_XX_sadtalker` | Fear | Neutral | Misclassified | Disgust | Disgust | Correct |
| 258 | `FAKE_T3_1020_IEO_ANG_MD__AUDIO_1020_IEO_DIS_MD_sadtalker` | Disgust | Disgust | Correct | Angry | Sad | Misclassified |
| 259 | `FAKE_T3_1033_TIE_ANG_XX__AUDIO_1033_TIE_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 260 | `FAKE_T3_1056_ITH_NEU_XX__AUDIO_1056_ITH_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 261 | `FAKE_T3_1046_ITS_HAP_XX__AUDIO_1046_ITS_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Neutral | Misclassified |
| 262 | `FAKE_T1_1071_ITS_ANG_XX__AUDIO_1071_ITS_DIS_XX_styletts` | Disgust | Neutral | Misclassified | Angry | Neutral | Misclassified |
| 263 | `FAKE_T1_1020_MTI_SAD_XX__AUDIO_1020_MTI_ANG_XX_styletts` | Angry | Sad | Misclassified | Sad | Sad | Correct |
| 264 | `FAKE_T1_1020_ITH_FEA_XX__AUDIO_1020_ITH_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Fear | Correct |
| 265 | `FAKE_T3_1033_DFA_HAP_XX__AUDIO_1033_DFA_NEU_XX_sadtalker` | Neutral | Happy | Misclassified | Happy | Happy | Correct |
| 266 | `FAKE_T3_1056_DFA_HAP_XX__AUDIO_1056_DFA_NEU_XX_sadtalker` | Neutral | Sad | Misclassified | Happy | Happy | Correct |
| 267 | `FAKE_T1_1063_WSI_DIS_XX__AUDIO_1063_WSI_FEA_XX_styletts` | Fear | Happy | Misclassified | Disgust | Disgust | Correct |
| 268 | `FAKE_T2_1068_IEO_FEA_MD__AUDIO_1068_IEO_HAP_MD_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 269 | `FAKE_T2_1020_IWL_DIS_XX__AUDIO_1020_IWL_FEA_XX_wav2lip` | Fear | Disgust | Misclassified | Disgust | Disgust | Correct |
| 270 | `FAKE_T3_1056_ITH_FEA_XX__AUDIO_1056_ITH_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Happy | Misclassified |
| 271 | `FAKE_T3_1048_IEO_DIS_LO__AUDIO_1048_IEO_FEA_LO_sadtalker` | Fear | Fear | Correct | Disgust | Fear | Misclassified |
| 272 | `FAKE_T3_1048_MTI_NEU_XX__AUDIO_1048_MTI_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 273 | `FAKE_T3_1046_ITS_DIS_XX__AUDIO_1046_ITS_FEA_XX_sadtalker` | Fear | Happy | Misclassified | Disgust | Disgust | Correct |
| 274 | `FAKE_T3_1033_IEO_FEA_LO__AUDIO_1033_IEO_HAP_LO_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 275 | `FAKE_T2_1048_IEO_SAD_LO__AUDIO_1048_IEO_ANG_LO_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 276 | `FAKE_T2_1063_TSI_NEU_XX__AUDIO_1063_TSI_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 277 | `FAKE_T3_1060_IEO_DIS_MD__AUDIO_1060_IEO_FEA_MD_sadtalker` | Fear | Fear | Correct | Disgust | Sad | Misclassified |
| 278 | `FAKE_T2_1048_WSI_NEU_XX__AUDIO_1048_WSI_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Disgust | Misclassified |
| 279 | `FAKE_T2_1060_WSI_NEU_XX__AUDIO_1060_WSI_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 280 | `FAKE_T3_1020_IEO_FEA_LO__AUDIO_1020_IEO_HAP_LO_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 281 | `FAKE_T1_1071_IEO_DIS_MD__AUDIO_1071_IEO_FEA_MD_styletts` | Fear | Angry | Misclassified | Disgust | Angry | Misclassified |
| 282 | `FAKE_T3_1056_ITS_ANG_XX__AUDIO_1056_ITS_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 283 | `FAKE_T3_1068_IOM_SAD_XX__AUDIO_1068_IOM_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Happy | Misclassified |
| 284 | `FAKE_T1_1020_IWL_FEA_XX__AUDIO_1020_IWL_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Fear | Correct |
| 285 | `FAKE_T3_1068_TSI_FEA_XX__AUDIO_1068_TSI_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 286 | `FAKE_T1_1068_WSI_ANG_XX__AUDIO_1068_WSI_DIS_XX_styletts` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 287 | `FAKE_T3_1033_ITH_FEA_XX__AUDIO_1033_ITH_HAP_XX_sadtalker` | Happy | Fear | Misclassified | Fear | Happy | Misclassified |
| 288 | `FAKE_T2_1056_TIE_HAP_XX__AUDIO_1056_TIE_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Sad | Misclassified |
| 289 | `FAKE_T3_1071_TIE_FEA_XX__AUDIO_1071_TIE_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 290 | `FAKE_T1_1046_IEO_HAP_LO__AUDIO_1046_IEO_NEU_XX_styletts` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 291 | `FAKE_T2_1056_IWW_DIS_XX__AUDIO_1056_IWW_FEA_XX_wav2lip` | Fear | Angry | Misclassified | Disgust | Disgust | Correct |
| 292 | `FAKE_T3_1046_TSI_HAP_XX__AUDIO_1046_TSI_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 293 | `FAKE_T3_1048_IOM_NEU_XX__AUDIO_1048_IOM_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 294 | `FAKE_T3_1020_WSI_ANG_XX__AUDIO_1020_WSI_DIS_XX_sadtalker` | Disgust | Angry | Misclassified | Angry | Angry | Correct |
| 295 | `FAKE_T3_1048_TIE_NEU_XX__AUDIO_1048_TIE_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 296 | `FAKE_T2_1060_IWW_SAD_XX__AUDIO_1060_IWW_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 297 | `FAKE_T2_1056_TSI_NEU_XX__AUDIO_1056_TSI_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 298 | `FAKE_T2_1048_IWW_NEU_XX__AUDIO_1048_IWW_SAD_XX_wav2lip` | Sad | Neutral | Misclassified | Neutral | Neutral | Correct |
| 299 | `FAKE_T2_1033_IWW_SAD_XX__AUDIO_1033_IWW_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Disgust | Misclassified |
| 300 | `FAKE_T3_1046_IWW_HAP_XX__AUDIO_1046_IWW_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 301 | `FAKE_T3_1048_ITS_HAP_XX__AUDIO_1048_ITS_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Sad | Misclassified |
| 302 | `FAKE_T3_1020_IEO_DIS_HI__AUDIO_1020_IEO_FEA_HI_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 303 | `FAKE_T3_1046_TIE_DIS_XX__AUDIO_1046_TIE_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 304 | `FAKE_T3_1056_IEO_HAP_LO__AUDIO_1056_IEO_NEU_XX_sadtalker` | Neutral | Fear | Misclassified | Happy | Happy | Correct |
| 305 | `FAKE_T3_1063_IWW_NEU_XX__AUDIO_1063_IWW_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Disgust | Misclassified |
| 306 | `FAKE_T3_1033_ITH_SAD_XX__AUDIO_1033_ITH_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 307 | `FAKE_T2_1046_MTI_NEU_XX__AUDIO_1046_MTI_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 308 | `FAKE_T3_1060_IEO_ANG_MD__AUDIO_1060_IEO_DIS_MD_sadtalker` | Disgust | Angry | Misclassified | Angry | Angry | Correct |
| 309 | `FAKE_T1_1071_ITS_SAD_XX__AUDIO_1071_ITS_ANG_XX_styletts` | Angry | Fear | Misclassified | Sad | Sad | Correct |
| 310 | `FAKE_T1_1063_TIE_SAD_XX__AUDIO_1063_TIE_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Neutral | Misclassified |
| 311 | `FAKE_T3_1048_TSI_SAD_XX__AUDIO_1048_TSI_ANG_XX_sadtalker` | Angry | Happy | Misclassified | Sad | Sad | Correct |
| 312 | `FAKE_T3_1068_IEO_DIS_MD__AUDIO_1068_IEO_FEA_MD_sadtalker` | Fear | Happy | Misclassified | Disgust | Disgust | Correct |
| 313 | `FAKE_T2_1020_ITH_HAP_XX__AUDIO_1020_ITH_NEU_XX_wav2lip` | Neutral | Angry | Misclassified | Happy | Happy | Correct |
| 314 | `FAKE_T2_1060_MTI_SAD_XX__AUDIO_1060_MTI_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 315 | `FAKE_T2_1048_IOM_SAD_XX__AUDIO_1048_IOM_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Neutral | Misclassified |
| 316 | `FAKE_T2_1056_IOM_FEA_XX__AUDIO_1056_IOM_HAP_XX_wav2lip` | Happy | Disgust | Misclassified | Fear | Fear | Correct |
| 317 | `FAKE_T2_1071_IWL_ANG_XX__AUDIO_1071_IWL_DIS_XX_wav2lip` | Disgust | Neutral | Misclassified | Angry | Angry | Correct |
| 318 | `FAKE_T2_1033_DFA_FEA_XX__AUDIO_1033_DFA_HAP_XX_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 319 | `FAKE_T3_1048_ITH_ANG_XX__AUDIO_1048_ITH_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Disgust | Misclassified |
| 320 | `FAKE_T2_1071_TAI_SAD_XX__AUDIO_1071_TAI_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 321 | `FAKE_T2_1060_ITH_DIS_XX__AUDIO_1060_ITH_FEA_XX_wav2lip` | Fear | Happy | Misclassified | Disgust | Sad | Misclassified |
| 322 | `FAKE_T1_1020_TIE_HAP_XX__AUDIO_1020_TIE_NEU_XX_styletts` | Neutral | Angry | Misclassified | Happy | Happy | Correct |
| 323 | `FAKE_T2_1056_IEO_FEA_LO__AUDIO_1056_IEO_HAP_LO_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 324 | `FAKE_T1_1068_ITS_ANG_XX__AUDIO_1068_ITS_DIS_XX_styletts` | Disgust | Disgust | Correct | Angry | Happy | Misclassified |
| 325 | `FAKE_T3_1071_IWW_FEA_XX__AUDIO_1071_IWW_HAP_XX_sadtalker` | Happy | Fear | Misclassified | Fear | Fear | Correct |
| 326 | `FAKE_T2_1046_ITH_FEA_XX__AUDIO_1046_ITH_HAP_XX_wav2lip` | Happy | Neutral | Misclassified | Fear | Fear | Correct |
| 327 | `FAKE_T3_1033_WSI_HAP_XX__AUDIO_1033_WSI_NEU_XX_sadtalker` | Neutral | Angry | Misclassified | Happy | Happy | Correct |
| 328 | `FAKE_T1_1071_DFA_SAD_XX__AUDIO_1071_DFA_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 329 | `FAKE_T2_1048_ITS_SAD_XX__AUDIO_1048_ITS_ANG_XX_wav2lip` | Angry | Fear | Misclassified | Sad | Sad | Correct |
| 330 | `FAKE_T3_1068_MTI_NEU_XX__AUDIO_1068_MTI_SAD_XX_sadtalker` | Sad | Disgust | Misclassified | Neutral | Neutral | Correct |
| 331 | `FAKE_T3_1071_IOM_ANG_XX__AUDIO_1071_IOM_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 332 | `FAKE_T1_1046_IEO_FEA_LO__AUDIO_1046_IEO_HAP_LO_styletts` | Happy | Happy | Correct | Fear | Fear | Correct |
| 333 | `FAKE_T3_1071_ITH_DIS_XX__AUDIO_1071_ITH_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 334 | `FAKE_T3_1020_TAI_FEA_XX__AUDIO_1020_TAI_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 335 | `FAKE_T3_1060_IEO_SAD_HI__AUDIO_1060_IEO_ANG_HI_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 336 | `FAKE_T3_1048_IWW_ANG_XX__AUDIO_1048_IWW_DIS_XX_sadtalker` | Disgust | Sad | Misclassified | Angry | Angry | Correct |
| 337 | `FAKE_T3_1033_TIE_NEU_XX__AUDIO_1033_TIE_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Disgust | Misclassified |
| 338 | `FAKE_T2_1020_IEO_HAP_HI__AUDIO_1020_IEO_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 339 | `FAKE_T3_1068_ITH_FEA_XX__AUDIO_1068_ITH_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 340 | `FAKE_T3_1056_WSI_HAP_XX__AUDIO_1056_WSI_NEU_XX_sadtalker` | Neutral | Fear | Misclassified | Happy | Happy | Correct |
| 341 | `FAKE_T1_1071_TSI_HAP_XX__AUDIO_1071_TSI_NEU_XX_styletts` | Neutral | Neutral | Correct | Happy | Disgust | Misclassified |
| 342 | `FAKE_T2_1033_DFA_DIS_XX__AUDIO_1033_DFA_FEA_XX_wav2lip` | Fear | Angry | Misclassified | Disgust | Fear | Misclassified |
| 343 | `FAKE_T2_1068_TIE_FEA_XX__AUDIO_1068_TIE_HAP_XX_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 344 | `FAKE_T2_1020_IWW_DIS_XX__AUDIO_1020_IWW_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 345 | `FAKE_T3_1048_IOM_ANG_XX__AUDIO_1048_IOM_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 346 | `FAKE_T3_1020_IOM_NEU_XX__AUDIO_1020_IOM_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 347 | `FAKE_T1_1056_TSI_FEA_XX__AUDIO_1056_TSI_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Fear | Correct |
| 348 | `FAKE_T1_1056_IEO_DIS_HI__AUDIO_1056_IEO_FEA_HI_styletts` | Fear | Neutral | Misclassified | Disgust | Sad | Misclassified |
| 349 | `FAKE_T2_1068_IWW_FEA_XX__AUDIO_1068_IWW_HAP_XX_wav2lip` | Happy | Happy | Correct | Fear | Sad | Misclassified |
| 350 | `FAKE_T1_1020_TAI_DIS_XX__AUDIO_1020_TAI_FEA_XX_styletts` | Fear | Angry | Misclassified | Disgust | Fear | Misclassified |
| 351 | `FAKE_T3_1071_IEO_FEA_HI__AUDIO_1071_IEO_HAP_HI_sadtalker` | Happy | Sad | Misclassified | Fear | Fear | Correct |
| 352 | `FAKE_T1_1068_IWL_DIS_XX__AUDIO_1068_IWL_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Sad | Misclassified |
| 353 | `FAKE_T3_1046_IEO_ANG_LO__AUDIO_1046_IEO_DIS_LO_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 354 | `FAKE_T3_1033_DFA_SAD_XX__AUDIO_1033_DFA_ANG_XX_sadtalker` | Angry | Disgust | Misclassified | Sad | Sad | Correct |
| 355 | `FAKE_T3_1056_IEO_HAP_HI__AUDIO_1056_IEO_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 356 | `FAKE_T3_1020_IWW_FEA_XX__AUDIO_1020_IWW_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Disgust | Misclassified |
| 357 | `FAKE_T3_1060_TIE_HAP_XX__AUDIO_1060_TIE_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 358 | `FAKE_T3_1071_IWL_FEA_XX__AUDIO_1071_IWL_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 359 | `FAKE_T3_1056_MTI_NEU_XX__AUDIO_1056_MTI_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 360 | `FAKE_T3_1020_ITS_SAD_XX__AUDIO_1020_ITS_ANG_XX_sadtalker` | Angry | Neutral | Misclassified | Sad | Sad | Correct |
| 361 | `FAKE_T3_1060_IWL_FEA_XX__AUDIO_1060_IWL_HAP_XX_sadtalker` | Happy | Sad | Misclassified | Fear | Fear | Correct |
| 362 | `FAKE_T3_1060_MTI_DIS_XX__AUDIO_1060_MTI_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 363 | `FAKE_T1_1056_ITS_HAP_XX__AUDIO_1056_ITS_NEU_XX` | Neutral | Angry | Misclassified | Happy | Happy | Correct |
| 364 | `FAKE_T3_1020_DFA_NEU_XX__AUDIO_1020_DFA_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 365 | `FAKE_T3_1063_TIE_ANG_XX__AUDIO_1063_TIE_DIS_XX_sadtalker` | Disgust | Neutral | Misclassified | Angry | Angry | Correct |
| 366 | `FAKE_T3_1071_IEO_ANG_MD__AUDIO_1071_IEO_DIS_MD_sadtalker` | Disgust | Happy | Misclassified | Angry | Angry | Correct |
| 367 | `FAKE_T3_1048_MTI_FEA_XX__AUDIO_1048_MTI_HAP_XX_sadtalker` | Happy | Angry | Misclassified | Fear | Fear | Correct |
| 368 | `FAKE_T3_1068_TSI_SAD_XX__AUDIO_1068_TSI_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 369 | `FAKE_T3_1056_ITS_FEA_XX__AUDIO_1056_ITS_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 370 | `FAKE_T3_1046_IOM_DIS_XX__AUDIO_1046_IOM_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 371 | `FAKE_T3_1048_TSI_ANG_XX__AUDIO_1048_TSI_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 372 | `FAKE_T3_1048_ITS_ANG_XX__AUDIO_1048_ITS_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Neutral | Misclassified |
| 373 | `FAKE_T3_1063_IWL_ANG_XX__AUDIO_1063_IWL_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 374 | `FAKE_T2_1046_ITS_SAD_XX__AUDIO_1046_ITS_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 375 | `FAKE_T3_1048_IEO_FEA_MD__AUDIO_1048_IEO_HAP_MD_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 376 | `FAKE_T1_1048_TAI_HAP_XX__AUDIO_1048_TAI_NEU_XX_styletts` | Neutral | Sad | Misclassified | Happy | Happy | Correct |
| 377 | `FAKE_T2_1046_IOM_SAD_XX__AUDIO_1046_IOM_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Angry | Misclassified |
| 378 | `FAKE_T3_1060_ITS_HAP_XX__AUDIO_1060_ITS_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 379 | `FAKE_T2_1060_IEO_ANG_LO__AUDIO_1060_IEO_DIS_LO_wav2lip` | Disgust | Happy | Misclassified | Angry | Disgust | Misclassified |
| 380 | `FAKE_T2_1020_ITS_DIS_XX__AUDIO_1020_ITS_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 381 | `FAKE_T3_1063_IEO_HAP_HI__AUDIO_1063_IEO_NEU_XX_sadtalker` | Neutral | Fear | Misclassified | Happy | Happy | Correct |
| 382 | `FAKE_T2_1048_ITH_NEU_XX__AUDIO_1048_ITH_SAD_XX_wav2lip` | Sad | Fear | Misclassified | Neutral | Disgust | Misclassified |
| 383 | `FAKE_T1_1060_TAI_FEA_XX__AUDIO_1060_TAI_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Fear | Correct |
| 384 | `FAKE_T1_1048_ITS_DIS_XX__AUDIO_1048_ITS_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 385 | `FAKE_T2_1060_IOM_SAD_XX__AUDIO_1060_IOM_ANG_XX_wav2lip` | Angry | Happy | Misclassified | Sad | Sad | Correct |
| 386 | `FAKE_T2_1056_MTI_DIS_XX__AUDIO_1056_MTI_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Fear | Misclassified |
| 387 | `FAKE_T3_1020_MTI_DIS_XX__AUDIO_1020_MTI_FEA_XX_sadtalker` | Fear | Angry | Misclassified | Disgust | Disgust | Correct |
| 388 | `FAKE_T3_1071_IEO_SAD_MD__AUDIO_1071_IEO_ANG_MD_sadtalker` | Angry | Angry | Correct | Sad | Fear | Misclassified |
| 389 | `FAKE_T2_1071_IWW_HAP_XX__AUDIO_1071_IWW_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 390 | `FAKE_T1_1068_DFA_HAP_XX__AUDIO_1068_DFA_NEU_XX_styletts` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 391 | `FAKE_T2_1056_IOM_SAD_XX__AUDIO_1056_IOM_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 392 | `FAKE_T2_1048_IEO_HAP_HI__AUDIO_1048_IEO_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 393 | `FAKE_T2_1020_TIE_DIS_XX__AUDIO_1020_TIE_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 394 | `FAKE_T1_1033_IWL_SAD_XX__AUDIO_1033_IWL_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 395 | `FAKE_T3_1063_IEO_FEA_LO__AUDIO_1063_IEO_HAP_LO_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 396 | `FAKE_T3_1048_IEO_SAD_MD__AUDIO_1048_IEO_ANG_MD_sadtalker` | Angry | Happy | Misclassified | Sad | Sad | Correct |
| 397 | `FAKE_T3_1046_TAI_NEU_XX__AUDIO_1046_TAI_SAD_XX_sadtalker` | Sad | Fear | Misclassified | Neutral | Neutral | Correct |
| 398 | `FAKE_T3_1060_IEO_HAP_LO__AUDIO_1060_IEO_NEU_XX_sadtalker` | Neutral | Disgust | Misclassified | Happy | Sad | Misclassified |
| 399 | `FAKE_T2_1063_TIE_HAP_XX__AUDIO_1063_TIE_NEU_XX_wav2lip` | Neutral | Disgust | Misclassified | Happy | Happy | Correct |
| 400 | `FAKE_T3_1068_IEO_DIS_LO__AUDIO_1068_IEO_FEA_LO_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 401 | `FAKE_T1_1033_TAI_DIS_XX__AUDIO_1033_TAI_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 402 | `FAKE_T3_1060_TIE_SAD_XX__AUDIO_1060_TIE_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 403 | `FAKE_T2_1048_IEO_DIS_MD__AUDIO_1048_IEO_FEA_MD_wav2lip` | Fear | Happy | Misclassified | Disgust | Disgust | Correct |
| 404 | `FAKE_T2_1068_WSI_HAP_XX__AUDIO_1068_WSI_NEU_XX_wav2lip` | Neutral | Fear | Misclassified | Happy | Happy | Correct |
| 405 | `FAKE_T3_1033_MTI_ANG_XX__AUDIO_1033_MTI_DIS_XX_sadtalker` | Disgust | Sad | Misclassified | Angry | Happy | Misclassified |
| 406 | `FAKE_T3_1046_TAI_FEA_XX__AUDIO_1046_TAI_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 407 | `FAKE_T3_1063_DFA_FEA_XX__AUDIO_1063_DFA_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 408 | `FAKE_T1_1060_TAI_ANG_XX__AUDIO_1060_TAI_DIS_XX_styletts` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 409 | `FAKE_T1_1068_WSI_SAD_XX__AUDIO_1068_WSI_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 410 | `FAKE_T1_1071_IEO_NEU_XX__AUDIO_1071_IEO_SAD_HI_styletts` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 411 | `FAKE_T3_1071_DFA_FEA_XX__AUDIO_1071_DFA_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Happy | Misclassified |
| 412 | `FAKE_T3_1071_ITS_FEA_XX__AUDIO_1071_ITS_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 413 | `FAKE_T3_1048_DFA_HAP_XX__AUDIO_1048_DFA_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 414 | `FAKE_T1_1068_WSI_DIS_XX__AUDIO_1068_WSI_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 415 | `FAKE_T2_1063_DFA_ANG_XX__AUDIO_1063_DFA_DIS_XX_wav2lip` | Disgust | Sad | Misclassified | Angry | Angry | Correct |
| 416 | `FAKE_T2_1048_TIE_DIS_XX__AUDIO_1048_TIE_FEA_XX_wav2lip` | Fear | Angry | Misclassified | Disgust | Disgust | Correct |
| 417 | `FAKE_T1_1063_WSI_ANG_XX__AUDIO_1063_WSI_DIS_XX_styletts` | Disgust | Fear | Misclassified | Angry | Neutral | Misclassified |
| 418 | `FAKE_T2_1068_TAI_FEA_XX__AUDIO_1068_TAI_HAP_XX_wav2lip` | Happy | Angry | Misclassified | Fear | Fear | Correct |
| 419 | `FAKE_T2_1060_TAI_NEU_XX__AUDIO_1060_TAI_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 420 | `FAKE_T3_1020_IOM_DIS_XX__AUDIO_1020_IOM_FEA_XX_sadtalker` | Fear | Neutral | Misclassified | Disgust | Disgust | Correct |
| 421 | `FAKE_T3_1060_TIE_ANG_XX__AUDIO_1060_TIE_DIS_XX_sadtalker` | Disgust | Angry | Misclassified | Angry | Angry | Correct |
| 422 | `FAKE_T1_1068_IWW_NEU_XX__AUDIO_1068_IWW_SAD_XX_styletts` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 423 | `FAKE_T3_1060_MTI_FEA_XX__AUDIO_1060_MTI_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 424 | `FAKE_T3_1046_WSI_HAP_XX__AUDIO_1046_WSI_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Sad | Misclassified |
| 425 | `FAKE_T3_1056_WSI_NEU_XX__AUDIO_1056_WSI_SAD_XX_sadtalker` | Sad | Happy | Misclassified | Neutral | Neutral | Correct |
| 426 | `FAKE_T3_1020_TIE_NEU_XX__AUDIO_1020_TIE_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 427 | `FAKE_T1_1033_ITH_ANG_XX__AUDIO_1033_ITH_DIS_XX_styletts` | Disgust | Angry | Misclassified | Angry | Angry | Correct |
| 428 | `FAKE_T2_1060_ITS_SAD_XX__AUDIO_1060_ITS_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Disgust | Misclassified |
| 429 | `FAKE_T3_1048_IEO_FEA_LO__AUDIO_1048_IEO_HAP_LO_sadtalker` | Happy | Happy | Correct | Fear | Disgust | Misclassified |
| 430 | `FAKE_T1_1033_TAI_SAD_XX__AUDIO_1033_TAI_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 431 | `FAKE_T2_1071_MTI_DIS_XX__AUDIO_1071_MTI_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 432 | `FAKE_T2_1033_IOM_HAP_XX__AUDIO_1033_IOM_NEU_XX_wav2lip` | Neutral | Fear | Misclassified | Happy | Happy | Correct |
| 433 | `FAKE_T3_1060_MTI_ANG_XX__AUDIO_1060_MTI_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 434 | `FAKE_T2_1046_IEO_FEA_MD__AUDIO_1046_IEO_HAP_MD_wav2lip` | Happy | Happy | Correct | Fear | Angry | Misclassified |
| 435 | `FAKE_T1_1020_IEO_DIS_LO__AUDIO_1020_IEO_FEA_LO_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 436 | `FAKE_T3_1056_MTI_FEA_XX__AUDIO_1056_MTI_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 437 | `FAKE_T2_1033_IWL_NEU_XX__AUDIO_1033_IWL_SAD_XX_wav2lip` | Sad | Fear | Misclassified | Neutral | Neutral | Correct |
| 438 | `FAKE_T1_1033_IOM_DIS_XX__AUDIO_1033_IOM_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Angry | Misclassified |
| 439 | `FAKE_T3_1071_DFA_NEU_XX__AUDIO_1071_DFA_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 440 | `FAKE_T3_1020_TSI_HAP_XX__AUDIO_1020_TSI_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 441 | `FAKE_T1_1020_TAI_ANG_XX__AUDIO_1020_TAI_DIS_XX_styletts` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 442 | `FAKE_T3_1063_TIE_NEU_XX__AUDIO_1063_TIE_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 443 | `FAKE_T2_1060_IWL_ANG_XX__AUDIO_1060_IWL_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 444 | `FAKE_T1_1048_TSI_FEA_XX__AUDIO_1048_TSI_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Angry | Misclassified |
| 445 | `FAKE_T3_1071_ITS_NEU_XX__AUDIO_1071_ITS_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Fear | Misclassified |
| 446 | `FAKE_T2_1048_TAI_NEU_XX__AUDIO_1048_TAI_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Sad | Misclassified |
| 447 | `FAKE_T3_1071_WSI_ANG_XX__AUDIO_1071_WSI_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 448 | `FAKE_T1_1046_DFA_SAD_XX__AUDIO_1046_DFA_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 449 | `FAKE_T2_1046_MTI_FEA_XX__AUDIO_1046_MTI_HAP_XX_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 450 | `FAKE_T2_1063_IOM_SAD_XX__AUDIO_1063_IOM_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Neutral | Misclassified |
| 451 | `FAKE_T1_1071_TSI_DIS_XX__AUDIO_1071_TSI_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 452 | `FAKE_T2_1063_ITH_HAP_XX__AUDIO_1063_ITH_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 453 | `FAKE_T1_1056_IWL_DIS_XX__AUDIO_1056_IWL_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 454 | `FAKE_T3_1063_ITH_NEU_XX__AUDIO_1063_ITH_SAD_XX_sadtalker` | Sad | Disgust | Misclassified | Neutral | Neutral | Correct |
| 455 | `FAKE_T2_1063_TSI_FEA_XX__AUDIO_1063_TSI_HAP_XX_wav2lip` | Happy | Neutral | Misclassified | Fear | Fear | Correct |
| 456 | `FAKE_T2_1048_TSI_HAP_XX__AUDIO_1048_TSI_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Sad | Misclassified |
| 457 | `FAKE_T3_1033_IEO_HAP_LO__AUDIO_1033_IEO_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 458 | `FAKE_T3_1046_WSI_ANG_XX__AUDIO_1046_WSI_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 459 | `FAKE_T1_1046_ITH_ANG_XX__AUDIO_1046_ITH_DIS_XX_styletts` | Disgust | Sad | Misclassified | Angry | Angry | Correct |
| 460 | `FAKE_T1_1060_IOM_DIS_XX__AUDIO_1060_IOM_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 461 | `FAKE_T2_1056_ITS_NEU_XX__AUDIO_1056_ITS_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 462 | `FAKE_T3_1071_IWL_SAD_XX__AUDIO_1071_IWL_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Neutral | Misclassified |
| 463 | `FAKE_T3_1046_ITS_FEA_XX__AUDIO_1046_ITS_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Sad | Misclassified |
| 464 | `FAKE_T3_1056_IWL_ANG_XX__AUDIO_1056_IWL_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Happy | Misclassified |
| 465 | `FAKE_T3_1020_ITH_SAD_XX__AUDIO_1020_ITH_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 466 | `FAKE_T1_1068_MTI_SAD_XX__AUDIO_1068_MTI_ANG_XX_styletts` | Angry | Disgust | Misclassified | Sad | Sad | Correct |
| 467 | `FAKE_T3_1071_MTI_HAP_XX__AUDIO_1071_MTI_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 468 | `FAKE_T1_1020_TAI_SAD_XX__AUDIO_1020_TAI_ANG_XX_styletts` | Angry | Sad | Misclassified | Sad | Sad | Correct |
| 469 | `FAKE_T2_1063_IWL_SAD_XX__AUDIO_1063_IWL_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 470 | `FAKE_T1_1020_IOM_SAD_XX__AUDIO_1020_IOM_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 471 | `FAKE_T3_1020_IWL_NEU_XX__AUDIO_1020_IWL_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Sad | Misclassified |
| 472 | `FAKE_T3_1056_IEO_HAP_MD__AUDIO_1056_IEO_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 473 | `FAKE_T3_1048_DFA_NEU_XX__AUDIO_1048_DFA_SAD_XX_sadtalker` | Sad | Fear | Misclassified | Neutral | Neutral | Correct |
| 474 | `FAKE_T3_1071_TAI_HAP_XX__AUDIO_1071_TAI_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Disgust | Misclassified |
| 475 | `FAKE_T2_1033_IEO_ANG_HI__AUDIO_1033_IEO_DIS_HI_wav2lip` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 476 | `FAKE_T3_1056_IWL_SAD_XX__AUDIO_1056_IWL_ANG_XX_sadtalker` | Angry | Fear | Misclassified | Sad | Sad | Correct |
| 477 | `FAKE_T3_1020_IEO_ANG_HI__AUDIO_1020_IEO_DIS_HI_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 478 | `FAKE_T2_1063_IEO_SAD_HI__AUDIO_1063_IEO_ANG_HI_wav2lip` | Angry | Disgust | Misclassified | Sad | Sad | Correct |
| 479 | `FAKE_T1_1056_IWW_SAD_XX__AUDIO_1056_IWW_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 480 | `FAKE_T3_1060_IOM_HAP_XX__AUDIO_1060_IOM_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 481 | `FAKE_T3_1068_ITS_SAD_XX__AUDIO_1068_ITS_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 482 | `FAKE_T3_1033_IOM_ANG_XX__AUDIO_1033_IOM_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 483 | `FAKE_T1_1046_IOM_HAP_XX__AUDIO_1046_IOM_NEU_XX_styletts` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 484 | `FAKE_T2_1046_IWL_HAP_XX__AUDIO_1046_IWL_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 485 | `FAKE_T2_1060_WSI_FEA_XX__AUDIO_1060_WSI_HAP_XX_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 486 | `FAKE_T1_1060_WSI_DIS_XX__AUDIO_1060_WSI_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 487 | `FAKE_T2_1068_ITH_NEU_XX__AUDIO_1068_ITH_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 488 | `FAKE_T3_1071_IOM_FEA_XX__AUDIO_1071_IOM_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Disgust | Misclassified |
| 489 | `FAKE_T3_1068_IWW_ANG_XX__AUDIO_1068_IWW_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 490 | `FAKE_T3_1048_IEO_DIS_HI__AUDIO_1048_IEO_FEA_HI_sadtalker` | Fear | Fear | Correct | Disgust | Neutral | Misclassified |
| 491 | `FAKE_T3_1063_ITS_NEU_XX__AUDIO_1063_ITS_SAD_XX_sadtalker` | Sad | Disgust | Misclassified | Neutral | Neutral | Correct |
| 492 | `FAKE_T1_1071_ITH_SAD_XX__AUDIO_1071_ITH_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Neutral | Misclassified |
| 493 | `FAKE_T3_1063_TAI_NEU_XX__AUDIO_1063_TAI_SAD_XX_sadtalker` | Sad | Angry | Misclassified | Neutral | Neutral | Correct |
| 494 | `FAKE_T3_1020_IOM_ANG_XX__AUDIO_1020_IOM_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 495 | `FAKE_T3_1068_ITH_DIS_XX__AUDIO_1068_ITH_FEA_XX_sadtalker` | Fear | Disgust | Misclassified | Disgust | Disgust | Correct |
| 496 | `FAKE_T3_1056_DFA_SAD_XX__AUDIO_1056_DFA_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Angry | Misclassified |
| 497 | `FAKE_T3_1048_MTI_SAD_XX__AUDIO_1048_MTI_ANG_XX_sadtalker` | Angry | Disgust | Misclassified | Sad | Sad | Correct |
| 498 | `FAKE_T2_1071_TSI_ANG_XX__AUDIO_1071_TSI_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 499 | `FAKE_T3_1071_IEO_SAD_HI__AUDIO_1071_IEO_ANG_HI_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 500 | `FAKE_T3_1048_TAI_FEA_XX__AUDIO_1048_TAI_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 501 | `FAKE_T3_1060_IWL_SAD_XX__AUDIO_1060_IWL_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 502 | `FAKE_T1_1063_WSI_HAP_XX__AUDIO_1063_WSI_NEU_XX_styletts` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 503 | `FAKE_T2_1033_IWW_ANG_XX__AUDIO_1033_IWW_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Neutral | Misclassified |
| 504 | `FAKE_T3_1060_IWL_HAP_XX__AUDIO_1060_IWL_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 505 | `FAKE_T3_1033_IOM_SAD_XX__AUDIO_1033_IOM_ANG_XX_sadtalker` | Angry | Fear | Misclassified | Sad | Sad | Correct |
| 506 | `FAKE_T1_1068_IEO_SAD_LO__AUDIO_1068_IEO_ANG_LO_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 507 | `FAKE_T1_1020_IWW_SAD_XX__AUDIO_1020_IWW_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 508 | `FAKE_T3_1063_IEO_SAD_LO__AUDIO_1063_IEO_ANG_LO_sadtalker` | Angry | Disgust | Misclassified | Sad | Happy | Misclassified |
| 509 | `FAKE_T3_1020_IEO_HAP_LO__AUDIO_1020_IEO_NEU_XX_sadtalker` | Neutral | Fear | Misclassified | Happy | Happy | Correct |
| 510 | `FAKE_T1_1020_ITH_DIS_XX__AUDIO_1020_ITH_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Neutral | Misclassified |
| 511 | `FAKE_T3_1056_IEO_SAD_MD__AUDIO_1056_IEO_ANG_MD_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 512 | `FAKE_T2_1046_IEO_SAD_LO__AUDIO_1046_IEO_ANG_LO_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 513 | `FAKE_T3_1063_DFA_HAP_XX__AUDIO_1063_DFA_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Sad | Misclassified |
| 514 | `FAKE_T3_1063_IEO_DIS_LO__AUDIO_1063_IEO_FEA_LO_sadtalker` | Fear | Sad | Misclassified | Disgust | Neutral | Misclassified |
| 515 | `FAKE_T3_1068_IEO_SAD_MD__AUDIO_1068_IEO_ANG_MD_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 516 | `FAKE_T3_1060_DFA_SAD_XX__AUDIO_1060_DFA_ANG_XX_sadtalker` | Angry | Fear | Misclassified | Sad | Sad | Correct |
| 517 | `FAKE_T2_1056_ITS_SAD_XX__AUDIO_1056_ITS_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 518 | `FAKE_T3_1071_TAI_ANG_XX__AUDIO_1071_TAI_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 519 | `FAKE_T2_1060_IEO_DIS_HI__AUDIO_1060_IEO_FEA_HI_wav2lip` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 520 | `FAKE_T1_1033_IWW_NEU_XX__AUDIO_1033_IWW_SAD_XX_styletts` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 521 | `FAKE_T3_1020_IEO_FEA_HI__AUDIO_1020_IEO_HAP_HI_sadtalker` | Happy | Angry | Misclassified | Fear | Fear | Correct |
| 522 | `FAKE_T1_1048_IWW_SAD_XX__AUDIO_1048_IWW_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 523 | `FAKE_T3_1060_IWW_NEU_XX__AUDIO_1060_IWW_SAD_XX_sadtalker` | Sad | Angry | Misclassified | Neutral | Happy | Misclassified |
| 524 | `FAKE_T3_1020_IEO_SAD_HI__AUDIO_1020_IEO_ANG_HI_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 525 | `FAKE_T3_1033_IEO_HAP_MD__AUDIO_1033_IEO_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 526 | `FAKE_T3_1056_IWW_NEU_XX__AUDIO_1056_IWW_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 527 | `FAKE_T3_1033_TAI_FEA_XX__AUDIO_1033_TAI_HAP_XX_sadtalker` | Happy | Fear | Misclassified | Fear | Neutral | Misclassified |
| 528 | `FAKE_T3_1060_IEO_FEA_HI__AUDIO_1060_IEO_HAP_HI_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 529 | `FAKE_T3_1048_MTI_DIS_XX__AUDIO_1048_MTI_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 530 | `FAKE_T3_1048_TAI_SAD_XX__AUDIO_1048_TAI_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Happy | Misclassified |
| 531 | `FAKE_T1_1060_ITS_FEA_XX__AUDIO_1060_ITS_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Fear | Correct |
| 532 | `FAKE_T3_1046_DFA_NEU_XX__AUDIO_1046_DFA_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 533 | `FAKE_T1_1071_TSI_NEU_XX__AUDIO_1071_TSI_SAD_XX_styletts` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 534 | `FAKE_T1_1020_ITS_FEA_XX__AUDIO_1020_ITS_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Fear | Correct |
| 535 | `FAKE_T1_1068_DFA_SAD_XX__AUDIO_1068_DFA_ANG_XX_styletts` | Angry | Happy | Misclassified | Sad | Sad | Correct |
| 536 | `FAKE_T3_1060_WSI_SAD_XX__AUDIO_1060_WSI_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 537 | `FAKE_T2_1046_TSI_ANG_XX__AUDIO_1046_TSI_DIS_XX_wav2lip` | Disgust | Fear | Misclassified | Angry | Angry | Correct |
| 538 | `FAKE_T1_1046_DFA_DIS_XX__AUDIO_1046_DFA_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 539 | `FAKE_T2_1068_TAI_ANG_XX__AUDIO_1068_TAI_DIS_XX_wav2lip` | Disgust | Sad | Misclassified | Angry | Angry | Correct |
| 540 | `FAKE_T2_1020_ITH_NEU_XX__AUDIO_1020_ITH_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Disgust | Misclassified |
| 541 | `FAKE_T3_1033_WSI_ANG_XX__AUDIO_1033_WSI_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Fear | Misclassified |
| 542 | `FAKE_T2_1046_IWL_FEA_XX__AUDIO_1046_IWL_HAP_XX_wav2lip` | Happy | Happy | Correct | Fear | Happy | Misclassified |
| 543 | `FAKE_T3_1063_IWW_SAD_XX__AUDIO_1063_IWW_ANG_XX_sadtalker` | Angry | Disgust | Misclassified | Sad | Sad | Correct |
| 544 | `FAKE_T1_1033_TSI_DIS_XX__AUDIO_1033_TSI_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 545 | `FAKE_T2_1020_WSI_FEA_XX__AUDIO_1020_WSI_HAP_XX_wav2lip` | Happy | Sad | Misclassified | Fear | Fear | Correct |
| 546 | `FAKE_T2_1071_IEO_HAP_MD__AUDIO_1071_IEO_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Sad | Misclassified |
| 547 | `FAKE_T3_1033_TIE_HAP_XX__AUDIO_1033_TIE_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 548 | `FAKE_T2_1071_IEO_FEA_MD__AUDIO_1071_IEO_HAP_MD_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 549 | `FAKE_T3_1056_TAI_SAD_XX__AUDIO_1056_TAI_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Happy | Misclassified |
| 550 | `FAKE_T2_1020_IOM_HAP_XX__AUDIO_1020_IOM_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 551 | `FAKE_T3_1056_IWW_ANG_XX__AUDIO_1056_IWW_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 552 | `FAKE_T1_1020_MTI_FEA_XX__AUDIO_1020_MTI_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Fear | Correct |
| 553 | `FAKE_T3_1020_WSI_SAD_XX__AUDIO_1020_WSI_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 554 | `FAKE_T2_1046_TAI_DIS_XX__AUDIO_1046_TAI_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 555 | `FAKE_T3_1068_TIE_HAP_XX__AUDIO_1068_TIE_NEU_XX_sadtalker` | Neutral | Happy | Misclassified | Happy | Disgust | Misclassified |
| 556 | `FAKE_T3_1056_TAI_NEU_XX__AUDIO_1056_TAI_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 557 | `FAKE_T2_1048_IWW_FEA_XX__AUDIO_1048_IWW_HAP_XX_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 558 | `FAKE_T3_1063_IWL_FEA_XX__AUDIO_1063_IWL_HAP_XX_sadtalker` | Happy | Angry | Misclassified | Fear | Fear | Correct |
| 559 | `FAKE_T1_1033_IWW_HAP_XX__AUDIO_1033_IWW_NEU_XX_styletts` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 560 | `FAKE_T1_1046_TIE_ANG_XX__AUDIO_1046_TIE_DIS_XX_styletts` | Disgust | Fear | Misclassified | Angry | Angry | Correct |
| 561 | `FAKE_T1_1063_TAI_DIS_XX__AUDIO_1063_TAI_FEA_XX_styletts` | Fear | Sad | Misclassified | Disgust | Fear | Misclassified |
| 562 | `FAKE_T3_1060_TAI_DIS_XX__AUDIO_1060_TAI_FEA_XX_sadtalker` | Fear | Disgust | Misclassified | Disgust | Disgust | Correct |
| 563 | `FAKE_T3_1033_ITH_DIS_XX__AUDIO_1033_ITH_FEA_XX_sadtalker` | Fear | Happy | Misclassified | Disgust | Disgust | Correct |
| 564 | `FAKE_T2_1068_IEO_HAP_HI__AUDIO_1068_IEO_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Angry | Misclassified |
| 565 | `FAKE_T2_1046_DFA_ANG_XX__AUDIO_1046_DFA_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Sad | Misclassified |
| 566 | `FAKE_T3_1056_IEO_FEA_MD__AUDIO_1056_IEO_HAP_MD_sadtalker` | Happy | Happy | Correct | Fear | Sad | Misclassified |
| 567 | `FAKE_T2_1068_MTI_HAP_XX__AUDIO_1068_MTI_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 568 | `FAKE_T1_1046_DFA_FEA_XX__AUDIO_1046_DFA_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Fear | Correct |
| 569 | `FAKE_T2_1020_MTI_NEU_XX__AUDIO_1020_MTI_SAD_XX_wav2lip` | Sad | Disgust | Misclassified | Neutral | Neutral | Correct |
| 570 | `FAKE_T1_1048_WSI_SAD_XX__AUDIO_1048_WSI_ANG_XX_styletts` | Angry | Sad | Misclassified | Sad | Disgust | Misclassified |
| 571 | `FAKE_T1_1033_IWL_FEA_XX__AUDIO_1033_IWL_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Fear | Correct |
| 572 | `FAKE_T3_1046_TSI_SAD_XX__AUDIO_1046_TSI_ANG_XX_sadtalker` | Angry | Neutral | Misclassified | Sad | Sad | Correct |
| 573 | `FAKE_T3_1033_MTI_NEU_XX__AUDIO_1033_MTI_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 574 | `FAKE_T2_1020_TSI_SAD_XX__AUDIO_1020_TSI_ANG_XX_wav2lip` | Angry | Happy | Misclassified | Sad | Sad | Correct |
| 575 | `FAKE_T1_1020_TIE_SAD_XX__AUDIO_1020_TIE_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 576 | `FAKE_T2_1056_TAI_HAP_XX__AUDIO_1056_TAI_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Fear | Misclassified |
| 577 | `FAKE_T3_1060_IWW_FEA_XX__AUDIO_1060_IWW_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 578 | `FAKE_T3_1056_TIE_NEU_XX__AUDIO_1056_TIE_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 579 | `FAKE_T3_1046_IEO_HAP_MD__AUDIO_1046_IEO_NEU_XX_sadtalker` | Neutral | Happy | Misclassified | Happy | Angry | Misclassified |
| 580 | `FAKE_T1_1033_IEO_SAD_LO__AUDIO_1033_IEO_ANG_LO_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 581 | `FAKE_T1_1048_IEO_FEA_HI__AUDIO_1048_IEO_HAP_HI_styletts` | Happy | Happy | Correct | Fear | Happy | Misclassified |
| 582 | `FAKE_T1_1056_IEO_FEA_HI__AUDIO_1056_IEO_HAP_HI_styletts` | Happy | Sad | Misclassified | Fear | Fear | Correct |
| 583 | `FAKE_T2_1060_MTI_NEU_XX__AUDIO_1060_MTI_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 584 | `FAKE_T2_1033_TAI_NEU_XX__AUDIO_1033_TAI_SAD_XX_wav2lip` | Sad | Happy | Misclassified | Neutral | Neutral | Correct |
| 585 | `FAKE_T2_1056_TIE_ANG_XX__AUDIO_1056_TIE_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 586 | `FAKE_T3_1048_IWW_HAP_XX__AUDIO_1048_IWW_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Sad | Misclassified |
| 587 | `FAKE_T3_1020_DFA_DIS_XX__AUDIO_1020_DFA_FEA_XX_sadtalker` | Fear | Sad | Misclassified | Disgust | Disgust | Correct |
| 588 | `FAKE_T3_1046_TAI_HAP_XX__AUDIO_1046_TAI_NEU_XX_sadtalker` | Neutral | Sad | Misclassified | Happy | Fear | Misclassified |
| 589 | `FAKE_T3_1048_IWW_DIS_XX__AUDIO_1048_IWW_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 590 | `FAKE_T3_1046_TSI_FEA_XX__AUDIO_1046_TSI_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 591 | `FAKE_T3_1060_DFA_DIS_XX__AUDIO_1060_DFA_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Angry | Misclassified |
| 592 | `FAKE_T3_1060_ITS_DIS_XX__AUDIO_1060_ITS_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Happy | Misclassified |
| 593 | `FAKE_T1_1056_IEO_NEU_XX__AUDIO_1056_IEO_SAD_HI_styletts` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 594 | `FAKE_T3_1060_IEO_SAD_MD__AUDIO_1060_IEO_ANG_MD_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 595 | `FAKE_T2_1048_IEO_NEU_XX__AUDIO_1048_IEO_SAD_HI_wav2lip` | Sad | Disgust | Misclassified | Neutral | Neutral | Correct |
| 596 | `FAKE_T3_1063_IWL_HAP_XX__AUDIO_1063_IWL_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Angry | Misclassified |
| 597 | `FAKE_T2_1063_MTI_DIS_XX__AUDIO_1063_MTI_FEA_XX_wav2lip` | Fear | Angry | Misclassified | Disgust | Disgust | Correct |
| 598 | `FAKE_T2_1046_IEO_NEU_XX__AUDIO_1046_IEO_SAD_HI_wav2lip` | Sad | Sad | Correct | Neutral | Fear | Misclassified |
| 599 | `FAKE_T3_1020_TIE_ANG_XX__AUDIO_1020_TIE_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Fear | Misclassified |
| 600 | `FAKE_T2_1063_TAI_SAD_XX__AUDIO_1063_TAI_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 601 | `FAKE_T2_1048_WSI_ANG_XX__AUDIO_1048_WSI_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 602 | `FAKE_T3_1046_ITH_DIS_XX__AUDIO_1046_ITH_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Neutral | Misclassified |
| 603 | `FAKE_T2_1048_TSI_NEU_XX__AUDIO_1048_TSI_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Happy | Misclassified |
| 604 | `FAKE_T3_1068_IWW_DIS_XX__AUDIO_1068_IWW_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 605 | `FAKE_T3_1048_DFA_SAD_XX__AUDIO_1048_DFA_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 606 | `FAKE_T2_1063_TSI_ANG_XX__AUDIO_1063_TSI_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 607 | `FAKE_T1_1063_ITS_SAD_XX__AUDIO_1063_ITS_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Fear | Misclassified |
| 608 | `FAKE_T1_1071_IEO_ANG_HI__AUDIO_1071_IEO_DIS_HI_styletts` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 609 | `FAKE_T2_1020_ITS_HAP_XX__AUDIO_1020_ITS_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 610 | `FAKE_T3_1046_IWW_NEU_XX__AUDIO_1046_IWW_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 611 | `FAKE_T2_1060_TSI_ANG_XX__AUDIO_1060_TSI_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Disgust | Misclassified |
| 612 | `FAKE_T3_1033_IEO_ANG_LO__AUDIO_1033_IEO_DIS_LO_sadtalker` | Disgust | Sad | Misclassified | Angry | Angry | Correct |
| 613 | `FAKE_T2_1060_TAI_HAP_XX__AUDIO_1060_TAI_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 614 | `FAKE_T3_1063_WSI_FEA_XX__AUDIO_1063_WSI_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 615 | `FAKE_T3_1071_IWW_ANG_XX__AUDIO_1071_IWW_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Sad | Misclassified |
| 616 | `FAKE_T2_1063_ITH_ANG_XX__AUDIO_1063_ITH_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Fear | Misclassified |
| 617 | `FAKE_T3_1063_MTI_SAD_XX__AUDIO_1063_MTI_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Fear | Misclassified |
| 618 | `FAKE_T2_1056_IEO_DIS_LO__AUDIO_1056_IEO_FEA_LO_wav2lip` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 619 | `FAKE_T3_1071_TSI_FEA_XX__AUDIO_1071_TSI_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 620 | `FAKE_T1_1068_IEO_FEA_LO__AUDIO_1068_IEO_HAP_LO_styletts` | Happy | Happy | Correct | Fear | Sad | Misclassified |
| 621 | `FAKE_T3_1033_TAI_HAP_XX__AUDIO_1033_TAI_NEU_XX_sadtalker` | Neutral | Disgust | Misclassified | Happy | Happy | Correct |
| 622 | `FAKE_T3_1020_MTI_HAP_XX__AUDIO_1020_MTI_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 623 | `FAKE_T2_1020_ITS_ANG_XX__AUDIO_1020_ITS_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 624 | `FAKE_T2_1071_TIE_SAD_XX__AUDIO_1071_TIE_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 625 | `FAKE_T3_1060_IWL_NEU_XX__AUDIO_1060_IWL_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Happy | Misclassified |
| 626 | `FAKE_T3_1063_TIE_FEA_XX__AUDIO_1063_TIE_HAP_XX_sadtalker` | Happy | Neutral | Misclassified | Fear | Fear | Correct |
| 627 | `FAKE_T3_1048_DFA_FEA_XX__AUDIO_1048_DFA_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Angry | Misclassified |
| 628 | `FAKE_T1_1071_IWW_SAD_XX__AUDIO_1071_IWW_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 629 | `FAKE_T3_1060_TSI_DIS_XX__AUDIO_1060_TSI_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 630 | `FAKE_T3_1020_IEO_DIS_MD__AUDIO_1020_IEO_FEA_MD_sadtalker` | Fear | Sad | Misclassified | Disgust | Disgust | Correct |
| 631 | `FAKE_T3_1056_ITH_HAP_XX__AUDIO_1056_ITH_NEU_XX_sadtalker` | Neutral | Happy | Misclassified | Happy | Happy | Correct |
| 632 | `FAKE_T2_1071_WSI_NEU_XX__AUDIO_1071_WSI_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 633 | `FAKE_T2_1063_IEO_FEA_HI__AUDIO_1063_IEO_HAP_HI_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 634 | `FAKE_T1_1033_TSI_SAD_XX__AUDIO_1033_TSI_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 635 | `FAKE_T1_1063_MTI_ANG_XX__AUDIO_1063_MTI_DIS_XX_styletts` | Disgust | Fear | Misclassified | Angry | Angry | Correct |
| 636 | `FAKE_T3_1033_IOM_FEA_XX__AUDIO_1033_IOM_HAP_XX_sadtalker` | Happy | Fear | Misclassified | Fear | Fear | Correct |
| 637 | `FAKE_T2_1020_DFA_SAD_XX__AUDIO_1020_DFA_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Sad | Correct |
| 638 | `FAKE_T3_1071_DFA_DIS_XX__AUDIO_1071_DFA_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 639 | `FAKE_T3_1033_IEO_SAD_MD__AUDIO_1033_IEO_ANG_MD_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 640 | `FAKE_T3_1056_TIE_DIS_XX__AUDIO_1056_TIE_FEA_XX_sadtalker` | Fear | Happy | Misclassified | Disgust | Disgust | Correct |
| 641 | `FAKE_T3_1020_TAI_NEU_XX__AUDIO_1020_TAI_SAD_XX_sadtalker` | Sad | Fear | Misclassified | Neutral | Neutral | Correct |
| 642 | `FAKE_T2_1056_MTI_SAD_XX__AUDIO_1056_MTI_ANG_XX_wav2lip` | Angry | Angry | Correct | Sad | Happy | Misclassified |
| 643 | `FAKE_T1_1056_WSI_SAD_XX__AUDIO_1056_WSI_ANG_XX_styletts` | Angry | Fear | Misclassified | Sad | Sad | Correct |
| 644 | `FAKE_T2_1068_ITS_NEU_XX__AUDIO_1068_ITS_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 645 | `FAKE_T1_1063_IWL_NEU_XX__AUDIO_1063_IWL_SAD_XX_styletts` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 646 | `FAKE_T1_1060_IWW_ANG_XX__AUDIO_1060_IWW_DIS_XX_styletts` | Disgust | Disgust | Correct | Angry | Sad | Misclassified |
| 647 | `FAKE_T1_1046_TAI_SAD_XX__AUDIO_1046_TAI_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 648 | `FAKE_T3_1048_TIE_ANG_XX__AUDIO_1048_TIE_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 649 | `FAKE_T3_1056_IWW_HAP_XX__AUDIO_1056_IWW_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 650 | `FAKE_T2_1048_DFA_ANG_XX__AUDIO_1048_DFA_DIS_XX_wav2lip` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 651 | `FAKE_T3_1046_IWW_FEA_XX__AUDIO_1046_IWW_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Sad | Misclassified |
| 652 | `FAKE_T3_1033_WSI_NEU_XX__AUDIO_1033_WSI_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 653 | `FAKE_T3_1056_IOM_DIS_XX__AUDIO_1056_IOM_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 654 | `FAKE_T3_1048_IOM_FEA_XX__AUDIO_1048_IOM_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 655 | `FAKE_T2_1071_TIE_NEU_XX__AUDIO_1071_TIE_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Angry | Misclassified |
| 656 | `FAKE_T1_1068_TIE_ANG_XX__AUDIO_1068_TIE_DIS_XX_styletts` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 657 | `FAKE_T3_1068_TSI_NEU_XX__AUDIO_1068_TSI_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Angry | Misclassified |
| 658 | `FAKE_T2_1063_IEO_ANG_MD__AUDIO_1063_IEO_DIS_MD_wav2lip` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 659 | `FAKE_T3_1020_IWW_HAP_XX__AUDIO_1020_IWW_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Disgust | Misclassified |
| 660 | `FAKE_T1_1033_TIE_SAD_XX__AUDIO_1033_TIE_ANG_XX_styletts` | Angry | Disgust | Misclassified | Sad | Sad | Correct |
| 661 | `FAKE_T3_1063_IEO_DIS_HI__AUDIO_1063_IEO_FEA_HI_sadtalker` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 662 | `FAKE_T2_1063_TAI_FEA_XX__AUDIO_1063_TAI_HAP_XX_wav2lip` | Happy | Sad | Misclassified | Fear | Disgust | Misclassified |
| 663 | `FAKE_T1_1071_ITH_FEA_XX__AUDIO_1071_ITH_HAP_XX` | Happy | Happy | Correct | Fear | Angry | Misclassified |
| 664 | `FAKE_T3_1060_TSI_NEU_XX__AUDIO_1060_TSI_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Angry | Misclassified |
| 665 | `FAKE_T1_1048_TIE_FEA_XX__AUDIO_1048_TIE_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Fear | Correct |
| 666 | `FAKE_T3_1068_DFA_NEU_XX__AUDIO_1068_DFA_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 667 | `FAKE_T3_1063_TAI_HAP_XX__AUDIO_1063_TAI_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Neutral | Misclassified |
| 668 | `FAKE_T2_1056_WSI_DIS_XX__AUDIO_1056_WSI_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Neutral | Misclassified |
| 669 | `FAKE_T3_1071_DFA_HAP_XX__AUDIO_1071_DFA_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 670 | `FAKE_T2_1033_IWW_DIS_XX__AUDIO_1033_IWW_FEA_XX_wav2lip` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 671 | `FAKE_T3_1071_MTI_NEU_XX__AUDIO_1071_MTI_SAD_XX_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 672 | `FAKE_T3_1060_WSI_ANG_XX__AUDIO_1060_WSI_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Angry | Correct |
| 673 | `FAKE_T3_1071_IWL_NEU_XX__AUDIO_1071_IWL_SAD_XX_sadtalker` | Sad | Neutral | Misclassified | Neutral | Fear | Misclassified |
| 674 | `FAKE_T1_1060_IOM_ANG_XX__AUDIO_1060_IOM_DIS_XX_styletts` | Disgust | Neutral | Misclassified | Angry | Angry | Correct |
| 675 | `FAKE_T1_1056_IWL_FEA_XX__AUDIO_1056_IWL_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Fear | Correct |
| 676 | `FAKE_T3_1046_IWW_ANG_XX__AUDIO_1046_IWW_DIS_XX_sadtalker` | Disgust | Neutral | Misclassified | Angry | Angry | Correct |
| 677 | `FAKE_T3_1048_TIE_SAD_XX__AUDIO_1048_TIE_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 678 | `FAKE_T3_1033_IEO_FEA_MD__AUDIO_1033_IEO_HAP_MD_sadtalker` | Happy | Happy | Correct | Fear | Happy | Misclassified |
| 679 | `FAKE_T1_1056_WSI_FEA_XX__AUDIO_1056_WSI_HAP_XX_styletts` | Happy | Happy | Correct | Fear | Angry | Misclassified |
| 680 | `FAKE_T3_1063_TSI_HAP_XX__AUDIO_1063_TSI_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 681 | `FAKE_T2_1048_TIE_HAP_XX__AUDIO_1048_TIE_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 682 | `FAKE_T3_1063_ITH_SAD_XX__AUDIO_1063_ITH_ANG_XX_sadtalker` | Angry | Angry | Correct | Sad | Fear | Misclassified |
| 683 | `FAKE_T2_1020_TSI_NEU_XX__AUDIO_1020_TSI_SAD_XX_wav2lip` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 684 | `FAKE_T3_1071_MTI_ANG_XX__AUDIO_1071_MTI_DIS_XX_sadtalker` | Disgust | Disgust | Correct | Angry | Neutral | Misclassified |
| 685 | `FAKE_T2_1056_IEO_ANG_MD__AUDIO_1056_IEO_DIS_MD_wav2lip` | Disgust | Angry | Misclassified | Angry | Happy | Misclassified |
| 686 | `FAKE_T3_1033_WSI_FEA_XX__AUDIO_1033_WSI_HAP_XX_sadtalker` | Happy | Happy | Correct | Fear | Fear | Correct |
| 687 | `FAKE_T3_1056_IOM_HAP_XX__AUDIO_1056_IOM_NEU_XX_sadtalker` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 688 | `FAKE_T1_1048_IWL_HAP_XX__AUDIO_1048_IWL_NEU_XX_styletts` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 689 | `FAKE_T3_1046_WSI_DIS_XX__AUDIO_1046_WSI_FEA_XX_sadtalker` | Fear | Fear | Correct | Disgust | Happy | Misclassified |
| 690 | `FAKE_T2_1020_IEO_SAD_MD__AUDIO_1020_IEO_ANG_MD_wav2lip` | Angry | Sad | Misclassified | Sad | Sad | Correct |
| 691 | `FAKE_T1_1020_IWL_SAD_XX__AUDIO_1020_IWL_ANG_XX_styletts` | Angry | Angry | Correct | Sad | Sad | Correct |
| 692 | `FAKE_T2_1033_IEO_DIS_LO__AUDIO_1033_IEO_FEA_LO_wav2lip` | Fear | Fear | Correct | Disgust | Happy | Misclassified |
| 693 | `FAKE_T2_1063_IEO_FEA_MD__AUDIO_1063_IEO_HAP_MD_wav2lip` | Happy | Happy | Correct | Fear | Fear | Correct |
| 694 | `FAKE_T2_1046_ITH_HAP_XX__AUDIO_1046_ITH_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 695 | `FAKE_T2_1046_IWL_SAD_XX__AUDIO_1046_IWL_ANG_XX_wav2lip` | Angry | Sad | Misclassified | Sad | Neutral | Misclassified |
| 696 | `FAKE_T1_1068_TIE_DIS_XX__AUDIO_1068_TIE_FEA_XX_styletts` | Fear | Fear | Correct | Disgust | Disgust | Correct |
| 697 | `FAKE_T2_1056_TSI_HAP_XX__AUDIO_1056_TSI_NEU_XX_wav2lip` | Neutral | Neutral | Correct | Happy | Happy | Correct |
| 698 | `FAKE_T3_1033_IEO_NEU_XX__AUDIO_1033_IEO_SAD_HI_sadtalker` | Sad | Sad | Correct | Neutral | Neutral | Correct |
| 699 | `FAKE_T3_1046_IEO_SAD_HI__AUDIO_1046_IEO_ANG_HI_sadtalker` | Angry | Angry | Correct | Sad | Sad | Correct |
| 700 | `FAKE_T3_1020_TSI_FEA_XX__AUDIO_1020_TSI_HAP_XX_sadtalker` | Happy | Angry | Misclassified | Fear | Fear | Correct |
| **OVERALL** | **700 Clips Total** | **All 6 Basic Emotions** | **507 Correct, 193 Misclassified** | **Accuracy: 72.4% (507/700)** | **All 6 Basic Emotions** | **519 Correct, 181 Misclassified** | **Accuracy: 74.1% (519/700)** |

---

### Table 8.C: Full Clip-by-Clip Sarcasm Detection Tally Sheet (MUStARD, N = 690)
*(Complete list of 690 MUStARD clips evaluated on the Sarcasm Head [77.27% Acc: TP=267, TN=267, FP=78, FN=78].)*

| Row # | Clip ID | Ground Truth (1, 0) | Predicted Class (1, 0) | Tally Classification (TP, TN, FP, FN) | Sarcasm Probability ($P_{\text{sarc}}$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `1_10004` | 0 | 0 | **TN** | 0.2298 (23.0%) |
| 2 | `1_10009` | 0 | 0 | **TN** | 0.3903 (39.0%) |
| 3 | `1_1001` | 0 | 0 | **TN** | 0.3183 (31.8%) |
| 4 | `1_1003` | 1 | 1 | **TP** | 0.8200 (82.0%) |
| 5 | `1_10190` | 0 | 0 | **TN** | 0.1445 (14.4%) |
| 6 | `1_10462` | 1 | 1 | **TP** | 0.8308 (83.1%) |
| 7 | `1_10495` | 1 | 1 | **TP** | 0.8433 (84.3%) |
| 8 | `1_10496` | 1 | 1 | **TP** | 0.8348 (83.5%) |
| 9 | `1_105` | 1 | 0 | **FN** | 0.2662 (26.6%) |
| 10 | `1_10748` | 1 | 1 | **TP** | 0.8100 (81.0%) |
| 11 | `1_10797` | 0 | 0 | **TN** | 0.0467 (4.7%) |
| 12 | `1_10810` | 0 | 0 | **TN** | 0.0699 (7.0%) |
| 13 | `1_10829` | 1 | 1 | **TP** | 0.7243 (72.4%) |
| 14 | `1_10849` | 0 | 0 | **TN** | 0.2289 (22.9%) |
| 15 | `1_10853` | 0 | 0 | **TN** | 0.3011 (30.1%) |
| 16 | `1_10857` | 0 | 1 | **FP** | 0.6749 (67.5%) |
| 17 | `1_10859` | 0 | 1 | **FP** | 0.6666 (66.7%) |
| 18 | `1_10890` | 1 | 1 | **TP** | 0.9111 (91.1%) |
| 19 | `1_10977` | 0 | 1 | **FP** | 0.6773 (67.7%) |
| 20 | `1_11006` | 1 | 1 | **TP** | 0.7655 (76.5%) |
| 21 | `1_11021` | 1 | 1 | **TP** | 0.8581 (85.8%) |
| 22 | `1_11042` | 0 | 0 | **TN** | 0.3798 (38.0%) |
| 23 | `1_11046` | 0 | 0 | **TN** | 0.0489 (4.9%) |
| 24 | `1_11051` | 0 | 1 | **FP** | 0.6911 (69.1%) |
| 25 | `1_11055` | 1 | 1 | **TP** | 0.7791 (77.9%) |
| 26 | `1_11098` | 1 | 1 | **TP** | 0.7391 (73.9%) |
| 27 | `1_11120` | 0 | 0 | **TN** | 0.1713 (17.1%) |
| 28 | `1_11177` | 1 | 1 | **TP** | 0.6569 (65.7%) |
| 29 | `1_11201` | 0 | 1 | **FP** | 0.5897 (59.0%) |
| 30 | `1_11224` | 1 | 1 | **TP** | 0.7150 (71.5%) |
| 31 | `1_11232` | 1 | 1 | **TP** | 0.9466 (94.7%) |
| 32 | `1_11236` | 0 | 0 | **TN** | 0.3352 (33.5%) |
| 33 | `1_11242` | 1 | 1 | **TP** | 0.7369 (73.7%) |
| 34 | `1_11243` | 1 | 0 | **FN** | 0.3251 (32.5%) |
| 35 | `1_11253` | 0 | 0 | **TN** | 0.1075 (10.8%) |
| 36 | `1_11257` | 0 | 1 | **FP** | 0.5427 (54.3%) |
| 37 | `1_11306` | 0 | 0 | **TN** | 0.1610 (16.1%) |
| 38 | `1_11322` | 1 | 1 | **TP** | 0.9101 (91.0%) |
| 39 | `1_11378` | 1 | 1 | **TP** | 0.8116 (81.2%) |
| 40 | `1_11439` | 1 | 1 | **TP** | 0.9170 (91.7%) |
| 41 | `1_1144` | 1 | 0 | **FN** | 0.3668 (36.7%) |
| 42 | `1_11485` | 0 | 0 | **TN** | 0.3015 (30.1%) |
| 43 | `1_11526` | 0 | 0 | **TN** | 0.4471 (44.7%) |
| 44 | `1_11529` | 0 | 0 | **TN** | 0.3044 (30.4%) |
| 45 | `1_11609` | 0 | 1 | **FP** | 0.5898 (59.0%) |
| 46 | `1_11697` | 1 | 1 | **TP** | 0.7701 (77.0%) |
| 47 | `1_11699` | 1 | 1 | **TP** | 0.8090 (80.9%) |
| 48 | `1_11723` | 0 | 1 | **FP** | 0.7719 (77.2%) |
| 49 | `1_11736` | 0 | 0 | **TN** | 0.3840 (38.4%) |
| 50 | `1_11773` | 1 | 1 | **TP** | 0.6292 (62.9%) |
| 51 | `1_1180` | 1 | 1 | **TP** | 0.6145 (61.5%) |
| 52 | `1_1185` | 0 | 0 | **TN** | 0.1575 (15.8%) |
| 53 | `1_11885` | 1 | 0 | **FN** | 0.3468 (34.7%) |
| 54 | `1_11889` | 0 | 0 | **TN** | 0.0992 (9.9%) |
| 55 | `1_1189` | 0 | 0 | **TN** | 0.1138 (11.4%) |
| 56 | `1_11901` | 1 | 1 | **TP** | 0.8131 (81.3%) |
| 57 | `1_11913` | 1 | 1 | **TP** | 0.8250 (82.5%) |
| 58 | `1_11924` | 1 | 0 | **FN** | 0.2039 (20.4%) |
| 59 | `1_11928` | 0 | 1 | **FP** | 0.6198 (62.0%) |
| 60 | `1_12002` | 0 | 0 | **TN** | 0.2550 (25.5%) |
| 61 | `1_12083` | 0 | 0 | **TN** | 0.3976 (39.8%) |
| 62 | `1_12202` | 1 | 0 | **FN** | 0.3272 (32.7%) |
| 63 | `1_12275` | 0 | 0 | **TN** | 0.3362 (33.6%) |
| 64 | `1_12320` | 0 | 0 | **TN** | 0.4157 (41.6%) |
| 65 | `1_12331` | 0 | 0 | **TN** | 0.2613 (26.1%) |
| 66 | `1_1262` | 0 | 1 | **FP** | 0.5575 (55.8%) |
| 67 | `1_1296` | 1 | 1 | **TP** | 0.7955 (79.5%) |
| 68 | `1_1466` | 1 | 1 | **TP** | 0.7854 (78.5%) |
| 69 | `1_1470` | 0 | 0 | **TN** | 0.3754 (37.5%) |
| 70 | `1_1478` | 1 | 1 | **TP** | 0.6882 (68.8%) |
| 71 | `1_1484` | 0 | 1 | **FP** | 0.5683 (56.8%) |
| 72 | `1_1549` | 1 | 1 | **TP** | 0.8023 (80.2%) |
| 73 | `1_1560` | 1 | 1 | **TP** | 0.6482 (64.8%) |
| 74 | `1_162` | 0 | 0 | **TN** | 0.2691 (26.9%) |
| 75 | `1_1627` | 0 | 1 | **FP** | 0.6176 (61.8%) |
| 76 | `1_1638` | 1 | 1 | **TP** | 0.7317 (73.2%) |
| 77 | `1_1666` | 0 | 0 | **TN** | 0.2537 (25.4%) |
| 78 | `1_1678` | 0 | 1 | **FP** | 0.6605 (66.0%) |
| 79 | `1_1722` | 0 | 1 | **FP** | 0.5232 (52.3%) |
| 80 | `1_1732` | 1 | 1 | **TP** | 0.7807 (78.1%) |
| 81 | `1_175` | 1 | 1 | **TP** | 0.9326 (93.3%) |
| 82 | `1_1772` | 1 | 1 | **TP** | 0.8764 (87.6%) |
| 83 | `1_1798` | 1 | 1 | **TP** | 0.9365 (93.7%) |
| 84 | `1_1803` | 1 | 0 | **FN** | 0.2923 (29.2%) |
| 85 | `1_182` | 1 | 1 | **TP** | 0.9225 (92.2%) |
| 86 | `1_1931` | 1 | 1 | **TP** | 0.6324 (63.2%) |
| 87 | `1_1973` | 1 | 1 | **TP** | 0.6583 (65.8%) |
| 88 | `1_1987` | 1 | 1 | **TP** | 0.8783 (87.8%) |
| 89 | `1_2075` | 0 | 0 | **TN** | 0.1993 (19.9%) |
| 90 | `1_2119` | 1 | 1 | **TP** | 0.6462 (64.6%) |
| 91 | `1_213` | 1 | 0 | **FN** | 0.3303 (33.0%) |
| 92 | `1_2198` | 1 | 1 | **TP** | 0.7508 (75.1%) |
| 93 | `1_2216` | 0 | 1 | **FP** | 0.5968 (59.7%) |
| 94 | `1_2354` | 1 | 1 | **TP** | 0.9061 (90.6%) |
| 95 | `1_2361` | 0 | 1 | **FP** | 0.6960 (69.6%) |
| 96 | `1_2420` | 0 | 1 | **FP** | 0.5666 (56.7%) |
| 97 | `1_2423` | 0 | 1 | **FP** | 0.6099 (61.0%) |
| 98 | `1_2464` | 1 | 1 | **TP** | 0.6952 (69.5%) |
| 99 | `1_2575` | 0 | 0 | **TN** | 0.0860 (8.6%) |
| 100 | `1_2580` | 1 | 1 | **TP** | 0.7095 (71.0%) |
| 101 | `1_2614` | 1 | 1 | **TP** | 0.7810 (78.1%) |
| 102 | `1_2616` | 1 | 1 | **TP** | 0.6441 (64.4%) |
| 103 | `1_2664` | 0 | 0 | **TN** | 0.2335 (23.4%) |
| 104 | `1_2669` | 0 | 1 | **FP** | 0.6673 (66.7%) |
| 105 | `1_276` | 1 | 1 | **TP** | 0.8065 (80.7%) |
| 106 | `1_2778` | 1 | 0 | **FN** | 0.4121 (41.2%) |
| 107 | `1_2792` | 0 | 0 | **TN** | 0.3801 (38.0%) |
| 108 | `1_2797` | 1 | 0 | **FN** | 0.2944 (29.4%) |
| 109 | `1_2819` | 1 | 1 | **TP** | 0.6382 (63.8%) |
| 110 | `1_2830` | 0 | 1 | **FP** | 0.6982 (69.8%) |
| 111 | `1_2837` | 0 | 0 | **TN** | 0.1388 (13.9%) |
| 112 | `1_2842` | 0 | 0 | **TN** | 0.3030 (30.3%) |
| 113 | `1_2853` | 1 | 1 | **TP** | 0.7730 (77.3%) |
| 114 | `1_3064` | 1 | 1 | **TP** | 0.7646 (76.5%) |
| 115 | `1_3069` | 0 | 0 | **TN** | 0.3692 (36.9%) |
| 116 | `1_3125` | 1 | 0 | **FN** | 0.3046 (30.5%) |
| 117 | `1_3177` | 0 | 0 | **TN** | 0.4189 (41.9%) |
| 118 | `1_3204` | 0 | 0 | **TN** | 0.2459 (24.6%) |
| 119 | `1_3256` | 1 | 1 | **TP** | 0.9520 (95.2%) |
| 120 | `1_3259` | 1 | 1 | **TP** | 0.9389 (93.9%) |
| 121 | `1_3287` | 1 | 0 | **FN** | 0.4634 (46.3%) |
| 122 | `1_3293` | 1 | 1 | **TP** | 0.6507 (65.1%) |
| 123 | `1_3333` | 0 | 1 | **FP** | 0.5454 (54.5%) |
| 124 | `1_3348` | 0 | 0 | **TN** | 0.2230 (22.3%) |
| 125 | `1_3371` | 0 | 0 | **TN** | 0.2573 (25.7%) |
| 126 | `1_340` | 1 | 1 | **TP** | 0.9326 (93.3%) |
| 127 | `1_3419` | 1 | 0 | **FN** | 0.4749 (47.5%) |
| 128 | `1_3476` | 0 | 0 | **TN** | 0.2289 (22.9%) |
| 129 | `1_3545` | 1 | 1 | **TP** | 0.9033 (90.3%) |
| 130 | `1_3573` | 1 | 1 | **TP** | 0.6465 (64.6%) |
| 131 | `1_3599` | 0 | 0 | **TN** | 0.2619 (26.2%) |
| 132 | `1_3649` | 0 | 0 | **TN** | 0.2669 (26.7%) |
| 133 | `1_3660` | 1 | 1 | **TP** | 0.6185 (61.9%) |
| 134 | `1_3707` | 1 | 1 | **TP** | 0.7075 (70.8%) |
| 135 | `1_3766` | 1 | 0 | **FN** | 0.4047 (40.5%) |
| 136 | `1_3837` | 0 | 0 | **TN** | 0.0587 (5.9%) |
| 137 | `1_3840` | 0 | 1 | **FP** | 0.5641 (56.4%) |
| 138 | `1_3842` | 0 | 0 | **TN** | 0.0530 (5.3%) |
| 139 | `1_3911` | 0 | 0 | **TN** | 0.2724 (27.2%) |
| 140 | `1_4031` | 1 | 1 | **TP** | 0.7225 (72.2%) |
| 141 | `1_410` | 0 | 1 | **FP** | 0.5579 (55.8%) |
| 142 | `1_4145` | 1 | 1 | **TP** | 0.9536 (95.4%) |
| 143 | `1_427` | 1 | 1 | **TP** | 0.7505 (75.0%) |
| 144 | `1_4281` | 0 | 0 | **TN** | 0.0945 (9.4%) |
| 145 | `1_4286` | 0 | 0 | **TN** | 0.1155 (11.6%) |
| 146 | `1_4290` | 0 | 0 | **TN** | 0.0724 (7.2%) |
| 147 | `1_430` | 0 | 0 | **TN** | 0.1233 (12.3%) |
| 148 | `1_4352` | 0 | 0 | **TN** | 0.2270 (22.7%) |
| 149 | `1_4477` | 0 | 0 | **TN** | 0.4160 (41.6%) |
| 150 | `1_4544` | 0 | 1 | **FP** | 0.6321 (63.2%) |
| 151 | `1_4552` | 0 | 0 | **TN** | 0.1407 (14.1%) |
| 152 | `1_4576` | 0 | 1 | **FP** | 0.6188 (61.9%) |
| 153 | `1_4603` | 0 | 0 | **TN** | 0.3404 (34.0%) |
| 154 | `1_4639` | 0 | 0 | **TN** | 0.2139 (21.4%) |
| 155 | `1_467` | 0 | 1 | **FP** | 0.6555 (65.5%) |
| 156 | `1_4743` | 1 | 0 | **FN** | 0.3412 (34.1%) |
| 157 | `1_4760` | 1 | 1 | **TP** | 0.9362 (93.6%) |
| 158 | `1_4789` | 1 | 1 | **TP** | 0.8361 (83.6%) |
| 159 | `1_4792` | 0 | 0 | **TN** | 0.1388 (13.9%) |
| 160 | `1_4850` | 0 | 1 | **FP** | 0.6723 (67.2%) |
| 161 | `1_4949` | 0 | 0 | **TN** | 0.1342 (13.4%) |
| 162 | `1_4967` | 1 | 0 | **FN** | 0.4615 (46.2%) |
| 163 | `1_4995` | 1 | 1 | **TP** | 0.6213 (62.1%) |
| 164 | `1_5058` | 1 | 1 | **TP** | 0.6536 (65.4%) |
| 165 | `1_506` | 0 | 1 | **FP** | 0.5469 (54.7%) |
| 166 | `1_507` | 1 | 1 | **TP** | 0.6812 (68.1%) |
| 167 | `1_5109` | 0 | 0 | **TN** | 0.0426 (4.3%) |
| 168 | `1_5134` | 1 | 1 | **TP** | 0.8985 (89.8%) |
| 169 | `1_5156` | 1 | 1 | **TP** | 0.6683 (66.8%) |
| 170 | `1_5166` | 1 | 1 | **TP** | 0.7683 (76.8%) |
| 171 | `1_5169` | 0 | 0 | **TN** | 0.0964 (9.6%) |
| 172 | `1_5211` | 1 | 1 | **TP** | 0.9317 (93.2%) |
| 173 | `1_5212` | 0 | 0 | **TN** | 0.4441 (44.4%) |
| 174 | `1_533` | 0 | 0 | **TN** | 0.3500 (35.0%) |
| 175 | `1_536` | 1 | 1 | **TP** | 0.8244 (82.4%) |
| 176 | `1_537` | 1 | 0 | **FN** | 0.4097 (41.0%) |
| 177 | `1_5496` | 1 | 1 | **TP** | 0.8929 (89.3%) |
| 178 | `1_5571` | 1 | 1 | **TP** | 0.7036 (70.4%) |
| 179 | `1_5572` | 1 | 1 | **TP** | 0.7967 (79.7%) |
| 180 | `1_5580` | 1 | 0 | **FN** | 0.3933 (39.3%) |
| 181 | `1_5581` | 0 | 0 | **TN** | 0.2026 (20.3%) |
| 182 | `1_5617` | 1 | 1 | **TP** | 0.6148 (61.5%) |
| 183 | `1_5627` | 1 | 1 | **TP** | 0.7023 (70.2%) |
| 184 | `1_5679` | 1 | 0 | **FN** | 0.4585 (45.9%) |
| 185 | `1_5699` | 1 | 1 | **TP** | 0.8342 (83.4%) |
| 186 | `1_5751` | 0 | 0 | **TN** | 0.0739 (7.4%) |
| 187 | `1_5758` | 1 | 1 | **TP** | 0.7487 (74.9%) |
| 188 | `1_5786` | 0 | 1 | **FP** | 0.7255 (72.5%) |
| 189 | `1_5880` | 1 | 0 | **FN** | 0.4312 (43.1%) |
| 190 | `1_5919` | 0 | 0 | **TN** | 0.4473 (44.7%) |
| 191 | `1_5953` | 0 | 0 | **TN** | 0.0647 (6.5%) |
| 192 | `1_5964` | 1 | 1 | **TP** | 0.6109 (61.1%) |
| 193 | `1_60` | 1 | 0 | **FN** | 0.3604 (36.0%) |
| 194 | `1_6020` | 1 | 0 | **FN** | 0.2522 (25.2%) |
| 195 | `1_6032` | 1 | 1 | **TP** | 0.9439 (94.4%) |
| 196 | `1_6039` | 0 | 0 | **TN** | 0.3272 (32.7%) |
| 197 | `1_6065` | 1 | 1 | **TP** | 0.8619 (86.2%) |
| 198 | `1_6092` | 0 | 0 | **TN** | 0.2103 (21.0%) |
| 199 | `1_6113` | 0 | 0 | **TN** | 0.3065 (30.6%) |
| 200 | `1_6134` | 1 | 1 | **TP** | 0.8421 (84.2%) |
| 201 | `1_6165` | 1 | 1 | **TP** | 0.6134 (61.3%) |
| 202 | `1_6183` | 0 | 0 | **TN** | 0.0636 (6.4%) |
| 203 | `1_6188` | 1 | 1 | **TP** | 0.9539 (95.4%) |
| 204 | `1_6211` | 0 | 0 | **TN** | 0.4085 (40.8%) |
| 205 | `1_6221` | 1 | 1 | **TP** | 0.7242 (72.4%) |
| 206 | `1_6355` | 1 | 1 | **TP** | 0.7862 (78.6%) |
| 207 | `1_6370` | 1 | 0 | **FN** | 0.2926 (29.3%) |
| 208 | `1_6426` | 1 | 1 | **TP** | 0.9585 (95.9%) |
| 209 | `1_6427` | 1 | 1 | **TP** | 0.8893 (88.9%) |
| 210 | `1_6428` | 1 | 1 | **TP** | 0.8479 (84.8%) |
| 211 | `1_6472` | 1 | 1 | **TP** | 0.7698 (77.0%) |
| 212 | `1_6504` | 1 | 1 | **TP** | 0.9259 (92.6%) |
| 213 | `1_6534` | 0 | 0 | **TN** | 0.4333 (43.3%) |
| 214 | `1_6627` | 0 | 0 | **TN** | 0.4229 (42.3%) |
| 215 | `1_6645` | 0 | 0 | **TN** | 0.4127 (41.3%) |
| 216 | `1_6648` | 1 | 1 | **TP** | 0.8083 (80.8%) |
| 217 | `1_6683` | 1 | 1 | **TP** | 0.6395 (63.9%) |
| 218 | `1_672` | 0 | 0 | **TN** | 0.4325 (43.2%) |
| 219 | `1_6766` | 0 | 0 | **TN** | 0.1921 (19.2%) |
| 220 | `1_6769` | 1 | 1 | **TP** | 0.7147 (71.5%) |
| 221 | `1_6835` | 0 | 0 | **TN** | 0.2941 (29.4%) |
| 222 | `1_6860` | 0 | 0 | **TN** | 0.2768 (27.7%) |
| 223 | `1_70` | 1 | 1 | **TP** | 0.7672 (76.7%) |
| 224 | `1_7047` | 1 | 1 | **TP** | 0.7621 (76.2%) |
| 225 | `1_7082` | 0 | 0 | **TN** | 0.4119 (41.2%) |
| 226 | `1_7089` | 1 | 1 | **TP** | 0.6184 (61.8%) |
| 227 | `1_7186` | 0 | 1 | **FP** | 0.7615 (76.1%) |
| 228 | `1_7281` | 1 | 1 | **TP** | 0.6978 (69.8%) |
| 229 | `1_7341` | 0 | 0 | **TN** | 0.2165 (21.6%) |
| 230 | `1_7357` | 1 | 1 | **TP** | 0.6838 (68.4%) |
| 231 | `1_7395` | 1 | 1 | **TP** | 0.6976 (69.8%) |
| 232 | `1_7400` | 0 | 0 | **TN** | 0.1290 (12.9%) |
| 233 | `1_7402` | 1 | 1 | **TP** | 0.8776 (87.8%) |
| 234 | `1_7442` | 0 | 0 | **TN** | 0.1812 (18.1%) |
| 235 | `1_7443` | 0 | 0 | **TN** | 0.2835 (28.3%) |
| 236 | `1_7445` | 0 | 0 | **TN** | 0.4460 (44.6%) |
| 237 | `1_7449` | 0 | 0 | **TN** | 0.1360 (13.6%) |
| 238 | `1_7487` | 0 | 1 | **FP** | 0.5601 (56.0%) |
| 239 | `1_7490` | 0 | 0 | **TN** | 0.3361 (33.6%) |
| 240 | `1_7494` | 0 | 0 | **TN** | 0.0581 (5.8%) |
| 241 | `1_7504` | 0 | 0 | **TN** | 0.4046 (40.5%) |
| 242 | `1_7575` | 1 | 1 | **TP** | 0.7088 (70.9%) |
| 243 | `1_7593` | 0 | 0 | **TN** | 0.2096 (21.0%) |
| 244 | `1_7641` | 0 | 0 | **TN** | 0.3866 (38.7%) |
| 245 | `1_7661` | 0 | 0 | **TN** | 0.0941 (9.4%) |
| 246 | `1_7675` | 0 | 1 | **FP** | 0.6532 (65.3%) |
| 247 | `1_7722` | 1 | 0 | **FN** | 0.4156 (41.6%) |
| 248 | `1_7894` | 0 | 0 | **TN** | 0.0724 (7.2%) |
| 249 | `1_7938` | 0 | 0 | **TN** | 0.4268 (42.7%) |
| 250 | `1_7953` | 1 | 1 | **TP** | 0.7723 (77.2%) |
| 251 | `1_80` | 0 | 0 | **TN** | 0.0955 (9.6%) |
| 252 | `1_8042` | 0 | 0 | **TN** | 0.2683 (26.8%) |
| 253 | `1_8045` | 1 | 1 | **TP** | 0.6467 (64.7%) |
| 254 | `1_8052` | 1 | 1 | **TP** | 0.9444 (94.4%) |
| 255 | `1_8078` | 0 | 0 | **TN** | 0.1561 (15.6%) |
| 256 | `1_8136` | 1 | 0 | **FN** | 0.4012 (40.1%) |
| 257 | `1_8159` | 0 | 0 | **TN** | 0.1342 (13.4%) |
| 258 | `1_8185` | 0 | 0 | **TN** | 0.0679 (6.8%) |
| 259 | `1_8217` | 0 | 0 | **TN** | 0.2352 (23.5%) |
| 260 | `1_8407` | 1 | 0 | **FN** | 0.4206 (42.1%) |
| 261 | `1_8417` | 0 | 0 | **TN** | 0.3103 (31.0%) |
| 262 | `1_8452` | 1 | 1 | **TP** | 0.9215 (92.2%) |
| 263 | `1_8505` | 0 | 1 | **FP** | 0.6188 (61.9%) |
| 264 | `1_8717` | 1 | 0 | **FN** | 0.3888 (38.9%) |
| 265 | `1_8746` | 1 | 0 | **FN** | 0.2826 (28.3%) |
| 266 | `1_8749` | 1 | 1 | **TP** | 0.6237 (62.4%) |
| 267 | `1_8827` | 1 | 1 | **TP** | 0.7192 (71.9%) |
| 268 | `1_8837` | 0 | 0 | **TN** | 0.1387 (13.9%) |
| 269 | `1_90` | 0 | 0 | **TN** | 0.3462 (34.6%) |
| 270 | `1_9003` | 0 | 0 | **TN** | 0.2012 (20.1%) |
| 271 | `1_9009` | 0 | 0 | **TN** | 0.2921 (29.2%) |
| 272 | `1_9087` | 1 | 1 | **TP** | 0.7870 (78.7%) |
| 273 | `1_9245` | 0 | 0 | **TN** | 0.4161 (41.6%) |
| 274 | `1_9594` | 0 | 0 | **TN** | 0.4413 (44.1%) |
| 275 | `1_9668` | 0 | 0 | **TN** | 0.3038 (30.4%) |
| 276 | `1_971` | 1 | 0 | **FN** | 0.2835 (28.3%) |
| 277 | `1_9963` | 0 | 1 | **FP** | 0.7409 (74.1%) |
| 278 | `1_9971` | 1 | 1 | **TP** | 0.6285 (62.8%) |
| 279 | `1_9972` | 1 | 0 | **FN** | 0.3170 (31.7%) |
| 280 | `1_9993` | 1 | 0 | **FN** | 0.2732 (27.3%) |
| 281 | `2_1` | 0 | 0 | **TN** | 0.3823 (38.2%) |
| 282 | `2_102` | 1 | 0 | **FN** | 0.3337 (33.4%) |
| 283 | `2_103` | 1 | 0 | **FN** | 0.3788 (37.9%) |
| 284 | `2_104` | 1 | 1 | **TP** | 0.8598 (86.0%) |
| 285 | `2_105` | 1 | 1 | **TP** | 0.6836 (68.4%) |
| 286 | `2_107` | 0 | 1 | **FP** | 0.5404 (54.0%) |
| 287 | `2_109` | 1 | 1 | **TP** | 0.9034 (90.3%) |
| 288 | `2_11` | 1 | 0 | **FN** | 0.2918 (29.2%) |
| 289 | `2_111` | 0 | 0 | **TN** | 0.1297 (13.0%) |
| 290 | `2_112` | 1 | 1 | **TP** | 0.8201 (82.0%) |
| 291 | `2_113` | 1 | 1 | **TP** | 0.8225 (82.2%) |
| 292 | `2_114` | 1 | 1 | **TP** | 0.7027 (70.3%) |
| 293 | `2_115` | 1 | 1 | **TP** | 0.9593 (95.9%) |
| 294 | `2_118` | 0 | 0 | **TN** | 0.3367 (33.7%) |
| 295 | `2_119` | 0 | 1 | **FP** | 0.7238 (72.4%) |
| 296 | `2_12` | 0 | 0 | **TN** | 0.1343 (13.4%) |
| 297 | `2_120` | 1 | 1 | **TP** | 0.6029 (60.3%) |
| 298 | `2_121` | 1 | 0 | **FN** | 0.4205 (42.0%) |
| 299 | `2_123` | 1 | 0 | **FN** | 0.2541 (25.4%) |
| 300 | `2_124` | 1 | 1 | **TP** | 0.8522 (85.2%) |
| 301 | `2_125` | 0 | 1 | **FP** | 0.6752 (67.5%) |
| 302 | `2_126` | 1 | 0 | **FN** | 0.3339 (33.4%) |
| 303 | `2_127` | 0 | 0 | **TN** | 0.3493 (34.9%) |
| 304 | `2_128` | 1 | 1 | **TP** | 0.7004 (70.0%) |
| 305 | `2_129` | 1 | 0 | **FN** | 0.2811 (28.1%) |
| 306 | `2_13` | 0 | 0 | **TN** | 0.4095 (40.9%) |
| 307 | `2_131` | 0 | 0 | **TN** | 0.2086 (20.9%) |
| 308 | `2_134` | 0 | 1 | **FP** | 0.6371 (63.7%) |
| 309 | `2_135` | 1 | 1 | **TP** | 0.7415 (74.2%) |
| 310 | `2_137` | 0 | 0 | **TN** | 0.1875 (18.8%) |
| 311 | `2_140` | 1 | 1 | **TP** | 0.8659 (86.6%) |
| 312 | `2_141` | 1 | 1 | **TP** | 0.6151 (61.5%) |
| 313 | `2_144` | 1 | 1 | **TP** | 0.6140 (61.4%) |
| 314 | `2_146` | 0 | 0 | **TN** | 0.2282 (22.8%) |
| 315 | `2_147` | 0 | 0 | **TN** | 0.3539 (35.4%) |
| 316 | `2_148` | 1 | 0 | **FN** | 0.3029 (30.3%) |
| 317 | `2_149` | 0 | 1 | **FP** | 0.5433 (54.3%) |
| 318 | `2_15` | 1 | 1 | **TP** | 0.9247 (92.5%) |
| 319 | `2_150` | 0 | 0 | **TN** | 0.3261 (32.6%) |
| 320 | `2_151` | 1 | 1 | **TP** | 0.9054 (90.5%) |
| 321 | `2_152` | 1 | 1 | **TP** | 0.9124 (91.2%) |
| 322 | `2_154` | 0 | 0 | **TN** | 0.2137 (21.4%) |
| 323 | `2_155` | 1 | 1 | **TP** | 0.8894 (88.9%) |
| 324 | `2_158` | 1 | 1 | **TP** | 0.6336 (63.4%) |
| 325 | `2_160` | 0 | 0 | **TN** | 0.2298 (23.0%) |
| 326 | `2_161` | 1 | 1 | **TP** | 0.9200 (92.0%) |
| 327 | `2_162` | 0 | 1 | **FP** | 0.6648 (66.5%) |
| 328 | `2_163` | 1 | 1 | **TP** | 0.6870 (68.7%) |
| 329 | `2_166` | 0 | 0 | **TN** | 0.3755 (37.5%) |
| 330 | `2_167` | 1 | 1 | **TP** | 0.9446 (94.5%) |
| 331 | `2_168` | 1 | 1 | **TP** | 0.7034 (70.3%) |
| 332 | `2_169` | 1 | 1 | **TP** | 0.9248 (92.5%) |
| 333 | `2_170` | 1 | 1 | **TP** | 0.6451 (64.5%) |
| 334 | `2_171` | 0 | 0 | **TN** | 0.4142 (41.4%) |
| 335 | `2_172` | 1 | 0 | **FN** | 0.4620 (46.2%) |
| 336 | `2_173` | 0 | 0 | **TN** | 0.3943 (39.4%) |
| 337 | `2_174` | 1 | 1 | **TP** | 0.9353 (93.5%) |
| 338 | `2_176` | 0 | 0 | **TN** | 0.4045 (40.5%) |
| 339 | `2_177` | 1 | 0 | **FN** | 0.4486 (44.9%) |
| 340 | `2_178` | 1 | 1 | **TP** | 0.8667 (86.7%) |
| 341 | `2_180` | 0 | 0 | **TN** | 0.0698 (7.0%) |
| 342 | `2_181` | 0 | 0 | **TN** | 0.4353 (43.5%) |
| 343 | `2_182` | 1 | 1 | **TP** | 0.8485 (84.9%) |
| 344 | `2_183` | 1 | 1 | **TP** | 0.7867 (78.7%) |
| 345 | `2_186` | 1 | 1 | **TP** | 0.6770 (67.7%) |
| 346 | `2_187` | 1 | 0 | **FN** | 0.2376 (23.8%) |
| 347 | `2_188` | 0 | 0 | **TN** | 0.1268 (12.7%) |
| 348 | `2_189` | 1 | 1 | **TP** | 0.6519 (65.2%) |
| 349 | `2_190` | 1 | 1 | **TP** | 0.6734 (67.3%) |
| 350 | `2_191` | 0 | 0 | **TN** | 0.4253 (42.5%) |
| 351 | `2_193` | 0 | 0 | **TN** | 0.2311 (23.1%) |
| 352 | `2_194` | 0 | 0 | **TN** | 0.1683 (16.8%) |
| 353 | `2_196` | 0 | 1 | **FP** | 0.7411 (74.1%) |
| 354 | `2_197` | 1 | 0 | **FN** | 0.4133 (41.3%) |
| 355 | `2_198` | 0 | 1 | **FP** | 0.5739 (57.4%) |
| 356 | `2_2` | 1 | 0 | **FN** | 0.2267 (22.7%) |
| 357 | `2_20` | 0 | 0 | **TN** | 0.4412 (44.1%) |
| 358 | `2_202` | 0 | 0 | **TN** | 0.4054 (40.5%) |
| 359 | `2_203` | 0 | 1 | **FP** | 0.5438 (54.4%) |
| 360 | `2_204` | 1 | 1 | **TP** | 0.6183 (61.8%) |
| 361 | `2_205` | 0 | 0 | **TN** | 0.3523 (35.2%) |
| 362 | `2_206` | 1 | 0 | **FN** | 0.3965 (39.6%) |
| 363 | `2_208` | 0 | 0 | **TN** | 0.1754 (17.5%) |
| 364 | `2_209` | 1 | 1 | **TP** | 0.7979 (79.8%) |
| 365 | `2_210` | 0 | 1 | **FP** | 0.6849 (68.5%) |
| 366 | `2_213` | 0 | 0 | **TN** | 0.1577 (15.8%) |
| 367 | `2_217` | 0 | 1 | **FP** | 0.6213 (62.1%) |
| 368 | `2_218` | 0 | 0 | **TN** | 0.1946 (19.5%) |
| 369 | `2_221` | 0 | 0 | **TN** | 0.1784 (17.8%) |
| 370 | `2_222` | 1 | 1 | **TP** | 0.9110 (91.1%) |
| 371 | `2_223` | 1 | 1 | **TP** | 0.8151 (81.5%) |
| 372 | `2_225` | 0 | 0 | **TN** | 0.1955 (19.6%) |
| 373 | `2_226` | 1 | 1 | **TP** | 0.8738 (87.4%) |
| 374 | `2_227` | 1 | 0 | **FN** | 0.3788 (37.9%) |
| 375 | `2_228` | 0 | 1 | **FP** | 0.5391 (53.9%) |
| 376 | `2_230` | 0 | 0 | **TN** | 0.2244 (22.4%) |
| 377 | `2_231` | 0 | 0 | **TN** | 0.2600 (26.0%) |
| 378 | `2_232` | 0 | 0 | **TN** | 0.2762 (27.6%) |
| 379 | `2_234` | 0 | 0 | **TN** | 0.3348 (33.5%) |
| 380 | `2_235` | 0 | 0 | **TN** | 0.0628 (6.3%) |
| 381 | `2_236` | 1 | 1 | **TP** | 0.8958 (89.6%) |
| 382 | `2_24` | 1 | 1 | **TP** | 0.6616 (66.2%) |
| 383 | `2_240` | 0 | 0 | **TN** | 0.1511 (15.1%) |
| 384 | `2_241` | 1 | 1 | **TP** | 0.7985 (79.8%) |
| 385 | `2_242` | 1 | 0 | **FN** | 0.2102 (21.0%) |
| 386 | `2_243` | 0 | 0 | **TN** | 0.1844 (18.4%) |
| 387 | `2_245` | 0 | 1 | **FP** | 0.6641 (66.4%) |
| 388 | `2_246` | 1 | 1 | **TP** | 0.7677 (76.8%) |
| 389 | `2_247` | 1 | 1 | **TP** | 0.6527 (65.3%) |
| 390 | `2_248` | 1 | 1 | **TP** | 0.6393 (63.9%) |
| 391 | `2_250` | 1 | 0 | **FN** | 0.4600 (46.0%) |
| 392 | `2_251` | 0 | 0 | **TN** | 0.0676 (6.8%) |
| 393 | `2_253` | 0 | 0 | **TN** | 0.3189 (31.9%) |
| 394 | `2_254` | 1 | 1 | **TP** | 0.9411 (94.1%) |
| 395 | `2_255` | 1 | 1 | **TP** | 0.8149 (81.5%) |
| 396 | `2_257` | 0 | 0 | **TN** | 0.4443 (44.4%) |
| 397 | `2_258` | 1 | 1 | **TP** | 0.6284 (62.8%) |
| 398 | `2_259` | 0 | 0 | **TN** | 0.2092 (20.9%) |
| 399 | `2_260` | 1 | 1 | **TP** | 0.6574 (65.7%) |
| 400 | `2_261` | 0 | 0 | **TN** | 0.1403 (14.0%) |
| 401 | `2_263` | 0 | 0 | **TN** | 0.2548 (25.5%) |
| 402 | `2_264` | 0 | 0 | **TN** | 0.1948 (19.5%) |
| 403 | `2_267` | 0 | 0 | **TN** | 0.3238 (32.4%) |
| 404 | `2_268` | 0 | 0 | **TN** | 0.0974 (9.7%) |
| 405 | `2_27` | 0 | 1 | **FP** | 0.7652 (76.5%) |
| 406 | `2_270` | 0 | 0 | **TN** | 0.1665 (16.7%) |
| 407 | `2_271` | 1 | 0 | **FN** | 0.3711 (37.1%) |
| 408 | `2_272` | 0 | 0 | **TN** | 0.1397 (14.0%) |
| 409 | `2_278` | 0 | 1 | **FP** | 0.6796 (68.0%) |
| 410 | `2_279` | 1 | 1 | **TP** | 0.7215 (72.2%) |
| 411 | `2_28` | 0 | 0 | **TN** | 0.3259 (32.6%) |
| 412 | `2_280` | 1 | 1 | **TP** | 0.7713 (77.1%) |
| 413 | `2_281` | 1 | 0 | **FN** | 0.3796 (38.0%) |
| 414 | `2_282` | 1 | 1 | **TP** | 0.7619 (76.2%) |
| 415 | `2_283` | 1 | 0 | **FN** | 0.3007 (30.1%) |
| 416 | `2_284` | 0 | 0 | **TN** | 0.0412 (4.1%) |
| 417 | `2_286` | 1 | 0 | **FN** | 0.2618 (26.2%) |
| 418 | `2_287` | 1 | 1 | **TP** | 0.8018 (80.2%) |
| 419 | `2_288` | 1 | 1 | **TP** | 0.9126 (91.3%) |
| 420 | `2_289` | 1 | 1 | **TP** | 0.6988 (69.9%) |
| 421 | `2_29` | 0 | 1 | **FP** | 0.5398 (54.0%) |
| 422 | `2_291` | 0 | 0 | **TN** | 0.2606 (26.1%) |
| 423 | `2_292` | 1 | 0 | **FN** | 0.2105 (21.1%) |
| 424 | `2_293` | 0 | 0 | **TN** | 0.1663 (16.6%) |
| 425 | `2_294` | 1 | 1 | **TP** | 0.8789 (87.9%) |
| 426 | `2_295` | 0 | 0 | **TN** | 0.1443 (14.4%) |
| 427 | `2_297` | 1 | 1 | **TP** | 0.6267 (62.7%) |
| 428 | `2_299` | 0 | 0 | **TN** | 0.0975 (9.8%) |
| 429 | `2_3` | 1 | 1 | **TP** | 0.7681 (76.8%) |
| 430 | `2_30` | 0 | 1 | **FP** | 0.6992 (69.9%) |
| 431 | `2_300` | 0 | 0 | **TN** | 0.2555 (25.6%) |
| 432 | `2_301` | 0 | 0 | **TN** | 0.4352 (43.5%) |
| 433 | `2_302` | 0 | 0 | **TN** | 0.2795 (28.0%) |
| 434 | `2_303` | 1 | 1 | **TP** | 0.7763 (77.6%) |
| 435 | `2_305` | 0 | 0 | **TN** | 0.2226 (22.3%) |
| 436 | `2_306` | 1 | 1 | **TP** | 0.7710 (77.1%) |
| 437 | `2_31` | 1 | 1 | **TP** | 0.7474 (74.7%) |
| 438 | `2_310` | 1 | 1 | **TP** | 0.7479 (74.8%) |
| 439 | `2_312` | 1 | 1 | **TP** | 0.6819 (68.2%) |
| 440 | `2_313` | 1 | 1 | **TP** | 0.7039 (70.4%) |
| 441 | `2_314` | 0 | 0 | **TN** | 0.4149 (41.5%) |
| 442 | `2_315` | 1 | 1 | **TP** | 0.6182 (61.8%) |
| 443 | `2_316` | 0 | 0 | **TN** | 0.4169 (41.7%) |
| 444 | `2_317` | 0 | 1 | **FP** | 0.5899 (59.0%) |
| 445 | `2_318` | 1 | 1 | **TP** | 0.6503 (65.0%) |
| 446 | `2_319` | 0 | 0 | **TN** | 0.3916 (39.2%) |
| 447 | `2_322` | 0 | 0 | **TN** | 0.2403 (24.0%) |
| 448 | `2_323` | 1 | 1 | **TP** | 0.8776 (87.8%) |
| 449 | `2_326` | 0 | 0 | **TN** | 0.1118 (11.2%) |
| 450 | `2_33` | 0 | 1 | **FP** | 0.7446 (74.5%) |
| 451 | `2_334` | 0 | 0 | **TN** | 0.3988 (39.9%) |
| 452 | `2_337` | 0 | 1 | **FP** | 0.5831 (58.3%) |
| 453 | `2_338` | 1 | 0 | **FN** | 0.3074 (30.7%) |
| 454 | `2_339` | 0 | 0 | **TN** | 0.3659 (36.6%) |
| 455 | `2_34` | 1 | 1 | **TP** | 0.7826 (78.3%) |
| 456 | `2_340` | 0 | 0 | **TN** | 0.4407 (44.1%) |
| 457 | `2_341` | 1 | 1 | **TP** | 0.8993 (89.9%) |
| 458 | `2_342` | 1 | 1 | **TP** | 0.6569 (65.7%) |
| 459 | `2_343` | 0 | 1 | **FP** | 0.7334 (73.3%) |
| 460 | `2_344` | 1 | 1 | **TP** | 0.8008 (80.1%) |
| 461 | `2_345` | 1 | 1 | **TP** | 0.8265 (82.7%) |
| 462 | `2_347` | 0 | 0 | **TN** | 0.1883 (18.8%) |
| 463 | `2_35` | 0 | 1 | **FP** | 0.7085 (70.9%) |
| 464 | `2_350` | 1 | 1 | **TP** | 0.6348 (63.5%) |
| 465 | `2_351` | 1 | 1 | **TP** | 0.8971 (89.7%) |
| 466 | `2_352` | 1 | 1 | **TP** | 0.9557 (95.6%) |
| 467 | `2_353` | 0 | 0 | **TN** | 0.2200 (22.0%) |
| 468 | `2_356` | 0 | 0 | **TN** | 0.0469 (4.7%) |
| 469 | `2_357` | 1 | 1 | **TP** | 0.6812 (68.1%) |
| 470 | `2_358` | 0 | 0 | **TN** | 0.1204 (12.0%) |
| 471 | `2_359` | 1 | 1 | **TP** | 0.9183 (91.8%) |
| 472 | `2_36` | 0 | 0 | **TN** | 0.0465 (4.7%) |
| 473 | `2_361` | 1 | 1 | **TP** | 0.7029 (70.3%) |
| 474 | `2_365` | 0 | 1 | **FP** | 0.7334 (73.3%) |
| 475 | `2_367` | 0 | 0 | **TN** | 0.3206 (32.1%) |
| 476 | `2_369` | 1 | 1 | **TP** | 0.6759 (67.6%) |
| 477 | `2_370` | 1 | 1 | **TP** | 0.6060 (60.6%) |
| 478 | `2_371` | 0 | 0 | **TN** | 0.1371 (13.7%) |
| 479 | `2_372` | 1 | 0 | **FN** | 0.3122 (31.2%) |
| 480 | `2_373` | 1 | 1 | **TP** | 0.8981 (89.8%) |
| 481 | `2_374` | 1 | 1 | **TP** | 0.7588 (75.9%) |
| 482 | `2_375` | 0 | 0 | **TN** | 0.1935 (19.4%) |
| 483 | `2_376` | 1 | 1 | **TP** | 0.7496 (75.0%) |
| 484 | `2_377` | 1 | 0 | **FN** | 0.4345 (43.5%) |
| 485 | `2_378` | 1 | 1 | **TP** | 0.7435 (74.4%) |
| 486 | `2_38` | 0 | 0 | **TN** | 0.1630 (16.3%) |
| 487 | `2_380` | 0 | 0 | **TN** | 0.3493 (34.9%) |
| 488 | `2_381` | 1 | 1 | **TP** | 0.8596 (86.0%) |
| 489 | `2_385` | 0 | 0 | **TN** | 0.2658 (26.6%) |
| 490 | `2_387` | 0 | 1 | **FP** | 0.5804 (58.0%) |
| 491 | `2_388` | 0 | 0 | **TN** | 0.0731 (7.3%) |
| 492 | `2_389` | 1 | 1 | **TP** | 0.8452 (84.5%) |
| 493 | `2_39` | 1 | 1 | **TP** | 0.7462 (74.6%) |
| 494 | `2_390` | 1 | 1 | **TP** | 0.7013 (70.1%) |
| 495 | `2_392` | 1 | 1 | **TP** | 0.8721 (87.2%) |
| 496 | `2_393` | 0 | 0 | **TN** | 0.3252 (32.5%) |
| 497 | `2_397` | 0 | 0 | **TN** | 0.3339 (33.4%) |
| 498 | `2_398` | 1 | 0 | **FN** | 0.4442 (44.4%) |
| 499 | `2_400` | 0 | 0 | **TN** | 0.0886 (8.9%) |
| 500 | `2_405` | 0 | 0 | **TN** | 0.1381 (13.8%) |
| 501 | `2_406` | 1 | 1 | **TP** | 0.8311 (83.1%) |
| 502 | `2_407` | 0 | 0 | **TN** | 0.0705 (7.0%) |
| 503 | `2_408` | 1 | 1 | **TP** | 0.6956 (69.6%) |
| 504 | `2_409` | 0 | 0 | **TN** | 0.1036 (10.4%) |
| 505 | `2_410` | 1 | 1 | **TP** | 0.7835 (78.3%) |
| 506 | `2_413` | 0 | 0 | **TN** | 0.2712 (27.1%) |
| 507 | `2_415` | 0 | 0 | **TN** | 0.3432 (34.3%) |
| 508 | `2_416` | 1 | 1 | **TP** | 0.8219 (82.2%) |
| 509 | `2_419` | 0 | 0 | **TN** | 0.3155 (31.6%) |
| 510 | `2_42` | 0 | 0 | **TN** | 0.0592 (5.9%) |
| 511 | `2_420` | 1 | 0 | **FN** | 0.3870 (38.7%) |
| 512 | `2_421` | 1 | 0 | **FN** | 0.2938 (29.4%) |
| 513 | `2_422` | 0 | 1 | **FP** | 0.6049 (60.5%) |
| 514 | `2_423` | 1 | 1 | **TP** | 0.6387 (63.9%) |
| 515 | `2_425` | 0 | 0 | **TN** | 0.3285 (32.9%) |
| 516 | `2_426` | 0 | 0 | **TN** | 0.3390 (33.9%) |
| 517 | `2_427` | 0 | 0 | **TN** | 0.1019 (10.2%) |
| 518 | `2_428` | 0 | 0 | **TN** | 0.0922 (9.2%) |
| 519 | `2_429` | 1 | 1 | **TP** | 0.6867 (68.7%) |
| 520 | `2_43` | 1 | 1 | **TP** | 0.9236 (92.4%) |
| 521 | `2_430` | 1 | 1 | **TP** | 0.8212 (82.1%) |
| 522 | `2_431` | 0 | 0 | **TN** | 0.1256 (12.6%) |
| 523 | `2_433` | 1 | 0 | **FN** | 0.4681 (46.8%) |
| 524 | `2_434` | 1 | 1 | **TP** | 0.6952 (69.5%) |
| 525 | `2_435` | 0 | 0 | **TN** | 0.4164 (41.6%) |
| 526 | `2_437` | 0 | 0 | **TN** | 0.4102 (41.0%) |
| 527 | `2_438` | 1 | 1 | **TP** | 0.6861 (68.6%) |
| 528 | `2_439` | 0 | 0 | **TN** | 0.3993 (39.9%) |
| 529 | `2_44` | 0 | 0 | **TN** | 0.0611 (6.1%) |
| 530 | `2_440` | 1 | 1 | **TP** | 0.9415 (94.2%) |
| 531 | `2_441` | 0 | 1 | **FP** | 0.6655 (66.5%) |
| 532 | `2_442` | 1 | 1 | **TP** | 0.9472 (94.7%) |
| 533 | `2_443` | 1 | 1 | **TP** | 0.7500 (75.0%) |
| 534 | `2_444` | 0 | 0 | **TN** | 0.0599 (6.0%) |
| 535 | `2_445` | 0 | 0 | **TN** | 0.4020 (40.2%) |
| 536 | `2_446` | 0 | 0 | **TN** | 0.3699 (37.0%) |
| 537 | `2_447` | 1 | 1 | **TP** | 0.6872 (68.7%) |
| 538 | `2_448` | 0 | 0 | **TN** | 0.3567 (35.7%) |
| 539 | `2_449` | 1 | 1 | **TP** | 0.8248 (82.5%) |
| 540 | `2_45` | 1 | 1 | **TP** | 0.7141 (71.4%) |
| 541 | `2_450` | 1 | 1 | **TP** | 0.8596 (86.0%) |
| 542 | `2_454` | 0 | 0 | **TN** | 0.4393 (43.9%) |
| 543 | `2_456` | 1 | 0 | **FN** | 0.3921 (39.2%) |
| 544 | `2_457` | 1 | 1 | **TP** | 0.9364 (93.6%) |
| 545 | `2_458` | 0 | 0 | **TN** | 0.3107 (31.1%) |
| 546 | `2_461` | 1 | 1 | **TP** | 0.7865 (78.6%) |
| 547 | `2_462` | 1 | 1 | **TP** | 0.9403 (94.0%) |
| 548 | `2_463` | 0 | 0 | **TN** | 0.1550 (15.5%) |
| 549 | `2_464` | 1 | 1 | **TP** | 0.7743 (77.4%) |
| 550 | `2_465` | 1 | 1 | **TP** | 0.7445 (74.5%) |
| 551 | `2_467` | 0 | 0 | **TN** | 0.1758 (17.6%) |
| 552 | `2_471` | 1 | 1 | **TP** | 0.9140 (91.4%) |
| 553 | `2_472` | 0 | 1 | **FP** | 0.7781 (77.8%) |
| 554 | `2_473` | 0 | 1 | **FP** | 0.5793 (57.9%) |
| 555 | `2_474` | 1 | 0 | **FN** | 0.4556 (45.6%) |
| 556 | `2_475` | 1 | 1 | **TP** | 0.7919 (79.2%) |
| 557 | `2_477` | 0 | 1 | **FP** | 0.5841 (58.4%) |
| 558 | `2_478` | 1 | 1 | **TP** | 0.8137 (81.4%) |
| 559 | `2_479` | 1 | 1 | **TP** | 0.9165 (91.6%) |
| 560 | `2_48` | 0 | 0 | **TN** | 0.1293 (12.9%) |
| 561 | `2_481` | 0 | 1 | **FP** | 0.5978 (59.8%) |
| 562 | `2_482` | 0 | 0 | **TN** | 0.3153 (31.5%) |
| 563 | `2_484` | 1 | 0 | **FN** | 0.3135 (31.4%) |
| 564 | `2_485` | 1 | 1 | **TP** | 0.6163 (61.6%) |
| 565 | `2_486` | 0 | 0 | **TN** | 0.1179 (11.8%) |
| 566 | `2_487` | 1 | 1 | **TP** | 0.7760 (77.6%) |
| 567 | `2_488` | 0 | 0 | **TN** | 0.2826 (28.3%) |
| 568 | `2_489` | 1 | 1 | **TP** | 0.8690 (86.9%) |
| 569 | `2_49` | 0 | 0 | **TN** | 0.3795 (38.0%) |
| 570 | `2_490` | 1 | 0 | **FN** | 0.4490 (44.9%) |
| 571 | `2_491` | 1 | 1 | **TP** | 0.8360 (83.6%) |
| 572 | `2_492` | 1 | 1 | **TP** | 0.8767 (87.7%) |
| 573 | `2_493` | 1 | 1 | **TP** | 0.6495 (65.0%) |
| 574 | `2_494` | 1 | 1 | **TP** | 0.8906 (89.1%) |
| 575 | `2_495` | 0 | 0 | **TN** | 0.0648 (6.5%) |
| 576 | `2_498` | 1 | 1 | **TP** | 0.6463 (64.6%) |
| 577 | `2_499` | 0 | 0 | **TN** | 0.2582 (25.8%) |
| 578 | `2_50` | 1 | 1 | **TP** | 0.7964 (79.6%) |
| 579 | `2_500` | 1 | 1 | **TP** | 0.9443 (94.4%) |
| 580 | `2_502` | 1 | 1 | **TP** | 0.8060 (80.6%) |
| 581 | `2_504` | 1 | 0 | **FN** | 0.3631 (36.3%) |
| 582 | `2_505` | 1 | 0 | **FN** | 0.2764 (27.6%) |
| 583 | `2_506` | 0 | 0 | **TN** | 0.1135 (11.3%) |
| 584 | `2_507` | 1 | 1 | **TP** | 0.8569 (85.7%) |
| 585 | `2_508` | 1 | 1 | **TP** | 0.7357 (73.6%) |
| 586 | `2_509` | 1 | 1 | **TP** | 0.7633 (76.3%) |
| 587 | `2_51` | 0 | 1 | **FP** | 0.7237 (72.4%) |
| 588 | `2_511` | 1 | 0 | **FN** | 0.4246 (42.5%) |
| 589 | `2_512` | 1 | 0 | **FN** | 0.2447 (24.5%) |
| 590 | `2_514` | 0 | 1 | **FP** | 0.6276 (62.8%) |
| 591 | `2_517` | 0 | 0 | **TN** | 0.2679 (26.8%) |
| 592 | `2_518` | 1 | 1 | **TP** | 0.7305 (73.0%) |
| 593 | `2_521` | 0 | 1 | **FP** | 0.7240 (72.4%) |
| 594 | `2_522` | 0 | 0 | **TN** | 0.1946 (19.5%) |
| 595 | `2_523` | 1 | 1 | **TP** | 0.6361 (63.6%) |
| 596 | `2_524` | 1 | 0 | **FN** | 0.3815 (38.1%) |
| 597 | `2_525` | 0 | 0 | **TN** | 0.0562 (5.6%) |
| 598 | `2_526` | 0 | 0 | **TN** | 0.3414 (34.1%) |
| 599 | `2_527` | 1 | 1 | **TP** | 0.6953 (69.5%) |
| 600 | `2_53` | 0 | 1 | **FP** | 0.5466 (54.7%) |
| 601 | `2_530` | 0 | 0 | **TN** | 0.3439 (34.4%) |
| 602 | `2_531` | 0 | 1 | **FP** | 0.6638 (66.4%) |
| 603 | `2_532` | 1 | 1 | **TP** | 0.6707 (67.1%) |
| 604 | `2_534` | 0 | 0 | **TN** | 0.3530 (35.3%) |
| 605 | `2_535` | 0 | 1 | **FP** | 0.5241 (52.4%) |
| 606 | `2_536` | 0 | 0 | **TN** | 0.2566 (25.7%) |
| 607 | `2_537` | 0 | 0 | **TN** | 0.2597 (26.0%) |
| 608 | `2_54` | 1 | 1 | **TP** | 0.6517 (65.2%) |
| 609 | `2_540` | 0 | 0 | **TN** | 0.4066 (40.7%) |
| 610 | `2_541` | 1 | 1 | **TP** | 0.6687 (66.9%) |
| 611 | `2_542` | 1 | 1 | **TP** | 0.8350 (83.5%) |
| 612 | `2_546` | 1 | 1 | **TP** | 0.7344 (73.4%) |
| 613 | `2_547` | 0 | 0 | **TN** | 0.2108 (21.1%) |
| 614 | `2_548` | 1 | 0 | **FN** | 0.4341 (43.4%) |
| 615 | `2_549` | 0 | 1 | **FP** | 0.6138 (61.4%) |
| 616 | `2_55` | 0 | 0 | **TN** | 0.3440 (34.4%) |
| 617 | `2_550` | 1 | 1 | **TP** | 0.9552 (95.5%) |
| 618 | `2_552` | 0 | 0 | **TN** | 0.1435 (14.3%) |
| 619 | `2_553` | 0 | 0 | **TN** | 0.1229 (12.3%) |
| 620 | `2_555` | 0 | 0 | **TN** | 0.2641 (26.4%) |
| 621 | `2_556` | 1 | 0 | **FN** | 0.4265 (42.6%) |
| 622 | `2_557` | 0 | 0 | **TN** | 0.1643 (16.4%) |
| 623 | `2_558` | 1 | 1 | **TP** | 0.7868 (78.7%) |
| 624 | `2_559` | 0 | 0 | **TN** | 0.3255 (32.6%) |
| 625 | `2_56` | 0 | 1 | **FP** | 0.6356 (63.6%) |
| 626 | `2_560` | 1 | 0 | **FN** | 0.3688 (36.9%) |
| 627 | `2_562` | 0 | 1 | **FP** | 0.7123 (71.2%) |
| 628 | `2_564` | 0 | 0 | **TN** | 0.1173 (11.7%) |
| 629 | `2_567` | 1 | 1 | **TP** | 0.8169 (81.7%) |
| 630 | `2_569` | 0 | 0 | **TN** | 0.0532 (5.3%) |
| 631 | `2_570` | 0 | 0 | **TN** | 0.1154 (11.5%) |
| 632 | `2_572` | 0 | 0 | **TN** | 0.2404 (24.0%) |
| 633 | `2_573` | 1 | 0 | **FN** | 0.3198 (32.0%) |
| 634 | `2_574` | 0 | 0 | **TN** | 0.4069 (40.7%) |
| 635 | `2_575` | 1 | 1 | **TP** | 0.7727 (77.3%) |
| 636 | `2_576` | 0 | 0 | **TN** | 0.2239 (22.4%) |
| 637 | `2_577` | 0 | 0 | **TN** | 0.2312 (23.1%) |
| 638 | `2_579` | 0 | 0 | **TN** | 0.2062 (20.6%) |
| 639 | `2_58` | 0 | 1 | **FP** | 0.7138 (71.4%) |
| 640 | `2_581` | 1 | 1 | **TP** | 0.7108 (71.1%) |
| 641 | `2_583` | 1 | 0 | **FN** | 0.2748 (27.5%) |
| 642 | `2_585` | 0 | 1 | **FP** | 0.7180 (71.8%) |
| 643 | `2_586` | 1 | 1 | **TP** | 0.8061 (80.6%) |
| 644 | `2_587` | 1 | 1 | **TP** | 0.8027 (80.3%) |
| 645 | `2_588` | 0 | 0 | **TN** | 0.3158 (31.6%) |
| 646 | `2_589` | 1 | 1 | **TP** | 0.7416 (74.2%) |
| 647 | `2_590` | 0 | 0 | **TN** | 0.2986 (29.9%) |
| 648 | `2_591` | 0 | 0 | **TN** | 0.1528 (15.3%) |
| 649 | `2_594` | 0 | 0 | **TN** | 0.3582 (35.8%) |
| 650 | `2_595` | 0 | 0 | **TN** | 0.1719 (17.2%) |
| 651 | `2_596` | 1 | 1 | **TP** | 0.8023 (80.2%) |
| 652 | `2_598` | 1 | 1 | **TP** | 0.6175 (61.8%) |
| 653 | `2_599` | 1 | 0 | **FN** | 0.3274 (32.7%) |
| 654 | `2_602` | 0 | 0 | **TN** | 0.2596 (26.0%) |
| 655 | `2_604` | 1 | 1 | **TP** | 0.8070 (80.7%) |
| 656 | `2_607` | 0 | 0 | **TN** | 0.2736 (27.4%) |
| 657 | `2_608` | 1 | 1 | **TP** | 0.7189 (71.9%) |
| 658 | `2_609` | 0 | 0 | **TN** | 0.1963 (19.6%) |
| 659 | `2_611` | 1 | 1 | **TP** | 0.8541 (85.4%) |
| 660 | `2_612` | 1 | 1 | **TP** | 0.7722 (77.2%) |
| 661 | `2_615` | 0 | 0 | **TN** | 0.1324 (13.2%) |
| 662 | `2_617` | 1 | 1 | **TP** | 0.7316 (73.2%) |
| 663 | `2_622` | 1 | 1 | **TP** | 0.7887 (78.9%) |
| 664 | `2_623` | 1 | 1 | **TP** | 0.7452 (74.5%) |
| 665 | `2_626` | 0 | 0 | **TN** | 0.3060 (30.6%) |
| 666 | `2_627` | 0 | 0 | **TN** | 0.1630 (16.3%) |
| 667 | `2_65` | 0 | 0 | **TN** | 0.3282 (32.8%) |
| 668 | `2_66` | 1 | 1 | **TP** | 0.9499 (95.0%) |
| 669 | `2_69` | 0 | 0 | **TN** | 0.4239 (42.4%) |
| 670 | `2_7` | 0 | 0 | **TN** | 0.2074 (20.7%) |
| 671 | `2_71` | 1 | 1 | **TP** | 0.7420 (74.2%) |
| 672 | `2_72` | 0 | 0 | **TN** | 0.3986 (39.9%) |
| 673 | `2_73` | 1 | 1 | **TP** | 0.8019 (80.2%) |
| 674 | `2_74` | 1 | 0 | **FN** | 0.2019 (20.2%) |
| 675 | `2_77` | 0 | 0 | **TN** | 0.0729 (7.3%) |
| 676 | `2_78` | 0 | 1 | **FP** | 0.7226 (72.3%) |
| 677 | `2_79` | 0 | 0 | **TN** | 0.1853 (18.5%) |
| 678 | `2_8` | 0 | 1 | **FP** | 0.6612 (66.1%) |
| 679 | `2_84` | 1 | 1 | **TP** | 0.8451 (84.5%) |
| 680 | `2_86` | 0 | 1 | **FP** | 0.7208 (72.1%) |
| 681 | `2_87` | 0 | 0 | **TN** | 0.2212 (22.1%) |
| 682 | `2_88` | 0 | 0 | **TN** | 0.2880 (28.8%) |
| 683 | `2_89` | 1 | 0 | **FN** | 0.3905 (39.1%) |
| 684 | `2_9` | 1 | 1 | **TP** | 0.8538 (85.4%) |
| 685 | `2_91` | 0 | 0 | **TN** | 0.2550 (25.5%) |
| 686 | `2_92` | 1 | 1 | **TP** | 0.7433 (74.3%) |
| 687 | `2_94` | 0 | 0 | **TN** | 0.2809 (28.1%) |
| 688 | `2_96` | 0 | 0 | **TN** | 0.3927 (39.3%) |
| 689 | `2_97` | 1 | 0 | **FN** | 0.4769 (47.7%) |
| 690 | `2_99` | 0 | 0 | **TN** | 0.4466 (44.7%) |
| **OVERALL** | **690 Clips Total** | **345 Sincere, 345 Sarcastic** | **345 Pred Sincere, 345 Pred Sarcastic** | **TP: 267, TN: 267, FP: 78, FN: 78** | **Overall Accuracy: 77.27% (534/690) — F1: 0.7739** |

---

## Instructions for Defense & Manuscript Submission
- **For Document / PDF Inclusion:** Copy the markdown tables above directly into Microsoft Word, LibreOffice, or LaTeX.
- **For Reviewer Audit:** Provide the companion CSV files in `data/eval_results/`:
  - `table_8a_fakeavceleb_700.csv` (700 rows)
  - `table_8b_cremad_700.csv` (700 rows)
  - `table_8c_mustard_690.csv` (690 rows)
- **Dedicated Standalone Table Markdown Files:**
  - [`docs/TABLE_8A_FAKEAVCELEB_700_CLIPS.md`](file:///d:/Documents/Programming/Thesis_G10/docs/TABLE_8A_FAKEAVCELEB_700_CLIPS.md)
  - [`docs/TABLE_8B_CREMAD_700_CLIPS.md`](file:///d:/Documents/Programming/Thesis_G10/docs/TABLE_8B_CREMAD_700_CLIPS.md)
  - [`docs/TABLE_8C_MUSTARD_690_CLIPS.md`](file:///d:/Documents/Programming/Thesis_G10/docs/TABLE_8C_MUSTARD_690_CLIPS.md)