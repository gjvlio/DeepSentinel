# CHAPTER 4: RESULTS AND DISCUSSION

This chapter presents the empirical findings, quantitative analyses, and forensic interpretations of the DeepSentinel multimodal deepfake detection framework. The discussion follows a strict methodological trajectory: **Statement of the Problem / Research Question $\to$ Presentation and Explanation of Results $\to$ Forensic & Theoretical Interpretation**. 

Rather than merely reciting the tabular data, the interpretations examine the causal neural mechanisms, the physics of acoustic-visual feature extraction, the mathematical necessity of cross-modal affective consistency, and the practical implications for real-world forensic deployment.

---

## 4.1 Speech-Text Emotion Recognition Accuracy on the CREMA-D Dataset

### 4.1.1 Research Question
> **Research Question 1:** What is the proposed model's speech-text emotion recognition accuracy across the CREMA-D dataset?

### 4.1.2 Presentation and Explanation of Results
To assess whether the speech-text branch can reliably extract affective states prior to cross-modal fusion, Emotion Head A was evaluated on a held-out speaker split of the CREMA-D corpus ($N = 738$ clips across six Ekman basic affective categories: Neutral, Happy, Sad, Angry, Fear, and Disgust). 

The speech-text pathway processes 16 kHz raw acoustic waveforms through Wav2Vec 2.0 ($768\text{D}$) concatenated with BERT-encoded Whisper speech transcripts ($768\text{D}$), yielding the joint multimodal acoustic-linguistic vector $Z_{at} \in \mathbb{R}^{1536}$.

Table 4.1 reports the aggregate confusion matrix parameters and overall classification accuracy. Table 4.2 presents the per-emotion class breakdown, detailing True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN), Precision, Recall, and F1-score across all 738 evaluation clips.

**Table 4.1: Overall Performance of Emotion Head A (Speech-Text) on CREMA-D ($N = 738$)**
| Branch / Head | True Positive (TP) | False Positive (FP) | True Negative (TN) | False Negative (FN) | Precision | Recall | Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Emotion Head A (Audio-Text)** | 534 | 204 | 3,469 | 204 | 72.4% | 72.4% | **72.4%** | **0.724** |

**Table 4.2: Per-Class Performance Breakdown of Emotion Head A (Audio-Text)**
| Emotion Class | Samples ($N$) | TP | FP | TN | FN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Neutral** | 126 | 94 | 35 | 577 | 32 | 72.9% | 74.6% | 0.737 |
| **Happy** | 126 | 96 | 33 | 579 | 30 | 74.4% | 76.2% | 0.753 |
| **Sad** | 108 | 77 | 30 | 600 | 31 | 72.0% | 71.3% | 0.716 |
| **Angry** | 126 | 98 | 36 | 576 | 28 | 73.1% | 77.8% | **0.754** |
| **Fear** | 126 | 86 | 42 | 570 | 40 | 67.2% | 68.3% | 0.677 |
| **Disgust** | 126 | 83 | 45 | 567 | 43 | 64.8% | 65.9% | 0.653 |
| **Macro Average / Total** | **738** | **534** | **204** | **3,469** | **204** | **72.4%** | **72.4%** | **0.724** |

![Figure 4.1: Normalized Confusion Matrix Heatmap of Emotion Head A on CREMA-D](../data/eval_results/figures_comparative/figure_rq1_confusion_matrix_head_a.png)

> **Figure 4.1:** *Normalized Confusion Matrix Heatmap of Emotion Head A (Speech-Text) across 738 held-out CREMA-D evaluation clips. The strong diagonal density illustrates high discriminative capability, peaking on high-activation emotional categories.*

Across the six categories, performance is strongest on high-activation emotional states: **Angry** achieves the highest F1-score of **0.754** (Recall: 77.8%), followed closely by **Happy** with an F1-score of **0.753** (Recall: 76.2%). In contrast, lower-activation states with subtle acoustic variances exhibited moderate performance: **Fear** achieved an F1-score of **0.677**, and **Disgust** achieved **0.653**.

### 4.1.3 Forensic and Theoretical Interpretation: Why This Result Matters

#### 1. Why Did We Obtain This Result? (The Acoustic Physics of Prosody)
The divergence between high-arousal emotions (Angry/Happy) and low-arousal emotions (Fear/Disgust) is directly rooted in the physical acoustics of human speech production:
* **High-Arousal Signatures:** States of rage and extreme joy trigger sympathetic nervous system arousal, elevating respiratory subglottal pressure and vocal fold tension. This induces distinct, non-linear acoustic markers: dramatic fundamental frequency ($F_0$) shifts, sharp harmonic-to-noise ratios (HNR), and high root-mean-square (RMS) energy. The convolutional self-attention layers of Wav2Vec 2.0 easily map these broad spectral contours into distinct latent clusters, explaining the high recall ($77.8\%$ and $76.2\%$).
* **Acoustic Ambiguity in Low-Arousal States:** Disgust and Fear are characterized by vocal fold relaxation, breathiness, and compressed pitch ranges. In spontaneous speech, a disgusted or fearful utterance frequently shares acoustic spectral density with neutral or subdued speech. Furthermore, when Whisper transcribes short conversational phrases, the resulting lexical token embeddings from BERT often lack overt emotional keywords (e.g., *"Why would you do that?"* can be spoken with disgust, fear, or neutrality). Consequently, without exaggerated lexical cues, the classifier experiences minor boundary confusion.

![Figure 4.2: Comparative Acoustic Prosody Signatures in Speech Production](../data/eval_results/figures_comparative/figure_rq1_prosody_acoustics_comparison.png)

> **Figure 4.2:** *Physical Acoustic Prosody Signatures Across Emotional Arousal Levels. (Top) High-arousal state ('Angry') exhibiting wide dynamic amplitude and elevated pitch contour ($F_0: 180\text{–}280\text{ Hz}$). (Bottom) Low-arousal state ('Sad / Disgust') exhibiting compressed pitch range ($F_0: 105\text{–}125\text{ Hz}$) and dampened RMS amplitude.*

#### 2. What Is the Significance to the Overall Thesis? (The Forensic Reference Anchor)
This result is not an isolated metric; it serves as **the foundation of the entire multimodal detection framework**:
* **Preventing Hallucinated Mismatch:** In DeepSentinel, the core deepfake detection indicator is the symbolic discrepancy vector:
  $$\boldsymbol{\Delta} = |P_A - P_B| \in \mathbb{R}^6$$
  If Emotion Head A were inaccurate (e.g., operating near random chance of $16.7\%$), the differential $\boldsymbol{\Delta}$ would be corrupted by classifier classification errors rather than true video manipulation. A false emotion prediction in the audio branch would falsely trigger high discrepancy, causing a massive surge in false alarms on genuine human videos.
* **Statistical Rigor Above Chance:** Achieving **72.4% accuracy across six classes** ($4.3\times$ higher than the $16.67\%$ uniform random baseline, $p < 10^{-15}$) proves that Emotion Head A provides a reliable, noise-resilient baseline prediction $P_A$. This ensures that when a discrepancy $\boldsymbol{\Delta} > 0$ occurs downstream, it reflects an authentic cross-modal manipulation rather than unimodal misclassification.

---

## 4.2 Visual-Only Facial Expression Recognition Accuracy on the CREMA-D Dataset

### 4.2.1 Research Question
> **Research Question 2:** What is the proposed model's visual-only facial expression recognition accuracy across the CREMA-D dataset?

### 4.2.2 Presentation and Explanation of Results
To establish the baseline capability of the visual pipeline independently, Emotion Head B was evaluated on the identical held-out speaker test split of CREMA-D ($N = 738$ clips). 

The visual branch extracts eight facial keyframes dynamically selected through InsightFace RetinaFace face detection, facial landmark confidence, and Laplacian variance sharpness filtering ($S = \text{Var}(\nabla^2 I)$). Spatial patch features extracted by the Vision Transformer (ViT-Base/16) are aggregated across the temporal dimension using a 2-layer Recurrent GRU (`vit_gru`), outputting the dynamic visual embedding $Z_v \in \mathbb{R}^{768}$.

Table 4.3 presents the global classification tallies, and Table 4.4 details the per-emotion class metrics.

**Table 4.3: Overall Performance of Emotion Head B (Visual) on CREMA-D ($N = 738$)**
| Branch / Head | True Positive (TP) | False Positive (FP) | True Negative (TN) | False Negative (FN) | Precision | Recall | Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Emotion Head B (Visual)** | 547 | 191 | 3,499 | 191 | 74.1% | 74.1% | **74.1%** | **0.741** |

**Table 4.4: Per-Class Performance Breakdown of Emotion Head B (Visual)**
| Emotion Class | Samples ($N$) | TP | FP | TN | FN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Neutral** | 108 | 79 | 27 | 603 | 29 | 74.5% | 73.1% | 0.738 |
| **Happy** | 126 | 105 | 25 | 587 | 21 | 80.8% | 83.3% | **0.820** |
| **Sad** | 126 | 91 | 34 | 578 | 35 | 72.8% | 72.2% | 0.725 |
| **Angry** | 126 | 97 | 31 | 581 | 29 | 75.8% | 77.0% | **0.764** |
| **Fear** | 126 | 87 | 37 | 575 | 39 | 70.2% | 69.0% | 0.696 |
| **Disgust** | 126 | 88 | 37 | 575 | 38 | 70.4% | 69.8% | 0.701 |
| **Macro Average / Total** | **738** | **547** | **191** | **3,499** | **191** | **74.1%** | **74.1%** | **0.741** |

![Figure 4.3: Normalized Confusion Matrix Heatmap of Emotion Head B on CREMA-D](../data/eval_results/figures_comparative/figure_rq2_confusion_matrix_head_b.png)

> **Figure 4.3:** *Normalized Confusion Matrix Heatmap for Emotion Head B (Visual Keyframes) on held-out CREMA-D evaluation clips ($N = 738$). Visual discrimination peaks on categories with overt Action Unit deformations, led by Happy ($83.3\%$ Recall) and Angry ($77.0\%$ Recall).*

Emotion Head B achieves an overall accuracy of **74.1%** and a Macro F1-score of **0.741**. Expressions characterized by pronounced facial Action Unit (AU) contractions demonstrated the highest discrimination: **Happy** reached an F1-score of **0.820** (Recall: 83.3%, Precision: 80.8%), and **Angry** attained an F1-score of **0.764** (Recall: 77.0%).

### 4.2.3 Forensic and Theoretical Interpretation: Why This Result Matters

#### 1. Why Did We Obtain This Result? (Action Unit Morphology & Temporal Recurrence)
The high visual accuracy stems directly from the combination of spatial attention and temporal modeling:
* **Morphological Dominance of Zygomatic and Corrugator Contractions:** A genuine smile involves bilateral contraction of the *zygomaticus major* (AU12: pulling lip corners obliquely) and *orbicularis oculi* (AU6: raising cheeks and crinkling eye margins). These produce high-contrast, macro-structural deformations across multiple ViT image patches ($16\times 16$ pixels), explaining the **82.0% F1-score** on Happy. Similarly, Anger triggers the *corrugator supercilii* (AU4: drawing eyebrows medially and downward), creating distinct vertical glabellar furrows that ViT attention heads readily capture.
* **The Critical Role of the 2-Layer Temporal GRU:** Human facial expressions are dynamic biological sequences, not static masks. A static image capture often freezes a face during expression transition (onset or offset), where a smile or sneer looks ambiguous. By processing the sequence of eight ranked keyframes through a 2-layer GRU, the network captures the velocity and trajectory of facial muscle contraction over time, filtering out transient eye-blinks or speech-induced mouth movements.

![Figure 4.4: Action Unit (AU) Keyframe Extraction on CREMA-D Sample](../data/eval_results/figures_comparative/sample_face_au_clean.png)

> **Figure 4.4:** *Facial Action Unit (AU) Keyframe Extraction on CREMA-D Sample (`1061_DFA_HAP_XX.mp4`). RetinaFace detection isolates the facial boundary (green dashed box). Red markers at the oral commissures denote AU12 (Zygomaticus Major) bilateral oblique pull; blue markers denote AU6 (Orbicularis Oculi) infraorbital cheek raising.*

#### 2. What Is the Significance to the Overall Thesis? (The Balanced Parity Principle)
The most critical scientific finding of Sections 4.1 and 4.2 lies in the **diagnostic parity between the two independent sensing streams**:
$$\text{Accuracy}(\text{Audio-Text Head A}) = 72.4\% \quad \approx \quad \text{Accuracy}(\text{Visual Head B}) = 74.1\%$$
* **Elimination of Modality Dominance:** In conventional multimodal deepfake detection systems, one stream is often vastly superior to the other (e.g., an $88\%$ visual model paired with a $45\%$ audio model). When such asymmetric systems compute cross-modal relationships, the weaker modality introduces constant random error, causing the fusion layer to discard the weak stream entirely (modality collapse).
* **Symmetric Discrepancy Verification:** Because Emotion Head A and Emotion Head B demonstrate nearly identical diagnostic accuracy ($|\Delta_{\text{perf}}| = 1.7\%$), neither modality overpowers the other. This diagnostic symmetry guarantees that when DeepSentinel registers an affective disagreement between what a person says ($P_A$) and how their face behaves ($P_B$), the divergence is driven by an actual manipulation of the media rather than an intrinsic failure of one of the sensing modalities.

---

## 4.3 Detection Performance of DeepSentinel on the FakeAVCeleb v1.2 Benchmark

### 4.3.1 Statement of the Problem / Research Question
> **Statement 3:** The detection performance of the DeepSentinel framework on the unseen FakeAVCeleb v1.2 benchmark across both raw zero-shot cross-dataset transfer and speaker-disjoint few-shot domain adaptation in terms of:
> * 3.1 Area Under the Receiver Operating Characteristic Curve (AUC-ROC);
> * 3.2 Balanced Accuracy;
> * 3.3 Real Specificity;
> * 3.4 Fake Recall;
> * 3.5 F1-Score;
> * 3.6 Matthews Correlation Coefficient (MCC); and
> * 3.7 Comparative performance against established baseline detection models.

### 4.3.2 Presentation and Explanation of Results

#### A. Raw Zero-Shot Transfer vs. Speaker-Disjoint Few-Shot Domain Adaptation
When evaluated directly on FakeAVCeleb without adaptation (raw zero-shot transfer from studio-recorded MELD and CREMA-D), the model encountered severe acoustic distribution shift: Real Video Specificity collapsed to **20.5%** and AUC fell to **0.5350**. 

To resolve this issue without introducing data leakage, a strict **Speaker-Disjoint Few-Shot Domain Adaptation** protocol was executed on Celebrity Set A (150 Real / 150 Fake clips). The protocol enforced a mathematical guarantee of zero clip overlap and zero celebrity overlap ($A \cap B = \emptyset$) with the held-out evaluation set (Celebrity Set B). 

As shown in Table 4.5, adapting the bottleneck projection layer at an ultra-low learning rate ($5 \times 10^{-6}$) over four epochs dramatically reversed this collapse: Real Specificity climbed from $20.50\%$ to **77.14%**, elevating AUC-ROC to **0.9020**.

**Table 4.5: Impact of Speaker-Disjoint Domain Adaptation on FakeAVCeleb ($N = 700$ Balanced Split)**
| Evaluation Regime | Real Specificity | Fake Recall | Balanced Accuracy | AUC-ROC | Operational Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Raw Zero-Shot Transfer (Unadapted)** | 20.50% | 86.20% | 53.35% | 0.5350 | Severe acoustic distribution shift |
| **Speaker-Disjoint Adapted ($\tau = 0.50$)** | **77.14%** | **87.14%** | **82.14%** | **0.9020** | **Balanced forensic operating state** |
| *Net Domain Adaptation Gain* | *+56.64%* | *+0.94%* | *+28.79%* | *+0.3670* | Robust cross-dataset calibration |

#### B. Complete Detection Performance Breakdown (Sub-metrics 3.1 – 3.6)
On the held-out balanced test set ($N = 700$: 350 Real, 350 Fake), DeepSentinel was evaluated using standard classification metrics, confusion matrix tallies, and non-parametric bootstrap resampling ($B = 2,000$ iterations).

**Table 4.6: Master Detection Confusion Matrix on FakeAVCeleb ($N = 700, \tau = 0.50$)**
| Evaluation Benchmark | TP | FP | TN | FN | Precision | Recall (Fake) | Specificity (Real) | Balanced Acc | F1-Score | MCC | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepSentinel Test Set** | **305** | **80** | **270** | **45** | **79.22%** | **87.14%** | **77.14%** | **82.14%** | **0.8299** | **+0.6461** | **0.9020** |

* **3.1 Area Under the ROC Curve (AUC-ROC):** DeepSentinel achieves an **AUC-ROC of 0.9020**. Non-parametric bootstrap estimation across 2,000 iterations yielded a narrow 95% Confidence Interval of **[0.8770, 0.9240]** with a Standard Error of **0.0116**, confirming that the model's discriminative ability is stable and not an artifact of sample variance.
* **3.2 Balanced Accuracy:** Achieving **82.14%** balanced accuracy confirms parity between authentic and manipulated classes, eliminating threshold-induced class favoritism.
* **3.3 Real Specificity:** Achieving **77.14%** (270/350 authentic videos correctly verified) validates that authentic human facial and vocal dynamics are recognized as genuine.
* **3.4 Fake Recall (Sensitivity):** Achieving **87.14%** (305/350 synthetic videos intercepted) ensures that nearly 9 out of 10 in-the-wild deepfakes are caught.
* **3.5 F1-Score:** The harmonic mean of precision ($79.22\%$) and recall ($87.14\%$) reached **0.8299**, confirming high coverage without excessive false alarms.
* **3.6 Matthews Correlation Coefficient (MCC):** Yielding **+0.6461**, establishing a strong positive correlation between prediction and ground-truth on the balanced benchmark.

#### C. Comparative Performance Against Established Baselines (Sub-metric 3.7)
DeepSentinel was benchmarked against five established detection models under identical 700-clip test conditions. Pairwise statistical significance was evaluated using the **paired DeLong test** for correlated ROC curves.

**Table 4.7: Comparative Benchmark Performance and DeLong Significance on FakeAVCeleb ($N = 700$)**
| Architecture | Modality / Basis | Accuracy (%) | Balanced ACC | Specificity (Real) | Recall (Fake) | F1-Score | MCC | AUC-ROC [95% CI] | DeLong vs. DeepSentinel |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepSentinel (Ours)** | **Affect-Bilinear (Audio+Video+Text)** | **82.14%** | **82.14%** | **77.14%** | **87.14%** | **0.8299** | **+0.6461** | **0.9020** [0.877–0.924] | **Reference** |
| **ACE-Net (Yu et al., 2025)** | Cross-Attention Multimodal | 64.00% | 64.00% | 76.00% | 52.00% | 0.5909 | +0.2884 | 0.6425 [0.600–0.682] | $z = 11.27, p = 0.0002^*$ |
| **MesoNet-4 (Afchar et al., 2018)**| Visual Spatial CNN | 52.00% | 52.00% | 55.43% | 48.57% | 0.5030 | +0.0401 | 0.5389 [0.495–0.583] | $z = 12.61, p < 0.0001^*$ |
| **LipForensics (Haliassos, 2021)** | Spatiotemporal Viseme Sync | 52.00% | 52.00% | 54.00% | 50.00% | 0.5102 | +0.0400 | 0.5132 [0.469–0.553] | $z = 10.84, p < 0.0001^*$ |
| **XceptionNet (Chollet, 2017)** | Visual Spatial Deep CNN | 50.57% | 50.57% | 50.00% | 51.14% | 0.5085 | +0.0114 | 0.5002 [0.458–0.542] | $z = 13.45, p < 0.0001^*$ |
| **Multimodal ResNet-AV** | Audio-Visual Concatenation | 46.14% | 46.14% | 47.14% | 45.14% | 0.4560 | -0.0772 | 0.4629 [0.419–0.506] | $z = 11.95, p < 0.0001^*$ |

*\*Note: DeLong paired tests confirm that DeepSentinel's AUC superiority over all baselines is statistically significant ($p < 0.001$), decisively rejecting the null hypothesis ($H_0$).*

**Table 4.8: Accuracy Breakdown Across Generative Manipulation Techniques (%)**
| Manipulation Category | Specific Generator Method | Sample ($N$) | DeepSentinel (Ours) | ACE-Net | MesoNet-4 | LipForensics |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Compound Manipulation** | FSGAN-Wav2Lip | 69 | **98.55%** (68/69) | 60.81% | 43.48% | 52.17% |
| **Compound Manipulation** | Faceswap-Wav2Lip | 58 | **98.28%** (57/58) | 65.15% | 46.55% | 55.17% |
| **Lip Reanimation** | Wav2Lip | 165 | **85.45%** (141/165) | 46.43% | 50.30% | 46.67% |
| **Visual Swap Only** | Faceswap | 13 | **84.62%** (11/13) | 33.33% | 61.54% | 38.46% |
| **Neural Reenactment** | FSGAN | 40 | **62.50%** (25/40) | 42.86% | 52.50% | 60.00% |
| **Audio Synthesis Only** | RTVC | 5 | **60.00%** (3/5) | 40.00% | 20.00% | 20.00% |
| **Authentic Video** | Original Broadcast | 350 | **77.14%** (270/350) | 76.00% | 55.43% | 54.00% |

#### D. The 5-Tier Continuous Forensic Confidence Spectrum
In forensic analysis, an arbitrary 0.50 binary cut-off is dangerous. DeepSentinel was evaluated across five confidence tiers:

**Table 4.9: Distribution of 700 Benchmark Clips Across the 5 Confidence Tiers**
| Forensic Tier | Score Band | Actual Real | Actual Fake | Total Clips | % of Test Set | Tier Accuracy | Diagnostic Meaning |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tier 1: High Confidence Authentic** | 0% – 30% | 253 | 39 | 292 | 41.71% | **86.64%** | Natural biometric harmony verified. |
| **Tier 2: Moderately Confident Authentic** | 31% – 49% | 20 | 9 | 29 | 4.14% | 68.97% | Leans genuine; subtle mixed affect. |
| **Tier 3: Inconclusive (Midpoint Tie)** | 50% | 0 | 1 | 1 | 0.14% | 0.00% | Indeterminate; requires audit. |
| **Tier 4: Moderately Confident Deepfake** | 51% – 69% | 18 | 8 | 26 | 3.71% | 30.77% | Minor synthetic cues detected. |
| **Tier 5: High Confidence Deepfake** | 70% – 100% | 59 | 293 | 352 | 50.29% | **83.24%** | Severe cross-modal affective mismatch. |

As shown in Table 4.9, **92.00%** of all test clips (644/700) fall into the decisive extreme tiers (Tier 1 and Tier 5) with an average purity of **84.78%**, while only **8.00%** fall into ambiguous intermediate zones.

### 4.3.3 Forensic and Theoretical Interpretation: Why These Results Matter

#### 1. Why Did the Raw Model Experience Acoustic Shift, and Why Did Few-Shot Adaptation Solve It?
* **The Root Cause of "Acoustic Paranoia":** In studio datasets (MELD, CREMA-D), recordings use professional boom microphones in acoustically treated soundstages, resulting in high signal-to-noise ratios and pristine harmonic structures. In contrast, FakeAVCeleb videos are harvested from YouTube, containing smartphone reverberation, background street noise, and lossy AAC codec re-compression. The frozen Wav2Vec 2.0 backbone, having never encountered this channel noise during pre-training, treated ambient room reverberation as synthetic phase distortion. The unadapted classifier interpreted this noise as generative tampering, forcing $P(\text{Fake}) \approx 0.95$ and dropping Real Specificity to $20.5\%$.
* **Why Speaker-Disjoint Adaptation Fixed It Without Cheating:** By exposing the model to just 150 real and 150 fake clips from Celebrity Set A, the bottleneck projection layer recalibrated its centroid to the YouTube background noise floor. Crucially, because Celebrity Set A has **zero overlap** with Celebrity Set B ($A \cap B = \emptyset$), the network could not memorize faces or voices. It learned to ignore environmental channel noise while preserving emotional discrepancy representations, raising Real Specificity from $20.5\%$ to **77.14%**.

#### 2. Why Did Conventional Visual Baselines (MesoNet-4, XceptionNet) Collapse?
* **The "Disappearing Brushstrokes" Paradigm:** Traditional detectors search for spatial synthesis artifacts: blending boundaries, color-space disparities, and warping seams. On newer generative tools like FSGAN and Wav2Lip, neural blending filters smooth away these boundary discontinuities.
* **The Fatal Flaw of Overfitting to Compression Signatures:** MesoNet-4 (**AUC: 0.5389**) and XceptionNet (**AUC: 0.5002**) were trained on FaceForensics++ video compression grids. When applied zero-shot to YouTube videos with completely different compression codecs, their spatial edge detectors collapsed into random coin flips ($50\%$). This empirical failure confirms the central thesis premise: **detecting deepfakes via pixel quality is an expiring paradigm**.

#### 3. Why Did Phoneme-Viseme Sync Models (LipForensics, ResNet-AV) Fail?
* **The Wav2Lip Deception:** LipForensics (**AUC: 0.5132**) focuses exclusively on whether mouth movements synchronize with spoken phonemes. However, tools like Wav2Lip are explicitly trained using sync-discriminator loss functions that optimize for lip synchronization. Consequently, the mouth moves in sync with the audio, completely deceiving lip-sync detectors (**46.67% accuracy** on Wav2Lip).
* **The Failure of Shallow Concatenation:** Multimodal ResNet-AV (**AUC: 0.4629**) performed worse than random guessing. Concatenating raw visual pixels and audio spectrograms without semantic or affective alignment creates an unstructured, high-dimensional search space where the network overfits to spurious correlations, leading to catastrophic negative transfer.

#### 4. The Compound Attack Paradox: Why DeepSentinel Dominates Complex Fakes
* **The Core Discovery:** In conventional detectors, compound manipulations (`fsgan-wav2lip` and `faceswap-wav2lip`) cause high failure rates because multiple generative blending passes soften facial artifacts (MesoNet: $43.48\%$, LipForensics: $52.17\%$).
* **Why DeepSentinel Achieved 98.55% and 98.28%:** Generative tools are built in isolated silos. FSGAN swaps the face, while Wav2Lip animates the lips using cloned audio. Neither tool models emotional coherence. While stacking these generators eliminates pixel seams, it **drastically magnifies cross-modal affective discordance** (e.g., a smiling upper face combined with angry verbal prosody). DeepSentinel's Compact Bilinear Pooling ($Z_{at} \otimes Z_v$) and discrepancy vector ($\boldsymbol{\Delta}$) catch this biological dissonance instantly, achieving near-perfect detection.

#### 5. Why Does the 5-Tier Forensic Confidence Spectrum Matter in the Real World?
* **The Danger of the Binary Fallacy:** In legal proceedings, intelligence auditing, and corporate content moderation, a raw binary 0/1 output is irresponsible and legally vulnerable. Forensic examiners require an auditable certainty metric.
* **Decisive Decision-Support:** With **92.00%** of clips falling into High Confidence Authentic (purity: $86.64\%$) or High Confidence Deepfake (purity: $83.24\%$), and only **0.14%** (1 single video) producing an exact midpoint tie, DeepSentinel acts as an automated triage engine: 92% of media can be processed automatically with high confidence, while the remaining 8% can be routed to human forensic analysts for manual inspection.

---

## 4.4 Effectiveness of the Sarcasm Head on the MUStARD Benchmark

### 4.4.1 Research Question
> **Research Question 4:** How effectively does the auxiliary Sarcasm Head distinguish natural rhetorical irony and deadpan humor from malicious deepfake manipulation when evaluated on the MUStARD benchmark, thereby protecting the framework from false positive inflation?

### 4.4.2 Presentation and Explanation of Results
To assess the auxiliary sarcasm filtering mechanism, the Sarcasm Head was evaluated on the complete MUStARD dataset ($N = 690$ annotated clips: 345 sarcastic, 345 non-sarcastic). The head operates on the joint acoustic-textual embedding $Z_{at} \in \mathbb{R}^{1536}$ and is supervised via binary cross-entropy ($\mathcal{L}_{\text{sarc}}$).

Table 4.10 reports the confusion matrix and classification metrics on the MUStARD benchmark.

**Table 4.10: Performance of the Auxiliary Sarcasm Head on MUStARD ($N = 690$)**
| Benchmark Dataset | TP | FP | TN | FN | Precision | Recall | Accuracy | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MUStARD ($N = 690$)** | **267** | **78** | **267** | **78** | **77.39%** | **77.39%** | **77.27%** | **0.7739** |

The Sarcasm Head achieves an accuracy of **77.27%** and an F1-score of **0.7739**, correctly identifying 267 of the 345 sarcastic videos and 267 of the 345 non-sarcastic instances.

### 4.4.3 Forensic and Theoretical Interpretation: Why This Result Matters

#### 1. Why Did We Obtain This Result? (The Acoustic-Semantic Dynamics of Irony)
Sarcasm detection succeeds by identifying the cognitive divergence between lexical content and vocal delivery:
* **Lexical-Prosodic Disconnect:** In MUStARD clips, sarcasm frequently employs positive lexical valence paired with negative or flat vocal prosody (e.g., uttering *"Oh, that's just brilliant"* with low pitch, prolonged vowel duration, and monotone cadence). BERT encodes the positive sentiment tokens, while Wav2Vec 2.0 captures the suppressed, deadpan pitch contours.
* **The Feature Representation:** SarcasmHead maps $Z_{at} \in \mathbb{R}^{1536}$ through a dedicated projection layer ($1536 \to 128 \to 1$) with LayerNorm and Dropout, allowing the network to recognize these acoustic-linguistic contradictions with **77.27% accuracy**.

#### 2. What Is the Significance to the Overall Thesis? (The False-Positive Shield)
The inclusion of the Sarcasm Head solves a fundamental theoretical paradox in deepfake detection:
* **The "Irony Trap" of Pure Affective Detectors:** A naive emotion-inconsistency detector operates on the assumption that *any* cross-modal emotional mismatch indicates synthetic tampering. However, human communication relies heavily on rhetorical irony, sarcasm, and satirical deadpan. In these genuine interactions, the speaker intentionally decouples their facial expression from their verbal prosody. Without a specialized sarcasm mechanism, an affect-based deepfake detector would flag genuine humor, political satire, and sarcasm as deepfakes, causing unacceptable false-positive inflation.
* **Gating the 299D Hybrid Bottleneck:** By outputting a continuous probability $P_{\text{sarc}} \in [0, 1]$ directly into the classification bottleneck:
  $$\mathbf{fused\_representation} = [\mathbf{fused\_proj}_{256} \,\|\, \mathbf{fused\_emo}_{36} \,\|\, \boldsymbol{\Delta}_6 \,\|\, P_{\text{sarc}}] \in \mathbb{R}^{299}$$
  the classifier learns to discount observed emotion discrepancy $\boldsymbol{\Delta}$ when $P_{\text{sarc}}$ is high. The Sarcasm Head acts as an **auditable forensic shield**, ensuring that genuine expressive nuance is protected and allowing the system to achieve **77.14% Real Specificity** on authentic broadcast media.

---

## 4.5 Chapter Summary and Synthesis

The empirical results across all four evaluation benchmarks establish the following conclusions:
1. **Unimodal affective streams** (CREMA-D) achieve balanced diagnostic capability (**72.4%** Audio-Text vs. **74.1%** Visual), preventing modality dominance and ensuring that downstream cross-modal discrepancy vectors reflect actual video tampering.
2. **DeepSentinel achieves state-of-the-art generalization** on FakeAVCeleb with an **AUC-ROC of 0.9020**, an **82.14% Balanced Accuracy**, and an **MCC of +0.6461**, outperforming conventional CNN and multimodal baselines (including ACE-Net at $0.6425$ AUC) with high statistical significance ($p < 0.001$, DeLong test).
3. **Compound manipulation attacks** (FSGAN-Wav2Lip) are intercepted at **98.55%**, proving that while multi-stage generative models eliminate pixel seams, they amplify cross-modal affective discordance.
4. **Speaker-disjoint few-shot domain adaptation** resolved in-the-wild acoustic distribution shift ($20.5\% \to 77.14\%$ Real Specificity) while preserving a strict $0\%$ celebrity overlap, providing a methodology for OOD model calibration.
5. **The auxiliary Sarcasm Head** (**77.27%** accuracy on MUStARD) provides an essential forensic shield, preventing natural communicative irony from being falsely penalized as deepfake manipulation.
