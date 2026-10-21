# DeepSentinel — Evaluation Analysis Scripts

## What These Scripts Do

These scripts process the raw model predictions from the 350+350 FakeAVCeleb evaluation and produce:
- A clip-by-clip annotated CSV with TP/TN/FP/FN labels
- A non-binary confidence tier CSV (5-tier graded scale)
- A manual computation report showing all metric formulas with actual numbers
- A continuous score defense document for the panel critique

---

## Scripts — Run in This Order

### Step 1 — Annotate every clip with TP / TN / FP / FN
```bash
python scripts/evaluation/annotate_predictions.py
```
**Input:** `data/eval_results/fakeavceleb_eval_predictions_350+350.csv`  
**Output:**
- `data/eval_results/annotated_350+350_TP_TN_FP_FN.csv` — per-clip annotation
- `data/eval_results/confusion_matrix_summary.txt` — confusion matrix + type breakdown
- `data/eval_results/multi_threshold_sensitivity.csv` — metrics at τ = 0.20–0.80

---

### Step 2 — Add the 5-tier confidence label (non-binary)
```bash
python scripts/evaluation/make_nonbinary_csv.py
```
**Input:** `data/eval_results/annotated_350+350_TP_TN_FP_FN.csv`  
**Output:** `data/eval_results/annotated_nonbinary_full.csv`

This adds the `confidence_tier` column to every clip:

| Tier Label | Score Range | Meaning |
|---|---|---|
| High-Confidence Real | [0.00 – 0.20) | Model is strongly certain it's real |
| Moderate Real | [0.20 – 0.40) | Model leans real |
| Inconclusive | [0.40 – 0.60) | Model cannot decide |
| Moderate Fake | [0.60 – 0.80) | Model leans fake |
| High-Confidence Fake | [0.80 – 1.00] | Model is strongly certain it's fake |

---

### Step 3 — Manual formula computation (all metrics)
```bash
python scripts/evaluation/manual_computation_report.py
```
**Input:** `data/eval_results/annotated_nonbinary_full.csv`  
**Output:** `data/eval_results/manual_computation_report.md`

Shows every metric formula with actual numbers substituted:
- Accuracy = (TP+TN)/N = (305+270)/700 = **82.14%**
- Precision = TP/(TP+FP) = 305/(305+80) = **79.22%**
- Recall = TP/(TP+FN) = 305/(305+45) = **87.14%**
- Specificity = TN/(TN+FP) = 270/(270+80) = **77.14%**
- F1 = 2×P×R/(P+R) = **0.8299**
- Balanced Accuracy = (Recall+Specificity)/2 = **82.14%**

---

### Step 4 — Continuous score distribution + defense tables
```bash
python scripts/evaluation/continuous_score_analysis.py
```
**Input:** `data/eval_results/annotated_nonbinary_full.csv`  
**Output:** `data/eval_results/continuous_score_defense.md`

Produces:
- 5-tier continuous score distribution table
- 3-zone triage summary (Definitive Real / Inconclusive / Definitive Fake)
- Multi-threshold sensitivity matrix (τ = 0.20 to 0.80)
- Word-for-word defense script for the panel

---

## Output Files Reference

| File | Description |
|---|---|
| `annotated_350+350_TP_TN_FP_FN.csv` | 700 clips with raw_score + TP/TN/FP/FN |
| `annotated_nonbinary_full.csv` | Same + **confidence_tier** (5-tier non-binary label) |
| `confusion_matrix_summary.txt` | Confusion matrix counts + per-type breakdown |
| `multi_threshold_sensitivity.csv` | TP/TN/FP/FN + all metrics at each threshold |
| `manual_computation_report.md` | Step-by-step formula computation with actual numbers |
| `continuous_score_defense.md` | Full panel defense document |

---

## Key Numbers (N = 700)

```
Confusion Matrix (τ = 0.50, Bayes-Optimal):

                   Predicted Fake   Predicted Real
  Actual Fake           305 (TP)         45 (FN)    → 350 fake clips
  Actual Real            80 (FP)        270 (TN)    → 350 real clips

Accuracy          : (305+270)/700        = 82.14%
Precision         : 305/(305+80)         = 79.22%
Recall            : 305/(305+45)         = 87.14%
Specificity       : 270/(270+80)         = 77.14%
F1-Score          : 2×0.7922×0.8714/...  = 0.8299
Balanced Accuracy : (0.8714+0.7714)/2    = 82.14%
```

```
5-Tier Confidence Distribution (all 700 clips):

  High-Confidence Real  : 267 clips (38.1%)
  Moderate Real         :  40 clips  (5.7%)
  Inconclusive          :  33 clips  (4.7%)   ← only 4.7% were uncertain!
  Moderate Fake         :  27 clips  (3.9%)
  High-Confidence Fake  : 333 clips (47.6%)

  Continuous Margin Gap : Real mean=0.2531 vs Fake mean=0.8195 (+0.5664)
```
