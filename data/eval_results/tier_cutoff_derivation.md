# DeepSentinel — Data-Driven 3-Tier Label Derivation
> Labels: **Deepfake** / **Likely Deepfake** / **Not Deepfake**

## Method

Since Set A (adaptation set) does not have model scores, Set B (350+350) is
split by row order to preserve zero-leakage:

| Split | Rows | Purpose |
|---|---|---|
| **Derivation half** | 1–350 | Find t_low and t_high |
| **Validation half** | 351–700 | Apply cutoffs, report precision-per-tier |

## Cutoff Selection Criteria

| Cutoff | Criterion | Meaning |
|---|---|---|
| **t_high** | FPR on real clips ≤ 2% | At most 2 in 100 real clips are wrongly called Deepfake |
| **t_low** | FNR on fake clips ≤ 5% | At least 95 in 100 fakes are caught (not missed entirely) |

## Derived Cutoffs (from Derivation Half)

| Cutoff | Value | Measured Error |
|---|---|---|
| **t_high** | **0.96** | FPR = 0.5% on real clips (target ≤ 2%) |
| **t_low** | **0.07** | FNR = 4.5% on fake clips (target ≤ 5%) |

## 3-Tier Label Scheme

```
 0.00 ─────────────── 0.07 ──────────────────── 0.96 ──── 1.00
   Not Deepfake        │     Likely Deepfake      │   Deepfake
  (Real, confident)    │  (Uncertain, review)     │  (Fake, confident)
```

| Score Range | Label | logit_calib equivalent |
|---|---|---|
| P(fake) ≥ 0.96 | **Deepfake** | ≥ 24.00 (approx.) |
| 0.07 ≤ P(fake) < 0.96 | **Likely Deepfake** | 0 to 24.00 |
| P(fake) < 0.07 | **Not Deepfake** | < 0 |

## Precision-Per-Tier (Validation Half, N=350)

| Tier | Count | Actually Fake | Precision |
|---|---|---|---|
| **Deepfake** | 35 | 35 | **100.0%** |
| **Likely Deepfake** | 232 | 149 | **64.2%** |
| **Not Deepfake** | 83 | 11 | **13.3%** |

## Full Distribution (All 700 Clips)

| Tier | Count | % of total | Actually Fake | Precision |
|---|---|---|---|---|
| **Deepfake** | 60 | 8.6% | 59 | **98.3%** |
| **Likely Deepfake** | 442 | 63.1% | 273 | **61.8%** |
| **Not Deepfake** | 198 | 28.3% | 18 | **9.1%** |

## Important Caveats

1. **Calibration:** `raw_score` is a sigmoid output, not a validated probability.
   Call it a *confidence score*, not a probability, unless ECE is reported.
2. **Base rate:** Set B is 50% fake. In real deployment, fakes are rarer,
   so precision in the 'Deepfake' tier will be lower than shown here.
3. **Harmony bonus:** The −2.70 logit adjustment shifts many scores into the
   middle range; inspect the histogram to confirm 'Likely Deepfake' is not
   artificially inflated.