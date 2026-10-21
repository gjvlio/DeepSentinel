# DeepSentinel — Manual Computation of Classification Metrics
> FakeAVCeleb 350 + 350 Balanced Evaluation (N = 700)
> Decision rule: pred_cal = 1 → Deepfake, pred_cal = 0 → Real (τ = 0.50, Bayes-Optimal)

---

## Step 1 — Count TP / TN / FP / FN from Annotated CSV

Each of the 700 clips in `annotated_nonbinary_full.csv` is labelled with one of four categories:

| Symbol | Full Name | Condition | Count |
|---|---|---|---|
| **TP** | True Positive  | fake_label = 1, pred_cal = 1 (Fake correctly detected) | **305** |
| **TN** | True Negative  | fake_label = 0, pred_cal = 0 (Real correctly accepted) | **270** |
| **FP** | False Positive | fake_label = 0, pred_cal = 1 (Real wrongly flagged as Fake) | **80** |
| **FN** | False Negative | fake_label = 1, pred_cal = 0 (Fake missed, called Real) | **45** |
| | **Total** | | **700** |

### Confusion Matrix

```
                        Predicted
                   Fake (1)   Real (0)     Row Total
  Actual  Fake (1)      305         45          350   (All Fake clips)
          Real (0)       80        270          350   (All Real clips)
  Col Total            385        315          700   (Grand Total)
```

---

## Step 2 — Manual Formula Computation

### 2.1 Accuracy

> *Proportion of all clips classified correctly.*

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$= \frac{305 + 270}{305 + 270 + 80 + 45}$$

$$= \frac{575}{700}$$

$$= 0.8214 \approx \mathbf{82.14\%}$$

### 2.2 Precision (Positive Predictive Value)

> *Of all clips the model called Deepfake, how many actually were?*

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$= \frac{305}{305 + 80}$$

$$= \frac{305}{385}$$

$$= 0.7922 \approx \mathbf{79.22\%}$$

### 2.3 Recall (Sensitivity / True Positive Rate)

> *Of all actual Deepfake clips, how many did the model catch?*

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$= \frac{305}{305 + 45}$$

$$= \frac{305}{350}$$

$$= 0.8714 \approx \mathbf{87.14\%}$$

### 2.4 Specificity (True Negative Rate)

> *Of all actual Real clips, how many did the model correctly accept?*

$$\text{Specificity} = \frac{TN}{TN + FP}$$

$$= \frac{270}{270 + 80}$$

$$= \frac{270}{350}$$

$$= 0.7714 \approx \mathbf{77.14\%}$$

### 2.5 F1-Score

> *Harmonic mean of Precision and Recall — balances both.*

$$F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$= \frac{2 \times 0.7922 \times 0.8714}{0.7922 + 0.8714}$$

$$= \frac{1.3807}{1.6636}$$

$$= 0.8299 \approx \mathbf{82.99\%}$$

### 2.6 Balanced Accuracy

> *Average of Recall and Specificity — handles class imbalance. Preferred metric for equal-split datasets.*

$$\text{Balanced Accuracy} = \frac{\text{Recall} + \text{Specificity}}{2}$$

$$= \frac{0.8714 + 0.7714}{2}$$

$$= \frac{1.6429}{2}$$

$$= 0.8214 \approx \mathbf{82.14\%}$$

### 2.7 Negative Predictive Value (NPV)

> *Of all clips the model called Real, how many actually were?*

$$\text{NPV} = \frac{TN}{TN + FN}$$

$$= \frac{270}{270 + 45}$$

$$= \frac{270}{315}$$

$$= 0.8571 \approx \mathbf{85.71\%}$$

### 2.8 False Positive Rate (FPR) & False Negative Rate (FNR)

$$\text{FPR} = \frac{FP}{FP + TN} = \frac{80}{80+270} = 0.2286$$

$$\text{FNR} = \frac{FN}{FN + TP} = \frac{45}{45+305} = 0.1286$$

---

## Step 3 — Complete Metrics Summary Table

| Metric | Formula | Computation | Result |
|---|---|---|---|
| **Accuracy** | (TP+TN) / N | (305+270) / 700 | **82.14%** |
| **Precision** | TP / (TP+FP) | 305 / (305+80) | **79.22%** |
| **Recall (Sensitivity)** | TP / (TP+FN) | 305 / (305+45) | **87.14%** |
| **Specificity** | TN / (TN+FP) | 270 / (270+80) | **77.14%** |
| **F1-Score** | 2·P·R / (P+R) | 2×0.7922×0.8714 / (0.7922+0.8714) | **0.8299** |
| **Balanced Accuracy** | (Recall+Specificity) / 2 | (0.8714+0.7714) / 2 | **82.14%** |
| **NPV** | TN / (TN+FN) | 270 / (270+45) | **85.71%** |
| **FPR** | FP / (FP+TN) | 80 / (80+270) | **22.86%** |
| **FNR** | FN / (FN+TP) | 45 / (45+305) | **12.86%** |

---

## Step 4 — Raw Score Continuous Statistics

| Statistic | Real Clips (N=350) | Fake Clips (N=350) |
|---|---|---|
| Mean P(Fake) | 0.2531 | 0.8195 |
| Median P(Fake) | 0.0660 | 0.9424 |
| Std Dev | 0.3200 | 0.2757 |
| Min | 0.0286 | 0.0306 |
| Max | 0.9617 | 0.9742 |
| **Continuous Margin Gap** | | **+0.5664** |
