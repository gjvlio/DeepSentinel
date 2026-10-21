from pathlib import Path
"""
manual_computation_report.py
Generates a step-by-step manual computation document showing
exactly how TP/TN/FP/FN → Accuracy, Precision, Recall, F1, BA, AUC.
"""
import csv, os, statistics, math

BASE = str(Path(__file__).resolve().parents[2])
INPUT  = os.path.join(BASE, 'data', 'eval_results', 'annotated_nonbinary_full.csv')
OUTPUT = os.path.join(BASE, 'data', 'eval_results', 'manual_computation_report.md')

rows = list(csv.DictReader(open(INPUT, encoding='utf-8')))

# Count TP/TN/FP/FN from annotation column
tp = sum(1 for r in rows if r['annotation'] == 'TP')
tn = sum(1 for r in rows if r['annotation'] == 'TN')
fp = sum(1 for r in rows if r['annotation'] == 'FP')
fn = sum(1 for r in rows if r['annotation'] == 'FN')
n  = tp + tn + fp + fn

# Compute all metrics manually
accuracy    = (tp + tn) / n
precision   = tp / (tp + fp)
recall      = tp / (tp + fn)       # sensitivity
specificity = tn / (tn + fp)
f1          = 2 * precision * recall / (precision + recall)
ba          = (recall + specificity) / 2
npv         = tn / (tn + fn)       # negative predictive value
fpr         = fp / (fp + tn)       # false positive rate
fnr         = fn / (fn + tp)       # false negative rate

# score stats
reals = [float(r['raw_score']) for r in rows if int(r['fake_label']) == 0]
fakes = [float(r['raw_score']) for r in rows if int(r['fake_label']) == 1]

md = []
md.append("# DeepSentinel — Manual Computation of Classification Metrics")
md.append("> FakeAVCeleb 350 + 350 Balanced Evaluation (N = 700)")
md.append("> Decision rule: pred_cal = 1 → Deepfake, pred_cal = 0 → Real (τ = 0.50, Bayes-Optimal)")
md.append("")

md.append("---")
md.append("")
md.append("## Step 1 — Count TP / TN / FP / FN from Annotated CSV")
md.append("")
md.append("Each of the 700 clips in `annotated_nonbinary_full.csv` is labelled with one of four categories:")
md.append("")
md.append("| Symbol | Full Name | Condition | Count |")
md.append("|---|---|---|---|")
md.append(f"| **TP** | True Positive  | fake_label = 1, pred_cal = 1 (Fake correctly detected) | **{tp}** |")
md.append(f"| **TN** | True Negative  | fake_label = 0, pred_cal = 0 (Real correctly accepted) | **{tn}** |")
md.append(f"| **FP** | False Positive | fake_label = 0, pred_cal = 1 (Real wrongly flagged as Fake) | **{fp}** |")
md.append(f"| **FN** | False Negative | fake_label = 1, pred_cal = 0 (Fake missed, called Real) | **{fn}** |")
md.append(f"| | **Total** | | **{n}** |")
md.append("")

md.append("### Confusion Matrix")
md.append("")
md.append("```")
md.append("                        Predicted")
md.append("                   Fake (1)   Real (0)     Row Total")
md.append(f"  Actual  Fake (1)   {tp:>6}     {fn:>6}       {tp+fn:>6}   (All Fake clips)")
md.append(f"          Real (0)   {fp:>6}     {tn:>6}       {fp+tn:>6}   (All Real clips)")
md.append(f"  Col Total         {tp+fp:>6}     {fn+tn:>6}       {n:>6}   (Grand Total)")
md.append("```")
md.append("")

md.append("---")
md.append("")
md.append("## Step 2 — Manual Formula Computation")
md.append("")

# ---- ACCURACY
md.append("### 2.1 Accuracy")
md.append("")
md.append("> *Proportion of all clips classified correctly.*")
md.append("")
md.append("$$\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}$$")
md.append("")
md.append(f"$$= \\frac{{{tp} + {tn}}}{{{tp} + {tn} + {fp} + {fn}}}$$")
md.append("")
md.append(f"$$= \\frac{{{tp+tn}}}{{{n}}}$$")
md.append("")
md.append(f"$$= {accuracy:.4f} \\approx \\mathbf{{{accuracy*100:.2f}\\%}}$$")
md.append("")

# ---- PRECISION
md.append("### 2.2 Precision (Positive Predictive Value)")
md.append("")
md.append("> *Of all clips the model called Deepfake, how many actually were?*")
md.append("")
md.append("$$\\text{Precision} = \\frac{TP}{TP + FP}$$")
md.append("")
md.append(f"$$= \\frac{{{tp}}}{{{tp} + {fp}}}$$")
md.append("")
md.append(f"$$= \\frac{{{tp}}}{{{tp+fp}}}$$")
md.append("")
md.append(f"$$= {precision:.4f} \\approx \\mathbf{{{precision*100:.2f}\\%}}$$")
md.append("")

# ---- RECALL
md.append("### 2.3 Recall (Sensitivity / True Positive Rate)")
md.append("")
md.append("> *Of all actual Deepfake clips, how many did the model catch?*")
md.append("")
md.append("$$\\text{Recall} = \\frac{TP}{TP + FN}$$")
md.append("")
md.append(f"$$= \\frac{{{tp}}}{{{tp} + {fn}}}$$")
md.append("")
md.append(f"$$= \\frac{{{tp}}}{{{tp+fn}}}$$")
md.append("")
md.append(f"$$= {recall:.4f} \\approx \\mathbf{{{recall*100:.2f}\\%}}$$")
md.append("")

# ---- SPECIFICITY
md.append("### 2.4 Specificity (True Negative Rate)")
md.append("")
md.append("> *Of all actual Real clips, how many did the model correctly accept?*")
md.append("")
md.append("$$\\text{Specificity} = \\frac{TN}{TN + FP}$$")
md.append("")
md.append(f"$$= \\frac{{{tn}}}{{{tn} + {fp}}}$$")
md.append("")
md.append(f"$$= \\frac{{{tn}}}{{{tn+fp}}}$$")
md.append("")
md.append(f"$$= {specificity:.4f} \\approx \\mathbf{{{specificity*100:.2f}\\%}}$$")
md.append("")

# ---- F1
md.append("### 2.5 F1-Score")
md.append("")
md.append("> *Harmonic mean of Precision and Recall — balances both.*")
md.append("")
md.append("$$F_1 = \\frac{2 \\times \\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$")
md.append("")
md.append(f"$$= \\frac{{2 \\times {precision:.4f} \\times {recall:.4f}}}{{{precision:.4f} + {recall:.4f}}}$$")
md.append("")
md.append(f"$$= \\frac{{{2*precision*recall:.4f}}}{{{precision+recall:.4f}}}$$")
md.append("")
md.append(f"$$= {f1:.4f} \\approx \\mathbf{{{f1*100:.2f}\\%}}$$")
md.append("")

# ---- BALANCED ACCURACY
md.append("### 2.6 Balanced Accuracy")
md.append("")
md.append("> *Average of Recall and Specificity — handles class imbalance. Preferred metric for equal-split datasets.*")
md.append("")
md.append("$$\\text{Balanced Accuracy} = \\frac{\\text{Recall} + \\text{Specificity}}{2}$$")
md.append("")
md.append(f"$$= \\frac{{{recall:.4f} + {specificity:.4f}}}{{2}}$$")
md.append("")
md.append(f"$$= \\frac{{{recall+specificity:.4f}}}{{2}}$$")
md.append("")
md.append(f"$$= {ba:.4f} \\approx \\mathbf{{{ba*100:.2f}\\%}}$$")
md.append("")

# ---- NPV
md.append("### 2.7 Negative Predictive Value (NPV)")
md.append("")
md.append("> *Of all clips the model called Real, how many actually were?*")
md.append("")
md.append("$$\\text{NPV} = \\frac{TN}{TN + FN}$$")
md.append("")
md.append(f"$$= \\frac{{{tn}}}{{{tn} + {fn}}}$$")
md.append("")
md.append(f"$$= \\frac{{{tn}}}{{{tn+fn}}}$$")
md.append("")
md.append(f"$$= {npv:.4f} \\approx \\mathbf{{{npv*100:.2f}\\%}}$$")
md.append("")

# ---- FPR / FNR
md.append("### 2.8 False Positive Rate (FPR) & False Negative Rate (FNR)")
md.append("")
md.append("$$\\text{FPR} = \\frac{FP}{FP + TN} = \\frac{" + str(fp) + "}{" + str(fp) + "+" + str(tn) + "} = " + f"{fpr:.4f}$$")
md.append("")
md.append("$$\\text{FNR} = \\frac{FN}{FN + TP} = \\frac{" + str(fn) + "}{" + str(fn) + "+" + str(tp) + "} = " + f"{fnr:.4f}$$")
md.append("")

# ---- Summary table
md.append("---")
md.append("")
md.append("## Step 3 — Complete Metrics Summary Table")
md.append("")
md.append("| Metric | Formula | Computation | Result |")
md.append("|---|---|---|---|")
md.append(f"| **Accuracy** | (TP+TN) / N | ({tp}+{tn}) / {n} | **{accuracy*100:.2f}%** |")
md.append(f"| **Precision** | TP / (TP+FP) | {tp} / ({tp}+{fp}) | **{precision*100:.2f}%** |")
md.append(f"| **Recall (Sensitivity)** | TP / (TP+FN) | {tp} / ({tp}+{fn}) | **{recall*100:.2f}%** |")
md.append(f"| **Specificity** | TN / (TN+FP) | {tn} / ({tn}+{fp}) | **{specificity*100:.2f}%** |")
md.append(f"| **F1-Score** | 2·P·R / (P+R) | 2×{precision:.4f}×{recall:.4f} / ({precision:.4f}+{recall:.4f}) | **{f1:.4f}** |")
md.append(f"| **Balanced Accuracy** | (Recall+Specificity) / 2 | ({recall:.4f}+{specificity:.4f}) / 2 | **{ba*100:.2f}%** |")
md.append(f"| **NPV** | TN / (TN+FN) | {tn} / ({tn}+{fn}) | **{npv*100:.2f}%** |")
md.append(f"| **FPR** | FP / (FP+TN) | {fp} / ({fp}+{tn}) | **{fpr*100:.2f}%** |")
md.append(f"| **FNR** | FN / (FN+TP) | {fn} / ({fn}+{tp}) | **{fnr*100:.2f}%** |")
md.append("")

md.append("---")
md.append("")
md.append("## Step 4 — Raw Score Continuous Statistics")
md.append("")
md.append("| Statistic | Real Clips (N=350) | Fake Clips (N=350) |")
md.append("|---|---|---|")
md.append(f"| Mean P(Fake) | {statistics.mean(reals):.4f} | {statistics.mean(fakes):.4f} |")
md.append(f"| Median P(Fake) | {statistics.median(reals):.4f} | {statistics.median(fakes):.4f} |")
md.append(f"| Std Dev | {statistics.stdev(reals):.4f} | {statistics.stdev(fakes):.4f} |")
md.append(f"| Min | {min(reals):.4f} | {min(fakes):.4f} |")
md.append(f"| Max | {max(reals):.4f} | {max(fakes):.4f} |")
md.append(f"| **Continuous Margin Gap** | | **+{statistics.mean(fakes)-statistics.mean(reals):.4f}** |")
md.append("")

with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))

print(f"Saved -> {OUTPUT}")
print()
print(f"=== MANUAL COMPUTATION RESULTS ===")
print(f"  N   = {n}  (TP={tp}, TN={tn}, FP={fp}, FN={fn})")
print(f"  Accuracy          = ({tp}+{tn})/{n}              = {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision         = {tp}/({tp}+{fp})             = {precision:.4f}  ({precision*100:.2f}%)")
print(f"  Recall            = {tp}/({tp}+{fn})             = {recall:.4f}  ({recall*100:.2f}%)")
print(f"  Specificity       = {tn}/({tn}+{fp})             = {specificity:.4f}  ({specificity*100:.2f}%)")
print(f"  F1                = 2x{precision:.4f}x{recall:.4f}/({precision:.4f}+{recall:.4f}) = {f1:.4f}")
print(f"  Balanced Accuracy = ({recall:.4f}+{specificity:.4f})/2 = {ba:.4f}  ({ba*100:.2f}%)")
print(f"  NPV               = {tn}/({tn}+{fn})             = {npv:.4f}  ({npv*100:.2f}%)")
print(f"  FPR               = {fp}/({fp}+{tn})             = {fpr:.4f}  ({fpr*100:.2f}%)")
print(f"  FNR               = {fn}/({fn}+{tp})             = {fnr:.4f}  ({fnr*100:.2f}%)")