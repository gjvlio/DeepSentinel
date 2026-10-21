from pathlib import Path
"""
annotate_tp_tn_fp_fn.py
────────────────────────────────────────────────────────────────────────────
Produces a full clip-by-clip TP / TN / FP / FN annotation table for the
350 + 350 FakeAVCeleb evaluation split, plus a summary confusion matrix and
multi-threshold sensitivity table.

Decision rule (matching the paper):
    pred_cal == 1  →  Model says "Deepfake"
    pred_cal == 0  →  Model says "Real"

Annotation legend:
    TP  –  fake_label=1, pred_cal=1   (correctly detected fake)
    TN  –  fake_label=0, pred_cal=0   (correctly accepted real)
    FP  –  fake_label=0, pred_cal=1   (real wrongly flagged as fake)
    FN  –  fake_label=1, pred_cal=0   (fake missed, called real)

Outputs
───────
data/eval_results/annotated_350+350_TP_TN_FP_FN.csv
    Full per-clip table with annotation column added.

data/eval_results/confusion_matrix_summary.txt
    Confusion-matrix counts + standard classification metrics.

data/eval_results/multi_threshold_sensitivity.csv
    Accuracy / Precision / Recall / F1 / BA across τ = 0.20 … 0.80.
"""

import csv
import math
import os

# ── paths ──────────────────────────────────────────────────────────────────
BASE = str(Path(__file__).resolve().parents[2])
INPUT_CSV  = os.path.join(BASE, "data", "eval_results",
                          "fakeavceleb_eval_predictions_350+350.csv")
OUTPUT_CSV = os.path.join(BASE, "data", "eval_results",
                          "annotated_350+350_TP_TN_FP_FN.csv")
SUMMARY_TXT = os.path.join(BASE, "data", "eval_results",
                            "confusion_matrix_summary.txt")
THRESH_CSV  = os.path.join(BASE, "data", "eval_results",
                            "multi_threshold_sensitivity.csv")

# ── helpers ─────────────────────────────────────────────────────────────────
def classify(label: int, pred: int) -> str:
    if   label == 1 and pred == 1: return "TP"
    elif label == 0 and pred == 0: return "TN"
    elif label == 0 and pred == 1: return "FP"
    else:                          return "FN"

def metrics_from_counts(tp, tn, fp, fn):
    total  = tp + tn + fp + fn
    acc    = (tp + tn) / total if total else 0
    prec   = tp / (tp + fp)   if (tp + fp) else 0
    rec    = tp / (tp + fn)   if (tp + fn) else 0
    spec   = tn / (tn + fp)   if (tn + fp) else 0
    f1     = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0
    ba     = (rec + spec) / 2
    return dict(total=total, tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=acc, precision=prec, recall=rec,
                specificity=spec, f1=f1, balanced_accuracy=ba)

# ── read input ───────────────────────────────────────────────────────────────
rows = []
with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(f"Loaded {len(rows)} clips from {INPUT_CSV}")

# ── annotate each clip ───────────────────────────────────────────────────────
annotated = []
for r in rows:
    label    = int(r["fake_label"])
    pred_cal = int(r["pred_cal"])
    annot    = classify(label, pred_cal)
    annotated.append({
        "row_no"     : len(annotated) + 1,
        "clip_id"    : r["clip_id"],
        "fake_label" : label,
        "method"     : r["method"],
        "type"       : r["type"],
        "raw_score"  : float(r["score"]),
        "pred_cal"   : pred_cal,
        "annotation" : annot,
    })

# ── write annotated CSV ──────────────────────────────────────────────────────
fieldnames = ["row_no","clip_id","fake_label","method","type",
              "raw_score","pred_cal","annotation"]
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(annotated)
print(f"[OK] Annotated CSV saved -> {OUTPUT_CSV}")

# ── confusion matrix at τ = 0.50 (pred_cal) ─────────────────────────────────
counts = {"TP":0,"TN":0,"FP":0,"FN":0}
for a in annotated:
    counts[a["annotation"]] += 1

tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
m = metrics_from_counts(tp, tn, fp, fn)

lines = []
lines.append("=" * 62)
lines.append("DeepSentinel - FakeAVCeleb 350+350 Evaluation")
lines.append("Manual Confusion Matrix  (threshold tau = 0.50 / pred_cal)")
lines.append("=" * 62)
lines.append("")
lines.append("                        Predicted")
lines.append("                   Fake (1)   Real (0)")
lines.append(f"  Actual  Fake (1)   {tp:>6}     {fn:>6}   <- TP | FN")
lines.append(f"          Real (0)   {fp:>6}     {tn:>6}   <- FP | TN")
lines.append("")
lines.append(f"  Total clips  : {m['total']}")
lines.append(f"  TP (Correct Fake)    : {tp:>4}  ({100*tp/m['total']:.2f}%)")
lines.append(f"  TN (Correct Real)    : {tn:>4}  ({100*tn/m['total']:.2f}%)")
lines.append(f"  FP (False Alarm)     : {fp:>4}  ({100*fp/m['total']:.2f}%)")
lines.append(f"  FN (Missed Fake)     : {fn:>4}  ({100*fn/m['total']:.2f}%)")
lines.append("")
lines.append(f"  Accuracy             : {m['accuracy']:.4f}  ({m['accuracy']*100:.2f}%)")
lines.append(f"  Precision            : {m['precision']:.4f}")
lines.append(f"  Recall (Sensitivity) : {m['recall']:.4f}")
lines.append(f"  Specificity          : {m['specificity']:.4f}")
lines.append(f"  F1-Score             : {m['f1']:.4f}")
lines.append(f"  Balanced Accuracy    : {m['balanced_accuracy']:.4f}  ({m['balanced_accuracy']*100:.2f}%)")
lines.append("")

# annotation breakdown by type
from collections import defaultdict
type_annot = defaultdict(lambda: defaultdict(int))
for a in annotated:
    type_annot[a["type"]][a["annotation"]] += 1

lines.append("-" * 62)
lines.append("  Breakdown by video / audio manipulation type:")
lines.append("")
header = f"  {'Type':<32} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}"
lines.append(header)
lines.append("  " + "-" * (len(header)-2))
for vtype, cnts in sorted(type_annot.items()):
    lines.append(f"  {vtype:<32} {cnts.get('TP',0):>4} {cnts.get('TN',0):>4} "
                 f"{cnts.get('FP',0):>4} {cnts.get('FN',0):>4}")
lines.append("")
lines.append("=" * 62)

with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"[OK] Confusion matrix summary saved -> {SUMMARY_TXT}")
print("\n".join(lines))

# ── multi-threshold sensitivity ──────────────────────────────────────────────
scores  = [a["raw_score"]  for a in annotated]
labels  = [a["fake_label"] for a in annotated]
threshs = [round(t/100, 2) for t in range(20, 85, 5)]   # 0.20 … 0.80

thresh_rows = []
for tau in threshs:
    preds = [1 if s >= tau else 0 for s in scores]
    t, n, p, e = 0, 0, 0, 0
    for lbl, pr in zip(labels, preds):
        ann = classify(lbl, pr)
        if ann == "TP": t += 1
        elif ann == "TN": n += 1
        elif ann == "FP": p += 1
        else: e += 1
    m2 = metrics_from_counts(t, n, p, e)
    thresh_rows.append({
        "threshold"       : tau,
        "TP"              : t,
        "TN"              : n,
        "FP"              : p,
        "FN"              : e,
        "accuracy"        : round(m2["accuracy"], 4),
        "precision"       : round(m2["precision"], 4),
        "recall"          : round(m2["recall"], 4),
        "specificity"     : round(m2["specificity"], 4),
        "f1"              : round(m2["f1"], 4),
        "balanced_accuracy": round(m2["balanced_accuracy"], 4),
    })

tfields = ["threshold","TP","TN","FP","FN",
           "accuracy","precision","recall","specificity","f1","balanced_accuracy"]
with open(THRESH_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=tfields)
    writer.writeheader()
    writer.writerows(thresh_rows)
print(f"\n[OK] Multi-threshold sensitivity saved -> {THRESH_CSV}")

# pretty-print threshold table
print("\n Multi-Threshold Sensitivity Table")
print(f" {'tau':>5} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} "
      f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'Spec':>7} {'F1':>7} {'BA':>7}")
print(" " + "-"*70)
for r in thresh_rows:
    star = " <-" if r["threshold"] == 0.50 else ""
    print(f" {r['threshold']:>5.2f} {r['TP']:>4} {r['TN']:>4} {r['FP']:>4} {r['FN']:>4} "
          f"{r['accuracy']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f} "
          f"{r['specificity']:>7.4f} {r['f1']:>7.4f} {r['balanced_accuracy']:>7.4f}{star}")

print("\nDone.")