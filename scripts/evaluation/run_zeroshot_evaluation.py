"""
scripts/evaluation/run_zeroshot_evaluation.py
=============================================
Zero-Shot Evaluation & Annotation Runner for DeepSentinel on FakeAVCeleb (700 clips).

Evaluates the unadapted, zero-shot base checkpoint:
    DeepSentinel (zero-shot): best_phase2_bottleneck.pt

Executes the identical multi-modal evaluation pipeline on the exact 700-clip test manifest:
    data/manifests/fakeavceleb_eval_350_350.csv
at the decision threshold tau = 0.50, and generates separate zero-shot prediction and
annotation artifacts without overwriting few-shot / adapted evaluation results.

Usage:
    python scripts/evaluation/run_zeroshot_evaluation.py --device cuda --batch_size 16
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "evaluation"

# EDIT THIS: directory of best_phase2_bottleneck.pt (zero-shot model)
DEFAULT_CHECKPOINT = "checkpoints/best_phase2_bottleneck.pt"

# EDIT THIS: path for raw zero-shot predictions CSV
DEFAULT_RAW_CSV_OUT = "data/eval_results/fakeavceleb_eval_predictions_zeroshot_350+350.csv"

# Dedicated zero-shot annotation output paths
DEFAULT_ANNOTATED_CSV_OUT = "data/eval_results/annotated_zeroshot_350+350_TP_TN_FP_FN.csv"
DEFAULT_NONBINARY_CSV_OUT = "data/eval_results/annotated_zeroshot_nonbinary_full.csv"
DEFAULT_SUMMARY_TXT_OUT   = "data/eval_results/confusion_matrix_summary_zeroshot.txt"
DEFAULT_SENSITIVITY_CSV   = "data/eval_results/multi_threshold_sensitivity_zeroshot.csv"


def classify(label: int, pred: int) -> str:
    if label == 1 and pred == 1:
        return "TP"
    elif label == 0 and pred == 0:
        return "TN"
    elif label == 0 and pred == 1:
        return "FP"
    else:
        return "FN"


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict:
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    ba = (rec + spec) / 2.0
    return dict(
        total=total, tp=tp, tn=tn, fp=fp, fn=fn,
        accuracy=acc, precision=prec, recall=rec,
        specificity=spec, f1=f1, balanced_accuracy=ba,
    )


def confidence_tier(s: float) -> str:
    pct = round(s * 100)
    if pct <= 30:
        return "High Confidence Authentic (0%-30%)"
    elif pct < 50:
        return "Moderately Confident Authentic (31%-49%)"
    elif pct == 50:
        return "Inconclusive / Neutral (50%)"
    elif pct < 70:
        return "Moderately Confident Deepfake (51%-69%)"
    else:
        return "High Confidence Deepfake (70%-100%)"


def annotate_zeroshot_results(raw_csv: Path, annot_csv: Path, nonbinary_csv: Path, summary_txt: Path, sens_csv: Path, threshold: float = 0.50) -> None:
    print(f"\n[Annotation] Annotating zero-shot predictions at threshold tau = {threshold:.2f}...")
    rows = []
    with open(raw_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    annotated = []
    for r in rows:
        label = int(r["fake_label"])
        score = float(r["score"]) if "score" in r else float(r["raw_score"])
        pred_cal = 1 if score >= threshold else 0
        annot = classify(label, pred_cal)
        annotated.append({
            "row_no": len(annotated) + 1,
            "clip_id": r["clip_id"],
            "fake_label": label,
            "method": r.get("method", "unknown"),
            "type": r.get("type", "unknown"),
            "raw_score": score,
            "pred_cal": pred_cal,
            "annotation": annot,
        })

    annot_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_no", "clip_id", "fake_label", "method", "type", "raw_score", "pred_cal", "annotation"]
    with open(annot_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotated)
    print(f"  [OK] Annotated zero-shot CSV saved -> {annot_csv}")

    # Nonbinary 5-tier classification
    nb_fields = ["row_no", "clip_id", "fake_label", "method", "type", "raw_score", "confidence_tier", "pred_cal", "annotation"]
    with open(nonbinary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=nb_fields)
        writer.writeheader()
        for a in annotated:
            writer.writerow({
                "row_no": a["row_no"],
                "clip_id": a["clip_id"],
                "fake_label": a["fake_label"],
                "method": a["method"],
                "type": a["type"],
                "raw_score": a["raw_score"],
                "confidence_tier": confidence_tier(a["raw_score"]),
                "pred_cal": a["pred_cal"],
                "annotation": a["annotation"],
            })
    print(f"  [OK] 5-Tier Non-binary zero-shot CSV saved -> {nonbinary_csv}")

    # Summary text
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for a in annotated:
        counts[a["annotation"]] += 1
    tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    m = metrics_from_counts(tp, tn, fp, fn)

    lines = [
        "=" * 65,
        "DeepSentinel (zero-shot) - FakeAVCeleb 350+350 Evaluation",
        f"Manual Confusion Matrix  (threshold tau = {threshold:.2f} / pred_cal)",
        "=" * 65,
        "",
        "                        Predicted",
        "                   Fake (1)   Real (0)",
        f"  Actual  Fake (1)   {tp:>6}     {fn:>6}   <- TP | FN",
        f"          Real (0)   {fp:>6}     {tn:>6}   <- FP | TN",
        "",
        f"  Total clips          : {m['total']}",
        f"  TP (Correct Fake)    : {tp:>4}  ({100*tp/m['total']:.2f}%)",
        f"  TN (Correct Real)    : {tn:>4}  ({100*tn/m['total']:.2f}%)",
        f"  FP (False Alarm)     : {fp:>4}  ({100*fp/m['total']:.2f}%)",
        f"  FN (Missed Fake)     : {fn:>4}  ({100*fn/m['total']:.2f}%)",
        "",
        f"  Accuracy             : {m['accuracy']:.4f}  ({m['accuracy']*100:.2f}%)",
        f"  Precision            : {m['precision']:.4f}",
        f"  Recall (Sensitivity) : {m['recall']:.4f}",
        f"  Specificity          : {m['specificity']:.4f}",
        f"  F1-Score             : {m['f1']:.4f}",
        f"  Balanced Accuracy    : {m['balanced_accuracy']:.4f}  ({m['balanced_accuracy']*100:.2f}%)",
        "=" * 65,
    ]
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [OK] Zero-shot confusion matrix summary saved -> {summary_txt}")

    # Multi-threshold sensitivity sweep
    threshs = [round(t / 100, 2) for t in range(20, 85, 5)]
    thresh_rows = []
    for tau in threshs:
        t_annot = [classify(int(r["fake_label"]), 1 if float(r.get("score", r.get("raw_score", 0))) >= tau else 0) for r in rows]
        c = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        for ca in t_annot:
            c[ca] += 1
        m_t = metrics_from_counts(c["TP"], c["TN"], c["FP"], c["FN"])
        thresh_rows.append({
            "threshold": tau,
            "TP": c["TP"],
            "TN": c["TN"],
            "FP": c["FP"],
            "FN": c["FN"],
            "accuracy": round(m_t["accuracy"] * 100, 2),
            "precision": round(m_t["precision"] * 100, 2),
            "recall": round(m_t["recall"] * 100, 2),
            "specificity": round(m_t["specificity"] * 100, 2),
            "f1_score": round(m_t["f1"], 4),
            "balanced_accuracy": round(m_t["balanced_accuracy"] * 100, 2),
        })

    tfields = ["threshold", "TP", "TN", "FP", "FN", "accuracy", "precision", "recall", "specificity", "f1_score", "balanced_accuracy"]
    with open(sens_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=tfields)
        writer.writeheader()
        writer.writerows(thresh_rows)
    print(f"  [OK] Zero-shot multi-threshold sensitivity saved -> {sens_csv}")


def main():
    parser = argparse.ArgumentParser(description="DeepSentinel Zero-Shot Cross-Dataset Benchmark Runner (FakeAVCeleb 700 clips)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help=f"Path to zero-shot checkpoint (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--manifest", type=str, default="data/manifests/fakeavceleb_eval_350_350.csv",
                        help="Path to 700-clip test manifest (default: data/manifests/fakeavceleb_eval_350_350.csv)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for evaluation (default: cuda)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Inference batch size (default: 16)")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Decision threshold (default: 0.50)")
    parser.add_argument("--save_csv", type=str, default=DEFAULT_RAW_CSV_OUT,
                        help=f"Path to save raw zero-shot predictions (default: {DEFAULT_RAW_CSV_OUT})")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip raw inference and only run annotation steps")
    args = parser.parse_args()

    python_exe = sys.executable

    print("\n" + "#" * 70)
    print("  DEEPSENTINEL — ZERO-SHOT BENCHMARK RUNNER (FakeAVCeleb 700 Clips)")
    print("#" * 70)
    print(f"  Model      : DeepSentinel (zero-shot)")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Manifest   : {args.manifest}")
    print(f"  Device     : {args.device}")
    print(f"  Batch Size : {args.batch_size}")
    print(f"  Threshold  : {args.threshold}")
    print(f"  Raw Output : {args.save_csv}")

    # Validate checkpoint existence
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path

    if not args.skip_inference:
        if not ckpt_path.exists():
            print("\n" + "=" * 70)
            print(f"[ERROR] Zero-shot checkpoint file not found: {ckpt_path}")
            print("Please place 'best_phase2_bottleneck.pt' at the path above,")
            print("or pass '--checkpoint <path>', or update 'DEFAULT_CHECKPOINT' in this script.")
            print("=" * 70 + "\n")
            sys.exit(1)

    raw_csv_out = Path(args.save_csv)
    if not raw_csv_out.is_absolute():
        raw_csv_out = REPO_ROOT / raw_csv_out
    raw_csv_out.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path

    annot_csv_out = REPO_ROOT / DEFAULT_ANNOTATED_CSV_OUT
    nonbinary_csv_out = REPO_ROOT / DEFAULT_NONBINARY_CSV_OUT
    summary_txt_out = REPO_ROOT / DEFAULT_SUMMARY_TXT_OUT
    sens_csv_out = REPO_ROOT / DEFAULT_SENSITIVITY_CSV

    if not args.skip_inference:
        eval_cmd = [
            python_exe,
            str(SCRIPTS_DIR / "evaluate_fakeavceleb.py"),
            "--checkpoint", str(ckpt_path),
            "--manifest", str(manifest_path),
            "--device", args.device,
            "--batch_size", str(args.batch_size),
            "--threshold", str(args.threshold),
            "--save_csv", str(raw_csv_out),
        ]
        print("\n" + "=" * 70)
        print("  STEP 1: Zero-Shot Multi-Modal Inference (evaluate_fakeavceleb.py)")
        print("  COMMAND: " + " ".join(eval_cmd))
        print("=" * 70)
        ret = subprocess.run(eval_cmd, cwd=str(REPO_ROOT))
        if ret.returncode != 0:
            print(f"\n[ERROR] Zero-shot evaluation failed with exit code {ret.returncode}.")
            sys.exit(ret.returncode)
    else:
        print("\n[SKIP] Skipping Step 1 (Inference) as requested. Using existing zero-shot prediction CSV.")

    # Validate raw prediction CSV row count
    with open(raw_csv_out, "r", encoding="utf-8") as f:
        raw_rows_count = sum(1 for _ in csv.DictReader(f))
    if raw_rows_count != 700:
        print(f"\n[ERROR] Zero-shot output CSV '{raw_csv_out}' must have exactly 700 rows, but found {raw_rows_count} rows.")
        sys.exit(1)

    # Step 2: Annotate zero-shot predictions
    annotate_zeroshot_results(
        raw_csv=raw_csv_out,
        annot_csv=annot_csv_out,
        nonbinary_csv=nonbinary_csv_out,
        summary_txt=summary_txt_out,
        sens_csv=sens_csv_out,
        threshold=args.threshold,
    )

    print("\n" + "#" * 70)
    print("  ZERO-SHOT EVALUATION COMPLETED SUCCESSFULLY!")
    print("#" * 70)
    print("  Generated Artifacts:")
    print(f"  - Raw Predictions CSV     : {raw_csv_out}")
    print(f"  - TP/TN/FP/FN Annotated   : {annot_csv_out}")
    print(f"  - 5-Tier Non-Binary CSV   : {nonbinary_csv_out}")
    print(f"  - Confusion Matrix Summary: {summary_txt_out}")
    print(f"  - Multi-Threshold CSV     : {sens_csv_out}")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()

