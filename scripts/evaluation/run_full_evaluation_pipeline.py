"""
scripts/evaluation/run_full_evaluation_pipeline.py
==================================================
End-to-End Master Evaluation Runner for DeepSentinel on Google Colab / Local GPU.

Executes the complete evaluation and annotation pipeline in a single command:
  1. Runs end-to-end multi-modal inference on the exact 700 balanced FakeAVCeleb test clips
     (with all updated ViT layer remappings, 5.0s audio-visual synchronization, and RetinaFace aspect ratios).
  2. Generates the raw prediction CSV:
     - data/eval_results/fakeavceleb_eval_predictions_350+350.csv
  3. Formats and annotates the clip-by-clip classifications:
     - data/eval_results/annotated_350+350_TP_TN_FP_FN.csv (TP / TN / FP / FN labels)
     - data/eval_results/annotated_nonbinary_full.csv (5-tier confidence classification)
     - data/eval_results/confusion_matrix_summary.txt
     - data/eval_results/multi_threshold_sensitivity.csv
  4. Updates comparative baseline benchmarks against ACE-Net, MesoNet, LipForensics, etc.:
     - data/eval_results/comparative_benchmark_data.json
     - docs/comparative_sota_benchmark_reference.md
  5. Updates clip-by-clip Table 8.A tally sheets:
     - data/eval_results/table_8a_fakeavceleb_700.csv
     - docs/TABLE_8A_FAKEAVCELEB_700_CLIPS.md
  6. Re-renders comparative SOTA plots:
     - data/eval_results/figures_comparative/

Usage:
  python scripts/evaluation/run_full_evaluation_pipeline.py --device cuda --batch_size 16
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "evaluation"


def run_step(step_name: str, cmd: list[str]) -> None:
    print("\n" + "=" * 70)
    print(f"  STEP: {step_name}")
    print("  COMMAND: " + " ".join(cmd))
    print("=" * 70)
    ret = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if ret.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with exit code {ret.returncode}.")
        sys.exit(ret.returncode)


def main():
    parser = argparse.ArgumentParser(description="DeepSentinel Complete End-to-End Evaluation & Annotation Runner")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_phase2_adapted.pt",
                        help="Path to adapted checkpoint (default: checkpoints/best_phase2_adapted.pt)")
    parser.add_argument("--manifest", type=str, default="data/manifests/fakeavceleb_eval_350_350.csv",
                        help="Path to 700-clip test manifest (default: data/manifests/fakeavceleb_eval_350_350.csv)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for evaluation (default: cuda)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Inference batch size (default: 16)")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip raw inference and only re-run annotation and benchmarking steps")
    args = parser.parse_args()

    python_exe = sys.executable

    print("\n" + "#" * 70)
    print("  DEEPSENTINEL — MASTER END-TO-END EVALUATION & ANNOTATION PIPELINE")
    print("#" * 70)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Manifest   : {args.manifest}")
    print(f"  Device     : {args.device}")
    print(f"  Batch Size : {args.batch_size}")

    # Step 1: Run End-to-End Model Inference
    if not args.skip_inference:
        raw_csv_out = REPO_ROOT / "data/eval_results/fakeavceleb_eval_predictions_350+350.csv"
        raw_csv_out.parent.mkdir(parents=True, exist_ok=True)

        eval_cmd = [
            python_exe,
            str(SCRIPTS_DIR / "evaluate_fakeavceleb.py"),
            "--checkpoint", args.checkpoint,
            "--manifest", args.manifest,
            "--device", args.device,
            "--batch_size", str(args.batch_size),
            "--threshold", "0.5",
            "--save_csv", str(raw_csv_out),
        ]
        run_step("1. End-to-End Multi-Modal Evaluation (Raw Inference)", eval_cmd)
    else:
        print("\n[SKIP] Skipping Step 1 (Inference) as requested. Using existing raw prediction CSV.")

    # Step 2: Annotate TP / TN / FP / FN Classifications
    annotate_cmd = [
        python_exe,
        str(SCRIPTS_DIR / "annotate_predictions.py"),
    ]
    run_step("2. Binary Classification Annotation (TP/TN/FP/FN & Sensitivity Curve)", annotate_cmd)

    # Step 3: Compute 5-Tier Continuous Confidence Scale
    nonbinary_cmd = [
        python_exe,
        str(SCRIPTS_DIR / "make_nonbinary_csv.py"),
    ]
    run_step("3. 5-Tier Forensic Confidence Tier Classification", nonbinary_cmd)

    # Step 4: Export SOTA Benchmark & DeLong Statistical Significance
    sota_cmd = [
        python_exe,
        str(SCRIPTS_DIR / "export_comparative_benchmark_reference.py"),
    ]
    run_step("4. SOTA Comparative Benchmarking & DeLong Significance Tests", sota_cmd)

    # Step 5: Build Clip-by-Clip Tally Sheets (Table 8.A)
    tally_cmd = [
        python_exe,
        str(SCRIPTS_DIR / "build_full_tally_sheets.py"),
    ]
    run_step("5. Generating Master Table 8.A Tally Sheets & Markdown Tables", tally_cmd)

    # Step 6: Render High-Resolution SOTA Comparison Figures
    try:
        plot_cmd = [
            python_exe,
            str(SCRIPTS_DIR / "plot_sota_comparisons.py"),
            "--deepsentinel", "data/eval_results/fakeavceleb_eval_predictions_350+350.csv",
            "--split", "700",
        ]
        run_step("6. Re-generating SOTA Comparison Curves and Plots", plot_cmd)
    except Exception as e:
        print(f"\n[NOTICE] Plotting step completed or skipped figure generation: {e}")

    # Summary
    print("\n" + "#" * 70)
    print("  ALL STEPS COMPLETED SUCCESSFULLY!")
    print("#" * 70)
    print("  Generated Artifacts:")
    print("  - Raw Predictions CSV     : data/eval_results/fakeavceleb_eval_predictions_350+350.csv")
    print("  - TP/TN/FP/FN Annotated   : data/eval_results/annotated_350+350_TP_TN_FP_FN.csv")
    print("  - 5-Tier Non-Binary CSV   : data/eval_results/annotated_nonbinary_full.csv")
    print("  - Confusion Matrix Report : data/eval_results/confusion_matrix_summary.txt")
    print("  - Comparative SOTA JSON   : data/eval_results/comparative_benchmark_data.json")
    print("  - Table 8.A Full Tally    : data/eval_results/table_8a_fakeavceleb_700.csv")
    print("  - Chapter 4 Ready Docs    : docs/TABLE_8A_FAKEAVCELEB_700_CLIPS.md")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()

