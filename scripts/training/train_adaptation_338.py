"""
train_adaptation_338.py
=======================
Few-Shot Domain Adaptation on FakeAVCeleb (338-Clip Cohort) for DeepSentinel.

Composition:
- 150 Real FakeAVCeleb clips
- 150 Fake FakeAVCeleb clips (Wav2Lip, FaceSwap, FSGAN)
- 38 Stage 2 Memory Anchor clips (MELD, CREMA-D, synthetic tracks)
Total = 338 clips.

Evaluated on:
- 700 Held-Out FakeAVCeleb clips (350 Real + 350 Fake) with 0% data leakage.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

try:
    import cv2
    cv2.setLogLevel(0)
except Exception:
    pass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.detection_model import DeepfakeDetector
from src.preprocessing.pipeline import PreprocessingPipeline
from scripts.evaluation.evaluate_fakeavceleb import (
    FakeAVCelebEvalDataset,
    run_inference,
    compute_metrics,
    per_method_breakdown
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def resolve_video_path(rel_path: str) -> str:
    """Resolves relative video path across local directories."""
    cand_roots = [
        REPO_ROOT / "data/raw/FakeAVCeleb_v1.2",
        REPO_ROOT / "data",
        REPO_ROOT / "data/raw",
    ]
    for root in cand_roots:
        p = root / rel_path
        if p.is_file():
            return str(p)
    return str(REPO_ROOT / rel_path)


def load_manifest(csv_path: Path) -> list[dict]:
    """Loads adaptation manifest and resolves paths."""
    clips = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["clip_id"]
            label = int(row["fake_label"])
            meth = row.get("method", "real")
            cat = row.get("type", "")
            spk = row.get("speaker_id", "unknown")
            v_rel = row.get("rel_video_path", row.get("video_path", ""))
            v_full = resolve_video_path(v_rel)

            z_at_p = REPO_ROOT / f"data/preprocessed/features/z_at/{cid}.pt"
            z_v_p = REPO_ROOT / f"data/preprocessed/features/z_v/{cid}.pt"
            cached = z_at_p.is_file() and z_v_p.is_file()

            clips.append({
                "clip_id": cid,
                "fake_label": label,
                "method": meth,
                "type": cat,
                "speaker_id": spk,
                "video_path": v_full,
                "cached": cached,
            })
    return clips


def main():
    parser = argparse.ArgumentParser(description="Few-Shot Domain Adaptation on FakeAVCeleb (338 Clips)")
    parser.add_argument("--adapt_manifest", type=str, default="data/manifests/fakeavceleb_adapt_set_338.csv")
    parser.add_argument("--eval_manifest", type=str, default="data/manifests/fakeavceleb_eval_350_350.csv")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/best_phase2_bottleneck.pt")
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/candidate_phase2_adapted_338.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=4, help="Adaptation epochs (default: 4)")
    parser.add_argument("--lr", type=float, default=5e-6, help="Adaptation learning rate (default: 5e-6)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8 for laptop GPU)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 65)
    print("  DeepSentinel — Few-Shot Adaptation (338-Clip Cohort)")
    print("=" * 65)
    print(f"  Device              : {args.device} ({torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'})")
    print(f"  Base Checkpoint     : {args.base_checkpoint}")
    print(f"  Adaptation Manifest : {args.adapt_manifest}")
    print(f"  Output Checkpoint   : {args.output_checkpoint}")
    print(f"  Epochs / Batch / LR : {args.epochs} / {args.batch_size} / {args.lr}")
    print("=" * 65)

    # 1. Load Adaptation Set
    adapt_path = REPO_ROOT / args.adapt_manifest
    if not adapt_path.exists():
        raise FileNotFoundError(f"Adaptation manifest not found: {adapt_path}")
    adapt_set = load_manifest(adapt_path)

    n_reals = sum(1 for c in adapt_set if c["fake_label"] == 0)
    n_fakes = sum(1 for c in adapt_set if c["fake_label"] == 1)
    n_anchors = sum(1 for c in adapt_set if c["speaker_id"] == "fav_spk")
    print(f"  Loaded {len(adapt_set)} adaptation clips:")
    print(f"    - Authentic FakeAVCeleb : {n_reals}")
    print(f"    - Manipulated FakeAVCeleb: {n_fakes - n_anchors}")
    print(f"    - Stage 2 Anchor Replay : {n_anchors}")

    # 2. Load Base Model
    ckpt_path = REPO_ROOT / args.base_checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {ckpt_path}")

    model = DeepfakeDetector(classifier_mode="bottleneck").to(args.device)
    model.load_backbones()
    model.to(args.device)

    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"  Loaded base checkpoint: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    # 3. Setup Adaptation Dataset & Loader
    pipeline = PreprocessingPipeline(cache_dir="data/preprocessed")
    adapt_dataset = FakeAVCelebEvalDataset(adapt_set, pipeline, no_cache=False)
    adapt_loader = DataLoader(adapt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 4. Optimizer, Scheduler & Multi-Task Loss with pos_weight adjustment
    pos_weight_val = n_reals / float(n_fakes) if n_fakes > 0 else 1.0
    pw_t = torch.tensor([pos_weight_val], device=args.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print("\n" + "=" * 65)
    print(f"  TRAINING {args.epochs} EPOCHS (Pos Weight={pos_weight_val:.4f}, Margin=1.5)")
    print("=" * 65)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        pbar = tqdm(adapt_loader, desc=f"Adaptation Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            audio_values    = batch["audio_values"].to(args.device)
            input_ids       = batch["input_ids"].to(args.device)
            attention_mask  = batch["attention_mask"].to(args.device)
            keyframe_pixels = batch["keyframe_pixels"].to(args.device)
            labels          = batch["fake_label"].to(args.device).float()

            optimizer.zero_grad()
            out = model(audio_values, input_ids, attention_mask, keyframe_pixels)
            bce_loss = criterion(out.logit.squeeze(1), labels)

            # Supervised Margin Separation: actively push Real and Fake distributions apart
            f_mask = (labels == 1)
            r_mask = (labels == 0)
            if f_mask.any() and r_mask.any():
                mean_fake = out.logit.squeeze(1)[f_mask].mean()
                mean_real = out.logit.squeeze(1)[r_mask].mean()
                margin_loss = torch.clamp(1.5 - (mean_fake - mean_real), min=0.0)
            else:
                margin_loss = out.logit.new_zeros(1).squeeze()

            loss = bce_loss + 0.20 * margin_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        scheduler.step()
        avg_loss = total_loss / len(adapt_set)
        print(f"  Epoch {epoch:02d}/{args.epochs:02d} — Adaptation Loss: {avg_loss:.4f} (LR: {scheduler.get_last_lr()[0]:.2e})")

    # 5. Save Candidate Checkpoint
    out_ckpt_path = REPO_ROOT / args.output_checkpoint
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": args.epochs,
        "model_state": model.state_dict(),
        "classifier_mode": "bottleneck",
        "adapted": True,
        "adapt_cohort_size": len(adapt_set),
    }, out_ckpt_path)
    print(f"\n✅ Saved candidate adapted checkpoint -> {out_ckpt_path}")

    # 6. Evaluate on 700-Clip Held-Out Test Set
    eval_path = REPO_ROOT / args.eval_manifest
    if not eval_path.exists():
        print(f"⚠️ Eval manifest not found: {eval_path}. Skipping evaluation.")
        return

    print("\n" + "=" * 65)
    print(f"  BENCHMARKING ON HELD-OUT 700 TEST CLIPS ({eval_path.name})")
    print("=" * 65)

    test_clips = []
    with open(eval_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["clip_id"]
            test_clips.append({
                "clip_id": cid,
                "fake_label": int(row["fake_label"]),
                "method": row.get("method", "real"),
                "type": row.get("type", ""),
                "video_path": str(REPO_ROOT / row.get("video_path", "")),
                "cached": (REPO_ROOT / f"data/preprocessed/features/z_at/{cid}.pt").is_file(),
            })

    results = run_inference(
        model=model,
        clips=test_clips,
        pipeline=pipeline,
        batch_size=args.batch_size,
        num_workers=0,
        device=args.device,
        no_cache=False,
    )

    # Save candidate predictions
    pred_out_csv = REPO_ROOT / "data/eval_results/candidate_predictions_338.csv"
    with open(pred_out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "fake_label", "method", "type", "score", "pred", "video_path"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "clip_id": r["clip_id"],
                "fake_label": r["fake_label"],
                "method": r["method"],
                "type": r["type"],
                "score": f"{r['score']:.4f}",
                "pred": 1 if r["score"] >= 0.50 else 0,
                "video_path": r.get("video_path", ""),
            })
    print(f"  Saved candidate predictions -> {pred_out_csv}")

    # Compute and display metrics
    m = compute_metrics(results, user_thresh=0.50)
    cal_t = m.get("youden_thresh", 0.50)
    eer_t = m.get("eer_thresh", 0.50)
    m_cal = compute_metrics(results, user_thresh=cal_t)

    print("\n" + "=" * 65)
    print("  338-ADAPTED CANDIDATE RESULTS VS CURRENT CHECKPOINT")
    print("=" * 65)
    print(f"  Accuracy (tau=0.50)   : {m['acc_50']*100:.2f}%  (Baseline: 79.71%)")
    print(f"  Balanced Accuracy     : {m['bal_acc_50']*100:.2f}%  (Baseline: 79.71%)")
    print(f"  AUC-ROC               : {m['auc']:.4f}  (Baseline: 0.8749)")
    print(f"  Precision (0.50)      : {m['prec_50']*100:.2f}%  (Baseline: 77.96%)")
    print(f"  Recall (0.50)         : {m['rec_50']*100:.2f}%  (Baseline: 82.86%)")
    print(f"  Specificity (0.50)    : {m['spec_50']*100:.2f}%  (Baseline: 76.57%)")
    print(f"  F1-Score (0.50)       : {m['f1_50']:.4f}  (Baseline: 0.8033)")
    print("=" * 65)

    per_method_breakdown(results, cal_thresh=cal_t)


if __name__ == "__main__":
    main()

