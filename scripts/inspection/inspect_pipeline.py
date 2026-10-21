"""
inspect_pipeline.py — Master Inspection CLI for DeepSentinel Thesis Defense & Review.

Located in: scripts/inspection/inspect_pipeline.py

Allows professors, reviewers, and evaluators to run individual modules or the full
pipeline directly from the terminal, inspecting the exact raw formulas, contingency
counts, and tensor transformations without third-party black-box abstractions.

Usage (from repository root):
    python scripts/inspection/inspect_pipeline.py --part evaluation
    python scripts/inspection/inspect_pipeline.py --part cbp
    python scripts/inspection/inspect_pipeline.py --part model
    python scripts/inspection/inspect_pipeline.py --part losses
    python scripts/inspection/inspect_pipeline.py --part significance
    python scripts/inspection/inspect_pipeline.py --part training_flow
    python scripts/inspection/inspect_pipeline.py --part all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path (scripts/inspection/ -> parent is scripts/ -> parent is project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_evaluation_inspection():
    """Module 1: Raw Contingency Counts & Step-by-Step Metric Derivations."""
    from src.evaluation.metrics import (
        DetectionMetrics,
        extract_confusion_counts,
        compute_accuracy,
        compute_precision,
        compute_recall,
        compute_specificity,
        compute_balanced_accuracy,
        compute_f1,
        compute_matthews_corrcoef,
        compute_roc_auc,
        compute_npv,
        compute_fpr,
        compute_fnr,
    )
    import csv
    import math
    import numpy as np

    print("\n" + "=" * 78)
    print("   MODULE 1: EVALUATION METRICS -- RAW CONTINGENCY & FIRST-PRINCIPLES MATH")
    print("=" * 78)
    print("References: Thesis Section 4 (Detection Evaluation on N=700 Balanced Test Set)\n")

    csv_path = PROJECT_ROOT / "data" / "eval_results" / "annotated_nonbinary_full.csv"
    if not csv_path.exists():
        print(f"[!] Warning: Annotated CSV not found at {csv_path}. Using synthetic balanced set.")
        y_true = np.array([1]*350 + [0]*350)
        y_pred = np.array([1]*305 + [0]*45 + [1]*80 + [0]*270)
        scores = np.array([0.85]*305 + [0.25]*45 + [0.75]*80 + [0.15]*270)
        pipelines = ["real"]*350 + ["wav2lip"]*350
        deltas = np.zeros(700)
    else:
        print(f"Data Source: {csv_path.name} (N = 700 clips)")
        labels, scores, preds, pipelines, deltas = [], [], [], [], []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(int(row["fake_label"]))
                scores.append(float(row["raw_score"]))
                pred_val = row.get("pred_cal") or row.get("predicted_binary") or row.get("pred") or (1 if float(row["raw_score"]) >= 0.5 else 0)
                preds.append(int(pred_val))
                pipelines.append(row.get("method") or row.get("source_pipeline", "unknown"))
                deltas.append(float(row.get("delta_norm", 0.0)))
        y_true = np.array(labels)
        scores = np.array(scores)
        y_pred = np.array(preds)
        deltas = np.array(deltas)

    tp, tn, fp, fn, n = extract_confusion_counts(y_true, y_pred)
    acc = compute_accuracy(y_true, y_pred)
    prec = compute_precision(y_true, y_pred)
    rec = compute_recall(y_true, y_pred)
    spec = compute_specificity(y_true, y_pred)
    ba = compute_balanced_accuracy(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred)
    mcc = compute_matthews_corrcoef(y_true, y_pred)
    npv = compute_npv(y_true, y_pred)
    fpr = compute_fpr(y_true, y_pred)
    fnr = compute_fnr(y_true, y_pred)
    auc = compute_roc_auc(y_true, scores)

    print("\n[STEP 1] RAW CONTINGENCY CARDINALITIES (CONFUSION MATRIX):")
    print(f"  True Positives  (TP) : {tp:<5} [fake_label=1, pred=1] (Synthetic detected)")
    print(f"  True Negatives  (TN) : {tn:<5} [fake_label=0, pred=0] (Authentic accepted)")
    print(f"  False Positives (FP) : {fp:<5} [fake_label=0, pred=1] (Authentic misflagged)")
    print(f"  False Negatives (FN) : {fn:<5} [fake_label=1, pred=0] (Synthetic missed)")
    print(f"  Total Samples   (N)  : {n:<5} = {tp} + {tn} + {fp} + {fn}\n")

    print("[STEP 2] STEP-BY-STEP ARITHMETIC FORMULA DERIVATIONS:")
    print(f"  Accuracy          = (TP + TN) / N")
    print(f"                    = ({tp} + {tn}) / {n} = {tp + tn} / {n}")
    print(f"                    = {acc:.6f}  -->  {acc * 100:.2f}%\n")

    print(f"  Precision (PPV)   = TP / (TP + FP)")
    print(f"                    = {tp} / ({tp} + {fp}) = {tp} / {tp + fp}")
    print(f"                    = {prec:.6f}  -->  {prec * 100:.2f}%\n")

    print(f"  Recall (Sens/TPR) = TP / (TP + FN)")
    print(f"                    = {tp} / ({tp} + {fn}) = {tp} / {tp + fn}")
    print(f"                    = {rec:.6f}  -->  {rec * 100:.2f}%\n")

    print(f"  Specificity (TNR) = TN / (TN + FP)")
    print(f"                    = {tn} / ({tn} + {fp}) = {tn} / {tn + fp}")
    print(f"                    = {spec:.6f}  -->  {spec * 100:.2f}%\n")

    print(f"  Balanced Accuracy = (Recall + Specificity) / 2")
    print(f"                    = ({rec:.6f} + {spec:.6f}) / 2")
    print(f"                    = {ba:.6f}  -->  {ba * 100:.2f}%\n")

    print(f"  F1-Score          = 2 * (Precision * Recall) / (Precision + Recall)")
    print(f"                    = 2 * ({prec:.6f} * {rec:.6f}) / ({prec:.6f} + {rec:.6f})")
    print(f"                    = {f1:.6f}  -->  {f1:.4f}\n")

    denom_mcc = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    print(f"  Matthews Corrcoef = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))")
    print(f"                    = ({tp}*{tn} - {fp}*{fn}) / sqrt({tp+fp} * {tp+fn} * {tn+fp} * {tn+fn})")
    print(f"                    = ({tp*tn} - {fp*fn}) / {denom_mcc:.2f}")
    print(f"                    = {mcc:+.6f}  -->  {mcc:+.4f}\n")

    print(f"  Negative Pred Val = TN / (TN + FN) = {tn} / ({tn} + {fn}) = {npv * 100:.2f}%")
    print(f"  False Pos Rate    = FP / (FP + TN) = {fp} / ({fp} + {tn}) = {fpr * 100:.2f}%")
    print(f"  False Neg Rate    = FN / (FN + TP) = {fn} / ({fn} + {tp}) = {fnr * 100:.2f}%")
    print(f"  ROC-AUC           = Mann-Whitney U Integral = {auc:.6f}  -->  {auc:.4f}")

    print("\n[STEP 3] OFFICIAL EVALUATION REPORT (PER-PIPELINE BREAKDOWN):")
    dm = DetectionMetrics(threshold=0.50)
    eps = 1e-7
    scores_clamped = np.clip(scores, eps, 1.0 - eps)
    score_logits = np.log(scores_clamped / (1.0 - scores_clamped))
    eval_res = dm.evaluate(score_logits, y_true, deltas, pipelines, all_preds=y_pred)
    print(dm.report(eval_res))
    print("=" * 78 + "\n")


def run_cbp_inspection():
    """Module 2: Compact Bilinear Pooling Mathematical Pipeline."""
    import torch
    from src.models.bilinear import (
        CompactBilinearFusion,
        compute_count_sketch,
        compute_circular_convolution_fft,
        compute_signed_sqrt_power_normalization,
        compute_l2_vector_normalization,
    )

    print("\n" + "=" * 78)
    print("   MODULE 2: COMPACT BILINEAR POOLING (CBP) -- MATHEMATICAL VERIFICATION")
    print("=" * 78)
    print("References: Pham & Pagh (2013) / Fukui et al. (2016) Outer-Product Approximation\n")

    torch.manual_seed(42)
    B = 2
    d_at = 1536
    d_v = 768
    D = 8192

    print(f"[STEP 1] INPUT FEATURE SPACES & DIMENSIONAL REDUCTION:")
    print(f"  Acoustic-Text vector  Z_at : ({B}, {d_at}) [Wav2Vec2 768 + BERT 768]")
    print(f"  Visual Facial vector  Z_v  : ({B}, {d_v})  [ViT-B/16 + GRU 768]")
    full_outer = d_at * d_v
    print(f"  Full Outer Product Dim     : {d_at} x {d_v} = {full_outer:,} dimensions")
    print(f"  CBP Target Sketch Dim  (D) : {D:,} dimensions")
    print(f"  Compression Ratio          : {full_outer / D:.1f}:1 reduction (prevents overfitting)\n")

    z_at = torch.randn(B, d_at)
    z_v = torch.randn(B, d_v)
    cbp = CompactBilinearFusion(z_at_dim=d_at, z_v_dim=d_v, output_dim=D, seed=42)

    print(f"[STEP 2] COUNT SKETCH PROJECTION (Psi_j = sum s(i) * x_i):")
    psi_at = compute_count_sketch(z_at, cbp.h_at, cbp.s_at, D)
    psi_v  = compute_count_sketch(z_v, cbp.h_v, cbp.s_v, D)
    print(f"  Psi(Z_at) sketch shape : {tuple(psi_at.shape)} | Mean: {psi_at.mean().item():+.5f} | Std: {psi_at.std().item():.5f}")
    print(f"  Psi(Z_v)  sketch shape : {tuple(psi_v.shape)}  | Mean: {psi_v.mean().item():+.5f} | Std: {psi_v.std().item():.5f}\n")

    print(f"[STEP 3] CIRCULAR CONVOLUTION VIA CONVOLUTION THEOREM (FFT):")
    print(f"  x_CBP = F^{{-1}}( F(Psi_at) (.) F(Psi_v) )")
    x_cbp = compute_circular_convolution_fft(psi_at, psi_v, D)
    print(f"  x_CBP raw shape        : {tuple(x_cbp.shape)}")
    print(f"  x_CBP raw L2 norm      : {torch.norm(x_cbp, p=2, dim=-1).tolist()}\n")

    print(f"[STEP 4] SIGNED SQUARE-ROOT POWER NORMALIZATION:")
    print(f"  y_k = sign(x_k) * sqrt(|x_k| + eps)")
    y = compute_signed_sqrt_power_normalization(x_cbp, eps=1e-5)
    print(f"  y shape                : {tuple(y.shape)}")
    print(f"  y L2 norm before L2    : {torch.norm(y, p=2, dim=-1).tolist()}\n")

    print(f"[STEP 5] L2 VECTOR NORMALIZATION (UNIT HYPER-SPHERE PROJECTION):")
    print(f"  fused = y / sqrt(sum y_k^2 + eps)")
    fused = compute_l2_vector_normalization(y, eps=1e-12)
    final_norms = torch.norm(fused, p=2, dim=-1).tolist()
    print(f"  Fused CBP shape        : {tuple(fused.shape)}")
    print(f"  Final L2 Norms         : {[f'{n:.6f}' for n in final_norms]}")
    assert all(abs(n - 1.0) < 1e-4 for n in final_norms), "Norm must equal 1.0!"
    print(f"  --> VERIFIED: All output vectors reside strictly on the unit hypersphere ||fused||_2 = 1.000000")
    print("=" * 78 + "\n")


def run_model_inspection():
    """Module 3: Multimodal Bottleneck Architecture & 299D Representation Assembly."""
    import torch
    from src.models.detection_model import (
        DeepfakeDetector,
        compute_emotion_probabilities,
        compute_emotion_discrepancy,
        compute_discrepancy_magnitude,
        compute_affect_cooccurrence_matrix,
        compute_bottleneck_projection,
        assemble_hybrid_299d_feature,
        compute_detection_probability,
    )

    print("\n" + "=" * 78)
    print("   MODULE 3: MULTIMODAL BOTTLENECK & 299D FEATURE ASSEMBLY")
    print("=" * 78)
    print("References: Thesis Section 3 (Cross-Modal Affective Incongruence Bottleneck)\n")

    torch.manual_seed(42)
    B = 2
    d_at = 1536
    d_v = 768

    model = DeepfakeDetector(classifier_mode="bottleneck")
    model.eval()

    z_at = torch.randn(B, d_at)
    z_v = torch.randn(B, d_v)

    with torch.no_grad():
        emo_a = model.emotion_head_a(z_at)
        emo_b = model.emotion_head_b(z_v)
        sarc = model.sarcasm_head(z_at)

        prob_a = compute_emotion_probabilities(emo_a)
        prob_b = compute_emotion_probabilities(emo_b)

        print("[STEP 1] EMOTION PROBABILITY DISTRIBUTIONS (SOFTMAX P in Delta^5):")
        print(f"  P_A (Acoustic-Text) : {prob_a[0].numpy().round(4)} | Sum: {prob_a[0].sum().item():.6f}")
        print(f"  P_V (Visual Face)   : {prob_b[0].numpy().round(4)} | Sum: {prob_b[0].sum().item():.6f}\n")

        delta = compute_emotion_discrepancy(prob_a, prob_b)
        delta_norm = compute_discrepancy_magnitude(delta)
        print("[STEP 2] AFFECTIVE INCONGRUENCE FORMULA DERIVATIONS:")
        print(f"  Delta = |P_A - P_V| : {delta[0].numpy().round(4)} in R^6")
        print(f"  ||Delta||_2 norm    : {delta_norm[0].item():.6f} = sqrt(sum Delta_k^2)\n")

        fused_emo = compute_affect_cooccurrence_matrix(prob_a, prob_b)
        print("[STEP 3] AFFECT CO-OCCURRENCE KRONECKER OUTER PRODUCT:")
        print(f"  fused_emo = vec(P_A (x) P_V) : shape {tuple(fused_emo.shape)} in R^36")
        print(f"  Vector sum                   : {fused_emo[0].sum().item():.6f} (= 1.000000)\n")

        fused_cbp = model.bilinear_fusion(z_at, z_v)
        fused_proj = compute_bottleneck_projection(fused_cbp, model.bilinear_proj, model.proj_ln)
        print("[STEP 4] CBP & BOTTLENECK PROJECTION:")
        print(f"  fused_cbp  : shape {tuple(fused_cbp.shape)} in R^8192 (Unit L2-norm)")
        print(f"  fused_proj : shape {tuple(fused_proj.shape)} in R^256 (GELU + LayerNorm)\n")

        x_299 = assemble_hybrid_299d_feature(fused_proj, fused_emo, delta, sarc)
        print("[STEP 5] 299D HYBRID FEATURE VECTOR ASSEMBLY (PAPER SECTION 3.4):")
        print(f"  Formula: x_299 = [ fused_proj (256) || fused_emo (36) || Delta (6) || sarc (1) ]")
        print(f"  Assembly: 256 + 36 + 6 + 1 = {256 + 36 + 6 + 1} dimensions")
        print(f"  Resulting Tensor Shape     : {tuple(x_299.shape)}\n")

        out = model._detect(z_at, z_v)
        prob_fake = compute_detection_probability(out.logit)
        print("[STEP 6] RAW LOGIT & FIRST-PRINCIPLES SIGMOID DETECTION:")
        for i in range(B):
            print(f"  Sample {i+1}: Logit = {out.logit[i].item():+.4f} --> P(fake) = 1/(1 + exp(-z)) = {prob_fake[i].item():.6f} ({prob_fake[i].item()*100:.2f}%)")
    print("=" * 78 + "\n")


def run_losses_inspection():
    """Module 4: Multi-Task Loss Objectives & Focal/Margin Formulations."""
    import torch
    from src.training.losses import (
        MultiTaskLoss,
        compute_binary_cross_entropy_with_logits,
        compute_focal_loss,
        compute_masked_categorical_cross_entropy,
        compute_contrastive_margin_loss,
    )

    print("\n" + "=" * 78)
    print("   MODULE 4: MULTI-TASK LOSS FUNCTIONS -- RAW FORMULA VERIFICATION")
    print("=" * 78)
    print("References: Thesis Section 3.5 (Joint Detection, Auxiliary Emotions, Domain, Margin)\n")

    torch.manual_seed(42)
    B = 4
    logits = torch.tensor([2.5, -2.1, 0.4, -0.2], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    bce_raw = compute_binary_cross_entropy_with_logits(logits, targets)
    fl_raw = compute_focal_loss(logits, targets, gamma=2.0, alpha=0.25)
    print("[STEP 1] DETECTION LOSS FORMULATIONS:")
    print(f"  Raw Binary Cross Entropy (BCE) per sample : {bce_raw.detach().numpy().round(5)}")
    print(f"  BCE Mean Loss                             : {bce_raw.mean().item():.6f}")
    print(f"  Focal Loss (alpha=0.25, gamma=2.0) Mean   : {fl_raw.item():.6f}")
    print(f"  --> Focal loss down-weights confident easy examples to focus on hard boundaries.\n")

    emo_logits = torch.randn(B, 6)
    emo_targets = torch.tensor([2, 0, -1, 4])
    l_emo = compute_masked_categorical_cross_entropy(emo_logits, emo_targets, ignore_index=-1)
    print("[STEP 2] MASKED EMOTION CATEGORICAL CROSS ENTROPY:")
    print(f"  Targets with missing/unlabeled mask (-1) : {emo_targets.tolist()}")
    print(f"  Masked Emotion NLL Loss                  : {l_emo.item():.6f}\n")

    fake_scores = torch.tensor([0.85, 0.92])
    real_scores = torch.tensor([0.15, 0.22])
    l_margin = compute_contrastive_margin_loss(fake_scores, real_scores, margin=1.5)
    print("[STEP 3] CONTRASTIVE MARGIN SEPARATION LOSS:")
    print(f"  Mean Fake Score: {fake_scores.mean().item():.3f} | Mean Real Score: {real_scores.mean().item():.3f}")
    print(f"  Separation Gap : {fake_scores.mean().item() - real_scores.mean().item():.3f}")
    print(f"  Margin Loss    : max(0, 1.5 - gap) = {l_margin.item():.6f}\n")

    criterion = MultiTaskLoss(
        lambda_a=0.3,
        lambda_b=0.3,
        lambda_sarcasm=0.2,
        lambda_domain=0.1,
        lambda_margin=0.05,
    )
    fake_logits_batch = logits.unsqueeze(1)
    targets_batch = targets
    emo_a_logits = torch.randn(B, 6)
    emo_b_logits = torch.randn(B, 6)
    sarc_logits = torch.randn(B, 1)
    sarc_labels = torch.tensor([1, -1, 0, -1])
    dom_logits = torch.randn(B, 5)
    dom_labels = torch.tensor([0, 1, 4, 2])

    loss_out = criterion(
        fake_logit=fake_logits_batch,
        fake_label=targets_batch,
        emotion_a_logits=emo_a_logits,
        emotion_b_logits=emo_b_logits,
        audio_emotion_label=emo_targets,
        visual_emotion_label=emo_targets,
        sarcasm_logit=sarc_logits,
        sarcasm_label=sarc_labels,
        domain_logits=dom_logits,
        domain_label=dom_labels,
    )

    print("[STEP 4] FULL MULTI-TASK TOTAL LOSS COMPOSITION:")
    print(f"  L_det     (Detection BCE)          : {loss_out.bce.item():.6f}")
    print(f"  L_emo_a   (Audio Emotion NLL)      : {loss_out.emotion_a.item():.6f} (lambda=0.30)")
    print(f"  L_emo_b   (Visual Emotion NLL)     : {loss_out.emotion_b.item():.6f} (lambda=0.30)")
    print(f"  L_sarc    (Sarcasm BCE)            : {loss_out.sarcasm.item():.6f} (lambda=0.20)")
    print(f"  L_domain  (Domain Adversarial NLL) : {loss_out.domain.item():.6f} (lambda=0.10)")
    print(f"  L_margin  (Cluster Separation)     : {loss_out.margin.item():.6f} (lambda=0.05)")
    print("  " + "-" * 60)
    print(f"  TOTAL MULTI-TASK LOSS              : {loss_out.total.item():.6f}")
    print("=" * 78 + "\n")


def run_significance_inspection():
    """Module 5: Paired DeLong's AUC Test & Paired Bootstrap Scorecard vs Baselines."""
    import csv
    import numpy as np
    from src.evaluation.significance import delong_test, paired_bootstrap_scorecard, print_scorecard

    print("\n" + "=" * 84)
    print("   MODULE 5: STATISTICAL SIGNIFICANCE -- DEEPSENTINEL VS BASELINES (PAIRED)")
    print("=" * 84)
    print("References: Sun & Xu (2014) Fast DeLong Algorithm + Efron Paired Bootstrap (N=2,000)\n")

    csv_path = PROJECT_ROOT / "data" / "eval_results" / "preds_acenet_adapted_700.csv"
    if csv_path.exists():
        print(f"Loading paired evaluation results from: {csv_path.name} (N = 700 clips)")
        labels, scores_ds, scores_ace = [], [], []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(int(row["fake_label"]))
                scores_ds.append(float(row["deepsentinel_score"]))
                scores_ace.append(float(row["acenet_score"]))

        labels = np.array(labels)
        scores_ds = np.array(scores_ds)
        scores_ace = np.array(scores_ace)

        delong_res = delong_test(labels, scores_ds, scores_ace)
        scorecard = paired_bootstrap_scorecard(labels, scores_ds, scores_ace, thresh=0.5, n_boot=2000, seed=42)
        print_scorecard("DeepSentinel", "ACE-Net", delong_res, scorecard)
    else:
        print(f"[!] Paired CSV not found at {csv_path}. Running with synthetic paired benchmark...")
        rng = np.random.default_rng(42)
        y = np.array([1]*350 + [0]*350)
        s_a = np.clip(y * 0.8 + rng.normal(0, 0.2, 700), 0.01, 0.99)
        s_b = np.clip(y * 0.6 + rng.normal(0, 0.3, 700), 0.01, 0.99)
        delong_res = delong_test(y, s_a, s_b)
        scorecard = paired_bootstrap_scorecard(y, s_a, s_b, thresh=0.5, n_boot=1000, seed=42)
        print_scorecard("Model_A", "Model_B", delong_res, scorecard)
    print("=" * 84 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="DeepSentinel Pipeline & Raw Formula Inspection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/inspection/inspect_pipeline.py --part evaluation
  python scripts/inspection/inspect_pipeline.py --part cbp
  python scripts/inspection/inspect_pipeline.py --part model
  python scripts/inspection/inspect_pipeline.py --part losses
  python scripts/inspection/inspect_pipeline.py --part significance
  python scripts/inspection/inspect_pipeline.py --part training_flow
  python scripts/inspection/inspect_pipeline.py --part all
        """,
    )
    parser.add_argument(
        "--part",
        type=str,
        default="evaluation",
        choices=["evaluation", "cbp", "model", "losses", "significance", "training_flow", "all"],
        help="Pipeline module to inspect (default: evaluation)",
    )
    parser.add_argument(
        "--clips",
        type=int,
        default=8,
        help="Number of demo clips for training flow demonstration (default: 8)",
    )

    args = parser.parse_args()

    if args.part in ("evaluation", "all"):
        run_evaluation_inspection()
    if args.part in ("cbp", "all"):
        run_cbp_inspection()
    if args.part in ("model", "all"):
        run_model_inspection()
    if args.part in ("losses", "all"):
        run_losses_inspection()
    if args.part in ("significance", "all"):
        run_significance_inspection()
    if args.part in ("training_flow", "all"):
        from scripts.inspection.demo_training_flow import run_training_pipeline_demonstration
        run_training_pipeline_demonstration(num_clips=args.clips, stage="all")


if __name__ == "__main__":
    main()
