"""
metrics.py — Detection evaluation metrics for RQ1 and RQ2.

Primary (RQ1 — detection accuracy):
    Directly translates all paper classification formulas from first principles:
    Contingency Counts: TP, TN, FP, FN, Total N
    Accuracy          = (TP + TN) / (TP + TN + FP + FN)
    Precision (PPV)   = TP / (TP + FP)
    Recall (TPR)      = TP / (TP + FN)
    Specificity (TNR) = TN / (TN + FP)
    Balanced Accuracy = (Recall + Specificity) / 2
    F1-Score          = 2 * (Precision * Recall) / (Precision + Recall) = 2*TP / (2*TP + FP + FN)
    MCC               = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    NPV               = TN / (TN + FN)
    FPR               = FP / (FP + TN) = 1 - Specificity
    FNR               = FN / (FN + TP) = 1 - Recall
    AUC-ROC           = Mann-Whitney U rank-sum integral (DeLong 1988)

Secondary (RQ2 — Δ as predictor of P(fake)):
    Pearson/Spearman correlation between ||Δ|| and P(fake)

Note on Software Engineering Design:
    Zero black-box library calls (no scikit-learn, no scipy). Every formula
    expressed in the thesis manuscript is directly, visibly implemented in native
    Python/NumPy so examiners and faculty can trace the algorithmic flow.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


# ── Step 1: Foundational Contingency Extraction (TP, TN, FP, FN) ─────────────

def extract_confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[int, int, int, int, int]:
    """
    Extracts the four fundamental cardinalities of the binary contingency table:
        TP: True Positives  (Actual=1, Pred=1) -> Fake correctly caught
        TN: True Negatives  (Actual=0, Pred=0) -> Real correctly accepted
        FP: False Positives (Actual=0, Pred=1) -> Real wrongly flagged as Fake
        FN: False Negatives (Actual=1, Pred=0) -> Fake missed, called Real
        N : Total Samples   (TP + TN + FP + FN)
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    n  = tp + tn + fp + fn
    return tp, tn, fp, fn, n


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Constructs standard 2x2 confusion matrix from foundational counts:
        [[TN, FP],
         [FN, TP]]
    """
    tp, tn, fp, fn, _ = extract_confusion_counts(y_true, y_pred)
    return np.array([[tn, fp], [fn, tp]], dtype=int)


# ── Step 2: Paper Formulas Directly Expressed from TP / TN / FP / FN ─────────

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Proportion of all video clips classified correctly.
    """
    tp, tn, fp, fn, n = extract_confusion_counts(y_true, y_pred)
    return float((tp + tn) / n) if n > 0 else 0.0


def compute_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: float = 0.0,
) -> float:
    """
    Precision (Positive Predictive Value, PPV) = TP / (TP + FP)
    Of all clips predicted as Deepfake, fraction that is genuinely synthetic.
    """
    tp, tn, fp, fn, _ = extract_confusion_counts(y_true, y_pred)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float(zero_division)


def compute_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: float = 0.0,
) -> float:
    """
    Recall (Sensitivity / True Positive Rate, TPR) = TP / (TP + FN)
    Of all actual Deepfake clips, fraction that the detector successfully caught.
    """
    tp, tn, fp, fn, _ = extract_confusion_counts(y_true, y_pred)
    return float(tp / (tp + fn)) if (tp + fn) > 0 else float(zero_division)


def compute_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: float = 0.0,
) -> float:
    """
    Specificity (True Negative Rate, TNR) = TN / (TN + FP)
    Of all actual Authentic clips, fraction that the detector correctly accepted.
    """
    tp, tn, fp, fn, _ = extract_confusion_counts(y_true, y_pred)
    return float(tn / (tn + fp)) if (tn + fp) > 0 else float(zero_division)


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Balanced Accuracy = (Recall + Specificity) / 2
                      = 0.5 * [ TP / (TP + FN) + TN / (TN + FP) ]
    Unbiased performance measure that accounts equally for fake and real classes.
    """
    rec = compute_recall(y_true, y_pred)
    spec = compute_specificity(y_true, y_pred)
    return float((rec + spec) / 2.0)


def compute_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: float = 0.0,
) -> float:
    """
    F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
             = (2 * TP) / (2 * TP + FP + FN)
    Harmonic mean of precision and recall.
    """
    tp, tn, fp, fn, _ = extract_confusion_counts(y_true, y_pred)
    denom = 2 * tp + fp + fn
    return float(2.0 * tp / denom) if denom > 0 else float(zero_division)


def compute_matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Matthews Correlation Coefficient (MCC):
        MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    Balanced correlation measure ranging from -1 (total disagreement) to +1 (perfect).
    """
    tp, tn, fp, fn, _ = extract_confusion_counts(y_true, y_pred)
    num = float(tp * tn - fp * fn)
    denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float(num / denom) if denom > 0 else 0.0


def compute_npv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: float = 0.0,
) -> float:
    """
    Negative Predictive Value (NPV) = TN / (TN + FN)
    Of all clips predicted as Real, fraction that was genuinely authentic.
    """
    tp, tn, fp, fn, _ = extract_confusion_counts(y_true, y_pred)
    return float(tn / (tn + fn)) if (tn + fn) > 0 else float(zero_division)


def compute_fpr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: float = 0.0,
) -> float:
    """
    False Positive Rate (FPR) = FP / (FP + TN) = 1 - Specificity
    """
    tp, tn, fp, fn, _ = extract_confusion_counts(y_true, y_pred)
    return float(fp / (fp + tn)) if (fp + tn) > 0 else float(zero_division)


def compute_fnr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: float = 0.0,
) -> float:
    """
    False Negative Rate (FNR) = FN / (FN + TP) = 1 - Recall
    """
    tp, tn, fp, fn, _ = extract_confusion_counts(y_true, y_pred)
    return float(fn / (fn + tp)) if (fn + tp) > 0 else float(zero_division)


def compute_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    zero_division: float = 0.0,
) -> Tuple[float, float, float, int]:
    """Compatibility tuple: (Precision, Recall, F1, Support)."""
    tp, tn, fp, fn, _ = extract_confusion_counts(y_true, y_pred)
    prec = compute_precision(y_true, y_pred, zero_division=zero_division)
    rec  = compute_recall(y_true, y_pred, zero_division=zero_division)
    f1   = compute_f1(y_true, y_pred, zero_division=zero_division)
    support = int(tp + fn)
    return prec, rec, f1, support


# ── Step 3: Area Under the ROC Curve (Mann-Whitney U Rank Integral) ──────────

def compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of vector x with tied observations averaged (Sun & Xu 2014)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=float)
    order = np.argsort(x)
    z = x[order]
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[i:j] = avg_rank
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Area Under the Receiver Operating Characteristic Curve (ROC-AUC).
    
    Formula (Mann-Whitney U formulation):
        AUC = [ Sum_{i in Pos} Rank(score_i) - n_pos * (n_pos + 1) / 2 ] / (n_pos * n_neg)
    Computes exact Wilcoxon rank integral with fractional midrank tie correction.
    """
    yt = np.asarray(y_true)
    ys = np.asarray(y_score, dtype=float)
    classes = np.unique(yt)
    if len(classes) != 2:
        raise ValueError("Only one class present in y_true. ROC AUC score is not defined in that case.")
    pos_mask = (yt == 1)
    n_pos = int(np.sum(pos_mask))
    n_neg = len(yt) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes must have at least one sample to compute ROC AUC.")

    ranks = compute_midrank(ys)
    sum_ranks_pos = float(np.sum(ranks[pos_mask]))
    u_pos = sum_ranks_pos - (n_pos * (n_pos + 1.0)) / 2.0
    auc = u_pos / (n_pos * n_neg)
    return float(auc)


# ── Step 4: Pearson & Spearman Correlation (RQ2 Δ Analysis) ──────────────────

@dataclass
class CorrelationResult:
    statistic: float


def compute_pearsonr(x: np.ndarray, y: np.ndarray) -> CorrelationResult:
    """
    Pearson product-moment correlation coefficient:
        r = Sum((x - x_bar) * (y - y_bar)) / [ sqrt(Sum(x - x_bar)^2) * sqrt(Sum(y - y_bar)^2) ]
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 2 or len(ya) < 2:
        return CorrelationResult(statistic=0.0)
    xm = xa - np.mean(xa)
    ym = ya - np.mean(ya)
    num = np.sum(xm * ym)
    denom = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))
    r = float(num / denom) if denom > 0 else 0.0
    return CorrelationResult(statistic=r)


def compute_spearmanr(x: np.ndarray, y: np.ndarray) -> CorrelationResult:
    """
    Spearman rank correlation coefficient:
        Rank transforms both vectors using midranks, then computes Pearson correlation on ranks.
    """
    rx = compute_midrank(x)
    ry = compute_midrank(y)
    return compute_pearsonr(rx, ry)


# Backward-compatible aliases
confusion_matrix = compute_confusion_matrix
accuracy_score = compute_accuracy
precision_recall_fscore_support = compute_precision_recall_f1
roc_auc_score = compute_roc_auc
pearsonr = compute_pearsonr
spearmanr = compute_spearmanr


# ── Step 5: Master Evaluation Results Structure ──────────────────────────────

@dataclass
class PipelineMetrics:
    pipeline:          str
    n:                 int
    accuracy:          float
    precision:         float
    recall:            float
    specificity:       float
    f1:                float
    balanced_accuracy: float
    tp:                int
    tn:                int
    fp:                int
    fn:                int


@dataclass
class EvalResults:
    # Primary classification metrics flowing from TP/TN/FP/FN
    accuracy:          float
    precision:         float
    recall:            float
    specificity:       float
    balanced_accuracy: float
    f1:                float
    mcc:               float
    npv:               float
    fpr:               float
    fnr:               float
    auc_roc:           float
    # Contingency elements
    tp:                int
    tn:                int
    fp:                int
    fn:                int
    n:                 int
    confusion:         np.ndarray                         # 2x2 contingency matrix
    # Per-pipeline breakdown & RQ2 Δ-correlations
    per_pipeline:      List[PipelineMetrics] = field(default_factory=list)
    delta_pearson:     Optional[float] = None
    delta_spearman:    Optional[float] = None


# ── Step 6: Master Detection Evaluator Class ─────────────────────────────────

class DetectionMetrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate(
        self,
        all_logits:    torch.Tensor | np.ndarray,        # (N, 1) or (N,)
        all_labels:    torch.Tensor | np.ndarray,        # (N,)
        all_delta_norms: Optional[torch.Tensor | np.ndarray] = None,   # (N,)
        all_pipelines: Optional[List[str]]  = None,
        all_preds:     Optional[torch.Tensor | np.ndarray] = None,     # (N,) optional explicit binary decisions
    ) -> EvalResults:

        if isinstance(all_logits, np.ndarray):
            all_logits = torch.from_numpy(all_logits)
        if isinstance(all_labels, np.ndarray):
            all_labels = torch.from_numpy(all_labels)
        if all_delta_norms is not None and isinstance(all_delta_norms, np.ndarray):
            all_delta_norms = torch.from_numpy(all_delta_norms)

        if all_logits.ndim >= 2:
            logits_1d = all_logits.squeeze(1) if all_logits.shape[1] == 1 else all_logits.squeeze()
        else:
            logits_1d = all_logits

        probs  = torch.sigmoid(logits_1d).numpy()
        if all_preds is not None:
            if isinstance(all_preds, torch.Tensor):
                preds = all_preds.cpu().numpy().astype(int)
            else:
                preds = np.asarray(all_preds).astype(int)
        else:
            preds  = (probs >= self.threshold).astype(int)

        labels = all_labels.numpy().astype(int)

        # 1. Extract foundational counts
        tp, tn, fp, fn, n = extract_confusion_counts(labels, preds)


        # 2. Compute every paper formula explicitly from TP/TN/FP/FN
        acc        = compute_accuracy(labels, preds)
        prec       = compute_precision(labels, preds, zero_division=0.0)
        rec        = compute_recall(labels, preds, zero_division=0.0)
        spec       = compute_specificity(labels, preds, zero_division=0.0)
        ba         = compute_balanced_accuracy(labels, preds)
        f1         = compute_f1(labels, preds, zero_division=0.0)
        mcc        = compute_matthews_corrcoef(labels, preds)
        npv        = compute_npv(labels, preds, zero_division=0.0)
        fpr        = compute_fpr(labels, preds, zero_division=0.0)
        fnr        = compute_fnr(labels, preds, zero_division=0.0)
        cm         = compute_confusion_matrix(labels, preds)

        try:
            auc = compute_roc_auc(labels, probs)
        except ValueError:
            auc = float("nan")

        # 3. Per-pipeline breakdown
        per_pipeline: List[PipelineMetrics] = []
        if all_pipelines:
            unique = sorted(set(all_pipelines))
            for pipe in unique:
                idx = [i for i, p in enumerate(all_pipelines) if p == pipe]
                p_l = labels[idx]
                p_p = preds[idx]
                p_tp, p_tn, p_fp, p_fn, p_n = extract_confusion_counts(p_l, p_p)
                per_pipeline.append(PipelineMetrics(
                    pipeline=pipe,
                    n=p_n,
                    accuracy=compute_accuracy(p_l, p_p),
                    precision=compute_precision(p_l, p_p, zero_division=0.0),
                    recall=compute_recall(p_l, p_p, zero_division=0.0),
                    specificity=compute_specificity(p_l, p_p, zero_division=0.0),
                    f1=compute_f1(p_l, p_p, zero_division=0.0),
                    balanced_accuracy=compute_balanced_accuracy(p_l, p_p),
                    tp=p_tp,
                    tn=p_tn,
                    fp=p_fp,
                    fn=p_fn,
                ))

        # 4. RQ2 — Δ correlation
        delta_p = delta_s = None
        if all_delta_norms is not None:
            d = all_delta_norms.numpy()
            try:
                delta_p = float(compute_pearsonr(d, probs).statistic)
                delta_s = float(compute_spearmanr(d, probs).statistic)
            except Exception:
                pass

        return EvalResults(
            accuracy=acc,
            precision=prec,
            recall=rec,
            specificity=spec,
            balanced_accuracy=ba,
            f1=f1,
            mcc=mcc,
            npv=npv,
            fpr=fpr,
            fnr=fnr,
            auc_roc=auc,
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            n=n,
            confusion=cm,
            per_pipeline=per_pipeline,
            delta_pearson=delta_p,
            delta_spearman=delta_s,
        )

    def report(self, results: EvalResults) -> str:
        lines = [
            "=" * 64,
            "     DEEPSENTINEL DETECTION EVALUATION (FORMULA BREAKDOWN)",
            "=" * 64,
            f"Total Samples (N) : {results.n}",
            f"True Positives (TP): {results.tp:<6} (Fake correctly detected)",
            f"True Negatives (TN): {results.tn:<6} (Real correctly accepted)",
            f"False Positives(FP): {results.fp:<6} (Real wrongly flagged as Fake)",
            f"False Negatives(FN): {results.fn:<6} (Fake missed, called Real)",
            "-" * 64,
            f"Accuracy          = (TP+TN)/N                 : {results.accuracy * 100:6.2f}%",
            f"Precision (PPV)   = TP/(TP+FP)                : {results.precision * 100:6.2f}%",
            f"Recall (Sens/TPR) = TP/(TP+FN)                : {results.recall * 100:6.2f}%",
            f"Specificity (TNR) = TN/(TN+FP)                : {results.specificity * 100:6.2f}%",
            f"Balanced Accuracy = (Recall+Specificity)/2    : {results.balanced_accuracy * 100:6.2f}%",
            f"F1-Score          = 2*TP/(2*TP+FP+FN)         : {results.f1:.4f}",
            f"MCC Correlation   = (TP*TN-FP*FN)/sqrt(...)   : {results.mcc:.4f}",
            f"NPV               = TN/(TN+FN)                : {results.npv * 100:6.2f}%",
            f"FPR               = FP/(FP+TN)                : {results.fpr * 100:6.2f}%",
            f"FNR               = FN/(FN+TP)                : {results.fnr * 100:6.2f}%",
            f"AUC-ROC           = Mann-Whitney U rank int   : {results.auc_roc:.4f}",
            "-" * 64,
            f"Confusion Matrix:\n{results.confusion}",
        ]
        if results.per_pipeline:
            lines.append("\nPer-Pipeline Breakdown:")
            for pm in results.per_pipeline:
                lines.append(
                    f"  {pm.pipeline:14s}  n={pm.n:4d} | "
                    f"Acc={pm.accuracy*100:5.2f}% | F1={pm.f1:.4f} | "
                    f"Rec={pm.recall*100:5.2f}% | Spec={pm.specificity*100:5.2f}%"
                )
        if results.delta_pearson is not None:
            lines.append(
                f"\nRQ2 -- ||Delta|| vs P(fake): "
                f"Pearson r={results.delta_pearson:.4f}  "
                f"Spearman rho={results.delta_spearman:.4f}"
            )
        lines.append("=" * 64)
        return "\n".join(lines)



if __name__ == "__main__":
    import csv
    from pathlib import Path

    print("\n" + "=" * 70)
    print("   DEEPSENTINEL CLASSIFICATION EVALUATION (FIRST-PRINCIPLES MATH)")
    print("=" * 70)

    # Attempt to load the official benchmark results
    default_csv = Path(__file__).resolve().parents[2] / "data" / "eval_results" / "annotated_nonbinary_full.csv"
    if default_csv.exists():
        print(f"Data Source: {default_csv.name} (N = 700 clips)")
        with open(default_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        y_true = np.array([int(r["fake_label"]) for r in rows])
        y_pred = np.array([int(r["pred_cal"]) for r in rows])
        scores = np.array([float(r["raw_score"]) for r in rows])
        deltas = np.array([float(r.get("delta_norm", 0.0)) for r in rows])
        pipelines = [r.get("method") or r.get("source_pipeline", "unknown") for r in rows]

        # 1. Fetch contingency cardinalities
        tp, tn, fp, fn, n = extract_confusion_counts(y_true, y_pred)

        print("\n[STEP 1] CONTINGENCY CARDINALITIES (SOURCE OF TRUTH):")
        print(f"  True Positives  (TP) : {tp:<5}  [fake_label=1, pred=1] (Synthetic detected)")
        print(f"  True Negatives  (TN) : {tn:<5}  [fake_label=0, pred=0] (Authentic accepted)")
        print(f"  False Positives (FP) : {fp:<5}  [fake_label=0, pred=1] (Authentic misflagged)")
        print(f"  False Negatives (FN) : {fn:<5}  [fake_label=1, pred=0] (Synthetic missed)")
        print(f"  Total Clips     (N)  : {n:<5}  = TP + TN + FP + FN = {tp} + {tn} + {fp} + {fn}")

        # 2. Compute every formula step-by-step
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

        print("\n[STEP 2] RAW FORMULA DERIVATIONS (PAPER SECTION 4):")
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

        # 3. Overall DetectionMetrics object verification
        dm = DetectionMetrics(threshold=0.50)
        eps = 1e-7
        scores_clamped = np.clip(scores, eps, 1.0 - eps)
        score_logits = np.log(scores_clamped / (1.0 - scores_clamped))
        eval_res = dm.evaluate(score_logits, y_true, deltas, pipelines, all_preds=y_pred)
        print("\n[STEP 3] OFFICIAL EVALUATION REPORT:")
        print(dm.report(eval_res))

    else:
        print(f"[!] Default CSV not found at {default_csv}. Generating synthetic test vectors...")
        y_true = np.array([1]*350 + [0]*350)
        y_pred = np.array([1]*305 + [0]*45 + [1]*80 + [0]*270)
        scores = np.array([0.8]*305 + [0.3]*45 + [0.7]*80 + [0.2]*270)
        dm = DetectionMetrics(threshold=0.50)
        res = dm.evaluate(scores, y_true)
        print(dm.report(res))

