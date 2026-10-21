"""
significance.py — Statistical comparison of two frameworks evaluated on the
SAME test set ("Battle of the Frameworks": DeepSentinel vs. reimplemented
ACE-Net baseline, both scored on identical FakeAVCeleb clips).

Two tests, both paired (same clips → correlated, not independent samples):

  delong_test()             — DeLong (1988) / Sun & Xu (2014) fast algorithm.
                                Paired significance test on AUC-ROC. Standard
                                in the deepfake-detection literature.
  paired_bootstrap_scorecard() — resamples the SAME indices for both
                                frameworks each iteration, tracks the metric
                                difference. Covers Accuracy/Precision/Recall/F1
                                (metrics DeLong doesn't reach) with a p-value.

Note on Software Engineering Design:
    Statistical significance formulas (DeLong variance-covariance matrix,
    paired bootstrap confidence intervals, and normal p-values) are implemented
    natively without third-party black-box libraries (scipy, scikit-learn).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .metrics import compute_roc_auc, extract_confusion_counts


# ── DeLong's test ────────────────────────────────────────────────────────────

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of x with ties averaged (Sun & Xu 2014, Algorithm 2)."""
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(scores: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """
    scores: (k, m+n) — k score vectors (rows), positive-class columns first.
    Returns (aucs (k,), covariance (k,k)) per Sun & Xu (2014).
    """
    m = n_pos
    n = scores.shape[1] - m
    k = scores.shape[0]

    pos = scores[:, :m]
    neg = scores[:, m:]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(scores[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, cov


def _standard_normal_sf(z: float) -> float:
    """
    Survival function (1 - CDF) of standard normal distribution:
        P(Z > z) = 0.5 * erfc(z / sqrt(2))
    Implemented via standard library math.erfc for high numerical precision.
    """
    return 0.5 * math.erfc(z / math.sqrt(2.0))


@dataclass
class DeLongResult:
    auc_a: float
    auc_b: float
    z: float
    p_value: float


def delong_test(y_true: Sequence[int], scores_a: Sequence[float], scores_b: Sequence[float]) -> DeLongResult:
    """
    Paired DeLong test comparing two correlated ROC-AUCs on the SAME clips.
    y_true must be binary {0, 1}. scores_a/scores_b are P(fake) per clip,
    same order, from framework A and framework B respectively.
    """
    y = np.asarray(y_true)
    assert set(np.unique(y).tolist()) <= {0, 1}, "y_true must be binary {0,1}"
    order = np.argsort(-y)  # positives (label=1) first
    n_pos = int(y[order].sum())

    stacked = np.vstack([np.asarray(scores_a)[order], np.asarray(scores_b)[order]])
    aucs, cov = _fast_delong(stacked, n_pos)

    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = 0.0 if var <= 0 else diff / np.sqrt(var)
    # Two-sided p-value from standard normal distribution
    p = 2.0 * _standard_normal_sf(abs(float(z)))
    return DeLongResult(auc_a=float(aucs[0]), auc_b=float(aucs[1]), z=float(z), p_value=float(p))


# ── Paired bootstrap scorecard (Accuracy/Precision/Recall/F1/AUC) ───────────

def binary_metrics(y: np.ndarray, scores: np.ndarray, thresh: float) -> dict:
    preds = (scores >= thresh).astype(int)
    tp, tn, fp, fn, n = extract_confusion_counts(y, preds)
    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ba = (rec + spec) / 2.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    try:
        auc = compute_roc_auc(y, scores) if len(np.unique(y)) == 2 else float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": acc, "precision": prec, "recall": rec,
        "specificity": spec, "balanced_accuracy": ba, "f1": f1, "auc": auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


@dataclass
class MetricComparison:
    name: str
    value_a: float
    value_b: float
    diff: float          # a - b
    ci_lo: float
    ci_hi: float
    p_value: float
    significant: bool


def paired_bootstrap_scorecard(
    y_true: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    thresh: float = 0.5,
    n_boot: int = 10_000,
    seed: int = 42,
    metric_names: tuple = ("accuracy", "precision", "recall", "f1", "auc"),
) -> list[MetricComparison]:
    """
    Resamples the SAME indices for both frameworks each iteration (paired),
    recomputes each metric for both, tracks the difference distribution.
    95% CI excluding 0 -> significant. p-value = 2 * min(P(diff<=0), P(diff>=0)).
    """
    y = np.asarray(y_true)
    a = np.asarray(scores_a)
    b = np.asarray(scores_b)
    n = len(y)
    rng = np.random.default_rng(seed)

    point_a = binary_metrics(y, a, thresh)
    point_b = binary_metrics(y, b, thresh)

    diffs = {m: [] for m in metric_names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        ma = binary_metrics(yb, a[idx], thresh)
        mb = binary_metrics(yb, b[idx], thresh)
        for m in metric_names:
            diffs[m].append(ma[m] - mb[m])

    out = []
    for m in metric_names:
        d = np.array(diffs[m], dtype=float)
        d = d[~np.isnan(d)]
        if len(d) == 0:
            continue
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        out.append(MetricComparison(
            name=m, value_a=point_a[m], value_b=point_b[m],
            diff=point_a[m] - point_b[m],
            ci_lo=float(lo), ci_hi=float(hi), p_value=float(min(p, 1.0)),
            significant=bool(lo > 0 or hi < 0),
        ))
    return out


def print_scorecard(name_a: str, name_b: str, delong: DeLongResult, bootstrap: list[MetricComparison]) -> None:
    width = 84
    print(f"\n{'=' * width}")
    print(f"  BATTLE OF THE FRAMEWORKS - {name_a}  vs  {name_b}")
    print(f"{'=' * width}")
    print(f"  {'Metric':<12}{name_a:>16}{name_b:>16}{'Diff':>10}{'95% CI (diff)':>20}{'p-value':>10}  Sig")
    print(f"  {'-' * (width - 2)}")
    for mc in bootstrap:
        sig = "*" if mc.significant else ""
        ci = f"[{mc.ci_lo * 100:6.2f}%, {mc.ci_hi * 100:6.2f}%]"
        print(f"  {mc.name:<12}{mc.value_a * 100:>15.2f}%{mc.value_b * 100:>15.2f}%"
              f"{mc.diff * 100:>9.2f}%  {ci:>20}{mc.p_value:>10.4f}  {sig}")
    print(f"  {'-' * (width - 2)}")
    sig_str = "SIGNIFICANT (p < 0.05)" if delong.p_value < 0.05 else "not significant (p >= 0.05)"
    print(f"  DeLong's test (AUC, paired):  AUC_a={delong.auc_a:.4f}  AUC_b={delong.auc_b:.4f}  "
          f"z={delong.z:.4f}  p={delong.p_value:.4f}  -> {sig_str}")
    print(f"{'=' * width}\n")


if __name__ == "__main__":
    import csv
    from pathlib import Path

    print("\n" + "=" * 84)
    print("   STATISTICAL SIGNIFICANCE TESTING -- DEEPSENTINEL VS BASELINES (PAIRED)")
    print("=" * 84)
    print("Sun & Xu (2014) Fast DeLong Algorithm + Efron Paired Bootstrap (N=2,000)\n")

    csv_path = Path("data/eval_results/preds_acenet_adapted_700.csv")
    if not csv_path.exists():
        csv_path = Path(__file__).resolve().parents[2] / "data" / "eval_results" / "preds_acenet_adapted_700.csv"

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

        # 1. Paired DeLong test for AUC
        delong_res = delong_test(labels, scores_ds, scores_ace)

        # 2. Paired bootstrap scorecard
        scorecard = paired_bootstrap_scorecard(labels, scores_ds, scores_ace, thresh=0.5, n_boot=2000, seed=42)

        # 3. Print report
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

