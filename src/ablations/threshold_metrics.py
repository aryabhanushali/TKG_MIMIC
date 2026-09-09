"""Shared threshold-based metrics (recall, F1) for the class-1-recall
hyperparameter sweeps.

Every other metric in this study (AUROC, AUPRC) is threshold-free -- a
deliberate choice given how rare these events are (0.4%-5.2% test-set
prevalence per disease/horizon). Recall and F1 require picking a decision
threshold, and with prevalence this low, recall on its own has a
degenerate maximum: flagging every patient (threshold below the lowest
score) always achieves recall=1.0, for every model and every
hyperparameter config, tying everything. `best_achievable_recall` reports
this directly so the degeneracy is visible rather than assumed; the
sweeps select on `best_achievable_f1` instead, which does not have this
problem since precision drops to the base rate as recall approaches 1.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve


def best_threshold_metrics(y: np.ndarray, s: np.ndarray) -> dict:
    """Best achievable recall (trivial, ~1.0) and best achievable F1 (real
    selection criterion) over every threshold on the precision-recall
    curve, plus the precision/recall/threshold at the best-F1 point."""
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return dict(max_recall=np.nan, best_f1=np.nan,
                    best_f1_precision=np.nan, best_f1_recall=np.nan, best_f1_threshold=np.nan)
    precision, recall, thresholds = precision_recall_curve(y, s)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0)
    best_idx = int(np.argmax(f1))
    # precision_recall_curve returns one more point (precision=1, recall=0) than thresholds
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else float(np.max(s))
    return dict(
        max_recall=float(np.max(recall)),
        best_f1=float(f1[best_idx]),
        best_f1_precision=float(precision[best_idx]),
        best_f1_recall=float(recall[best_idx]),
        best_f1_threshold=best_thresh,
    )
