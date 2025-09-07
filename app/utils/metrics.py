import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    confusion_matrix, precision_score, recall_score, f1_score, brier_score_loss
)

def compute_all_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= thr).astype(int)

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc  = average_precision_score(y_true, y_prob)
    acc     = (y_pred == y_true).mean()
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    cm      = confusion_matrix(y_true, y_pred)

    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    precs, recs, pr_thr = precision_recall_curve(y_true, y_prob)

    # sensitivity at 90% specificity and specificity at 90% sensitivity
    spec = 1 - fpr
    sens_at_90spec = tpr[spec >= 0.90].max() if np.any(spec >= 0.90) else None
    spec_at_90sens = spec[tpr >= 0.90].max() if np.any(tpr >= 0.90) else None

    calib = brier_score_loss(y_true, y_prob)

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "sensitivity_at_90_specificity": sens_at_90spec,
        "specificity_at_90_sensitivity": spec_at_90sens,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thr": roc_thr.tolist()},
        "pr_curve": {"precision": precs.tolist(), "recall": recs.tolist(), "thr": pr_thr.tolist()},
        "brier": float(calib)
    }
