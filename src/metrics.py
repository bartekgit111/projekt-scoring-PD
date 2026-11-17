"""
Metryki skuteczności i kalibracji.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score, roc_curve)

from . import config as cfg

EPS = 1e-6


def ks(y, p):
    """
    Statystyka Kołmogorowa-Smirnowa
    """
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def ece(y, p, n_bins=10):

    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    return float(sum(np.mean(idx == m) * abs(y[idx == m].mean() - p[idx == m].mean())
                     for m in range(n_bins) if np.any(idx == m)))


def ace(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    return float(sum(np.mean(idx == m) * abs(y[idx == m].mean() - p[idx == m].mean())
                     for m in range(len(edges) - 1) if np.any(idx == m)))


def metrics(y, p) -> dict:
    """Komplet metryk: dyskryminacja (AUC, Gini, KS, PR-AUC) i jakość PD (log-loss, Brier, ECE, ACE)."""
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    return {"n": len(y), "DR": y.mean(), "średnie PD": p.mean(), "AUC": auc, "Gini": 2 * auc - 1,
            "KS": ks(y, p), "PR-AUC": average_precision_score(y, p),
            "log-loss": log_loss(y, np.clip(p, EPS, 1 - EPS)), "Brier": brier_score_loss(y, p),
            "ECE": ece(y, p), "ACE": ace(y, p)}


def bootstrap_ci(y, p, B=500, seed=cfg.SEED) -> pd.DataFrame:
    """95% przedziały bootstrapowe dla AUC, Gini, KS i PR-AUC."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    w = {"AUC": [], "Gini": [], "KS": [], "PR-AUC": []}
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        auc = roc_auc_score(y[i], p[i])
        w["AUC"].append(auc); w["Gini"].append(2 * auc - 1)
        w["KS"].append(ks(y[i], p[i])); w["PR-AUC"].append(average_precision_score(y[i], p[i]))
    return pd.DataFrame({k: np.percentile(v, [2.5, 97.5]) for k, v in w.items()},
                        index=["CI 2,5%", "CI 97,5%"]).T


def bootstrap_gini_diff(y, p_a, p_b, B=500, seed=cfg.SEED):
    """Sparowany bootstrap różnicy Gini (model B - model A) na tych samych obserwacjach."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(2 * (roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i])))
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5])