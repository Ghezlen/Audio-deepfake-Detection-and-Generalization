"""Metrics. Bona fide (label 1) is the positive class. Higher score = more bona fide."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .data import NON_UNSEEN_EVAL_ATTACKS


def compute_eer(labels, scores):
    """labels: 1 bona fide / 0 spoof; scores: higher = bona fide."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def compute_auc(labels, scores):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def per_attack_eer(df):
    """df columns: attack, label, score (eval). bona fide pooled vs each spoof attack."""
    out = {}
    bona = df[df.label == 1]
    for a in sorted(df.attack[df.label == 0].unique()):
        sub = df[df.attack == a]
        y = np.concatenate([bona.label.values, sub.label.values])
        sc = np.concatenate([bona.score.values, sub.score.values])
        out[a] = compute_eer(y, sc)
    return out


def summarize_eval(df):
    """Full eval summary: overall EER/AUC, per-attack EER, unseen mean, A16/A19."""
    pae = per_attack_eer(df)
    unseen = [v for k, v in pae.items() if k not in NON_UNSEEN_EVAL_ATTACKS]
    return {
        "eval_eer": compute_eer(df.label.values, df.score.values),
        "eval_auc": compute_auc(df.label.values, df.score.values),
        "per_attack_eer": pae,
        "unseen_mean_eer": float(np.nanmean(unseen)) if unseen else float("nan"),
        "A16_eer": pae.get("A16", float("nan")),
        "A19_eer": pae.get("A19", float("nan")),
    }
