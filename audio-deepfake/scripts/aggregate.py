#!/usr/bin/env python
"""P9: read every results/runs/*/metrics.json and build tables + figures.

Outputs:
  results/tables/main_results.csv     mean/std over seeds: dev EER, eval EER, eval AUC
  results/tables/per_attack_eer.csv   eval EER per attack, mean/std over seeds, per model
  results/tables/trim_effect.csv      A_trim vs A_untrim
  results/figures/*.png
Every number comes from a metrics.json you actually produced (rule zero).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"


def load_runs():
    recs = []
    for mj in sorted(RUNS.glob("*/metrics.json")):
        with open(mj) as f:
            recs.append(json.load(f))
    if not recs:
        sys.exit(f"no metrics.json under {RUNS}. Train some runs first.")
    return recs


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    recs = load_runs()

    base = pd.DataFrame([{
        "run_id": r["run_id"], "model": r["model"], "trim": r["trim"],
        "seed": r["seed"], "dev_eer": r["dev_eer"], "eval_eer": r["eval_eer"],
        "eval_auc": r["eval_auc"], "unseen_mean_eer": r["unseen_mean_eer"],
        "A16_eer": r["A16_eer"], "A19_eer": r["A19_eer"],
    } for r in recs])

    # 1) main results
    agg = base.groupby("run_id").agg(
        model=("model", "first"), trim=("trim", "first"), n_seeds=("seed", "nunique"),
        dev_eer_mean=("dev_eer", "mean"), dev_eer_std=("dev_eer", "std"),
        eval_eer_mean=("eval_eer", "mean"), eval_eer_std=("eval_eer", "std"),
        eval_auc_mean=("eval_auc", "mean"), eval_auc_std=("eval_auc", "std"),
        unseen_mean_eer=("unseen_mean_eer", "mean"),
    ).reset_index()
    agg.to_csv(TABLES / "main_results.csv", index=False)
    print("== main_results ==")
    print(agg.to_string(index=False))

    # 2) per-attack eer
    pa_rows = []
    for r in recs:
        for atk, eer in r["per_attack_eer"].items():
            pa_rows.append(dict(run_id=r["run_id"], model=r["model"],
                                seed=r["seed"], attack=atk, eer=eer))
    pa = pd.DataFrame(pa_rows)
    pa_agg = pa.groupby(["run_id", "attack"]).agg(
        eer_mean=("eer", "mean"), eer_std=("eer", "std")).reset_index()
    pa_agg["reused_algo"] = pa_agg.attack.isin(["A16", "A19"])
    pa_agg.to_csv(TABLES / "per_attack_eer.csv", index=False)

    # 3) trim effect (A_trim vs A_untrim)
    te = agg[agg.run_id.isin(["A_trim", "A_untrim"])][
        ["run_id", "eval_eer_mean", "eval_eer_std", "dev_eer_mean"]]
    te.to_csv(TABLES / "trim_effect.csv", index=False)

    # ---- figures ----
    # eer_by_model: dev vs eval EER with error bars
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(agg)); w = 0.38
    ax.bar(x - w / 2, agg.dev_eer_mean * 100, w, yerr=agg.dev_eer_std.fillna(0) * 100,
           label="dev EER", capsize=3)
    ax.bar(x + w / 2, agg.eval_eer_mean * 100, w, yerr=agg.eval_eer_std.fillna(0) * 100,
           label="eval EER", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(agg.run_id, rotation=20, ha="right")
    ax.set_ylabel("EER (%)"); ax.set_title("Dev vs Eval EER by run"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "eer_by_model.png", dpi=150); plt.close(fig)

    # per_attack_eer grouped bars (trimmed models only, if present)
    trimmed = [rid for rid in agg.run_id if rid.endswith("_trim")]
    sub = pa_agg[pa_agg.run_id.isin(trimmed)]
    if not sub.empty:
        attacks = sorted(sub.attack.unique())
        fig, ax = plt.subplots(figsize=(11, 4))
        x = np.arange(len(attacks)); n = len(trimmed); w = 0.8 / max(n, 1)
        for j, rid in enumerate(trimmed):
            vals = [sub[(sub.run_id == rid) & (sub.attack == a)].eer_mean.mean()
                    for a in attacks]
            ax.bar(x + j * w, np.array(vals) * 100, w, label=rid)
        ax.set_xticks(x + w * (n - 1) / 2); ax.set_xticklabels(attacks, rotation=45)
        ax.set_ylabel("eval EER (%)"); ax.set_title("Per-attack eval EER (A16/A19 reuse A04/A06)")
        ax.legend(); fig.tight_layout()
        fig.savefig(FIGS / "per_attack_eer.png", dpi=150); plt.close(fig)

    # score distributions for A_trim seed 0, if available
    sc_path = RUNS / "A_trim_s0" / "eval_scores.csv"
    if sc_path.exists():
        sc = pd.read_csv(sc_path)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(sc[sc.label == 1].score, bins=60, alpha=0.6, label="bona fide", density=True)
        ax.hist(sc[sc.label == 0].score, bins=60, alpha=0.6, label="spoof", density=True)
        ax.set_xlabel("model logit (higher = bona fide)"); ax.set_ylabel("density")
        ax.set_title("Eval score distribution — A_trim seed 0"); ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "score_distributions_A_trim.png", dpi=150); plt.close(fig)

    # shortcut before/after, if the shortcut step ran
    sr = ROOT / "results" / "shortcut" / "shortcut_results.csv"
    if sr.exists():
        s = pd.read_csv(sr)
        fig, ax = plt.subplots(figsize=(9, 4))
        groups = ["duration", "silence", "loudness", "first_0.5s", "all"]
        x = np.arange(len(groups)); w = 0.2
        for k, (setting, split) in enumerate(
                [("raw", "dev"), ("raw", "eval"), ("trimmed", "dev"), ("trimmed", "eval")]):
            vals = [s[(s.setting == setting) & (s.split == split) & (s.group == g)].auc.mean()
                    for g in groups]
            ax.bar(x + k * w, vals, w, label=f"{setting}/{split}")
        ax.axhline(0.6, ls="--", c="k", lw=1, label="G1 target 0.60")
        ax.set_xticks(x + 1.5 * w); ax.set_xticklabels(groups, rotation=20)
        ax.set_ylabel("ROC-AUC"); ax.set_title("Shortcut AUC: raw vs trimmed"); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / "shortcut_before_after.png", dpi=150); plt.close(fig)

    print(f"\nwrote tables to {TABLES} and figures to {FIGS}")


if __name__ == "__main__":
    main()
