#!/usr/bin/env python
"""P3/P5: shortcut check. Extract non-speech features, cache them, train HGB, score.

Usage:
  python scripts/run_shortcut.py --setting raw     --data_root <ROOT>
  python scripts/run_shortcut.py --setting trimmed --data_root <ROOT>
Appends rows (setting, group, split, auc, eer) to results/shortcut/shortcut_results.csv.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.shortcut import extract_features, run_classifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["raw", "trimmed"], required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--recompute", action="store_true",
                    help="ignore cached features parquet and recompute")
    args = ap.parse_args()

    root = Path(args.data_root) / "LA"
    df = pd.read_csv(args.manifest)
    cache = ROOT / "results" / "shortcut" / f"features_{args.setting}.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists() and not args.recompute:
        print(f"loading cached features {cache}")
        feat = pd.read_parquet(cache)
    else:
        print(f"extracting '{args.setting}' features for {len(df)} files ...")
        feat = extract_features(df, root, args.setting, n_jobs=args.n_jobs)
        try:
            feat.to_parquet(cache)
        except Exception:
            cache = cache.with_suffix(".csv"); feat.to_csv(cache, index=False)
        print(f"cached features -> {cache}")

    errs = int((feat.get("_err", "") != "").sum())
    if errs:
        print(f"WARNING: {errs} files errored during feature extraction.")

    print("training HistGradientBoosting per feature group ...")
    res = run_classifier(feat)
    res.insert(0, "setting", args.setting)

    out = ROOT / "results" / "shortcut" / "shortcut_results.csv"
    if out.exists():
        prev = pd.read_csv(out)
        prev = prev[prev.setting != args.setting]       # overwrite this setting
        res = pd.concat([prev, res], ignore_index=True)
    res.to_csv(out, index=False)
    print(f"wrote {out}")
    print(res[res.setting == args.setting].to_string(index=False))


if __name__ == "__main__":
    main()
