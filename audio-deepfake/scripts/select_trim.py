#!/usr/bin/env python
"""P4 (proper): choose trimming parameters by the DEV AUC of the 'all' shortcut group.

Grid: thresh_db in {30,35,40}, margin_s in {0.02,0.05,0.10}. Lower dev 'all' AUC is
better (less leakage). Writes results/shortcut/trim_grid.csv and prints the winner.
Eval is NOT used here. This is heavier (re-extracts features per grid point on a
subsample); use --subsample to keep it fast, then confirm the winner on full data.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import compute_trim_columns          # noqa: E402
from src.shortcut import extract_features, run_classifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--subsample", type=int, default=6000,
                    help="train+dev files per class per split to use (0 = all)")
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    root = Path(args.data_root) / "LA"
    df = pd.read_csv(args.manifest)
    df = df[df.split.isin(["train", "dev"])].copy()

    if args.subsample:
        parts = []
        for sp in ["train", "dev"]:
            s = df[df.split == sp]
            for lbl in (0, 1):
                parts.append(s[s.label == lbl].sample(
                    n=min(args.subsample, (s.label == lbl).sum()), random_state=0))
        df = pd.concat(parts, ignore_index=True)
        print(f"grid search on a subsample of {len(df)} files")

    grid = [(t, m) for t in (30.0, 35.0, 40.0) for m in (0.02, 0.05, 0.10)]
    rows = []
    for thresh_db, margin_s in grid:
        tdf = compute_trim_columns(df, root, thresh_db=thresh_db,
                                   margin_s=margin_s, n_jobs=args.n_jobs)
        feat = extract_features(tdf, root, "trimmed", n_jobs=args.n_jobs)
        res = run_classifier(feat)
        dev_all = res[(res.group == "all") & (res.split == "dev")].auc.iloc[0]
        rows.append(dict(thresh_db=thresh_db, margin_s=margin_s, dev_all_auc=dev_all))
        print(f"thresh_db={thresh_db} margin_s={margin_s}  dev all AUC={dev_all:.4f}")

    out = pd.DataFrame(rows).sort_values("dev_all_auc").reset_index(drop=True)
    path = ROOT / "results" / "shortcut" / "trim_grid.csv"
    out.to_csv(path, index=False)
    best = out.iloc[0]
    print(f"\nwrote {path}")
    print(f"BEST: thresh_db={best.thresh_db} margin_s={best.margin_s} "
          f"(dev all AUC={best.dev_all_auc:.4f})")
    print("Now run: python scripts/compute_trim.py --data_root <ROOT> "
          f"--thresh_db {best.thresh_db} --margin_s {best.margin_s}")


if __name__ == "__main__":
    main()
