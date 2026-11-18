#!/usr/bin/env python
"""P4: compute trim_start/trim_end for every file and write them into the manifest.

Default parameters (thresh_db=35, margin_s=0.05) are the guide's start values.
Use scripts/select_trim.py first if you want to pick them by dev AUC (gate G1).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import compute_trim_columns  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--thresh_db", type=float, default=35.0)
    ap.add_argument("--margin_s", type=float, default=0.05)
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    root = Path(args.data_root) / "LA"
    df = pd.read_csv(args.manifest)
    print(f"Computing trim indices (thresh_db={args.thresh_db}, "
          f"margin_s={args.margin_s}) for {len(df)} files ...")
    out = compute_trim_columns(df, root, thresh_db=args.thresh_db,
                               margin_s=args.margin_s, n_jobs=args.n_jobs)
    out.to_csv(args.manifest, index=False)
    kept = (out.trim_end - out.trim_start) / out.sr
    print(f"wrote {args.manifest}")
    print(f"trimmed duration: median {kept.median():.2f}s  mean {kept.mean():.2f}s")
    fails = int((out.get("trim_error", "") != "").sum())
    if fails:
        print(f"WARNING: {fails} files failed during trimming (kept whole file).")


if __name__ == "__main__":
    main()
