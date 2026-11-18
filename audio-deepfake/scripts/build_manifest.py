#!/usr/bin/env python
"""P2: build data/manifest.csv and results/tables/audit.csv from DATA_ROOT/LA."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import audit, build_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="folder that CONTAINS the LA/ directory (DATA_ROOT)")
    ap.add_argument("--out", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    data_root = Path(args.data_root) / "LA"
    if not data_root.exists():
        sys.exit(f"Expected {data_root} to exist. Point --data_root at the folder "
                 f"that contains LA/ (after unzipping LA.zip).")

    print(f"Building manifest from {data_root} ...")
    df = build_manifest(data_root, n_jobs=args.n_jobs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(df)} rows)")

    aud = audit(df)
    aud_path = ROOT / "results" / "tables" / "audit.csv"
    aud_path.parent.mkdir(parents=True, exist_ok=True)
    aud.to_csv(aud_path, index=False)
    print(f"wrote {aud_path}")
    print(aud.to_string(index=False))
    fails = int((df.decode_error != "").sum())
    if fails:
        print(f"WARNING: {fails} files failed to decode (see decode_error column).")


if __name__ == "__main__":
    main()
