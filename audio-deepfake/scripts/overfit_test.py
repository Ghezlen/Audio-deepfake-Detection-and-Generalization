#!/usr/bin/env python
"""P6 gate: a model must overfit 64 samples to near-zero loss with no NaN."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train import overfit_test      # noqa: E402
from src.utils import load_config       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()

    cfg = load_config(args.config)
    trim = "trim" if cfg["trim"] else "untrim"
    norm_npz = ROOT / "data" / f"norm_{trim}.npz"
    norm = str(norm_npz) if norm_npz.exists() else None
    root = Path(args.data_root) / "LA"
    r = overfit_test(cfg, root, args.manifest, norm, steps=args.steps)
    print("PASS" if r["passed"] else "FAIL", r)
    sys.exit(0 if r["passed"] else 1)


if __name__ == "__main__":
    main()
