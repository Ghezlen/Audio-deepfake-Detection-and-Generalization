#!/usr/bin/env python
"""P7/P8: train one run = one config (model+trim) x one seed.

Writes to results/runs/<run_id>_s<seed>/ : best.pt, last.pt, dev/eval scores, metrics.json.
Resume a disconnected run with --resume (reads last.pt).

Example:
  python scripts/train_run.py --config configs/A_trim.yaml --seed 0 --data_root <ROOT>
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train import train_one          # noqa: E402
from src.utils import load_config        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--runs_dir", default=str(ROOT / "results" / "runs"),
                    help="put this on Google Drive so checkpoints survive disconnects")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_epochs", type=int, default=None,
                    help="override epochs (e.g. 1 for a smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    trim = "trim" if cfg["trim"] else "untrim"
    norm_npz = ROOT / "data" / f"norm_{trim}.npz"
    if not norm_npz.exists():
        sys.exit(f"missing {norm_npz} — run compute_norm_stats.py --setting {trim} first")

    out_dir = Path(args.runs_dir) / f"{cfg['run_id']}_s{args.seed}"
    root = Path(args.data_root) / "LA"
    train_one(cfg, args.seed, root, args.manifest, str(norm_npz), out_dir,
              resume=args.resume, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
