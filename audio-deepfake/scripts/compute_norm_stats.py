#!/usr/bin/env python
"""P6: per-bin mean/std from 3,000 random TRAIN files (seed 0), trim-setting specific.

Writes data/norm_trim.npz or data/norm_untrim.npz with keys
mel_mean(80), mel_std(80), lfcc_mean(60), lfcc_std(60).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import LADataset               # noqa: E402
from src.features import LFCCDD, LogMel       # noqa: E402


def accumulate(frontend, loader, device, n_bins):
    s = torch.zeros(n_bins, dtype=torch.float64, device=device)
    ss = torch.zeros(n_bins, dtype=torch.float64, device=device)
    count = 0
    frontend = frontend.to(device).eval()
    with torch.no_grad():
        for wav, _ in loader:
            wav = wav.to(device)
            feat = frontend(wav)              # (B,1,C,frames)
            feat = feat.squeeze(1).permute(0, 2, 1).reshape(-1, n_bins).double()
            s += feat.sum(0)
            ss += (feat ** 2).sum(0)
            count += feat.shape[0]
    mean = (s / count)
    var = (ss / count) - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    return mean.cpu().numpy(), std.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--setting", choices=["trim", "untrim"], required=True)
    ap.add_argument("--n_files", type=int, default=3000)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root) / "LA"
    df = pd.read_csv(args.manifest)
    tr = df[df.split == "train"].sample(n=min(args.n_files, (df.split == "train").sum()),
                                        random_state=0).reset_index(drop=True)
    trim = args.setting == "trim"
    ds = LADataset(tr, root, train=False, trim=trim)   # center crop, deterministic
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    print(f"computing mel stats on {len(tr)} files ...")
    mel_mean, mel_std = accumulate(LogMel(), loader, device, 80)
    print(f"computing lfcc stats on {len(tr)} files ...")
    lfcc_mean, lfcc_std = accumulate(LFCCDD(), loader, device, 60)

    out = ROOT / "data" / f"norm_{args.setting}.npz"
    np.savez(out, mel_mean=mel_mean, mel_std=mel_std,
             lfcc_mean=lfcc_mean, lfcc_std=lfcc_std)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
