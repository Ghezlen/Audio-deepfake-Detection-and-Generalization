"""Training (section 7): train on train, early-stop on dev EER, score dev+eval once.

Checkpoints every epoch to persistent storage so a disconnected Colab/Kaggle
session can resume with resume=True.
"""
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import LADataset
from .metrics import compute_auc, compute_eer, summarize_eval
from .models import build_model
from .utils import git_commit, lib_versions, save_json, set_seed

WINDOW = 64000


def _loader(df, root, train, trim, batch_size, num_workers, shuffle):
    ds = LADataset(df, root, train=train, trim=trim, window=WINDOW)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0, drop_last=False)


@torch.no_grad()
def score_split(model, df_split, root, trim, device, batch_size=64, num_workers=2):
    model.eval()
    loader = _loader(df_split, root, train=False, trim=trim,
                     batch_size=batch_size, num_workers=num_workers, shuffle=False)
    scores = []
    use_amp = device.type == "cuda"
    for wav, _ in loader:
        wav = wav.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logit = model(wav)
        scores.append(logit.float().cpu().numpy())
    out = df_split[["file_id", "attack", "label"]].reset_index(drop=True).copy()
    out["score"] = np.concatenate(scores)       # higher logit = more bona fide
    return out


def _save_ckpt(path, model, opt, scaler, epoch, best_eer, patience_left):
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_eer": best_eer,
        "patience_left": patience_left,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }, path)


def _restore_rng(rng):
    torch.set_rng_state(rng["torch"].cpu())
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in rng["cuda"]])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])


def train_one(cfg, seed, data_root, manifest_csv, norm_npz, out_dir,
              resume=False, device=None, max_epochs=None):
    """Run one (model, trim, seed) training. Writes best.pt/last.pt and metrics.json."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    df = pd.read_csv(manifest_csv)
    trim = bool(cfg["trim"])
    tr = df[df.split == "train"].reset_index(drop=True)
    dev = df[df.split == "dev"].reset_index(drop=True)
    ev = df[df.split == "eval"].reset_index(drop=True)

    n_bona = int((tr.label == 1).sum())
    n_spoof = int((tr.label == 0).sum())
    pos_weight = torch.tensor([n_spoof / max(n_bona, 1)], device=device)

    model = build_model(cfg["model"], norm_npz=norm_npz).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    epochs = int(max_epochs or cfg["epochs"])
    patience = int(cfg["patience"])
    bs = int(cfg["batch_size"])
    nw = int(cfg["num_workers"])
    grad_clip = float(cfg["grad_clip"])

    start_epoch, best_eer, patience_left = 0, float("inf"), patience
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); _restore_rng(ck["rng"])
        start_epoch = ck["epoch"] + 1
        best_eer = ck["best_eer"]; patience_left = ck["patience_left"]
        print(f"[resume] from epoch {start_epoch}, best dev EER {best_eer:.4f}")

    train_loader = _loader(tr, data_root, train=True, trim=trim,
                           batch_size=bs, num_workers=nw, shuffle=True)
    use_amp = device.type == "cuda"
    t0 = time.time()
    history = []

    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        for wav, y in train_loader:
            wav, y = wav.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logit = model(wav)
                loss = crit(logit, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update()
            running += loss.item() * len(y)
        train_loss = running / len(tr)

        dev_sc = score_split(model, dev, data_root, trim, device, bs, nw)
        dev_eer = compute_eer(dev_sc.label.values, dev_sc.score.values)
        dev_auc = compute_auc(dev_sc.label.values, dev_sc.score.values)
        history.append(dict(epoch=epoch, train_loss=train_loss,
                            dev_eer=dev_eer, dev_auc=dev_auc))
        improved = dev_eer < best_eer - 1e-5
        if improved:
            best_eer = dev_eer; patience_left = patience
            _save_ckpt(best_path, model, opt, scaler, epoch, best_eer, patience_left)
        else:
            patience_left -= 1
        _save_ckpt(last_path, model, opt, scaler, epoch, best_eer, patience_left)
        print(f"ep {epoch:02d} loss {train_loss:.4f} dev_eer {dev_eer:.4f} "
              f"dev_auc {dev_auc:.4f} best {best_eer:.4f} pat {patience_left}")
        if patience_left <= 0:
            print("[early stop]"); break

    # ---- final scoring with best checkpoint ----
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    best_epoch = ck["epoch"]
    dev_sc = score_split(model, dev, data_root, trim, device, bs, nw)
    ev_sc = score_split(model, ev, data_root, trim, device, bs, nw)
    dev_sc.to_csv(out_dir / "dev_scores.csv", index=False)
    ev_sc.to_csv(out_dir / "eval_scores.csv", index=False)

    metrics = {
        "run_id": cfg["run_id"], "model": cfg["model"], "trim": trim, "seed": seed,
        "dev_eer": compute_eer(dev_sc.label.values, dev_sc.score.values),
        "dev_auc": compute_auc(dev_sc.label.values, dev_sc.score.values),
        "best_epoch": int(best_epoch),
        "train_time_s": round(time.time() - t0, 1),
        "batch_size": bs, "pos_weight": float(pos_weight.item()),
        "git_commit": git_commit(), "versions": lib_versions(),
        "config": cfg, "history": history,
    }
    metrics.update(summarize_eval(ev_sc))
    save_json(metrics, out_dir / "metrics.json")
    print(f"[done] dev_eer {metrics['dev_eer']:.4f} eval_eer {metrics['eval_eer']:.4f} "
          f"eval_auc {metrics['eval_auc']:.4f} unseen_mean_eer {metrics['unseen_mean_eer']:.4f}")
    return metrics


def overfit_test(cfg, data_root, manifest_csv, norm_npz, n=64, steps=60, device=None):
    """P6 sanity gate: memorize 64 samples to near-zero loss, no NaN."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(0)
    df = pd.read_csv(manifest_csv)
    trim = bool(cfg["trim"])
    tr = df[df.split == "train"].sample(n=n, random_state=0).reset_index(drop=True)
    loader = _loader(tr, data_root, train=True, trim=trim, batch_size=min(n, 64),
                     num_workers=2, shuffle=True)
    model = build_model(cfg["model"], norm_npz=norm_npz).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()
    model.train()
    last_loss, saw_nan = None, False
    for step in range(steps):
        for wav, y in loader:
            wav, y = wav.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logit = model(wav)                   # fp32, no autocast: clean sanity signal
            loss = crit(logit, y)
            if torch.isnan(loss):
                saw_nan = True
            loss.backward()
            opt.step()
            last_loss = loss.item()
        if step % 10 == 0:
            print(f"overfit step {step} loss {last_loss:.4f}")
    print(f"[overfit] final loss {last_loss:.4f} nan={saw_nan}")
    return {"final_loss": last_loss, "saw_nan": saw_nan,
            "passed": (last_loss is not None and last_loss < 0.05 and not saw_nan)}
