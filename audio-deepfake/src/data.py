"""Manifest, trimming, fixed-window cropping and the torch Dataset.

Label convention (fixed everywhere): label = 1 bona fide, label = 0 spoof.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed

# Protocol filenames inside DATA_ROOT/LA/ASVspoof2019_LA_cm_protocols/.
# These are the standard names for the 2019 LA release; build_manifest verifies
# they exist and raises a clear error listing the folder if they do not.
PROTO = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval":  "ASVspoof2019.LA.cm.eval.trl.txt",
}

EXPECTED_COUNTS = {  # from the guide, section 3.2 (bona, spoof, total)
    "train": (2580, 22800, 25380),
    "dev":   (2548, 22296, 24844),
    "eval":  (7355, 63882, 71237),
}

# Eval attacks A16 and A19 reuse the algorithms of A04 and A06, so they are NOT
# truly unseen. 11 of the 13 eval attacks are unseen.
NON_UNSEEN_EVAL_ATTACKS = {"A16", "A19"}


def _proto_path(root: Path, split: str) -> Path:
    folder = root / "ASVspoof2019_LA_cm_protocols"
    p = folder / PROTO[split]
    if not p.exists():
        listing = sorted(x.name for x in folder.glob("*.txt")) if folder.exists() else []
        raise FileNotFoundError(
            f"Protocol for '{split}' not found at {p}.\n"
            f"Files present in {folder}: {listing}\n"
            "Edit PROTO in src/data.py to match the real filenames."
        )
    return p


def read_protocol(root: Path, split: str) -> pd.DataFrame:
    rows = []
    with open(_proto_path(root, split)) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            spk, fid, _, atk, key = parts[0], parts[1], parts[2], parts[3], parts[4]
            rows.append(dict(
                file_id=fid, split=split, speaker_id=spk,
                attack="bonafide" if key == "bonafide" else atk,
                label=1 if key == "bonafide" else 0,
                path=str(Path(f"ASVspoof2019_LA_{split}") / "flac" / f"{fid}.flac"),
            ))
    return pd.DataFrame(rows)


def _probe(root: Path, rel: str):
    try:
        info = sf.info(str(root / rel))
        return info.frames, info.samplerate, ""
    except Exception as e:  # keep going; log the failure
        return -1, -1, str(e)


def build_manifest(root, n_jobs=-1) -> pd.DataFrame:
    root = Path(root)
    df = pd.concat([read_protocol(root, s) for s in ["train", "dev", "eval"]],
                   ignore_index=True)
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_probe)(root, p) for p in df.path)
    df["n_samples"] = [r[0] for r in res]
    df["sr"] = [r[1] for r in res]
    df["decode_error"] = [r[2] for r in res]
    df["duration"] = df.n_samples / df.sr.replace(0, np.nan)
    # placeholders filled by compute_trim.py
    df["trim_start"] = 0
    df["trim_end"] = df["n_samples"].clip(lower=0)
    return df


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Return an audit table and never raise: deviations are data, not crashes."""
    rows = []
    for split in ["train", "dev", "eval"]:
        s = df[df.split == split]
        bona = int((s.label == 1).sum())
        spoof = int((s.label == 0).sum())
        exp = EXPECTED_COUNTS[split]
        rows.append(dict(
            split=split, bona=bona, spoof=spoof, total=len(s),
            exp_bona=exp[0], exp_spoof=exp[1], exp_total=exp[2],
            counts_match=(bona, spoof, len(s)) == exp,
            bona_fraction=round(bona / max(len(s), 1), 4),
            sr_all_16k=bool((s.sr == 16000).all()),
            decode_failures=int((s.decode_error != "").sum()),
            dup_file_ids=int(s.file_id.duplicated().sum()),
        ))
    # speaker overlap across splits
    spk = {sp: set(df[df.split == sp].speaker_id) for sp in ["train", "dev", "eval"]}
    rows.append(dict(
        split="_speaker_overlap",
        bona=len(spk["train"] & spk["dev"]),
        spoof=len(spk["train"] & spk["eval"]),
        total=len(spk["dev"] & spk["eval"]),
        exp_bona=0, exp_spoof=0, exp_total=0,
        counts_match=not (spk["train"] & spk["dev"] or spk["train"] & spk["eval"]
                          or spk["dev"] & spk["eval"]),
        bona_fraction=0.0, sr_all_16k=True, decode_failures=0, dup_file_ids=0,
    ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Trimming (section 4)
# --------------------------------------------------------------------------- #
def trim_indices(wav, sr=16000, frame=400, hop=160, thresh_db=35.0, margin_s=0.05):
    """Energy-based trim. Returns (start, end) in samples; identical rule for all."""
    import librosa
    if len(wav) < frame:
        return 0, len(wav)
    rms = librosa.feature.rms(y=wav, frame_length=frame, hop_length=hop,
                              center=False)[0]
    db = 20 * np.log10(rms + 1e-10)
    active = db > (db.max() - thresh_db)
    if not active.any():
        return 0, len(wav)
    first = int(np.argmax(active))
    last = len(active) - 1 - int(np.argmax(active[::-1]))
    m = int(margin_s * sr)
    s = max(0, first * hop - m)
    e = min(len(wav), last * hop + frame + m)
    if e - s < int(0.1 * sr):
        return 0, len(wav)
    return int(s), int(e)


def _trim_one(root, rel, thresh_db, margin_s):
    try:
        wav, sr = sf.read(str(root / rel), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        s, e = trim_indices(wav, sr=sr, thresh_db=thresh_db, margin_s=margin_s)
        return s, e, ""
    except Exception as ex:
        return 0, -1, str(ex)


def compute_trim_columns(df, root, thresh_db=35.0, margin_s=0.05, n_jobs=-1):
    root = Path(root)
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_trim_one)(root, p, thresh_db, margin_s) for p in df.path)
    out = df.copy()
    out["trim_start"] = [r[0] for r in res]
    out["trim_end"] = [r[1] if r[1] > 0 else n for r, n in zip(res, df.n_samples)]
    out["trim_error"] = [r[2] for r in res]
    return out


# --------------------------------------------------------------------------- #
# Fixed 4 s window (section 4.2)
# --------------------------------------------------------------------------- #
def fix_length(wav, n=64000, random_crop=False, rng=None):
    if rng is None:
        rng = np.random
    if len(wav) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(wav) < n:
        wav = np.tile(wav, int(np.ceil(n / len(wav))))[:n]
    elif len(wav) > n:
        start = rng.randint(0, len(wav) - n + 1) if random_crop else (len(wav) - n) // 2
        wav = wav[start:start + n]
    return wav.astype(np.float32)


def read_audio(path, retries=3):
    """sf.read with retries; the error always names the file (soundfile's own message can be empty)."""
    import time
    last = None
    for k in range(retries):
        try:
            return sf.read(str(path), dtype="float32")
        except Exception as e:
            last = e
            time.sleep(0.5 * (k + 1))
    raise RuntimeError(f"cannot read audio file {path}: {type(last).__name__}") from last


class LADataset:
    """torch Dataset. Returns (waveform[64000], label). Import torch lazily."""

    def __init__(self, df, root, train, trim, window=64000):
        import torch  # noqa: F401  (ensures torch present; tensor built below)
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.train = train
        self.trim = trim
        self.window = window

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        import torch
        r = self.df.iloc[i]
        wav, sr = read_audio(self.root / r.path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        assert sr == 16000, f"unexpected sr={sr} for {r.file_id}"
        if self.trim:
            wav = wav[int(r.trim_start):int(r.trim_end)]
        wav = fix_length(wav, self.window, random_crop=self.train)
        return torch.from_numpy(wav), torch.tensor(float(r.label), dtype=torch.float32)
