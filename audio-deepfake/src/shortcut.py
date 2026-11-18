"""Shortcut check (section 5): can non-speech cues alone separate bona/spoof?

Feature groups: duration, silence, loudness, first_0.5s, all.
Classifier: HistGradientBoostingClassifier, trained on train only, scored on dev/eval.
Two settings: 'raw' (full file) and 'trimmed' (exactly what the network sees).
"""
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed

from .data import fix_length
from .metrics import compute_auc, compute_eer

SR = 16000
FRAME = 400
HOP = 160
SIL_DB = 35.0            # a frame is silent if >35 dB below the loudest frame
N_MEL_F05 = 40
F05_SAMPLES = int(0.5 * SR)

GROUPS = {
    "duration": ["dur_"],
    "silence": ["sil_"],
    "loudness": ["loud_"],
    "first_0.5s": ["f05_"],
    "all": ["dur_", "sil_", "loud_", "f05_"],
}


def _frame_db(wav):
    import librosa
    rms = librosa.feature.rms(y=wav, frame_length=FRAME, hop_length=HOP,
                              center=False)[0]
    return 20 * np.log10(rms + 1e-10)


def _silence_feats(wav):
    if len(wav) < FRAME:
        return dict(sil_lead_s=0.0, sil_trail_s=0.0, sil_ratio=1.0)
    db = _frame_db(wav)
    active = db > (db.max() - SIL_DB)
    if not active.any():
        return dict(sil_lead_s=len(wav) / SR, sil_trail_s=len(wav) / SR, sil_ratio=1.0)
    first = int(np.argmax(active))
    last = len(active) - 1 - int(np.argmax(active[::-1]))
    lead = first * HOP / SR
    trail = (len(active) - 1 - last) * HOP / SR
    ratio = float((~active).mean())
    return dict(sil_lead_s=float(lead), sil_trail_s=float(trail), sil_ratio=ratio)


def _loudness_feats(wav):
    rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-10)
    peak = float(np.max(np.abs(wav)) + 1e-10)
    return dict(loud_rms_db=20 * np.log10(rms), loud_peak_db=20 * np.log10(peak))


def _first05_feats(wav):
    import librosa
    seg = wav[:F05_SAMPLES]
    if len(seg) < F05_SAMPLES:
        seg = np.pad(seg, (0, F05_SAMPLES - len(seg)))
    mel = librosa.feature.melspectrogram(
        y=seg, sr=SR, n_fft=512, win_length=FRAME, hop_length=HOP,
        n_mels=N_MEL_F05, fmin=0.0, fmax=8000.0)
    logmel = np.log(mel + 1e-6)                 # (40, frames)
    mean = logmel.mean(axis=1)
    std = logmel.std(axis=1)
    out = {}
    for i in range(N_MEL_F05):
        out[f"f05_mel_mean_{i:02d}"] = float(mean[i])
        out[f"f05_mel_std_{i:02d}"] = float(std[i])
    return out


def _extract_one(root, rel, trim_start, trim_end, setting):
    try:
        wav, sr = sf.read(str(root + "/" + rel), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if setting == "raw":
            sig_dur = wav                       # full file
            model_input = wav                   # measure on full file
        else:  # trimmed: duration on trimmed clip, the rest on the 4 s model input
            clip = wav[int(trim_start):int(trim_end)]
            sig_dur = clip
            model_input = fix_length(clip, 64000, random_crop=False)
        feats = {"dur_seconds": len(sig_dur) / sr}
        feats.update(_silence_feats(model_input))
        feats.update(_loudness_feats(model_input))
        feats.update(_first05_feats(model_input))
        feats["_err"] = ""
        return feats
    except Exception as ex:
        return {"dur_seconds": 0.0, "_err": str(ex)}


def extract_features(df, root, setting, n_jobs=-1):
    """Return a DataFrame: file_id, split, label, attack + feature columns."""
    root = str(root)
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_extract_one)(root, r.path, r.trim_start, r.trim_end, setting)
        for r in df.itertuples(index=False))
    feat = pd.DataFrame(res)
    meta = df[["file_id", "split", "label", "attack"]].reset_index(drop=True)
    return pd.concat([meta, feat], axis=1)


def _cols_for_group(df, group):
    prefixes = GROUPS[group]
    return [c for c in df.columns
            if any(c.startswith(p) for p in prefixes) and c != "_err"]


def run_classifier(feat_df):
    """Train HGB per group on train, score dev+eval. Returns long results DataFrame."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    tr = feat_df[feat_df.split == "train"]
    rows = []
    for group in GROUPS:
        cols = _cols_for_group(feat_df, group)
        Xtr, ytr = tr[cols].values, tr.label.values
        # equal total weight per class
        w = np.ones(len(ytr), dtype=float)
        for c in (0, 1):
            m = ytr == c
            if m.sum() > 0:
                w[m] = len(ytr) / (2.0 * m.sum())
        clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, random_state=0)
        clf.fit(Xtr, ytr, sample_weight=w)
        for split in ["dev", "eval"]:
            s = feat_df[feat_df.split == split]
            score = clf.predict_proba(s[cols].values)[:, 1]  # P(bona fide)
            rows.append(dict(
                group=group, split=split,
                auc=compute_auc(s.label.values, score),
                eer=compute_eer(s.label.values, score)))
    return pd.DataFrame(rows)
