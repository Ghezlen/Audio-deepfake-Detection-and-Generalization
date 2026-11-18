"""Frontends (section 6.1). Run in float32, outside autocast. No learnable params."""
import torch
import torch.nn as nn
import torchaudio

N_MELS = 80
N_LFCC = 20
LFCC_DD = N_LFCC * 3  # 60: lfcc + delta + delta-delta


class LogMel(nn.Module):
    def __init__(self, mean=None, std=None):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
            f_min=0.0, f_max=8000.0, n_mels=N_MELS)
        self.register_buffer("mean", torch.zeros(N_MELS, 1) if mean is None else mean)
        self.register_buffer("std",  torch.ones(N_MELS, 1) if std is None else std)

    @torch.no_grad()
    def forward(self, x):                      # x: (B, T)
        with torch.autocast(device_type=x.device.type, enabled=False):
            m = torch.log(self.mel(x.float()) + 1e-6)      # (B, 80, frames)
            m = (m - self.mean) / (self.std + 1e-5)
        return m.unsqueeze(1)                   # (B, 1, 80, frames)


class LFCCDD(nn.Module):
    def __init__(self, mean=None, std=None):
        super().__init__()
        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=16000, n_filter=70, n_lfcc=N_LFCC, f_min=0.0, f_max=8000.0,
            speckwargs=dict(n_fft=512, win_length=400, hop_length=160))
        self.register_buffer("mean", torch.zeros(LFCC_DD, 1) if mean is None else mean)
        self.register_buffer("std",  torch.ones(LFCC_DD, 1) if std is None else std)

    @torch.no_grad()
    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            c = self.lfcc(x.float())                           # (B, 20, frames)
            d1 = torchaudio.functional.compute_deltas(c)
            d2 = torchaudio.functional.compute_deltas(d1)
            f = torch.cat([c, d1, d2], dim=1)                  # (B, 60, frames)
            f = (f - self.mean) / (self.std + 1e-5)
        return f.unsqueeze(1)                                   # (B, 1, 60, frames)


def load_norm(npz_path, kind):
    """kind in {'mel','lfcc'}. Returns (mean, std) as (C,1) float tensors or None."""
    import numpy as np
    d = np.load(npz_path)
    mean = torch.tensor(d[f"{kind}_mean"], dtype=torch.float32).reshape(-1, 1)
    std = torch.tensor(d[f"{kind}_std"], dtype=torch.float32).reshape(-1, 1)
    return mean, std
