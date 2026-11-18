"""Models A (Log-Mel+ResNet18), B (LFCC+CNN), F (fusion). One logit, higher=bona fide."""
import torch
import torch.nn as nn
import torchvision

from .features import LFCCDD, LogMel, load_norm


def resnet18_1ch():
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    m.fc = nn.Identity()                  # 512-d embedding
    return m


class SmallCNN(nn.Module):                # for LFCC, 256-d embedding
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.net = nn.Sequential(
            blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class ModelA(nn.Module):                  # Log-Mel + ResNet18
    def __init__(self, front):
        super().__init__()
        self.front = front
        self.enc = resnet18_1ch()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 1))

    def forward(self, wav):
        return self.head(self.enc(self.front(wav))).squeeze(-1)


class ModelB(nn.Module):                  # LFCC + small CNN
    def __init__(self, front):
        super().__init__()
        self.front = front
        self.enc = SmallCNN()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, wav):
        return self.head(self.enc(self.front(wav))).squeeze(-1)


class ModelF(nn.Module):                  # fusion: concat(512,256) -> proj -> logit
    def __init__(self, front_mel, front_lfcc):
        super().__init__()
        self.fm, self.fl = front_mel, front_lfcc
        self.em, self.el = resnet18_1ch(), SmallCNN()
        self.head = nn.Sequential(
            nn.Linear(512 + 256, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, wav):
        z = torch.cat([self.em(self.fm(wav)), self.el(self.fl(wav))], dim=1)
        return self.head(z).squeeze(-1)


def build_model(kind, norm_npz=None):
    """kind in {'A','B','F'}. norm_npz path loads per-bin train stats into frontends."""
    mel_mean = mel_std = lfcc_mean = lfcc_std = None
    if norm_npz is not None:
        mel_mean, mel_std = load_norm(norm_npz, "mel")
        lfcc_mean, lfcc_std = load_norm(norm_npz, "lfcc")
    if kind == "A":
        return ModelA(LogMel(mel_mean, mel_std))
    if kind == "B":
        return ModelB(LFCCDD(lfcc_mean, lfcc_std))
    if kind == "F":
        return ModelF(LogMel(mel_mean, mel_std), LFCCDD(lfcc_mean, lfcc_std))
    raise ValueError(f"unknown model kind {kind!r}")
