"""Small shared helpers: seeding, config IO, environment capture."""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np


def set_seed(seed: int):
    """Seed python, numpy and torch (if present). Call once per run."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # keep cudnn fast but note: full determinism is not required by the guide
    except ImportError:
        pass


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "no-git"


def lib_versions():
    out = {"python": sys.version.split()[0]}
    for name in ["numpy", "pandas", "scipy", "sklearn", "soundfile",
                 "librosa", "torch", "torchaudio", "torchvision"]:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            out[name] = "not-installed"
    return out


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def env_report():
    """One-shot environment summary string for DECISIONS.md (section 2.1)."""
    lines = []
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"], stderr=subprocess.DEVNULL
        ).decode().strip()
        lines.append(f"GPU: {gpu}")
    except Exception:
        lines.append("GPU: none detected")
    lines.append(f"CPUs: {os.cpu_count()}")
    lines.append(f"versions: {json.dumps(lib_versions())}")
    return "\n".join(lines)
