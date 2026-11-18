# DECISIONS.md

Running log of every deviation from the guide, every bug, every assumption, and the
gate outcomes. Rule zero: no number here or in any table/plot is invented — each comes
from a run actually executed, with config and seed saved.

## P0 — Environment
- Platform: Google Colab (fill GPU name, RAM, disk from the notebook's env cell).
- Dataset storage: LA.zip kept on Google Drive; unzipped to local /content for fast reads.
- (paste the output of the env cell here)

## P1 — Data
- Source: ASVspoof 2019 LA, Edinburgh DataShare https://datashare.ed.ac.uk/handle/10283/3336
- License: Open Data Commons Attribution License (ODC-By 1.0). LA.zip ~7.1 GB.
- MD5 of LA.zip (DataShare record): b53f0668f3b17ce72535fb60cac78923  <- verify after download.

## P2 — Manifest & audit
- (paste audit.csv summary; note any count mismatch vs README.LA.txt)

## P3 — Shortcut check (raw)
- (paste raw AUC per group)

## P4 — Trimming parameters
- Chosen thresh_db / margin_s and how (default 35 / 0.05, or select_trim.py winner).

## P5 — Shortcut check (trimmed) — Gate G1
- G1 target: every group AUC < 0.60 on dev and eval.
- Outcome: PASS / limitation / STOP (paste trimmed AUC per group).

## P6 — Frontends, models, overfit test
- Overfit test (64 samples) final loss, NaN? PASS/FAIL.
- Norm stats: computed on 3,000 train files, seed 0, for trim and untrim.

## P7 — First A_trim run — Gate G2
- Dev EER clearly < 50%? dev-to-eval gap present?
- Measured minutes per run on this GPU: ____ (replaces the guide's estimate).

## P8 — Remaining core runs
- 12 metrics.json present? list run_ids and seeds done.

## P9 — Aggregation
- Tables and figures generated from metrics.json.

## Deviations / bugs / assumptions
-
- BUG FIXED (src/train.py): PyTorch 2.6+ defaults torch.load to weights_only=True, which rejects our
  checkpoints (they store numpy/python RNG state). Every run would have crashed at final scoring and on
  --resume. Both torch.load calls now pass weights_only=False (safe: we only load our own checkpoints).
  Found by a smoke test on synthetic LA-like data (manifest -> trim -> shortcut -> norm -> overfit ->
  train A/B/F -> resume -> aggregate), not on real data.
