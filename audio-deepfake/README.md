# Audio Deepfake Detection — reduced, leakage-aware study (ASVspoof 2019 LA)

A small, reproducible pipeline that answers one question: **does fusing Log-Mel and
LFCC help detect spoofed speech from unseen generators, once the known silence
shortcut is controlled?** It audits a dataset shortcut, trims silence, trains three
models over three seeds, and reports EER / ROC-AUC with per-attack breakdowns.

Built to run on **Google Colab** with the dataset on **Google Drive**. Everything
is driven from `notebooks/audio_deepfake_colab.ipynb`, or from the CLI scripts below.

> **Rule zero:** no number in any table, plot, or report is invented. Every figure
> comes from a run you actually executed, with its config and seed saved.

---

## 0. Get the data (you download it; it is never committed)

**ASVspoof 2019 LA** — Edinburgh DataShare
<https://datashare.ed.ac.uk/handle/10283/3336>
- File: `LA.zip` (~7.1 GB). License: **Open Data Commons Attribution License (ODC-By 1.0)** — free for academic and commercial use with attribution. Record DOI: 10.7488/ds/2555.
- Direct bitstream (verify on the page first): `https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip`
- HuggingFace mirror (handy inside Colab): dataset `LanceaKing/asvspoof2019`.
- After unzip you get `LA/ASVspoof2019_LA_{train,dev,eval}/flac/*.flac`, `LA/ASVspoof2019_LA_cm_protocols/`, `README.LA.txt`.

**In-the-Wild** (only for stretch goal S1, the sealed external test)
<https://huggingface.co/datasets/mueller91/In-The-Wild> — `release_in_the_wild.zip`
(~38 h, 58 speakers). Note the license is stated as Apache-2.0 on the Fraunhofer page
and CC-BY-SA-4.0 on the HF mirror; record whichever the source you use states.

### Recommended Colab layout
Download `LA.zip` **once** to your Drive (e.g. `MyDrive/asvspoof/LA.zip`). Each session,
the notebook copies it to local `/content` and unzips there. Reading 121k tiny FLAC
files **directly from mounted Drive is very slow** — local disk is the whole point.
Keep **checkpoints and results on Drive** so a disconnect doesn't lose progress.

```
Google Drive/MyDrive/asvspoof/
  LA.zip                      <- you put this here once
  runs/                       <- checkpoints + metrics.json written here (survive disconnects)
Colab /content/
  data_root/LA/...            <- unzipped here each session (fast local reads)
  audio-deepfake/             <- this repo
```

---

## 1. Quick start on Colab

1. Open `notebooks/audio_deepfake_colab.ipynb` in Colab. Runtime → Change runtime type → **GPU** (T4 is enough).
2. Run the cells top to bottom. They: mount Drive, copy+unzip `LA.zip` locally, install the
   few extra packages, then walk P0→P9. Checkpoints go to `Drive/.../runs`.
3. If the session drops mid-training, reopen, re-run the setup cells, and re-run the
   training cell — it resumes from `last.pt` automatically.

## 2. Or run from the CLI (same steps)

```bash
DATA=/content/data_root        # folder that CONTAINS LA/
RUNS=/content/drive/MyDrive/asvspoof/runs   # persistent

# P2  manifest + audit
python scripts/build_manifest.py --data_root $DATA

# P4  trimming (defaults 35 dB / 0.05 s). Optional: pick params by dev AUC first:
# python scripts/select_trim.py --data_root $DATA --subsample 6000
python scripts/compute_trim.py --data_root $DATA --thresh_db 35 --margin_s 0.05

# P3/P5  shortcut audit, raw then trimmed  (Gate G1: every group AUC < 0.60)
python scripts/run_shortcut.py --setting raw     --data_root $DATA
python scripts/run_shortcut.py --setting trimmed --data_root $DATA

# P6  norm stats (both settings) + overfit sanity gate
python scripts/compute_norm_stats.py --data_root $DATA --setting trim
python scripts/compute_norm_stats.py --data_root $DATA --setting untrim
python scripts/overfit_test.py --config configs/A_trim.yaml --data_root $DATA

# P7/P8  the 12 core runs = 4 configs x seeds {0,1,2}. One command per (config,seed):
for s in 0 1 2; do
  python scripts/train_run.py --config configs/A_trim.yaml   --seed $s --data_root $DATA --runs_dir $RUNS --resume
  python scripts/train_run.py --config configs/A_untrim.yaml --seed $s --data_root $DATA --runs_dir $RUNS --resume
  python scripts/train_run.py --config configs/B_trim.yaml   --seed $s --data_root $DATA --runs_dir $RUNS --resume
  python scripts/train_run.py --config configs/F_trim.yaml   --seed $s --data_root $DATA --runs_dir $RUNS --resume
done

# P9  tables + figures (reads every runs/*/metrics.json)
python scripts/aggregate.py
```

`--resume` is safe to leave on always: it continues `last.pt` if present, else starts fresh.

---

## 3. What each piece does

| Path | Role |
|---|---|
| `src/data.py` | protocol reader, manifest, audit, energy trimming, 4 s window, `LADataset` |
| `src/features.py` | Log-Mel (80) and LFCC+Δ+ΔΔ (60) frontends, fp32, outside autocast |
| `src/models.py` | A = Log-Mel+ResNet18, B = LFCC+CNN, F = late fusion; one logit, higher = bona fide |
| `src/metrics.py` | EER, ROC-AUC, per-attack EER, unseen-attack aggregation |
| `src/train.py` | train loop, dev early stopping on EER, checkpoint/resume, final dev+eval scoring, `metrics.json` |
| `src/shortcut.py` | non-speech features (duration/silence/loudness/first-0.5 s) + HistGradientBoosting audit |
| `scripts/*` | thin CLI wrappers for each pipeline step |
| `configs/*.yaml` | the 4 runs; training settings are fixed, not tuned |

## 4. Fixed conventions (don't change silently)

- **Labels:** `1` = bona fide, `0` = spoof. Model emits one logit; higher = more bona fide.
  EER/AUC computed with bona fide as positive.
- **Splits:** train on `train`; early stopping and every choice use `dev`; `eval` is final
  reporting only — never used to pick anything, including trimming parameters.
- **Norm stats** come from `train` only (3,000 files, seed 0), computed per setting.
- **One preprocessing rule** for every file, class, and split.
- **Frontends in fp32** (outside autocast); only the CNN runs under autocast + GradScaler.
- `A16` and `A19` reuse the algorithms of `A04`/`A06` → only 11 of 13 eval attacks are truly unseen; they are reported separately.

## 5. Gates and hypotheses

- **G1** (after trimming): every shortcut group AUC < 0.60 on dev and eval. 0.60–0.75 = continue as a limitation; > 0.75 = stop and reconsider the dataset. `duration` may stay high (trimming doesn't equalise speech lengths) — record it, don't cheat.
- **G2** (after first `A_trim`): dev EER clearly below 50% (near 50% = a bug: check label direction / normalization / trimming), and a measurable dev→eval gap.
- **R1** silence etc. leak on raw, trimming largely removes it. **R2** eval worse than dev.
  **R3** fusion F beats A and B on eval (a negative result is fine — report it honestly).
  **R4** trimming raises EER vs untrimmed.

## 6. Outputs to hand back

`data/manifest.csv`, `results/tables/audit.csv`, `results/shortcut/shortcut_results.csv`,
`results/runs/*/metrics.json`, the tables and figures in `results/`, a filled-in
`DECISIONS.md`, and a one-page summary (what was done, G1/G2 outcomes, main numbers with
spread, what was skipped, and open uncertainties).

## 7. Colab gotchas (already handled or worth knowing)

- **Slow Drive reads** → copy+unzip to `/content`; train from local disk.
- **Disconnects** → checkpoints every epoch to Drive; `--resume`.
- **OOM** → set `batch_size: 32` in the config and record it in `DECISIONS.md`.
- **`num_workers`** → Colab often has 2 CPU cores; 2–4 workers is realistic (not the guide's 4+).
- **First run time** → measure it and replace the guide's 20–60 min/run estimate with the real figure.

## 8. Not in scope

WavLM/Wav2Vec, gated fusion, leave-one-attack-out, full ablation grid, hyperparameter
search, demo/web app, training any foundation model. Stretch goals S1–S4 only after P0–P9
are complete and documented (see the project guide, section 10).
