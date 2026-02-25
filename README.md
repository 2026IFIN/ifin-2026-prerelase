# IFIN

This repository contains IFIN training, evaluation, and inference code with dataset/config utilities and reproducibility settings.

## Highlights

- Clean project layout with `src/`, `configs/`, `scripts/`, `tests/`
- Naming:
  - `IFIN`: Integrated Forward-Inverse Network (`IFINNet`)
  - `IFIB`: Integrated Forward-Inverse Block (`IFIB`)
  - `ISO`: Inverse System Operator (`ISO`)
  - `FSO`: Forward System Operator (`FSO`)
  - `RB`: Refinement block (`RB`)
- Checkpoint compatibility utility for common key-format issues (`module.` prefixes)
- Reproducibility utilities (seed + deterministic flags)
- Lightweight tests (forward equivalence, 1-step train/save/load)
- Default config aligned with Waller dataset usage and PSF preprocessing from the original training script

## Repository structure

- `src/models`: native IFIN/IFIB/ISO/FSO/RB implementation
- `src/data`: WallerDataset + synthetic fallback dataset
- `src/losses`: minimal deterministic losses
- `src/metrics`: metrics
- `src/utils`: seed and checkpoint utilities
- `configs/default.yaml`: default reproducible config
- `scripts/`: train/eval/infer entry implementations
- root `train.py`, `eval.py`, `infer.py`: convenience entry points
- `tests/`: smoke and equivalence tests

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducibility

Configuration fields in `configs/default.yaml`:

- `seed`: global seed for Python/NumPy/PyTorch
- `deterministic`: sets cuDNN deterministic behavior
- `device`: `auto` / `cpu` / `cuda`

## Train

```bash
python train.py --config configs/default.yaml
```

## Eval

```bash
python eval.py --config configs/default.yaml --checkpoint outputs/checkpoints/ifin_last.pth
```

## Inference

```bash
python infer.py --config configs/default.yaml --checkpoint outputs/checkpoints/ifin_last.pth
```

Notebook walkthrough:

```bash
jupyter notebook notebooks/inference_waller.ipynb
```

## Tests

Run all tests:

```bash
python -m pytest -q tests
```

Run only equivalence sanity check:

```bash
python -m pytest -q tests/test_equivalence.py
```

## Data preparation

Default config uses WallerDataset inputs, matching the original script behavior:

- `data.dataset: waller`
- `data.waller_path: ../../wallerlab/dataset`
- `data.psf_path: ../../wallerlab/dataset/psf.tiff`
- model input size `270 x 480`

### Required WallerDataset directory layout

`data.waller_path` should point to a folder containing:

- `dataset_train.csv`
- `dataset_test.csv`
- `diffuser_images/`
- `ground_truth_lensed/`

`data.psf_path` should point to `psf.tiff` for the same dataset.

### Which paths users should revise

Update these two fields in `configs/default.yaml` for your machine:

- `data.waller_path`
- `data.psf_path`

Both absolute and relative paths are supported. Relative paths such as `../../wallerlab/dataset` are resolved automatically.
At runtime, IFIN validates dataset layout and PSF path early, and raises a clear error if a required file/folder is missing.

PSF preprocessing is aligned to the original script:

- read grayscale PSF
- resize to `(480, 270)`
- normalize by `255`
- subtract local background mean from top-left `15x15`
- clamp negative values to `0`

For environments without WallerDataset data, set `data.dataset: synthetic` in config.

### Optional: Hugging Face download route

You can use any source (including Hugging Face) as long as the extracted folder matches the required layout above.

After download/extract, set:

- `data.waller_path` to the extracted dataset root (must match the Waller CSV + folder layout above)
- `data.psf_path` to the corresponding `psf.tiff`

## Notes on checkpoint compatibility

This codebase preserves legacy checkpoint loading via key-compatibility utilities.
If checkpoints contain `module.` prefixes, `load_model_state_compat` automatically normalizes keys.
