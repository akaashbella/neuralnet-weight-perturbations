# Neural Net Weight Perturbations

Research prototype for studying **robustness to Gaussian weight perturbations** and whether **training-time weight noise** improves post-training robustness. Models are trained in two regimes (clean vs. noisy); at evaluation time we sweep over perturbation strengths (α_test) and measure accuracy drop.

## Features

- **Dataset:** CIFAR-10 (32×32 RGB, 10 classes; optional resize per architecture). Standard channel-wise normalization; training uses random crop (padding=4) + horizontal flip.
- **Architectures (core set):** `cnn`, `mlp`, `plainnet20`, `resnet20`, `mobilenet_v2`, `vit_lite`. All CIFAR-10–safe (32×32 input). Optional: `mlp_small`, `mlp_medium`, `mlp_large`, `resnet32`, `resnet18`.
- **Training:** Clean regime (α_train=0.0) and noisy regime (α_train=0.05); per-batch weight noise in noisy regime; optimizer updates clean weights.
- **Evaluation:** Robustness sweep over α_test; accuracy drop at α=0.1; results merged into dataset-specific CSVs and JSON.
- **Plots:** `scripts/plot_robustness_curves.py` produces robustness decay curves and drop@0.1 bar chart.

## Setup

```bash
pip install -r requirements.txt
```

Requires: `torch>=2.0.0`, `torchvision>=0.15.0`, `matplotlib>=3.5.0`. Data is downloaded under `./data` on first run.

---

## Running Experiments

### Full experiment (all architectures, one dataset)

Trains every architecture in both regimes over 3 seeds, then runs the robustness sweep and writes results. Checkpoints and results are under `checkpoints/cifar10/` and `results/cifar10/`.

```bash
# Default: CIFAR-10, core architectures (cnn, mlp, plainnet20, resnet20, mobilenet_v2, vit_lite) × 2 regimes × 3 seeds
python run_experiments.py
```

- **Checkpoints:** `checkpoints/cifar10/{arch}_{regime}_seed{seed}_a{alpha_train}.pt` (clean a=0.00, noisy a=0.05)
- **Results:** `results/cifar10/summary.csv`, `results/cifar10/sweep.csv`, `results/cifar10/results.json`
- Existing checkpoints are **skipped**; new runs are **merged** by (architecture, regime, seed, alpha_train).

### Choose architectures

Pass architecture names as positional arguments. If none are given, the full core set is run.

```bash
# Core set only (faster)
python run_experiments.py cnn mlp plainnet20 resnet20

# With ViT and MobileNet
python run_experiments.py cnn mlp resnet20 mobilenet_v2 vit_lite

# All core (default)
python run_experiments.py
```

**Core architecture names:** `cnn`, `mlp`, `plainnet20`, `resnet20`, `mobilenet_v2`, `vit_lite`. Optional: `mlp_small`, `mlp_medium`, `mlp_large`, `resnet32`, `resnet18`.

### Plot figures

After running experiments, generate the two main figures (robustness decay curves and drop@0.1 bar chart):

```bash
python scripts/plot_robustness_curves.py --dataset cifar10
```

Outputs: `results/cifar10/robustness_curves.png`, `results/cifar10/drop_at_01_bars.png`.

### Quick sanity check (1 epoch, MLP only)

Trains `mlp_small` for 1 epoch in both regimes, evaluates at α_test ∈ {0, 0.1}, and checks that the noisy-trained model has smaller or similar accuracy drop.

```bash
python sanity_check.py
```

Expect: `Sanity OK: noisy-trained has smaller or similar drop ...` (with 1 epoch, variance can occasionally reverse this).

### Optional module checks

```bash
python data.py    # DataLoader shapes for CIFAR-10
python -c "from models import get_model, ARCH_NAMES; [get_model(n) for n in ARCH_NAMES]"  # Forward pass
python noise.py   # Weight noise apply/remove restores params
python evaluate.py  # Weights unchanged after sweep (invariant check), then optional checkpoint sweep
```

---

## Customization

### Config (`config.py`)

Single place for seeds, noise strengths, and training/eval settings.

| Variable | Meaning | Default |
|----------|---------|--------|
| `SEEDS` | Random seeds (one model per seed × regime × arch) | `[0, 1, 2]` |
| `ALPHA_TRAIN` | Fallback when alpha_train not passed (orchestration uses ALPHA_TRAIN_LIST) | `0.05` |
| `ALPHA_TRAIN_LIST` | [clean_alpha, noisy_alpha]; run_experiments trains both regimes from this | `[0.0, 0.05]` |
| `ALPHA_TEST_LIST` | Evaluation perturbation strengths for robustness sweep (optionally add 0.5 in config for a “break” point) | `[0, 0.01, 0.02, 0.05, 0.1, 0.2]` |
| `ROBUSTNESS_NUM_SAMPLES` | Noise samples per α_test for averaging | `5` |
| `EPOCHS` | Training epochs (all architectures) | `40` |
| `BATCH_SIZE` | Training/eval batch size | `128` |
| `LEARNING_RATE` | Adam learning rate | `1e-3` |
| `WEIGHT_DECAY` | Adam weight decay (same for all architectures) | `5e-4` |
| `DATA_DIR` | Dataset root | `./data` |
| `CHECKPOINT_DIR` | Checkpoint root (per-dataset subdirs added by script) | `./checkpoints` |
| `RESULTS_DIR` | Results root (per-dataset subdirs) | `./results` |
| `DEVICE` | `"cuda"` if available, else `"cpu"` | auto |

Reproducibility: `set_seed()` (in `train.py`) sets PyTorch and CUDA seeds and uses `cudnn.deterministic=True`, `cudnn.benchmark=False` for tighter GPU reproducibility.

**Training recipe:** A single recipe (Adam, one LR, one weight decay, CIFAR-style augmentation, no scheduler) is used for all architectures so that robustness comparisons reflect geometry, not training quirks. Change config and re-run; checkpoint filenames include `alpha_train` so different noise levels do not overwrite.

### Dataset and data recipe

- **Supported:** CIFAR-10 only (see `data.DATASET_CHOICES`). Images are 32×32 RGB; optional `resize` in `get_loaders()` for architectures that need a different input size.
- **Transforms:** Train = RandomCrop(32, padding=4) + RandomHorizontalFlip + ToTensor + Normalize(CIFAR10_MEAN, CIFAR10_STD). Test = ToTensor + Normalize. If `resize` is set (e.g. for larger backbones), Resize is applied first, and RandomCrop uses that size. This standard recipe improves CNN/ResNet baselines and makes robustness comparisons defensible; MLPs work fine with normalized inputs.

### Neural architectures

- **Defined in:** `models/` package (`models/__init__.py`, `resnet_cifar.py`, `vit_lite.py`, etc.).
- **Factory:** `get_model(name, num_classes=10)`.
- **Core names:** `ARCH_NAMES = ["cnn", "mlp", "plainnet20", "resnet20", "mobilenet_v2", "vit_lite"]`.
- **Resize:** Centralized in `models.ARCH_INPUT_RESIZE` (default 32 for CIFAR); used by `run_experiments.py` for loaders.

To **add an architecture:** implement the model (input `(B, 3, 32, 32)` or documented resize, output `(B, num_classes)`), add a branch in `get_model()` in `models/__init__.py`, and append to `ARCH_NAMES` or `ARCH_NAMES_OPTIONAL`. Add an entry to `ARCH_INPUT_RESIZE` if the input size is not 32.

---

## Outputs

- **summary.csv:** One row per (architecture, regime, seed, dataset, alpha_train): `acc_0`, `acc_01`, `drop_at_01`.
- **sweep.csv:** Per (architecture, regime, seed, alpha_train, alpha_test): `acc`, `loss`.
- **results.json:** Same summary and sweep as structured JSON.

Summary table and mean drop at α=0.1 per (arch, regime) are printed to stdout after each run.
