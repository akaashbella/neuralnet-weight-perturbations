# Neural Net Weight Perturbations

Research prototype for studying **robustness to Gaussian weight perturbations** and whether **training-time weight noise** improves post-training robustness. Models are trained in two regimes (clean vs. noisy); at evaluation time we sweep over perturbation strengths (α_test) and measure accuracy drop.

## Features

- **Dataset:** CIFAR-10 (32×32 RGB, 10 classes; optional resize per architecture).
- **Architectures:** 5 models — MLP (small / medium / large), Simple CNN, ResNet-18. All use 32×32 input.
- **Training:** Clean regime (standard training) or noisy regime (per-batch Gaussian weight noise before forward/backward; optimizer updates clean weights).
- **Evaluation:** Robustness sweep over α_test; accuracy drop at α=0.1 as main metric. Results merged into dataset-specific CSVs and JSON.

## Setup

```bash
pip install -r requirements.txt
```

Requires: `torch>=2.0.0`, `torchvision>=0.15.0`. Data is downloaded under `./data` on first run.

---

## Running Experiments

### Full experiment (all architectures, one dataset)

Trains every architecture in both regimes over 3 seeds, then runs the robustness sweep and writes results. Checkpoints and results are under `checkpoints/cifar10/` and `results/cifar10/`.

```bash
# Default: CIFAR-10, all 5 architectures (5 × 2 regimes × 3 seeds = 30 models)
python run_experiments.py
```

- **Checkpoints:** `checkpoints/cifar10/{arch}_{regime}_seed{seed}_a{alpha_train}.pt`
- **Results:** `results/cifar10/summary.csv`, `results/cifar10/sweep.csv`, `results/cifar10/results.json`
- Existing checkpoints are **skipped**; new runs are **merged** into existing summary/sweep CSVs by (architecture, regime, seed, alpha_train).

### Choose architectures

- **Architectures:** pass as positional arguments. If none are given, all are run.

```bash
# Only MLP variants and CNN (faster)
python run_experiments.py mlp_small mlp_medium mlp_large cnn

# Only ResNet-18
python run_experiments.py resnet18

# All 5 architectures (default when no args)
python run_experiments.py mlp_small mlp_medium mlp_large cnn resnet18
```

**Architecture names:** `mlp_small`, `mlp_medium`, `mlp_large`, `cnn`, `resnet18`.

### Quick sanity check (1 epoch, MLP only)

Trains `mlp_small` for 1 epoch in both regimes, evaluates at α_test ∈ {0, 0.1}, and checks that the noisy-trained model has smaller or similar accuracy drop.

```bash
python sanity_check.py
```

Expect: `Sanity OK: noisy-trained has smaller or similar drop ...` (with 1 epoch, variance can occasionally reverse this).

### Optional module checks

```bash
python data.py    # DataLoader shapes for CIFAR-10
python models.py  # All architectures: (2, 3, 32, 32) → (2, 10)
python noise.py   # Weight noise apply/remove restores params
```

---

## Customization

### Config (`config.py`)

Single place for seeds, noise strengths, and training/eval settings.

| Variable | Meaning | Default |
|----------|---------|--------|
| `SEEDS` | Random seeds (one model per seed × regime × arch) | `[0, 1, 2]` |
| `ALPHA_TRAIN` | Training-time weight noise (noisy regime only) | `0.2` |
| `ALPHA_TEST_LIST` | Evaluation perturbation strengths | `[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]` |
| `ROBUSTNESS_NUM_SAMPLES` | Noise samples per α_test for averaging | `5` |
| `EPOCHS` | Training epochs (all architectures) | `10` |
| `BATCH_SIZE` | Training/eval batch size | `128` |
| `LEARNING_RATE` | Adam learning rate | `1e-3` |
| `DATA_DIR` | Dataset root | `./data` |
| `CHECKPOINT_DIR` | Checkpoint root (per-dataset subdirs added by script) | `./checkpoints` |
| `RESULTS_DIR` | Results root (per-dataset subdirs) | `./results` |
| `DEVICE` | `"cuda"` if available, else `"cpu"` | auto |

Change these and re-run; checkpoint filenames include `alpha_train` so different noise levels do not overwrite.

### Dataset

- **Supported:** CIFAR-10 only (see `data.DATASET_CHOICES`). Images are 32×32 RGB; optional `resize` in `get_loaders()` for architectures that need a different input size.

### Neural architectures

- **Defined in:** `models.py`.
- **Factory:** `get_model(name, num_classes=10)`.
- **Names:** `ARCH_NAMES = ["mlp_small", "mlp_medium", "mlp_large", "cnn", "resnet18"]`.

To **add an architecture:**

1. Implement the model (or wrap a torchvision model) so it takes `(B, C, H, W)` and returns `(B, num_classes)` logits.
2. In `get_model()`, add a branch: `if name == "your_arch": return YourModel(...)`.
3. Append `"your_arch"` to `ARCH_NAMES`.
4. If the model expects a different input size (e.g. 224×224), use `get_loaders(dataset, resize=224)` for that architecture in `run_experiments.py` (or add a `RESIZE_ARCHS` pattern if you extend the script).

MLP variants differ only by hidden dims: `mlp_small` (256, 128), `mlp_medium` (512, 256), `mlp_large` (1024, 512, 256). All use 3×32×32 flattened input and 10 classes unless you change `num_classes` or the data pipeline.

---

## Outputs

- **summary.csv:** One row per (architecture, regime, seed, dataset, alpha_train): `acc_0`, `acc_01`, `drop_at_01`.
- **sweep.csv:** Per (architecture, regime, seed, alpha_train, alpha_test): `acc`, `loss`.
- **results.json:** Same summary and sweep as structured JSON.

Summary table and mean drop at α=0.1 per (arch, regime) are printed to stdout after each run.
