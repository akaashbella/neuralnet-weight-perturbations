# Neural Net Weight Noise Experiment

Research prototype: study robustness to Gaussian weight perturbations and whether training-time weight noise improves post-training robustness.

- **Phase 1:** Config + MNIST (3-channel) + DataLoaders.

## Setup

```bash
pip install -r requirements.txt
```

## How to run

**Full experiment (24 models + robustness sweep + results):**

```bash
python run_experiments.py
```

- Trains 4 architectures × 2 regimes (clean/noisy) × 3 seeds = 24 models; skips any checkpoint that already exists.
- Runs robustness evaluation at α_test ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2} (5 noise samples per α for stability).
- Prints a summary table and mean drop at α=0.1; writes `checkpoints/*.pt`, `results/summary.csv`, `results/sweep.csv`, `results/results.json`.

**Run only selected architectures** (e.g. skip ViT locally; run ViT on HPC):

```bash
python run_experiments.py mlp cnn resnet18    # first 3 only (fast)
python run_experiments.py vit                 # ViT only (heavy; use HPC)
python run_experiments.py                    # all four (default)
```

- Each run trains and evaluates only the listed architectures. Results files are overwritten with the current run’s subset; combine CSVs manually if you run in separate batches (e.g. local + HPC).

**Quick sanity (MLP only, 1 epoch):**

```bash
python sanity_check.py
```

- Trains MLP clean + noisy for 1 epoch, evaluates at α ∈ {0, 0.1}, prints “Sanity OK” if noisy is more robust.

**Optional sanity checks** (data shapes, models, noise):

```bash
python data.py && python models.py && python noise.py
```

## Sanity checks (details)

```bash
python data.py   # Train/test batch shapes (B, 3, 28, 28)
python models.py # All 4 architectures: (2, 3, 28, 28) -> (2, 10)
python noise.py  # Weight noise apply/remove restores params
```

Expect: `Train batch: x.shape=torch.Size([128, 3, 28, 28]), ...`, `mlp/cnn/resnet18/vit: ok, ...`, and `noise.py: apply/remove ok`.

Phase 4: `python train.py` runs a short sanity train (1 epoch MLP clean + noisy, saves checkpoints).

Phase 5: `python evaluate.py` runs a robustness sweep on MLP checkpoints (if present) and prints acc@0, acc@0.1, drop.

Phase 6–7: `python run_experiments.py` trains all 24 models, runs robustness sweep, prints summary table (Architecture, Regime, Seed, Acc(0), Acc(0.1), Drop) and mean drop per (arch, regime), and writes `results/summary.csv`, `results/sweep.csv`, `results/results.json`.

Phase 8: Code has short “why” comments at major blocks. Run the sanity check:

```bash
python sanity_check.py
```

Trains MLP 1 epoch (clean + noisy), evaluates at α_test ∈ {0, 0.1}, and prints clean vs noisy drop. Expect “Sanity OK: noisy-trained has smaller or similar drop.” With only 1 epoch, noisy drop can occasionally exceed clean; use more epochs for stable comparison.
