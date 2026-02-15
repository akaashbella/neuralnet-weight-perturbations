# Scripts

Visualization and plotting scripts for thesis-style figures. Run from the **repository root**:

```bash
python scripts/viz_sweep.py --dataset cifar10
```

## viz_sweep.py

Generates robustness and degradation plots from existing sweep/summary CSVs under `./results/<dataset>/`.

**Usage**

```bash
python scripts/viz_sweep.py --dataset cifar10 [options]
```

**Required**

- `--dataset` - Dataset subdir under `results/` (e.g. `cifar10`).

**Options**

- `--regimes` - Comma-separated regimes (default: `clean,noisy`).
- `--architectures` - Optional comma-separated architecture filter; empty = all.
- `--out_dir` - Output directory; if empty, uses `./results/figures/<dataset>/`.
- `--show` - Call `plt.show()` after saving (interactive).
- `--log_x` - Use log scale for alpha_test on curve plots.

**Inputs**

- Reads `./results/<dataset>/sweep.csv` with exact headers: `architecture`, `regime`, `seed`, `dataset`, `alpha_train`, `alpha_test`, `acc`, `loss`. Uses `acc` and `loss` directly (no column guessing).
- Reads `./results/<dataset>/summary.csv` if present (optional; used for drop_at_01 plot).

If a file is missing or sweep.csv has wrong columns, the script prints a clear message and still generates what it can.

**Outputs (saved to `./results/figures/<dataset>/` by default)**

- `robustness_curves_acc_<regime>.png` - Accuracy vs alpha_test, one line per architecture.
- `robustness_curves_loss_<regime>.png` - Loss vs alpha_test.
- `degradation_curves_acc_<regime>.png` - acc(alpha0) - acc(alpha) per regime.
- `degradation_curves_loss_<regime>.png` - loss(alpha) - loss(alpha0) per regime.
- `auc_by_arch_acc.png` - Trapezoidal AUC of accuracy by architecture and regime.
- `auc_by_arch_loss.png` - Trapezoidal AUC of loss by architecture and regime.
- `initial_sensitivity_acc.png` - Initial sensitivity (acc drop from baseline to min positive alpha).
- `initial_sensitivity_loss.png` - Initial sensitivity (loss rise from baseline to min positive alpha).
- `drop_at_01.png` - Drop at alpha_test=0.1 from summary.csv (if available).

Baseline alpha0 is the alpha_test value closest to 0 (tolerance 1e-12). Aggregation: mean and std over seeds per (architecture, regime, alpha_test). A text table (architecture, regime, acc0, loss0, auc_acc, auc_loss, sens_acc, sens_loss) is printed to stdout. If an (arch, regime) is missing alpha_test values vs the global set, a warning is printed and AUC/sensitivity use the available range only.

---

## viz_loss_landscape_slice.py (optional, expensive)

Computes a 2D loss slice around a single trained checkpoint and saves a contour and surface plot. **Expensive**: evaluates loss on a grid of weight-space directions with limited batches per point.

**Usage**

```bash
python scripts/viz_loss_landscape_slice.py --dataset cifar10 --arch cnn_large --regime clean --seed 0 --grid 31 --span 0.5 --batch_limit 10
```

**Required**

- `--dataset` - Dataset name (e.g. `cifar10`).
- `--arch` - Architecture name (e.g. `cnn_large`, `mlp`, `resnet20`).
- `--regime` - `clean` or `noisy`.

**Options**

- `--seed` - Checkpoint seed (default: 0).
- `--grid` - Grid size, e.g. 31 for 31x31 points (default: 31).
- `--span` - Range for each direction axis: [-span, span] (default: 0.5).
- `--batch_limit` - Max number of batches from the test loader per grid point (default: 10).
- `--dir_seed` - Random seed for the two direction vectors (default: 42).

**Inputs**

- Checkpoint: `./checkpoints/<dataset>/<arch>_<regime>_seed<seed>_*.pt` (pattern match).
- Model is created via the repo model factory (`models.get_model`).
- Data: repo data loaders for the given dataset.

**Outputs**

Saved under:

`./results/figures/<dataset>/loss_landscape/<arch>_<regime>_seed<seed>/`

- `loss_contour.png` - Contour plot of loss over the 2D slice.
- `loss_surface.png` - 3D surface plot.

If the checkpoint is missing, the model cannot be created, or data loaders fail, the script prints a clear error and exits.
