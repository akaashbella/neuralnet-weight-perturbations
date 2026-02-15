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
- `--log_x` - Use symlog scale for alpha_test on curve plots (linthresh=1e-3 so alpha=0 is allowed).

**Inputs**

- Reads `./results/<dataset>/sweep.csv` with exact headers: `architecture`, `regime`, `seed`, `dataset`, `alpha_train`, `alpha_test`, `acc`, `loss`. Uses `acc` and `loss` directly. If sweep.csv is missing, tries `./results/<dataset>/results.json` and uses its `"sweep"` array (fallback only).
- Reads `./results/<dataset>/summary.csv` if present (optional; used for drop_at_01 plot).

If a file is missing or sweep.csv has wrong columns, the script prints a clear message and still generates what it can.

**Outputs (saved to `./results/figures/<dataset>/` by default)**

- `robustness_curves_acc_<regime>.png` - Accuracy vs alpha_test, one line per architecture.
- `robustness_curves_loss_<regime>.png` - Loss vs alpha_test.
- `degradation_curves_acc_<regime>.png` - acc(alpha0) - acc(alpha) per regime; std band is computed per seed then aggregated (correct degradation std).
- `degradation_curves_loss_<regime>.png` - loss(alpha) - loss(alpha0) per regime; same per-seed std.
- `auc_by_arch_acc.png` - Trapezoidal AUC of accuracy by architecture and regime.
- `auc_by_arch_loss.png` - Trapezoidal AUC of loss by architecture and regime.
- `initial_sensitivity_acc.png` - Initial sensitivity (acc drop from baseline to min positive alpha).
- `initial_sensitivity_loss.png` - Initial sensitivity (loss rise from baseline to min positive alpha).
- `drop_at_01.png` - Drop at alpha_test=0.1 from summary.csv (if available).

Baseline alpha0 is the alpha_test value closest to 0 (argmin |alpha|). A warning is printed if baseline is not exactly 0 (missing alpha=0 data). Aggregation: mean and std over seeds per (architecture, regime, alpha_test); degradation curves use per-seed degradation then mean/std over seeds. The script prints the expected global alpha_test list once to stderr; if an (arch, regime) is missing values vs that set, a warning is printed and AUC/sensitivity use the available range only.

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
- `--batch_limit` - Max number of batches per grid point (default: 10).
- `--dir_seed` - Random seed for the two direction vectors (default: 42).
- `--split` - Data split for loss: `train` or `test` (default: `train`). Train loss aligns with training objective; test loss mixes geometry with generalization.

**Inputs**

- Checkpoint: `./checkpoints/<dataset>/<arch>_<regime>_seed<seed>_*.pt` (pattern match; most recent by modification time is chosen). Loading supports wrapped formats: if the file contains a dict with `state_dict` or `model_state_dict`, that is used; otherwise param-like keys (with dots) are treated as state_dict.
- Model is created via the repo model factory (`models.get_model`).
- Data: repo `get_loaders`; train or test loader is selected by `--split`.

**Outputs**

Saved under:

`./results/figures/<dataset>/loss_landscape/<arch>_<regime>_seed<seed>/`

- `loss_contour.png` - Contour plot of loss over the 2D slice.
- `loss_surface.png` - 3D surface plot.

If the checkpoint is missing, the model cannot be created, or data loaders fail, the script prints a clear error and exits.
