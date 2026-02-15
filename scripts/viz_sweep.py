"""
Generate plots from sweep/summary CSV outputs.
Assumes sweep.csv has headers: architecture, regime, seed, dataset, alpha_train, alpha_test, acc, loss.
Run from repo root: python scripts/viz_sweep.py --dataset cifar10
Uses matplotlib only; saves to results/figures/<dataset>/.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

EXPECTED_SWEEP_HEADERS = ["architecture", "regime", "seed", "dataset", "alpha_train", "alpha_test", "acc", "loss"]
ALPHA0_TOL = 1e-12


def _sweep_rows_from_csv(path):
    """Parse sweep CSV with exact headers; return list of dicts or empty if columns missing."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        colnames = reader.fieldnames or []
        missing = [h for h in EXPECTED_SWEEP_HEADERS if h not in colnames]
        if missing:
            print(f"sweep.csv missing columns: {missing}. Expected: {EXPECTED_SWEEP_HEADERS}. Skipping.", file=sys.stderr)
            return []
        for r in reader:
            try:
                r["alpha_test"] = float(r["alpha_test"])
                r["acc"] = float(r["acc"])
                r["loss"] = float(r["loss"])
            except (TypeError, ValueError):
                continue
            r["seed"] = int(r["seed"])
            rows.append(r)
    return rows


def load_sweep(results_dir):
    """Load sweep from sweep.csv; fallback to results.json 'sweep' if CSV missing. Return list of dicts with float alpha_test, acc, loss."""
    path = os.path.join(results_dir, "sweep.csv")
    if os.path.isfile(path):
        return _sweep_rows_from_csv(path)
    # Fallback: results.json
    json_path = os.path.join(results_dir, "results.json")
    if not os.path.isfile(json_path):
        return []
    try:
        import json
        with open(json_path) as f:
            data = json.load(f)
        sweep = data.get("sweep")
        if not isinstance(sweep, list):
            return []
        rows = []
        for r in sweep:
            try:
                r = dict(r)
                r["alpha_test"] = float(r.get("alpha_test", 0))
                r["acc"] = float(r.get("acc", 0))
                r["loss"] = float(r.get("loss", 0))
                r["seed"] = int(r.get("seed", 0))
                for k in ["architecture", "regime", "dataset"]:
                    if k in r and not isinstance(r[k], str):
                        r[k] = str(r[k])
                rows.append(r)
            except (TypeError, ValueError, KeyError):
                continue
        if rows:
            print("Loaded sweep from results.json (sweep.csv not found).", file=sys.stderr)
        return rows
    except Exception as e:
        print(f"Failed to load results.json: {e}", file=sys.stderr)
        return []


def load_summary(results_dir):
    """Load summary.csv; return list of dicts. Empty if file missing."""
    path = os.path.join(results_dir, "summary.csv")
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                r["acc_0"] = float(r.get("acc_0", 0))
                r["acc_01"] = float(r.get("acc_01", 0))
                r["drop_at_01"] = float(r.get("drop_at_01", 0))
            except (TypeError, ValueError):
                continue
            r["architecture"] = r.get("architecture", "")
            r["regime"] = r.get("regime", "")
            rows.append(r)
    return rows


def filter_rows(rows, regimes, architectures):
    """Filter by regime list and optional architecture list."""
    regime_set = {s.strip() for s in regimes.split(",") if s.strip()}
    arch_set = {s.strip() for s in architectures.split(",") if s.strip()} if architectures else None
    out = []
    for r in rows:
        if r.get("regime") not in regime_set:
            continue
        if arch_set is not None and r.get("architecture") not in arch_set:
            continue
        out.append(r)
    return out


def aggregate_sweep(sweep_rows):
    """
    Build per-seed curves then aggregate. Robustness: mean and std over seeds at each alpha.
    Degradation: per seed deg(alpha) = acc(alpha0) - acc(alpha) or loss(alpha) - loss(alpha0); then mean and std over seeds.
    Returns: dict (arch, regime) -> alphas, acc_mean, acc_std, loss_mean, loss_std, acc_deg_mean, acc_deg_std, loss_deg_mean, loss_deg_std.
    """
    # (arch, regime, seed) -> {alpha: (acc, loss)}
    curves = defaultdict(dict)
    for r in sweep_rows:
        key = (r["architecture"], r["regime"], r["seed"])
        alpha = r["alpha_test"]
        curves[key][alpha] = (r["acc"], r["loss"])

    out = {}
    # Group by (arch, regime)
    keys_ar = defaultdict(set)
    for (arch, reg, seed) in curves:
        keys_ar[(arch, reg)].add(seed)
    for (arch, reg), seeds in keys_ar.items():
        seeds = sorted(seeds)
        global_alphas = sorted(set().union(*(set(curves[(arch, reg, s)].keys()) for s in seeds)))
        if not global_alphas:
            continue
        alpha0 = min(global_alphas, key=lambda a: abs(a))
        i0 = global_alphas.index(alpha0)

        acc_means, acc_stds, loss_means, loss_stds = [], [], [], []
        acc_deg_vals, loss_deg_vals = defaultdict(list), defaultdict(list)
        for alpha in global_alphas:
            acc_vals = [curves[(arch, reg, s)][alpha][0] for s in seeds if alpha in curves[(arch, reg, s)]]
            loss_vals = [curves[(arch, reg, s)][alpha][1] for s in seeds if alpha in curves[(arch, reg, s)]]
            acc_means.append(np.mean(acc_vals) if acc_vals else float("nan"))
            acc_stds.append(np.std(acc_vals) if len(acc_vals) > 1 else 0.0)
            loss_means.append(np.mean(loss_vals) if loss_vals else float("nan"))
            loss_stds.append(np.std(loss_vals) if len(loss_vals) > 1 else 0.0)
            for s in seeds:
                c = curves[(arch, reg, s)]
                if alpha0 in c and alpha in c:
                    acc_deg_vals[alpha].append(c[alpha0][0] - c[alpha][0])
                    loss_deg_vals[alpha].append(c[alpha][1] - c[alpha0][1])
        acc_deg_mean = [np.mean(acc_deg_vals[a]) if acc_deg_vals[a] else float("nan") for a in global_alphas]
        acc_deg_std = [np.std(acc_deg_vals[a]) if len(acc_deg_vals[a]) > 1 else 0.0 for a in global_alphas]
        loss_deg_mean = [np.mean(loss_deg_vals[a]) if loss_deg_vals[a] else float("nan") for a in global_alphas]
        loss_deg_std = [np.std(loss_deg_vals[a]) if len(loss_deg_vals[a]) > 1 else 0.0 for a in global_alphas]

        out[(arch, reg)] = {
            "alphas": global_alphas,
            "acc_mean": np.array(acc_means),
            "acc_std": np.array(acc_stds) if any(s > 0 for s in acc_stds) else None,
            "loss_mean": np.array(loss_means),
            "loss_std": np.array(loss_stds) if any(s > 0 for s in loss_stds) else None,
            "acc_deg_mean": np.array(acc_deg_mean),
            "acc_deg_std": np.array(acc_deg_std) if any(s > 0 for s in acc_deg_std) else None,
            "loss_deg_mean": np.array(loss_deg_mean),
            "loss_deg_std": np.array(loss_deg_std) if any(s > 0 for s in loss_deg_std) else None,
        }
    return out


def global_alpha_set(aggregated):
    """Return sorted set of all alpha_test values across (arch, regime)."""
    all_alphas = set()
    for agg in aggregated.values():
        all_alphas.update(agg["alphas"])
    return sorted(all_alphas)


def baseline_alpha_index(alphas, tol=ALPHA0_TOL):
    """Index of alpha_test closest to 0 (argmin |alpha|)."""
    if not alphas:
        return None
    return int(np.argmin(np.abs(np.asarray(alphas))))


def warn_baseline_not_zero(aggregated, tol=ALPHA0_TOL):
    """Warn once per (arch, regime) if baseline alpha is not exactly 0."""
    for (arch, reg), agg in sorted(aggregated.items()):
        alphas = agg["alphas"]
        if not alphas:
            continue
        i0 = baseline_alpha_index(alphas)
        alpha0_val = alphas[i0]
        if abs(alpha0_val) > tol:
            print(f"Warning: ({arch}, {reg}) baseline alpha_test is {alpha0_val} (closest to 0); missing alpha=0 data.", file=sys.stderr)


def compute_auc(alphas, mean_vals):
    """Trapezoidal AUC over (alphas, mean_vals)."""
    if len(alphas) < 2:
        return 0.0
    return float(np.trapz(mean_vals, alphas))


def warn_missing_alpha(aggregated, global_alphas):
    """Print expected global alphas once; warn for each (arch, regime) missing values vs global set."""
    print(f"Expected global alpha_test values: {global_alphas}", file=sys.stderr)
    global_set = set(global_alphas)
    for (arch, reg), agg in sorted(aggregated.items()):
        have = set(agg["alphas"])
        if have != global_set:
            missing = global_set - have
            print(f"Warning: ({arch}, {reg}) missing alpha_test values {sorted(missing)}; AUC/sensitivity over available range only.", file=sys.stderr)


def initial_sensitivity(agg, metric):
    """
    For acc: acc(alpha0) - acc(min positive alpha). For loss: loss(min positive alpha) - loss(alpha0).
    Returns (baseline_value, sensitivity). baseline_value is acc0 or loss0; sensitivity is the drop or rise.
    """
    alphas = agg["alphas"]
    if not alphas:
        return float("nan"), float("nan")
    i0 = baseline_alpha_index(alphas)
    if i0 is None:
        return float("nan"), float("nan")
    if metric == "acc":
        mean = agg["acc_mean"]
        baseline = mean[i0]
        positive_idx = [i for i, a in enumerate(alphas) if a > alphas[i0] + ALPHA0_TOL]
        if not positive_idx:
            return baseline, 0.0
        j = min(positive_idx, key=lambda i: alphas[i])
        return baseline, baseline - mean[j]
    else:
        mean = agg["loss_mean"]
        baseline = mean[i0]
        positive_idx = [i for i, a in enumerate(alphas) if a > alphas[i0] + ALPHA0_TOL]
        if not positive_idx:
            return baseline, 0.0
        j = min(positive_idx, key=lambda i: alphas[i])
        return baseline, mean[j] - baseline


def plot_robustness_curves(aggregated, regime, metric, out_path, dataset, log_x=False):
    """One figure: metric vs alpha_test, one line per architecture, optional std band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"
    ylabel = "Accuracy" if metric == "acc" else "Loss"

    fig, ax = plt.subplots()
    for (arch, reg), agg in sorted(aggregated.items()):
        if reg != regime:
            continue
        alphas = agg["alphas"]
        mean = agg[mean_key]
        std = agg[std_key]
        ax.plot(alphas, mean, label=arch)
        if std is not None and np.any(std > 0):
            ax.fill_between(alphas, mean - std, mean + std, alpha=0.25)
    ax.set_xlabel("alpha_test")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset} - {regime}: robustness ({metric})")
    if log_x:
        ax.set_xscale("symlog", linthresh=1e-3)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_degradation_curves(aggregated, regime, metric, out_path, dataset, log_x=False):
    """Degradation computed per seed then mean/std. Acc: acc(alpha0)-acc(alpha). Loss: loss(alpha)-loss(alpha0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_key = f"{metric}_deg_mean"
    std_key = f"{metric}_deg_std"
    if metric == "acc":
        ylabel = "Degradation (acc(alpha0) - acc(alpha))"
    else:
        ylabel = "Degradation (loss(alpha) - loss(alpha0))"

    fig, ax = plt.subplots()
    for (arch, reg), agg in sorted(aggregated.items()):
        if reg != regime:
            continue
        alphas = agg["alphas"]
        mean = agg[mean_key]
        std = agg[std_key]
        ax.plot(alphas, mean, label=arch)
        if std is not None and np.any(std > 0):
            ax.fill_between(alphas, mean - std, mean + std, alpha=0.25)
    ax.set_xlabel("alpha_test")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset} - {regime}: degradation ({metric})")
    if log_x:
        ax.set_xscale("symlog", linthresh=1e-3)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_auc_bars(aggregated, metric, out_path, dataset, regimes_order):
    """Grouped bars: AUC (trapezoidal on mean) per architecture and regime."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_key = f"{metric}_mean"
    regimes_order = regimes_order or ["clean", "noisy"]
    archs = sorted({k[0] for k in aggregated.keys()})
    if not archs:
        return

    auc_map = {}
    for (arch, reg), agg in aggregated.items():
        auc_map[(arch, reg)] = compute_auc(agg["alphas"], agg[mean_key])

    x = np.arange(len(archs))
    width = 0.35
    fig, ax = plt.subplots()
    for i, reg in enumerate(regimes_order):
        vals = [auc_map.get((a, reg), float("nan")) for a in archs]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=reg)
    ax.set_ylabel(f"Robustness AUC ({metric})")
    ax.set_title(f"{dataset}: AUC by architecture and regime ({metric})")
    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_initial_sensitivity_bars(aggregated, metric, out_path, dataset, regimes_order=None):
    """Grouped bars: initial sensitivity (acc drop or loss rise from alpha0 to min positive alpha)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes_order = regimes_order or ["clean", "noisy"]
    archs = sorted({k[0] for k in aggregated.keys()})
    if not archs:
        return

    sens_map = {}
    for (arch, reg), agg in aggregated.items():
        _, sens = initial_sensitivity(agg, metric)
        sens_map[(arch, reg)] = sens

    x = np.arange(len(archs))
    width = 0.35
    fig, ax = plt.subplots()
    for i, reg in enumerate(regimes_order):
        vals = [sens_map.get((a, reg), float("nan")) for a in archs]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=reg)
    ax.set_ylabel(f"Initial sensitivity ({metric})")
    ax.set_title(f"{dataset}: initial sensitivity by architecture and regime ({metric})")
    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_drop_at_01_from_summary(summary_rows, out_path, dataset, regimes_order=None, architectures_filter=None):
    """Bar chart of drop_at_01 from summary.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes_order = regimes_order or ["clean", "noisy"]
    by_key = defaultdict(list)
    for r in summary_rows:
        key = (r["architecture"], r["regime"])
        if architectures_filter and r["architecture"] not in architectures_filter:
            continue
        by_key[key].append(r["drop_at_01"])
    mean_drop = {k: np.mean(v) for k, v in by_key.items()}
    archs = sorted({k[0] for k in mean_drop.keys()})
    if not archs:
        return

    x = np.arange(len(archs))
    width = 0.35
    fig, ax = plt.subplots()
    for i, reg in enumerate(regimes_order):
        vals = [mean_drop.get((a, reg), float("nan")) for a in archs]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=reg)
    ax.set_ylabel("Drop at alpha_test = 0.1")
    ax.set_title(f"{dataset}: drop_at_01 (from summary)")
    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def print_table(aggregated, regimes_order=None):
    """Print text table: architecture, regime, acc0, loss0, auc_acc, auc_loss, initial_sens_acc, initial_sens_loss."""
    regimes_order = regimes_order or ["clean", "noisy"]
    rows = []
    for (arch, reg), agg in aggregated.items():
        acc0, _ = initial_sensitivity(agg, "acc")
        loss0, _ = initial_sensitivity(agg, "loss")
        _, sens_acc = initial_sensitivity(agg, "acc")
        _, sens_loss = initial_sensitivity(agg, "loss")
        auc_acc = compute_auc(agg["alphas"], agg["acc_mean"])
        auc_loss = compute_auc(agg["alphas"], agg["loss_mean"])
        rows.append((arch, reg, acc0, loss0, auc_acc, auc_loss, sens_acc, sens_loss))
    rows.sort(key=lambda r: (r[0], r[1]))

    fmt = "{:12} {:8} {:8} {:8} {:10} {:10} {:12} {:12}"
    print(fmt.format("architecture", "regime", "acc0", "loss0", "auc_acc", "auc_loss", "sens_acc", "sens_loss"))
    print("-" * 82)
    for arch, reg, acc0, loss0, auc_acc, auc_loss, sens_acc, sens_loss in rows:
        print(fmt.format(arch, reg, f"{acc0:.4f}", f"{loss0:.4f}", f"{auc_acc:.4f}", f"{auc_loss:.4f}", f"{sens_acc:.4f}", f"{sens_loss:.4f}"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from sweep.csv (exact headers: architecture, regime, seed, dataset, alpha_train, alpha_test, acc, loss).",
    )
    parser.add_argument("--dataset", required=True, help="Dataset subdir under results (e.g. cifar10).")
    parser.add_argument("--regimes", default="clean,noisy", help="Comma-separated regimes (default: clean,noisy).")
    parser.add_argument("--architectures", default="", help="Optional comma-separated architecture filter; empty = all.")
    parser.add_argument("--out_dir", default="", help="Output directory; if empty, use ./results/figures/<dataset>/.")
    parser.add_argument("--show", action="store_true", help="Call plt.show() after saving.")
    parser.add_argument("--log_x", action="store_true", help="Use log scale for alpha_test on curve plots.")
    args = parser.parse_args()

    results_dir = os.path.join(config.RESULTS_DIR, args.dataset)
    out_dir = args.out_dir or os.path.join("results", "figures", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(results_dir):
        print(f"Results dir not found: {results_dir}", file=sys.stderr)
        return 1

    sweep_rows = load_sweep(results_dir)
    summary_rows = load_summary(results_dir)

    if not sweep_rows:
        print("No sweep data (sweep.csv and results.json sweep not found or empty). Generate what we can from summary only.", file=sys.stderr)

    sweep_rows = filter_rows(sweep_rows, args.regimes, args.architectures)
    summary_rows = [r for r in summary_rows if r.get("regime") in {s.strip() for s in args.regimes.split(",")}]
    if args.architectures:
        arch_set = {s.strip() for s in args.architectures.split(",")}
        summary_rows = [r for r in summary_rows if r.get("architecture") in arch_set]

    regimes_list = [s.strip() for s in args.regimes.split(",") if s.strip()]
    generated_any = False

    if sweep_rows:
        aggregated = aggregate_sweep(sweep_rows)
        global_alphas = global_alpha_set(aggregated)
        warn_missing_alpha(aggregated, global_alphas)
        warn_baseline_not_zero(aggregated)
        print_table(aggregated, regimes_order=regimes_list)

        for regime in regimes_list:
            if not any(k[1] == regime for k in aggregated):
                continue
            plot_robustness_curves(
                aggregated, regime, "acc",
                os.path.join(out_dir, f"robustness_curves_acc_{regime}.png"),
                args.dataset, log_x=args.log_x,
            )
            plot_robustness_curves(
                aggregated, regime, "loss",
                os.path.join(out_dir, f"robustness_curves_loss_{regime}.png"),
                args.dataset, log_x=args.log_x,
            )
            plot_degradation_curves(
                aggregated, regime, "acc",
                os.path.join(out_dir, f"degradation_curves_acc_{regime}.png"),
                args.dataset, log_x=args.log_x,
            )
            plot_degradation_curves(
                aggregated, regime, "loss",
                os.path.join(out_dir, f"degradation_curves_loss_{regime}.png"),
                args.dataset, log_x=args.log_x,
            )
            generated_any = True

        plot_auc_bars(aggregated, "acc", os.path.join(out_dir, "auc_by_arch_acc.png"), args.dataset, regimes_list)
        plot_auc_bars(aggregated, "loss", os.path.join(out_dir, "auc_by_arch_loss.png"), args.dataset, regimes_list)
        plot_initial_sensitivity_bars(aggregated, "acc", os.path.join(out_dir, "initial_sensitivity_acc.png"), args.dataset, regimes_list)
        plot_initial_sensitivity_bars(aggregated, "loss", os.path.join(out_dir, "initial_sensitivity_loss.png"), args.dataset, regimes_list)
        generated_any = True

    if summary_rows and any("drop_at_01" in r for r in summary_rows):
        arch_filter = {s.strip() for s in args.architectures.split(",")} if args.architectures else None
        plot_drop_at_01_from_summary(
            summary_rows,
            os.path.join(out_dir, "drop_at_01.png"),
            args.dataset,
            regimes_order=regimes_list,
            architectures_filter=arch_filter,
        )
        generated_any = True

    if args.show and generated_any:
        import matplotlib.pyplot as plt
        plt.show()

    if not generated_any:
        print("No plots generated. Ensure sweep.csv exists with expected columns and contains data.", file=sys.stderr)
        return 1

    print(f"Plots saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
