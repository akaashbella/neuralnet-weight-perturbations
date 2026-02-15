"""
Generate plots from sweep/summary CSV outputs.
Run from repo root: python scripts/viz_sweep.py --dataset cifar10
Uses matplotlib only; saves to results/figures/<dataset>/.
"""

import argparse
import csv
import os
import sys

import numpy as np

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Candidate column names for accuracy (first match wins)
ACC_COL_CANDIDATES = ["acc", "acc_mean", "accuracy", "top1", "test_acc"]


def _pick_acc_column(columns, metric_hint="acc"):
    """Return first column name that exists and is usable for accuracy. Warn which one is used."""
    for c in ACC_COL_CANDIDATES:
        if c in columns:
            if c != metric_hint:
                print(f"Using accuracy column: '{c}' (requested '{metric_hint}' not found)", file=sys.stderr)
            return c
    return None


def load_sweep(results_dir, alpha_col="alpha_test", metric="acc"):
    """Load sweep.csv; return list of dicts with float alpha and chosen acc column. Empty if file missing."""
    path = os.path.join(results_dir, "sweep.csv")
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        colnames = reader.fieldnames or []
        acc_col = _pick_acc_column(colnames, metric)
        if acc_col is None:
            print(f"No accuracy column found in sweep.csv (tried {ACC_COL_CANDIDATES}). Skipping sweep.", file=sys.stderr)
            return []
        for r in reader:
            try:
                r[alpha_col] = float(r.get(alpha_col, r.get("alpha_test", 0)))
                r["_acc"] = float(r.get(acc_col, 0))
            except (TypeError, ValueError):
                continue
            r["seed"] = int(r.get("seed", 0))
            r["architecture"] = r.get("architecture", "")
            r["regime"] = r.get("regime", "")
            rows.append(r)
    return rows


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


def filter_rows(rows, regimes, architectures, alpha_col="alpha_test", acc_key="_acc"):
    """Filter by regime list and optional architecture list. regimes/comma list; architectures/comma list or empty = all."""
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


def aggregate_sweep(sweep_rows, alpha_col="alpha_test", acc_key="_acc"):
    """
    Aggregate by (architecture, regime, alpha_test): mean and std across seeds.
    Returns: dict keyed by (arch, regime) -> {"alphas": sorted list, "mean": array, "std": array or None}.
    """
    from collections import defaultdict
    # (arch, regime) -> alpha -> list of acc values (one per seed/sample)
    by_key = defaultdict(lambda: defaultdict(list))
    for r in sweep_rows:
        key = (r["architecture"], r["regime"])
        alpha = r[alpha_col]
        by_key[key][alpha].append(r[acc_key])

    out = {}
    for key, alpha_to_vals in by_key.items():
        alphas = sorted(alpha_to_vals.keys())
        means = []
        stds = []
        for a in alphas:
            vals = alpha_to_vals[a]
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)
        out[key] = {
            "alphas": alphas,
            "mean": np.array(means),
            "std": np.array(stds) if any(s > 0 for s in stds) else None,
        }
    return out


def compute_auc(alphas, mean_acc):
    """Trapezoidal AUC over (alphas, mean_acc)."""
    if len(alphas) < 2:
        return 0.0
    return float(np.trapz(mean_acc, alphas))


def compute_initial_drop(agg):
    """Drop from alpha=0 to smallest positive alpha. Returns (acc0, drop) or (nan, nan) if no alpha=0."""
    alphas = agg["alphas"]
    mean = agg["mean"]
    if not alphas or alphas[0] != 0:
        return float("nan"), float("nan")
    acc0 = mean[0]
    positive = [a for a in alphas if a > 0]
    if not positive:
        return acc0, 0.0
    min_pos = min(positive)
    idx = alphas.index(min_pos)
    drop = acc0 - mean[idx]
    return acc0, drop


def plot_robustness_curves(aggregated, regime, out_path, dataset, log_x=False):
    """One figure: accuracy vs alpha_test, one line per architecture, optional std band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for (arch, reg), agg in sorted(aggregated.items()):
        if reg != regime:
            continue
        alphas = agg["alphas"]
        mean = agg["mean"]
        std = agg["std"]
        ax.plot(alphas, mean, label=arch)
        if std is not None and np.any(std > 0):
            ax.fill_between(alphas, mean - std, mean + std, alpha=0.25)
    ax.set_xlabel("alpha_test")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{dataset} - {regime}: robustness curves")
    if log_x:
        ax.set_xscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_degradation_curves(aggregated, regime, out_path, dataset, log_x=False):
    """Degradation = acc(0) - acc(alpha). One line per architecture."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for (arch, reg), agg in sorted(aggregated.items()):
        if reg != regime:
            continue
        alphas = agg["alphas"]
        mean = agg["mean"]
        if not alphas or alphas[0] != 0:
            continue
        acc0 = mean[0]
        degradation = acc0 - mean
        std = agg["std"]
        ax.plot(alphas, degradation, label=arch)
        if std is not None and np.any(std > 0):
            # std of (acc0 - acc) = std of acc
            ax.fill_between(alphas, degradation - std, degradation + std, alpha=0.25)
    ax.set_xlabel("alpha_test")
    ax.set_ylabel("Degradation (acc(0) - acc(alpha))")
    ax.set_title(f"{dataset} - {regime}: degradation curves")
    if log_x:
        ax.set_xscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_auc_bars(aggregated, out_path, dataset, regimes_order=None):
    """Grouped bars: per architecture, one group per regime (clean vs noisy), AUC value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    regimes_order = regimes_order or ["clean", "noisy"]
    archs = sorted({k[0] for k in aggregated.keys()})
    if not archs:
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    # (arch, regime) -> auc
    auc_map = {}
    for (arch, reg), agg in aggregated.items():
        auc_map[(arch, reg)] = compute_auc(agg["alphas"], agg["mean"])

    x = np.arange(len(archs))
    width = 0.35
    fig, ax = plt.subplots()
    for i, reg in enumerate(regimes_order):
        vals = [auc_map.get((a, reg), float("nan")) for a in archs]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=reg)
    ax.set_ylabel("Robustness AUC")
    ax.set_title(f"{dataset}: AUC by architecture and regime")
    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_initial_drop_bars(aggregated, out_path, dataset, regimes_order=None):
    """Grouped bars: initial drop (acc(0) - acc(min positive alpha)) by architecture and regime."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    regimes_order = regimes_order or ["clean", "noisy"]
    archs = sorted({k[0] for k in aggregated.keys()})
    if not archs:
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    drop_map = {}
    for (arch, reg), agg in aggregated.items():
        _, drop = compute_initial_drop(agg)
        drop_map[(arch, reg)] = drop

    x = np.arange(len(archs))
    width = 0.35
    fig, ax = plt.subplots()
    for i, reg in enumerate(regimes_order):
        vals = [drop_map.get((a, reg), float("nan")) for a in archs]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=reg)
    ax.set_ylabel("Initial drop (acc(0) - acc(min alpha>0))")
    ax.set_title(f"{dataset}: initial drop by architecture and regime")
    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_drop_at_01_from_summary(summary_rows, out_path, dataset, regimes_order=None, architectures_filter=None):
    """Bar or scatter of drop_at_01 from summary.csv across architectures and regimes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    regimes_order = regimes_order or ["clean", "noisy"]
    from collections import defaultdict
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
    """Print text table: architecture, regime, acc0, auc, initial_drop."""
    regimes_order = regimes_order or ["clean", "noisy"]
    rows = []
    for (arch, reg), agg in aggregated.items():
        acc0, initial_drop = compute_initial_drop(agg)
        auc = compute_auc(agg["alphas"], agg["mean"])
        rows.append((arch, reg, acc0, auc, initial_drop))
    rows.sort(key=lambda r: (r[0], r[1]))

    fmt = "{:14} {:8} {:8} {:10} {:10}"
    print(fmt.format("architecture", "regime", "acc0", "auc", "initial_drop"))
    print("-" * 52)
    for arch, reg, acc0, auc, drop in rows:
        print(fmt.format(arch, reg, f"{acc0:.4f}", f"{auc:.4f}", f"{drop:.4f}"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis plots from sweep/summary CSVs.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset subdir under results (e.g. cifar10).")
    parser.add_argument(
        "--regimes",
        default="clean,noisy",
        help="Comma-separated regimes to include (default: clean,noisy).",
    )
    parser.add_argument(
        "--architectures",
        default="",
        help="Optional comma-separated architecture filter; empty = all.",
    )
    parser.add_argument(
        "--metric",
        default="acc",
        help="Accuracy column hint (default: acc). First match from acc, acc_mean, accuracy, top1, test_acc.",
    )
    parser.add_argument(
        "--alpha_col",
        default="alpha_test",
        help="Column name for perturbation strength in sweep.csv (default: alpha_test).",
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="Output directory; if empty, use ./results/figures/<dataset>/.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Call plt.show() after saving (for interactive display).",
    )
    parser.add_argument(
        "--log_x",
        action="store_true",
        help="Use log scale for alpha_test on curve plots.",
    )
    args = parser.parse_args()

    results_dir = os.path.join(config.RESULTS_DIR, args.dataset)
    out_dir = args.out_dir or os.path.join("results", "figures", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(results_dir):
        print(f"Results dir not found: {results_dir}", file=sys.stderr)
        return 1

    sweep_rows = load_sweep(results_dir, alpha_col=args.alpha_col, metric=args.metric)
    summary_rows = load_summary(results_dir)

    if not sweep_rows:
        print("No sweep.csv or no usable accuracy column. Generate what we can from summary only.", file=sys.stderr)

    sweep_rows = filter_rows(
        sweep_rows,
        args.regimes,
        args.architectures,
        alpha_col=args.alpha_col,
        acc_key="_acc",
    )
    summary_rows = [r for r in summary_rows if r.get("regime") in {s.strip() for s in args.regimes.split(",")}]
    if args.architectures:
        arch_set = {s.strip() for s in args.architectures.split(",")}
        summary_rows = [r for r in summary_rows if r.get("architecture") in arch_set]

    regimes_list = [s.strip() for s in args.regimes.split(",") if s.strip()]
    generated_any = False

    if sweep_rows:
        aggregated = aggregate_sweep(sweep_rows, alpha_col=args.alpha_col, acc_key="_acc")
        print_table(aggregated, regimes_order=regimes_list)

        for regime in regimes_list:
            if not any(k[1] == regime for k in aggregated):
                continue
            plot_robustness_curves(
                aggregated,
                regime,
                os.path.join(out_dir, f"robustness_curves_{regime}.png"),
                args.dataset,
                log_x=args.log_x,
            )
            plot_degradation_curves(
                aggregated,
                regime,
                os.path.join(out_dir, f"degradation_curves_{regime}.png"),
                args.dataset,
                log_x=args.log_x,
            )
            generated_any = True

        plot_auc_bars(aggregated, os.path.join(out_dir, "auc_by_arch.png"), args.dataset, regimes_order=regimes_list)
        plot_initial_drop_bars(
            aggregated,
            os.path.join(out_dir, "initial_drop.png"),
            args.dataset,
            regimes_order=regimes_list,
        )
        generated_any = True

    if summary_rows and any("drop_at_01" in r for r in summary_rows):
        arch_filter = None
        if args.architectures:
            arch_filter = {s.strip() for s in args.architectures.split(",")}
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
        print("No plots generated. Ensure sweep.csv (and optionally summary.csv) exist and contain data.", file=sys.stderr)
        return 1

    print(f"Plots saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
