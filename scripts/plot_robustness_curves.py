"""
Produce the two main figures for the thesis:
1. Robustness decay curves: accuracy vs alpha_test, one line per (architecture, regime); clean vs noisy by linestyle.
2. Drop@0.1 bar plot: one bar per architecture, clean vs noisy side-by-side.

Reads results from RESULTS_DIR/dataset/sweep.csv and summary.csv (or results.json).
Writes figures to RESULTS_DIR/dataset/.
"""

import argparse
import csv
import os
import sys

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def load_sweep(results_dir):
    path = os.path.join(results_dir, "sweep.csv")
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["alpha_test"] = float(r["alpha_test"])
            r["acc"] = float(r["acc"])
            r["alpha_train"] = float(r.get("alpha_train", 0))
            rows.append(r)
    return rows


def load_summary(results_dir):
    path = os.path.join(results_dir, "summary.csv")
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["acc_0"] = float(r["acc_0"])
            r["acc_01"] = float(r["acc_01"])
            r["drop_at_01"] = float(r["drop_at_01"])
            r["alpha_train"] = float(r.get("alpha_train", 0))
            rows.append(r)
    return rows


def aggregate_sweep_by_arch_regime(sweep_rows):
    """For each (arch, regime, alpha_train), for each alpha_test, mean acc over seeds."""
    from collections import defaultdict
    # Key: (arch, regime, alpha_train), value: list of (alpha_test, acc) per seed then we average
    by_key = defaultdict(lambda: defaultdict(list))
    for r in sweep_rows:
        key = (r["architecture"], r["regime"], r["alpha_train"])
        by_key[key][r["alpha_test"]].append(r["acc"])
    out = {}
    for key, alphas in by_key.items():
        out[key] = {a: sum(vals) / len(vals) for a, vals in alphas.items()}
    return out


def aggregate_drop_by_arch_regime(summary_rows):
    """Mean drop_at_01 per (arch, regime, alpha_train)."""
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in summary_rows:
        key = (r["architecture"], r["regime"], r["alpha_train"])
        by_key[key].append(r["drop_at_01"])
    return {k: sum(v) / len(v) for k, v in by_key.items()}


def plot_decay_curves(aggregated, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Prefer clean=0.0, noisy=0.05 for thesis; one line per (arch, regime)
    preferred_alpha = {"clean": 0.0, "noisy": 0.05}
    keys_by_arch_regime = {}
    for (arch, regime, alpha_train) in aggregated.keys():
        k = (arch, regime)
        pref = preferred_alpha.get(regime, 0.05)
        if k not in keys_by_arch_regime or alpha_train == pref:
            keys_by_arch_regime[k] = (arch, regime, alpha_train)
    for (arch, regime) in sorted(keys_by_arch_regime.keys()):
        _, _, alpha_train = keys_by_arch_regime[(arch, regime)]
        curve = aggregated[(arch, regime, alpha_train)]
        alphas = sorted(curve.keys())
        accs = [curve[a] for a in alphas]
        linestyle = "-" if regime == "noisy" else "--"
        label = f"{arch} ({regime})"
        plt.plot(alphas, accs, linestyle=linestyle, label=label)

    plt.xlabel(r"$\alpha$ (test perturbation)")
    plt.ylabel("Accuracy")
    plt.title("Robustness decay: accuracy vs weight-noise strength")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(results_dir, "robustness_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def plot_drop_bars(drop_by_key, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Collect (arch, regime) and drop; then group by arch, clean vs noisy
    arch_order = []
    seen = set()
    for (arch, regime, _) in sorted(drop_by_key.keys()):
        if arch not in seen:
            arch_order.append(arch)
            seen.add(arch)
    # Prefer alpha_train 0.0/0.05
    clean_drops = []
    noisy_drops = []
    for arch in arch_order:
        c = drop_by_key.get((arch, "clean", 0.0)) or drop_by_key.get((arch, "clean", 0.05))
        n = drop_by_key.get((arch, "noisy", 0.05)) or drop_by_key.get((arch, "noisy", 0.0))
        clean_drops.append(c if c is not None else float("nan"))
        noisy_drops.append(n if n is not None else float("nan"))

    x = np.arange(len(arch_order))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, clean_drops, width, label="clean", color="C0", alpha=0.8)
    ax.bar(x + width / 2, noisy_drops, width, label="noisy", color="C1", alpha=0.8)
    ax.set_ylabel("Accuracy drop at α = 0.1")
    ax.set_title("Robustness: drop at α_test = 0.1")
    ax.set_xticks(x)
    ax.set_xticklabels(arch_order, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(results_dir, "drop_at_01_bars.png")
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot robustness curves and drop@0.1 bars from results.")
    parser.add_argument("--dataset", default="cifar10", help="Dataset subdir under RESULTS_DIR")
    parser.add_argument("--results-dir", default=None, help="Defaults to config.RESULTS_DIR")
    args = parser.parse_args()
    results_dir = args.results_dir or os.path.join(config.RESULTS_DIR, args.dataset)
    if not os.path.isdir(results_dir):
        print(f"Results dir not found: {results_dir}")
        return 1

    sweep_rows = load_sweep(results_dir)
    summary_rows = load_summary(results_dir)
    if not sweep_rows:
        print("No sweep.csv found; run experiments first.")
        return 1

    aggregated = aggregate_sweep_by_arch_regime(sweep_rows)
    plot_decay_curves(aggregated, results_dir)

    if summary_rows:
        drop_by_key = aggregate_drop_by_arch_regime(summary_rows)
        plot_drop_bars(drop_by_key, results_dir)
    else:
        print("No summary.csv; skipping drop bar plot.")

    return 0


if __name__ == "__main__":
    exit(main())
