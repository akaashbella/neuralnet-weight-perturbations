"""
Train models, run robustness sweep, log results, save to CSV/JSON.
Checkpoints under config.CHECKPOINT_DIR; results under config.RESULTS_DIR.
Run all architectures (default) or selected: python run_experiments.py [cnn] [mlp] [plainnet20] [resnet20] [mobilenet_v2] [vit_lite]
"""

import argparse
import csv
import json
import os

import config
from data import get_loaders
from evaluate import accuracy_drop_at_01, load_and_sweep
from models import ARCH_NAMES, ARCH_INPUT_RESIZE, ALL_ARCH_NAMES
from train import train_one


def checkpoint_path(arch, regime, seed, alpha_train=None, base_dir=None):
    """Path includes alpha_train so different noise levels don't overwrite each other. base_dir is dataset-specific."""
    if alpha_train is None:
        alpha_train = config.ALPHA_TRAIN
    base_dir = base_dir or config.CHECKPOINT_DIR
    return os.path.join(base_dir, f"{arch}_{regime}_seed{seed}_a{alpha_train:.2f}.pt")


def run_all(dataset="cifar10", architectures=None):
    """
    Run training + robustness sweep for given dataset and architectures.
    Regimes and alpha_train come from config.ALPHA_TRAIN_LIST (no hardcoding).
    Per-arch resize from ARCH_INPUT_RESIZE (default 32).
    """
    archs = architectures if architectures is not None else ARCH_NAMES
    for a in archs:
        if a not in ALL_ARCH_NAMES:
            raise ValueError(f"Unknown architecture: {a}. Choose from {ALL_ARCH_NAMES}")

    alpha_train_list = config.ALPHA_TRAIN_LIST
    if len(alpha_train_list) < 2:
        raise ValueError("ALPHA_TRAIN_LIST must have at least 2 values (clean, noisy).")
    # First = clean (no noise), second = noisy (noise-regularized)
    clean_alpha, noisy_alpha = alpha_train_list[0], alpha_train_list[1]
    train_regimes = [
        ("clean", False, clean_alpha),
        ("noisy", True, noisy_alpha),
    ]

    device = config.DEVICE
    checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, dataset)
    results_dir = os.path.join(config.RESULTS_DIR, dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Per-arch resize (centralized); default 32 for CIFAR
    def resize_for(arch):
        return ARCH_INPUT_RESIZE.get(arch, 32)

    # Loaders per unique resize to avoid redundant builds
    unique_resizes = {resize_for(a) for a in archs}
    loaders_by_resize = {}
    for r in unique_resizes:
        loaders_by_resize[r] = get_loaders(dataset, resize=r if r != 32 else None)

    print(f"Using device: {device}")
    print(f"Dataset: {dataset}")
    print(f"Alpha (train) from ALPHA_TRAIN_LIST: clean={clean_alpha}, noisy={noisy_alpha}")
    print(f"Architectures: {archs}")

    # Train: for each (arch, seed, regime) with regime-specific alpha_train
    for seed in config.SEEDS:
        for arch in archs:
            train_loader, _ = loaders_by_resize[resize_for(arch)]
            for regime, noisy, alpha_train in train_regimes:
                path = checkpoint_path(arch, regime, seed, alpha_train, base_dir=checkpoint_dir)
                if os.path.isfile(path):
                    print(f"skip training {arch} {regime} seed={seed}")
                    continue
                print(f"\n--- train {arch} {regime} seed={seed} a_train={alpha_train} -> {path}")
                train_one(
                    arch, train_loader, noisy=noisy, seed=seed, device=device,
                    save_path=path, alpha_train=alpha_train,
                )

    # Robustness sweep: same resize per arch as training
    print("\n" + "=" * 60 + "\nRobustness evaluation\n")
    summary_rows = []
    sweep_rows = []
    drops_by_arch_regime = {}

    for arch in archs:
        _, test_loader = loaders_by_resize[resize_for(arch)]
        for regime, _, alpha_train in train_regimes:
            drops_by_arch_regime[(arch, regime)] = []
            for seed in config.SEEDS:
                path = checkpoint_path(arch, regime, seed, alpha_train, base_dir=checkpoint_dir)
                if not os.path.isfile(path):
                    print(f"skip {arch} {regime} seed={seed} (no checkpoint)")
                    continue
                _, sweep = load_and_sweep(arch, path, test_loader, device)
                drop = accuracy_drop_at_01(sweep)
                acc0 = next(r["acc"] for r in sweep if r["alpha"] == 0.0)
                acc01 = next(r["acc"] for r in sweep if r["alpha"] == 0.1)
                summary_rows.append({
                    "architecture": arch,
                    "regime": regime,
                    "seed": seed,
                    "dataset": dataset,
                    "alpha_train": float(alpha_train),
                    "acc_0": acc0,
                    "acc_01": acc01,
                    "drop_at_01": drop,
                })
                for rec in sweep:
                    sweep_rows.append({
                        "architecture": arch,
                        "regime": regime,
                        "seed": seed,
                        "dataset": dataset,
                        "alpha_train": float(alpha_train),
                        "alpha_test": rec["alpha"],
                        "acc": rec["acc"],
                        "loss": rec["loss"],
                    })
                drops_by_arch_regime[(arch, regime)].append(drop)
                print(f"{arch} {regime} seed={seed}: acc@0={acc0:.4f} acc@0.1={acc01:.4f} drop={drop:.4f}")

    # Print summary table: Architecture, Regime, Seed, Dataset, Alpha_train, Acc(0), Acc(0.1), Drop
    print("\n" + "-" * 70 + "\nSummary table: Architecture, Regime, Seed, Dataset, Alpha_train, Acc(0), Acc(0.1), Drop at 0.1\n")
    fmt = "{:12} {:8} {:4} {:8} {:10} {:8} {:8} {:10}"
    print(fmt.format("Architecture", "Regime", "Seed", "Dataset", "Alpha_tr", "Acc(0)", "Acc(0.1)", "Drop"))
    print("-" * 72)
    for row in summary_rows:
        print(fmt.format(
            row["architecture"], row["regime"], row["seed"], row["dataset"],
            f"{row['alpha_train']:.2f}",
            f"{row['acc_0']:.4f}", f"{row['acc_01']:.4f}", f"{row['drop_at_01']:.4f}",
        ))

    # Mean drop at α=0.1 per (arch, regime) — only for architectures we ran
    print("\n" + "-" * 40 + "\nMean drop at α_test=0.1 (over seeds)")
    for arch in archs:
        for regime in ["clean", "noisy"]:
            vals = drops_by_arch_regime.get((arch, regime), [])
            mean_drop = sum(vals) / len(vals) if vals else float("nan")
            print(f"  {arch} {regime}: {mean_drop:.4f}")

    # Phase 7: merge with existing results (by arch, regime, seed, alpha_train) and save to dataset-specific dir
    summary_fieldnames = ["architecture", "regime", "seed", "dataset", "alpha_train", "acc_0", "acc_01", "drop_at_01"]
    sweep_fieldnames = ["architecture", "regime", "seed", "dataset", "alpha_train", "alpha_test", "acc", "loss"]

    def _norm_alpha(x):
        """Normalize alpha_train to float for consistent merge keys (CSV gives strings)."""
        if x is None or x == "":
            return None
        try:
            return float(x)
        except (TypeError, ValueError):
            return x

    def _key_summary(row):
        return (row["architecture"], row["regime"], row["seed"], _norm_alpha(row.get("alpha_train")))

    def _load_summary_csv(path, dataset_for_legacy=None):
        rows = []
        if not os.path.isfile(path):
            return rows
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if "alpha_train" not in row or row.get("alpha_train") == "":
                    row["alpha_train"] = None  # legacy file
                else:
                    row["alpha_train"] = _norm_alpha(row["alpha_train"])
                if "dataset" not in row and dataset_for_legacy is not None:
                    row["dataset"] = dataset_for_legacy
                rows.append(row)
        return rows

    def _load_sweep_csv(path, dataset_for_legacy=None):
        rows = []
        if not os.path.isfile(path):
            return rows
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if "alpha_train" not in row or row.get("alpha_train") == "":
                    row["alpha_train"] = None
                else:
                    row["alpha_train"] = _norm_alpha(row["alpha_train"])
                if "dataset" not in row and dataset_for_legacy is not None:
                    row["dataset"] = dataset_for_legacy
                rows.append(row)
        return rows

    current_run_keys = {_key_summary(r) for r in summary_rows}
    existing_summary = _load_summary_csv(os.path.join(results_dir, "summary.csv"), dataset_for_legacy=dataset)
    merged_summary = [r for r in existing_summary if _key_summary(r) not in current_run_keys] + summary_rows
    for r in merged_summary:
        if r.get("alpha_train") is None:
            r["alpha_train"] = 0.0  # legacy rows: serialize as 0.0 in CSV/JSON

    current_run_sweep_keys = {(r["architecture"], r["regime"], r["seed"], r["alpha_train"]) for r in sweep_rows}
    existing_sweep = _load_sweep_csv(os.path.join(results_dir, "sweep.csv"), dataset_for_legacy=dataset)
    merged_sweep = [r for r in existing_sweep if (r["architecture"], r["regime"], r["seed"], r.get("alpha_train")) not in current_run_sweep_keys] + sweep_rows
    for r in merged_sweep:
        if r.get("alpha_train") is None:
            r["alpha_train"] = 0.0

    summary_path = os.path.join(results_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fieldnames)
        w.writeheader()
        w.writerows(merged_summary)
    print(f"\nWrote {summary_path} (merged, {len(merged_summary)} rows)")

    sweep_path = os.path.join(results_dir, "sweep.csv")
    with open(sweep_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sweep_fieldnames)
        w.writeheader()
        w.writerows(merged_sweep)
    print(f"Wrote {sweep_path} (merged, {len(merged_sweep)} rows)")

    results_json = os.path.join(results_dir, "results.json")
    with open(results_json, "w") as f:
        json.dump({"summary": merged_summary, "sweep": merged_sweep}, f, indent=2)
    print(f"Wrote {results_json}")


if __name__ == "__main__":
    from data import DATASET_CHOICES
    parser = argparse.ArgumentParser(
        description="Train and evaluate weight-noise experiments. Choose dataset and optionally architectures.",
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=DATASET_CHOICES,
        help="Dataset to train and evaluate on (default: cifar10).",
    )
    parser.add_argument(
        "architectures",
        nargs="*",
        choices=ALL_ARCH_NAMES,
        help=f"Architectures to run (default: core set). Choices: {ALL_ARCH_NAMES}",
    )
    args = parser.parse_args()
    run_all(dataset=args.dataset, architectures=args.architectures if args.architectures else None)
