"""
Phase 6–7: Train models, run robustness sweep, log results, save to CSV/JSON.
Checkpoints under config.CHECKPOINT_DIR; results under config.RESULTS_DIR.
Run all architectures (default) or only selected ones: python run_experiments.py [mlp] [cnn] [resnet18] [vit]
"""

import argparse
import csv
import json
import os

import config
from data import get_mnist_loaders
from evaluate import accuracy_drop_at_01, load_and_sweep
from models import ARCH_NAMES
from train import train_one


def checkpoint_path(arch, regime, seed):
    return os.path.join(config.CHECKPOINT_DIR, f"{arch}_{regime}_seed{seed}.pt")


# ViT expects 224×224; use loaders with resize so we don't resize in model forward.
VIT_INPUT_SIZE = 224


def run_all(architectures=None):
    """
    Run training + robustness sweep for given architectures.
    architectures: list of arch names (e.g. ["mlp", "cnn", "resnet18"]). If None, run all ARCH_NAMES.
    """
    archs = architectures if architectures is not None else ARCH_NAMES
    for a in archs:
        if a not in ARCH_NAMES:
            raise ValueError(f"Unknown architecture: {a}. Choose from {ARCH_NAMES}")

    device = config.DEVICE
    print(f"Using device: {device}")
    print(f"Architectures: {archs}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    train_loader, test_loader = get_mnist_loaders()
    train_loader_vit, test_loader_vit = (get_mnist_loaders(resize=VIT_INPUT_SIZE) if "vit" in archs else (None, None))

    # Train (skip if checkpoint already exists)
    for seed in config.SEEDS:
        for arch in archs:
            for regime, noisy in [("clean", False), ("noisy", True)]:
                path = checkpoint_path(arch, regime, seed)
                if os.path.isfile(path):
                    print(f"skip training {arch} {regime} seed={seed}")
                    continue
                tr_loader = train_loader_vit if arch == "vit" and train_loader_vit is not None else train_loader
                print(f"\n--- train {arch} {regime} seed={seed} -> {path}")
                train_one(arch, tr_loader, noisy=noisy, seed=seed, device=device, save_path=path)

    # Robustness sweep for each checkpoint; collect results
    print("\n" + "=" * 60 + "\nRobustness evaluation\n")
    summary_rows = []  # Architecture, Regime, Seed, Acc(0), Acc(0.1), Drop
    sweep_rows = []    # Architecture, Regime, Seed, alpha_test, Acc, Loss
    drops_by_arch_regime = {}

    for arch in ARCH_NAMES:
        for regime in ["clean", "noisy"]:
            drops_by_arch_regime[(arch, regime)] = []
            for seed in config.SEEDS:
                path = checkpoint_path(arch, regime, seed)
                if not os.path.isfile(path):
                    print(f"skip {arch} {regime} seed={seed} (no checkpoint)")
                    continue
                te_loader = test_loader_vit if arch == "vit" and test_loader_vit is not None else test_loader
                _, sweep = load_and_sweep(arch, path, te_loader, device)
                drop = accuracy_drop_at_01(sweep)
                acc0 = next(r["acc"] for r in sweep if r["alpha"] == 0.0)
                acc01 = next(r["acc"] for r in sweep if r["alpha"] == 0.1)
                summary_rows.append({
                    "architecture": arch,
                    "regime": regime,
                    "seed": seed,
                    "acc_0": acc0,
                    "acc_01": acc01,
                    "drop_at_01": drop,
                })
                for r in sweep:
                    sweep_rows.append({
                        "architecture": arch,
                        "regime": regime,
                        "seed": seed,
                        "alpha_test": r["alpha"],
                        "acc": r["acc"],
                        "loss": r["loss"],
                    })
                drops_by_arch_regime[(arch, regime)].append(drop)
                print(f"{arch} {regime} seed={seed}: acc@0={acc0:.4f} acc@0.1={acc01:.4f} drop={drop:.4f}")

    # Print summary table: Architecture, Regime, Seed, Acc(0), Acc(0.1), Drop
    print("\n" + "-" * 70 + "\nSummary table: Architecture, Regime, Seed, Acc(0), Acc(0.1), Drop at 0.1\n")
    fmt = "{:12} {:8} {:4} {:8} {:8} {:10}"
    print(fmt.format("Architecture", "Regime", "Seed", "Acc(0)", "Acc(0.1)", "Drop"))
    print("-" * 54)
    for row in summary_rows:
        print(fmt.format(
            row["architecture"], row["regime"], row["seed"],
            f"{row['acc_0']:.4f}", f"{row['acc_01']:.4f}", f"{row['drop_at_01']:.4f}",
        ))

    # Mean drop at α=0.1 per (arch, regime)
    print("\n" + "-" * 40 + "\nMean drop at α_test=0.1 (over seeds)")
    for arch in ARCH_NAMES:
        for regime in ["clean", "noisy"]:
            vals = drops_by_arch_regime.get((arch, regime), [])
            mean_drop = sum(vals) / len(vals) if vals else float("nan")
            print(f"  {arch} {regime}: {mean_drop:.4f}")

    # Phase 7: save results for plotting (CSV and JSON)
    summary_path = os.path.join(config.RESULTS_DIR, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["architecture", "regime", "seed", "acc_0", "acc_01", "drop_at_01"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote {summary_path}")

    sweep_path = os.path.join(config.RESULTS_DIR, "sweep.csv")
    with open(sweep_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["architecture", "regime", "seed", "alpha_test", "acc", "loss"])
        w.writeheader()
        w.writerows(sweep_rows)
    print(f"Wrote {sweep_path}")

    results_json = os.path.join(config.RESULTS_DIR, "results.json")
    with open(results_json, "w") as f:
        json.dump({"summary": summary_rows, "sweep": sweep_rows}, f, indent=2)
    print(f"Wrote {results_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate weight-noise experiments. Omit archs to run all.")
    parser.add_argument(
        "architectures",
        nargs="*",
        choices=ARCH_NAMES,
        help=f"Architectures to run (default: all). Choices: {ARCH_NAMES}",
    )
    args = parser.parse_args()
    run_all(architectures=args.architectures if args.architectures else None)
