"""
OPTIONAL, EXPENSIVE: 2D loss landscape slice around a trained checkpoint.
Run from repo root: python scripts/viz_loss_landscape_slice.py --dataset cifar10 --arch cnn_large --regime clean --seed 0 --grid 31 --span 0.5 --batch_limit 10
Uses existing model factory and data loaders; saves contour and surface under results/figures/<dataset>/loss_landscape/<arch>_<regime>_seed<seed>/.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data import get_loaders
from models import get_model, ARCH_INPUT_RESIZE
from noise import get_weight_only_params


def find_checkpoint(checkpoint_dir, arch, regime, seed):
    """Return path to checkpoint matching {arch}_{regime}_seed{seed}_*.pt, or None."""
    pattern = os.path.join(checkpoint_dir, f"{arch}_{regime}_seed{seed}_*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return candidates[0]


def make_random_directions(weight_params, device, seed=42):
    """
    Two random direction vectors in weight space. Same shape as each param.
    Layer-wise normalization: each direction scaled so ||dir|| = ||param|| (Li et al style).
    Returns (orig_weights, d1_list, d2_list).
    """
    torch.manual_seed(seed)
    orig = [p.data.clone() for p in weight_params]
    d1_list = []
    d2_list = []
    for p in weight_params:
        d1 = torch.randn_like(p.data, device=p.device, dtype=p.dtype)
        d2 = torch.randn_like(p.data, device=p.device, dtype=p.dtype)
        # Normalize and scale to param norm (per-parameter / layer-wise)
        norm_p = p.data.norm().item()
        eps = 1e-8
        d1_norm = d1.norm().item() + eps
        d2_norm = d2.norm().item() + eps
        d1 = d1 / d1_norm * norm_p
        d2 = d2 / d2_norm * norm_p
        d1_list.append(d1)
        d2_list.append(d2)
    return orig, d1_list, d2_list


def set_weights(weight_params, orig_list, a, b, d1_list, d2_list):
    """Set weight_params to orig + a*d1 + b*d2."""
    for i, p in enumerate(weight_params):
        p.data.copy_(orig_list[i] + a * d1_list[i] + b * d2_list[i])


def restore_weights(weight_params, orig_list):
    """Restore original weights."""
    for i, p in enumerate(weight_params):
        p.data.copy_(orig_list[i])


def evaluate_loss(model, loader, device, batch_limit):
    """Run up to batch_limit batches; return mean loss (sum loss / total samples)."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= batch_limit:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            n += y.size(0)
    return total_loss / n if n else float("nan")


def main():
    parser = argparse.ArgumentParser(
        description="Compute 2D loss landscape slice (optional, expensive). Requires checkpoint and model.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset (e.g. cifar10).")
    parser.add_argument("--arch", required=True, help="Architecture name (e.g. cnn_large).")
    parser.add_argument("--regime", required=True, help="Regime (clean or noisy).")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the checkpoint.")
    parser.add_argument("--grid", type=int, default=31, help="Grid size (grid x grid points).")
    parser.add_argument("--span", type=float, default=0.5, help="Range for each axis: [-span, span].")
    parser.add_argument("--batch_limit", type=int, default=10, help="Max batches per grid point for loss eval.")
    parser.add_argument("--dir_seed", type=int, default=42, help="Seed for random direction vectors.")
    args = parser.parse_args()

    checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, args.dataset)
    if not os.path.isdir(checkpoint_dir):
        print(f"Checkpoint dir not found: {checkpoint_dir}", file=sys.stderr)
        return 1

    ckpt_path = find_checkpoint(checkpoint_dir, args.arch, args.regime, args.seed)
    if ckpt_path is None:
        print(f"No checkpoint found for {args.arch} {args.regime} seed{args.seed} in {checkpoint_dir}", file=sys.stderr)
        return 1

    try:
        model = get_model(args.arch).to(config.DEVICE)
    except Exception as e:
        print(f"Failed to create model for architecture '{args.arch}': {e}", file=sys.stderr)
        return 1

    try:
        state = torch.load(ckpt_path, map_location=config.DEVICE)
        model.load_state_dict(state)
    except Exception as e:
        print(f"Failed to load checkpoint {ckpt_path}: {e}", file=sys.stderr)
        return 1

    resize = ARCH_INPUT_RESIZE.get(args.arch, 32)
    resize = resize if resize != 32 else None
    try:
        _, test_loader = get_loaders(args.dataset, resize=resize)
    except Exception as e:
        print(f"Failed to get data loaders for {args.dataset}: {e}", file=sys.stderr)
        return 1

    weight_params = get_weight_only_params(model)
    if not weight_params:
        print("No weight parameters found (get_weight_only_params returned empty).", file=sys.stderr)
        return 1

    orig_list, d1_list, d2_list = make_random_directions(weight_params, config.DEVICE, seed=args.dir_seed)

    # Grid
    grid = args.grid
    span = args.span
    avals = np.linspace(-span, span, grid)
    bvals = np.linspace(-span, span, grid)
    loss_grid = np.zeros((grid, grid))
    device = config.DEVICE

    print(f"Evaluating loss on {grid}x{grid} grid (batch_limit={args.batch_limit} per point)...")
    for i, a in enumerate(avals):
        for j, b in enumerate(bvals):
            set_weights(weight_params, orig_list, float(a), float(b), d1_list, d2_list)
            loss_grid[i, j] = evaluate_loss(model, test_loader, device, args.batch_limit)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  row {i + 1}/{grid} done")
    restore_weights(weight_params, orig_list)

    # Save outputs
    out_dir = os.path.join(
        config.RESULTS_DIR,
        "figures",
        args.dataset,
        "loss_landscape",
        f"{args.arch}_{args.regime}_seed{args.seed}",
    )
    os.makedirs(out_dir, exist_ok=True)

    A, B = np.meshgrid(avals, bvals, indexing="ij")
    L = loss_grid

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Contour
    fig, ax = plt.subplots()
    cs = ax.contour(A, B, L, levels=15)
    ax.clabel(cs, inline=True, fontsize=8)
    ax.set_xlabel("direction 1")
    ax.set_ylabel("direction 2")
    ax.set_title(f"Loss contour: {args.dataset} {args.arch} {args.regime} seed{args.seed}")
    fig.tight_layout()
    contour_path = os.path.join(out_dir, "loss_contour.png")
    fig.savefig(contour_path, dpi=150)
    plt.close()
    print(f"Saved {contour_path}")

    # Surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(A, B, L, cmap="viridis", alpha=0.9)
    ax.set_xlabel("direction 1")
    ax.set_ylabel("direction 2")
    ax.set_zlabel("Loss")
    ax.set_title(f"Loss surface: {args.dataset} {args.arch} {args.regime} seed{args.seed}")
    fig.tight_layout()
    surface_path = os.path.join(out_dir, "loss_surface.png")
    fig.savefig(surface_path, dpi=150)
    plt.close()
    print(f"Saved {surface_path}")

    print(f"All outputs under {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
