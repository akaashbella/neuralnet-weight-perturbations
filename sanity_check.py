"""
Phase 8 sanity check: train MLP 1 epoch (clean + noisy), evaluate at α_test ∈ {0, 0.1}.
Spot-check that noisy-trained has smaller accuracy drop at α=0.1 than clean-trained.
"""

import os
import tempfile

import config
from data import get_loaders
from evaluate import accuracy_drop_at_01, load_and_sweep
from train import train_one


def main():
    device = config.DEVICE
    train_loader, test_loader = get_loaders("cifar10")
    alpha_sanity = [0.0, 0.1]  # only 0 and 0.1 for quick check
    arch = "mlp_small"

    with tempfile.TemporaryDirectory() as tmp:
        clean_path = os.path.join(tmp, "mlp_clean.pt")
        noisy_path = os.path.join(tmp, "mlp_noisy.pt")

        print(f"Training {arch} clean (1 epoch)...")
        train_one(arch, train_loader, noisy=False, seed=0, device=device, save_path=clean_path, epochs=1)
        print(f"Training {arch} noisy (1 epoch)...")
        train_one(arch, train_loader, noisy=True, seed=0, device=device, save_path=noisy_path, epochs=1)

        print("\nRobustness at alpha_test in {0, 0.1}:")
        _, sweep_clean = load_and_sweep(arch, clean_path, test_loader, device, alpha_list=alpha_sanity)
        _, sweep_noisy = load_and_sweep(arch, noisy_path, test_loader, device, alpha_list=alpha_sanity)

        drop_clean = accuracy_drop_at_01(sweep_clean)
        drop_noisy = accuracy_drop_at_01(sweep_noisy)
        print(f"  clean drop at 0.1: {drop_clean:.4f}")
        print(f"  noisy drop at 0.1: {drop_noisy:.4f}")

        # Expect noisy-trained to be more robust (smaller drop). Allow small tolerance for 1-epoch variance.
        tol = 0.02
        if drop_noisy <= drop_clean + tol:
            print("\nSanity OK: noisy-trained has smaller or similar drop (noisy more robust).")
        else:
            print(f"\nSanity: noisy drop > clean drop by {drop_noisy - drop_clean:.4f}. With 1 epoch this can happen by chance; re-run or use more epochs.")


if __name__ == "__main__":
    main()
