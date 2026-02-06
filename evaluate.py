"""
Robustness evaluation: test accuracy/loss under post-training weight perturbation.
Sweep over α_test; optionally compute accuracy drop at α_test = 0.1.
"""

import torch
import torch.nn as nn
from contextlib import nullcontext

import config
from models import get_model
from noise import weight_noise


def evaluate_one(model, test_loader, device, alpha_test=0.0):
    """
    One test pass: optional weight noise (α_test), no grad.
    Returns (mean_loss, accuracy) where accuracy is correct/total.
    Skips noise when alpha_test == 0 to avoid unnecessary tensor ops.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    n = 0
    ctx = weight_noise(model, alpha_test) if alpha_test > 0 else nullcontext()
    with torch.no_grad(), ctx:
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            n += y.size(0)
    mean_loss = total_loss / n if n else 0.0
    accuracy = correct / n if n else 0.0
    return mean_loss, accuracy


def run_robustness_sweep(model, test_loader, device, alpha_list=None, num_samples=None):
    """
    Robustness loop: for each α_test, evaluate (averaging over num_samples noise draws if α > 0).
    Reduces variance and avoids spurious "negative drop"; single draw for α=0.
    Returns list of dicts: [{"alpha": a, "loss": l, "acc": acc}, ...].
    """
    if alpha_list is None:
        alpha_list = config.ALPHA_TEST_LIST
    num_samples = num_samples if num_samples is not None else config.ROBUSTNESS_NUM_SAMPLES
    results = []
    for alpha in alpha_list:
        if alpha == 0:
            loss, acc = evaluate_one(model, test_loader, device, alpha_test=0.0)
            results.append({"alpha": alpha, "loss": loss, "acc": acc})
        else:
            losses, accs = [], []
            for _ in range(num_samples):
                l, a = evaluate_one(model, test_loader, device, alpha_test=alpha)
                losses.append(l)
                accs.append(a)
            results.append({
                "alpha": alpha,
                "loss": sum(losses) / len(losses),
                "acc": sum(accs) / len(accs),
            })
    return results


def accuracy_drop_at_01(sweep_results):
    """
    Summary metric: acc(α=0) − acc(α=0.1).
    sweep_results is list of {"alpha", "loss", "acc"} from run_robustness_sweep.
    """
    acc_at_0 = next((r["acc"] for r in sweep_results if r["alpha"] == 0.0), None)
    acc_at_01 = next((r["acc"] for r in sweep_results if r["alpha"] == 0.1), None)
    if acc_at_0 is None or acc_at_01 is None:
        return None
    return acc_at_0 - acc_at_01


def load_and_sweep(arch_name, checkpoint_path, test_loader, device, alpha_list=None):
    """
    Load model from checkpoint, run robustness sweep.
    Returns (model, sweep_results) for logging/saving.
    """
    model = get_model(arch_name).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    results = run_robustness_sweep(model, test_loader, device, alpha_list=alpha_list)
    return model, results


def assert_weights_unchanged_after_sweep(model, test_loader, device, alpha_list=None):
    """
    Sanity check: run_robustness_sweep uses a context manager that adds/removes noise,
    so after the sweep the model's weights must equal the original. Catches "noise not removed" regressions.
    """
    state_before = {n: p.data.clone() for n, p in model.named_parameters()}
    run_robustness_sweep(model, test_loader, device, alpha_list=alpha_list or [0.0, 0.1], num_samples=2)
    for n, p in model.named_parameters():
        assert torch.allclose(p.data, state_before[n]), f"Param {n} changed after sweep (noise not restored)"
    return True


if __name__ == "__main__":
    from data import get_loaders
    device = config.DEVICE
    _, test_loader = get_loaders("cifar10")

    # Sanity: weights unchanged after sweep (noise is add/remove, not permanent)
    model = get_model("mlp").to(device)
    assert_weights_unchanged_after_sweep(model, test_loader, device)
    print("evaluate.py: weights unchanged after sweep (OK)")

    # Optional: load MLP checkpoints (if present), run short sweep
    for name, path in [("mlp_clean", "checkpoint_mlp_clean.pt"), ("mlp_noisy", "checkpoint_mlp_noisy.pt")]:
        try:
            _, results = load_and_sweep("mlp", path, test_loader, device)
            drop = accuracy_drop_at_01(results)
            print(f"{name}: acc@0={results[0]['acc']:.4f}, acc@0.1={next(r['acc'] for r in results if r['alpha']==0.1):.4f}, drop={drop:.4f}")
        except FileNotFoundError:
            print(f"{name}: skip (no {path})")
