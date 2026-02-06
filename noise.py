"""
Weight-only Gaussian noise for training-time and evaluation-time perturbation.
Used for both: (1) noisy training per batch, (2) robustness sweep at evaluation.
Excludes biases and BatchNorm/LayerNorm parameters.
"""

import torch
import torch.nn as nn


# Small constant so we never divide by zero when scaling noise
_STD_EPS = 1e-8


def get_weight_only_params(model):
    """
    Return list of weight tensors that should receive noise.
    Guarantees: biases excluded; BatchNorm/LayerNorm params excluded; all other
    weights included (Linear, Conv2d, attention projections, etc.).
    Uses the owning module (get_submodule) so nesting and naming are robust.
    """
    weight_params = []
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)
    for name, param in model.named_parameters():
        if name.split(".")[-1] != "weight":
            continue
        module_path = name.rsplit(".", 1)[0]
        owner = model.get_submodule(module_path) if module_path else model
        if isinstance(owner, norm_types):
            continue
        weight_params.append(param)
    return weight_params


def apply_weight_noise(model, alpha):
    """
    Add Gaussian noise to weight-only parameters in-place.
    Used in training (per batch) and in evaluation (per α_test).
    For each weight W: std_W = std(W), then ε ~ N(0, (α * std_W)²), W += ε.
    Returns a list of (param, noise) so caller can remove the same noise later.
    Used for both training (per batch) and evaluation (per α_test).
    """
    weight_params = get_weight_only_params(model)
    noise_list = []
    for w in weight_params:
        std_w = w.data.std().item()
        scale = max(std_w, _STD_EPS) * alpha
        eps = torch.randn_like(w, device=w.device, dtype=w.dtype) * scale
        w.data.add_(eps)
        noise_list.append((w, eps))
    return noise_list


def remove_weight_noise(noise_list):
    """
    Subtract the same noise that was added in apply_weight_noise.
    Call this after backward() and before optimizer.step() in noisy training.
    """
    for w, eps in noise_list:
        w.data.sub_(eps)


class weight_noise:
    """
    Context manager: add noise on enter, remove on exit.
    Useful for evaluation-time perturbation (add noise, run test pass, exit restores clean weights).
    """

    def __init__(self, model, alpha):
        self.model = model
        self.alpha = alpha
        self._noise_list = None

    def __enter__(self):
        self._noise_list = apply_weight_noise(self.model, self.alpha)
        return self

    def __exit__(self, *args):
        remove_weight_noise(self._noise_list)
        return False


if __name__ == "__main__":
    # Sanity check: apply noise changes params; remove restores them
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    w0 = model[0].weight.data.clone()
    noise_list = apply_weight_noise(model, 0.1)
    assert not torch.allclose(model[0].weight.data, w0), "params should change"
    remove_weight_noise(noise_list)
    assert torch.allclose(model[0].weight.data, w0), "params should restore"
    print("noise.py: apply/remove ok")
