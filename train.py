"""
Single training run: clean or noisy (weight noise per batch).
Optimizer updates the clean weights; noisy training uses W+ε only for forward/backward.
"""

import torch
import torch.nn as nn

import config
from data import get_loaders
from models import get_model
from noise import apply_weight_noise, remove_weight_noise


def set_seed(seed):
    """Set seed for reproducibility before creating model/optimizer."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one(arch_name, train_loader, noisy, seed, device, save_path=None, epochs=None, alpha_train=None):
    """
    Train one model: clean or noisy regime.
    Noisy: apply weight noise each batch before forward, remove after backward, then step (so optimizer updates clean W).
    alpha_train: noise strength when noisy=True; when None, uses config.ALPHA_TRAIN.
    Returns the trained model (and saves state_dict to save_path if given).
    """
    set_seed(seed)
    model = get_model(arch_name).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    n_epochs = epochs if epochs is not None else config.EPOCHS
    alpha = alpha_train if alpha_train is not None else config.ALPHA_TRAIN
    model.train()

    for epoch in range(n_epochs):
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Noisy: forward/backward use W+ε; optimizer updates clean W.
            if noisy and alpha > 0:
                noise_list = apply_weight_noise(model, alpha)
            try:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
            finally:
                if noisy and alpha > 0:
                    remove_weight_noise(noise_list)
            optimizer.step()

            running_loss += loss.item()
        # Minimal logging: mean loss per epoch
        n_batches = len(train_loader)
        print(f"  epoch {epoch + 1}/{n_epochs} loss={running_loss / n_batches:.4f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"  saved {save_path}")
    return model


if __name__ == "__main__":
    # Quick sanity run: 1 epoch, MLP, clean then noisy
    device = config.DEVICE
    train_loader, _ = get_loaders("cifar10")
    print("train_one(mlp, clean, seed=0)")
    train_one("mlp", train_loader, noisy=False, seed=0, device=device, save_path="checkpoint_mlp_clean.pt", epochs=1, alpha_train=0.0)
    print("train_one(mlp, noisy, seed=0)")
    train_one("mlp", train_loader, noisy=True, seed=0, device=device, save_path="checkpoint_mlp_noisy.pt", epochs=1, alpha_train=0.05)