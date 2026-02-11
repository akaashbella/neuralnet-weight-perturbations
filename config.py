"""
Experiment config: seeds, noise strengths, training hyperparameters.
Single place so training recipe is consistent across all models.
"""

import torch

# Random seeds for reproducibility (10 seeds: 0 through 9)
SEEDS = list(range(10))

# Orchestration uses ALPHA_TRAIN_LIST; this is only the fallback when alpha_train is not passed (e.g. ad-hoc scripts).
ALPHA_TRAIN = 0.05

# Train each arch in both regimes: [clean_alpha, noisy_alpha]. run_experiments.py loops over this (no hardcoding).
ALPHA_TRAIN_LIST = [0.0, 0.05]

# Evaluation-time perturbation strengths for robustness sweep (sorted)
ALPHA_TEST_LIST = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
# Number of noise samples per α_test to average over (reduces variance; 1 = single draw)
ROBUSTNESS_NUM_SAMPLES = 10

# Training recipe (same for all architectures; single recipe for comparability)
# AdamW + cosine LR decay over full training; identical across archs for thesis comparability.
EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2

# Data, checkpoints, and results
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"
IMAGE_SIZE = 32  # CIFAR-10 is 32×32; resize in transform if needed for an architecture

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # set to "cpu" if no GPU
