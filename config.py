"""
Experiment config: seeds, noise strengths, training hyperparameters.
Single place so training recipe is consistent across all models.
"""

import torch

# Random seeds for reproducibility (3 seeds as specified)
SEEDS = [0, 1, 2]

# Training-time weight noise strength (noisy regime only)
ALPHA_TRAIN = 0.05

# Evaluation-time perturbation strengths for robustness sweep
ALPHA_TEST_LIST = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
# Number of noise samples per α_test to average over (reduces variance; 1 = single draw)
ROBUSTNESS_NUM_SAMPLES = 5

# Training recipe (same for all architectures)
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# Data, checkpoints, and results
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"
IMAGE_SIZE = 32  # CIFAR-10 is 32×32; resize in transform if needed for an architecture

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # set to "cpu" if no GPU
