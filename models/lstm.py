"""
Type: RNN (LSTM).
LSTM classifier for CIFAR-10: treats each image row as a timestep.
Input: (B, 3, 32, 32). Output: (B, num_classes).
"""

import torch.nn as nn


class LSTM(nn.Module):
    """LSTM: hidden_size=256/512, num_layers=1/2, dropout only when num_layers > 1."""

    def __init__(self, num_classes=10, hidden_size=256, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 32 rows, 96 features per row (32 * 3)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(96, hidden_size, num_layers=num_layers, batch_first=True, dropout=lstm_dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B = x.size(0)
        # (B, 3, 32, 32) -> (B, 32, 3, 32) -> (B, 32, 96)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, 32, 96)
        _, (h_n, _) = self.lstm(x)
        feat = h_n[-1]  # (B, hidden_size)
        return self.fc(feat)
