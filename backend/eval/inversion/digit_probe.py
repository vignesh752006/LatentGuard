"""Spatial digit-string classifier for inversion benchmark embeddings."""

import torch
from torch import nn
from torch.nn import functional as F


class DigitGridProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1152, 32, kernel_size=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 16 * 10),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        features = self.encoder(tensor)
        features = F.adaptive_avg_pool2d(features, (4, 4)).flatten(1)
        return self.classifier(features).reshape(-1, 16, 10)
