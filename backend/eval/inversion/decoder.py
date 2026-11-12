"""Convolutional decoder shared by training and evaluation."""

import torch
from torch import nn
from torch.nn import functional as F


class ConvDecoder(nn.Module):
    """Reduce channels at 24x24, then upsample with learned transposed convolutions."""

    def __init__(self):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(1152, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.upsample(self.reduce(latent))
        features = F.interpolate(features, size=(224, 224), mode="bilinear", align_corners=False)
        return torch.sigmoid(features)
