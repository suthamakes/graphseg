"""
models.cnn2d.model
==================
2-D spatial-spectral CNN for patch-based HSI classification.

Each sample is a spatial patch of shape (B, P, P) centred on a labeled
pixel.  2-D convolutions slide over the spatial dimensions while treating
spectral bands as input channels, learning joint spatial-spectral features.

Input shape : (N, B, P, P)  where B = spectral bands, P = patch side length
Output shape: (N, C)         logits ready for CrossEntropyLoss
"""

import torch
import torch.nn as nn


class Net2D(nn.Module):
    """
    2-D spatial-spectral CNN.

    Architecture
    ------------
    Block 1 : Conv2d(B  →  64, 3×3, pad=1) → BatchNorm2d → ReLU → Dropout2d
    Block 2 : Conv2d(64 → 128, 3×3, pad=1) → BatchNorm2d → ReLU → Dropout2d
    Head    : AdaptiveAvgPool2d(1) → flatten → Linear(128 → C)

    Parameters
    ----------
    in_channels : int
        Number of spectral bands (default: 200).
    num_classes : int
        Number of land-cover classes (default: 16).
    dropout : float
        Dropout2d probability after each conv block.
    """

    def __init__(self, in_channels: int = 200, num_classes: int = 16, dropout: float = 0.4):
        super(Net2D, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

        # Global average pool: (N, 128, P, P) → (N, 128, 1, 1) → (N, 128)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, num_classes)

        # Xavier uniform initialisation, zero bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor  shape (N, B, P, P)

        Returns
        -------
        torch.Tensor  shape (N, C)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x)             # (N, 128, 1, 1)
        x = x.flatten(start_dim=1)  # (N, 128)
        x = self.head(x)            # (N, C)
        return x


def initialize_parameters(
    in_channels: int = 200,
    num_classes: int = 16,
    dropout: float = 0.4,
    seed: int = 1,
) -> Net2D:
    """
    Create a seeded Net2D instance for reproducible initialisation.

    Parameters
    ----------
    in_channels : int
    num_classes  : int
    dropout      : float
    seed         : int  (default 1)

    Returns
    -------
    Net2D
    """
    torch.manual_seed(seed)
    return Net2D(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
