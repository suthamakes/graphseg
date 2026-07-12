"""
models.cnn3d.model
==================
3-D joint spectral-spatial CNN for patch-based HSI classification.

Each sample is a volume of shape (1, B, P, P).  The 3-D convolution kernel
slides simultaneously over spectral depth *and* both spatial dimensions,
capturing correlations that 1-D (spectral-only) or 2-D (spatial-only) kernels
cannot.

Input shape : (N, 1, B, P, P)
  1  = single input channel
  B  = spectral bands (depth axis)
  P  = spatial patch side length

Output shape: (N, C)  logits ready for CrossEntropyLoss
"""

import torch
import torch.nn as nn


class Net3D(nn.Module):
    """
    3-D joint spectral-spatial CNN.

    Architecture
    ------------
    Block 1 : Conv3d(1  →  8, (7,3,3), pad=(3,1,1)) → BatchNorm3d → ReLU → Dropout3d
    Block 2 : Conv3d(8  → 16, (5,3,3), pad=(2,1,1)) → BatchNorm3d → ReLU → Dropout3d
    Block 3 : Conv3d(16 → 32, (3,1,1), pad=(1,0,0)) → BatchNorm3d → ReLU
    Head    : AdaptiveAvgPool3d(1) → flatten → Linear(32 → C)

    Kernel design
    -------------
    Large spectral kernels (7 / 5 / 3) capture broad spectral correlations.
    Small spatial kernels (3×3 then 1×1) focus on local spatial structure.
    Symmetric padding keeps spatial dimensions intact; spectral depth is
    preserved across all three blocks.

    Parameters
    ----------
    in_channels : int
        Number of input channels — always 1 for the single-channel 3-D volume.
    num_classes : int
        Number of land-cover classes (default: 16).
    dropout : float
        Dropout3d probability applied after blocks 1 and 2.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 16, dropout: float = 0.4):
        super(Net3D, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(8, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(16, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
        )

        self.block3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32, momentum=0.1),
            nn.ReLU(inplace=True),
        )

        # Global average pool: (N, 32, B, P, P) → (N, 32, 1, 1, 1) → (N, 32)
        self.gap  = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(32, num_classes)

        # Xavier uniform initialisation, zero bias
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
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
        x : torch.Tensor  shape (N, 1, B, P, P)

        Returns
        -------
        torch.Tensor  shape (N, C)
        """
        x = self.block1(x)          # (N,  8, B, P, P)
        x = self.block2(x)          # (N, 16, B, P, P)
        x = self.block3(x)          # (N, 32, B, P, P)
        x = self.gap(x)             # (N, 32, 1, 1, 1)
        x = x.flatten(start_dim=1)  # (N, 32)
        x = self.head(x)            # (N, C)
        return x


def initialize_parameters(
    in_channels: int = 1,
    num_classes: int = 16,
    dropout: float = 0.4,
    seed: int = 1,
) -> Net3D:
    """
    Create a seeded Net3D instance for reproducible initialisation.

    Parameters
    ----------
    in_channels : int
    num_classes  : int
    dropout      : float
    seed         : int  (default 1)

    Returns
    -------
    Net3D
    """
    torch.manual_seed(seed)
    return Net3D(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
