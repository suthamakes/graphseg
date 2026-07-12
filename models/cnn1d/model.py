"""
models.cnn1d.model
==================
1-D spectral CNN for pixel-wise HSI classification.

The model treats each pixel as a 1-D spectral vector and applies
1×1 convolutions (equivalent to per-band linear projections) to learn
a compact spectral embedding before classification.

Input shape : (N, B, 1, 1)  where B = number of spectral bands
Output shape: (N, C, 1, 1)  where C = number of classes
              → squeeze to (N, C) before loss
"""

import torch
import torch.nn as nn


class Net(nn.Module):
    """
    1-D spectral CNN using Conv2d(1×1) kernels.

    Architecture
    ------------
    Conv2d(B → 128, k=1) → BatchNorm2d → ReLU → Dropout(0.2)
    Conv2d(128 → C,  k=1)

    Parameters
    ----------
    in_channels : int
        Number of spectral bands (default: 200 for Indian Pines corrected).
    num_classes : int
        Number of land-cover classes (default: 16).
    dropout : float
        Dropout probability applied after the first conv block.
    """

    def __init__(self, in_channels: int = 200, num_classes: int = 16, dropout: float = 0.2):
        super(Net, self).__init__()

        self.conv1   = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.bn1     = nn.BatchNorm2d(128, momentum=0.1)
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2   = nn.Conv2d(128, num_classes, kernel_size=1)

        # Xavier uniform initialisation, zero bias
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv1.bias.data.zero_()
        self.conv2.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor  shape (N, B, 1, 1)

        Returns
        -------
        torch.Tensor  shape (N, C, 1, 1)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


def initialize_parameters(
    in_channels: int = 200,
    num_classes: int = 16,
    dropout: float = 0.2,
    seed: int = 1,
) -> Net:
    """
    Create a seeded Net instance for reproducible initialisation.

    Parameters
    ----------
    in_channels : int
    num_classes  : int
    dropout      : float
    seed         : int  (default 1)

    Returns
    -------
    Net
    """
    torch.manual_seed(seed)
    return Net(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
