"""
models.cnn1d.model
==================
1-D spectral CNN for pixel-wise HSI classification.

The model treats each pixel as a 1-D spectral vector and applies
Conv1d layers with kernels > 1 that slide along the spectral dimension,
learning local spectral patterns.

Input shape : (N, 1, B)  where B = number of spectral bands
Output shape: (N, C)
"""

import torch
import torch.nn as nn


class Net(nn.Module):
    """
    1-D spectral CNN using Conv1d with spectral kernels.

    Architecture
    ------------
    Conv1d(1 → 64, k=7, pad=3) → BatchNorm1d → ReLU → Dropout
    Conv1d(64 → 128, k=5, pad=2) → BatchNorm1d → ReLU → Dropout
    AdaptiveAvgPool1d(1) → Linear(128 → C)

    Parameters
    ----------
    in_channels : int
        Number of input channels (default: 1 for spectral vector).
    num_classes : int
        Number of land-cover classes (default: 16).
    dropout : float
        Dropout probability applied after each conv block.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 16, dropout: float = 0.2):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(64, momentum=0.1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(128, momentum=0.1)
        self.relu  = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(128, num_classes)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        self.conv1.bias.data.zero_()
        self.conv2.bias.data.zero_()
        self.fc.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor  shape (N, 1, B)

        Returns
        -------
        torch.Tensor  shape (N, C)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


def initialize_parameters(
    in_channels: int = 1,
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
