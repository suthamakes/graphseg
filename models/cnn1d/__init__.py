"""
models.cnn1d
============
1-D spectral CNN for pixel-wise HSI classification.

Exports
-------
    Net                  — model class
    initialize_parameters — seeded factory function
"""

from .model import Net, initialize_parameters

__all__ = ["Net", "initialize_parameters"]
