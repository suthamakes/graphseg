"""
models.cnn2d
============
2-D spatial-spectral CNN for patch-based HSI classification.

Exports
-------
    Net2D                — model class
    initialize_parameters — seeded factory function
"""

from .model import Net2D, initialize_parameters

__all__ = ["Net2D", "initialize_parameters"]
