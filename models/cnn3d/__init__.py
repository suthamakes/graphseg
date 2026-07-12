"""
models.cnn3d
============
3-D joint spectral-spatial CNN for patch-based HSI classification.

Exports
-------
    Net3D                — model class
    initialize_parameters — seeded factory function
"""

from .model import Net3D, initialize_parameters

__all__ = ["Net3D", "initialize_parameters"]
