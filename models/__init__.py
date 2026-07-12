"""
models
======
Top-level re-export of every model class.

Usage
-----
    from models.cnn1d import Net
    from models.cnn2d import Net2D
    from models.cnn3d import Net3D
    from models.gcn   import GCN
"""

from models.cnn1d.model import Net
from models.cnn2d.model import Net2D
from models.cnn3d.model import Net3D
from models.gcn.model   import GCN

__all__ = ["Net", "Net2D", "Net3D", "GCN"]
