"""
models.gcn
==========
Graph Convolutional Network for superpixel-based HSI classification.

Exports
-------
    GCN                  — main GCN model (projection + 2 GCN layers)
    GraphConvolution     — single GCN layer (sparse A·X·W)
    normalize_adjacency  — symmetric normalisation D^{-1/2}(A+I)D^{-1/2}
    compute_loss         — cross-entropy on labeled nodes only
"""

from .model import GCN, GraphConvolution, normalize_adjacency, compute_loss

__all__ = ["GCN", "GraphConvolution", "normalize_adjacency", "compute_loss"]
