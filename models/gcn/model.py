"""
models.gcn.model
================
Graph Convolutional Network for superpixel-based HSI classification.

The network operates on a graph where each node is a superpixel.
Node features are PCA-reduced spectral means; edges encode spatial
and spectral proximity.

Components
----------
GraphConvolution   — a single GCN layer computing  Â·X·W
GCN                — full model: linear projection + 2 GCN layers
normalize_adjacency — symmetric normalisation  D^{-1/2}(A+I)D^{-1/2}
compute_loss        — cross-entropy restricted to labeled nodes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np


# ---------------------------------------------------------------------------
# Adjacency normalisation
# ---------------------------------------------------------------------------

def normalize_adjacency(A_sparse: sp.csr_matrix) -> torch.Tensor:
    r"""
    Symmetrically normalise a sparse adjacency matrix.

    Computes  Â = D^{-1/2} (A + I) D^{-1/2}
    where I is the identity (self-loops) and D is the diagonal degree matrix.

    Parameters
    ----------
    A_sparse : scipy.sparse.csr_matrix  shape (N, N)

    Returns
    -------
    torch.sparse_coo_tensor  shape (N, N), dtype float32
    """
    A = A_sparse + sp.eye(A_sparse.shape[0])
    degree     = np.array(A.sum(1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_norm     = D_inv_sqrt.dot(A).dot(D_inv_sqrt).tocoo()

    indices = torch.from_numpy(np.vstack((A_norm.row, A_norm.col)).astype(np.int64))
    values  = torch.from_numpy(A_norm.data.astype(np.float32))
    shape   = torch.Size(A_norm.shape)

    return torch.sparse_coo_tensor(indices, values, shape)


# ---------------------------------------------------------------------------
# GCN layer
# ---------------------------------------------------------------------------

class GraphConvolution(nn.Module):
    """
    A single Graph Convolutional layer.

    Computes  output = Â · (input · W) + b
    where Â is the pre-normalised adjacency matrix passed at call time.

    Parameters
    ----------
    in_features  : int
    out_features : int
    bias         : bool  (default True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : torch.Tensor        shape (N, in_features)
        adj : torch.sparse_tensor shape (N, N)

        Returns
        -------
        torch.Tensor  shape (N, out_features)
        """
        support = torch.mm(x, self.weight)       # (N, out_features)
        output  = torch.spmm(adj, support)        # (N, out_features)
        if self.bias is not None:
            return output + self.bias
        return output


# ---------------------------------------------------------------------------
# Full GCN model
# ---------------------------------------------------------------------------

class GCN(nn.Module):
    """
    Two-layer Graph Convolutional Network.

    Architecture
    ------------
    Linear(in_features → hidden) → ReLU → Dropout
    GCNLayer(hidden → hidden)    → ReLU → Dropout
    GCNLayer(hidden → C)

    Parameters
    ----------
    in_features     : int   — number of node features (PCA components)
    hidden_features : int   — hidden layer width
    num_classes     : int   — number of land-cover classes
    dropout         : float — dropout probability (default 0.5)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
        dropout: float = 0.5,
    ):
        super(GCN, self).__init__()

        # Linear projection to hidden dimension
        self.proj = nn.Linear(in_features, hidden_features)

        # Two GCN layers
        self.gcn1    = GraphConvolution(hidden_features, hidden_features)
        self.gcn2    = GraphConvolution(hidden_features, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : torch.Tensor  shape (N, in_features)  — node feature matrix
        adj : torch.Tensor  shape (N, N) sparse     — normalised adjacency

        Returns
        -------
        torch.Tensor  shape (N, num_classes)  — raw logits per node
        """
        # Projection
        x = self.proj(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # GCN layer 1
        x = self.gcn1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # GCN layer 2 — produces logits
        x = self.gcn2(x, adj)
        return x


# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------

def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labeled_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss computed only on labeled nodes.

    Parameters
    ----------
    logits       : torch.Tensor  shape (N, C)  — raw model output
    labels       : torch.Tensor  shape (N,)    — 1-indexed class labels (1..C)
    labeled_mask : torch.BoolTensor shape (N,) — True for labeled nodes

    Returns
    -------
    torch.Tensor  scalar loss
    """
    if labeled_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    # Labels are 1-indexed (1..C); convert to 0-indexed for CrossEntropy
    selected_logits = logits[labeled_mask]
    selected_labels = labels[labeled_mask] - 1

    return F.cross_entropy(selected_logits, selected_labels)
