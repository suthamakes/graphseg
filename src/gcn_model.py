import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

def normalize_adjacency(A_sparse: sp.csr_matrix) -> torch.sparse.FloatTensor:
    r"""
    Symmetrically normalize adjacency matrix.
    \hat{A}_i = D_i^{-1/2} (A_i + I) D_i^{-1/2}
    """
    A = A_sparse + sp.eye(A_sparse.shape[0])
    degree = np.array(A.sum(1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_norm = D_inv_sqrt.dot(A).dot(D_inv_sqrt).tocoo()
    
    indices = torch.from_numpy(np.vstack((A_norm.row, A_norm.col)).astype(np.int64))
    values = torch.from_numpy(A_norm.data.astype(np.float32))
    shape = torch.Size(A_norm.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        # 1x1 convolution / linear projection
        self.proj = nn.Linear(in_features, hidden_features)
        
        # 2 GCN layers
        self.gcn1 = GraphConvolution(hidden_features, hidden_features)
        self.gcn2 = GraphConvolution(hidden_features, num_classes)
        self.dropout = dropout
        
    def forward(self, x, adj):
        # x is (N, F)
        x = self.proj(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Layer 1
        x = self.gcn1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Layer 2
        x = self.gcn2(x, adj)
        return x

def compute_loss(logits, labels, labeled_mask):
    """
    Cross entropy loss only on labeled nodes.
    labels: (N,) true labels
    labeled_mask: (N,) boolean mask
    """
    if labeled_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)
        
    # Note: labels are 1..num_classes, so we shift by -1 for 0-indexed PyTorch CrossEntropy
    # or ensure they are already 0-indexed.
    # We will assume labels are 1-indexed and class 0 is background.
    # Therefore the logits should correspond to classes 1..16.
    # We subtract 1 to make it 0..15.
    selected_logits = logits[labeled_mask]
    selected_labels = labels[labeled_mask] - 1
    
    return F.cross_entropy(selected_logits, selected_labels)
