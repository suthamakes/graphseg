import numpy as np
import scipy.sparse as sp
import pymetis
import logging

def partition_graph(A_sparse: sp.csr_matrix, X: np.ndarray, num_clusters: int, node_labels: np.ndarray):
    """
    Partition graph using METIS.
    
    Args:
        A_sparse: Adjacency matrix in CSR format (N, N)
        X: Node features (N, F)
        num_clusters: Number of clusters (c)
        node_labels: Ground truth labels for nodes (N,)
        
    Returns:
        sub_graphs: List of dicts, each containing:
            'V': List of original node indices
            'A': Sub-adjacency matrix (sparse)
            'X': Sub-feature matrix
            'Y': Sub-labels
        edge_cut_ratio: float
    """
    N = A_sparse.shape[0]
    logging.info(f"Partitioning graph of {N} nodes into {num_clusters} clusters with METIS")
    
    # pymetis expects adjacency list
    # since A_sparse is csr, it's easy to extract
    A_sparse.eliminate_zeros()
    xadj = A_sparse.indptr
    adjncy = A_sparse.indices
    
    # Call pymetis
    n_cuts, membership = pymetis.part_graph(num_clusters, xadj=xadj, adjncy=adjncy)
    
    membership = np.array(membership)
    
    total_edges = A_sparse.nnz // 2 # undirected
    edge_cut_ratio = n_cuts / max(total_edges, 1)
    logging.info(f"METIS edge cut: {n_cuts}/{total_edges} (ratio: {edge_cut_ratio:.4f})")
    
    sub_graphs = []
    
    for c in range(num_clusters):
        # Nodes in this partition
        V_i = np.where(membership == c)[0]
        
        # Sub-adjacency matrix
        A_i = A_sparse[V_i][:, V_i]
        
        # Sub-features
        X_i = X[V_i]
        
        # Sub-labels
        Y_i = node_labels[V_i]
        
        sub_graphs.append({
            'V': V_i,
            'A': A_i,
            'X': X_i,
            'Y': Y_i
        })
        
    return sub_graphs, edge_cut_ratio
