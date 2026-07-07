import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import scipy.sparse as sp
import logging

def build_graph(image: np.ndarray, superpixels: np.ndarray, gt: np.ndarray, 
                pca_components: int = 30, top_k: int = 10, hops: list = [1, 2]):
    """
    Build graph based on superpixel segmentation.
    
    Args:
        image: HSI cube (H, W, B)
        superpixels: Superpixel label map (H, W)
        gt: Ground truth labels (H, W)
        pca_components: Number of PCA components
        top_k: Top-k nearest neighbors to keep
        hops: List of hop orders to consider
        
    Returns:
        X: Attribute matrix (N, F)
        A: Adjacency matrix (N, N) sparse
        superpixel_map: (H, W)
        node_to_pixel_indices: dict[int, np.ndarray]
        node_labels: np.ndarray (N,)
    """
    H, W, B = image.shape
    
    # 1. PCA on raw cube
    logging.info(f"Applying PCA to reduce from {B} to {pca_components} components")
    flat_image = image.reshape(-1, B)
    pca = PCA(n_components=pca_components)
    flat_pca = pca.fit_transform(flat_image)
    
    # Get unique superpixels and their contiguous IDs
    unique_ids = np.unique(superpixels)
    N = len(unique_ids)
    
    # Ensure IDs are 0 to N-1
    superpixel_map = np.zeros_like(superpixels)
    id_map = {}
    for new_id, old_id in enumerate(unique_ids):
        superpixel_map[superpixels == old_id] = new_id
        id_map[new_id] = old_id
        
    # 2. Node attributes and mapping
    logging.info(f"Computing node attributes for {N} superpixels")
    X = np.zeros((N, pca_components), dtype=np.float32)
    node_to_pixel_indices = {}
    node_labels = np.zeros(N, dtype=np.int32)
    
    flat_superpixels = superpixel_map.flatten()
    flat_gt = gt.flatten()
    
    for i in range(N):
        idx = np.where(flat_superpixels == i)[0]
        node_to_pixel_indices[i] = idx
        X[i] = flat_pca[idx].mean(axis=0)
        
        # Node label: if any pixel is labeled, use that label.
        # Handle cases where multiple labels might exist by taking the mode or first non-zero.
        node_gt = flat_gt[idx]
        labeled_pixels = node_gt[node_gt > 0]
        if len(labeled_pixels) > 0:
            # We take the most frequent non-zero label
            counts = np.bincount(labeled_pixels)
            node_labels[i] = np.argmax(counts)
            
    # 3. Spatial adjacency (Region Adjacency Graph)
    logging.info("Building spatial region adjacency graph")
    spatial_graph = nx.Graph()
    spatial_graph.add_nodes_from(range(N))
    
    # Find spatial neighbors by shifting map
    # We can check right and down neighbors for every pixel
    right_edges = np.vstack([superpixel_map[:, :-1].ravel(), superpixel_map[:, 1:].ravel()]).T
    down_edges = np.vstack([superpixel_map[:-1, :].ravel(), superpixel_map[1:, :].ravel()]).T
    
    all_edges = np.vstack([right_edges, down_edges])
    # Keep only edges between different superpixels
    diff_mask = all_edges[:, 0] != all_edges[:, 1]
    boundary_edges = all_edges[diff_mask]
    
    spatial_graph.add_edges_from(boundary_edges)
    
    # 4. Pairwise distance
    logging.info("Computing pairwise spectral distances")
    dist_matrix = cdist(X, X, metric='euclidean')
    
    # 5 & 6. Multi-hop extension and combining
    logging.info(f"Computing multi-hop top-{top_k} adjacency for hops {hops}")
    A_final = np.zeros((N, N), dtype=np.float32)
    
    # Pre-calculate shortest path lengths in spatial graph up to max hop
    max_hop = max(hops)
    # Using single source shortest path length
    hop_neighbors = {h: {i: set() for i in range(N)} for h in hops}
    
    for n in range(N):
        lengths = nx.single_source_shortest_path_length(spatial_graph, n, cutoff=max_hop)
        for target, length in lengths.items():
            if length in hops:
                hop_neighbors[length][n].add(target)
                
    for h in hops:
        A_h = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            candidates = list(hop_neighbors[h][i])
            if len(candidates) == 0:
                continue
            
            # Get distances to candidates
            cand_dists = dist_matrix[i, candidates]
            
            # Select top k
            k_actual = min(top_k, len(candidates))
            if k_actual > 0:
                # get indices of smallest k distances
                top_idx_local = np.argsort(cand_dists)[:k_actual]
                top_idx_global = [candidates[idx] for idx in top_idx_local]
                
                A_h[i, top_idx_global] = 1.0
                
        # Symmetrize A_h just in case, though the paper top-k is directed, we usually symmetrize for GCN
        # Let's keep it directed first, or symmetrize? Paper says "keeps only the top-k... fixes edge weight to 1". 
        A_final += A_h
        
    # Make A symmetric
    A_final = np.maximum(A_final, A_final.T)
    
    # Normalize node features (zero mean, unit variance per feature)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X = (X - X_mean) / X_std
    
    # Output sparse A
    A_sparse = sp.csr_matrix(A_final)
    
    logging.info(f"Graph construction complete. Nodes: {N}, Edges: {A_sparse.nnz}")
    return X, A_sparse, superpixel_map, node_to_pixel_indices, node_labels
