"""
preprocessing.GCN
=================
Runtime graph-construction pipeline for the GCN model.

This module is imported by src/train.py at runtime (not a run-once script).
It consolidates three steps into straightforward functions:

    1. segment          — SLIC superpixel segmentation on the HSI cube
    2. build_graph      — node attributes + multi-hop spatial adjacency
    3. partition_graph  — METIS balanced graph partitioning for mini-batch GCN

Flow
----
    cube, gt, metadata = load_dataset("indian_pines")
                │
                ▼
    superpixel_map = segment(cube, n_segments, compactness)
                │
                ▼
    X, A, superpixel_map, node_to_pixel_indices, node_labels
                = build_graph(cube, superpixel_map, gt, ...)
                │
                ▼
    sub_graphs, edge_cut_ratio
                = partition_graph(A, X, num_clusters, node_labels)
"""

import logging

import numpy as np
import scipy.sparse as sp
import networkx as nx
import pymetis

from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from skimage.segmentation import slic


# ---------------------------------------------------------------------------
# Step 1 — Superpixel segmentation
# ---------------------------------------------------------------------------

def segment(image: np.ndarray, n_segments: int, compactness: float) -> np.ndarray:
    """
    Cluster spectrally-homogeneous pixels into superpixels using SLIC.

    Reduces the HSI cube to 3 PCA components first so that SLIC operates in a
    perceptually meaningful low-dimensional space.

    Parameters
    ----------
    image       : np.ndarray  shape (H, W, B)
    n_segments  : int         approximate number of superpixels
    compactness : float       spatial vs. spectral trade-off (higher = more square)

    Returns
    -------
    np.ndarray  shape (H, W)  integer superpixel label map
    """
    H, W, B = image.shape
    logging.info(f"[segment] PCA: {B} → 3 components for SLIC")

    flat = image.reshape(-1, B)
    pca  = PCA(n_components=3)
    pca_flat = pca.fit_transform(flat)
    image_pca = pca_flat.reshape(H, W, 3)

    # Normalise each PCA channel to [0, 1] for consistent compactness
    for i in range(3):
        lo, hi = image_pca[..., i].min(), image_pca[..., i].max()
        if hi > lo:
            image_pca[..., i] = (image_pca[..., i] - lo) / (hi - lo)

    logging.info(f"[segment] SLIC: n_segments={n_segments}, compactness={compactness}")
    segments = slic(
        image_pca,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        enforce_connectivity=True,
    )

    logging.info(f"[segment] Generated {len(np.unique(segments))} superpixels")
    return segments


# ---------------------------------------------------------------------------
# Step 2 — Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    image: np.ndarray,
    superpixels: np.ndarray,
    gt: np.ndarray,
    pca_components: int = 30,
    top_k: int = 10,
    hops: list = [1, 2],
):
    """
    Build a superpixel graph from the HSI cube.

    Steps
    -----
    1. PCA on the raw cube → node feature matrix X  (N, pca_components)
    2. Compute mean PCA feature per superpixel node
    3. Spatial Region Adjacency Graph (RAG) via pixel-shift trick
    4. Multi-hop top-k adjacency: for each hop order h, keep the k nearest
       spectral neighbours within h hops in the spatial RAG
    5. Symmetrise and normalise node features (zero-mean, unit-variance)

    Parameters
    ----------
    image          : np.ndarray  shape (H, W, B)
    superpixels    : np.ndarray  shape (H, W)   initial superpixel map
    gt             : np.ndarray  shape (H, W)   ground-truth labels (0 = background)
    pca_components : int         number of PCA components for node features
    top_k          : int         number of spectral neighbours to keep per hop
    hops           : list[int]   hop orders to include in the adjacency

    Returns
    -------
    X                   : np.ndarray       shape (N, pca_components)  node features
    A_sparse            : sp.csr_matrix    shape (N, N)               adjacency
    superpixel_map      : np.ndarray       shape (H, W)               re-indexed labels
    node_to_pixel_indices : dict[int, np.ndarray]   node → flat pixel indices
    node_labels         : np.ndarray       shape (N,)                 majority GT label
    """
    H, W, B = image.shape

    # 1. PCA on raw cube
    logging.info(f"[build_graph] PCA: {B} → {pca_components} components")
    flat_image = image.reshape(-1, B)
    pca      = PCA(n_components=pca_components)
    flat_pca = pca.fit_transform(flat_image)

    # Remap superpixel IDs to contiguous 0..N-1
    unique_ids    = np.unique(superpixels)
    N             = len(unique_ids)
    superpixel_map = np.zeros_like(superpixels)
    for new_id, old_id in enumerate(unique_ids):
        superpixel_map[superpixels == old_id] = new_id

    # 2. Node attributes and GT labels
    logging.info(f"[build_graph] Computing node attributes for {N} superpixels")
    X                     = np.zeros((N, pca_components), dtype=np.float32)
    node_to_pixel_indices = {}
    node_labels           = np.zeros(N, dtype=np.int32)

    flat_sp = superpixel_map.flatten()
    flat_gt = gt.flatten()

    for i in range(N):
        idx                      = np.where(flat_sp == i)[0]
        node_to_pixel_indices[i] = idx
        X[i]                     = flat_pca[idx].mean(axis=0)

        # Majority non-zero label for the node
        node_gt        = flat_gt[idx]
        labeled_pixels = node_gt[node_gt > 0]
        if len(labeled_pixels) > 0:
            counts         = np.bincount(labeled_pixels)
            node_labels[i] = np.argmax(counts)

    # 3. Spatial Region Adjacency Graph
    logging.info("[build_graph] Building spatial RAG")
    spatial_graph = nx.Graph()
    spatial_graph.add_nodes_from(range(N))

    right_edges = np.vstack([superpixel_map[:, :-1].ravel(), superpixel_map[:, 1:].ravel()]).T
    down_edges  = np.vstack([superpixel_map[:-1, :].ravel(), superpixel_map[1:, :].ravel()]).T
    all_edges   = np.vstack([right_edges, down_edges])
    diff_mask   = all_edges[:, 0] != all_edges[:, 1]
    spatial_graph.add_edges_from(all_edges[diff_mask])

    # 4. Multi-hop top-k spectral adjacency
    logging.info(f"[build_graph] Pairwise spectral distances + multi-hop top-{top_k} for hops {hops}")
    dist_matrix = cdist(X, X, metric="euclidean")

    max_hop      = max(hops)
    hop_neighbors = {h: {i: set() for i in range(N)} for h in hops}

    for n in range(N):
        lengths = nx.single_source_shortest_path_length(spatial_graph, n, cutoff=max_hop)
        for target, length in lengths.items():
            if length in hops:
                hop_neighbors[length][n].add(target)

    A_final = np.zeros((N, N), dtype=np.float32)
    for h in hops:
        A_h = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            candidates = list(hop_neighbors[h][i])
            if not candidates:
                continue
            k_actual = min(top_k, len(candidates))
            top_local  = np.argsort(dist_matrix[i, candidates])[:k_actual]
            top_global = [candidates[j] for j in top_local]
            A_h[i, top_global] = 1.0
        A_final += A_h

    # Make symmetric
    A_final = np.maximum(A_final, A_final.T)

    # 5. Normalise node features (zero-mean, unit-variance per feature)
    X_mean          = X.mean(axis=0)
    X_std           = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X = (X - X_mean) / X_std

    A_sparse = sp.csr_matrix(A_final)
    logging.info(f"[build_graph] Done. Nodes: {N}, Edges: {A_sparse.nnz}")

    return X, A_sparse, superpixel_map, node_to_pixel_indices, node_labels


# ---------------------------------------------------------------------------
# Step 3 — Graph partitioning
# ---------------------------------------------------------------------------

def partition_graph(
    A_sparse: sp.csr_matrix,
    X: np.ndarray,
    num_clusters: int,
    node_labels: np.ndarray,
):
    """
    Partition the graph into balanced sub-graphs using METIS.

    Mini-batch GCN training samples one sub-graph per step, so balanced
    partitions improve gradient variance.

    Parameters
    ----------
    A_sparse     : sp.csr_matrix  shape (N, N)
    X            : np.ndarray     shape (N, F)
    num_clusters : int            number of partitions  (c)
    node_labels  : np.ndarray     shape (N,)  GT labels per node

    Returns
    -------
    sub_graphs     : list[dict]  each dict has keys:
                       'V' — original node indices  (np.ndarray)
                       'A' — sub-adjacency           (sp.csr_matrix)
                       'X' — sub-features            (np.ndarray)
                       'Y' — sub-labels              (np.ndarray)
    edge_cut_ratio : float  fraction of edges crossing partition boundaries
    """
    N = A_sparse.shape[0]
    logging.info(f"[partition_graph] METIS: {N} nodes → {num_clusters} clusters")

    A_sparse.eliminate_zeros()
    xadj   = A_sparse.indptr
    adjncy = A_sparse.indices

    n_cuts, membership = pymetis.part_graph(num_clusters, xadj=xadj, adjncy=adjncy)
    membership = np.array(membership)

    total_edges    = A_sparse.nnz // 2  # undirected
    edge_cut_ratio = n_cuts / max(total_edges, 1)
    logging.info(f"[partition_graph] Edge cut: {n_cuts}/{total_edges} (ratio {edge_cut_ratio:.4f})")

    sub_graphs = []
    for c in range(num_clusters):
        V_i = np.where(membership == c)[0]
        sub_graphs.append({
            "V": V_i,
            "A": A_sparse[V_i][:, V_i],
            "X": X[V_i],
            "Y": node_labels[V_i],
        })

    return sub_graphs, edge_cut_ratio
