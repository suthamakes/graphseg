import numpy as np
import logging

def split_data(gt_map, num_classes=16, seed=0):
    """
    Split the dataset according to the paper protocol.
    Returns boolean masks for train, val, and test pixels.
    """
    np.random.seed(seed)
    
    H, W = gt_map.shape
    train_mask = np.zeros((H, W), dtype=bool)
    val_mask = np.zeros((H, W), dtype=bool)
    test_mask = np.zeros((H, W), dtype=bool)
    
    # Paper specific sampling per class
    # 30 per class except 7 and 9 which get 15.
    samples_per_class = {c: 30 for c in range(1, num_classes + 1)}
    samples_per_class[7] = 15
    samples_per_class[9] = 15
    
    for c in range(1, num_classes + 1):
        # find all pixels of class c
        ys, xs = np.where(gt_map == c)
        indices = np.arange(len(ys))
        
        if len(indices) == 0:
            continue
            
        np.random.shuffle(indices)
        
        n_labeled = samples_per_class[c]
        
        # If class has fewer than n_labeled, take all (shouldn't happen with correct config)
        n_labeled = min(n_labeled, len(indices))
        
        # 90% train, 10% val of the labeled set
        n_train = int(np.round(n_labeled * 0.9))
        n_val = n_labeled - n_train
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_labeled]
        test_idx = indices[n_labeled:]
        
        train_mask[ys[train_idx], xs[train_idx]] = True
        val_mask[ys[val_idx], xs[val_idx]] = True
        test_mask[ys[test_idx], xs[test_idx]] = True
        
    logging.info(f"Split sizes -> Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    return train_mask, val_mask, test_mask

def propagate_labels_to_nodes(node_to_pixel_indices, train_mask, val_mask, test_mask, gt_map):
    """
    Given the pixel-level splits, determine which superpixel nodes are labeled.
    A node is in train if it contains any train pixel.
    A node is in val if it contains any val pixel (and not train).
    """
    N = len(node_to_pixel_indices)
    
    node_train = np.zeros(N, dtype=bool)
    node_val = np.zeros(N, dtype=bool)
    node_test = np.zeros(N, dtype=bool)
    
    flat_train = train_mask.flatten()
    flat_val = val_mask.flatten()
    flat_test = test_mask.flatten()
    
    for i in range(N):
        idx = node_to_pixel_indices[i]
        
        if np.any(flat_train[idx]):
            node_train[i] = True
        elif np.any(flat_val[idx]):
            node_val[i] = True
        elif np.any(flat_test[idx]):
            node_test[i] = True
            
    logging.info(f"Node split -> Train: {node_train.sum()}, Val: {node_val.sum()}, Test: {node_test.sum()}")
    return node_train, node_val, node_test
