"""
src/dataset.py
==============
Shared PyTorch Dataset and split utilities used by src/train.py.

HSIDataset      — generic loader for the .npy arrays produced by preprocessing/
split_data      — pixel-level stratified train/val/test split (GCN protocol)
propagate_labels_to_nodes — map pixel splits to superpixel nodes (GCN)
"""

import os
import logging

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# CNN dataset
# ---------------------------------------------------------------------------

class HSIDataset(Dataset):
    """
    Generic PyTorch Dataset for CNN-based HSI classification.

    Loads .npy arrays produced by any of the preprocessing scripts:
        preprocessing/1DCNN.py  → data/extracted/
        preprocessing/2DCNN.py  → data/extracted_2d/
        preprocessing/3DCNN.py  → data/extracted_3d/

    The labels are one-hot encoded; __getitem__ converts them to integer
    class indices expected by PyTorch CrossEntropyLoss.

    Parameters
    ----------
    data_dir : str   path to the extracted .npy directory
    is_train : bool  True → load Train_X / TrLabel, False → Test_X / TeLabel
    transform        optional transform applied to the input tensor
    logger           optional logging.Logger
    """

    def __init__(
        self,
        data_dir: str = "data/extracted",
        is_train: bool = True,
        transform=None,
        logger: logging.Logger = None,
    ):
        self.transform = transform
        self.logger    = logger

        prefix     = "Train" if is_train else "Test"
        lbl_prefix = "Tr"    if is_train else "Te"

        data_path  = os.path.join(data_dir, f"{prefix}_X.npy")
        label_path = os.path.join(data_dir, f"{lbl_prefix}Label.npy")

        if logger:
            logger.info(f"Loading {'train' if is_train else 'test'} data from {data_dir}")

        self.data   = np.load(data_path)   # any shape: (N, B,1,1) / (N,B,P,P) / (N,1,B,P,P)
        self.labels = np.load(label_path)  # (N, num_classes) one-hot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_label  = torch.tensor(int(y.argmax()), dtype=torch.long)

        if self.transform:
            x_tensor = self.transform(x_tensor)

        return x_tensor, y_label


# ---------------------------------------------------------------------------
# GCN split utilities
# ---------------------------------------------------------------------------

def split_data(gt_map: np.ndarray, num_classes: int = 16, seed: int = 0):
    """
    Stratified pixel-level train / val / test split following the GCN paper
    protocol (30 labeled pixels per class; classes 7 and 9 get 15 each).

    Parameters
    ----------
    gt_map      : np.ndarray  shape (H, W)  ground-truth labels (0 = background)
    num_classes : int
    seed        : int

    Returns
    -------
    train_mask, val_mask, test_mask  — boolean np.ndarray each of shape (H, W)
    """
    np.random.seed(seed)

    H, W = gt_map.shape
    train_mask = np.zeros((H, W), dtype=bool)
    val_mask   = np.zeros((H, W), dtype=bool)
    test_mask  = np.zeros((H, W), dtype=bool)

    samples_per_class = {c: 30 for c in range(1, num_classes + 1)}
    samples_per_class[7] = 15
    samples_per_class[9] = 15

    for c in range(1, num_classes + 1):
        ys, xs  = np.where(gt_map == c)
        indices = np.arange(len(ys))
        if len(indices) == 0:
            continue

        np.random.shuffle(indices)
        n_labeled = min(samples_per_class[c], len(indices))
        n_train   = int(np.round(n_labeled * 0.9))
        n_val     = n_labeled - n_train

        train_mask[ys[indices[:n_train]],              xs[indices[:n_train]]]              = True
        val_mask  [ys[indices[n_train:n_labeled]],     xs[indices[n_train:n_labeled]]]     = True
        test_mask [ys[indices[n_labeled:]],            xs[indices[n_labeled:]]]            = True

    logging.info(
        f"Split → Train: {train_mask.sum()}  Val: {val_mask.sum()}  Test: {test_mask.sum()}"
    )
    return train_mask, val_mask, test_mask


def propagate_labels_to_nodes(
    node_to_pixel_indices: dict,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    gt_map: np.ndarray,
):
    """
    Map pixel-level splits to superpixel nodes.

    A node is assigned to the first split that contains any of its pixels
    (priority: train > val > test).

    Parameters
    ----------
    node_to_pixel_indices : dict[int, np.ndarray]  flat pixel indices per node
    train_mask, val_mask, test_mask : np.ndarray   shape (H, W)
    gt_map                          : np.ndarray   shape (H, W)  (unused, kept for API compat)

    Returns
    -------
    node_train, node_val, node_test  — boolean np.ndarray each of shape (N,)
    """
    N          = len(node_to_pixel_indices)
    node_train = np.zeros(N, dtype=bool)
    node_val   = np.zeros(N, dtype=bool)
    node_test  = np.zeros(N, dtype=bool)

    flat_train = train_mask.flatten()
    flat_val   = val_mask.flatten()
    flat_test  = test_mask.flatten()

    for i in range(N):
        idx = node_to_pixel_indices[i]
        if np.any(flat_train[idx]):
            node_train[i] = True
        elif np.any(flat_val[idx]):
            node_val[i]   = True
        elif np.any(flat_test[idx]):
            node_test[i]  = True

    logging.info(
        f"Node split → Train: {node_train.sum()}  Val: {node_val.sum()}  Test: {node_test.sum()}"
    )
    return node_train, node_val, node_test
