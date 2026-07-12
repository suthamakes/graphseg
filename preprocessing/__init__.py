"""
preprocessing
=============
Data loading and model-specific preprocessing pipelines.

Offline scripts (run once to prepare .npy arrays on disk)
---------------------------------------------------------
    preprocessing/1DCNN.py   → data/extracted/
    preprocessing/2DCNN.py   → data/extracted_2d/
    preprocessing/3DCNN.py   → data/extracted_3d/

Runtime library (imported by src/train.py)
------------------------------------------
    preprocessing.load_hsi   — load raw .mat files
    preprocessing.GCN        — superpixel segmentation, graph construction,
                               graph partitioning
"""

from .load_hsi import load_dataset

__all__ = ["load_dataset"]
