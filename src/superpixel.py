import numpy as np
from sklearn.decomposition import PCA
from skimage.segmentation import slic
import logging

def segment(image: np.ndarray, n_segments: int, compactness: float) -> np.ndarray:
    """
    Cluster spectrally-homogeneous pixels into superpixels using SLIC.
    
    Args:
        image: np.ndarray of shape (H, W, B)
        n_segments: Approximate number of superpixels (p)
        compactness: Balances color proximity and space proximity
        
    Returns:
        np.ndarray of shape (H, W) containing integer labels for each superpixel.
    """
    H, W, B = image.shape
    
    logging.info(f"Applying PCA to reduce from {B} to 3 components for SLIC segmentation")
    # Reshape for PCA
    flat_image = image.reshape(-1, B)
    
    pca = PCA(n_components=3)
    flat_pca = pca.fit_transform(flat_image)
    
    # Reshape back to image dimensions
    image_pca = flat_pca.reshape(H, W, 3)
    
    # Normalize PCA components for SLIC (usually it expects values in reasonable range)
    # SLIC works well with lab or float inputs, but normalizing helps compactness be consistent
    for i in range(3):
        min_val, max_val = image_pca[..., i].min(), image_pca[..., i].max()
        if max_val > min_val:
            image_pca[..., i] = (image_pca[..., i] - min_val) / (max_val - min_val)
            
    logging.info(f"Running SLIC with n_segments={n_segments}, compactness={compactness}")
    # Run SLIC. Setting start_label=0 is convenient.
    segments = slic(image_pca, n_segments=n_segments, compactness=compactness, 
                    start_label=0, enforce_connectivity=True)
    
    logging.info(f"Generated {len(np.unique(segments))} superpixels")
    return segments
