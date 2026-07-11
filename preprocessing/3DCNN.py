import numpy as np
import scipy.io as sio
import os
from sklearn.model_selection import train_test_split

# change data path for different datasets
DATA_PATH  = 'data/raw/Indian_pines_corrected.mat'
GT_PATH    = 'data/raw/Indian_pines_gt.mat'
OUTPUT_DIR = 'data/extracted_3d'
PATCH_SIZE  = 5    # spatial window: PATCH_SIZE x PATCH_SIZE pixels per sample
NUM_CLASSES = 16
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------

def load_mat_data(data_path, gt_path):
    """Load hyperspectral cube and ground-truth map from .mat files."""
    print("Loading .mat files from Source...")
    data_mat = sio.loadmat(data_path)
    gt_mat   = sio.loadmat(gt_path)

    raw_data     = data_mat['indian_pines_corrected']   # (H, W, B)
    ground_truth = gt_mat['indian_pines_gt']            # (H, W)

    print(f"Raw Data Shape: {raw_data.shape} | Ground Truth Shape: {ground_truth.shape}")
    return raw_data, ground_truth


def remove_noise_bands(data_cube, noise_band_indices):
    """Remove water-absorption / noise bands from the spectral axis (axis=2)."""
    total_bands = data_cube.shape[2]
    all_bands   = np.arange(total_bands)

    # Keep indices that are NOT in noise_band_indices
    clean_band_indices = np.isin(all_bands, noise_band_indices, invert=True)
    cleaned_cube = data_cube[:, :, clean_band_indices]

    print(
        f"Original bands: {total_bands} | "
        f"Removed: {len(noise_band_indices)} | "
        f"Clean bands kept: {cleaned_cube.shape[2]}"
    )
    return cleaned_cube


def mat2gray_normalize(data_cube):
    """
    Per-band min-max normalization (mat2gray) applied to the 3-D cube.

    Args:
        data_cube (np.ndarray): Shape (H, W, B)

    Returns:
        np.ndarray: Normalized cube of the same shape, dtype float32.
    """
    print("Normalizing spectral bands (mat2gray)...")
    cube = data_cube.astype(np.float32)

    # Compute per-band stats over spatial axes (axis=(0,1))
    band_mins = cube.min(axis=(0, 1), keepdims=True)   # (1, 1, B)
    band_maxs = cube.max(axis=(0, 1), keepdims=True)   # (1, 1, B)

    band_ranges = band_maxs - band_mins
    band_ranges[band_ranges == 0] = 1e-8

    cube = (cube - band_mins) / band_ranges
    return cube


def pad_cube(data_cube, pad_size):
    """
    Mirror-pad the spatial dimensions of the cube so that edge pixels
    can still have a full patch_size x patch_size neighbourhood.

    Args:
        data_cube (np.ndarray): Shape (H, W, B)
        pad_size  (int):        Number of pixels to pad on each side.

    Returns:
        np.ndarray: Padded cube of shape (H+2*pad, W+2*pad, B).
    """
    return np.pad(
        data_cube,
        ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
        mode='reflect'
    )


def extract_patches(padded_cube, gt_map, patch_size):
    """
    Extract a spatial patch centred on every labeled pixel.

    Args:
        padded_cube (np.ndarray): Mirror-padded cube (H+2p, W+2p, B).
        gt_map      (np.ndarray): Original ground-truth map (H, W).
        patch_size  (int):        Side length of square patch (must be odd).

    Returns:
        patches (np.ndarray): Shape (N_labeled, patch_size, patch_size, B)
        labels  (np.ndarray): Shape (N_labeled,) -- integer class indices 0..C-1
    """
    print(f"Extracting {patch_size}x{patch_size} spatial patches around labeled pixels...")
    half       = patch_size // 2
    rows, cols = gt_map.shape
    patches, labels = [], []

    for r in range(rows):
        for c in range(cols):
            label = gt_map[r, c]
            if label == 0:      # skip background
                continue
            # Padded coordinates: original (r,c) -> padded (r+half, c+half)
            pr, pc = r + half, c + half
            patch = padded_cube[pr - half: pr + half + 1,
                                pc - half: pc + half + 1, :]  # (P, P, B)
            patches.append(patch)
            labels.append(int(label) - 1)   # convert 1-based -> 0-based

    patches = np.array(patches, dtype=np.float32)   # (N, P, P, B)
    labels  = np.array(labels,  dtype=np.int64)     # (N,)
    print(f"Total labeled patches extracted: {len(patches)}")
    return patches, labels


def one_hot_encode_labels(labels, num_classes=16):
    """One-hot encode integer class labels."""
    print(f"One-hot encoding labels into {num_classes} categorical channels...")
    return np.eye(num_classes, dtype=np.float32)[labels]


def split_data(patches, labels, test_size=0.5, random_state=42):
    """Stratified train/test split on extracted patches."""
    print("Splitting dataset (stratified)...")
    Train_X, Test_X, TrLabel, TeLabel = train_test_split(
        patches, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    print(f"Train samples: {len(Train_X)} | Test samples: {len(Test_X)}")
    return Train_X, Test_X, TrLabel, TeLabel


# ---------------------------------------------------------------------------
# Main pipeline execution
# ---------------------------------------------------------------------------

# 1. Load raw data
raw_data, ground_truth = load_mat_data(DATA_PATH, GT_PATH)

# 2. Normalize cube
cube_normalized = mat2gray_normalize(raw_data)

# 3. Mirror-pad for patch extraction
pad_size    = PATCH_SIZE // 2
padded_cube = pad_cube(cube_normalized, pad_size)

# 4. Extract spatial patches for every labeled pixel
patches, labels = extract_patches(padded_cube, ground_truth, PATCH_SIZE)

# 5. Stratified split
Train_X, Test_X, TrLabel, TeLabel = split_data(patches, labels, test_size=0.5)

# 6. One-hot encode split labels
TrLabel_encoded = one_hot_encode_labels(TrLabel, num_classes=NUM_CLASSES)
TeLabel_encoded = one_hot_encode_labels(TeLabel, num_classes=NUM_CLASSES)

# 7. Reshape for 3D CNN input: PyTorch Conv3d expects (N, C, D, H, W)
#    We treat: C=1 (single channel), D=B (spectral bands as depth), H=P, W=P
#    Current shape from extraction: (N, P, P, B)
#    Step 1: transpose to (N, B, P, P)
#    Step 2: unsqueeze channel dim -> (N, 1, B, P, P)
Train_X_ready = np.transpose(Train_X, (0, 3, 1, 2))[:, np.newaxis, ...]  # (N, 1, B, P, P)
Test_X_ready  = np.transpose(Test_X,  (0, 3, 1, 2))[:, np.newaxis, ...]  # (N, 1, B, P, P)

print("\nSaving finalized arrays to extracted_3d folder...")
np.save(os.path.join(OUTPUT_DIR, 'Train_X.npy'),  Train_X_ready)
np.save(os.path.join(OUTPUT_DIR, 'TrLabel.npy'),  TrLabel_encoded)
np.save(os.path.join(OUTPUT_DIR, 'Test_X.npy'),   Test_X_ready)
np.save(os.path.join(OUTPUT_DIR, 'TeLabel.npy'),  TeLabel_encoded)

print(f"\nFinal shapes:")
print(f"  Train_X : {Train_X_ready.shape}  (N, 1, Bands, {PATCH_SIZE}, {PATCH_SIZE})")
print(f"  TrLabel : {TrLabel_encoded.shape}")
print(f"  Test_X  : {Test_X_ready.shape}")
print(f"  TeLabel : {TeLabel_encoded.shape}")
print("\nPipeline Ran successfully")
