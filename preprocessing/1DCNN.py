import numpy as np
import scipy.io as sio
import os
from sklearn.model_selection import train_test_split

# change data path for different datasets
DATA_PATH = 'data/raw/Indian_pines_corrected.mat'
GT_PATH = 'data/raw/Indian_pines_gt.mat'
OUTPUT_DIR = 'data/extracted'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mat_data(data_path, gt_path):
    print("Loading .mat files from Source...")
    data_mat = sio.loadmat(data_path)
    gt_mat = sio.loadmat(gt_path)

    raw_data = data_mat['indian_pines_corrected']
    ground_truth = gt_mat['indian_pines_gt']

    print(f"Raw Data Shape: {raw_data.shape} | Ground Truth Shape: {ground_truth.shape}")
    return raw_data, ground_truth

def remove_noise_bands(data_cube, noise_band_indices):
    # Find the total number of bands currently in the cube
    total_bands = data_cube.shape[2]

    # Generate an array of all band indices: [0, 1, 2, ..., total_bands-1]
    all_bands = np.arange(total_bands)

    # Keep indices that are NOT present in the noise_band_indices list
    clean_band_indices = np.isin(all_bands, noise_band_indices, invert=True)

    # Slice the cube along the 3rd axis (axis=2 represents the spectral bands)
    cleaned_cube = data_cube[:, :, clean_band_indices]

    print(f"Original bands: {total_bands} | Removed: {len(noise_band_indices)} | Clean bands kept: {cleaned_cube.shape[2]}")
    return cleaned_cube

def flatten_cube(data_cube, gt_map):
    print("Flattening 3D cube dimensions into 2D spectral rows...")
    rows, cols, bands = data_cube.shape
    total_pixels = rows * cols

    matrix_2d = np.reshape(data_cube, (total_pixels, bands))
    labels_1d = np.reshape(gt_map, (total_pixels,))
    return matrix_2d, labels_1d

def mat2gray_normalize(matrix_2d):
    print("Normalizing spectral bands (mat2gray)...")
    matrix_normalized = matrix_2d.astype(np.float32)
    band_mins = matrix_normalized.min(axis=0)
    band_maxs = matrix_normalized.max(axis=0)

    band_ranges = band_maxs - band_mins
    band_ranges[band_ranges == 0] = 1e-8

    matrix_normalized = (matrix_normalized - band_mins) / band_ranges
    return matrix_normalized

def one_hot_encode_labels(labels, num_classes=16):
    print(f"One-hot encoding labels into {num_classes} categorical channels...")
    return np.eye(num_classes)[labels]

def extract_and_split_data(X_normalized, labels_1d, test_size=0.8, random_state=42):
    print("Extracting labeled pixels and splitting dataset...")
    labeled_mask = labels_1d > 0
    X_labeled = X_normalized[labeled_mask]
    y_labeled = labels_1d[labeled_mask]

    # Convert 1-16 labels to 0-15 labels
    y_labeled = y_labeled - 1

    Train_X, Test_X, TrLabel, TeLabel = train_test_split(
        X_labeled, y_labeled, test_size=test_size, stratify=y_labeled, random_state=random_state
    )
    return Train_X, Test_X, TrLabel, TeLabel

# 1. Run Data Loader
raw_data, ground_truth = load_mat_data(DATA_PATH, GT_PATH)

# 2. Reshape
matrix_2D, labels_1D = flatten_cube(raw_data, ground_truth)

# 3. Scale Features
X_normalized = mat2gray_normalize(matrix_2D)

# 4. Filter Background & Split
Train_X, Test_X, TrLabel, TeLabel = extract_and_split_data(X_normalized, labels_1D, test_size=0.5)

# 5. Categorical Encoding
TrLabel_encoded = one_hot_encode_labels(TrLabel, num_classes=16)
TeLabel_encoded = one_hot_encode_labels(TeLabel, num_classes=16)

# 6. Reshape for 1D CNN: (Samples, Channels, Spectral_Length)
Train_X_ready = np.expand_dims(Train_X, axis=1)
Test_X_ready = np.expand_dims(Test_X, axis=1)

print("\nSaving finalized arrays to sample_data folder...")
np.save(os.path.join(OUTPUT_DIR, 'Train_X.npy'), Train_X_ready)
np.save(os.path.join(OUTPUT_DIR, 'TrLabel.npy'), TrLabel_encoded)
np.save(os.path.join(OUTPUT_DIR, 'Test_X.npy'), Test_X_ready)
np.save(os.path.join(OUTPUT_DIR, 'TeLabel.npy'), TeLabel_encoded)

print("\nPipeline Ran successfully")