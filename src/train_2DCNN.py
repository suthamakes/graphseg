import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import logging
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import scipy.io as sio
import matplotlib.pyplot as plt

# Allow importing shared utilities from src/
sys.path.insert(0, os.path.dirname(__file__))
from utils import setup_logger, log_experiment


class Net2D(nn.Module):
    """
    2D CNN model for spatial-spectral hyperspectral image classification.

    Input shape : (N, B, P, P)  where B = spectral bands, P = patch size.

    Architecture
    ------------
    Block 1 : Conv2d(B, 64, 3x3) -> BN -> ReLU -> Dropout
    Block 2 : Conv2d(64, 128, 3x3) -> BN -> ReLU -> Dropout
    Head    : GlobalAveragePool -> Linear(128, num_classes)
    """

    def __init__(self, in_channels=200, num_classes=16, dropout=0.4):
        super(Net2D, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

        # Global average pool collapses (N, 128, P, P) -> (N, 128)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, num_classes)

        # Weight initialisation (Xavier uniform, zero bias)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass. x: (N, B, P, P) -> logits: (N, num_classes)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x)            # (N, 128, 1, 1)
        x = x.flatten(start_dim=1) # (N, 128)
        x = self.head(x)           # (N, num_classes)
        return x


def initialize_parameters(in_channels=200, num_classes=16, dropout=0.4):
    """
    Initialise the 2D CNN with a fixed seed for reproducibility.

    Returns:
        Net2D: Initialized model instance.
    """
    torch.manual_seed(1)
    model = Net2D(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    return model


class dataset(Dataset):
    """PyTorch Dataset for 2D CNN patch-based HSI classification."""

    def __init__(self, data_dir='data/extracted_2d', is_train=True, transform=None, logger=None):
        self.data_dir  = data_dir
        self.transform = transform
        self.logger    = logger

        if is_train:
            data_path  = os.path.join(data_dir, 'Train_X.npy')
            label_path = os.path.join(data_dir, 'TrLabel.npy')
        else:
            data_path  = os.path.join(data_dir, 'Test_X.npy')
            label_path = os.path.join(data_dir, 'TeLabel.npy')

        if self.logger:
            self.logger.info(f"Loading dataset from {data_path}")

        self.data   = np.load(data_path)    # (N, B, P, P)
        self.labels = np.load(label_path)   # (N, num_classes) one-hot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]     # (B, P, P) — already channel-first
        y = self.labels[idx]   # (num_classes,) one-hot

        x_tensor = torch.tensor(x, dtype=torch.float32)

        # Convert one-hot label to class index (scalar) for CrossEntropyLoss
        y_label = torch.tensor(y.argmax(), dtype=torch.long)

        if self.transform:
            x_tensor = self.transform(x_tensor)

        return x_tensor, y_label


def train_models(model, train_loader, criterion, optimizer, device, epochs=50, logger=None):
    """Training loop with per-epoch loss and accuracy reporting."""
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct      = 0
        total        = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)           # (N, num_classes)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted  = torch.max(outputs, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc  = correct / total

        msg = f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)


def evaluate_models(model, test_loader, device, logger=None):
    """Evaluation with accuracy, precision, recall, F1 and confusion matrix."""
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs  = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    cm       = confusion_matrix(all_labels, all_preds)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

    msg = (
        f"\nEvaluation Results:\n"
        f"Accuracy:  {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1-Score:  {f1:.4f}\n"
    )

    if logger:
        logger.info(msg)
        logger.info(f"Confusion Matrix:\n{cm}")
    else:
        print(msg)
        print(f"Confusion Matrix:\n{cm}")

    return {
        'accuracy':  round(float(accuracy),  4),
        'precision': round(float(precision), 4),
        'recall':    round(float(recall),    4),
        'f1':        round(float(f1),        4),
    }


def plot_predictions(
    model,
    device,
    patch_size=5,
    data_path='data/raw/Indian_pines_corrected.mat',
    gt_path='data/raw/Indian_pines_gt.mat',
    output_plot_path='results/plots/segmentation_comparison_2d.png',
    logger=None,
):
    """
    Run full-image inference with the trained 2D CNN and save a 3-panel plot:
      [False-colour composite | Ground Truth | 2D CNN Predicted Classes].
    """
    model.eval()

    # 1. Load raw .mat data
    data_mat     = sio.loadmat(data_path)
    gt_mat       = sio.loadmat(gt_path)
    raw_data     = data_mat['indian_pines_corrected']   # (H, W, B)
    ground_truth = gt_mat['indian_pines_gt']            # (H, W)

    rows, cols, bands = raw_data.shape

    # 2. Normalize cube (per-band mat2gray)
    cube = raw_data.astype(np.float32)
    band_mins   = cube.min(axis=(0, 1), keepdims=True)
    band_maxs   = cube.max(axis=(0, 1), keepdims=True)
    band_ranges = band_maxs - band_mins
    band_ranges[band_ranges == 0] = 1e-8
    cube = (cube - band_mins) / band_ranges   # (H, W, B)

    # 3. Mirror-pad for patch extraction
    half        = patch_size // 2
    padded_cube = np.pad(
        cube,
        ((half, half), (half, half), (0, 0)),
        mode='reflect'
    )

    # 4. Build all-pixel patch tensor (H*W, B, P, P)
    all_patches = []
    for r in range(rows):
        for c in range(cols):
            pr, pc = r + half, c + half
            patch = padded_cube[pr - half: pr + half + 1,
                                pc - half: pc + half + 1, :]   # (P, P, B)
            all_patches.append(patch.transpose(2, 0, 1))       # (B, P, P)

    all_patches = np.array(all_patches, dtype=np.float32)      # (H*W, B, P, P)
    x_tensor    = torch.tensor(all_patches)

    # 5. Predict in batches
    predictions = []
    batch_size  = 512
    with torch.no_grad():
        for i in range(0, len(x_tensor), batch_size):
            batch   = x_tensor[i: i + batch_size].to(device)
            outputs = model(batch)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)

    predictions = np.array(predictions)

    # 6. Reshape to spatial map; convert back to 1-indexed, mask background
    predicted_map = (predictions + 1).reshape((rows, cols))
    predicted_map[ground_truth == 0] = 0

    # 7. Discrete colormap (0=background, 1-16=classes)
    from matplotlib.colors import ListedColormap
    class_colors = [
        '#000000',  # 0: Background
        '#FF0000',  # 1: Alfalfa
        '#00FF00',  # 2: Corn-notill
        '#0000FF',  # 3: Corn-mintill
        '#FFFF00',  # 4: Corn
        '#FF00FF',  # 5: Grass-pasture
        '#00FFFF',  # 6: Grass-trees
        '#FFA500',  # 7: Grass-pasture-mowed
        '#800080',  # 8: Hay-windrowed
        '#A52A2A',  # 9: Oats
        '#808080',  # 10: Soybean-notill
        '#FFC0CB',  # 11: Soybean-mintill
        '#E6E6FA',  # 12: Soybean-clean
        '#008080',  # 13: Wheat
        '#FFD700',  # 14: Woods
        '#4B0082',  # 15: Buildings-Grass-Trees-Drives
        '#ADFF2F',  # 16: Stone-Steel-Towers
    ]
    cmap = ListedColormap(class_colors)

    # 8. False-colour composite (bands 29, 19, 9 -> R, G, B)
    r = raw_data[:, :, 29].astype(np.float32)
    g = raw_data[:, :, 19].astype(np.float32)
    b = raw_data[:, :, 9].astype(np.float32)

    r = (r - r.min()) / (r.max() - r.min() + 1e-8)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    b = (b - b.min()) / (b.max() - b.min() + 1e-8)
    rgb_img = np.dstack((r, g, b))

    # 9. 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(rgb_img)
    axes[0].set_title("True/False Color Composite (RGB)")
    axes[0].axis('off')

    axes[1].imshow(ground_truth, cmap=cmap)
    axes[1].set_title("Ground Truth Classes")
    axes[1].axis('off')

    axes[2].imshow(predicted_map, cmap=cmap)
    axes[2].set_title("2D CNN Predicted Classes")
    axes[2].axis('off')

    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    msg = f"3-panel plot saved to {output_plot_path}"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def main():
    # --- Hyperparameters (edit these to run experiments) ---
    EPOCHS     = 100
    LR         = 0.001
    BATCH_SIZE = 64
    DROPOUT    = 0.4
    TEST_SPLIT = 0.5
    PATCH_SIZE = 5
    IN_CHANNELS = 200
    NUM_CLASSES = 16
    # --------------------------------------------------------

    # Setup logging using shared helper function
    logger = setup_logger(log_dir='results', log_filename='training_2d.log')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load datasets
    train_dataset = dataset(
        data_dir='data/extracted_2d',
        is_train=True,
        transform=None,
        logger=logger,
    )

    test_dataset = dataset(
        data_dir='data/extracted_2d',
        is_train=False,
        transform=None,
        logger=logger,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # Class-weighted loss to handle imbalanced classes
    labels       = train_dataset.labels.argmax(axis=1)
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_counts[class_counts == 0] = 1
    weights       = 1.0 / class_counts
    weights       = weights / weights.sum() * NUM_CLASSES
    class_weights = torch.FloatTensor(weights).to(device)

    # Initialize model, loss, optimizer
    model     = initialize_parameters(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train
    logger.info("Starting training...")
    train_models(model, train_loader, criterion, optimizer, device, epochs=EPOCHS, logger=logger)

    # Evaluate
    logger.info("Evaluating model...")
    metrics = evaluate_models(model, test_loader, device, logger=logger)

    # Log experiment results to CSV
    params = {
        'epochs':        EPOCHS,
        'lr':            LR,
        'batch_size':    BATCH_SIZE,
        'dropout':       DROPOUT,
        'test_split':    TEST_SPLIT,
        'patch_size':    PATCH_SIZE,
        'train_samples': len(train_dataset),
        'test_samples':  len(test_dataset),
    }
    csv_path = log_experiment(params, metrics, log_dir='results', csv_filename='experiments_2d.csv')
    logger.info(f"Experiment results logged to {csv_path}")

    # Plot and save segmentation maps
    logger.info("Generating segmentation plots...")
    plot_predictions(
        model,
        device,
        patch_size=PATCH_SIZE,
        output_plot_path='results/plots/segmentation_comparison_2d.png',
        logger=logger,
    )


if __name__ == '__main__':
    main()
