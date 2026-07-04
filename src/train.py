import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from utils import setup_logger, log_experiment
import scipy.io as sio
import matplotlib.pyplot as plt

class Net(nn.Module):
    """1D CNN model class to learn spectral features."""
    def __init__(self): 
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(200, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(128, momentum=0.1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(128, 16, kernel_size=1)

        # Weight initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv1.bias.data.zero_()
        self.conv2.bias.data.zero_()

    def forward(self, x):
        """Forward pass through the network."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

def initialize_parameters():
    """
    Initialize the model with seed 1 for reproducibility.

    Returns:
        Net: Initialized model instance
    """
    torch.manual_seed(1)
    model = Net()
    return model
    
class dataset(Dataset):   
    def __init__(self, data_dir='data/extracted', is_train=True, transform=None, logger=None):
        self.data_dir = data_dir
        self.transform = transform
        self.logger = logger
        
        if is_train:
            data_path = os.path.join(data_dir, 'Train_X.npy')
            label_path = os.path.join(data_dir, 'TrLabel.npy')
        else:
            data_path = os.path.join(data_dir, 'Test_X.npy')
            label_path = os.path.join(data_dir, 'TeLabel.npy')
            
        if self.logger:
            self.logger.info(f"Loading dataset from {data_path}")
        
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]  # Shape: (200, 1)
        y = self.labels[idx]  # Shape: (16,)
        
        # Convert spectral bands to float tensor and unsqueeze to shape (200, 1, 1) for Conv2d (C, H, W)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        
        # Convert one-hot encoded label to class index (scalar integer) for CrossEntropyLoss
        y_label = torch.tensor(y.argmax(), dtype=torch.long)
        
        if self.transform:
            x_tensor = self.transform(x_tensor)
            
        return x_tensor, y_label

def train_models(model, train_loader, criterion, optimizer, device, epochs=50, logger=None):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: shape returned is (batch_size, 16, 1, 1), squeeze to (batch_size, 16)
            outputs = model(inputs).squeeze(-1).squeeze(-1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        msg = f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)

def evaluate_models(model, test_loader, device, logger=None):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze(-1).squeeze(-1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    msg = f"\nEvaluation Results:\n" \
          f"Accuracy:  {accuracy:.4f}\n" \
          f"Precision: {precision:.4f}\n" \
          f"Recall:    {recall:.4f}\n" \
          f"F1-Score:  {f1:.4f}\n"
          
    if logger:
        logger.info(msg)
        logger.info(f"Confusion Matrix:\n{cm}")
    else:
        print(msg)
        print(f"Confusion Matrix:\n{cm}")
    
    return {'accuracy': round(float(accuracy), 4), 'precision': round(float(precision), 4), 'recall': round(float(recall), 4), 'f1': round(float(f1), 4)}

def plot_predictions(model, device, data_path='data/raw/Indian_pines_corrected.mat', gt_path='data/raw/Indian_pines_gt.mat', output_plot_path='results/plots/segmentation_comparison.png', logger=None):
    model.eval()
    
    # 1. Load raw .mat data
    data_mat = sio.loadmat(data_path)
    gt_mat = sio.loadmat(gt_path)
    
    raw_data = data_mat['indian_pines_corrected']  # (145, 145, 200)
    ground_truth = gt_mat['indian_pines_gt']      # (145, 145)
    
    rows, cols, bands = raw_data.shape
    total_pixels = rows * cols
    
    # 2. Reshape and Normalize
    matrix_2d = np.reshape(raw_data, (total_pixels, bands))
    
    matrix_normalized = matrix_2d.astype(np.float32)
    band_mins = matrix_normalized.min(axis=0)
    band_maxs = matrix_normalized.max(axis=0)
    band_ranges = band_maxs - band_mins
    band_ranges[band_ranges == 0] = 1e-8
    matrix_normalized = (matrix_normalized - band_mins) / band_ranges
    
    # 3. Add dimension for 1D CNN: (total_pixels, 200, 1, 1)
    x_tensor = torch.tensor(matrix_normalized, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
    
    # 4. Predict in batches
    predictions = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, total_pixels, batch_size):
            batch_x = x_tensor[i:i+batch_size].to(device)
            outputs = model(batch_x).squeeze(-1).squeeze(-1)
            preds = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
            
    predictions = np.array(predictions)
    
    # 5. Convert back to 1..16, set background to 0
    predicted_map = (predictions + 1).reshape((rows, cols))
    predicted_map[ground_truth == 0] = 0
    
    # 6. Create custom ListedColormap for discrete class coloring (0 is black, 1-16 are distinct colors)
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
        '#ADFF2F'   # 16: Stone-Steel-Towers
    ]
    cmap = ListedColormap(class_colors)

    # 7. Generate False/True Color Composite (RGB) from hyperspectral bands
    # We extract bands 29 (Red), 19 (Green), and 9 (Blue) and normalize to [0, 1]
    r = raw_data[:, :, 29].astype(np.float32)
    g = raw_data[:, :, 19].astype(np.float32)
    b = raw_data[:, :, 9].astype(np.float32)
    
    r = (r - r.min()) / (r.max() - r.min() + 1e-8)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    b = (b - b.min()) / (b.max() - b.min() + 1e-8)
    rgb_img = np.dstack((r, g, b))

    # 8. Plot 3-panel visual
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(rgb_img)
    axes[0].set_title("True/False Color Composite (RGB)")
    axes[0].axis('off')
    
    axes[1].imshow(ground_truth, cmap=cmap)
    axes[1].set_title("Ground Truth Classes")
    axes[1].axis('off')
    
    axes[2].imshow(predicted_map, cmap=cmap)
    axes[2].set_title("1D CNN Predicted Classes")
    axes[2].axis('off')
    
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    
    msg = f"🖼️ 3-panel plot saved to {output_plot_path}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

def main():
    # --- Hyperparameters (edit these to run experiments) ---
    EPOCHS     = 100
    LR         = 0.001
    BATCH_SIZE = 64
    DROPOUT    = 0.2
    TEST_SPLIT = 0.5
    # --------------------------------------------------------
    # Setup logging using helper function
    logger = setup_logger(log_dir='results', log_filename='training.log')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load datasets
    train_dataset = dataset(
        data_dir='data/extracted',
        is_train=True,
        transform=None,
        logger=logger
    )
    
    test_dataset = dataset(
        data_dir='data/extracted',
        is_train=False,
        transform=None,
        logger=logger
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model, criterion with class weights to handle imbalance, and optimizer
    labels = train_dataset.labels.argmax(axis=1)
    class_counts = np.bincount(labels, minlength=16)
    class_counts[class_counts == 0] = 1
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * 16
    class_weights = torch.FloatTensor(weights).to(device)
    
    model = initialize_parameters().to(device)
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
        'epochs':     EPOCHS,
        'lr':         LR,
        'batch_size': BATCH_SIZE,
        'dropout':    DROPOUT,
        'test_split': TEST_SPLIT,
        'train_samples': len(train_dataset),
        'test_samples':  len(test_dataset),
    }
    csv_path = log_experiment(params, metrics, log_dir='results')
    logger.info(f"📊 Experiment results logged to {csv_path}")
    
    # Plot and save segmented images
    logger.info("Generating segmentation plots...")
    plot_predictions(model, device, logger=logger)

if __name__ == '__main__':
    main()
