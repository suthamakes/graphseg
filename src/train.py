"""
src/train.py
============
Unified training entry-point for all HSI classification models.

Usage
-----
    # CNN models (require offline preprocessing first)
    python src/train.py --model cnn1d
    python src/train.py --model cnn2d
    python src/train.py --model cnn3d

    # GCN model (graph built at runtime, no offline step needed)
    python src/train.py --model gcn --config configs/gcn_config.yaml

Offline preprocessing (run once before CNN training)
-----------------------------------------------------
    python preprocessing/1DCNN.py    # → data/extracted/
    python preprocessing/2DCNN.py    # → data/extracted_2d/
    python preprocessing/3DCNN.py    # → data/extracted_3d/
"""

import os
import sys
import time
import json
import yaml
import pickle
import random
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Add project root to path so imports work regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn1d import Net, initialize_parameters as init_cnn1d
from models.cnn2d import Net2D, initialize_parameters as init_cnn2d
from models.cnn3d import Net3D, initialize_parameters as init_cnn3d
from models.gcn   import GCN, normalize_adjacency, compute_loss

from preprocessing        import load_dataset
from preprocessing.GCN   import segment, build_graph, partition_graph
from src.dataset          import HSIDataset, split_data, propagate_labels_to_nodes
from src.evaluate         import evaluate_metrics
from src.utils            import setup_logger, log_experiment


# ---------------------------------------------------------------------------
# Shared colormap for all segmentation plots
# ---------------------------------------------------------------------------
_CLASS_COLORS = [
    '#000000',  # 0:  Background
    '#FF0000',  # 1:  Alfalfa
    '#00FF00',  # 2:  Corn-notill
    '#0000FF',  # 3:  Corn-mintill
    '#FFFF00',  # 4:  Corn
    '#FF00FF',  # 5:  Grass-pasture
    '#00FFFF',  # 6:  Grass-trees
    '#FFA500',  # 7:  Grass-pasture-mowed
    '#800080',  # 8:  Hay-windrowed
    '#A52A2A',  # 9:  Oats
    '#808080',  # 10: Soybean-notill
    '#FFC0CB',  # 11: Soybean-mintill
    '#E6E6FA',  # 12: Soybean-clean
    '#008080',  # 13: Wheat
    '#FFD700',  # 14: Woods
    '#4B0082',  # 15: Buildings-Grass-Trees-Drives
    '#ADFF2F',  # 16: Stone-Steel-Towers
]
CMAP = ListedColormap(_CLASS_COLORS)


# ===========================================================================
# Section 1 — Model factory
# ===========================================================================

def get_model(model_name: str, config: dict, device: torch.device) -> nn.Module:
    """
    Instantiate and return the requested model on the given device.

    Parameters
    ----------
    model_name : str    one of 'cnn1d', 'cnn2d', 'cnn3d', 'gcn'
    config     : dict   hyperparameters (from CLI args or YAML)
    device     : torch.device

    Returns
    -------
    nn.Module
    """
    if model_name == "cnn1d":
        model = init_cnn1d(
            in_channels=1,
            num_classes=config.get("num_classes") or 16,
            dropout=config.get("dropout") or 0.2,
        )

    elif model_name == "cnn2d":
        model = init_cnn2d(
            in_channels=config.get("in_channels") or 200,
            num_classes=config.get("num_classes") or 16,
            dropout=config.get("dropout") or 0.4,
        )

    elif model_name == "cnn3d":
        model = init_cnn3d(
            in_channels=1,
            num_classes=config.get("num_classes") or 16,
            dropout=config.get("dropout") or 0.4,
        )

    elif model_name == "gcn":
        model = GCN(
            in_features=config["pca_components"],
            hidden_features=config["hidden_units"],
            num_classes=config["num_classes"],
            dropout=config.get("dropout", 0.5),
        )

    else:
        raise ValueError(f"Unknown model: '{model_name}'. Choose from cnn1d, cnn2d, cnn3d, gcn.")

    return model.to(device)


# ===========================================================================
# Section 2 — Data loading
# ===========================================================================

def get_cnn_data(model_name: str, config: dict, logger: logging.Logger):
    """
    Load preprocessed .npy arrays for CNN models and return DataLoaders.

    Parameters
    ----------
    model_name : str   'cnn1d', 'cnn2d', or 'cnn3d'
    config     : dict  must contain 'batch_size' and 'num_classes'
    logger     : logging.Logger

    Returns
    -------
    train_loader, test_loader, num_classes, class_weights (torch.FloatTensor)
    """
    data_dirs = {
        "cnn1d": "data/extracted",
        "cnn2d": "data/extracted_2d",
        "cnn3d": "data/extracted_3d",
    }
    data_dir    = data_dirs[model_name]
    num_classes = config.get("num_classes", 16)
    batch_size  = config.get("batch_size", 64)

    train_dataset = HSIDataset(data_dir=data_dir, is_train=True,  logger=logger)
    test_dataset  = HSIDataset(data_dir=data_dir, is_train=False, logger=logger)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # Class-weighted loss to handle imbalanced distribution
    labels       = train_dataset.labels.argmax(axis=1)
    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts[class_counts == 0] = 1
    weights      = 1.0 / class_counts
    weights      = weights / weights.sum() * num_classes
    class_weights = torch.FloatTensor(weights)

    logger.info(f"Loaded {len(train_dataset)} train / {len(test_dataset)} test samples from {data_dir}")
    return train_loader, test_loader, num_classes, class_weights


def get_gcn_data(config: dict, seed: int, device: torch.device, run_dir: str, logger: logging.Logger):
    """
    Build the superpixel graph at runtime and return partitioned sub-graphs
    along with pixel-level splits.

    Parameters
    ----------
    config   : dict           GCN YAML config for the dataset
    seed     : int
    device   : torch.device
    run_dir  : str            directory for saving intermediate artefacts
    logger   : logging.Logger

    Returns
    -------
    pt_subgraphs  : list[dict]   PyTorch-ready sub-graph dicts
    node_train    : np.ndarray   boolean node train mask
    node_val      : np.ndarray   boolean node val mask
    gt            : np.ndarray   ground-truth map (H, W)
    node_to_pixel_indices : dict
    N             : int          number of superpixel nodes
    num_classes   : int
    """
    # 1. Load raw data
    cube, gt, metadata = load_dataset(config.get("dataset", "indian_pines"))
    num_classes = metadata["num_classes"]

    np.save(os.path.join(run_dir, "cube.npy"), cube)
    np.save(os.path.join(run_dir, "gt.npy"),   gt)

    # 2. Superpixel segmentation
    superpixel_map = segment(cube, config["n_segments"], config["compactness"])
    np.save(os.path.join(run_dir, "superpixel_map.npy"), superpixel_map)

    # 3. Graph construction
    X, A_sparse, superpixel_map, node_to_pixel_indices, node_labels = build_graph(
        cube, superpixel_map, gt,
        pca_components=config["pca_components"],
        top_k=config["top_k"],
        hops=config["hops"],
    )
    with open(os.path.join(run_dir, "node_to_pixel_indices.pkl"), "wb") as f:
        pickle.dump(node_to_pixel_indices, f)

    N = X.shape[0]
    logger.info(f"Graph: {N} superpixel nodes")

    # 4. Pixel-level splits → propagate to nodes
    train_mask, val_mask, test_mask = split_data(gt, num_classes=num_classes, seed=seed)
    node_train, node_val, node_test = propagate_labels_to_nodes(
        node_to_pixel_indices, train_mask, val_mask, test_mask, gt
    )

    # 5. Graph partitioning
    sub_graphs, edge_cut_ratio = partition_graph(
        A_sparse, X, config["num_clusters"], node_labels
    )

    # 6. Convert to PyTorch tensors
    pt_subgraphs = []
    for sg in sub_graphs:
        norm_A     = normalize_adjacency(sg["A"]).to(device)
        pt_X       = torch.FloatTensor(sg["X"]).to(device)
        pt_Y       = torch.LongTensor(sg["Y"]).to(device)
        V_i        = sg["V"]
        mask_train = torch.BoolTensor(node_train[V_i]).to(device)
        mask_val   = torch.BoolTensor(node_val[V_i]).to(device)
        pt_subgraphs.append({
            "V_i":        V_i,
            "A":          norm_A,
            "X":          pt_X,
            "Y":          pt_Y,
            "train_mask": mask_train,
            "val_mask":   mask_val,
        })

    return pt_subgraphs, node_train, node_val, node_test, test_mask, gt, node_to_pixel_indices, N, num_classes, edge_cut_ratio


# ===========================================================================
# Section 3 — Training loops
# ===========================================================================

def train_cnn(model, model_name, train_loader, criterion, optimizer, device, epochs, logger):
    """
    Standard mini-batch training loop for CNN models.

    Reports loss and accuracy every epoch.
    """
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 2D/3D CNN data may need an extra dim; Conv1d data is already (N, 1, B)
            if inputs.dim() == 3 and model_name != "cnn1d":
                inputs = inputs.unsqueeze(-1)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Old Conv2d 1D CNN output was (N, C, 1, 1); squeeze to (N, C)
            if outputs.dim() == 4:
                outputs = outputs.squeeze(-1).squeeze(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted  = torch.max(outputs, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {running_loss/total:.4f}  "
            f"Acc: {correct/total:.4f}"
        )


def train_gcn(model, pt_subgraphs, optimizer, config, logger):
    """
    Mini-cluster training loop for the GCN model.

    Each step randomly samples one sub-graph and backpropagates
    the cross-entropy loss on labeled nodes only.

    Returns loss_history and val_acc_history for later logging.
    """
    epochs           = config["epochs"]
    steps_per_epoch  = 5 * config["num_clusters"]

    loss_history     = []
    val_acc_history  = []

    logger.info("Starting GCN training loop...")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            sg = random.choice(pt_subgraphs)

            optimizer.zero_grad()
            logits = model(sg["X"], sg["A"])
            loss   = compute_loss(logits, sg["Y"], sg["train_mask"])

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= steps_per_epoch
        loss_history.append(epoch_loss)

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for sg in pt_subgraphs:
                if sg["val_mask"].sum() > 0:
                    logits = model(sg["X"], sg["A"])
                    preds  = logits.argmax(dim=1) + 1  # shift to 1-indexed
                    correct += (preds[sg["val_mask"]] == sg["Y"][sg["val_mask"]]).sum().item()
                    total   += sg["val_mask"].sum().item()

        val_acc = correct / max(total, 1)
        val_acc_history.append(val_acc)
        model.train()

        if (epoch + 1) % 50 == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

    return loss_history, val_acc_history


# ===========================================================================
# Section 4 — Evaluation
# ===========================================================================

def evaluate_cnn(model, model_name, test_loader, device, logger):
    """
    Evaluate a CNN model on the test DataLoader.

    Returns a metrics dict with accuracy, precision, recall, f1.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            # 2D/3D CNN data may need an extra dim; Conv1d data is already (N, 1, B)
            if inputs.dim() == 3 and model_name != "cnn1d":
                inputs = inputs.unsqueeze(-1)
            outputs = model(inputs)
            if outputs.dim() == 4:
                outputs = outputs.squeeze(-1).squeeze(-1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    cm       = confusion_matrix(all_labels, all_preds)

    logger.info(
        f"\nEvaluation Results:\n"
        f"  Accuracy : {accuracy:.4f}\n"
        f"  Precision: {precision:.4f}\n"
        f"  Recall   : {recall:.4f}\n"
        f"  F1-Score : {f1:.4f}\n"
    )
    logger.info(f"Confusion Matrix:\n{cm}")

    return {
        "accuracy":  round(float(accuracy),  4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall),    4),
        "f1":        round(float(f1),        4),
    }


def evaluate_gcn(model, pt_subgraphs, gt, node_to_pixel_indices, pixel_test_mask, N, num_classes, logger):
    """
    Evaluate the GCN model using pixel-level OA, AA, and Kappa.

    Propagates node predictions to pixels, then computes metrics on the
    held-out test pixels only (pixel_test_mask is the boolean (H, W) mask
    from split_data, NOT the node-level mask).

    Returns a metrics dict with OA, AA, Kappa, PerClass.
    """
    model.eval()
    node_predictions = np.zeros(N, dtype=np.int32)

    with torch.no_grad():
        for sg in pt_subgraphs:
            logits = model(sg["X"], sg["A"])
            preds  = logits.argmax(dim=1).cpu().numpy() + 1  # 1-indexed
            node_predictions[sg["V_i"]] = preds

    # Map node predictions back to pixels
    H, W     = gt.shape
    pred_map = np.zeros((H, W), dtype=np.int32)
    for i in range(N):
        pred_map.ravel()[node_to_pixel_indices[i]] = node_predictions[i]

    # pixel_test_mask is already (H, W) boolean — index directly
    test_true = gt[pixel_test_mask]
    test_pred = pred_map[pixel_test_mask]

    metrics = evaluate_metrics(test_true, test_pred, num_classes)
    logger.info(
        f"Final Test Metrics: "
        f"OA={metrics['OA']:.2f}  "
        f"AA={metrics['AA']:.2f}  "
        f"Kappa={metrics['Kappa']:.2f}"
    )
    return metrics, node_predictions, pred_map


# ===========================================================================
# Section 5 — Segmentation plots
# ===========================================================================

def _load_raw_rgb(data_path: str):
    """Load Indian Pines raw data and build a false-colour RGB composite."""
    data_mat = sio.loadmat(data_path)
    raw      = data_mat["indian_pines_corrected"].astype(np.float32)   # (H, W, B)

    r = raw[:, :, 29]; r = (r - r.min()) / (r.max() - r.min() + 1e-8)
    g = raw[:, :, 19]; g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    b = raw[:, :,  9]; b = (b - b.min()) / (b.max() - b.min() + 1e-8)
    return raw, np.dstack((r, g, b))


def save_segmentation_plot(rgb_img, ground_truth, predicted_map, model_label, output_path):
    """Save a 3-panel [RGB | Ground Truth | Predictions] figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(rgb_img);        axes[0].set_title("False-Colour Composite (RGB)"); axes[0].axis("off")
    axes[1].imshow(ground_truth, cmap=CMAP); axes[1].set_title("Ground Truth");        axes[1].axis("off")
    axes[2].imshow(predicted_map, cmap=CMAP); axes[2].set_title(f"{model_label} Predictions"); axes[2].axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Segmentation plot saved → {output_path}")


def plot_cnn_predictions(model, model_name, device, patch_size,
                         data_path, gt_path, output_path, logger):
    """
    Run full-image inference for a CNN model and save the segmentation plot.
    Handles 1D (pixel), 2D (patch), and 3D (volumetric patch) formats.
    """
    model.eval()
    raw, rgb_img = _load_raw_rgb(data_path)
    gt_mat       = sio.loadmat(gt_path)
    ground_truth = gt_mat["indian_pines_gt"]

    rows, cols, bands = raw.shape

    # Normalise
    cube       = raw.copy()
    band_mins  = cube.min(axis=(0, 1), keepdims=True)
    band_maxs  = cube.max(axis=(0, 1), keepdims=True)
    band_ranges = band_maxs - band_mins
    band_ranges[band_ranges == 0] = 1e-8
    cube = (cube - band_mins) / band_ranges

    # Build all-pixel input tensor
    if model_name == "cnn1d":
        flat    = cube.reshape(-1, bands).astype(np.float32)
        x_input = torch.tensor(flat).unsqueeze(1)   # (H*W, 1, B)
        batch_size = 512

    else:  # cnn2d or cnn3d
        half        = patch_size // 2
        padded      = np.pad(cube, ((half, half), (half, half), (0, 0)), mode="reflect")
        all_patches = []
        for r in range(rows):
            for c in range(cols):
                pr, pc = r + half, c + half
                patch  = padded[pr - half: pr + half + 1,
                                pc - half: pc + half + 1, :]   # (P, P, B)
                if model_name == "cnn2d":
                    all_patches.append(patch.transpose(2, 0, 1))           # (B, P, P)
                else:
                    all_patches.append(patch.transpose(2, 0, 1)[np.newaxis])  # (1, B, P, P)

        x_input    = torch.tensor(np.array(all_patches, dtype=np.float32))
        batch_size = 512 if model_name == "cnn2d" else 256

    # Predict
    predictions = []
    with torch.no_grad():
        for i in range(0, len(x_input), batch_size):
            batch   = x_input[i: i + batch_size].to(device)
            outputs = model(batch)
            if outputs.dim() == 4:
                outputs = outputs.squeeze(-1).squeeze(-1)
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())

    predicted_map = (np.array(predictions) + 1).reshape(rows, cols)
    predicted_map[ground_truth == 0] = 0

    labels = {"cnn1d": "1D CNN", "cnn2d": "2D CNN", "cnn3d": "3D CNN"}
    save_segmentation_plot(rgb_img, ground_truth, predicted_map, labels[model_name], output_path)
    logger.info(f"Segmentation plot saved → {output_path}")


# ===========================================================================
# Section 6 — Argument parsing & main
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train HSI classification models")
    parser.add_argument("--model",   type=str, required=True,
                        choices=["cnn1d", "cnn2d", "cnn3d", "gcn"],
                        help="Model to train")
    parser.add_argument("--config",  type=str, default="configs/gcn_config.yaml",
                        help="Path to YAML config (GCN only)")
    parser.add_argument("--dataset", type=str, default="indian_pines",
                        help="Dataset name (GCN only, default: indian_pines)")
    parser.add_argument("--seed",    type=int, default=0,
                        help="Random seed (default: 0)")

    # CNN hyperparameters (override defaults)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--dropout",    type=float, default=None,
                        help="Dropout (default: model-specific)")
    parser.add_argument("--num_classes",type=int,   default=16)
    parser.add_argument("--patch_size", type=int,   default=5,
                        help="Spatial patch size for 2D/3D CNN (default: 5)")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args   = parse_args()
    set_seed(args.seed)

    # -----------------------------------------------------------------------
    # Setup output directory and logger
    # -----------------------------------------------------------------------
    run_dir = os.path.join("results", args.model, f"seed_{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(
        log_dir=run_dir,
        log_filename=f"training_{args.model}.log",
    )
    logger.info(f"Model: {args.model} | Seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # -----------------------------------------------------------------------
    # CNN branch
    # -----------------------------------------------------------------------
    if args.model in ("cnn1d", "cnn2d", "cnn3d"):
        config = {
            "num_classes": args.num_classes,
            "batch_size":  args.batch_size,
            "dropout":     args.dropout,
            "in_channels": 200,
        }

        # 1. Load data
        train_loader, test_loader, num_classes, class_weights = get_cnn_data(
            args.model, config, logger
        )

        # 2. Build model
        model     = get_model(args.model, config, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # 3. Train
        logger.info("Starting training...")
        train_cnn(model, args.model, train_loader, criterion, optimizer, device, args.epochs, logger)

        # 4. Evaluate
        logger.info("Evaluating...")
        metrics = evaluate_cnn(model, args.model, test_loader, device, logger)

        # 5. Log to CSV
        params = {
            "model":      args.model,
            "epochs":     args.epochs,
            "lr":         args.lr,
            "batch_size": args.batch_size,
            "seed":       args.seed,
        }
        log_experiment(params, metrics, log_dir="results", csv_filename="experiments.csv")

        # 6. Segmentation plot
        logger.info("Generating segmentation plot...")
        plot_cnn_predictions(
            model, args.model, device,
            patch_size=args.patch_size,
            data_path="data/raw/Indian_pines_corrected.mat",
            gt_path="data/raw/Indian_pines_gt.mat",
            output_path=os.path.join(run_dir, "segmentation_comparison.png"),
            logger=logger,
        )

    # -----------------------------------------------------------------------
    # GCN branch
    # -----------------------------------------------------------------------
    elif args.model == "gcn":
        # Load YAML config
        with open(args.config, "r") as f:
            gcn_cfg = yaml.safe_load(f)[args.dataset]
        gcn_cfg["dataset"]     = args.dataset
        gcn_cfg["num_classes"] = gcn_cfg.get("num_classes", 16)

        # 1. Build graph data
        (pt_subgraphs, node_train, node_val, node_test, pixel_test_mask,
         gt, node_to_pixel_indices, N, num_classes, edge_cut_ratio) = get_gcn_data(
            gcn_cfg, args.seed, device, run_dir, logger
        )

        # 2. Build model
        model     = get_model("gcn", gcn_cfg, device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=gcn_cfg["learning_rate"],
            weight_decay=5e-4,
        )

        # 3. Train
        start_time = time.time()
        loss_history, val_acc_history = train_gcn(
            model, pt_subgraphs, optimizer, gcn_cfg, logger
        )
        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f}s")

        # 4. Evaluate
        metrics, node_predictions, pred_map = evaluate_gcn(
            model, pt_subgraphs, gt, node_to_pixel_indices,
            pixel_test_mask, N, num_classes, logger
        )
        metrics["edge_cut_ratio"] = edge_cut_ratio

        # 5. Save histories and metrics
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump({"loss": loss_history, "val_acc": val_acc_history}, f)
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)

        np.save(os.path.join(run_dir, "node_predictions.npy"), node_predictions)

        # 6. Log to shared experiments CSV
        import csv
        csv_path   = os.path.join("results", "experiments.csv")
        file_exists = os.path.exists(csv_path)
        fieldnames  = ["model", "dataset", "seed", "p", "c", "F",
                       "top_k", "hops", "edge_cut_ratio",
                       "OA", "AA", "Kappa", "wall_clock"]
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "model":          "GCN",
                "dataset":        args.dataset,
                "seed":           args.seed,
                "p":              N,
                "c":              gcn_cfg["num_clusters"],
                "F":              gcn_cfg["pca_components"],
                "top_k":          gcn_cfg["top_k"],
                "hops":           str(gcn_cfg["hops"]),
                "edge_cut_ratio": edge_cut_ratio,
                "OA":             metrics["OA"],
                "AA":             metrics["AA"],
                "Kappa":          metrics["Kappa"],
                "wall_clock":     train_time,
            })
        logger.info(f"Results appended → {csv_path}")

        # 7. Segmentation plot
        _, rgb_img = _load_raw_rgb("data/raw/Indian_pines_corrected.mat")
        save_segmentation_plot(
            rgb_img, gt, pred_map, "GCN",
            os.path.join(run_dir, "segmentation_comparison.png"),
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
