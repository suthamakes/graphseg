import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.segmentation import mark_boundaries
import os
import json

# Fixed palette for Indian Pines (17 colors: 0=background, 1..16=classes)
# We can just define a robust set of 16 distinctive colors.
# Example using tab20
tab20 = plt.get_cmap('tab20', 20)
INDIAN_PINES_COLORS = np.zeros((17, 3), dtype=np.uint8)
INDIAN_PINES_COLORS[0] = [0, 0, 0] # Background is black
for i in range(1, 17):
    # slice tab20 colors
    INDIAN_PINES_COLORS[i] = np.array(tab20(i - 1)[:3]) * 255

def get_class_colormap(num_classes: int, background: str = "black") -> np.ndarray:
    return INDIAN_PINES_COLORS

def plot_false_color_image(cube: np.ndarray, band_indices=(29, 19, 9), out_path: str = "") -> None:
    rgb = np.zeros((cube.shape[0], cube.shape[1], 3), dtype=np.float32)
    for i, b in enumerate(band_indices):
        band = cube[:, :, b]
        min_v, max_v = band.min(), band.max()
        if max_v > min_v:
            rgb[:, :, i] = (band - min_v) / (max_v - min_v)
    
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    plt.imsave(out_path, rgb_uint8)
    return rgb_uint8

def plot_label_map(label_map: np.ndarray, class_colors: np.ndarray, class_names: list, title: str, out_path: str, show_legend: bool = True) -> None:
    H, W = label_map.shape
    rgb_map = np.zeros((H, W, 3), dtype=np.uint8)
    
    for c in range(class_colors.shape[0]):
        rgb_map[label_map == c] = class_colors[c]
        
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb_map)
    ax.set_title(title)
    ax.axis('off')
    
    if show_legend:
        patches = [mpatches.Patch(color=class_colors[i+1]/255.0, label=class_names[i]) for i in range(len(class_names))]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def build_prediction_map(node_predictions: np.ndarray, node_to_pixel_indices: dict, H: int, W: int) -> np.ndarray:
    pred_map = np.zeros((H, W), dtype=np.int32)
    flat_map = pred_map.flatten()
    
    for i in range(len(node_predictions)):
        idx = node_to_pixel_indices[i] # This might have been saved with string keys if loaded from JSON
        if isinstance(idx, list):
            idx = np.array(idx)
        flat_map[idx] = node_predictions[i]
        
    return flat_map.reshape(H, W)

def plot_superpixel_overlay(cube: np.ndarray, superpixel_map: np.ndarray, out_path: str, boundary_color=(1.0, 1.0, 0.0)) -> None:
    # Get false color base
    # We will compute it quickly again for overlay
    rgb = np.zeros((cube.shape[0], cube.shape[1], 3), dtype=np.float32)
    band_indices = (29, 19, 9)
    for i, b in enumerate(band_indices):
        band = cube[:, :, b]
        min_v, max_v = band.min(), band.max()
        if max_v > min_v:
            rgb[:, :, i] = (band - min_v) / (max_v - min_v)
            
    overlay = mark_boundaries(rgb, superpixel_map, color=boundary_color)
    plt.imsave(out_path, (overlay * 255).astype(np.uint8))

def plot_gt_vs_prediction(gt_map, pred_map, class_colors, class_names, metrics: dict, out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    H, W = gt_map.shape
    gt_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    pred_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    for c in range(class_colors.shape[0]):
        gt_rgb[gt_map == c] = class_colors[c]
        pred_rgb[pred_map == c] = class_colors[c]
        
    ax1.imshow(gt_rgb)
    ax1.set_title("Ground Truth")
    ax1.axis('off')
    
    subtitle = f"Prediction\nOA: {metrics.get('OA', 0):.2f}% | AA: {metrics.get('AA', 0):.2f}% | Kappa: {metrics.get('Kappa', 0):.2f}%"
    ax2.imshow(pred_rgb)
    ax2.set_title(subtitle)
    ax2.axis('off')
    
    patches = [mpatches.Patch(color=class_colors[i+1]/255.0, label=class_names[i]) for i in range(len(class_names))]
    fig.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_training_curve(loss_history, val_acc_history, out_path: str) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    epochs = range(len(loss_history))
    ax1.plot(epochs, loss_history, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_acc_history, 'r-', label='Validation OA')
    ax2.set_ylabel('Validation OA', color='r')
    ax2.tick_params('y', colors='r')
    
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, out_path: str, normalize: bool = True) -> None:
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(1, len(class_names)+1))
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
           
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def generate_all_plots(run_dir: str, dataset_name: str = "indian_pines") -> None:
    print(f"Generating plots for run: {run_dir}")
    
    # Load cached arrays (assume they exist in run_dir)
    cube = np.load(os.path.join(run_dir, "cube.npy"))
    gt = np.load(os.path.join(run_dir, "gt.npy"))
    superpixel_map = np.load(os.path.join(run_dir, "superpixel_map.npy"))
    node_preds = np.load(os.path.join(run_dir, "node_predictions.npy"))
    
    import pickle
    with open(os.path.join(run_dir, "node_to_pixel_indices.pkl"), "rb") as f:
        node_to_pixel_indices = pickle.load(f)
        
    with open(os.path.join(run_dir, "metrics.json"), "r") as f:
        metrics = json.load(f)
        
    with open(os.path.join(run_dir, "history.json"), "r") as f:
        history = json.load(f)
        
    class_names = [
        "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
        "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
        "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean",
        "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"
    ]
    colors = get_class_colormap(16)
    
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join("results", "plots"), exist_ok=True)
    
    fc_path = os.path.join(run_dir, "plots", "indian_pines_falsecolor.png")
    plot_false_color_image(cube, out_path=fc_path)
    
    sp_path = os.path.join("results", "plots", "segmentation_comparison.png")
    plot_superpixel_overlay(cube, superpixel_map, sp_path)
    
    gt_path = os.path.join(run_dir, "plots", "indian_pines_gt.png")
    plot_label_map(gt, colors, class_names, "Indian Pines - Ground Truth", gt_path)
    
    H, W = gt.shape
    pred_map = build_prediction_map(node_preds, node_to_pixel_indices, H, W)
    
    pred_path = os.path.join("results", "plots", "indian_pines_gcn_prediction.png")
    plot_label_map(pred_map, colors, class_names, "Indian Pines - GCN Prediction", pred_path)
    
    gt_vs_pred_path = os.path.join(run_dir, "plots", "indian_pines_gt_vs_pred.png")
    plot_gt_vs_prediction(gt, pred_map, colors, class_names, metrics, gt_vs_pred_path)
    
    curve_path = os.path.join(run_dir, "plots", "indian_pines_training_curve.png")
    plot_training_curve(history["loss"], history["val_acc"], curve_path)
    
    cm_path = os.path.join(run_dir, "plots", "indian_pines_confusion_matrix.png")
    # For CM, only evaluate on test mask or all evaluated pixels? Usually test.
    # The metrics we have don't include test mask directly, but we can reconstruct it or just use all labeled pixels for this plot.
    # We will just reconstruct the prediction map for all pixels with gt > 0.
    valid_mask = gt > 0
    plot_confusion_matrix(gt[valid_mask], pred_map[valid_mask], class_names, cm_path)
    
    print("Generated all plots successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="indian_pines")
    parser.add_argument("--run", type=str, required=True)
    args = parser.parse_args()
    
    if args.dataset != "indian_pines":
        raise NotImplementedError("Only indian_pines is supported")
        
    generate_all_plots(args.run, args.dataset)
