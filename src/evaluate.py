import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import argparse
import os
import json
import logging

def evaluate_metrics(y_true, y_pred, num_classes=16):
    """
    Compute pixel-level metrics: OA, AA, Kappa, and per-class accuracy.
    y_true and y_pred should be 1D arrays of the same length containing only test pixels.
    """
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(1, num_classes + 1))
    
    per_class_acc = []
    for i in range(num_classes):
        if cm[i, :].sum() == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(cm[i, i] / cm[i, :].sum())
            
    aa = np.mean(per_class_acc)
    
    return {
        "OA": oa * 100,
        "AA": aa * 100,
        "Kappa": kappa * 100,
        "PerClass": [acc * 100 for acc in per_class_acc]
    }

def print_metrics_table(metrics_list, class_names):
    """
    Given a list of metrics dicts (e.g. from 10 seeds), compute mean/std and print.
    """
    oas = [m["OA"] for m in metrics_list]
    aas = [m["AA"] for m in metrics_list]
    kappas = [m["Kappa"] for m in metrics_list]
    per_class = np.array([m["PerClass"] for m in metrics_list])
    
    mean_oa, std_oa = np.mean(oas), np.std(oas)
    mean_aa, std_aa = np.mean(aas), np.std(aas)
    mean_kappa, std_kappa = np.mean(kappas), np.std(kappas)
    
    mean_per_class = np.mean(per_class, axis=0)
    std_per_class = np.std(per_class, axis=0)
    
    print("-" * 50)
    print(f"{'Class Name':<30} | Accuracy (%)")
    print("-" * 50)
    for i, name in enumerate(class_names):
        print(f"{name:<30} | {mean_per_class[i]:.2f} ± {std_per_class[i]:.2f}")
    print("-" * 50)
    print(f"{'Overall Accuracy (OA)':<30} | {mean_oa:.2f} ± {std_oa:.2f}")
    print(f"{'Average Accuracy (AA)':<30} | {mean_aa:.2f} ± {std_aa:.2f}")
    print(f"{'Kappa':<30} | {mean_kappa:.2f} ± {std_kappa:.2f}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    
    class_names = [
        "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
        "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
        "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean",
        "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"
    ]
    
    metrics_list = []
    
    for run_dir in args.runs:
        metric_file = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metric_file):
            with open(metric_file, "r") as f:
                m = json.load(f)
                metrics_list.append(m)
        else:
            print(f"Warning: No metrics.json found in {run_dir}")
            
    if len(metrics_list) > 0:
        print_metrics_table(metrics_list, class_names)
    else:
        print("No metrics loaded.")
