import os
import time
import json
import yaml
import pickle
import argparse
import random
import numpy as np
import torch
import logging
import csv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.load_hsi import load_dataset
from src.superpixel import segment
from src.graph_builder import build_graph
from src.graph_partition import partition_graph
from src.gcn_model import GCN, compute_loss, normalize_adjacency
from src.dataset import split_data, propagate_labels_to_nodes
from src.evaluate import evaluate_metrics
from src.visualize import generate_all_plots

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="indian_pines")
    parser.add_argument("--config", type=str, default="configs/gcn_config.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    if args.dataset != "indian_pines":
        raise NotImplementedError("Only indian_pines is implemented right now.")
        
    set_seed(args.seed)
    
    # Setup run dir
    run_dir = os.path.join("results", "gcn", args.dataset, f"seed_{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(run_dir, "training.log")),
                            logging.StreamHandler()
                        ])
                        
    logging.info(f"Starting GCN pipeline for {args.dataset} (seed {args.seed})")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)[args.dataset]
        
    # 1. Load data
    cube, gt, metadata = load_dataset(args.dataset)
    num_classes = metadata['num_classes']
    
    # Save arrays for visualization later
    np.save(os.path.join(run_dir, "cube.npy"), cube)
    np.save(os.path.join(run_dir, "gt.npy"), gt)
    
    # 2. Stage A: Superpixel Segmentation
    superpixel_map = segment(cube, config['n_segments'], config['compactness'])
    np.save(os.path.join(run_dir, "superpixel_map.npy"), superpixel_map)
    
    # 3. Stage B: Graph Construction
    X, A_sparse, superpixel_map, node_to_pixel_indices, node_labels = build_graph(
        cube, superpixel_map, gt, 
        pca_components=config['pca_components'], 
        top_k=config['top_k'], 
        hops=config['hops']
    )
    
    with open(os.path.join(run_dir, "node_to_pixel_indices.pkl"), "wb") as f:
        pickle.dump(node_to_pixel_indices, f)
        
    N = X.shape[0]
    logging.info(f"Generated {N} superpixel nodes")
    
    # Splitting logic at pixel level, then propagated to nodes
    train_mask, val_mask, test_mask = split_data(gt, num_classes=num_classes, seed=args.seed)
    node_train, node_val, node_test = propagate_labels_to_nodes(
        node_to_pixel_indices, train_mask, val_mask, test_mask, gt
    )
    
    # 4. Stage C: Graph Partition
    sub_graphs, edge_cut_ratio = partition_graph(A_sparse, X, config['num_clusters'], node_labels)
    
    # 5. Stage D & E: GCN Model & Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN(in_features=config['pca_components'], 
                hidden_features=config['hidden_units'], 
                num_classes=num_classes).to(device)
                
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=5e-4)
    
    # Pre-process sub-graphs for PyTorch
    pt_subgraphs = []
    for sg in sub_graphs:
        norm_A = normalize_adjacency(sg['A']).to(device)
        pt_X = torch.FloatTensor(sg['X']).to(device)
        pt_Y = torch.LongTensor(sg['Y']).to(device)
        V_i = sg['V']
        mask_train = torch.BoolTensor(node_train[V_i]).to(device)
        mask_val = torch.BoolTensor(node_val[V_i]).to(device)
        pt_subgraphs.append({
            'V_i': V_i,
            'A': norm_A,
            'X': pt_X,
            'Y': pt_Y,
            'train_mask': mask_train,
            'val_mask': mask_val
        })
        
    steps_per_epoch = 5 * config['num_clusters']
    max_iter = config['epochs'] * steps_per_epoch
    
    loss_history = []
    val_acc_history = []
    
    logging.info("Starting training loop...")
    start_time = time.time()
    
    model.train()
    
    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        
        for step in range(steps_per_epoch):
            sg = random.choice(pt_subgraphs)
            
            optimizer.zero_grad()
            logits = model(sg['X'], sg['A'])
            
            loss = compute_loss(logits, sg['Y'], sg['train_mask'])
            
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            epoch_loss += loss.item()
            
        epoch_loss /= steps_per_epoch
        loss_history.append(epoch_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for sg in pt_subgraphs:
                if sg['val_mask'].sum() > 0:
                    logits = model(sg['X'], sg['A'])
                    preds = logits.argmax(dim=1) + 1 # +1 because classes are 1..16
                    correct += (preds[sg['val_mask']] == sg['Y'][sg['val_mask']]).sum().item()
                    total += sg['val_mask'].sum().item()
                    
            val_acc = correct / max(total, 1)
            val_acc_history.append(val_acc)
            
        model.train()
        
        if (epoch + 1) % 50 == 0:
            logging.info(f"Epoch {epoch+1}/{config['epochs']} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")
            
    train_time = time.time() - start_time
    logging.info(f"Training completed in {train_time:.2f} seconds")
    
    # 6. Inference & Pixel mapping
    model.eval()
    node_predictions = np.zeros(N, dtype=np.int32)
    
    with torch.no_grad():
        for sg in pt_subgraphs:
            logits = model(sg['X'], sg['A'])
            preds = logits.argmax(dim=1).cpu().numpy() + 1
            node_predictions[sg['V_i']] = preds
            
    np.save(os.path.join(run_dir, "node_predictions.npy"), node_predictions)
    
    # 7. Evaluate on Test set
    # Create pixel-level prediction map
    H, W = gt.shape
    pred_map = np.zeros((H, W), dtype=np.int32)
    for i in range(N):
        pred_map.ravel()[node_to_pixel_indices[i]] = node_predictions[i]
        
    test_true = gt[test_mask]
    test_pred = pred_map[test_mask]
    
    metrics = evaluate_metrics(test_true, test_pred, num_classes)
    
    logging.info(f"Final Test Metrics: OA={metrics['OA']:.2f}, AA={metrics['AA']:.2f}, Kappa={metrics['Kappa']:.2f}")
    
    # Save histories and metrics
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump({"loss": loss_history, "val_acc": val_acc_history}, f)
        
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        metrics['edge_cut_ratio'] = edge_cut_ratio
        json.dump(metrics, f)
        
    # Append to experiments.csv
    csv_path = os.path.join("results", "experiments.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        fieldnames = ["dataset", "method", "seed", "p", "c", "F", "top_k", "hops", "edge_cut_ratio", "OA", "AA", "Kappa", "wall_clock"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "dataset": args.dataset,
            "method": "GCN",
            "seed": args.seed,
            "p": N,
            "c": config['num_clusters'],
            "F": config['pca_components'],
            "top_k": config['top_k'],
            "hops": str(config['hops']),
            "edge_cut_ratio": edge_cut_ratio,
            "OA": metrics['OA'],
            "AA": metrics['AA'],
            "Kappa": metrics['Kappa'],
            "wall_clock": train_time
        })
        
    # 8. Generate Plots
    generate_all_plots(run_dir, args.dataset)
    logging.info("Pipeline complete.")

if __name__ == "__main__":
    main()
