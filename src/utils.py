import os
import csv
import logging
from datetime import datetime

def setup_logger(log_dir='results', log_filename='training'):
    """Sets up a logger that outputs to both console and a file."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    log_path = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def log_experiment(params: dict, metrics: dict, log_dir='results', csv_filename='experiments.csv'):
    """
    Appends a row of hyperparameters and evaluation metrics to results/experiments.csv.
    
    Args:
        params (dict): Hyperparameters for this run (e.g. epochs, lr, dropout, batch_size).
        metrics (dict): Evaluation metrics (e.g. accuracy, precision, recall, f1).
        log_dir (str): Directory to save the CSV file.
        csv_filename (str): Name of the CSV file.
    """
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, csv_filename)
    
    row = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    row.update(params)
    row.update(metrics)
    
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    return csv_path
