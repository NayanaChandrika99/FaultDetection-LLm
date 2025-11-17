"""
Main Training Script for MultiROCKET Classifier
Handles data loading, preprocessing, training, and evaluation.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import yaml
import json
from datetime import datetime

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.loaders.slurry_loader import load_and_window
from data.loaders.window_creator import assign_labels, filter_windows_by_quality
from utils.physical_checks import filter_windows_by_physics
from models.rocket_heads import MultiROCKETClassifier
from models.fusion import LateFusionClassifier


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(
    csv_path: str,
    config: dict,
    maintenance_logs: list = None,
    use_heuristics: bool = True
):
    """
    Load and prepare data for training.
    
    Returns:
        Tuple of (windows, labels, metadata, window_dfs)
    """
    print(f"\n{'='*60}")
    print("DATA LOADING")
    print(f"{'='*60}")
    
    # Load and create windows
    print(f"Loading data from: {csv_path}")
    windows, metadata = load_and_window(
        csv_path=csv_path,
        window_sec=config['data']['window_sec'],
        stride_sec=config['data']['stride_sec'],
        sample_rate=config['data']['sample_rate'],
        max_gap_sec=config['data']['max_gap_sec'],
        max_missing_ratio=config['data'].get('max_missing_ratio', 0.1)
    )
    print(f"Created {len(windows)} windows")
    print(f"Window shape: {windows.shape}")
    
    # Load original DataFrame for labeling
    from data.loaders.slurry_loader import load_slurry_csv
    df = load_slurry_csv(
        csv_path,
        sample_rate=config['data']['sample_rate'],
        max_gap_sec=config['data']['max_gap_sec']
    )
    
    # Assign labels
    print("\nAssigning labels...")
    labels = assign_labels(
        windows=windows,
        metadata=metadata,
        df=df,
        maintenance_logs=maintenance_logs,
        use_heuristics=use_heuristics
    )
    
    # Show label distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
    
    # Quality filtering
    print("\nFiltering by quality...")
    windows, labels, metadata = filter_windows_by_quality(
        windows, labels, metadata,
        min_valid_ratio=config['data'].get('min_valid_ratio', 0.9)
    )
    print(f"Remaining windows after quality filter: {len(windows)}")
    
    # Physical validation
    if config['data'].get('physical_validation', True):
        print("\nPhysical validation...")
        # Create window DataFrames for validation
        window_dfs = []
        for i in range(len(windows)):
            start_time = metadata.iloc[i]['start_time']
            end_time = metadata.iloc[i]['end_time']
            start_idx = df.index.get_loc(start_time)
            end_idx = df.index.get_loc(end_time) + 1
            window_dfs.append(df.iloc[start_idx:end_idx])
        
        windows, labels, metadata, validation_log = filter_windows_by_physics(
            windows=windows,
            window_dfs=window_dfs,
            labels=labels,
            metadata=metadata
        )
        print(f"Remaining windows after physical validation: {len(windows)}")
        
        # Show validation stats
        n_failed_mass = (~validation_log['mass_balance_valid']).sum()
        n_failed_density = (~validation_log['density_sg_valid']).sum()
        print(f"  Failed mass balance: {n_failed_mass}")
        print(f"  Failed density-SG: {n_failed_density}")
    
    return windows, labels, metadata


def train_model(
    windows: np.ndarray,
    labels: np.ndarray,
    config: dict,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train the classifier model.
    
    Returns:
        Tuple of (model, X_train, X_test, y_train, y_test)
    """
    print(f"\n{'='*60}")
    print("MODEL TRAINING")
    print(f"{'='*60}")
    
    # Fill any remaining NaN values (MultiRocket cannot handle them)
    print(f"\nHandling missing values...")
    nan_count_before = np.isnan(windows).sum()
    if nan_count_before > 0:
        print(f"  Found {nan_count_before} NaN values ({nan_count_before/windows.size*100:.3f}%)")
        # Forward fill, then backward fill, then fill remaining with 0
        for i in range(len(windows)):
            for j in range(windows.shape[1]):
                series = windows[i, j, :]
                # Forward fill
                mask = np.isnan(series)
                if mask.any():
                    indices = np.where(~mask)[0]
                    if len(indices) > 0:
                        series[mask] = np.interp(np.where(mask)[0], indices, series[indices])
                    # Fill any remaining with mean of non-nan values
                    if np.isnan(series).any():
                        mean_val = np.nanmean(series)
                        if not np.isnan(mean_val):
                            series[np.isnan(series)] = mean_val
                        else:
                            series[np.isnan(series)] = 0
                windows[i, j, :] = series
        
        nan_count_after = np.isnan(windows).sum()
        print(f"  Remaining NaN values: {nan_count_after}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        windows, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Select model type
    model_family = config['model']['family']
    fusion_type = config['model'].get('fusion', 'late')
    
    if model_family == 'multirocket' and fusion_type == 'late':
        print(f"\nTraining Late Fusion MultiROCKET...")
        model = LateFusionClassifier(
            sensor_groups=config['model'].get('sensor_groups'),
            kernels_per_group=config['model']['multirocket'].get('kernels_per_group'),
            alphas=np.array(config['model']['classifier']['alphas']),
            random_state=random_state
        )
    else:
        print(f"\nTraining Baseline MultiROCKET...")
        model = MultiROCKETClassifier(
            n_kernels=config['model']['multirocket']['n_kernels'],
            alphas=np.array(config['model']['classifier']['alphas']),
            random_state=random_state
        )
    
    # Fit model
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """
    Evaluate the trained model.
    
    Returns:
        Dict of evaluation metrics
    """
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    confidence = model.get_confidence(X_test)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Macro F1
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\nMacro F1-Score: {macro_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Mean confidence
    mean_conf = confidence.mean()
    print(f"\nMean Prediction Confidence: {mean_conf:.4f}")
    
    return {
        'macro_f1': float(macro_f1),
        'mean_confidence': float(mean_conf),
        'y_pred': y_pred.tolist(),
        'y_proba': y_proba.tolist(),
        'confidence': confidence.tolist(),
        'confusion_matrix': cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description='Train MultiROCKET Fault Classifier')
    parser.add_argument('--config', type=str, default='experiments/configs/baseline.yaml',
                        help='Path to config file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to CSV data file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for models and results')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name (default: timestamp)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples for smoke test')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set ratio')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Generate run name
    if args.run_name is None:
        args.run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"FD-LLM Training Run: {args.run_name}")
    print(f"{'='*60}")
    
    # Prepare data
    windows, labels, metadata = prepare_data(
        csv_path=args.data,
        config=config
    )
    
    # Limit samples for smoke test
    if args.max_samples:
        print(f"\n[SMOKE TEST] Limiting to {args.max_samples} samples")
        windows = windows[:args.max_samples]
        labels = labels[:args.max_samples]
    
    # Train model
    model, X_train, X_test, y_train, y_test = train_model(
        windows=windows,
        labels=labels,
        config=config,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test)
    
    # Save model
    model_path = output_dir / 'model.pkl'
    model.save(str(model_path))
    
    # Save results
    results_dict = {
        'run_name': args.run_name,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'data': {
            'csv_path': args.data,
            'n_windows': len(windows),
            'n_train': len(X_train),
            'n_test': len(X_test),
        },
        'metrics': results,
    }
    
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")
    print(f"Macro F1: {results['macro_f1']:.4f}")


if __name__ == '__main__':
    main()

