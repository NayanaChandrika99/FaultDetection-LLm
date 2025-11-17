"""
Evaluation Metrics
Compute classification metrics and generate visualizations.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    recall_score
)
from typing import Dict, List, Optional
import json


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro F1 score (average across classes)."""
    return f1_score(y_true, y_pred, average='macro')


def compute_per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Compute recall for each class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional list of class names
    
    Returns:
        Dict mapping class name to recall
    """
    unique_labels = np.unique(y_true)
    recalls = recall_score(y_true, y_pred, labels=unique_labels, average=None)
    
    if labels is None:
        labels = unique_labels
    
    return {str(label): float(recall) for label, recall in zip(labels, recalls)}


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray, pos_label: int = 1) -> float:
    """
    Compute Precision-Recall AUC for binary or one-vs-rest.
    
    Args:
        y_true: True labels (binary)
        y_score: Prediction scores
        pos_label: Positive class label
    
    Returns:
        PR-AUC score
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
    return auc(recall, precision)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    normalize: bool = False
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class names
        save_path: Path to save figure
        normalize: Normalize by row (true label)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_class_metrics(
    recall_dict: Dict[str, float],
    save_path: Optional[str] = None
):
    """
    Plot per-class recall as a bar chart.
    
    Args:
        recall_dict: Dict mapping class to recall
        save_path: Path to save figure
    """
    classes = list(recall_dict.keys())
    recalls = list(recall_dict.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(classes)), recalls, color='steelblue', alpha=0.8)
    
    # Color bars based on performance
    for i, (bar, recall) in enumerate(zip(bars, recalls)):
        if recall < 0.5:
            bar.set_color('red')
        elif recall < 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title('Per-Class Recall', fontsize=14)
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='50% threshold')
    plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.3, label='70% threshold')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_full(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Comprehensive evaluation with all metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        labels: Class names
        output_dir: Directory to save plots and results
    
    Returns:
        Dict of all computed metrics
    """
    results = {}
    
    # Basic accuracy
    results['accuracy'] = float(accuracy_score(y_true, y_pred))
    
    # Macro F1
    results['macro_f1'] = float(compute_macro_f1(y_true, y_pred))
    
    # Per-class recall
    results['per_class_recall'] = compute_per_class_recall(y_true, y_pred, labels)
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    results['classification_report'] = report
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = cm.tolist()
    
    # Mean confidence (if probabilities provided)
    if y_proba is not None:
        confidence = y_proba.max(axis=1)
        results['mean_confidence'] = float(confidence.mean())
        results['confidence_std'] = float(confidence.std())
    
    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plots
        plot_confusion_matrix(
            y_true, y_pred, labels,
            save_path=str(output_dir / 'confusion_matrix.png')
        )
        plot_confusion_matrix(
            y_true, y_pred, labels,
            save_path=str(output_dir / 'confusion_matrix_normalized.png'),
            normalize=True
        )
        plot_per_class_metrics(
            results['per_class_recall'],
            save_path=str(output_dir / 'per_class_recall.png')
        )
        
        # Save metrics as JSON
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to: {output_dir / 'metrics.json'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Model Performance')
    parser.add_argument('--run_id', type=str, required=True,
                        help='Run ID to evaluate')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--robustness', action='store_true',
                        help='Run robustness tests')
    
    args = parser.parse_args()
    
    # Load results
    results_path = Path(args.output_dir) / args.run_id / 'results.json'
    
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract predictions
    y_pred = np.array(results['metrics']['y_pred'])
    y_proba = np.array(results['metrics']['y_proba'])
    
    # Note: Would need to load y_true from somewhere
    # For now, just print loaded metrics
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {args.run_id}")
    print(f"{'='*60}")
    print(f"\nMacro F1: {results['metrics']['macro_f1']:.4f}")
    print(f"Mean Confidence: {results['metrics']['mean_confidence']:.4f}")
    
    if args.robustness:
        print("\nRunning robustness tests...")
        from evaluation.robustness_tests import run_robustness_suite
        # Would implement robustness tests here
        print("Robustness tests not yet implemented")


if __name__ == '__main__':
    main()

