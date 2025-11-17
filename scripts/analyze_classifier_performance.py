"""
Analyze classifier performance to investigate high fault rate.
Check if the classifier is over-predicting faults.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_predictions(
    predictions_file: str = "outputs/exp_full_dataset/predictions_for_colab.parquet"
):
    """
    Analyze classifier predictions to diagnose high fault rate.
    """
    print("="*60)
    print("CLASSIFIER PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Load predictions
    print(f"\nLoading predictions from: {predictions_file}")
    pred_df = pd.read_parquet(predictions_file)
    
    print(f"Total windows: {len(pred_df)}")
    
    # Overall distribution
    print("\n" + "="*60)
    print("PREDICTION DISTRIBUTION")
    print("="*60)
    
    prediction_counts = pred_df['prediction'].value_counts()
    for pred, count in prediction_counts.items():
        pct = count / len(pred_df) * 100
        print(f"{pred:25s}: {count:5d} ({pct:5.1f}%)")
    
    fault_rate = (len(pred_df) - prediction_counts.get('Normal', 0)) / len(pred_df) * 100
    print(f"\n{'TOTAL FAULT RATE':25s}: {fault_rate:5.1f}%")
    
    # Confidence analysis
    print("\n" + "="*60)
    print("CONFIDENCE ANALYSIS")
    print("="*60)
    
    print("\nAverage confidence by prediction:")
    for pred in prediction_counts.index:
        pred_subset = pred_df[pred_df['prediction'] == pred]
        avg_conf = pred_subset['confidence'].mean()
        std_conf = pred_subset['confidence'].std()
        print(f"{pred:25s}: {avg_conf:.3f} ± {std_conf:.3f}")
    
    # Check for low-confidence faults (likely false positives)
    print("\n" + "="*60)
    print("LOW-CONFIDENCE PREDICTIONS (Potential False Positives)")
    print("="*60)
    
    low_conf_threshold = 0.6
    low_conf = pred_df[pred_df['confidence'] < low_conf_threshold]
    
    print(f"\nPredictions with confidence < {low_conf_threshold}:")
    low_conf_counts = low_conf['prediction'].value_counts()
    for pred, count in low_conf_counts.items():
        pct = count / len(low_conf) * 100
        print(f"{pred:25s}: {count:5d} ({pct:5.1f}%)")
    
    # Check temporal patterns
    print("\n" + "="*60)
    print("TEMPORAL PATTERN")
    print("="*60)
    
    # Group by time segments
    pred_df['window_group'] = pred_df['window_id'] // 500  # Groups of 500 windows
    
    print("\nFault rate over time (groups of 500 windows):")
    for group in pred_df['window_group'].unique()[:10]:  # First 10 groups
        group_data = pred_df[pred_df['window_group'] == group]
        fault_count = len(group_data[group_data['prediction'] != 'Normal'])
        fault_pct = fault_count / len(group_data) * 100
        print(f"Windows {group*500:5d}-{(group+1)*500:5d}: {fault_pct:5.1f}% faults")
    
    # Recommendations
    print("\n" + "="*60)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("="*60)
    
    if fault_rate > 50:
        print("\n⚠️  VERY HIGH FAULT RATE (>50%)")
        print("\nLikely causes:")
        print("  1. Training data was heavily imbalanced (too many fault examples)")
        print("  2. This data file is from an abnormal period (startup, maintenance)")
        print("  3. Classifier is over-sensitive (needs re-training or threshold tuning)")
        
        print("\nRecommended actions:")
        print("  ✓ Check the training data distribution - was it balanced?")
        print("  ✓ Review the source of this CSV file - is it from normal operations?")
        print("  ✓ Consider re-training with more 'Normal' examples")
        print("  ✓ For now, filter to only high-confidence faults (confidence > 0.7)")
    
    elif fault_rate > 20:
        print("\n⚠️  HIGH FAULT RATE (20-50%)")
        print("\nThis might be acceptable if:")
        print("  - The data is from a problematic period")
        print("  - The process is inherently unstable")
        
        print("\nBut consider:")
        print("  ✓ Reviewing low-confidence predictions")
        print("  ✓ Checking if 'Normal' examples were under-represented in training")
    
    else:
        print("\n✅ FAULT RATE LOOKS REASONABLE (<20%)")
        print("This is typical for a well-operating system with occasional faults.")
    
    # Suggest high-confidence filtering
    print("\n" + "="*60)
    print("FILTERING OPTIONS")
    print("="*60)
    
    high_conf_threshold = 0.7
    high_conf_faults = pred_df[
        (pred_df['prediction'] != 'Normal') & 
        (pred_df['confidence'] >= high_conf_threshold)
    ]
    
    print(f"\nOption A: Filter to high-confidence faults only (confidence ≥ {high_conf_threshold}):")
    print(f"  Total windows: {len(high_conf_faults)}")
    print(f"  Reduction: {(1 - len(high_conf_faults)/len(pred_df))*100:.1f}%")
    print(f"  Distribution:")
    for pred, count in high_conf_faults['prediction'].value_counts().items():
        print(f"    {pred}: {count}")
    
    return pred_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze classifier predictions")
    parser.add_argument(
        "--predictions",
        default="outputs/exp_full_dataset/predictions_for_colab.parquet",
        help="Predictions file to analyze"
    )
    
    args = parser.parse_args()
    
    analyze_predictions(args.predictions)

