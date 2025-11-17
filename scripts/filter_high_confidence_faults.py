"""
Filter predictions to only high-confidence faults.
This removes low-confidence predictions that are likely false positives.
"""

import pandas as pd
from pathlib import Path

def filter_high_confidence_faults(
    input_file: str = "outputs/exp_full_dataset/predictions_for_colab.parquet",
    output_file: str = "outputs/exp_full_dataset/predictions_high_conf_faults.parquet",
    confidence_threshold: float = 0.7
):
    """
    Filter to only high-confidence fault predictions.
    
    Args:
        input_file: Path to full predictions file
        output_file: Path to save filtered predictions
        confidence_threshold: Minimum confidence to keep (default: 0.7)
    """
    print("="*60)
    print("FILTERING TO HIGH-CONFIDENCE FAULTS")
    print("="*60)
    
    # Load predictions
    print(f"\nLoading predictions from: {input_file}")
    pred_df = pd.read_parquet(input_file)
    total_windows = len(pred_df)
    print(f"  Total windows: {total_windows}")
    
    # Show original distribution
    print("\nOriginal Distribution:")
    for pred, count in pred_df['prediction'].value_counts().items():
        pct = count / total_windows * 100
        avg_conf = pred_df[pred_df['prediction'] == pred]['confidence'].mean()
        print(f"  {pred:25s}: {count:5d} ({pct:5.1f}%) - avg conf: {avg_conf:.3f}")
    
    # Filter to high-confidence faults only
    print(f"\nFiltering to faults with confidence ≥ {confidence_threshold}...")
    filtered_df = pred_df[
        (pred_df['prediction'] != 'Normal') & 
        (pred_df['confidence'] >= confidence_threshold)
    ].copy()
    
    filtered_windows = len(filtered_df)
    
    print(f"\n  ✓ Filtered: {filtered_windows} high-confidence fault windows")
    print(f"  ✓ Removed: {total_windows - filtered_windows} windows")
    print(f"  ✓ Reduction: {(1 - filtered_windows/total_windows)*100:.1f}%")
    
    # Show filtered distribution
    print("\nFiltered Distribution (High-Confidence Faults Only):")
    for pred, count in filtered_df['prediction'].value_counts().items():
        pct = count / filtered_windows * 100
        avg_conf = filtered_df[filtered_df['prediction'] == pred]['confidence'].mean()
        print(f"  {pred:25s}: {count:5d} ({pct:5.1f}%) - avg conf: {avg_conf:.3f}")
    
    # Save filtered file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_parquet(output_file)
    
    print(f"\n✓ Filtered predictions saved to: {output_file}")
    
    # Calculate time savings
    print("\n" + "="*60)
    print("ESTIMATED PROCESSING TIME")
    print("="*60)
    
    time_per_window_sec = 90  # Based on your Colab output
    
    # With k=5
    time_k5_hours = (filtered_windows * time_per_window_sec) / 3600
    # With k=3 (recommended)
    time_k3_hours = time_k5_hours * (3/5)
    
    print(f"\nWith self_consistency_k=5: {filtered_windows} × 90s = {time_k5_hours:.1f} hours")
    print(f"With self_consistency_k=3: {filtered_windows} × 54s = {time_k3_hours:.1f} hours (RECOMMENDED)")
    
    print("\n" + "="*60)
    print("NEXT STEPS FOR COLAB")
    print("="*60)
    print("\n1. Upload the filtered file to Google Drive:")
    print(f"   {output_file}")
    print("\n2. In Colab CONFIG cell, update:")
    print("   CONFIG = {")
    print(f"       'predictions_file': '/content/drive/MyDrive/fd-llm/outputs/exp_full_dataset/predictions_high_conf_faults.parquet',")
    print("       'self_consistency_k': 3,  # Recommended for speed")
    print("       # ... rest of config")
    print("   }")
    print(f"\n3. Expected processing time: ~{time_k3_hours:.1f} hours ({time_k3_hours/24:.1f} days)")
    
    return filtered_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter to high-confidence faults")
    parser.add_argument(
        "--input",
        default="outputs/exp_full_dataset/predictions_for_colab.parquet",
        help="Input predictions file"
    )
    parser.add_argument(
        "--output",
        default="outputs/exp_full_dataset/predictions_high_conf_faults.parquet",
        help="Output filtered predictions file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    filter_high_confidence_faults(args.input, args.output, args.threshold)

