"""
Filter predictions to only include fault windows (exclude Normal).
This dramatically reduces the number of windows that need LLM explanations.
"""

import pandas as pd
from pathlib import Path

def filter_fault_windows(
    input_file: str = "outputs/exp_full_dataset/predictions_for_colab.parquet",
    output_file: str = "outputs/exp_full_dataset/predictions_faults_only.parquet"
):
    """
    Filter predictions to only include fault windows.
    
    Args:
        input_file: Path to full predictions file
        output_file: Path to save filtered predictions
    """
    print("="*60)
    print("FILTERING PREDICTIONS TO FAULTS ONLY")
    print("="*60)
    
    # Load predictions
    print(f"\nLoading predictions from: {input_file}")
    pred_df = pd.read_parquet(input_file)
    total_windows = len(pred_df)
    print(f"  Total windows: {total_windows}")
    
    # Show distribution
    print("\nPrediction Distribution (BEFORE filtering):")
    prediction_counts = pred_df['prediction'].value_counts()
    for pred, count in prediction_counts.items():
        pct = count / total_windows * 100
        print(f"  {pred}: {count} ({pct:.1f}%)")
    
    # Filter to only faults (exclude "Normal")
    print("\nFiltering to fault windows only...")
    fault_df = pred_df[pred_df['prediction'] != 'Normal'].copy()
    fault_windows = len(fault_df)
    
    print(f"\n  ✓ Filtered: {fault_windows} fault windows")
    print(f"  ✓ Removed: {total_windows - fault_windows} normal windows")
    print(f"  ✓ Reduction: {(1 - fault_windows/total_windows)*100:.1f}%")
    
    # Show fault distribution
    print("\nFault Distribution (AFTER filtering):")
    fault_counts = fault_df['prediction'].value_counts()
    for fault, count in fault_counts.items():
        pct = count / fault_windows * 100
        print(f"  {fault}: {count} ({pct:.1f}%)")
    
    # Save filtered file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fault_df.to_parquet(output_file)
    
    print(f"\n✓ Filtered predictions saved to: {output_file}")
    
    # Calculate time savings
    print("\n" + "="*60)
    print("ESTIMATED TIME SAVINGS")
    print("="*60)
    
    time_per_window = 90  # seconds (based on your output)
    
    original_time_hours = (total_windows * time_per_window) / 3600
    filtered_time_hours = (fault_windows * time_per_window) / 3600
    time_saved_hours = original_time_hours - filtered_time_hours
    
    print(f"\nOriginal (all windows): {total_windows} × 90s = {original_time_hours:.1f} hours")
    print(f"Filtered (faults only): {fault_windows} × 90s = {filtered_time_hours:.1f} hours")
    print(f"\n⚡ TIME SAVED: {time_saved_hours:.1f} hours ({time_saved_hours/24:.1f} days)")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Upload the new file to Google Drive:")
    print(f"   {output_file}")
    print("\n2. In Colab, update CONFIG to use the filtered file:")
    print("   CONFIG['predictions_file'] = '/content/drive/MyDrive/fd-llm/outputs/exp_full_dataset/predictions_faults_only.parquet'")
    print("\n3. Re-run the explanation generation")
    print(f"   Expected time: ~{filtered_time_hours:.1f} hours instead of {original_time_hours:.1f} hours")
    
    return fault_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter predictions to fault windows only")
    parser.add_argument(
        "--input",
        default="outputs/exp_full_dataset/predictions_for_colab.parquet",
        help="Input predictions file"
    )
    parser.add_argument(
        "--output",
        default="outputs/exp_full_dataset/predictions_faults_only.parquet",
        help="Output filtered predictions file"
    )
    
    args = parser.parse_args()
    
    filter_fault_windows(args.input, args.output)

