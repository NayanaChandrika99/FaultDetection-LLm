"""
Export Predictions for Colab LLM Processing
Creates a parquet file with predictions and features for Google Colab GPU processing.
"""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path

from data.loaders.slurry_loader import load_and_window, load_slurry_csv
from models.rocket_heads import MultiROCKETClassifier
from models.fusion import LateFusionClassifier
from models.encoders.feature_extractor import extract_window_features


def export_predictions_with_features(
    model_path: str,
    csv_path: str,
    output_path: str,
    max_samples: int = None
):
    """
    Export predictions with extracted features for Colab processing.
    
    Args:
        model_path: Path to trained model (.pkl)
        csv_path: Path to original CSV data
        output_path: Path to save parquet file
        max_samples: Optional limit on number of windows
    """
    print(f"\n{'='*60}")
    print("EXPORTING FOR COLAB")
    print(f"{'='*60}")
    
    # Load trained model
    print(f"\nLoading model from: {model_path}")
    try:
        model = MultiROCKETClassifier.load(model_path)
        print("  ✓ Loaded MultiROCKET model")
    except:
        model = LateFusionClassifier.load(model_path)
        print("  ✓ Loaded Late Fusion model")
    
    # Load and create windows
    print(f"\nLoading data from: {csv_path}")
    windows, metadata = load_and_window(
        csv_path=csv_path,
        window_sec=60,
        stride_sec=15
    )
    print(f"  ✓ Created {len(windows)} windows")
    
    # Limit samples if requested
    if max_samples and max_samples < len(windows):
        print(f"  Limiting to {max_samples} samples")
        windows = windows[:max_samples]
        metadata = metadata.iloc[:max_samples]
    
    # Handle NaN values (MultiRocket cannot handle them)
    print("\nHandling missing values...")
    nan_count_before = np.isnan(windows).sum()
    if nan_count_before > 0:
        print(f"  Found {nan_count_before} NaN values ({nan_count_before/windows.size*100:.3f}%)")
        for i in range(len(windows)):
            for j in range(windows.shape[1]):
                series = windows[i, j, :]
                mask = np.isnan(series)
                if mask.any():
                    indices = np.where(~mask)[0]
                    if len(indices) > 0:
                        series[mask] = np.interp(np.where(mask)[0], indices, series[indices])
                    if np.isnan(series).any():
                        mean_val = np.nanmean(series)
                        if not np.isnan(mean_val):
                            series[np.isnan(series)] = mean_val
                        else:
                            series[np.isnan(series)] = 0
                windows[i, j, :] = series
        print(f"  ✓ All NaN values handled")
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = model.predict(windows)
    confidences = model.get_confidence(windows)
    print(f"  ✓ Generated predictions")
    
    # Extract features for each window
    print("\nExtracting features...")
    df = load_slurry_csv(csv_path)
    features_list = []
    
    for i in range(len(windows)):
        start_time = metadata.iloc[i]['start_time']
        end_time = metadata.iloc[i]['end_time']
        start_idx = df.index.get_loc(start_time)
        end_idx = df.index.get_loc(end_time) + 1
        window_df = df.iloc[start_idx:end_idx]
        
        features = extract_window_features(window_df)
        features_list.append(features)
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(windows)}")
    
    print(f"  ✓ Extracted features for all windows")
    
    # Create export DataFrame
    print("\nCreating export file...")
    export_df = pd.DataFrame({
        'window_id': range(len(windows)),
        'prediction': predictions,
        'confidence': confidences,
        'features': [json.dumps(f) for f in features_list]
    })
    
    # Save as parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_df.to_parquet(output_path, compression='snappy')
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n{'='*60}")
    print("EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Exported: {len(export_df)} predictions")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Saved to: {output_path}")
    
    # Show preview
    print(f"\nPreview:")
    print(export_df[['window_id', 'prediction', 'confidence']].head())
    
    # Show prediction distribution
    print(f"\nPrediction Distribution:")
    pred_counts = pd.Series(predictions).value_counts()
    for pred, count in pred_counts.items():
        print(f"  {pred}: {count} ({count/len(predictions)*100:.1f}%)")
    
    print(f"\nNext steps:")
    print(f"  1. Upload this file to Google Drive")
    print(f"  2. Copy explainer/ folder to Google Drive")
    print(f"  3. Follow instructions in COLAB_SETUP.md")


def main():
    parser = argparse.ArgumentParser(description='Export predictions for Colab processing')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pkl)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to original CSV data')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for parquet file (default: same dir as model)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to export')
    
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        model_dir = Path(args.model).parent
        args.output = str(model_dir / 'predictions_for_colab.parquet')
    
    export_predictions_with_features(
        model_path=args.model,
        csv_path=args.data,
        output_path=args.output,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()

