"""
Example Usage Script
Demonstrates the complete FD-LLM workflow.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import modules
from data.loaders.slurry_loader import load_and_window
from data.loaders.window_creator import assign_labels
from utils.physical_checks import filter_windows_by_physics
from models.rocket_heads import MultiROCKETClassifier
from models.fusion import LateFusionClassifier
from models.encoders.feature_extractor import extract_window_features
from evaluation.metrics import evaluate_full
from sklearn.model_selection import train_test_split


def example_data_loading():
    """Example: Load and process CSV data."""
    print("\n" + "="*60)
    print("EXAMPLE 1: DATA LOADING")
    print("="*60)
    
    csv_path = "data/raw/your_data.csv"  # Replace with your file
    
    # Load and create windows
    windows, metadata = load_and_window(
        csv_path=csv_path,
        window_sec=60,
        stride_sec=15,
        sample_rate=1.0
    )
    
    print(f"Created {len(windows)} windows")
    print(f"Window shape: {windows.shape}")
    print(f"Metadata columns: {list(metadata.columns)}")
    
    return windows, metadata


def example_feature_extraction():
    """Example: Extract features from window."""
    print("\n" + "="*60)
    print("EXAMPLE 2: FEATURE EXTRACTION")
    print("="*60)
    
    # Create sample window DataFrame
    window_data = {
        'Slurry Flow (m3/s)': np.random.normal(2.0, 0.1, 60),
        'Slurry Density (kg/m3)': np.random.normal(1200, 20, 60),
        'Pressure (kPa)': np.random.normal(500, 30, 60),
        'Temperature (C)': np.random.normal(25, 2, 60),
    }
    window_df = pd.DataFrame(window_data)
    
    # Extract features
    features = extract_window_features(window_df)
    
    print("Extracted features:")
    for name, value in list(features.items())[:10]:
        print(f"  {name}: {value:.4f}")
    
    return features


def example_training():
    """Example: Train MultiROCKET classifier."""
    print("\n" + "="*60)
    print("EXAMPLE 3: MODEL TRAINING")
    print("="*60)
    
    # Generate synthetic data for demonstration
    n_samples = 100
    n_sensors = 11
    window_length = 60
    
    X = np.random.randn(n_samples, n_sensors, window_length)
    y = np.random.choice(['Normal', 'Pipeline Blockage', 'Air Entrainment'], n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train baseline classifier
    print("\n1. Training Baseline MultiROCKET...")
    baseline_model = MultiROCKETClassifier(
        n_kernels=1000,  # Reduced for speed
        alphas=np.logspace(-3, 3, 5)
    )
    baseline_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = baseline_model.predict(X_test)
    confidence = baseline_model.get_confidence(X_test)
    
    print(f"Test predictions: {len(y_pred)}")
    print(f"Mean confidence: {confidence.mean():.3f}")
    
    # Train late fusion model
    print("\n2. Training Late Fusion Classifier...")
    fusion_model = LateFusionClassifier(
        sensor_groups={
            'flow': [0, 1, 2, 3],
            'density': [4, 5, 6, 7],
            'process': [8, 9, 10],
        },
        kernels_per_group={
            'flow': 500,
            'density': 300,
            'process': 200,
        }
    )
    fusion_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_fusion = fusion_model.predict(X_test)
    confidence_fusion = fusion_model.get_confidence(X_test)
    
    print(f"Fusion predictions: {len(y_pred_fusion)}")
    print(f"Fusion mean confidence: {confidence_fusion.mean():.3f}")
    
    return baseline_model, fusion_model, X_test, y_test


def example_evaluation():
    """Example: Evaluate model performance."""
    print("\n" + "="*60)
    print("EXAMPLE 4: MODEL EVALUATION")
    print("="*60)
    
    # Generate synthetic predictions
    n_samples = 100
    classes = ['Normal', 'Pipeline Blockage', 'Air Entrainment']
    
    y_true = np.random.choice(classes, n_samples)
    y_pred = np.random.choice(classes, n_samples)
    
    # Create probability matrix
    y_proba = np.random.dirichlet([2, 1, 1], n_samples)
    
    # Evaluate
    results = evaluate_full(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=classes,
        output_dir='outputs/example_evaluation'
    )
    
    print(f"\nMacro F1: {results['macro_f1']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Mean Confidence: {results.get('mean_confidence', 0):.4f}")
    
    return results


def example_llm_explanation():
    """Example: Generate LLM explanation (requires GPU)."""
    print("\n" + "="*60)
    print("EXAMPLE 5: LLM EXPLANATION (REQUIRES GPU)")
    print("="*60)
    
    # Sample features
    features = {
        'flow_mean': 1.2,
        'flow_std': 0.15,
        'flow_cv': 0.125,
        'flow_n_zero_events': 0,
        'density_mean': 1250.0,
        'density_std': 25.0,
        'density_trend': 5.0,
        'sg_mean': 1.25,
        'sg_target_deviation': 0.05,
        'pressure_mean': 545.0,
        'pressure_variability': 0.08,
        'temp_mean': 28.0,
    }
    
    prediction = "Pipeline Blockage"
    confidence = 0.85
    
    print(f"\nClassifier Prediction: {prediction} (confidence: {confidence:.3f})")
    print("\nTo generate LLM explanation, run:")
    print("  from explainer.llm_setup import LLMExplainer")
    print("  from explainer.self_consistency import explain_with_self_consistency")
    print()
    print("  llm = LLMExplainer(load_in_4bit=True)")
    print("  explanation = explain_with_self_consistency(")
    print("      llm_explainer=llm,")
    print("      features=features,")
    print("      prediction=prediction,")
    print("      confidence=confidence,")
    print("      k=5")
    print("  )")
    
    print("\nNote: This requires ~8-10GB GPU memory")


def example_robustness_testing():
    """Example: Run robustness tests."""
    print("\n" + "="*60)
    print("EXAMPLE 6: ROBUSTNESS TESTING")
    print("="*60)
    
    from evaluation.robustness_tests import (
        test_noise_robustness,
        test_sensor_dropout_robustness
    )
    
    # Generate synthetic test data
    n_samples = 50
    n_sensors = 11
    window_length = 60
    
    X_test = np.random.randn(n_samples, n_sensors, window_length)
    y_test = np.random.choice(['Normal', 'Blockage'], n_samples)
    
    # Create simple mock model
    class MockModel:
        def predict(self, X):
            return np.random.choice(['Normal', 'Blockage'], len(X))
    
    model = MockModel()
    
    print("\n1. Testing noise robustness...")
    noise_results = test_noise_robustness(
        model, X_test, y_test,
        snr_levels=[30, 20, 10]
    )
    
    print("\n2. Testing sensor dropout...")
    dropout_results = test_sensor_dropout_robustness(
        model, X_test, y_test,
        sensor_groups={
            'flow': [0, 1, 2, 3],
            'density': [4, 5, 6, 7],
        }
    )
    
    return noise_results, dropout_results


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("FD-LLM EXAMPLE USAGE")
    print("="*60)
    print("\nThis script demonstrates the complete FD-LLM workflow.")
    print("Some examples use synthetic data for demonstration.")
    
    # Example 2: Feature extraction (works without data)
    example_feature_extraction()
    
    # Example 3: Training (synthetic data)
    example_training()
    
    # Example 4: Evaluation (synthetic data)
    example_evaluation()
    
    # Example 5: LLM explanation (info only)
    example_llm_explanation()
    
    # Example 6: Robustness testing (synthetic data)
    example_robustness_testing()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETE")
    print("="*60)
    print("\nFor real usage:")
    print("1. Place your CSV data in data/raw/")
    print("2. Run: python training/train_rocket.py --data data/raw/your_data.csv")
    print("3. Run: python explainer/run_explainer.py --pred_file predictions.parquet")
    print("\nSee README.md for more details.")


if __name__ == '__main__':
    main()

