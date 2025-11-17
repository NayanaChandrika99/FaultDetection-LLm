"""
Robustness Testing Suite
Tests model performance under various perturbations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def add_gaussian_noise(
    X: np.ndarray,
    snr_db: float
) -> np.ndarray:
    """
    Add Gaussian noise at specified SNR level.
    
    Args:
        X: Input data [n_samples, n_channels, length]
        snr_db: Signal-to-noise ratio in dB
    
    Returns:
        Noisy data
    """
    X_noisy = X.copy()
    
    for i in range(X.shape[0]):
        for ch in range(X.shape[1]):
            signal = X[i, ch, :]
            signal_power = np.mean(signal ** 2)
            
            # Calculate noise power from SNR
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            
            # Generate and add noise
            noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
            X_noisy[i, ch, :] = signal + noise
    
    return X_noisy


def apply_calibration_drift(
    X: np.ndarray,
    sensor_indices: List[int],
    drift_percent: float
) -> np.ndarray:
    """
    Apply systematic calibration drift to specified sensors.
    
    Args:
        X: Input data [n_samples, n_channels, length]
        sensor_indices: List of sensor indices to perturb
        drift_percent: Drift percentage (e.g., 5.0 for +5%)
    
    Returns:
        Drifted data
    """
    X_drifted = X.copy()
    multiplier = 1.0 + (drift_percent / 100.0)
    
    for sensor_idx in sensor_indices:
        X_drifted[:, sensor_idx, :] *= multiplier
    
    return X_drifted


def simulate_sensor_dropout(
    X: np.ndarray,
    sensor_indices: List[int],
    replacement_value: float = 0.0
) -> np.ndarray:
    """
    Simulate sensor dropout by replacing with constant value.
    
    Args:
        X: Input data [n_samples, n_channels, length]
        sensor_indices: List of sensor indices to drop
        replacement_value: Value to replace with (default: 0.0)
    
    Returns:
        Data with dropped sensors
    """
    X_dropped = X.copy()
    
    for sensor_idx in sensor_indices:
        X_dropped[:, sensor_idx, :] = replacement_value
    
    return X_dropped


def test_noise_robustness(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    snr_levels: List[float] = [30, 20, 10, 5]
) -> Dict[float, Dict[str, float]]:
    """
    Test model robustness to noise at various SNR levels.
    
    Args:
        model: Trained classifier
        X_test: Test data
        y_test: Test labels
        snr_levels: List of SNR levels in dB
    
    Returns:
        Dict mapping SNR to metrics
    """
    results = {}
    
    # Baseline (no noise)
    y_pred_baseline = model.predict(X_test)
    baseline_f1 = f1_score(y_test, y_pred_baseline, average='macro')
    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    
    results['baseline'] = {
        'macro_f1': float(baseline_f1),
        'accuracy': float(baseline_acc),
        'degradation_f1': 0.0,
        'degradation_acc': 0.0,
    }
    
    print(f"\nNoise Robustness Test:")
    print(f"  Baseline - F1: {baseline_f1:.4f}, Acc: {baseline_acc:.4f}")
    
    # Test each SNR level
    for snr_db in snr_levels:
        X_noisy = add_gaussian_noise(X_test, snr_db)
        y_pred = model.predict(X_noisy)
        
        f1 = f1_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)
        
        degradation_f1 = ((baseline_f1 - f1) / baseline_f1) * 100
        degradation_acc = ((baseline_acc - acc) / baseline_acc) * 100
        
        results[snr_db] = {
            'macro_f1': float(f1),
            'accuracy': float(acc),
            'degradation_f1': float(degradation_f1),
            'degradation_acc': float(degradation_acc),
        }
        
        print(f"  SNR {snr_db}dB - F1: {f1:.4f} ({degradation_f1:+.1f}%), Acc: {acc:.4f} ({degradation_acc:+.1f}%)")
    
    return results


def test_sensor_dropout_robustness(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensor_groups: Dict[str, List[int]]
) -> Dict[str, Dict[str, float]]:
    """
    Test model robustness to individual sensor dropouts.
    
    Args:
        model: Trained classifier
        X_test: Test data
        y_test: Test labels
        sensor_groups: Dict mapping group name to sensor indices
    
    Returns:
        Dict mapping dropped sensor group to metrics
    """
    results = {}
    
    # Baseline
    y_pred_baseline = model.predict(X_test)
    baseline_f1 = f1_score(y_test, y_pred_baseline, average='macro')
    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    
    results['baseline'] = {
        'macro_f1': float(baseline_f1),
        'accuracy': float(baseline_acc),
        'degradation_f1': 0.0,
    }
    
    print(f"\nSensor Dropout Robustness Test:")
    print(f"  Baseline - F1: {baseline_f1:.4f}, Acc: {baseline_acc:.4f}")
    
    # Test dropping each sensor group
    for group_name, sensor_indices in sensor_groups.items():
        X_dropped = simulate_sensor_dropout(X_test, sensor_indices)
        y_pred = model.predict(X_dropped)
        
        f1 = f1_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)
        
        degradation_f1 = ((baseline_f1 - f1) / baseline_f1) * 100
        
        results[group_name] = {
            'macro_f1': float(f1),
            'accuracy': float(acc),
            'degradation_f1': float(degradation_f1),
        }
        
        print(f"  Dropped {group_name} - F1: {f1:.4f} ({degradation_f1:+.1f}%), Acc: {acc:.4f}")
    
    return results


def test_calibration_drift_robustness(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensor_groups: Dict[str, List[int]],
    drift_levels: List[float] = [5, 10]
) -> Dict[str, Dict[float, Dict[str, float]]]:
    """
    Test model robustness to calibration drift.
    
    Args:
        model: Trained classifier
        X_test: Test data
        y_test: Test labels
        sensor_groups: Dict mapping group name to sensor indices
        drift_levels: List of drift percentages
    
    Returns:
        Nested dict mapping group -> drift_level -> metrics
    """
    results = {}
    
    # Baseline
    y_pred_baseline = model.predict(X_test)
    baseline_f1 = f1_score(y_test, y_pred_baseline, average='macro')
    
    print(f"\nCalibration Drift Robustness Test:")
    print(f"  Baseline F1: {baseline_f1:.4f}")
    
    # Test each sensor group
    for group_name, sensor_indices in sensor_groups.items():
        results[group_name] = {}
        print(f"\n  Testing {group_name} drift:")
        
        for drift_percent in drift_levels:
            X_drifted = apply_calibration_drift(X_test, sensor_indices, drift_percent)
            y_pred = model.predict(X_drifted)
            
            f1 = f1_score(y_test, y_pred, average='macro')
            acc = accuracy_score(y_test, y_pred)
            
            degradation_f1 = ((baseline_f1 - f1) / baseline_f1) * 100
            
            results[group_name][drift_percent] = {
                'macro_f1': float(f1),
                'accuracy': float(acc),
                'degradation_f1': float(degradation_f1),
            }
            
            print(f"    {drift_percent:+.0f}% drift - F1: {f1:.4f} ({degradation_f1:+.1f}%)")
    
    return results


def plot_robustness_results(
    noise_results: Dict,
    dropout_results: Dict,
    drift_results: Dict,
    save_dir: Optional[str] = None
):
    """
    Plot robustness test results.
    
    Args:
        noise_results: Results from noise test
        dropout_results: Results from dropout test
        drift_results: Results from drift test
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Noise robustness
    snr_levels = [k for k in noise_results.keys() if k != 'baseline']
    f1_scores = [noise_results[snr]['macro_f1'] for snr in snr_levels]
    baseline_f1 = noise_results['baseline']['macro_f1']
    
    axes[0].plot(snr_levels, f1_scores, marker='o', linewidth=2, markersize=8)
    axes[0].axhline(y=baseline_f1, color='green', linestyle='--', label='Baseline')
    axes[0].set_xlabel('SNR (dB)', fontsize=12)
    axes[0].set_ylabel('Macro F1', fontsize=12)
    axes[0].set_title('Noise Robustness', fontsize=14)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    
    # Sensor dropout
    groups = [k for k in dropout_results.keys() if k != 'baseline']
    f1_scores = [dropout_results[g]['macro_f1'] for g in groups]
    
    axes[1].bar(range(len(groups)), f1_scores, color='steelblue', alpha=0.8)
    axes[1].axhline(y=baseline_f1, color='green', linestyle='--', label='Baseline')
    axes[1].set_xlabel('Dropped Sensor Group', fontsize=12)
    axes[1].set_ylabel('Macro F1', fontsize=12)
    axes[1].set_title('Sensor Dropout Robustness', fontsize=14)
    axes[1].set_xticks(range(len(groups)))
    axes[1].set_xticklabels(groups, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend()
    
    # Calibration drift
    # Show heatmap of degradation
    groups = list(drift_results.keys())
    drift_levels = list(drift_results[groups[0]].keys())
    degradation_matrix = np.array([
        [drift_results[g][d]['degradation_f1'] for d in drift_levels]
        for g in groups
    ])
    
    sns.heatmap(
        degradation_matrix,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        xticklabels=[f'{d:+.0f}%' for d in drift_levels],
        yticklabels=groups,
        ax=axes[2],
        cbar_kws={'label': 'F1 Degradation (%)'}
    )
    axes[2].set_xlabel('Calibration Drift', fontsize=12)
    axes[2].set_ylabel('Sensor Group', fontsize=12)
    axes[2].set_title('Calibration Drift Robustness', fontsize=14)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'robustness_tests.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Robustness plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def run_robustness_suite(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensor_groups: Optional[Dict[str, List[int]]] = None,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Run complete robustness test suite.
    
    Args:
        model: Trained classifier
        X_test: Test data
        y_test: Test labels
        sensor_groups: Sensor group definitions
        output_dir: Directory to save results
    
    Returns:
        Dict of all robustness test results
    """
    if sensor_groups is None:
        sensor_groups = {
            'flow': [0, 1, 2, 3],
            'density': [4, 5, 6, 7],
            'process': [8, 9, 10],
        }
    
    print(f"\n{'='*60}")
    print("ROBUSTNESS TEST SUITE")
    print(f"{'='*60}")
    
    # Run tests
    noise_results = test_noise_robustness(model, X_test, y_test)
    dropout_results = test_sensor_dropout_robustness(model, X_test, y_test, sensor_groups)
    drift_results = test_calibration_drift_robustness(model, X_test, y_test, sensor_groups)
    
    # Combine results
    all_results = {
        'noise': noise_results,
        'sensor_dropout': dropout_results,
        'calibration_drift': drift_results,
    }
    
    # Save and plot
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        import json
        with open(output_dir / 'robustness_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Plot
        plot_robustness_results(noise_results, dropout_results, drift_results, str(output_dir))
    
    return all_results

