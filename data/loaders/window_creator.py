"""
Window Creation and Labeling
Handles label assignment from heuristics, maintenance logs, and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum


class FaultClass(Enum):
    """Fault classification labels"""
    NORMAL = "Normal"
    PUMP_CAVITATION = "Pump Cavitation"
    PIPELINE_BLOCKAGE = "Pipeline Blockage"
    SETTLING_SEGREGATION = "Settling/Segregation"
    AIR_ENTRAINMENT = "Air Entrainment"
    DILUTION = "Dilution"
    CONCENTRATION = "Concentration"
    VALVE_TRANSIENT = "Valve Transient"
    INSTRUMENT_FAULT = "Instrument Fault"
    PROCESS_UPSET = "Process Upset"


def label_from_heuristics(
    window_df: pd.DataFrame,
    thresholds: Optional[Dict] = None
) -> str:
    """
    Apply threshold-based heuristics to assign labels.
    
    Args:
        window_df: DataFrame for a single window
        thresholds: Dict of threshold parameters (uses defaults if None)
    
    Returns:
        Label string
    
    Heuristics:
        - All zero flow → Valve Transient or Planned Shutdown
        - High density spike (>3σ) → Settling/Segregation or Blockage
        - Low density spike (<-2σ) → Air Entrainment or Dilution
        - High pressure + low flow → Pipeline Blockage
        - Flow oscillations + low density → Air Entrainment
    """
    if thresholds is None:
        thresholds = {
            'zero_flow_threshold': 0.01,  # m³/s
            'density_spike_sigma': 3.0,
            'pressure_spike_sigma': 2.5,
            'cv_unstable': 0.15,
        }
    
    # Extract key sensors
    flow = window_df.get('Slurry Flow (m3/s)', pd.Series([np.nan]))
    density = window_df.get('Slurry Density (kg/m3)', pd.Series([np.nan]))
    pressure = window_df.get('Pressure (kPa)', pd.Series([np.nan]))
    
    flow_mean = flow.mean()
    flow_std = flow.std()
    density_mean = density.mean()
    density_std = density.std()
    
    # Rule 1: Zero flow events
    if flow_mean < thresholds['zero_flow_threshold']:
        # Check if gradual decrease (planned) vs sudden (fault)
        flow_change_rate = abs(flow.iloc[-1] - flow.iloc[0]) / len(flow)
        if flow_change_rate < 0.001:  # Gradual
            return FaultClass.VALVE_TRANSIENT.value
        else:
            return FaultClass.PROCESS_UPSET.value
    
    # Rule 2: High density spike
    if not np.isnan(density_mean) and not np.isnan(density_std):
        density_zscore = (density.max() - density_mean) / (density_std + 1e-6)
        if density_zscore > thresholds['density_spike_sigma']:
            # Check if flow also decreased → blockage
            if flow_mean < flow.quantile(0.25):
                return FaultClass.PIPELINE_BLOCKAGE.value
            else:
                return FaultClass.SETTLING_SEGREGATION.value
    
    # Rule 3: Low density spike
    if not np.isnan(density_mean) and not np.isnan(density_std):
        density_zscore_low = (density_mean - density.min()) / (density_std + 1e-6)
        if density_zscore_low > 2.0:
            # Check flow variability
            flow_cv = flow_std / (flow_mean + 1e-6)
            if flow_cv > thresholds['cv_unstable']:
                return FaultClass.AIR_ENTRAINMENT.value
            else:
                return FaultClass.DILUTION.value
    
    # Rule 4: High pressure + low flow
    if not pressure.isna().all():
        pressure_mean = pressure.mean()
        if pressure_mean > pressure.quantile(0.75) and flow_mean < flow.quantile(0.25):
            return FaultClass.PIPELINE_BLOCKAGE.value
    
    # Default to Normal
    return FaultClass.NORMAL.value


def label_from_maintenance_logs(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    logs: List[Dict]
) -> Optional[str]:
    """
    Assign label based on maintenance log entries.
    
    Args:
        window_start: Start time of window
        window_end: End time of window
        logs: List of maintenance log dicts with 'start', 'end', 'fault' keys
    
    Returns:
        Label string if window overlaps with log, None otherwise
    
    Example log entry:
        {
            'start': pd.Timestamp('2024-01-01 10:00:00'),
            'end': pd.Timestamp('2024-01-01 10:15:00'),
            'fault': 'Pipeline Blockage'
        }
    """
    for log in logs:
        log_start = log['start']
        log_end = log['end']
        
        # Check for overlap
        if window_start <= log_end and window_end >= log_start:
            return log['fault']
    
    return None


def assign_labels(
    windows: np.ndarray,
    metadata: pd.DataFrame,
    df: pd.DataFrame,
    maintenance_logs: Optional[List[Dict]] = None,
    use_heuristics: bool = True
) -> np.ndarray:
    """
    Assign labels to all windows using maintenance logs and/or heuristics.
    
    Args:
        windows: Array of shape [n_windows, n_sensors, window_sec]
        metadata: DataFrame with window metadata (start_time, end_time)
        df: Original DataFrame with sensor data
        maintenance_logs: Optional list of maintenance log dicts
        use_heuristics: Use threshold heuristics if no log match (default: True)
    
    Returns:
        Array of label strings, shape [n_windows]
    """
    labels = []
    
    for i in range(len(windows)):
        window_start = metadata.iloc[i]['start_time']
        window_end = metadata.iloc[i]['end_time']
        
        # Try maintenance logs first
        label = None
        if maintenance_logs:
            label = label_from_maintenance_logs(window_start, window_end, maintenance_logs)
        
        # Fall back to heuristics if no log match
        if label is None and use_heuristics:
            # Get window data
            start_idx = df.index.get_loc(window_start)
            end_idx = df.index.get_loc(window_end) + 1
            window_df = df.iloc[start_idx:end_idx]
            label = label_from_heuristics(window_df)
        
        # Default to Normal if still no label
        if label is None:
            label = FaultClass.NORMAL.value
        
        labels.append(label)
    
    return np.array(labels)


def validate_window_quality(
    window: np.ndarray,
    min_valid_ratio: float = 0.9
) -> Tuple[bool, str]:
    """
    Validate window data quality.
    
    Args:
        window: Single window array of shape [n_sensors, window_sec]
        min_valid_ratio: Minimum ratio of non-NaN values required
    
    Returns:
        Tuple of (is_valid, reason)
    """
    # Check for missing data
    valid_ratio = np.sum(~np.isnan(window)) / window.size
    if valid_ratio < min_valid_ratio:
        return False, f"insufficient_data (valid={valid_ratio:.2%})"
    
    # Check for constant values (sensor fault indicator)
    for sensor_idx in range(window.shape[0]):
        sensor_data = window[sensor_idx, :]
        if np.std(sensor_data[~np.isnan(sensor_data)]) < 1e-6:
            return False, f"constant_sensor_{sensor_idx}"
    
    # Check for extreme outliers (>10 sigma from mean)
    for sensor_idx in range(window.shape[0]):
        sensor_data = window[sensor_idx, :]
        valid_data = sensor_data[~np.isnan(sensor_data)]
        if len(valid_data) > 0:
            mean = np.mean(valid_data)
            std = np.std(valid_data)
            if std > 0 and np.any(np.abs(valid_data - mean) > 10 * std):
                return False, f"extreme_outlier_sensor_{sensor_idx}"
    
    return True, "valid"


def filter_windows_by_quality(
    windows: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    min_valid_ratio: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Filter out low-quality windows.
    
    Args:
        windows: Array of shape [n_windows, n_sensors, window_sec]
        labels: Array of label strings
        metadata: DataFrame with window metadata
        min_valid_ratio: Minimum valid data ratio
    
    Returns:
        Tuple of filtered (windows, labels, metadata)
    """
    valid_indices = []
    
    for i in range(len(windows)):
        is_valid, reason = validate_window_quality(windows[i], min_valid_ratio)
        if is_valid:
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        raise ValueError("No valid windows after quality filtering")
    
    filtered_windows = windows[valid_indices]
    filtered_labels = labels[valid_indices]
    filtered_metadata = metadata.iloc[valid_indices].reset_index(drop=True)
    
    return filtered_windows, filtered_labels, filtered_metadata

