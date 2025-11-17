"""
Pressure and Temperature Feature Extraction
Extracts features from process variables (pressure, temperature).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats


def compute_pressure_features(
    window_df: pd.DataFrame,
    pressure_col: str = 'Pressure (kPa)',
    flow_col: Optional[str] = 'Slurry Flow (m3/s)'
) -> Dict[str, float]:
    """
    Extract pressure-related features.
    
    Args:
        window_df: DataFrame for a single window
        pressure_col: Name of pressure column
        flow_col: Optional flow column for correlation analysis
    
    Returns:
        Dictionary of pressure features
    
    Features:
        - pressure_mean, pressure_std: Basic statistics
        - pressure_variability: Normalized variability (std/mean)
        - pressure_spike_count: Number of spikes >3σ
        - pressure_flow_correlation: Correlation with flow (if available)
    """
    features = {}
    
    if pressure_col in window_df.columns:
        pressure = window_df[pressure_col].values
        pressure_valid = pressure[~np.isnan(pressure)]
        
        if len(pressure_valid) > 0:
            pressure_mean = np.mean(pressure_valid)
            pressure_std = np.std(pressure_valid)
            
            features['pressure_mean'] = float(pressure_mean)
            features['pressure_std'] = float(pressure_std)
            features['pressure_min'] = float(np.min(pressure_valid))
            features['pressure_max'] = float(np.max(pressure_valid))
            
            # Normalized variability
            pressure_variability = pressure_std / (pressure_mean + 1e-6)
            features['pressure_variability'] = float(pressure_variability)
            
            # Spike count (>3σ)
            if pressure_std > 0:
                z_scores = np.abs(pressure_valid - pressure_mean) / pressure_std
                spike_count = np.sum(z_scores > 3.0)
                features['pressure_spike_count'] = int(spike_count)
            else:
                features['pressure_spike_count'] = 0
            
            # Correlation with flow
            if flow_col and flow_col in window_df.columns:
                flow = window_df[flow_col].values
                # Align valid indices
                valid_mask = ~(np.isnan(pressure) | np.isnan(flow))
                if np.sum(valid_mask) > 2:
                    corr, _ = stats.pearsonr(pressure[valid_mask], flow[valid_mask])
                    features['pressure_flow_correlation'] = float(corr)
                else:
                    features['pressure_flow_correlation'] = np.nan
            else:
                features['pressure_flow_correlation'] = np.nan
        else:
            features.update({
                'pressure_mean': np.nan,
                'pressure_std': np.nan,
                'pressure_min': np.nan,
                'pressure_max': np.nan,
                'pressure_variability': np.nan,
                'pressure_spike_count': 0,
                'pressure_flow_correlation': np.nan,
            })
    
    return features


def compute_temperature_features(
    window_df: pd.DataFrame,
    temp_col: str = 'Temperature (C)',
    sample_rate: float = 1.0
) -> Dict[str, float]:
    """
    Extract temperature features.
    
    Args:
        window_df: DataFrame for a single window
        temp_col: Name of temperature column
        sample_rate: Sampling rate in Hz
    
    Returns:
        Dictionary of temperature features
    
    Features:
        - temp_mean: Average temperature
        - temp_change_per_10min: Rate of change per 10 minutes
        - temp_state: Categorical state (normal/elevated/high)
    
    Temperature states:
        - Normal: <30°C
        - Elevated: 30-40°C
        - High: >40°C
    """
    features = {}
    
    if temp_col in window_df.columns:
        temp = window_df[temp_col].values
        temp_valid = temp[~np.isnan(temp)]
        
        if len(temp_valid) > 0:
            temp_mean = np.mean(temp_valid)
            features['temp_mean'] = float(temp_mean)
            features['temp_std'] = float(np.std(temp_valid))
            
            # Change per 10 minutes
            if len(temp_valid) > 1:
                window_duration_min = len(temp_valid) / sample_rate / 60.0
                temp_change = temp_valid[-1] - temp_valid[0]
                temp_change_per_10min = (temp_change / window_duration_min) * 10.0
                features['temp_change_per_10min'] = float(temp_change_per_10min)
            else:
                features['temp_change_per_10min'] = 0.0
            
            # Temperature state
            if temp_mean < 30:
                temp_state = 0  # Normal
            elif temp_mean < 40:
                temp_state = 1  # Elevated
            else:
                temp_state = 2  # High
            features['temp_state'] = int(temp_state)
        else:
            features.update({
                'temp_mean': np.nan,
                'temp_std': np.nan,
                'temp_change_per_10min': np.nan,
                'temp_state': 0,
            })
    
    return features


def extract_all_process_features(
    window_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Extract all process variable features.
    
    Args:
        window_df: DataFrame for a single window
    
    Returns:
        Combined dictionary of all process features
    """
    features = {}
    
    # Pressure features
    pressure_features = compute_pressure_features(window_df)
    features.update(pressure_features)
    
    # Temperature features
    temp_features = compute_temperature_features(window_df)
    features.update(temp_features)
    
    return features

