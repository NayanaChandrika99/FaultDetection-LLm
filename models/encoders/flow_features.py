"""
Flow Feature Extraction
Extracts statistical and process features from flow measurements.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_flow_features(
    window_df: pd.DataFrame,
    flow_col: str = 'Slurry Flow (m3/s)',
    zero_threshold: float = 0.01,
    sample_rate: float = 1.0
) -> Dict[str, float]:
    """
    Extract flow-related features from a time window.
    
    Args:
        window_df: DataFrame for a single window
        flow_col: Name of the flow column
        zero_threshold: Threshold below which flow is considered zero (m³/s)
        sample_rate: Sampling rate in Hz
    
    Returns:
        Dictionary of flow features
    
    Features:
        - flow_mean: Average flow rate
        - flow_std: Standard deviation
        - flow_cv: Coefficient of variation (std/mean)
        - flow_min, flow_max: Range
        - flow_rate_of_change: Change rate per minute
        - flow_n_zero_events: Number of zero-flow samples
        - flow_stability: 1 - CV (higher = more stable)
        - flow_trend: Linear trend slope
    """
    if flow_col not in window_df.columns:
        raise ValueError(f"Column '{flow_col}' not found in window data")
    
    flow = window_df[flow_col].values
    flow_valid = flow[~np.isnan(flow)]
    
    if len(flow_valid) == 0:
        return {
            'flow_mean': np.nan,
            'flow_std': np.nan,
            'flow_cv': np.nan,
            'flow_min': np.nan,
            'flow_max': np.nan,
            'flow_rate_of_change': np.nan,
            'flow_n_zero_events': 0,
            'flow_stability': np.nan,
            'flow_trend': np.nan,
        }
    
    # Basic statistics
    flow_mean = np.mean(flow_valid)
    flow_std = np.std(flow_valid)
    flow_cv = flow_std / (flow_mean + 1e-6)
    
    # Rate of change (per minute)
    if len(flow_valid) > 1:
        flow_roc = (flow_valid[-1] - flow_valid[0]) / (len(flow_valid) / sample_rate / 60.0)
    else:
        flow_roc = 0.0
    
    # Zero-flow events
    n_zero = np.sum(flow_valid < zero_threshold)
    
    # Stability (inverse of CV, capped at 1)
    stability = max(0, min(1, 1 - flow_cv))
    
    # Linear trend
    if len(flow_valid) > 2:
        x = np.arange(len(flow_valid))
        trend = np.polyfit(x, flow_valid, 1)[0]  # Slope
    else:
        trend = 0.0
    
    return {
        'flow_mean': float(flow_mean),
        'flow_std': float(flow_std),
        'flow_cv': float(flow_cv),
        'flow_min': float(np.min(flow_valid)),
        'flow_max': float(np.max(flow_valid)),
        'flow_rate_of_change': float(flow_roc),
        'flow_n_zero_events': int(n_zero),
        'flow_stability': float(stability),
        'flow_trend': float(trend),
    }


def compute_mass_flow_features(
    window_df: pd.DataFrame,
    mass_flow_col: str = 'Slurry Mass Flow (kg/s)',
    solids_flow_col: Optional[str] = 'Solids Mass Flow (kg/s)'
) -> Dict[str, float]:
    """
    Extract mass flow features.
    
    Args:
        window_df: DataFrame for a single window
        mass_flow_col: Name of the mass flow column
        solids_flow_col: Optional solids mass flow column
    
    Returns:
        Dictionary of mass flow features
    """
    features = {}
    
    if mass_flow_col in window_df.columns:
        mass_flow = window_df[mass_flow_col].values
        mass_flow_valid = mass_flow[~np.isnan(mass_flow)]
        
        if len(mass_flow_valid) > 0:
            features['mass_flow_mean'] = float(np.mean(mass_flow_valid))
            features['mass_flow_std'] = float(np.std(mass_flow_valid))
        else:
            features['mass_flow_mean'] = np.nan
            features['mass_flow_std'] = np.nan
    
    if solids_flow_col and solids_flow_col in window_df.columns:
        solids_flow = window_df[solids_flow_col].values
        solids_flow_valid = solids_flow[~np.isnan(solids_flow)]
        
        if len(solids_flow_valid) > 0:
            features['solids_flow_mean'] = float(np.mean(solids_flow_valid))
            features['solids_flow_std'] = float(np.std(solids_flow_valid))
        else:
            features['solids_flow_mean'] = np.nan
            features['solids_flow_std'] = np.nan
    
    return features


def extract_all_flow_features(
    window_df: pd.DataFrame,
    include_mass_flow: bool = True
) -> Dict[str, float]:
    """
    Extract all flow-related features.
    
    Args:
        window_df: DataFrame for a single window
        include_mass_flow: Include mass flow features (default: True)
    
    Returns:
        Combined dictionary of all flow features
    """
    features = compute_flow_features(window_df)
    
    if include_mass_flow:
        mass_features = compute_mass_flow_features(window_df)
        features.update(mass_features)
    
    return features

