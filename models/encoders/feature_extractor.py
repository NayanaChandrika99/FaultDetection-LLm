"""
Unified Feature Extraction
Combines all feature extractors into a single interface.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from .flow_features import extract_all_flow_features
from .density_features import extract_all_density_features
from .pressure_features import extract_all_process_features


def extract_window_features(
    window_df: pd.DataFrame,
    include_flow: bool = True,
    include_density: bool = True,
    include_process: bool = True
) -> Dict[str, float]:
    """
    Extract all features from a single window.
    
    Args:
        window_df: DataFrame for a single window
        include_flow: Include flow features (default: True)
        include_density: Include density features (default: True)
        include_process: Include process features (default: True)
    
    Returns:
        Dictionary of all features
    """
    features = {}
    
    if include_flow:
        flow_features = extract_all_flow_features(window_df)
        features.update(flow_features)
    
    if include_density:
        density_features = extract_all_density_features(window_df)
        features.update(density_features)
    
    if include_process:
        process_features = extract_all_process_features(window_df)
        features.update(process_features)
    
    return features


def extract_all_features(
    window_dfs: List[pd.DataFrame],
    include_flow: bool = True,
    include_density: bool = True,
    include_process: bool = True
) -> pd.DataFrame:
    """
    Extract features from all windows.
    
    Args:
        window_dfs: List of DataFrames, one per window
        include_flow: Include flow features
        include_density: Include density features
        include_process: Include process features
    
    Returns:
        DataFrame with all features, shape [n_windows, n_features]
    """
    all_features = []
    
    for window_df in window_dfs:
        features = extract_window_features(
            window_df,
            include_flow=include_flow,
            include_density=include_density,
            include_process=include_process
        )
        all_features.append(features)
    
    return pd.DataFrame(all_features)


def get_feature_names(
    include_flow: bool = True,
    include_density: bool = True,
    include_process: bool = True
) -> List[str]:
    """
    Get list of all feature names that would be extracted.
    
    Args:
        include_flow: Include flow features
        include_density: Include density features
        include_process: Include process features
    
    Returns:
        List of feature names
    """
    # Create a dummy DataFrame with required columns
    dummy_data = {
        'Slurry Flow (m3/s)': [1.0] * 60,
        'Slurry Mass Flow (kg/s)': [1000.0] * 60,
        'Slurry Density (kg/m3)': [1200.0] * 60,
        'SG': [1.2] * 60,
        'SG Shift: Target SG': [1.2] * 60,
        'Percent Solids (mass)': [30.0] * 60,
        'DV (um)': [50.0] * 60,
        'Pressure (kPa)': [500.0] * 60,
        'Temperature (C)': [25.0] * 60,
    }
    dummy_df = pd.DataFrame(dummy_data)
    
    features = extract_window_features(
        dummy_df,
        include_flow=include_flow,
        include_density=include_density,
        include_process=include_process
    )
    
    return list(features.keys())

