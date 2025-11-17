"""
Density Feature Extraction
Extracts features from density, SG, and solids concentration measurements.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_density_features(
    window_df: pd.DataFrame,
    density_col: str = 'Slurry Density (kg/m3)',
    sg_col: str = 'SG',
    target_sg_col: Optional[str] = 'SG Shift: Target SG',
    percent_solids_col: Optional[str] = 'Percent Solids (mass)',
    sample_rate: float = 1.0
) -> Dict[str, float]:
    """
    Extract density and SG-related features.
    
    Args:
        window_df: DataFrame for a single window
        density_col: Name of density column
        sg_col: Name of specific gravity column
        target_sg_col: Optional target SG column
        percent_solids_col: Optional percent solids column
        sample_rate: Sampling rate in Hz
    
    Returns:
        Dictionary of density features
    
    Features:
        - density_mean, density_std: Basic statistics
        - density_trend: Trend in kg/m³ per 5 minutes
        - density_spike_count: Number of spikes >3σ
        - sg_mean, sg_std: SG statistics
        - sg_target_deviation: Deviation from target SG
        - percent_solids_mean: Average solids concentration
    """
    features = {}
    
    # Density features
    if density_col in window_df.columns:
        density = window_df[density_col].values
        density_valid = density[~np.isnan(density)]
        
        if len(density_valid) > 0:
            density_mean = np.mean(density_valid)
            density_std = np.std(density_valid)
            
            features['density_mean'] = float(density_mean)
            features['density_std'] = float(density_std)
            features['density_min'] = float(np.min(density_valid))
            features['density_max'] = float(np.max(density_valid))
            
            # Trend: change per 5 minutes
            if len(density_valid) > 1:
                window_duration_min = len(density_valid) / sample_rate / 60.0
                density_change = density_valid[-1] - density_valid[0]
                density_trend = (density_change / window_duration_min) * 5.0  # per 5 min
                features['density_trend'] = float(density_trend)
            else:
                features['density_trend'] = 0.0
            
            # Spike count (>3σ from mean)
            if density_std > 0:
                z_scores = np.abs(density_valid - density_mean) / density_std
                spike_count = np.sum(z_scores > 3.0)
                features['density_spike_count'] = int(spike_count)
            else:
                features['density_spike_count'] = 0
        else:
            features.update({
                'density_mean': np.nan,
                'density_std': np.nan,
                'density_min': np.nan,
                'density_max': np.nan,
                'density_trend': np.nan,
                'density_spike_count': 0,
            })
    
    # SG features
    if sg_col in window_df.columns:
        sg = window_df[sg_col].values
        sg_valid = sg[~np.isnan(sg)]
        
        if len(sg_valid) > 0:
            sg_mean = np.mean(sg_valid)
            sg_std = np.std(sg_valid)
            
            features['sg_mean'] = float(sg_mean)
            features['sg_std'] = float(sg_std)
            
            # Deviation from target SG
            if target_sg_col and target_sg_col in window_df.columns:
                target_sg = window_df[target_sg_col].values
                target_sg_valid = target_sg[~np.isnan(target_sg)]
                
                if len(target_sg_valid) > 0:
                    target_sg_mean = np.mean(target_sg_valid)
                    sg_deviation = sg_mean - target_sg_mean
                    features['sg_target_deviation'] = float(sg_deviation)
                else:
                    features['sg_target_deviation'] = np.nan
            else:
                features['sg_target_deviation'] = np.nan
        else:
            features.update({
                'sg_mean': np.nan,
                'sg_std': np.nan,
                'sg_target_deviation': np.nan,
            })
    
    # Percent solids
    if percent_solids_col and percent_solids_col in window_df.columns:
        percent_solids = window_df[percent_solids_col].values
        ps_valid = percent_solids[~np.isnan(percent_solids)]
        
        if len(ps_valid) > 0:
            features['percent_solids_mean'] = float(np.mean(ps_valid))
            features['percent_solids_std'] = float(np.std(ps_valid))
        else:
            features['percent_solids_mean'] = np.nan
            features['percent_solids_std'] = np.nan
    
    return features


def compute_particle_size_features(
    window_df: pd.DataFrame,
    dv_col: str = 'DV (um)'
) -> Dict[str, float]:
    """
    Extract particle size distribution features.
    
    Args:
        window_df: DataFrame for a single window
        dv_col: Name of DV (particle size) column
    
    Returns:
        Dictionary of particle size features
    
    Features:
        - dv_mean: Average particle size
        - dv_drift: Change over window (indicates segregation)
        - dv_std: Variability
    """
    features = {}
    
    if dv_col in window_df.columns:
        dv = window_df[dv_col].values
        dv_valid = dv[~np.isnan(dv)]
        
        if len(dv_valid) > 0:
            features['dv_mean'] = float(np.mean(dv_valid))
            features['dv_std'] = float(np.std(dv_valid))
            
            # Drift: total change over window
            if len(dv_valid) > 1:
                dv_drift = dv_valid[-1] - dv_valid[0]
                features['dv_drift'] = float(dv_drift)
            else:
                features['dv_drift'] = 0.0
        else:
            features.update({
                'dv_mean': np.nan,
                'dv_std': np.nan,
                'dv_drift': np.nan,
            })
    
    return features


def extract_all_density_features(
    window_df: pd.DataFrame,
    include_particle_size: bool = True
) -> Dict[str, float]:
    """
    Extract all density-related features.
    
    Args:
        window_df: DataFrame for a single window
        include_particle_size: Include particle size features (default: True)
    
    Returns:
        Combined dictionary of all density features
    """
    features = compute_density_features(window_df)
    
    if include_particle_size:
        ps_features = compute_particle_size_features(window_df)
        features.update(ps_features)
    
    return features

