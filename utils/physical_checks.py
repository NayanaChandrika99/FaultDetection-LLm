"""
Physical Consistency Validation
Validates sensor data against physical laws and engineering constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def validate_mass_balance(
    window_df: pd.DataFrame,
    flow_col: str = 'Slurry Flow (m3/s)',
    density_col: str = 'Slurry Density (kg/m3)',
    mass_flow_col: str = 'Slurry Mass Flow (kg/s)',
    max_error: float = 0.15
) -> Tuple[bool, float, str]:
    """
    Validate mass balance: measured mass flow ≈ volumetric flow × density
    
    Args:
        window_df: DataFrame for a single window
        flow_col: Volumetric flow column name
        density_col: Density column name
        mass_flow_col: Mass flow column name
        max_error: Maximum allowed relative error (default: 0.15 = 15%)
    
    Returns:
        Tuple of (is_valid, error_ratio, message)
    
    Physics:
        mass_flow = volumetric_flow × density
        Error = |measured - calculated| / measured
    """
    required_cols = [flow_col, density_col, mass_flow_col]
    if not all(col in window_df.columns for col in required_cols):
        return True, 0.0, "mass_balance_not_checkable"
    
    # Get mean values
    flow_mean = window_df[flow_col].mean()
    density_mean = window_df[density_col].mean()
    mass_flow_measured = window_df[mass_flow_col].mean()
    
    # Check for NaN
    if np.isnan(flow_mean) or np.isnan(density_mean) or np.isnan(mass_flow_measured):
        return True, 0.0, "mass_balance_nan"
    
    # Calculate expected mass flow
    mass_flow_calculated = flow_mean * density_mean
    
    # Relative error
    if abs(mass_flow_measured) < 1e-6:
        # Near-zero flow, can't calculate error
        return True, 0.0, "zero_flow"
    
    error = abs(mass_flow_measured - mass_flow_calculated) / abs(mass_flow_measured)
    
    if error > max_error:
        return False, error, f"mass_balance_violation (error={error:.1%})"
    
    return True, error, "mass_balance_ok"


def validate_density_sg(
    window_df: pd.DataFrame,
    density_col: str = 'Slurry Density (kg/m3)',
    sg_col: str = 'SG',
    max_deviation: float = 0.05,
    water_density: float = 1000.0
) -> Tuple[bool, float, str]:
    """
    Validate density-SG relationship: SG ≈ Density / 1000
    
    Args:
        window_df: DataFrame for a single window
        density_col: Density column name
        sg_col: Specific gravity column name
        max_deviation: Maximum allowed absolute deviation (default: 0.05)
        water_density: Reference water density in kg/m³ (default: 1000.0)
    
    Returns:
        Tuple of (is_valid, deviation, message)
    
    Physics:
        SG = density / density_water
        Typically density_water = 1000 kg/m³
    """
    if density_col not in window_df.columns or sg_col not in window_df.columns:
        return True, 0.0, "density_sg_not_checkable"
    
    density_mean = window_df[density_col].mean()
    sg_mean = window_df[sg_col].mean()
    
    if np.isnan(density_mean) or np.isnan(sg_mean):
        return True, 0.0, "density_sg_nan"
    
    # Calculate expected SG from density
    sg_calculated = density_mean / water_density
    
    # Absolute deviation
    deviation = abs(sg_mean - sg_calculated)
    
    if deviation > max_deviation:
        return False, deviation, f"density_sg_mismatch (dev={deviation:.3f})"
    
    return True, deviation, "density_sg_ok"


def validate_solids_consistency(
    window_df: pd.DataFrame,
    solids_mass_flow_col: str = 'Solids Mass Flow (kg/s)',
    total_mass_flow_col: str = 'Slurry Mass Flow (kg/s)',
    percent_solids_col: str = 'Percent Solids (mass)',
    max_error: float = 0.15
) -> Tuple[bool, float, str]:
    """
    Validate solids concentration consistency.
    
    Args:
        window_df: DataFrame for a single window
        solids_mass_flow_col: Solids mass flow column
        total_mass_flow_col: Total mass flow column
        percent_solids_col: Percent solids column
        max_error: Maximum allowed relative error
    
    Returns:
        Tuple of (is_valid, error_ratio, message)
    
    Physics:
        percent_solids = solids_mass_flow / total_mass_flow * 100
    """
    required_cols = [solids_mass_flow_col, total_mass_flow_col, percent_solids_col]
    if not all(col in window_df.columns for col in required_cols):
        return True, 0.0, "solids_not_checkable"
    
    solids_flow = window_df[solids_mass_flow_col].mean()
    total_flow = window_df[total_mass_flow_col].mean()
    percent_solids = window_df[percent_solids_col].mean()
    
    if np.isnan(solids_flow) or np.isnan(total_flow) or np.isnan(percent_solids):
        return True, 0.0, "solids_nan"
    
    if abs(total_flow) < 1e-6:
        return True, 0.0, "zero_flow"
    
    # Calculate expected percent solids
    percent_calculated = (solids_flow / total_flow) * 100.0
    
    # Relative error
    if abs(percent_solids) < 0.1:
        return True, 0.0, "near_zero_solids"
    
    error = abs(percent_solids - percent_calculated) / percent_solids
    
    if error > max_error:
        return False, error, f"solids_inconsistency (error={error:.1%})"
    
    return True, error, "solids_ok"


def validate_window(
    window_df: pd.DataFrame,
    checks: Optional[Dict[str, bool]] = None
) -> Tuple[bool, Dict[str, Tuple[bool, float, str]]]:
    """
    Perform all physical validation checks on a window.
    
    Args:
        window_df: DataFrame for a single window
        checks: Dict specifying which checks to perform (default: all True)
    
    Returns:
        Tuple of (all_valid, check_results)
            - all_valid: True if all enabled checks pass
            - check_results: Dict mapping check name to (is_valid, metric, message)
    
    Example:
        >>> is_valid, results = validate_window(window_df)
        >>> if not is_valid:
        >>>     print(f"Failed checks: {[k for k,v in results.items() if not v[0]]}")
    """
    if checks is None:
        checks = {
            'mass_balance': True,
            'density_sg': True,
            'solids_consistency': True,
        }
    
    results = {}
    
    # Mass balance check
    if checks.get('mass_balance', True):
        results['mass_balance'] = validate_mass_balance(window_df)
    
    # Density-SG check
    if checks.get('density_sg', True):
        results['density_sg'] = validate_density_sg(window_df)
    
    # Solids consistency check
    if checks.get('solids_consistency', True):
        results['solids_consistency'] = validate_solids_consistency(window_df)
    
    # Overall validity
    all_valid = all(result[0] for result in results.values())
    
    return all_valid, results


def filter_windows_by_physics(
    windows: np.ndarray,
    window_dfs: list,
    labels: Optional[np.ndarray] = None,
    metadata: Optional[pd.DataFrame] = None,
    checks: Optional[Dict[str, bool]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[pd.DataFrame], pd.DataFrame]:
    """
    Filter windows based on physical consistency checks.
    
    Args:
        windows: Array of shape [n_windows, n_sensors, window_sec]
        window_dfs: List of DataFrames, one per window
        labels: Optional array of labels
        metadata: Optional metadata DataFrame
        checks: Dict specifying which checks to perform
    
    Returns:
        Tuple of (filtered_windows, filtered_labels, filtered_metadata, validation_log)
    """
    valid_indices = []
    validation_log = []
    
    for i, window_df in enumerate(window_dfs):
        is_valid, results = validate_window(window_df, checks)
        
        log_entry = {
            'window_id': i,
            'is_valid': is_valid,
        }
        
        # Add check results
        for check_name, (check_valid, metric, message) in results.items():
            log_entry[f'{check_name}_valid'] = check_valid
            log_entry[f'{check_name}_metric'] = metric
            log_entry[f'{check_name}_message'] = message
        
        validation_log.append(log_entry)
        
        if is_valid:
            valid_indices.append(i)
    
    validation_df = pd.DataFrame(validation_log)
    
    if len(valid_indices) == 0:
        raise ValueError("No valid windows after physical validation")
    
    # Filter arrays
    filtered_windows = windows[valid_indices]
    filtered_labels = labels[valid_indices] if labels is not None else None
    filtered_metadata = metadata.iloc[valid_indices].reset_index(drop=True) if metadata is not None else None
    
    return filtered_windows, filtered_labels, filtered_metadata, validation_df

