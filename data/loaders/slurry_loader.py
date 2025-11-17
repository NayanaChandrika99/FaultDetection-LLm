"""
Slurry CSV Data Loader
Handles parsing MM:SS.s timestamps, resampling to 1Hz, and interpolation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def parse_timestamp(timestamp_str: str) -> pd.Timedelta:
    """
    Parse MM:SS.s timestamp format to Timedelta.
    NOTE: For datetime timestamps, this is not used.
    
    Args:
        timestamp_str: Timestamp in format "MM:SS.s" (e.g., "01:23.7")
    
    Returns:
        pd.Timedelta object
    
    Examples:
        >>> parse_timestamp("01:23.7")
        Timedelta('0 days 00:01:23.700000')
    """
    parts = timestamp_str.split(':')
    minutes = int(parts[0])
    seconds = float(parts[1])
    return pd.Timedelta(minutes=minutes, seconds=seconds)


def load_slurry_csv(
    path: str,
    sample_rate: float = 1.0,
    max_gap_sec: int = 3,
    add_zero_flow_flag: bool = True
) -> pd.DataFrame:
    """
    Load and preprocess slurry sensor CSV data.
    
    Args:
        path: Path to CSV file
        sample_rate: Target sampling rate in Hz (default: 1.0)
        max_gap_sec: Maximum gap to interpolate in seconds (default: 3)
        add_zero_flow_flag: Add binary flag for zero-flow events (default: True)
    
    Returns:
        DataFrame with uniform time index and interpolated values
    
    Expected CSV columns:
        - Timestamp (datetime or MM:SS.s format)
        - Sensor columns (varies by file)
    """
    # Load CSV
    df = pd.read_csv(path)
    
    # Parse timestamps - try datetime first, then MM:SS.s format
    try:
        df['timestamp'] = pd.to_datetime(df['Timestamp'])
    except:
        df['timestamp'] = df['Timestamp'].apply(parse_timestamp)
    
    df = df.drop('Timestamp', axis=1).set_index('timestamp')
    
    # Sort by timestamp (ensure chronological order)
    df = df.sort_index()
    
    # Resample to uniform rate and interpolate gaps
    resample_rule = f"{int(1000/sample_rate)}ms"  # Convert Hz to ms
    df = df.resample(resample_rule).mean()
    
    # Interpolate missing values (limited to max_gap_sec)
    max_limit = int(max_gap_sec * sample_rate)
    df = df.interpolate(method='linear', limit=max_limit, limit_direction='both')
    
    # Add zero-flow flag if requested and flow column exists
    if add_zero_flow_flag and 'Slurry Flow (m3/s)' in df.columns:
        df['zero_flow_flag'] = (df['Slurry Flow (m3/s)'] == 0).astype(int)
    
    return df


def create_windows(
    df: pd.DataFrame,
    window_sec: int = 60,
    stride_sec: int = 15,
    max_missing_ratio: float = 0.1
) -> Tuple[np.ndarray, list]:
    """
    Create overlapping windows from time-series data.
    
    Args:
        df: DataFrame with time index and sensor columns
        window_sec: Window size in seconds (default: 60)
        stride_sec: Stride between windows in seconds (default: 15)
        max_missing_ratio: Maximum allowed ratio of missing values (default: 0.1)
    
    Returns:
        Tuple of (windows, indices):
            - windows: ndarray of shape [n_windows, n_sensors, window_sec]
            - indices: List of (start_idx, end_idx) tuples for each window
    
    Example:
        60s windows with 15s stride = 75% overlap
    """
    windows = []
    window_indices = []
    n_samples = len(df)
    
    # Filter out columns that are entirely NaN
    sensor_cols = [col for col in df.columns if col != 'zero_flow_flag']
    valid_sensor_cols = [col for col in sensor_cols if df[col].notna().sum() > len(df) * 0.1]
    
    if len(valid_sensor_cols) == 0:
        raise ValueError("No valid sensor columns found with data")
    
    n_sensors = len(valid_sensor_cols)
    
    for start in range(0, n_samples - window_sec + 1, stride_sec):
        end = start + window_sec
        window_data = df.iloc[start:end][valid_sensor_cols].values
        
        # Check missing data ratio (only for columns that should have data)
        missing_ratio = np.isnan(window_data).sum() / window_data.size
        
        if missing_ratio <= max_missing_ratio:
            # Transpose to shape [n_sensors, window_sec]
            windows.append(window_data.T)
            window_indices.append((start, end))
    
    if len(windows) == 0:
        raise ValueError(f"No valid windows created. Check data quality and parameters. "
                        f"Valid sensors: {len(valid_sensor_cols)}, Max missing ratio: {max_missing_ratio}")
    
    # Stack to [n_windows, n_sensors, window_sec]
    windows_array = np.array(windows)
    
    return windows_array, window_indices


def extract_window_metadata(
    df: pd.DataFrame,
    window_indices: list
) -> pd.DataFrame:
    """
    Extract metadata for each window (timestamps, labels, etc.).
    
    Args:
        df: Original DataFrame with time index
        window_indices: List of (start_idx, end_idx) tuples
    
    Returns:
        DataFrame with window metadata
    """
    metadata = []
    
    for i, (start, end) in enumerate(window_indices):
        window_df = df.iloc[start:end]
        
        meta = {
            'window_id': i,
            'start_time': df.index[start],
            'end_time': df.index[end-1],
            'duration_sec': (df.index[end-1] - df.index[start]).total_seconds(),
            'n_samples': end - start,
        }
        
        # Add zero-flow event count if available
        if 'zero_flow_flag' in window_df.columns:
            meta['n_zero_flow_events'] = window_df['zero_flow_flag'].sum()
        
        metadata.append(meta)
    
    return pd.DataFrame(metadata)


def load_and_window(
    csv_path: str,
    window_sec: int = 60,
    stride_sec: int = 15,
    sample_rate: float = 1.0,
    max_gap_sec: int = 3,
    max_missing_ratio: float = 0.1
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convenience function to load CSV and create windows in one call.
    
    Args:
        csv_path: Path to CSV file
        window_sec: Window size in seconds
        stride_sec: Stride between windows in seconds
        sample_rate: Target sampling rate in Hz
        max_gap_sec: Maximum gap to interpolate
        max_missing_ratio: Maximum allowed missing data ratio per window
    
    Returns:
        Tuple of (windows, metadata):
            - windows: ndarray of shape [n_windows, n_sensors, window_sec]
            - metadata: DataFrame with window information
    """
    # Load and preprocess data
    df = load_slurry_csv(csv_path, sample_rate, max_gap_sec)
    
    # Create windows
    windows, indices = create_windows(df, window_sec, stride_sec, max_missing_ratio)
    
    # Extract metadata
    metadata = extract_window_metadata(df, indices)
    
    return windows, metadata

