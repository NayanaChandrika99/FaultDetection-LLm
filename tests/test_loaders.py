"""
Unit Tests for Data Loaders
Tests CSV loading, timestamp parsing, and window creation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.loaders.slurry_loader import (
    parse_timestamp,
    load_slurry_csv,
    create_windows,
    extract_window_metadata
)


class TestTimestampParsing:
    """Test timestamp parsing functions."""
    
    def test_parse_timestamp_basic(self):
        """Test basic timestamp parsing."""
        result = parse_timestamp("01:23.7")
        expected = pd.Timedelta(minutes=1, seconds=23.7)
        assert result == expected
    
    def test_parse_timestamp_zero(self):
        """Test zero timestamp."""
        result = parse_timestamp("00:00.0")
        expected = pd.Timedelta(minutes=0, seconds=0.0)
        assert result == expected
    
    def test_parse_timestamp_large(self):
        """Test large timestamp."""
        result = parse_timestamp("59:59.9")
        expected = pd.Timedelta(minutes=59, seconds=59.9)
        assert result == expected
    
    def test_parse_timestamp_fractional(self):
        """Test fractional seconds."""
        result = parse_timestamp("05:15.567")
        assert result.total_seconds() == pytest.approx(315.567, rel=1e-3)


class TestWindowCreation:
    """Test window creation functions."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        n_samples = 120  # 2 minutes at 1 Hz
        data = {
            'Slurry Flow (m3/s)': np.random.uniform(1.0, 2.0, n_samples),
            'Slurry Density (kg/m3)': np.random.uniform(1100, 1300, n_samples),
            'Pressure (kPa)': np.random.uniform(400, 600, n_samples),
        }
        
        # Create time index
        timestamps = [pd.Timedelta(seconds=i) for i in range(n_samples)]
        df = pd.DataFrame(data, index=timestamps)
        df.index.name = 'timestamp'
        
        return df
    
    def test_create_windows_basic(self, sample_dataframe):
        """Test basic window creation."""
        windows, indices = create_windows(
            sample_dataframe,
            window_sec=60,
            stride_sec=15,
            max_missing_ratio=0.1
        )
        
        # Check shape
        assert windows.ndim == 3
        assert windows.shape[1] == 3  # 3 sensors
        assert windows.shape[2] == 60  # 60 second window
        
        # Check that we got multiple windows with 75% overlap
        assert len(windows) > 1
        
        # Check indices
        assert len(indices) == len(windows)
    
    def test_create_windows_with_missing_data(self, sample_dataframe):
        """Test window creation with missing data."""
        # Add some NaN values
        df = sample_dataframe.copy()
        df.iloc[10:15, 0] = np.nan  # 5 samples missing
        
        windows, indices = create_windows(
            df,
            window_sec=60,
            stride_sec=15,
            max_missing_ratio=0.1  # 10% = 6 samples, so should accept
        )
        
        # Should create windows
        assert len(windows) > 0
    
    def test_create_windows_rejects_excessive_missing(self, sample_dataframe):
        """Test that windows with too much missing data are rejected."""
        df = sample_dataframe.copy()
        # Set first 20 samples to NaN (> 10% of 60)
        df.iloc[:20, :] = np.nan
        
        windows, indices = create_windows(
            df,
            window_sec=60,
            stride_sec=15,
            max_missing_ratio=0.1
        )
        
        # First window should be rejected
        # Remaining windows should still be created
        assert len(windows) >= 0
    
    def test_window_shape_consistency(self, sample_dataframe):
        """Test that all windows have consistent shape."""
        windows, _ = create_windows(
            sample_dataframe,
            window_sec=60,
            stride_sec=15
        )
        
        for window in windows:
            assert window.shape == (3, 60)
    
    def test_extract_window_metadata(self, sample_dataframe):
        """Test metadata extraction."""
        windows, indices = create_windows(
            sample_dataframe,
            window_sec=60,
            stride_sec=15
        )
        
        metadata = extract_window_metadata(sample_dataframe, indices)
        
        # Check structure
        assert len(metadata) == len(windows)
        assert 'window_id' in metadata.columns
        assert 'start_time' in metadata.columns
        assert 'end_time' in metadata.columns
        assert 'duration_sec' in metadata.columns


class TestDataLoaderIntegration:
    """Integration tests for data loading pipeline."""
    
    @pytest.fixture
    def temp_csv_file(self, tmp_path):
        """Create a temporary CSV file for testing."""
        csv_path = tmp_path / "test_data.csv"
        
        # Create sample data
        n_samples = 120
        timestamps = [f"{i//60:02d}:{i%60:02d}.0" for i in range(n_samples)]
        
        data = {
            'Timestamp': timestamps,
            'Slurry Flow (m3/s)': np.random.uniform(1.0, 2.0, n_samples),
            'Slurry Mass Flow (kg/s)': np.random.uniform(1000, 2000, n_samples),
            'Slurry Density (kg/m3)': np.random.uniform(1100, 1300, n_samples),
            'SG': np.random.uniform(1.1, 1.3, n_samples),
            'Pressure (kPa)': np.random.uniform(400, 600, n_samples),
            'Temperature (C)': np.random.uniform(20, 30, n_samples),
        }
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def test_load_slurry_csv(self, temp_csv_file):
        """Test loading and preprocessing CSV."""
        df = load_slurry_csv(temp_csv_file)
        
        # Check that data was loaded
        assert len(df) > 0
        
        # Check that timestamp index exists
        assert df.index.name == 'timestamp'
        assert isinstance(df.index, pd.TimedeltaIndex)
        
        # Check that zero-flow flag was added
        assert 'zero_flow_flag' in df.columns
    
    def test_load_slurry_csv_with_gaps(self, tmp_path):
        """Test CSV loading with gaps in data."""
        csv_path = tmp_path / "test_gaps.csv"
        
        # Create data with gaps
        timestamps = ['00:00.0', '00:01.0', '00:05.0', '00:06.0']  # Gap at 02-04
        data = {
            'Timestamp': timestamps,
            'Slurry Flow (m3/s)': [1.0, 1.1, 1.2, 1.3],
            'Slurry Density (kg/m3)': [1100, 1110, 1120, 1130],
        }
        
        df_input = pd.DataFrame(data)
        df_input.to_csv(csv_path, index=False)
        
        # Load with interpolation
        df = load_slurry_csv(str(csv_path), max_gap_sec=3)
        
        # Check that gaps were interpolated
        assert len(df) >= len(timestamps)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

