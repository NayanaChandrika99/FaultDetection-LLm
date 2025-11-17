"""
Integration Tests
End-to-end tests for the complete pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.loaders.slurry_loader import load_and_window
from data.loaders.window_creator import assign_labels
from models.rocket_heads import MultiROCKETClassifier
from models.encoders.feature_extractor import extract_window_features


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def sample_csv_path(self, tmp_path):
        """Create a sample CSV file."""
        csv_path = tmp_path / "sample_data.csv"
        
        # Create 5 minutes of data at 1 Hz
        n_samples = 300
        timestamps = [f"{i//60:02d}:{i%60:02d}.0" for i in range(n_samples)]
        
        # Generate realistic sensor data
        np.random.seed(42)
        base_flow = 2.0
        base_density = 1200.0
        
        data = {
            'Timestamp': timestamps,
            'Slurry Flow (m3/s)': np.random.normal(base_flow, 0.1, n_samples),
            'Slurry Mass Flow (kg/s)': np.random.normal(base_flow * base_density, 100, n_samples),
            'Solids Flow (m3/s)': np.random.normal(0.6, 0.05, n_samples),
            'Solids Mass Flow (kg/s)': np.random.normal(720, 50, n_samples),
            'Slurry Density (kg/m3)': np.random.normal(base_density, 20, n_samples),
            'SG': np.random.normal(1.2, 0.02, n_samples),
            'Percent Solids (mass)': np.random.normal(30.0, 2.0, n_samples),
            'Percent Solids (vol)': np.random.normal(15.0, 1.0, n_samples),
            'SG Shift: Target SG': [1.2] * n_samples,
            'Pressure (kPa)': np.random.normal(500, 30, n_samples),
            'Temperature (C)': np.random.normal(25, 2, n_samples),
            'DV (um)': np.random.normal(50, 3, n_samples),
        }
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def test_data_loading_pipeline(self, sample_csv_path):
        """Test complete data loading and windowing."""
        # Load and create windows
        windows, metadata = load_and_window(
            csv_path=sample_csv_path,
            window_sec=60,
            stride_sec=15,
            sample_rate=1.0
        )
        
        # Check outputs
        assert windows.shape[0] > 0  # At least one window
        assert windows.shape[1] >= 11  # At least 11 sensors
        assert windows.shape[2] == 60  # 60 seconds
        assert len(metadata) == len(windows)
    
    def test_feature_extraction(self, sample_csv_path):
        """Test feature extraction from windows."""
        from data.loaders.slurry_loader import load_slurry_csv
        
        # Load data
        df = load_slurry_csv(sample_csv_path)
        
        # Extract features from first 60 seconds
        window_df = df.iloc[:60]
        features = extract_window_features(window_df)
        
        # Check that features were extracted
        assert len(features) > 0
        assert 'flow_mean' in features
        assert 'density_mean' in features
        assert 'pressure_mean' in features
        
        # Check that values are reasonable
        assert not np.isnan(features['flow_mean'])
        assert features['flow_mean'] > 0
    
    def test_classifier_training_smoke(self, sample_csv_path):
        """Smoke test for classifier training (quick)."""
        # Load data
        windows, metadata = load_and_window(
            csv_path=sample_csv_path,
            window_sec=60,
            stride_sec=15
        )
        
        # Create dummy labels
        labels = np.array(['Normal'] * len(windows))
        # Add some variety
        labels[::3] = 'Pipeline Blockage'
        
        # Limit to small sample for speed
        n_samples = min(20, len(windows))
        X = windows[:n_samples]
        y = labels[:n_samples]
        
        # Train classifier with minimal kernels
        classifier = MultiROCKETClassifier(
            n_kernels=100,  # Very small for speed
            alphas=np.array([1.0, 10.0])  # Just 2 alphas
        )
        
        classifier.fit(X, y)
        
        # Check that model is fitted
        assert classifier.is_fitted_ is True
        assert classifier.classes_ is not None
        
        # Make predictions
        y_pred = classifier.predict(X)
        assert len(y_pred) == len(y)
        
        # Get confidence
        confidence = classifier.get_confidence(X)
        assert len(confidence) == len(y)
        assert all(0 <= c <= 1 for c in confidence)
    
    def test_model_save_load(self, tmp_path, sample_csv_path):
        """Test model saving and loading."""
        # Load data
        windows, _ = load_and_window(sample_csv_path, window_sec=60, stride_sec=30)
        
        # Create minimal training set
        X = windows[:10]
        y = np.array(['Normal'] * 10)
        
        # Train model
        model = MultiROCKETClassifier(n_kernels=50, alphas=np.array([1.0]))
        model.fit(X, y)
        
        # Save
        model_path = tmp_path / 'test_model.pkl'
        model.save(str(model_path))
        
        # Load
        loaded_model = MultiROCKETClassifier.load(str(model_path))
        
        # Check that loaded model works
        assert loaded_model.is_fitted_ is True
        y_pred = loaded_model.predict(X)
        assert len(y_pred) == len(y)
    
    def test_physical_validation_integration(self, sample_csv_path):
        """Test physical validation in pipeline."""
        from data.loaders.slurry_loader import load_slurry_csv
        from utils.physical_checks import validate_window
        
        # Load data
        df = load_slurry_csv(sample_csv_path)
        
        # Validate first window
        window_df = df.iloc[:60]
        is_valid, results = validate_window(window_df)
        
        # Should have results for checks
        assert 'mass_balance' in results
        assert 'density_sg' in results
        
        # Results should be tuples
        for check_result in results.values():
            assert len(check_result) == 3  # (is_valid, metric, message)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_windows(self):
        """Test handling of empty windows."""
        from data.loaders.slurry_loader import create_windows
        
        # Create minimal DataFrame
        df = pd.DataFrame({
            'Slurry Flow (m3/s)': [1.0] * 30  # Only 30 samples
        })
        
        # Try to create 60-second windows (should fail gracefully)
        with pytest.raises(ValueError):
            windows, indices = create_windows(df, window_sec=60, stride_sec=15)
    
    def test_missing_columns(self):
        """Test handling of missing sensor columns."""
        from models.encoders.flow_features import compute_flow_features
        
        # DataFrame without expected column
        df = pd.DataFrame({'Wrong Column': [1.0] * 60})
        
        # Should raise error
        with pytest.raises(ValueError):
            features = compute_flow_features(df)
    
    def test_all_nan_data(self):
        """Test handling of all-NaN data."""
        from models.encoders.flow_features import compute_flow_features
        
        df = pd.DataFrame({
            'Slurry Flow (m3/s)': [np.nan] * 60
        })
        
        features = compute_flow_features(df)
        
        # Should return NaN features
        assert np.isnan(features['flow_mean'])
        assert np.isnan(features['flow_std'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

