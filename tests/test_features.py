"""
Unit Tests for Feature Extraction
Tests flow, density, and process feature extractors.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.encoders.flow_features import compute_flow_features
from models.encoders.density_features import compute_density_features
from models.encoders.pressure_features import compute_pressure_features
from utils.physical_checks import validate_mass_balance, validate_density_sg, validate_window


class TestFlowFeatures:
    """Test flow feature extraction."""
    
    @pytest.fixture
    def constant_flow_df(self):
        """Create DataFrame with constant flow."""
        data = {
            'Slurry Flow (m3/s)': [2.0] * 60,
            'Slurry Mass Flow (kg/s)': [2400.0] * 60,
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def zero_flow_df(self):
        """Create DataFrame with zero flow events."""
        flow = np.ones(60) * 2.0
        flow[45:] = 0  # Last 15 samples are zero
        data = {
            'Slurry Flow (m3/s)': flow,
        }
        return pd.DataFrame(data)
    
    def test_flow_features_constant(self, constant_flow_df):
        """Test features for constant flow."""
        features = compute_flow_features(constant_flow_df)
        
        # Check basic stats
        assert features['flow_mean'] == pytest.approx(2.0)
        assert features['flow_std'] == pytest.approx(0.0, abs=1e-6)
        assert features['flow_cv'] == pytest.approx(0.0, abs=1e-6)
        
        # Check stability
        assert features['flow_stability'] == pytest.approx(1.0)
        
        # Check zero events
        assert features['flow_n_zero_events'] == 0
    
    def test_flow_features_with_zeros(self, zero_flow_df):
        """Test features with zero flow events."""
        features = compute_flow_features(zero_flow_df)
        
        # Should detect 15 zero events
        assert features['flow_n_zero_events'] == 15
        
        # Mean should be less than 2.0
        assert features['flow_mean'] < 2.0
        
        # Should have variability
        assert features['flow_std'] > 0
    
    def test_flow_cv_calculation(self):
        """Test coefficient of variation calculation."""
        # High variability flow
        flow = np.array([1.0, 3.0] * 30)  # Alternating
        df = pd.DataFrame({'Slurry Flow (m3/s)': flow})
        
        features = compute_flow_features(df)
        
        # CV should be > 0.15 (unstable)
        assert features['flow_cv'] > 0.15
        assert features['flow_stability'] < 0.85


class TestDensityFeatures:
    """Test density feature extraction."""
    
    @pytest.fixture
    def constant_density_df(self):
        """Create DataFrame with constant density."""
        data = {
            'Slurry Density (kg/m3)': [1200.0] * 60,
            'SG': [1.2] * 60,
            'SG Shift: Target SG': [1.2] * 60,
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def trending_density_df(self):
        """Create DataFrame with density trend."""
        density = np.linspace(1200, 1250, 60)  # Increasing trend
        data = {
            'Slurry Density (kg/m3)': density,
            'SG': density / 1000,
        }
        return pd.DataFrame(data)
    
    def test_density_features_constant(self, constant_density_df):
        """Test features for constant density."""
        features = compute_density_features(constant_density_df)
        
        assert features['density_mean'] == pytest.approx(1200.0)
        assert features['density_std'] == pytest.approx(0.0, abs=1e-6)
        assert features['density_trend'] == pytest.approx(0.0, abs=1e-6)
        assert features['sg_mean'] == pytest.approx(1.2)
        assert features['sg_target_deviation'] == pytest.approx(0.0, abs=1e-6)
    
    def test_density_trend_detection(self, trending_density_df):
        """Test density trend detection."""
        features = compute_density_features(trending_density_df, sample_rate=1.0)
        
        # Should detect positive trend
        assert features['density_trend'] > 0
        
        # Mean should be around 1225
        assert features['density_mean'] == pytest.approx(1225.0, abs=5.0)
    
    def test_density_spike_detection(self):
        """Test spike detection."""
        # Create data with spike
        density = np.ones(60) * 1200
        density[30] = 1400  # Large spike
        
        df = pd.DataFrame({'Slurry Density (kg/m3)': density})
        features = compute_density_features(df)
        
        # Should detect at least one spike
        assert features['density_spike_count'] > 0


class TestPressureFeatures:
    """Test pressure feature extraction."""
    
    @pytest.fixture
    def correlated_data_df(self):
        """Create DataFrame with correlated pressure-flow."""
        flow = np.linspace(1.0, 2.0, 60)
        pressure = flow * 300  # Strong correlation
        
        data = {
            'Pressure (kPa)': pressure,
            'Slurry Flow (m3/s)': flow,
        }
        return pd.DataFrame(data)
    
    def test_pressure_flow_correlation(self, correlated_data_df):
        """Test pressure-flow correlation detection."""
        features = compute_pressure_features(correlated_data_df)
        
        # Should have strong positive correlation
        assert features['pressure_flow_correlation'] > 0.9
    
    def test_pressure_variability(self):
        """Test pressure variability calculation."""
        # High variability pressure
        pressure = np.random.normal(500, 100, 60)
        df = pd.DataFrame({'Pressure (kPa)': pressure})
        
        features = compute_pressure_features(df)
        
        assert features['pressure_mean'] == pytest.approx(500, abs=50)
        assert features['pressure_variability'] > 0


class TestPhysicalValidation:
    """Test physical consistency checks."""
    
    @pytest.fixture
    def valid_physics_df(self):
        """Create DataFrame with valid physics."""
        flow = 2.0  # m³/s
        density = 1200.0  # kg/m³
        mass_flow = flow * density  # Should be 2400 kg/s
        
        data = {
            'Slurry Flow (m3/s)': [flow] * 60,
            'Slurry Density (kg/m3)': [density] * 60,
            'Slurry Mass Flow (kg/s)': [mass_flow] * 60,
            'SG': [density/1000] * 60,
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def invalid_mass_balance_df(self):
        """Create DataFrame with invalid mass balance."""
        data = {
            'Slurry Flow (m3/s)': [2.0] * 60,
            'Slurry Density (kg/m3)': [1200.0] * 60,
            'Slurry Mass Flow (kg/s)': [3000.0] * 60,  # Should be 2400
        }
        return pd.DataFrame(data)
    
    def test_valid_mass_balance(self, valid_physics_df):
        """Test that valid mass balance passes."""
        is_valid, error, message = validate_mass_balance(valid_physics_df)
        
        assert is_valid is True
        assert error < 0.01  # < 1% error
    
    def test_invalid_mass_balance(self, invalid_mass_balance_df):
        """Test that invalid mass balance fails."""
        is_valid, error, message = validate_mass_balance(
            invalid_mass_balance_df,
            max_error=0.15
        )
        
        assert is_valid is False
        assert error > 0.15
    
    def test_valid_density_sg(self, valid_physics_df):
        """Test that valid density-SG relationship passes."""
        is_valid, deviation, message = validate_density_sg(valid_physics_df)
        
        assert is_valid is True
        assert deviation < 0.01
    
    def test_invalid_density_sg(self):
        """Test that invalid density-SG fails."""
        data = {
            'Slurry Density (kg/m3)': [1200.0] * 60,
            'SG': [1.5] * 60,  # Should be ~1.2
        }
        df = pd.DataFrame(data)
        
        is_valid, deviation, message = validate_density_sg(df, max_deviation=0.05)
        
        assert is_valid is False
        assert deviation > 0.05
    
    def test_validate_window_all_checks(self, valid_physics_df):
        """Test complete window validation."""
        is_valid, results = validate_window(valid_physics_df)
        
        assert is_valid is True
        assert 'mass_balance' in results
        assert 'density_sg' in results
        
        # All checks should pass
        for check_result in results.values():
            assert check_result[0] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

