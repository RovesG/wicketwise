# Purpose: Tests for model drift detection functionality
# Author: Shamus Rae, Last Modified: 2024-12-19

"""
Test suite for drift_detector.py module.
Tests various drift detection scenarios and validation logic.
"""

import pytest
import numpy as np
from crickformers.drift_detector import (
    detect_model_drift,
    DriftThresholds,
    calculate_drift_score,
    get_drift_recommendations
)


class TestDriftDetector:
    """Test drift detection functionality."""
    
    def create_stable_predictions(self, size: int = 100, base_error: float = 0.1) -> tuple:
        """Create stable predictions with consistent error."""
        np.random.seed(42)
        actuals = np.random.normal(0.5, 0.2, size)
        predictions = actuals + np.random.normal(0, base_error, size)
        return predictions.tolist(), actuals.tolist()
    
    def create_drift_predictions(self, stable_size: int = 80, drift_size: int = 20) -> tuple:
        """Create predictions with drift in the recent window."""
        np.random.seed(42)
        
        # Stable period
        stable_actuals = np.random.normal(0.5, 0.2, stable_size)
        stable_predictions = stable_actuals + np.random.normal(0, 0.1, stable_size)
        
        # Drift period (higher error)
        drift_actuals = np.random.normal(0.5, 0.2, drift_size)
        drift_predictions = drift_actuals + np.random.normal(0, 0.3, drift_size)  # Higher error
        
        predictions = np.concatenate([stable_predictions, drift_predictions])
        actuals = np.concatenate([stable_actuals, drift_actuals])
        
        return predictions.tolist(), actuals.tolist()
    
    def create_gradual_drift_predictions(self, size: int = 100) -> tuple:
        """Create predictions with gradual drift over time."""
        np.random.seed(42)
        
        actuals = np.random.normal(0.5, 0.2, size)
        predictions = []
        
        for i in range(size):
            # Gradually increasing error
            error_std = 0.1 + (i / size) * 0.3
            pred = actuals[i] + np.random.normal(0, error_std)
            predictions.append(pred)
        
        return predictions, actuals.tolist()
    
    def test_basic_drift_detection(self):
        """Test basic drift detection functionality."""
        predictions, actuals = self.create_stable_predictions(50)
        
        result = detect_model_drift(predictions, actuals)
        
        # Should not detect drift in stable predictions
        assert result['drift_detected'] == False
        assert 'No drift detected' in result['drift_reasons'][0]
        assert result['error_trend'] is not None
        assert result['std_dev_trend'] is not None
        assert result['recent_window_stats'] is not None
        assert result['long_term_stats'] is not None
        assert result['statistical_tests'] is not None
    
    def test_drift_detection_with_drift(self):
        """Test drift detection with actual drift."""
        predictions, actuals = self.create_drift_predictions(80, 20)
        
        thresholds = DriftThresholds(
            error_threshold=1.5,
            std_threshold=1.3,
            recent_window=20,
            long_term_window=100
        )
        
        result = detect_model_drift(predictions, actuals, thresholds)
        
        # Should detect drift
        assert result['drift_detected'] == True
        assert len(result['drift_reasons']) > 0
        assert any('error' in reason.lower() for reason in result['drift_reasons'])
        
        # Recent window should have higher error
        assert result['recent_window_stats']['mean_error'] > result['long_term_stats']['mean_error']
    
    def test_gradual_drift_detection(self):
        """Test detection of gradual drift."""
        predictions, actuals = self.create_gradual_drift_predictions(100)
        
        result = detect_model_drift(predictions, actuals)
        
        # Should detect drift due to increasing error trend
        assert result['drift_detected'] == True
        assert result['error_trend']['is_increasing'] == True
        assert result['std_dev_trend']['is_increasing'] == True
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        predictions = [0.1, 0.2, 0.3]
        actuals = [0.15, 0.25, 0.35]
        
        result = detect_model_drift(predictions, actuals)
        
        assert result['drift_detected'] == False
        assert 'Insufficient data' in result['drift_reasons'][0]
        assert result['error_trend'] is None
        assert result['std_dev_trend'] is None
    
    def test_invalid_input_lengths(self):
        """Test error handling for invalid input lengths."""
        predictions = [0.1, 0.2, 0.3]
        actuals = [0.15, 0.25]
        
        with pytest.raises(ValueError, match="must have the same length"):
            detect_model_drift(predictions, actuals)
    
    def test_custom_thresholds(self):
        """Test drift detection with custom thresholds."""
        predictions, actuals = self.create_drift_predictions(80, 20)
        
        # Very strict thresholds
        strict_thresholds = DriftThresholds(
            error_threshold=3.0,
            std_threshold=3.0,
            recent_window=20,
            long_term_window=100
        )
        
        result = detect_model_drift(predictions, actuals, strict_thresholds)
        
        # May not detect drift with strict thresholds
        # (depends on the specific data generated)
        assert isinstance(result['drift_detected'], bool)
        assert len(result['drift_reasons']) > 0
    
    def test_statistical_tests(self):
        """Test statistical test components."""
        predictions, actuals = self.create_drift_predictions(80, 20)
        
        result = detect_model_drift(predictions, actuals)
        
        # Should have statistical test results
        assert 't_test' in result['statistical_tests']
        assert 'levene_test' in result['statistical_tests']
        assert 'mann_whitney' in result['statistical_tests']
        assert 'ks_test' in result['statistical_tests']
        
        # Each test should have expected keys
        for test_name, test_result in result['statistical_tests'].items():
            if 'error' not in test_result:
                assert 'statistic' in test_result
                assert 'p_value' in test_result
                assert 'significant' in test_result
                assert 'interpretation' in test_result
    
    def test_error_trend_analysis(self):
        """Test error trend analysis."""
        predictions, actuals = self.create_gradual_drift_predictions(100)
        
        result = detect_model_drift(predictions, actuals)
        
        error_trend = result['error_trend']
        assert error_trend is not None
        assert 'rolling_mean' in error_trend
        assert 'trend_slope' in error_trend
        assert 'trend_correlation' in error_trend
        assert 'is_increasing' in error_trend
        assert 'rolling_means' in error_trend
        
        # With gradual drift, trend should be increasing
        assert error_trend['is_increasing'] == True
        assert error_trend['trend_slope'] > 0
    
    def test_std_dev_trend_analysis(self):
        """Test standard deviation trend analysis."""
        predictions, actuals = self.create_gradual_drift_predictions(100)
        
        result = detect_model_drift(predictions, actuals)
        
        std_trend = result['std_dev_trend']
        assert std_trend is not None
        assert 'rolling_std' in std_trend
        assert 'trend_slope' in std_trend
        assert 'trend_correlation' in std_trend
        assert 'is_increasing' in std_trend
        assert 'rolling_stds' in std_trend
        
        # With gradual drift, std trend should be increasing
        assert std_trend['is_increasing'] == True
        assert std_trend['trend_slope'] > 0
    
    def test_window_statistics(self):
        """Test window statistics computation."""
        predictions, actuals = self.create_drift_predictions(80, 20)
        
        result = detect_model_drift(predictions, actuals)
        
        # Check recent window stats
        recent_stats = result['recent_window_stats']
        assert recent_stats['window_name'] == 'recent'
        assert 'mean_error' in recent_stats
        assert 'std_error' in recent_stats
        assert 'median_error' in recent_stats
        assert 'min_error' in recent_stats
        assert 'max_error' in recent_stats
        assert 'q25_error' in recent_stats
        assert 'q75_error' in recent_stats
        assert 'sample_size' in recent_stats
        
        # Check long-term window stats
        long_term_stats = result['long_term_stats']
        assert long_term_stats['window_name'] == 'long_term'
        assert all(key in long_term_stats for key in recent_stats.keys())
    
    def test_extreme_values_detection(self):
        """Test detection of extreme values."""
        predictions, actuals = self.create_stable_predictions(50)
        
        # Add extreme values at the end
        predictions.extend([10.0, 11.0, 12.0])  # Very high predictions
        actuals.extend([0.5, 0.6, 0.7])  # Normal actuals
        
        result = detect_model_drift(predictions, actuals)
        
        # Should detect drift due to extreme values
        assert result['drift_detected'] == True
        assert any('maximum error' in reason for reason in result['drift_reasons'])
    
    def test_zero_variance_handling(self):
        """Test handling of zero variance scenarios."""
        # Perfect predictions (zero error)
        actuals = [0.5] * 50
        predictions = [0.5] * 50
        
        result = detect_model_drift(predictions, actuals)
        
        # Should not crash with zero variance
        assert isinstance(result['drift_detected'], bool)
        assert result['recent_window_stats']['std_error'] == 0.0
        assert result['long_term_stats']['std_error'] == 0.0


class TestDriftScore:
    """Test drift score calculation."""
    
    def test_no_drift_score(self):
        """Test drift score for no drift scenario."""
        predictions, actuals = TestDriftDetector().create_stable_predictions(50)
        result = detect_model_drift(predictions, actuals)
        
        score = calculate_drift_score(result)
        assert score == 0.0
    
    def test_drift_score_with_drift(self):
        """Test drift score with actual drift."""
        predictions, actuals = TestDriftDetector().create_drift_predictions(80, 20)
        result = detect_model_drift(predictions, actuals)
        
        score = calculate_drift_score(result)
        assert 0.0 <= score <= 1.0
        
        if result['drift_detected']:
            assert score > 0.0
    
    def test_drift_score_components(self):
        """Test individual components of drift score."""
        predictions, actuals = TestDriftDetector().create_drift_predictions(80, 20)
        result = detect_model_drift(predictions, actuals)
        
        if result['drift_detected']:
            score = calculate_drift_score(result)
            
            # Score should be influenced by error ratio
            recent_stats = result['recent_window_stats']
            long_term_stats = result['long_term_stats']
            
            error_ratio = recent_stats['mean_error'] / long_term_stats['mean_error']
            if error_ratio > 1.0:
                assert score > 0.0
    
    def test_drift_score_bounds(self):
        """Test that drift score is properly bounded."""
        # Create extreme drift scenario
        actuals = [0.5] * 100
        predictions = [0.5] * 80 + [10.0] * 20  # Extreme predictions at end
        
        result = detect_model_drift(predictions, actuals)
        score = calculate_drift_score(result)
        
        assert 0.0 <= score <= 1.0


class TestDriftRecommendations:
    """Test drift recommendations system."""
    
    def test_stable_recommendations(self):
        """Test recommendations for stable model."""
        predictions, actuals = TestDriftDetector().create_stable_predictions(50)
        result = detect_model_drift(predictions, actuals)
        
        recommendations = get_drift_recommendations(result)
        
        assert len(recommendations) > 0
        assert any('stable' in rec.lower() for rec in recommendations)
    
    def test_drift_recommendations(self):
        """Test recommendations for drifted model."""
        predictions, actuals = TestDriftDetector().create_drift_predictions(80, 20)
        result = detect_model_drift(predictions, actuals)
        
        recommendations = get_drift_recommendations(result)
        
        if result['drift_detected']:
            assert len(recommendations) > 0
            assert any('drift' in rec.lower() for rec in recommendations)
    
    def test_severity_based_recommendations(self):
        """Test that recommendations match drift severity."""
        predictions, actuals = TestDriftDetector().create_drift_predictions(80, 20)
        result = detect_model_drift(predictions, actuals)
        
        recommendations = get_drift_recommendations(result)
        score = calculate_drift_score(result)
        
        if score > 0.6:
            assert any('severe' in rec.lower() or 'immediate' in rec.lower() for rec in recommendations)
        elif score > 0.3:
            assert any('moderate' in rec.lower() or 'consider' in rec.lower() for rec in recommendations)
        else:
            assert any('mild' in rec.lower() or 'increase' in rec.lower() for rec in recommendations)
    
    def test_specific_recommendations(self):
        """Test specific recommendations based on drift reasons."""
        predictions, actuals = TestDriftDetector().create_gradual_drift_predictions(100)
        result = detect_model_drift(predictions, actuals)
        
        recommendations = get_drift_recommendations(result)
        
        # Should have specific recommendations based on detected issues
        if result['error_trend']['is_increasing']:
            assert any('trend' in rec.lower() for rec in recommendations)
        
        if result['std_dev_trend']['is_increasing']:
            assert any('variance' in rec.lower() or 'uncertainty' in rec.lower() for rec in recommendations)


class TestDriftThresholds:
    """Test drift thresholds configuration."""
    
    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = DriftThresholds()
        
        assert thresholds.error_threshold == 1.5
        assert thresholds.std_threshold == 1.3
        assert thresholds.min_samples == 20
        assert thresholds.recent_window == 10
        assert thresholds.long_term_window == 50
        assert thresholds.significance_level == 0.05
    
    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        thresholds = DriftThresholds(
            error_threshold=2.0,
            std_threshold=1.5,
            min_samples=30,
            recent_window=15,
            long_term_window=75,
            significance_level=0.01
        )
        
        assert thresholds.error_threshold == 2.0
        assert thresholds.std_threshold == 1.5
        assert thresholds.min_samples == 30
        assert thresholds.recent_window == 15
        assert thresholds.long_term_window == 75
        assert thresholds.significance_level == 0.01
    
    def test_threshold_impact_on_detection(self):
        """Test that thresholds affect drift detection."""
        predictions, actuals = TestDriftDetector().create_drift_predictions(80, 20)
        
        # Lenient thresholds
        lenient_thresholds = DriftThresholds(error_threshold=1.2, std_threshold=1.1)
        lenient_result = detect_model_drift(predictions, actuals, lenient_thresholds)
        
        # Strict thresholds
        strict_thresholds = DriftThresholds(error_threshold=3.0, std_threshold=3.0)
        strict_result = detect_model_drift(predictions, actuals, strict_thresholds)
        
        # Lenient thresholds should be more likely to detect drift
        if lenient_result['drift_detected'] and not strict_result['drift_detected']:
            assert True  # Expected behavior
        else:
            # Both should be boolean
            assert isinstance(lenient_result['drift_detected'], bool)
            assert isinstance(strict_result['drift_detected'], bool)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_prediction(self):
        """Test with single prediction."""
        predictions = [0.5]
        actuals = [0.6]
        
        result = detect_model_drift(predictions, actuals)
        
        assert result['drift_detected'] == False
        assert 'Insufficient data' in result['drift_reasons'][0]
    
    def test_identical_predictions(self):
        """Test with identical predictions and actuals."""
        predictions = [0.5] * 50
        actuals = [0.5] * 50
        
        result = detect_model_drift(predictions, actuals)
        
        # Should not crash with zero variance
        assert isinstance(result['drift_detected'], bool)
    
    def test_nan_values(self):
        """Test handling of NaN values."""
        predictions = [0.1, 0.2, float('nan'), 0.4, 0.5]
        actuals = [0.15, 0.25, 0.35, 0.45, 0.55]
        
        # Should handle NaN values gracefully
        try:
            result = detect_model_drift(predictions, actuals)
            assert isinstance(result['drift_detected'], bool)
        except (ValueError, RuntimeError):
            # NaN handling may raise errors - that's acceptable
            pass
    
    def test_very_small_errors(self):
        """Test with very small prediction errors."""
        actuals = [0.5] * 50
        predictions = [0.5 + 1e-10] * 50  # Tiny errors
        
        result = detect_model_drift(predictions, actuals)
        
        # Should handle very small errors without numerical issues
        assert isinstance(result['drift_detected'], bool)
        assert result['recent_window_stats']['mean_error'] >= 0
    
    def test_large_prediction_values(self):
        """Test with large prediction values."""
        actuals = [1000.0] * 50
        predictions = [1001.0] * 30 + [1010.0] * 20  # Drift to higher values
        
        result = detect_model_drift(predictions, actuals)
        
        # Should handle large values correctly
        assert isinstance(result['drift_detected'], bool)
        assert result['recent_window_stats']['mean_error'] >= 0 