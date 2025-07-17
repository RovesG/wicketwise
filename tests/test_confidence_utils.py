# Purpose: Tests for Monte Carlo dropout confidence estimation utilities
# Author: Shamus Rae, Last Modified: 2024-12-19

"""
Test suite for confidence_utils.py module.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch

from crickformers.confidence_utils import (
    predict_with_uncertainty,
    has_dropout_layers,
    get_dropout_modules,
    estimate_prediction_confidence,
    ConfidenceTracker
)


class MockModelWithDropout(nn.Module):
    """Mock model with dropout layers for testing."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class MockModelWithoutDropout(nn.Module):
    """Mock model without dropout layers for testing."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class MockModelWithTupleInputs(nn.Module):
    """Mock model that accepts tuple inputs."""
    
    def __init__(self, input_dim1=10, input_dim2=5, output_dim=3):
        super().__init__()
        self.layer1 = nn.Linear(input_dim1, 15)
        self.layer2 = nn.Linear(input_dim2, 10)
        self.dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(25, output_dim)
    
    def forward(self, x1, x2):
        out1 = self.layer1(x1)
        out2 = self.layer2(x2)
        combined = torch.cat([out1, out2], dim=1)
        dropped = self.dropout(combined)
        return self.output_layer(dropped)


class MockModelWithDictInputs(nn.Module):
    """Mock model that accepts dictionary inputs."""
    
    def __init__(self, input_dim=10, output_dim=3):
        super().__init__()
        self.layer = nn.Linear(input_dim, 15)
        self.dropout = nn.Dropout(0.4)
        self.output_layer = nn.Linear(15, output_dim)
    
    def forward(self, features, mask=None):
        out = self.layer(features)
        if mask is not None:
            out = out * mask
        dropped = self.dropout(out)
        return self.output_layer(dropped)


class TestPredictWithUncertainty:
    """Test predict_with_uncertainty function."""
    
    def test_basic_functionality(self):
        """Test basic uncertainty prediction with dropout model."""
        model = MockModelWithDropout(input_dim=10, output_dim=5)
        inputs = torch.randn(3, 10)
        
        mean_pred, std_pred, lower_bound, upper_bound = predict_with_uncertainty(
            model, inputs, n_samples=10, confidence_level=0.95
        )
        
        # Check shapes
        assert mean_pred.shape == (3, 5)
        assert std_pred.shape == (3, 5)
        assert lower_bound.shape == (3, 5)
        assert upper_bound.shape == (3, 5)
        
        # Check that bounds are reasonable
        assert torch.all(lower_bound <= mean_pred)
        assert torch.all(mean_pred <= upper_bound)
        
        # Check that standard deviation is positive
        assert torch.all(std_pred >= 0)
    
    def test_variance_with_dropout(self):
        """Test that dropout creates variance in predictions."""
        model = MockModelWithDropout(dropout_rate=0.5)
        inputs = torch.randn(2, 10)
        
        # Test with high number of samples for more reliable statistics
        mean_pred, std_pred, _, _ = predict_with_uncertainty(
            model, inputs, n_samples=50
        )
        
        # With dropout, there should be some variance
        assert torch.any(std_pred > 0.01)  # Some elements should have noticeable variance
    
    def test_different_confidence_levels(self):
        """Test different confidence levels produce different bounds."""
        model = MockModelWithDropout()
        inputs = torch.randn(2, 10)
        
        # Test 90% confidence
        _, _, lower_90, upper_90 = predict_with_uncertainty(
            model, inputs, n_samples=20, confidence_level=0.90
        )
        
        # Test 99% confidence  
        _, _, lower_99, upper_99 = predict_with_uncertainty(
            model, inputs, n_samples=20, confidence_level=0.99
        )
        
        # 99% intervals should be wider than 90%
        interval_90 = upper_90 - lower_90
        interval_99 = upper_99 - lower_99
        assert torch.all(interval_99 >= interval_90)
    
    def test_tuple_inputs(self):
        """Test with tuple inputs."""
        model = MockModelWithTupleInputs()
        inputs = (torch.randn(2, 10), torch.randn(2, 5))
        
        mean_pred, std_pred, lower_bound, upper_bound = predict_with_uncertainty(
            model, inputs, n_samples=10
        )
        
        # Check shapes
        assert mean_pred.shape == (2, 3)
        assert std_pred.shape == (2, 3)
        assert lower_bound.shape == (2, 3)
        assert upper_bound.shape == (2, 3)
    
    def test_dict_inputs(self):
        """Test with dictionary inputs."""
        model = MockModelWithDictInputs()
        inputs = {
            'features': torch.randn(2, 10),
            'mask': torch.ones(2, 15)
        }
        
        mean_pred, std_pred, lower_bound, upper_bound = predict_with_uncertainty(
            model, inputs, n_samples=10
        )
        
        # Check shapes
        assert mean_pred.shape == (2, 3)
        assert std_pred.shape == (2, 3)
        assert lower_bound.shape == (2, 3)
        assert upper_bound.shape == (2, 3)
    
    def test_training_mode_restoration(self):
        """Test that model training mode is restored after prediction."""
        model = MockModelWithDropout()
        inputs = torch.randn(2, 10)
        
        # Set model to eval mode
        model.eval()
        assert not model.training
        
        # Run prediction
        predict_with_uncertainty(model, inputs, n_samples=5)
        
        # Should be back to eval mode
        assert not model.training
        
        # Set model to train mode
        model.train()
        assert model.training
        
        # Run prediction
        predict_with_uncertainty(model, inputs, n_samples=5)
        
        # Should be back to train mode
        assert model.training
    
    def test_device_handling(self):
        """Test proper device handling."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = MockModelWithDropout().to(device)
            inputs = torch.randn(2, 10).to(device)
            
            mean_pred, std_pred, lower_bound, upper_bound = predict_with_uncertainty(
                model, inputs, n_samples=5, device=device
            )
            
            # All outputs should be on the same device
            assert mean_pred.device == device
            assert std_pred.device == device
            assert lower_bound.device == device
            assert upper_bound.device == device
    
    def test_invalid_input_type(self):
        """Test error handling for invalid input types."""
        model = MockModelWithDropout()
        
        with pytest.raises(ValueError, match="Unsupported input type"):
            predict_with_uncertainty(model, "invalid_input", n_samples=5)


class TestDropoutDetection:
    """Test dropout layer detection utilities."""
    
    def test_has_dropout_layers_with_dropout(self):
        """Test dropout detection with dropout model."""
        model = MockModelWithDropout()
        assert has_dropout_layers(model) is True
    
    def test_has_dropout_layers_without_dropout(self):
        """Test dropout detection without dropout model."""
        model = MockModelWithoutDropout()
        assert has_dropout_layers(model) is False
    
    def test_get_dropout_modules(self):
        """Test getting dropout modules."""
        model = MockModelWithDropout()
        dropout_modules = get_dropout_modules(model)
        
        # Should find 2 dropout layers
        assert len(dropout_modules) == 2
        
        # Check that they are indeed dropout modules
        for name, module in dropout_modules:
            assert isinstance(module, nn.Dropout)
            assert 'dropout' in name.lower() or isinstance(module, nn.Dropout)
    
    def test_different_dropout_types(self):
        """Test detection of different dropout types."""
        class ModelWithVariousDropouts(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout1d = nn.Dropout1d(0.2)
                self.dropout2d = nn.Dropout2d(0.3)
                self.dropout3d = nn.Dropout3d(0.4)
                self.regular_dropout = nn.Dropout(0.5)
        
        model = ModelWithVariousDropouts()
        assert has_dropout_layers(model) is True
        
        dropout_modules = get_dropout_modules(model)
        assert len(dropout_modules) == 4


class TestConfidenceEstimation:
    """Test confidence estimation utilities."""
    
    def test_estimate_prediction_confidence(self):
        """Test confidence score estimation."""
        predictions = torch.randn(10, 5)
        uncertainty = torch.rand(10, 5) * 0.5  # Random uncertainty values
        
        confidence = estimate_prediction_confidence(predictions, uncertainty)
        
        # Confidence should be between 0 and 1
        assert torch.all(confidence >= 0)
        assert torch.all(confidence <= 1)
        
        # Lower uncertainty should give higher confidence
        low_uncertainty = torch.ones(10, 5) * 0.01
        high_uncertainty = torch.ones(10, 5) * 0.5
        
        low_conf = estimate_prediction_confidence(predictions, low_uncertainty)
        high_conf = estimate_prediction_confidence(predictions, high_uncertainty)
        
        assert torch.all(low_conf >= high_conf)
    
    def test_confidence_with_threshold(self):
        """Test confidence estimation with different thresholds."""
        predictions = torch.randn(5, 3)
        uncertainty = torch.ones(5, 3) * 0.1
        
        # Lower threshold should give lower confidence for same uncertainty
        conf_low_thresh = estimate_prediction_confidence(predictions, uncertainty, 0.05)
        conf_high_thresh = estimate_prediction_confidence(predictions, uncertainty, 0.2)
        
        assert torch.all(conf_low_thresh <= conf_high_thresh)


class TestConfidenceTracker:
    """Test ConfidenceTracker class."""
    
    def test_initialization(self):
        """Test tracker initialization."""
        tracker = ConfidenceTracker()
        
        assert len(tracker.predictions) == 0
        assert len(tracker.uncertainties) == 0
        assert len(tracker.confidence_scores) == 0
    
    def test_add_prediction(self):
        """Test adding predictions to tracker."""
        tracker = ConfidenceTracker()
        
        prediction = torch.randn(2, 3)
        uncertainty = torch.rand(2, 3)
        confidence = torch.rand(2, 3)
        
        tracker.add_prediction(prediction, uncertainty, confidence)
        
        assert len(tracker.predictions) == 1
        assert len(tracker.uncertainties) == 1
        assert len(tracker.confidence_scores) == 1
    
    def test_get_statistics(self):
        """Test getting statistics from tracker."""
        tracker = ConfidenceTracker()
        
        # Add some predictions
        for i in range(3):
            prediction = torch.randn(2, 3)
            uncertainty = torch.rand(2, 3) * 0.5
            confidence = torch.rand(2, 3)
            tracker.add_prediction(prediction, uncertainty, confidence)
        
        stats = tracker.get_statistics()
        
        expected_keys = [
            'mean_uncertainty', 'std_uncertainty', 'mean_confidence', 
            'std_confidence', 'min_confidence', 'max_confidence', 'num_predictions'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats['num_predictions'] == 3
        assert 0 <= stats['mean_confidence'] <= 1
        assert 0 <= stats['min_confidence'] <= 1
        assert 0 <= stats['max_confidence'] <= 1
    
    def test_clear(self):
        """Test clearing tracker."""
        tracker = ConfidenceTracker()
        
        # Add a prediction
        prediction = torch.randn(2, 3)
        uncertainty = torch.rand(2, 3)
        confidence = torch.rand(2, 3)
        tracker.add_prediction(prediction, uncertainty, confidence)
        
        # Clear
        tracker.clear()
        
        assert len(tracker.predictions) == 0
        assert len(tracker.uncertainties) == 0
        assert len(tracker.confidence_scores) == 0
    
    def test_empty_statistics(self):
        """Test statistics with empty tracker."""
        tracker = ConfidenceTracker()
        stats = tracker.get_statistics()
        
        assert stats == {}


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_workflow(self):
        """Test complete workflow with model, prediction, and tracking."""
        # Create model and inputs
        model = MockModelWithDropout(dropout_rate=0.3)
        inputs = torch.randn(4, 10)
        
        # Get uncertainty predictions
        mean_pred, std_pred, lower_bound, upper_bound = predict_with_uncertainty(
            model, inputs, n_samples=15
        )
        
        # Estimate confidence
        confidence = estimate_prediction_confidence(mean_pred, std_pred)
        
        # Track results
        tracker = ConfidenceTracker()
        tracker.add_prediction(mean_pred, std_pred, confidence)
        
        # Get statistics
        stats = tracker.get_statistics()
        
        # Validate everything works together
        assert stats['num_predictions'] == 1
        assert 0 <= stats['mean_confidence'] <= 1
        assert torch.all(confidence >= 0)
        assert torch.all(confidence <= 1)
        assert torch.all(lower_bound <= upper_bound)
    
    def test_model_without_dropout_warning(self):
        """Test behavior with model without dropout."""
        model = MockModelWithoutDropout()
        inputs = torch.randn(2, 10)
        
        # Should still work but with minimal variance
        mean_pred, std_pred, lower_bound, upper_bound = predict_with_uncertainty(
            model, inputs, n_samples=10
        )
        
        # Standard deviation should be very small (numerical precision effects only)
        assert torch.all(std_pred < 0.001)
        
        # Bounds should be very close to mean
        assert torch.allclose(lower_bound, mean_pred, atol=0.001)
        assert torch.allclose(upper_bound, mean_pred, atol=0.001) 