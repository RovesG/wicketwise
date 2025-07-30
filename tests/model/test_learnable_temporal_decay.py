# Purpose: Unit tests for learnable temporal decay mechanism
# Author: Shamus Rae, Last Modified: 2024-01-15

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
import math

from crickformers.model.learnable_temporal_decay import (
    LearnableTemporalDecay,
    AdaptiveTemporalEncoder,
    TemporalDecayLoss,
    create_learnable_temporal_decay
)


class TestLearnableTemporalDecay:
    """Test learnable temporal decay module."""
    
    @pytest.fixture
    def feature_names(self):
        """Sample feature names."""
        return ['batting_avg', 'strike_rate', 'recent_form', 'pressure_score']
    
    @pytest.fixture
    def temporal_decay(self, feature_names):
        """Create temporal decay module."""
        return LearnableTemporalDecay(
            feature_names=feature_names,
            initial_half_life=30.0,
            min_half_life=1.0,
            max_half_life=365.0,
            learnable=True
        )
    
    @pytest.fixture
    def fixed_temporal_decay(self, feature_names):
        """Create non-learnable temporal decay module."""
        return LearnableTemporalDecay(
            feature_names=feature_names,
            initial_half_life=30.0,
            learnable=False
        )
    
    def test_initialization(self, temporal_decay, feature_names):
        """Test temporal decay initialization."""
        assert temporal_decay.feature_names == feature_names
        assert temporal_decay.num_features == len(feature_names)
        assert temporal_decay.learnable
        assert temporal_decay.min_half_life == 1.0
        assert temporal_decay.max_half_life == 365.0
        
        # Check parameter initialization
        assert hasattr(temporal_decay, 'log_half_lives')
        assert temporal_decay.log_half_lives.requires_grad
        assert temporal_decay.log_half_lives.shape == (len(feature_names),)
        
        # Check feature mapping
        for i, name in enumerate(feature_names):
            assert temporal_decay.feature_to_idx[name] == i
    
    def test_fixed_initialization(self, fixed_temporal_decay, feature_names):
        """Test non-learnable temporal decay initialization."""
        assert not fixed_temporal_decay.learnable
        assert not fixed_temporal_decay.log_half_lives.requires_grad
    
    def test_get_half_lives(self, temporal_decay):
        """Test half-life parameter retrieval."""
        half_lives = temporal_decay.get_half_lives()
        
        assert isinstance(half_lives, torch.Tensor)
        assert half_lives.shape == (temporal_decay.num_features,)
        assert torch.all(half_lives >= temporal_decay.min_half_life)
        assert torch.all(half_lives <= temporal_decay.max_half_life)
        
        # Check initial values are around 30 days
        assert torch.allclose(half_lives, torch.tensor(30.0), atol=5.0)
    
    def test_compute_temporal_weights_basic(self, temporal_decay):
        """Test basic temporal weight computation."""
        days_ago = torch.tensor([0.0, 10.0, 30.0, 90.0])
        weights = temporal_decay.compute_temporal_weights(days_ago)
        
        assert weights.shape == days_ago.shape
        assert torch.all(weights > 0)
        assert torch.all(weights <= 1.0)
        
        # Weights should decrease with time
        assert weights[0] > weights[1] > weights[2] > weights[3]
        
        # Weight at 0 days should be 1.0
        assert torch.allclose(weights[0], torch.tensor(1.0), atol=1e-6)
    
    def test_compute_temporal_weights_with_features(self, temporal_decay):
        """Test temporal weight computation with feature indices."""
        days_ago = torch.tensor([30.0, 30.0, 30.0, 30.0])
        feature_indices = torch.tensor([0, 1, 2, 3])
        
        weights = temporal_decay.compute_temporal_weights(days_ago, feature_indices)
        
        assert weights.shape == days_ago.shape
        
        # With same days_ago and initial identical half-lives, weights should be similar
        assert torch.allclose(weights[0], weights[1], atol=1e-3)
    
    def test_temporal_weight_formula(self, temporal_decay):
        """Test that temporal weights follow the correct formula."""
        # Set specific half-life for testing
        with torch.no_grad():
            temporal_decay.log_half_lives.fill_(math.log(30.0))  # 30-day half-life
        
        days_ago = torch.tensor([30.0])  # Exactly one half-life
        weights = temporal_decay.compute_temporal_weights(days_ago)
        
        # Weight should be 0.5 after one half-life
        expected_weight = 0.5
        assert torch.allclose(weights, torch.tensor(expected_weight), atol=1e-4)
        
        # Test multiple half-lives
        days_ago = torch.tensor([60.0])  # Two half-lives
        weights = temporal_decay.compute_temporal_weights(days_ago)
        expected_weight = 0.25  # (1/2)^2
        assert torch.allclose(weights, torch.tensor(expected_weight), atol=1e-4)
    
    def test_compute_feature_weights(self, temporal_decay, feature_names):
        """Test feature weight computation."""
        batch_size = 3
        days_ago = torch.tensor([0.0, 15.0, 45.0])
        feature_values = torch.randn(batch_size, len(feature_names))
        
        weighted_features = temporal_decay.compute_feature_weights(
            days_ago, feature_values, feature_names
        )
        
        assert weighted_features.shape == feature_values.shape
        
        # Recent data should have higher weights
        recent_magnitude = torch.norm(weighted_features[0])
        old_magnitude = torch.norm(weighted_features[2])
        assert recent_magnitude >= old_magnitude
    
    def test_compute_edge_weights(self, temporal_decay):
        """Test edge weight computation."""
        num_edges = 5
        days_ago = torch.tensor([0.0, 10.0, 20.0, 30.0, 60.0])
        base_weights = torch.ones(num_edges)
        
        edge_weights = temporal_decay.compute_edge_weights(days_ago, base_weights)
        
        assert edge_weights.shape == base_weights.shape
        assert torch.all(edge_weights > 0)
        
        # Recent edges should have higher weights
        assert edge_weights[0] > edge_weights[1] > edge_weights[2]
    
    def test_get_aggregated_form_vector(self, temporal_decay, feature_names):
        """Test form vector aggregation."""
        history_length = 10
        feature_history = torch.randn(history_length, len(feature_names))
        days_ago_history = torch.linspace(0, 90, history_length)
        
        form_vector = temporal_decay.get_aggregated_form_vector(
            feature_history, days_ago_history, feature_names
        )
        
        assert form_vector.shape == (len(feature_names),)
        assert torch.all(torch.isfinite(form_vector))
    
    def test_statistics_tracking(self, temporal_decay, feature_names):
        """Test statistics tracking during training."""
        # Simulate training mode
        temporal_decay.train()
        
        # Perform some forward passes
        for _ in range(5):
            days_ago = torch.randn(10).abs()
            feature_values = torch.randn(10, len(feature_names))
            temporal_decay.compute_feature_weights(days_ago, feature_values, feature_names)
        
        # Check statistics
        stats = temporal_decay.get_statistics()
        assert len(stats) > 0
        
        for feature_name in feature_names:
            assert f"{feature_name}_half_life" in stats
            assert f"{feature_name}_weight_mean" in stats
    
    def test_regularization_loss(self, temporal_decay):
        """Test regularization loss computation."""
        l1_weight = 0.01
        l2_weight = 0.01
        
        reg_loss = temporal_decay.regularization_loss(l1_weight, l2_weight)
        
        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0
        assert reg_loss.requires_grad == temporal_decay.learnable
    
    def test_forward_pass(self, temporal_decay, feature_names):
        """Test forward pass functionality."""
        days_ago = torch.tensor([10.0, 20.0, 30.0])
        
        # Test without feature values
        weights = temporal_decay(days_ago)
        assert weights.shape == days_ago.shape
        
        # Test with feature values
        feature_values = torch.randn(3, len(feature_names))
        weights, weighted_features = temporal_decay(days_ago, feature_values, feature_names)
        
        assert weights.shape == (3, len(feature_names))
        assert weighted_features.shape == feature_values.shape
    
    def test_temperature_scaling(self, feature_names):
        """Test temperature scaling effect."""
        temp_1 = LearnableTemporalDecay(feature_names, temperature=1.0)
        temp_2 = LearnableTemporalDecay(feature_names, temperature=2.0)
        
        days_ago = torch.tensor([30.0])
        
        weights_1 = temp_1.compute_temporal_weights(days_ago)
        weights_2 = temp_2.compute_temporal_weights(days_ago)
        
        # Higher temperature should produce smoother (closer to 1) weights
        assert weights_2 > weights_1
    
    def test_gradient_flow(self, temporal_decay):
        """Test gradient flow through temporal parameters."""
        days_ago = torch.tensor([10.0, 20.0, 30.0])
        feature_values = torch.randn(3, temporal_decay.num_features, requires_grad=True)
        
        weighted_features = temporal_decay.compute_feature_weights(
            days_ago, feature_values, temporal_decay.feature_names
        )
        
        loss = weighted_features.sum()
        loss.backward()
        
        # Check gradients exist
        assert temporal_decay.log_half_lives.grad is not None
        assert feature_values.grad is not None
        
        # Gradients should be non-zero
        assert torch.any(temporal_decay.log_half_lives.grad != 0)
    
    def test_half_life_bounds(self, temporal_decay):
        """Test half-life parameter bounds."""
        # Set extreme values
        with torch.no_grad():
            temporal_decay.log_half_lives.fill_(math.log(1000.0))  # Very large
        
        half_lives = temporal_decay.get_half_lives()
        assert torch.all(half_lives <= temporal_decay.max_half_life)
        
        # Set very small values
        with torch.no_grad():
            temporal_decay.log_half_lives.fill_(math.log(0.1))  # Very small
        
        half_lives = temporal_decay.get_half_lives()
        assert torch.all(half_lives >= temporal_decay.min_half_life)


class TestAdaptiveTemporalEncoder:
    """Test adaptive temporal encoder."""
    
    @pytest.fixture
    def feature_names(self):
        """Sample feature names."""
        return ['batting_avg', 'strike_rate', 'recent_form']
    
    @pytest.fixture
    def encoder(self, feature_names):
        """Create adaptive temporal encoder."""
        return AdaptiveTemporalEncoder(
            feature_names=feature_names,
            embed_dim=32,
            max_days=365,
            use_positional_encoding=True
        )
    
    def test_initialization(self, encoder, feature_names):
        """Test encoder initialization."""
        assert encoder.feature_names == feature_names
        assert encoder.embed_dim == 32
        assert encoder.max_days == 365
        assert encoder.use_positional_encoding
        
        # Check modules
        assert hasattr(encoder, 'temporal_decay')
        assert hasattr(encoder, 'temporal_embedding')
        assert hasattr(encoder, 'feature_encoders')
        assert len(encoder.feature_encoders) == len(feature_names)
    
    def test_forward_pass(self, encoder, feature_names):
        """Test encoder forward pass."""
        batch_size = 4
        days_ago = torch.tensor([0.0, 10.0, 30.0, 90.0])
        feature_values = torch.randn(batch_size, len(feature_names))
        
        output = encoder(days_ago, feature_values, feature_names)
        
        assert output.shape == (batch_size, len(feature_names))
        assert torch.all(torch.isfinite(output))
    
    def test_without_positional_encoding(self, feature_names):
        """Test encoder without positional encoding."""
        encoder = AdaptiveTemporalEncoder(
            feature_names=feature_names,
            embed_dim=32,
            use_positional_encoding=False
        )
        
        days_ago = torch.tensor([10.0, 20.0])
        feature_values = torch.randn(2, len(feature_names))
        
        output = encoder(days_ago, feature_values)
        assert output.shape == (2, len(feature_names))


class TestTemporalDecayLoss:
    """Test temporal decay loss function."""
    
    @pytest.fixture
    def feature_names(self):
        """Sample feature names."""
        return ['batting_avg', 'strike_rate']
    
    @pytest.fixture
    def temporal_decay(self, feature_names):
        """Create temporal decay module."""
        return LearnableTemporalDecay(feature_names)
    
    @pytest.fixture
    def loss_fn(self, temporal_decay):
        """Create temporal decay loss."""
        return TemporalDecayLoss(
            decay_module=temporal_decay,
            target_half_lives={'batting_avg': 25.0, 'strike_rate': 35.0},
            consistency_weight=0.1,
            smoothness_weight=0.05
        )
    
    def test_initialization(self, loss_fn, temporal_decay):
        """Test loss function initialization."""
        assert loss_fn.decay_module == temporal_decay
        assert 'batting_avg' in loss_fn.target_half_lives
        assert loss_fn.consistency_weight == 0.1
        assert loss_fn.smoothness_weight == 0.05
    
    def test_forward_pass(self, loss_fn):
        """Test loss function forward pass."""
        batch_size = 8
        predictions = torch.randn(batch_size, 1)
        targets = torch.randn(batch_size, 1)
        days_ago = torch.randn(batch_size).abs()
        feature_values = torch.randn(batch_size, 2)
        
        losses = loss_fn(predictions, targets, days_ago, feature_values)
        
        assert isinstance(losses, dict)
        assert 'main_loss' in losses
        assert 'total_loss' in losses
        assert 'consistency_loss' in losses
        assert 'smoothness_loss' in losses
        assert 'supervised_loss' in losses
        
        # All losses should be non-negative
        for key, value in losses.items():
            assert value.item() >= 0
    
    def test_supervised_loss(self, loss_fn):
        """Test supervised loss for target half-lives."""
        # Set specific half-lives
        with torch.no_grad():
            loss_fn.decay_module.log_half_lives[0] = math.log(25.0)  # batting_avg
            loss_fn.decay_module.log_half_lives[1] = math.log(35.0)  # strike_rate
        
        supervised_loss = loss_fn._compute_supervised_loss()
        
        # Loss should be very small when half-lives match targets
        assert supervised_loss.item() < 0.1
    
    def test_consistency_loss(self, loss_fn):
        """Test temporal consistency loss."""
        # Create sorted time series
        days_ago = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        feature_values = torch.randn(5, 2)
        
        consistency_loss = loss_fn._compute_consistency_loss(days_ago, feature_values)
        
        assert isinstance(consistency_loss, torch.Tensor)
        assert consistency_loss.item() >= 0
    
    def test_smoothness_loss(self, loss_fn):
        """Test smoothness regularization loss."""
        smoothness_loss = loss_fn._compute_smoothness_loss()
        
        assert isinstance(smoothness_loss, torch.Tensor)
        assert smoothness_loss.item() >= 0


class TestTemporalDecayValidation:
    """Test temporal decay behavior validation."""
    
    @pytest.fixture
    def feature_names(self):
        """Sample feature names."""
        return ['feature_a', 'feature_b', 'feature_c']
    
    def test_decay_weighting_over_time(self, feature_names):
        """Test that decay weighting changes correctly over time."""
        temporal_decay = LearnableTemporalDecay(feature_names, initial_half_life=30.0)
        
        # Test at different time points
        time_points = torch.tensor([0.0, 15.0, 30.0, 60.0, 90.0])
        weights = temporal_decay.compute_temporal_weights(time_points)
        
        # Weights should decrease monotonically
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1]
        
        # Weight at 0 should be 1.0
        assert torch.allclose(weights[0], torch.tensor(1.0), atol=1e-6)
        
        # Weight at one half-life should be 0.5
        half_life_weight = temporal_decay.compute_temporal_weights(torch.tensor([30.0]))
        assert torch.allclose(half_life_weight, torch.tensor(0.5), atol=1e-3)
    
    def test_feature_specific_behavior(self, feature_names):
        """Test feature-specific decay behavior."""
        temporal_decay = LearnableTemporalDecay(feature_names, learnable=True)
        
        # Set different half-lives for different features
        with torch.no_grad():
            temporal_decay.log_half_lives[0] = math.log(10.0)  # Fast decay
            temporal_decay.log_half_lives[1] = math.log(30.0)  # Medium decay
            temporal_decay.log_half_lives[2] = math.log(90.0)  # Slow decay
        
        days_ago = torch.tensor([30.0])
        feature_indices = torch.tensor([0, 1, 2])
        
        weights = temporal_decay.compute_temporal_weights(
            days_ago.unsqueeze(0), feature_indices.unsqueeze(0)
        )
        
        # Feature with shorter half-life should have lower weight
        assert weights[0, 0] < weights[0, 1] < weights[0, 2]
    
    def test_learnability_gradient_flow(self, feature_names):
        """Test that parameters are learnable with proper gradient flow."""
        temporal_decay = LearnableTemporalDecay(feature_names, learnable=True)
        
        # Create dummy data
        days_ago = torch.tensor([10.0, 20.0, 30.0])
        feature_values = torch.randn(3, len(feature_names), requires_grad=True)
        
        # Forward pass
        weighted_features = temporal_decay.compute_feature_weights(
            days_ago, feature_values, feature_names
        )
        
        # Create loss and backpropagate
        loss = weighted_features.mean()
        loss.backward()
        
        # Check that temporal parameters have gradients
        assert temporal_decay.log_half_lives.grad is not None
        assert torch.any(temporal_decay.log_half_lives.grad != 0)
        
        # Check gradient magnitudes are reasonable
        grad_norms = torch.norm(temporal_decay.log_half_lives.grad)
        assert 1e-6 < grad_norms < 1e2  # Reasonable gradient magnitude
    
    def test_optimization_step(self, feature_names):
        """Test that parameters can be optimized."""
        temporal_decay = LearnableTemporalDecay(feature_names, learnable=True)
        optimizer = torch.optim.Adam(temporal_decay.parameters(), lr=0.01)
        
        # Store initial half-lives
        initial_half_lives = temporal_decay.get_half_lives().clone()
        
        # Simulate training steps
        for _ in range(10):
            optimizer.zero_grad()
            
            days_ago = torch.randn(5).abs()
            feature_values = torch.randn(5, len(feature_names))
            
            weighted_features = temporal_decay.compute_feature_weights(
                days_ago, feature_values, feature_names
            )
            
            # Create artificial loss that prefers larger half-lives
            target_half_lives = torch.tensor([60.0] * len(feature_names))
            current_half_lives = temporal_decay.get_half_lives()
            loss = torch.nn.functional.mse_loss(current_half_lives, target_half_lives)
            
            loss.backward()
            optimizer.step()
        
        # Half-lives should have changed
        final_half_lives = temporal_decay.get_half_lives()
        assert not torch.allclose(initial_half_lives, final_half_lives, atol=1e-3)
        
        # Half-lives should have moved toward target
        initial_distance = torch.norm(initial_half_lives - target_half_lives)
        final_distance = torch.norm(final_half_lives - target_half_lives)
        assert final_distance < initial_distance
    
    def test_bounds_enforcement(self, feature_names):
        """Test that half-life bounds are enforced."""
        min_half_life = 5.0
        max_half_life = 100.0
        
        temporal_decay = LearnableTemporalDecay(
            feature_names,
            min_half_life=min_half_life,
            max_half_life=max_half_life,
            learnable=True
        )
        
        # Try to set extreme values
        with torch.no_grad():
            temporal_decay.log_half_lives.fill_(math.log(1000.0))  # Very large
        
        half_lives = temporal_decay.get_half_lives()
        assert torch.all(half_lives <= max_half_life)
        
        # Try very small values
        with torch.no_grad():
            temporal_decay.log_half_lives.fill_(math.log(0.1))  # Very small
        
        half_lives = temporal_decay.get_half_lives()
        assert torch.all(half_lives >= min_half_life)
    
    def test_numerical_stability(self, feature_names):
        """Test numerical stability with extreme inputs."""
        temporal_decay = LearnableTemporalDecay(feature_names)
        
        # Test with very large days_ago
        large_days = torch.tensor([1000.0, 10000.0])
        weights = temporal_decay.compute_temporal_weights(large_days)
        
        assert torch.all(torch.isfinite(weights))
        assert torch.all(weights >= 0)
        
        # Test with zero days_ago
        zero_days = torch.tensor([0.0])
        weights = temporal_decay.compute_temporal_weights(zero_days)
        
        assert torch.allclose(weights, torch.tensor(1.0), atol=1e-6)
    
    def test_batch_processing(self, feature_names):
        """Test batch processing capabilities."""
        temporal_decay = LearnableTemporalDecay(feature_names)
        
        batch_size = 32
        days_ago = torch.randn(batch_size).abs()
        feature_values = torch.randn(batch_size, len(feature_names))
        
        # Should handle large batches efficiently
        weighted_features = temporal_decay.compute_feature_weights(
            days_ago, feature_values, feature_names
        )
        
        assert weighted_features.shape == (batch_size, len(feature_names))
        assert torch.all(torch.isfinite(weighted_features))


class TestFactoryFunction:
    """Test factory function for creating temporal decay."""
    
    def test_create_with_defaults(self):
        """Test factory function with default configuration."""
        feature_names = ['feature1', 'feature2']
        temporal_decay = create_learnable_temporal_decay(feature_names)
        
        assert isinstance(temporal_decay, LearnableTemporalDecay)
        assert temporal_decay.feature_names == feature_names
        assert temporal_decay.learnable
    
    def test_create_with_config(self):
        """Test factory function with custom configuration."""
        feature_names = ['feature1', 'feature2']
        config = {
            'initial_half_life': 45.0,
            'min_half_life': 5.0,
            'max_half_life': 180.0,
            'learnable': False,
            'temperature': 2.0
        }
        
        temporal_decay = create_learnable_temporal_decay(feature_names, config)
        
        assert temporal_decay.min_half_life == 5.0
        assert temporal_decay.max_half_life == 180.0
        assert not temporal_decay.learnable
        assert temporal_decay.temperature == 2.0
        
        # Check initial half-life
        half_lives = temporal_decay.get_half_lives()
        assert torch.allclose(half_lives, torch.tensor(45.0), atol=1.0)