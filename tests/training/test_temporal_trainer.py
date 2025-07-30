# Purpose: Unit tests for temporal decay trainer
# Author: Shamus Rae, Last Modified: 2024-01-15

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

from crickformers.model.learnable_temporal_decay import LearnableTemporalDecay
from crickformers.training.temporal_trainer import (
    TemporalDecayTrainer,
    create_temporal_trainer
)


class MockMainModel(nn.Module):
    """Mock main model for testing."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


class TestTemporalDecayTrainer:
    """Test temporal decay trainer."""
    
    @pytest.fixture
    def feature_names(self):
        """Sample feature names."""
        return ['batting_avg', 'strike_rate', 'recent_form']
    
    @pytest.fixture
    def temporal_decay(self, feature_names):
        """Create temporal decay module."""
        return LearnableTemporalDecay(
            feature_names=feature_names,
            initial_half_life=30.0,
            learnable=True
        )
    
    @pytest.fixture
    def main_model(self, feature_names):
        """Create mock main model."""
        return MockMainModel(len(feature_names), 1)
    
    @pytest.fixture
    def trainer_separate_optimizers(self, temporal_decay, main_model):
        """Create trainer with separate optimizers."""
        return TemporalDecayTrainer(
            temporal_decay=temporal_decay,
            main_model=main_model,
            temporal_lr=0.001,
            main_lr=0.0001,
            use_separate_optimizers=True
        )
    
    @pytest.fixture
    def trainer_single_optimizer(self, temporal_decay, main_model):
        """Create trainer with single optimizer."""
        return TemporalDecayTrainer(
            temporal_decay=temporal_decay,
            main_model=main_model,
            temporal_lr=0.001,
            main_lr=0.0001,
            use_separate_optimizers=False
        )
    
    @pytest.fixture
    def sample_batch(self, feature_names):
        """Create sample training batch."""
        batch_size = 8
        return {
            'days_ago': torch.randn(batch_size).abs(),
            'feature_values': torch.randn(batch_size, len(feature_names)),
            'targets': torch.randn(batch_size, 1)
        }
    
    def test_initialization_separate_optimizers(self, trainer_separate_optimizers, temporal_decay, main_model):
        """Test trainer initialization with separate optimizers."""
        trainer = trainer_separate_optimizers
        
        assert trainer.temporal_decay == temporal_decay
        assert trainer.main_model == main_model
        assert trainer.use_separate_optimizers
        assert trainer.temporal_lr == 0.001
        assert trainer.main_lr == 0.0001
        
        # Check optimizers
        assert hasattr(trainer, 'temporal_optimizer')
        assert hasattr(trainer, 'main_optimizer')
        assert trainer.temporal_optimizer is not None
        assert trainer.main_optimizer is not None
        
        # Check schedulers
        assert hasattr(trainer, 'temporal_scheduler')
        assert hasattr(trainer, 'main_scheduler')
    
    def test_initialization_single_optimizer(self, trainer_single_optimizer):
        """Test trainer initialization with single optimizer."""
        trainer = trainer_single_optimizer
        
        assert not trainer.use_separate_optimizers
        assert hasattr(trainer, 'optimizer')
        assert trainer.optimizer is not None
        assert trainer.temporal_optimizer is None
        assert trainer.main_optimizer is None
        
        # Check parameter groups
        param_groups = trainer.optimizer.param_groups
        assert len(param_groups) == 2
        assert param_groups[0]['lr'] == 0.001  # Temporal parameters
        assert param_groups[1]['lr'] == 0.0001  # Main parameters
    
    def test_train_step_separate_optimizers(self, trainer_separate_optimizers, sample_batch):
        """Test training step with separate optimizers."""
        trainer = trainer_separate_optimizers
        
        losses = trainer.train_step(sample_batch)
        
        assert isinstance(losses, dict)
        assert 'main_loss' in losses
        assert 'total_loss' in losses
        assert 'temporal_grad_norm' in losses
        assert 'main_grad_norm' in losses
        
        # All losses should be non-negative
        for key, value in losses.items():
            if 'loss' in key:
                assert value >= 0
        
        # Gradient norms should be positive
        assert losses['temporal_grad_norm'] >= 0
        assert losses['main_grad_norm'] >= 0
    
    def test_train_step_single_optimizer(self, trainer_single_optimizer, sample_batch):
        """Test training step with single optimizer."""
        trainer = trainer_single_optimizer
        
        losses = trainer.train_step(sample_batch)
        
        assert isinstance(losses, dict)
        assert 'main_loss' in losses
        assert 'total_loss' in losses
        assert 'total_grad_norm' in losses
        
        # Should not have separate gradient norms
        assert 'temporal_grad_norm' not in losses
        assert 'main_grad_norm' not in losses
    
    def test_train_step_with_temporal_loss(self, trainer_separate_optimizers, sample_batch):
        """Test training step with temporal loss computation."""
        trainer = trainer_separate_optimizers
        
        losses = trainer.train_step(
            sample_batch,
            compute_temporal_loss=True,
            temporal_loss_weight=0.1
        )
        
        # Should have temporal loss components
        temporal_keys = [k for k in losses.keys() if k.startswith('temporal_')]
        assert len(temporal_keys) > 0
        
        # Check specific temporal loss components
        assert 'temporal_total_loss' in losses
    
    def test_train_step_without_temporal_loss(self, trainer_separate_optimizers, sample_batch):
        """Test training step without temporal loss computation."""
        trainer = trainer_separate_optimizers
        
        losses = trainer.train_step(
            sample_batch,
            compute_temporal_loss=False
        )
        
        # Should not have temporal loss components
        temporal_keys = [k for k in losses.keys() if k.startswith('temporal_') and 'grad' not in k]
        assert len(temporal_keys) == 0
    
    def test_validate_step(self, trainer_separate_optimizers, sample_batch):
        """Test validation step."""
        trainer = trainer_separate_optimizers
        
        val_losses = trainer.validate_step(sample_batch)
        
        assert isinstance(val_losses, dict)
        assert 'val_main_loss' in val_losses
        assert 'val_total_loss' in val_losses
        
        # Validation losses should be non-negative
        for key, value in val_losses.items():
            assert value >= 0
    
    def test_gradient_flow(self, trainer_separate_optimizers, sample_batch):
        """Test gradient flow through temporal parameters."""
        trainer = trainer_separate_optimizers
        
        # Store initial parameters
        initial_temporal_params = [p.clone() for p in trainer.temporal_decay.parameters()]
        initial_main_params = [p.clone() for p in trainer.main_model.parameters()]
        
        # Perform training step
        losses = trainer.train_step(sample_batch)
        
        # Check that gradients were computed
        assert losses['temporal_grad_norm'] > 0
        assert losses['main_grad_norm'] > 0
        
        # Check that parameters were updated
        for initial, current in zip(initial_temporal_params, trainer.temporal_decay.parameters()):
            assert not torch.allclose(initial, current, atol=1e-6)
        
        for initial, current in zip(initial_main_params, trainer.main_model.parameters()):
            assert not torch.allclose(initial, current, atol=1e-6)
    
    def test_learning_rate_schedulers(self, trainer_separate_optimizers):
        """Test learning rate scheduler functionality."""
        trainer = trainer_separate_optimizers
        
        # Store initial learning rates
        initial_temporal_lr = trainer.temporal_optimizer.param_groups[0]['lr']
        initial_main_lr = trainer.main_optimizer.param_groups[0]['lr']
        
        # Step schedulers (plateau scheduler needs validation loss)
        validation_loss = 0.5
        trainer.step_schedulers(validation_loss)
        
        # Learning rates should be accessible
        lrs = trainer.get_learning_rates()
        assert 'temporal_lr' in lrs
        assert 'main_lr' in lrs
        assert lrs['temporal_lr'] > 0
        assert lrs['main_lr'] > 0
    
    def test_half_life_tracking(self, trainer_separate_optimizers, sample_batch, feature_names):
        """Test half-life parameter tracking."""
        trainer = trainer_separate_optimizers
        
        # Initial half-lives
        initial_half_lives = trainer.get_half_lives()
        assert len(initial_half_lives) == len(feature_names)
        
        for feature_name in feature_names:
            assert feature_name in initial_half_lives
            assert initial_half_lives[feature_name] > 0
        
        # Perform training steps
        for _ in range(5):
            trainer.train_step(sample_batch)
        
        # Half-lives should be tracked in history
        assert len(trainer.half_life_history) > 0
        for feature_name in feature_names:
            assert feature_name in trainer.half_life_history
            assert len(trainer.half_life_history[feature_name]) == 5
    
    def test_training_statistics(self, trainer_separate_optimizers, sample_batch):
        """Test training statistics collection."""
        trainer = trainer_separate_optimizers
        
        # Perform several training steps
        for _ in range(10):
            trainer.train_step(sample_batch)
        
        stats = trainer.get_training_statistics()
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
        
        # Check for expected statistics
        expected_keys = ['main_loss_mean', 'total_loss_mean', 'temporal_lr', 'main_lr']
        for key in expected_keys:
            assert key in stats
        
        # Half-life statistics
        for feature_name in trainer.temporal_decay.feature_names:
            assert f'{feature_name}_half_life_current' in stats
    
    def test_gradient_analysis(self, trainer_separate_optimizers, sample_batch):
        """Test gradient flow analysis."""
        trainer = trainer_separate_optimizers
        
        # Perform training step to generate gradients
        trainer.train_step(sample_batch)
        
        analysis = trainer.analyze_gradient_flow()
        
        assert isinstance(analysis, dict)
        assert 'temporal_params_with_grad' in analysis
        assert 'temporal_params_without_grad' in analysis
        assert 'temporal_gradient_flow_health' in analysis
        
        # Should have good gradient flow
        assert analysis['temporal_gradient_flow_health'] > 0.5
        assert len(analysis['temporal_params_with_grad']) > 0
    
    def test_save_and_load_checkpoint(self, trainer_separate_optimizers, sample_batch):
        """Test checkpoint saving and loading."""
        trainer = trainer_separate_optimizers
        
        # Perform some training
        for _ in range(5):
            trainer.train_step(sample_batch)
        
        # Store current state
        original_temporal_state = trainer.temporal_decay.state_dict()
        original_main_state = trainer.main_model.state_dict()
        original_stats = dict(trainer.training_stats)
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            trainer.save_checkpoint(checkpoint_path, epoch=10)
            
            # Modify state
            with torch.no_grad():
                for param in trainer.temporal_decay.parameters():
                    param.fill_(0.5)
                for param in trainer.main_model.parameters():
                    param.fill_(0.3)
            
            trainer.training_stats.clear()
            
            # Load checkpoint
            loaded_info = trainer.load_checkpoint(checkpoint_path)
            
            assert loaded_info['epoch'] == 10
            
            # Check state restoration
            loaded_temporal_state = trainer.temporal_decay.state_dict()
            loaded_main_state = trainer.main_model.state_dict()
            
            for key in original_temporal_state:
                assert torch.allclose(original_temporal_state[key], loaded_temporal_state[key])
            
            for key in original_main_state:
                assert torch.allclose(original_main_state[key], loaded_main_state[key])
            
            # Check statistics restoration
            assert len(trainer.training_stats) > 0
            
        finally:
            os.unlink(checkpoint_path)
    
    def test_missing_batch_data(self, trainer_separate_optimizers):
        """Test handling of missing batch data."""
        incomplete_batch = {'days_ago': torch.tensor([1.0, 2.0])}
        
        with pytest.raises(ValueError, match="Missing required batch data"):
            trainer_separate_optimizers.train_step(incomplete_batch)
    
    def test_optimizer_parameter_groups(self, trainer_single_optimizer):
        """Test optimizer parameter group configuration."""
        trainer = trainer_single_optimizer
        
        param_groups = trainer.optimizer.param_groups
        assert len(param_groups) == 2
        
        # Check that temporal and main parameters are in different groups
        temporal_param_count = len(list(trainer.temporal_decay.parameters()))
        main_param_count = len(list(trainer.main_model.parameters()))
        
        assert len(param_groups[0]['params']) == temporal_param_count
        assert len(param_groups[1]['params']) == main_param_count
        assert param_groups[0]['lr'] == trainer.temporal_lr
        assert param_groups[1]['lr'] == trainer.main_lr
    
    def test_gradient_clipping(self, trainer_separate_optimizers, sample_batch):
        """Test gradient clipping functionality."""
        trainer = trainer_separate_optimizers
        
        # Set very low clipping thresholds
        trainer.temporal_max_grad_norm = 0.01
        trainer.max_grad_norm = 0.01
        
        losses = trainer.train_step(sample_batch)
        
        # Gradients should be clipped
        assert losses['temporal_grad_norm'] <= trainer.temporal_max_grad_norm + 1e-6
        assert losses['main_grad_norm'] <= trainer.max_grad_norm + 1e-6
    
    def test_scheduler_types(self, temporal_decay, main_model):
        """Test different scheduler types."""
        # Test cosine scheduler
        trainer_cosine = TemporalDecayTrainer(
            temporal_decay=temporal_decay,
            main_model=main_model,
            scheduler_type="cosine",
            scheduler_config={'T_max': 50}
        )
        
        assert trainer_cosine.temporal_scheduler is not None
        assert trainer_cosine.main_scheduler is not None
        
        # Test no scheduler
        trainer_none = TemporalDecayTrainer(
            temporal_decay=temporal_decay,
            main_model=main_model,
            scheduler_type="none"
        )
        
        assert trainer_none.temporal_scheduler is None
        assert trainer_none.main_scheduler is None
    
    def test_training_mode_effects(self, trainer_separate_optimizers, sample_batch):
        """Test effects of training vs evaluation mode."""
        trainer = trainer_separate_optimizers
        
        # Training mode
        trainer.temporal_decay.train()
        trainer.main_model.train()
        
        train_losses = trainer.train_step(sample_batch)
        
        # Evaluation mode
        trainer.temporal_decay.eval()
        trainer.main_model.eval()
        
        val_losses = trainer.validate_step(sample_batch)
        
        # Should have different behavior (validation should not update statistics)
        assert 'val_main_loss' in val_losses
        assert 'main_loss' in train_losses


class TestFactoryFunction:
    """Test factory function for temporal trainer."""
    
    @pytest.fixture
    def feature_names(self):
        return ['feature1', 'feature2']
    
    @pytest.fixture
    def temporal_decay(self, feature_names):
        return LearnableTemporalDecay(feature_names)
    
    @pytest.fixture
    def main_model(self, feature_names):
        return MockMainModel(len(feature_names), 1)
    
    def test_create_with_defaults(self, temporal_decay, main_model):
        """Test factory function with default configuration."""
        trainer = create_temporal_trainer(temporal_decay, main_model)
        
        assert isinstance(trainer, TemporalDecayTrainer)
        assert trainer.temporal_decay == temporal_decay
        assert trainer.main_model == main_model
        assert trainer.use_separate_optimizers
    
    def test_create_with_config(self, temporal_decay, main_model):
        """Test factory function with custom configuration."""
        config = {
            'temporal_lr': 0.005,
            'main_lr': 0.0005,
            'use_separate_optimizers': False,
            'scheduler_type': 'cosine',
            'scheduler_config': {'T_max': 100}
        }
        
        trainer = create_temporal_trainer(temporal_decay, main_model, config)
        
        assert trainer.temporal_lr == 0.005
        assert trainer.main_lr == 0.0005
        assert not trainer.use_separate_optimizers
        assert trainer.scheduler is not None


class TestTemporalTrainerEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def feature_names(self):
        return ['feature1']
    
    @pytest.fixture
    def temporal_decay(self, feature_names):
        return LearnableTemporalDecay(feature_names)
    
    @pytest.fixture
    def main_model(self, feature_names):
        return MockMainModel(len(feature_names), 1)
    
    def test_zero_learning_rates(self, temporal_decay, main_model):
        """Test handling of zero learning rates."""
        trainer = TemporalDecayTrainer(
            temporal_decay=temporal_decay,
            main_model=main_model,
            temporal_lr=0.0,
            main_lr=0.0
        )
        
        # Should not crash
        assert trainer.temporal_lr == 0.0
        assert trainer.main_lr == 0.0
    
    def test_extreme_gradient_norms(self, temporal_decay, main_model):
        """Test handling of extreme gradient norms."""
        trainer = TemporalDecayTrainer(
            temporal_decay=temporal_decay,
            main_model=main_model,
            temporal_lr=1000.0,  # Very high learning rate
            main_lr=1000.0
        )
        
        batch_data = {
            'days_ago': torch.tensor([1.0]),
            'feature_values': torch.tensor([[1000.0]]),  # Extreme values
            'targets': torch.tensor([[0.0]])
        }
        
        # Should handle extreme gradients with clipping
        losses = trainer.train_step(batch_data)
        
        assert 'temporal_grad_norm' in losses
        assert 'main_grad_norm' in losses
        assert np.isfinite(losses['temporal_grad_norm'])
        assert np.isfinite(losses['main_grad_norm'])
    
    def test_empty_batch(self, temporal_decay, main_model):
        """Test handling of empty batch."""
        trainer = TemporalDecayTrainer(temporal_decay, main_model)
        
        empty_batch = {
            'days_ago': torch.tensor([]),
            'feature_values': torch.tensor([]).reshape(0, 1),
            'targets': torch.tensor([]).reshape(0, 1)
        }
        
        # Should handle empty batch gracefully
        try:
            losses = trainer.train_step(empty_batch)
            # If it doesn't crash, losses should be reasonable
            assert isinstance(losses, dict)
        except RuntimeError:
            # It's acceptable to raise an error for empty batches
            pass
    
    def test_nan_in_data(self, temporal_decay, main_model):
        """Test handling of NaN values in data."""
        trainer = TemporalDecayTrainer(temporal_decay, main_model)
        
        batch_with_nan = {
            'days_ago': torch.tensor([1.0, float('nan'), 3.0]),
            'feature_values': torch.tensor([[1.0], [2.0], [float('nan')]]),
            'targets': torch.tensor([[1.0], [2.0], [3.0]])
        }
        
        # Should detect NaN values
        try:
            losses = trainer.train_step(batch_with_nan)
            # If it completes, check for NaN in losses
            for key, value in losses.items():
                if np.isnan(value):
                    pytest.fail(f"NaN detected in loss {key}")
        except RuntimeError:
            # It's acceptable to raise an error for NaN inputs
            pass