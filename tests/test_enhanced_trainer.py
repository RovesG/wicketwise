# Purpose: Test suite for EnhancedTrainer with drift detection and confidence monitoring
# Author: Shamus Rae, Last Modified: 2024

import pytest
import torch
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

import sys
sys.path.append('..')

from crickformers.enhanced_trainer import EnhancedTrainer
from crickformers.drift_detector import DriftDetector
from crickformers.confidence_utils import predict_with_uncertainty


class TestEnhancedTrainer:
    """Test suite for the enhanced trainer with monitoring capabilities."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "batch_size": 8,
            "num_epochs": 2,
            "learning_rate": 0.001,
            "log_interval": 10,
            "validation_interval": 20,
            "drift_threshold": 0.1,
            "drift_window_size": 100,
            "weight_decay": 0.01,
            "model": {
                "numeric_dim": 10,
                "categorical_vocab_sizes": {"player": 50, "team": 10},
                "categorical_embedding_dims": {"player": 16, "team": 8},
                "video_dim": 128,
                "sequence_length": 5,
                "hidden_dim": 64,
                "num_layers": 2,
                "num_heads": 4,
                "context_dim": 32,
                "dropout_rate": 0.2,
                "gnn_config": {},
                "enable_temporal_decay": True,
                "temporal_decay_factor": 0.1
            }
        }
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_dataset(self):
        """Mock dataset for testing."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.get_match_ids = Mock(return_value=['match_1', 'match_2'])
        return mock_dataset
    
    @pytest.fixture
    def mock_dataloader(self):
        """Mock dataloader for testing."""
        sample_batch = {
            "inputs": {
                "numeric_features": torch.randn(8, 10),
                "categorical_features": torch.randint(0, 50, (8, 2)),
                "video_features": torch.randn(8, 128),
                "video_mask": torch.ones(8, 1),
                "sequence_features": torch.randn(8, 5, 64),
                "sequence_mask": torch.ones(8, 5)
            },
            "targets": {
                "win_prob": torch.rand(8, 1),
                "next_ball_outcome": torch.randint(0, 10, (8,)),
                "mispricing": torch.randn(8, 1)
            }
        }
        
        mock_loader = Mock()
        mock_loader.__len__ = Mock(return_value=10)
        mock_loader.__iter__ = Mock(return_value=iter([sample_batch] * 10))
        return mock_loader
    
    def test_enhanced_trainer_initialization(self, sample_config):
        """Test EnhancedTrainer initialization."""
        trainer = EnhancedTrainer(sample_config, device="cpu")
        
        assert trainer.config == sample_config
        assert trainer.device == torch.device("cpu")
        assert trainer.batch_size == 8
        assert trainer.num_epochs == 2
        assert trainer.learning_rate == 0.001
        assert trainer.log_interval == 10
        assert trainer.validation_interval == 20
        
        # Check monitoring components
        assert isinstance(trainer.drift_detector, DriftDetector)
        assert trainer.drift_detector.threshold == 0.1
        assert trainer.drift_detector.window_size == 100
        
        # Check tracking variables
        assert trainer.step_count == 0
        assert trainer.running_loss == 0.0
        assert len(trainer.confidence_scores) == 0
        assert len(trainer.drift_alerts) == 0
        assert len(trainer.loss_history) == 0
    
    def test_model_setup(self, sample_config):
        """Test model setup with configuration."""
        trainer = EnhancedTrainer(sample_config, device="cpu")
        trainer.setup_model()
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        
        # Check model parameters
        total_params = sum(p.numel() for p in trainer.model.parameters())
        assert total_params > 0
        
    @patch('crickformers.enhanced_trainer.CrickformerDataset')
    @patch('crickformers.enhanced_trainer.DataLoader')
    def test_dataset_setup_with_csv(self, mock_dataloader, mock_dataset_class, sample_config, mock_dataset):
        """Test dataset setup with CSV adapter."""
        mock_dataset_class.return_value = mock_dataset
        mock_dataloader.return_value = Mock()
        
        trainer = EnhancedTrainer(sample_config, device="cpu")
        trainer.setup_dataset(
            data_path="/test/data",
            use_csv=True,
            train_matches_path="/test/train_matches.txt",
            val_matches_path="/test/val_matches.txt"
        )
        
        # Check dataset creation calls
        assert mock_dataset_class.call_count == 2  # Train and validation
        assert trainer.train_loader is not None
        assert trainer.val_loader is not None
    
    def test_compute_loss(self, sample_config):
        """Test loss computation."""
        trainer = EnhancedTrainer(sample_config, device="cpu")
        trainer.setup_model()
        
        # Mock outputs and targets
        batch_size = 4
        outputs = {
            "win_prob": torch.rand(batch_size, 1),
            "next_ball_outcome": torch.randn(batch_size, 10),  # Logits
            "mispricing": torch.randn(batch_size, 1)
        }
        targets = {
            "win_prob": torch.rand(batch_size, 1),
            "next_ball_outcome": torch.randint(0, 10, (batch_size,)),
            "mispricing": torch.randn(batch_size, 1)
        }
        
        total_loss, loss_dict = trainer.compute_loss(outputs, targets)
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert "total" in loss_dict
        assert "win_prob" in loss_dict
        assert "outcome" in loss_dict
        assert "mispricing" in loss_dict
        assert all(isinstance(v, float) for v in loss_dict.values())
    
    def test_train_step(self, sample_config):
        """Test single training step."""
        trainer = EnhancedTrainer(sample_config, device="cpu")
        trainer.setup_model()
        
        # Mock batch
        batch = {
            "inputs": {
                "numeric_features": torch.randn(4, 10),
                "categorical_features": torch.randint(0, 50, (4, 2)),
                "video_features": torch.randn(4, 128),
                "video_mask": torch.ones(4, 1),
                "sequence_features": torch.randn(4, 5, 64),
                "sequence_mask": torch.ones(4, 5)
            },
            "targets": {
                "win_prob": torch.rand(4, 1),
                "next_ball_outcome": torch.randint(0, 10, (4,)),
                "mispricing": torch.randn(4, 1)
            }
        }
        
        initial_step_count = trainer.step_count
        loss_dict = trainer.train_step(batch)
        
        # Check that step count increased
        assert trainer.step_count == initial_step_count + 1
        
        # Check loss dictionary
        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict
        assert "win_prob" in loss_dict
        assert "outcome" in loss_dict
        
        # Check that loss history was updated
        assert len(trainer.loss_history) == 1
        assert trainer.loss_history[-1] == loss_dict["total"]
    
    def test_validate_with_confidence(self, sample_config, mock_dataloader):
        """Test validation with confidence estimation."""
        trainer = EnhancedTrainer(sample_config, device="cpu")
        trainer.setup_model()
        trainer.val_loader = mock_dataloader
        
        # Mock predict_with_uncertainty
        with patch('crickformers.enhanced_trainer.predict_with_uncertainty') as mock_predict:
            mock_predict.return_value = (
                {"win_prob": torch.rand(8, 1)},  # mean_pred
                {"win_prob": torch.rand(8, 1) * 0.1},  # std_pred
                {"win_prob": (torch.rand(8, 1), torch.rand(8, 1))}  # conf_interval
            )
            
            with patch('crickformers.enhanced_trainer.calculate_confidence_score') as mock_conf:
                mock_conf.return_value = 0.8
                
                metrics = trainer.validate_with_confidence()
        
        # Check returned metrics
        assert isinstance(metrics, dict)
        assert "val_loss" in metrics
        assert "avg_confidence" in metrics
        assert "min_confidence" in metrics
        assert "max_confidence" in metrics
        assert "std_confidence" in metrics
        
        # Check that confidence scores were stored
        assert len(trainer.confidence_scores) > 0
        assert len(trainer.confidence_history) > 0
        assert len(trainer.validation_metrics) > 0
    
    def test_drift_detection_integration(self, sample_config):
        """Test drift detection integration in training step."""
        trainer = EnhancedTrainer(sample_config, device="cpu")
        trainer.setup_model()
        
        # Mock model with intermediate representations
        trainer.model.get_intermediate_representations = Mock(return_value=torch.randn(4, 64))
        
        # Mock drift detector to return drift detected
        trainer.drift_detector.detect_drift = Mock(return_value=True)
        trainer.drift_detector.get_last_drift_score = Mock(return_value=0.15)
        
        batch = {
            "inputs": {
                "numeric_features": torch.randn(4, 10),
                "categorical_features": torch.randint(0, 50, (4, 2)),
                "video_features": torch.randn(4, 128),
                "video_mask": torch.ones(4, 1),
                "sequence_features": torch.randn(4, 5, 64),
                "sequence_mask": torch.ones(4, 5)
            },
            "targets": {
                "win_prob": torch.rand(4, 1),
                "next_ball_outcome": torch.randint(0, 10, (4,)),
                "mispricing": torch.randn(4, 1)
            }
        }
        
        trainer.train_step(batch)
        
        # Check that drift alert was recorded
        assert len(trainer.drift_alerts) == 1
        assert trainer.drift_alerts[0]['step'] == 1
        assert trainer.drift_alerts[0]['drift_score'] == 0.15
        
        # Check drift detector was called
        trainer.drift_detector.detect_drift.assert_called_once()
    
    def test_create_monitoring_plots(self, sample_config, temp_dir):
        """Test monitoring plot creation."""
        trainer = EnhancedTrainer(sample_config, device="cpu")
        
        # Add some mock data
        trainer.loss_history = [0.5, 0.4, 0.3, 0.2]
        trainer.confidence_scores.extend([0.8, 0.7, 0.9, 0.6])
        trainer.confidence_history = [0.75, 0.80, 0.85]
        trainer.validation_metrics = [
            {"val_loss": 0.4, "avg_confidence": 0.75},
            {"val_loss": 0.3, "avg_confidence": 0.80}
        ]
        trainer.epoch_metrics = defaultdict(list)
        trainer.epoch_metrics['loss'] = [0.4, 0.3]
        trainer.epoch_metrics['time'] = [100, 95]
        trainer.drift_alerts = [
            {"step": 50, "drift_score": 0.12},
            {"step": 150, "drift_score": 0.15}
        ]
        
        # Create plots
        plots_dir = Path(temp_dir) / "plots"
        trainer.create_monitoring_plots(str(plots_dir))
        
        # Check that plot files were created
        assert (plots_dir / "loss_curves.png").exists()
        assert (plots_dir / "confidence_analysis.png").exists()
        assert (plots_dir / "drift_alerts.png").exists()
        assert (plots_dir / "training_summary.png").exists()
    
    def test_generate_training_report(self, sample_config, temp_dir):
        """Test training report generation."""
        trainer = EnhancedTrainer(sample_config, device="cpu")
        
        # Add some mock data
        trainer.step_count = 100
        trainer.loss_history = [0.5, 0.4, 0.3, 0.2]
        trainer.confidence_scores.extend([0.8, 0.7, 0.9, 0.6])
        trainer.confidence_history = [0.75, 0.80, 0.85]
        trainer.validation_metrics = [
            {"val_loss": 0.4, "avg_confidence": 0.75},
            {"val_loss": 0.3, "avg_confidence": 0.80}
        ]
        trainer.epoch_metrics = defaultdict(list)
        trainer.epoch_metrics['loss'] = [0.4, 0.3]
        trainer.epoch_metrics['time'] = [100, 95]
        trainer.drift_alerts = [
            {"step": 50, "drift_score": 0.12}
        ]
        
        report_path = Path(temp_dir) / "report.json"
        trainer.generate_training_report(str(report_path))
        
        # Check that report was created
        assert report_path.exists()
        
        # Load and validate report content
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        assert "training_config" in report
        assert "training_summary" in report
        assert "drift_detection" in report
        assert "confidence_analysis" in report
        assert "performance_metrics" in report
        
        # Check specific content
        assert report["training_summary"]["total_steps"] == 100
        assert report["drift_detection"]["total_drift_alerts"] == 1
        assert "average_confidence" in report["confidence_analysis"]
        assert len(report["performance_metrics"]["loss_history"]) > 0
    
    def test_save_model_with_monitoring_data(self, sample_config, temp_dir):
        """Test model saving with monitoring data."""
        trainer = EnhancedTrainer(sample_config, device="cpu")
        trainer.setup_model()
        
        # Add some mock monitoring data
        trainer.step_count = 50
        trainer.loss_history = [0.5, 0.4, 0.3]
        trainer.confidence_scores.extend([0.8, 0.7, 0.9])
        trainer.drift_alerts = [{"step": 25, "drift_score": 0.12}]
        trainer.validation_metrics = [{"val_loss": 0.4, "avg_confidence": 0.75}]
        trainer.epoch_metrics = defaultdict(list)
        trainer.epoch_metrics['loss'] = [0.4, 0.3]
        
        # Mock drift detector state
        trainer.drift_detector.get_state = Mock(return_value={"threshold": 0.1, "window_size": 100})
        
        model_path = Path(temp_dir) / "model.pth"
        trainer.save_model(str(model_path))
        
        # Check that model was saved
        assert model_path.exists()
        
        # Load and validate saved data
        checkpoint = torch.load(model_path, map_location="cpu")
        
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "config" in checkpoint
        assert "step_count" in checkpoint
        assert "drift_detector_state" in checkpoint
        assert "training_metrics" in checkpoint
        
        # Check monitoring data
        training_metrics = checkpoint["training_metrics"]
        assert "loss_history" in training_metrics
        assert "epoch_metrics" in training_metrics
        assert "validation_metrics" in training_metrics
        assert "confidence_scores" in training_metrics
        assert "drift_alerts" in training_metrics
        
        assert checkpoint["step_count"] == 50
        assert len(training_metrics["loss_history"]) == 3
        assert len(training_metrics["drift_alerts"]) == 1
    
    def test_train_epoch_with_monitoring(self, sample_config, mock_dataloader):
        """Test training epoch with monitoring features."""
        trainer = EnhancedTrainer(sample_config, device="cpu")
        trainer.setup_model()
        trainer.train_loader = mock_dataloader
        trainer.val_loader = mock_dataloader
        
        # Mock validation function
        trainer.validate_with_confidence = Mock(return_value={
            "val_loss": 0.4,
            "avg_confidence": 0.75,
            "min_confidence": 0.6,
            "max_confidence": 0.9,
            "std_confidence": 0.1
        })
        
        # Run one epoch
        trainer.train_epoch(0)
        
        # Check that metrics were recorded
        assert len(trainer.epoch_metrics['loss']) > 0
        assert len(trainer.epoch_metrics['time']) > 0
        assert trainer.step_count > 0
        assert len(trainer.loss_history) > 0
    
    def test_full_training_workflow(self, sample_config, mock_dataloader):
        """Test complete training workflow with monitoring."""
        trainer = EnhancedTrainer(sample_config, device="cpu")
        trainer.setup_model()
        trainer.train_loader = mock_dataloader
        trainer.val_loader = mock_dataloader
        
        # Mock methods to avoid actual file operations
        trainer.create_monitoring_plots = Mock()
        trainer.generate_training_report = Mock()
        trainer.validate_with_confidence = Mock(return_value={
            "val_loss": 0.4,
            "avg_confidence": 0.75,
            "min_confidence": 0.6,
            "max_confidence": 0.9,
            "std_confidence": 0.1
        })
        
        # Run training
        trainer.train()
        
        # Check that training completed
        assert trainer.step_count > 0
        assert len(trainer.loss_history) > 0
        assert len(trainer.epoch_metrics['loss']) == trainer.num_epochs
        assert len(trainer.epoch_metrics['time']) == trainer.num_epochs
        
        # Check that monitoring functions were called
        trainer.create_monitoring_plots.assert_called_once()
        trainer.generate_training_report.assert_called_once()
    
    def test_error_handling(self, sample_config):
        """Test error handling in various scenarios."""
        trainer = EnhancedTrainer(sample_config, device="cpu")
        
        # Test with empty confidence scores
        trainer.confidence_scores = []
        
        # Should not crash when creating plots with empty data
        with patch('matplotlib.pyplot.savefig'):
            trainer.create_monitoring_plots()
        
        # Test report generation with minimal data
        report_path = "test_report.json"
        trainer.generate_training_report(report_path)
        
        # Clean up
        if Path(report_path).exists():
            Path(report_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 