# Purpose: Unit tests for the Crickformer training pipeline
# Author: Assistant, Last Modified: 2024

import pytest
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent))

from crickformers.train import CrickformerTrainer, collate_fn
from crickformers.crickformer_dataset import CrickformerDataset
from crickformers.csv_data_adapter import CSVDataAdapter, CSVDataConfig


class TestCollateFunction:
    """Test the custom collate function."""
    
    def test_collate_fn_basic(self):
        """Test basic collate function behavior."""
        # Create mock samples
        batch_size = 4
        batch = []
        
        for i in range(batch_size):
            sample = {
                'numeric_ball_features': torch.randn(15),
                'categorical_ball_features': torch.randint(0, 5, (4,)),
                'ball_history': torch.randn(5, 6),
                'video_features': torch.randn(99),
                'video_mask': torch.ones(1),
                'gnn_embeddings': torch.randn(320),
                'market_odds': torch.randn(7),
                'market_odds_mask': torch.ones(1)
            }
            batch.append(sample)
        
        # Test collate function
        result = collate_fn(batch)
        
        # Check structure
        assert 'inputs' in result
        assert 'targets' in result
        assert 'masks' in result
        
        # Check input shapes
        inputs = result['inputs']
        assert inputs['recent_ball_history'].shape == (batch_size, 5, 6)
        assert inputs['numeric_features'].shape == (batch_size, 15)
        assert inputs['categorical_features'].shape == (batch_size, 4)
        assert inputs['gnn_embeddings'].shape == (batch_size, 1, 320)
        assert inputs['video_features'].shape == (batch_size, 99)
        assert inputs['video_mask'].shape == (batch_size, 1)
        
        # Check target shapes
        targets = result['targets']
        assert targets['win_prob'].shape == (batch_size, 1)
        assert targets['outcome'].shape == (batch_size,)
        assert targets['mispricing'].shape == (batch_size, 1)
        
        # Check masks
        masks = result['masks']
        assert masks['video_mask'].shape == (batch_size, 1)
        assert masks['market_odds_mask'].shape == (batch_size, 1)


class TestCrickformerTrainer:
    """Test the main trainer class."""
    
    @pytest.fixture
    def trainer_config(self):
        """Basic trainer configuration."""
        return {
            "batch_size": 8,
            "num_epochs": 1,
            "learning_rate": 1e-3,
            "log_interval": 5,
            "loss_weights": {
                "win_prob": 1.0,
                "outcome": 1.0,
                "mispricing": 0.5
            }
        }
    
    @pytest.fixture
    def trainer(self, trainer_config):
        """Create trainer instance."""
        return CrickformerTrainer(trainer_config)
    
    def test_trainer_init(self, trainer):
        """Test trainer initialization."""
        assert trainer.batch_size == 8
        assert trainer.num_epochs == 1
        assert trainer.learning_rate == 1e-3
        assert trainer.log_interval == 5
        assert trainer.device is not None
        assert trainer.step_count == 0
        assert trainer.running_loss == 0.0
    
    def test_setup_model(self, trainer):
        """Test model setup."""
        trainer.setup_model()
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        
        # Check model is on correct device
        assert next(trainer.model.parameters()).device == trainer.device
        
        # Check optimizer
        assert isinstance(trainer.optimizer, torch.optim.Adam)
    
    def test_compute_loss(self, trainer):
        """Test loss computation."""
        trainer.setup_model()
        
        batch_size = 4
        
        # Mock outputs (with gradients)
        outputs = {
            "win_probability": torch.randn(batch_size, 1, requires_grad=True),
            "next_ball_outcome": torch.randn(batch_size, 7, requires_grad=True),
            "odds_mispricing": torch.randn(batch_size, 1, requires_grad=True)
        }
        
        # Mock targets
        targets = {
            "win_prob": torch.sigmoid(torch.randn(batch_size, 1)),
            "outcome": torch.randint(0, 7, (batch_size,)),
            "mispricing": torch.bernoulli(torch.full((batch_size, 1), 0.1))
        }
        
        # Compute loss
        total_loss, loss_dict = trainer.compute_loss(outputs, targets)
        
        # Check loss is computed
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        
        # Check loss components
        assert 'total' in loss_dict
        assert 'win_prob' in loss_dict
        assert 'outcome' in loss_dict
        assert 'mispricing' in loss_dict
        
        # Check all losses are positive
        for loss_name, loss_value in loss_dict.items():
            assert loss_value >= 0.0


class TestTrainingPipelineIntegration:
    """Integration tests with real data."""
    
    @pytest.fixture
    def mock_csv_data(self):
        """Create mock CSV data for testing."""
        # Create temporary CSV files
        temp_dir = tempfile.mkdtemp()
        
        # Mock NVPlay data
        nvplay_data = {
            'Competition': ['Test League'] * 10,
            'Match': ['Test Match'] * 10,
            'Date': ['2024-01-01'] * 10,
            'Innings': [1] * 5 + [2] * 5,
            'Over': [1, 1, 1, 2, 2, 1, 1, 1, 2, 2],
            'Ball': [1, 2, 3, 1, 2, 1, 2, 3, 1, 2],
            'Innings Ball': list(range(1, 11)),
            'Batter': ['Player A'] * 10,
            'Batter ID': ['A001'] * 10,
            'Bowler': ['Player B'] * 10,
            'Bowler ID': ['B001'] * 10,
            'Runs': [0, 1, 4, 0, 2, 1, 0, 6, 0, 1],
            'Extra Runs': [0] * 10,
            'Wicket': ['No Wicket'] * 9 + ['Bowled'],
            'Team Runs': [0, 1, 5, 5, 7, 8, 8, 14, 14, 15],
            'Team Wickets': [0] * 9 + [1],
            'Batting Team': ['Team A'] * 5 + ['Team B'] * 5,
            'Bowling Team': ['Team B'] * 5 + ['Team A'] * 5,
            'Batting Hand': ['RHB'] * 10,
            'Bowler Type': ['RM'] * 10,
            'FieldX': [100.0] * 10,
            'FieldY': [150.0] * 10,
            'PitchX': [0.0] * 10,
            'PitchY': [0.0] * 10,
            'Power Play': [1.0] * 6 + [0.0] * 4,
            'Run Rate After': [0.0, 3.0, 15.0, 10.0, 8.4, 4.8, 4.8, 8.4, 8.4, 9.0],
            'Req Run Rate After': [8.0] * 10,
            'Venue': ['Test Ground'] * 10
        }
        
        # Mock decimal data
        decimal_data = {
            'date': ['2024-01-01'] * 10,
            'competition': ['Test League'] * 10,
            'home': ['Team A'] * 10,
            'away': ['Team B'] * 10,
            'innings': [1] * 5 + [2] * 5,
            'ball': list(range(1, 11)),
            'win_prob': [0.5, 0.52, 0.58, 0.55, 0.60, 0.40, 0.38, 0.45, 0.42, 0.48]
        }
        
        # Save as CSV files
        import pandas as pd
        nvplay_df = pd.DataFrame(nvplay_data)
        decimal_df = pd.DataFrame(decimal_data)
        
        nvplay_path = Path(temp_dir) / 'nvplay_data_v3.csv'
        decimal_path = Path(temp_dir) / 'decimal_data_v3.csv'
        
        nvplay_df.to_csv(nvplay_path, index=False)
        decimal_df.to_csv(decimal_path, index=False)
        
        return temp_dir
    
    def test_dataset_loading_with_mock_data(self, mock_csv_data):
        """Test dataset loading with mock CSV data."""
        dataset = CrickformerDataset(
            data_root=mock_csv_data,
            use_csv_adapter=True,
            csv_config=CSVDataConfig(),
            history_length=5,
            load_video=True,
            load_embeddings=True,
            load_market_odds=True
        )
        
        assert len(dataset) == 10  # 10 balls in mock data
        assert len(dataset.get_match_ids()) == 1  # 1 match
        
        # Test sample structure
        sample = dataset[0]
        expected_keys = [
            'numeric_ball_features', 'categorical_ball_features', 'ball_history',
            'video_features', 'video_mask', 'gnn_embeddings', 'market_odds', 'market_odds_mask'
        ]
        
        for key in expected_keys:
            assert key in sample
    
    def test_training_step_with_real_batch(self, mock_csv_data):
        """Test training step with real batch from dataset."""
        # Setup trainer
        config = {
            "batch_size": 4,
            "num_epochs": 1,
            "learning_rate": 1e-3,
            "log_interval": 1
        }
        trainer = CrickformerTrainer(config)
        trainer.setup_model()
        
        # Create dataset and dataloader
        dataset = CrickformerDataset(
            data_root=mock_csv_data,
            use_csv_adapter=True,
            csv_config=CSVDataConfig(),
            history_length=5
        )
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Get a batch
        batch = next(iter(dataloader))
        
        # Test training step
        initial_step_count = trainer.step_count
        loss_dict = trainer.train_step(batch)
        
        # Verify training step worked
        assert trainer.step_count == initial_step_count + 1
        assert 'total' in loss_dict
        assert 'win_prob' in loss_dict
        assert 'outcome' in loss_dict
        assert 'mispricing' in loss_dict
        
        # Check all losses are reasonable
        for loss_name, loss_value in loss_dict.items():
            assert 0.0 <= loss_value <= 100.0  # Reasonable loss range
    
    def test_forward_backward_pass(self, mock_csv_data):
        """Test complete forward and backward pass."""
        # Setup trainer
        config = {"batch_size": 2, "num_epochs": 1}
        trainer = CrickformerTrainer(config)
        trainer.setup_model()
        
        # Create minimal dataset
        dataset = CrickformerDataset(
            data_root=mock_csv_data,
            use_csv_adapter=True,
            history_length=5
        )
        
        # Get sample and convert to batch format
        sample = dataset[0]
        batch = [sample, sample]  # Duplicate for batch of 2
        collated_batch = collate_fn(batch)
        
        # Move to device
        inputs = {k: v.to(trainer.device) for k, v in collated_batch["inputs"].items()}
        targets = {k: v.to(trainer.device) for k, v in collated_batch["targets"].items()}
        
        # Forward pass
        trainer.model.train()
        outputs = trainer.model(inputs)
        
        # Check output shapes
        assert outputs["win_probability"].shape == (2, 1)
        assert outputs["next_ball_outcome"].shape == (2, 7)
        assert outputs["odds_mispricing"].shape == (2, 1)
        
        # Backward pass
        total_loss, loss_dict = trainer.compute_loss(outputs, targets)
        total_loss.backward()
        
        # Check gradients exist
        for name, param in trainer.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_model_output_shapes(self, mock_csv_data):
        """Test that model outputs have correct shapes."""
        config = {"batch_size": 3}
        trainer = CrickformerTrainer(config)
        trainer.setup_model()
        
        # Create mock inputs with correct shapes
        batch_size = 3
        inputs = {
            "recent_ball_history": torch.randn(batch_size, 5, 6),
            "numeric_features": torch.randn(batch_size, 15),
            "categorical_features": torch.randint(0, 5, (batch_size, 4)),
            "video_features": torch.randn(batch_size, 99),
            "video_mask": torch.ones(batch_size, 1),
            "gnn_embeddings": torch.randn(batch_size, 1, 320)
        }
        
        # Forward pass
        trainer.model.eval()
        with torch.no_grad():
            outputs = trainer.model(inputs)
        
        # Validate output shapes
        assert outputs["win_probability"].shape == (batch_size, 1)
        assert outputs["next_ball_outcome"].shape == (batch_size, 7)
        assert outputs["odds_mispricing"].shape == (batch_size, 1)
        
        # Validate output ranges (after sigmoid/softmax)
        win_probs = torch.sigmoid(outputs["win_probability"])
        assert torch.all(win_probs >= 0.0) and torch.all(win_probs <= 1.0)
        
        outcomes = torch.softmax(outputs["next_ball_outcome"], dim=1)
        assert torch.allclose(outcomes.sum(dim=1), torch.ones(batch_size))


class TestTrainingConfiguration:
    """Test training configuration and CLI integration."""
    
    def test_config_loading(self):
        """Test configuration loading from dict."""
        config = {
            "batch_size": 16,
            "num_epochs": 5,
            "learning_rate": 2e-4,
            "loss_weights": {
                "win_prob": 1.5,
                "outcome": 1.0,
                "mispricing": 0.3
            }
        }
        
        trainer = CrickformerTrainer(config)
        
        assert trainer.batch_size == 16
        assert trainer.num_epochs == 5
        assert trainer.learning_rate == 2e-4
        assert trainer.loss_weights["win_prob"] == 1.5
        assert trainer.loss_weights["outcome"] == 1.0
        assert trainer.loss_weights["mispricing"] == 0.3
    
    def test_default_config(self):
        """Test default configuration values."""
        trainer = CrickformerTrainer({})
        
        assert trainer.batch_size == 32  # Default
        assert trainer.num_epochs == 10  # Default
        assert trainer.learning_rate == 1e-4  # Default
        assert trainer.log_interval == 100  # Default


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 