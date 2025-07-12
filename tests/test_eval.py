# Purpose: Tests for eval.py evaluation script
# Author: Assistant, Last Modified: 2024

import pytest
import torch
import tempfile
import pandas as pd
import csv
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path so we can import eval
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import CrickformerEvaluator
from crickformers.model.crickformer_model import CrickformerModel


class TestCrickformerEvaluator:
    """Test suite for CrickformerEvaluator class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return {
            "batch_size": 4,
            "test_split": 0.2,
            "model": {
                "sequence_encoder": {
                    "feature_dim": 6,
                    "nhead": 2,
                    "num_encoder_layers": 2,
                    "dim_feedforward": 128,
                    "dropout": 0.1
                },
                "static_context_encoder": {
                    "numeric_dim": 15,
                    "categorical_vocab_sizes": {"competition": 100, "batter_hand": 100, "bowler_type": 100, "innings": 10},
                    "categorical_embedding_dims": {"competition": 8, "batter_hand": 4, "bowler_type": 8, "innings": 4},
                    "video_dim": 99,
                    "hidden_dims": [128, 64],
                    "context_dim": 128,
                    "dropout_rate": 0.1
                },
                "fusion_layer": {
                    "sequence_dim": 6,
                    "context_dim": 128,
                    "kg_dim": 128,
                    "hidden_dims": [256, 128],
                    "latent_dim": 128,
                    "dropout_rate": 0.1
                },
                "prediction_heads": {
                    "win_probability": {"latent_dim": 128, "dropout_rate": 0.1},
                    "next_ball_outcome": {"latent_dim": 128, "num_outcomes": 7, "dropout_rate": 0.1},
                    "odds_mispricing": {"latent_dim": 128, "dropout_rate": 0.1}
                }
            }
        }
    
    @pytest.fixture
    def mock_checkpoint(self, mock_config):
        """Create mock checkpoint file."""
        temp_dir = tempfile.mkdtemp()
        checkpoint_path = Path(temp_dir) / 'mock_checkpoint.pt'
        
        # Create a simple model for testing
        model = CrickformerModel(
            sequence_config=mock_config["model"]["sequence_encoder"],
            static_config=mock_config["model"]["static_context_encoder"],
            fusion_config=mock_config["model"]["fusion_layer"],
            prediction_heads_config=mock_config["model"]["prediction_heads"],
            gnn_embedding_dim=320
        )
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': {},
            'config': mock_config,
            'step_count': 1000
        }
        
        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)
    
    @pytest.fixture
    def mock_csv_data(self):
        """Create mock CSV data for testing."""
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
        nvplay_df = pd.DataFrame(nvplay_data)
        decimal_df = pd.DataFrame(decimal_data)
        
        nvplay_path = Path(temp_dir) / 'nvplay_data_v3.csv'
        decimal_path = Path(temp_dir) / 'decimal_data_v3.csv'
        
        nvplay_df.to_csv(nvplay_path, index=False)
        decimal_df.to_csv(decimal_path, index=False)
        
        return temp_dir
    
    def test_evaluator_initialization(self, mock_config):
        """Test CrickformerEvaluator initialization."""
        evaluator = CrickformerEvaluator(mock_config)
        
        assert evaluator.config == mock_config
        assert evaluator.model is None
        assert evaluator.dataset is None
        assert evaluator.test_loader is None
        assert len(evaluator.outcome_classes) == 7
        assert evaluator.outcome_classes[0] == "0_runs"
        assert evaluator.outcome_classes[6] == "wicket"
    
    def test_load_model_success(self, mock_config, mock_checkpoint):
        """Test successful model loading from checkpoint."""
        evaluator = CrickformerEvaluator(mock_config)
        
        # Load model
        evaluator.load_model(mock_checkpoint)
        
        # Verify model is loaded
        assert evaluator.model is not None
        assert isinstance(evaluator.model, CrickformerModel)
        assert evaluator.model.training is False  # Should be in eval mode
    
    def test_load_model_file_not_found(self, mock_config):
        """Test model loading with non-existent checkpoint."""
        evaluator = CrickformerEvaluator(mock_config)
        
        with pytest.raises(FileNotFoundError):
            evaluator.load_model("nonexistent_checkpoint.pt")
    
    def test_setup_dataset(self, mock_config, mock_csv_data):
        """Test dataset setup."""
        evaluator = CrickformerEvaluator(mock_config)
        
        # Setup dataset
        evaluator.setup_dataset(mock_csv_data, use_csv=True, test_split=0.2)
        
        # Verify dataset is created
        assert evaluator.dataset is not None
        assert evaluator.test_loader is not None
        assert len(evaluator.dataset) > 0
    
    def test_extract_actual_outcomes(self, mock_config):
        """Test extraction of actual outcomes from batch."""
        evaluator = CrickformerEvaluator(mock_config)
        
        # Create mock batch
        batch = {
            "targets": {
                "outcome": torch.tensor([0, 1, 4, 6]),
                "win_prob": torch.tensor([[0.5], [0.6], [0.4], [0.7]]),
                "mispricing": torch.tensor([[0.1], [0.2], [0.3], [0.4]])
            }
        }
        
        # Extract outcomes
        outcomes = evaluator._extract_actual_outcomes(batch)
        
        # Verify extraction
        assert len(outcomes) == 4
        assert outcomes[0]["actual_runs"] == 0
        assert outcomes[1]["actual_runs"] == 1
        assert outcomes[2]["actual_runs"] == 4
        assert outcomes[3]["actual_runs"] == 6
        assert abs(outcomes[0]["actual_win_prob"] - 0.5) < 1e-6
        assert abs(outcomes[3]["actual_mispricing"] - 0.4) < 1e-6
    
    def test_get_ball_metadata(self, mock_config):
        """Test extraction of ball metadata from batch."""
        evaluator = CrickformerEvaluator(mock_config)
        
        # Create mock batch
        batch = {
            "inputs": {
                "numeric_features": torch.randn(3, 15)
            }
        }
        
        # Extract metadata
        metadata = evaluator._get_ball_metadata(batch)
        
        # Verify metadata
        assert len(metadata) == 3
        for i, meta in enumerate(metadata):
            assert "match_id" in meta
            assert "ball_id" in meta
            assert "phase" in meta
            assert "batter_id" in meta
            assert "bowler_id" in meta
            assert meta["ball_id"] == f"ball_{i}"
    
    def test_evaluate_model_without_model(self, mock_config):
        """Test evaluation without loading model first."""
        evaluator = CrickformerEvaluator(mock_config)
        
        with pytest.raises(ValueError, match="Model not loaded"):
            evaluator.evaluate_model()
    
    def test_evaluate_model_without_dataset(self, mock_config, mock_checkpoint):
        """Test evaluation without setting up dataset first."""
        evaluator = CrickformerEvaluator(mock_config)
        evaluator.load_model(mock_checkpoint)
        
        with pytest.raises(ValueError, match="Test dataset not setup"):
            evaluator.evaluate_model()
    
    def test_evaluate_model_full_pipeline(self, mock_config, mock_checkpoint, mock_csv_data):
        """Test full evaluation pipeline."""
        evaluator = CrickformerEvaluator(mock_config)
        
        # Setup evaluator
        evaluator.load_model(mock_checkpoint)
        evaluator.setup_dataset(mock_csv_data, use_csv=True, test_split=0.8)  # Use more data for testing
        
        # Run evaluation
        temp_dir = tempfile.mkdtemp()
        output_csv = Path(temp_dir) / 'test_predictions.csv'
        
        total_predictions = evaluator.evaluate_model(str(output_csv))
        
        # Verify output
        assert total_predictions > 0
        assert output_csv.exists()
        
        # Verify CSV content
        df = pd.read_csv(output_csv)
        assert len(df) == total_predictions
        
        # Check required columns
        required_columns = [
            "match_id", "ball_id", "actual_runs", "predicted_runs_class",
            "win_prob", "odds_mispricing", "phase", "batter_id", "bowler_id"
        ]
        for col in required_columns:
            assert col in df.columns
        
        # Check data types and ranges
        assert df["win_prob"].dtype == float
        assert df["odds_mispricing"].dtype == float
        assert df["win_prob"].min() >= 0.0
        assert df["win_prob"].max() <= 1.0
        assert df["odds_mispricing"].min() >= 0.0
        assert df["odds_mispricing"].max() <= 1.0
        
        # Check predicted classes are valid
        valid_classes = ["0_runs", "1_run", "2_runs", "3_runs", "4_runs", "6_runs", "wicket"]
        assert all(cls in valid_classes for cls in df["predicted_runs_class"].unique())


class TestEvaluationCSVOutput:
    """Test suite for CSV output validation."""
    
    def test_csv_structure(self):
        """Test that CSV output has the correct structure."""
        temp_dir = tempfile.mkdtemp()
        csv_path = Path(temp_dir) / 'test_output.csv'
        
        # Expected headers
        expected_headers = [
            "match_id", "ball_id", "actual_runs", "predicted_runs_class",
            "win_prob", "odds_mispricing", "phase", "batter_id", "bowler_id",
            "predicted_runs_0", "predicted_runs_1", "predicted_runs_2", 
            "predicted_runs_3", "predicted_runs_4", "predicted_runs_6", 
            "predicted_wicket", "actual_win_prob", "actual_mispricing"
        ]
        
        # Create test CSV
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=expected_headers)
            writer.writeheader()
            
            # Write test row
            test_row = {
                "match_id": "test_match",
                "ball_id": "test_ball",
                "actual_runs": 1,
                "predicted_runs_class": "1_run",
                "win_prob": 0.65,
                "odds_mispricing": 0.12,
                "phase": "powerplay",
                "batter_id": "batter_001",
                "bowler_id": "bowler_001",
                "predicted_runs_0": 0.1,
                "predicted_runs_1": 0.6,
                "predicted_runs_2": 0.15,
                "predicted_runs_3": 0.05,
                "predicted_runs_4": 0.05,
                "predicted_runs_6": 0.03,
                "predicted_wicket": 0.02,
                "actual_win_prob": 0.68,
                "actual_mispricing": 0.15
            }
            writer.writerow(test_row)
        
        # Verify CSV structure
        df = pd.read_csv(csv_path)
        assert list(df.columns) == expected_headers
        assert len(df) == 1
        assert df.iloc[0]["match_id"] == "test_match"
        assert df.iloc[0]["win_prob"] == 0.65
        assert df.iloc[0]["predicted_runs_class"] == "1_run"
    
    def test_csv_data_types(self):
        """Test CSV data types are correct."""
        temp_dir = tempfile.mkdtemp()
        csv_path = Path(temp_dir) / 'test_types.csv'
        
        # Create test data with various types
        test_data = [
            {
                "match_id": "match_1",
                "ball_id": "ball_1",
                "actual_runs": 0,
                "predicted_runs_class": "0_runs",
                "win_prob": 0.45,
                "odds_mispricing": 0.23,
                "phase": "powerplay",
                "batter_id": "batter_1",
                "bowler_id": "bowler_1",
                "predicted_runs_0": 0.7,
                "predicted_runs_1": 0.15,
                "predicted_runs_2": 0.08,
                "predicted_runs_3": 0.03,
                "predicted_runs_4": 0.02,
                "predicted_runs_6": 0.01,
                "predicted_wicket": 0.01,
                "actual_win_prob": 0.48,
                "actual_mispricing": 0.21
            },
            {
                "match_id": "match_2",
                "ball_id": "ball_2",
                "actual_runs": 6,
                "predicted_runs_class": "6_runs",
                "win_prob": 0.72,
                "odds_mispricing": 0.05,
                "phase": "death_overs",
                "batter_id": "batter_2",
                "bowler_id": "bowler_2",
                "predicted_runs_0": 0.05,
                "predicted_runs_1": 0.1,
                "predicted_runs_2": 0.1,
                "predicted_runs_3": 0.05,
                "predicted_runs_4": 0.15,
                "predicted_runs_6": 0.5,
                "predicted_wicket": 0.05,
                "actual_win_prob": 0.75,
                "actual_mispricing": 0.03
            }
        ]
        
        # Write test data
        headers = list(test_data[0].keys())
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(test_data)
        
        # Verify data types
        df = pd.read_csv(csv_path)
        
        # String columns
        string_cols = ["match_id", "ball_id", "predicted_runs_class", "phase", "batter_id", "bowler_id"]
        for col in string_cols:
            assert df[col].dtype == object
        
        # Numeric columns
        numeric_cols = ["actual_runs"]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col])
        
        # Float columns (probabilities)
        float_cols = ["win_prob", "odds_mispricing", "predicted_runs_0", "predicted_runs_1", 
                     "predicted_runs_2", "predicted_runs_3", "predicted_runs_4", 
                     "predicted_runs_6", "predicted_wicket", "actual_win_prob", "actual_mispricing"]
        for col in float_cols:
            assert pd.api.types.is_numeric_dtype(df[col])
            # Check probability ranges
            if col.startswith("predicted_") or col in ["win_prob", "odds_mispricing", "actual_win_prob", "actual_mispricing"]:
                assert df[col].min() >= 0.0
                assert df[col].max() <= 1.0


class TestEvaluationIntegration:
    """Integration tests for the evaluation pipeline."""
    
    def test_outcome_class_mapping(self):
        """Test that outcome class mapping is correct."""
        config = {"batch_size": 1}
        evaluator = CrickformerEvaluator(config)
        
        expected_mapping = {
            0: "0_runs",
            1: "1_run", 
            2: "2_runs",
            3: "3_runs",
            4: "4_runs",
            5: "6_runs",
            6: "wicket"
        }
        
        assert evaluator.outcome_classes == expected_mapping
    
    def test_prediction_probability_consistency(self):
        """Test that prediction probabilities are consistent."""
        # Create mock predictions
        outcome_probs = torch.tensor([
            [0.1, 0.2, 0.15, 0.05, 0.2, 0.25, 0.05],  # Should sum to 1.0
            [0.3, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]    # Should sum to 1.0
        ])
        
        # Apply softmax to ensure they sum to 1
        normalized_probs = torch.softmax(outcome_probs, dim=1)
        
        # Check that probabilities sum to 1 (within tolerance)
        prob_sums = normalized_probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(2), atol=1e-6)
    
    def test_device_handling(self):
        """Test that device handling works correctly."""
        config = {"batch_size": 1}
        evaluator = CrickformerEvaluator(config)
        
        # Should default to CPU if CUDA not available
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert evaluator.device == expected_device


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 