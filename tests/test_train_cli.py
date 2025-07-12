# Purpose: Tests for the training CLI entrypoint.
# Author: Shamus Rae, Last Modified: 2024-07-30

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from crickformers.train import main as train_main
import torch


@pytest.fixture
def mock_config_file(tmp_path):
    """Creates a mock JSON config file for testing."""
    config = {
        "model": {
            "sequence_encoder": {"input_dim": 10, "hidden_dim": 32, "num_layers": 2},
            "static_context_encoder": {"input_dim": 5, "hidden_dim": 16},
            "fusion_layer": {
                "sequence_dim": 32,
                "context_dim": 16,
                "kg_dim": 128,
                "hidden_dims": [64],
                "latent_dim": 64,
            },
            "prediction_heads": {
                "next_ball_outcome": {"input_dim": 64, "output_dim": 7},
                "win_probability": {"input_dim": 64, "output_dim": 1},
                "odds_mispricing": {"input_dim": 64, "output_dim": 1},
            },
        },
        "training_params": {
            "learning_rate": 0.001,
            "epochs": 2,
            "loss_weights": {"next_ball_outcome": 0.6, "win_probability": 0.4},
        },
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return str(config_path)


@patch("crickformers.train.train_model")
@patch("crickformers.train.CrickformerModel")
def test_train_cli_runs_successfully(
    mock_crickformer_model, mock_train_model, mock_config_file, tmp_path
):
    """
    Tests that the CLI entrypoint can be called, parses args correctly,
    and launches the training process.
    """
    save_path = tmp_path / "model.pth"
    # Mock the train_model to return a dummy model object
    mock_trained_model = MagicMock()
    mock_trained_model.state_dict.return_value = {"param": "value"}
    mock_train_model.return_value = mock_trained_model

    test_args = [
        "train.py",
        "--config",
        mock_config_file,
        "--save-path",
        str(save_path),
    ]

    with patch("sys.argv", test_args):
        train_main()

    # Assertions
    mock_crickformer_model.assert_called_once()
    mock_train_model.assert_called_once()
    assert os.path.exists(save_path)
    # Check that torch.save was called by checking the mock's state_dict was called
    mock_trained_model.state_dict.assert_called_once()

@patch("torch.load")
@patch("crickformers.train.train_model")
@patch("crickformers.train.CrickformerModel")
def test_train_cli_resume_option(
    mock_crickformer_model, mock_train_model, mock_torch_load, mock_config_file, tmp_path
):
    """Tests that the --resume argument correctly loads a model state_dict."""
    save_path = tmp_path / "model.pth"
    resume_path = tmp_path / "resume_model.pth"
    open(resume_path, "a").close()

    # Configure the mock to return a serializable nn.Module
    mock_trained_model = torch.nn.Linear(1, 1)
    mock_train_model.return_value = mock_trained_model

    mock_instance = mock_crickformer_model.return_value
    
    test_args = [
        "train.py",
        "--config",
        mock_config_file,
        "--save-path",
        str(save_path),
        "--resume",
        str(resume_path),
    ]

    with patch("sys.argv", test_args):
        train_main()

    mock_torch_load.assert_called_with(str(resume_path))
    mock_instance.load_state_dict.assert_called_once() 