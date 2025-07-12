# Purpose: Tests for the configuration loading module.
# Author: Shamus Rae, Last Modified: 2024-07-30

import pytest
import yaml
from crickformers.config import CrickformerConfig, ModelArchConfig


@pytest.fixture
def mock_config_yaml(tmp_path):
    """Creates a mock YAML config file for testing."""
    config_content = {
        "model": {
            "sequence_encoder": {"input_dim": 4, "hidden_dim": 32, "num_layers": 2},
            "static_context_encoder": {"input_dim": 8, "hidden_dim": 16},
            "fusion_layer": {
                "sequence_dim": 32,
                "context_dim": 16,
                "kg_dim": 128,
                "hidden_dims": [64],
                "latent_dim": 64,
            },
            "prediction_heads": {
                "win_probability": {"input_dim": 64, "output_dim": 1}
            },
        },
        "gnn": {
            "embedding_dim": 128,
            "embeddings_path": "data/test_embeddings.pt",
        },
        "agent": {
            "value_threshold": 0.08,
            "risk_confidence_threshold": 0.8,
        },
        "data_paths": {
            "video_features_dir": "data/videos/",
            "market_odds_dir": "data/odds/",
        },
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
    return config_path


def test_load_from_yaml_file(mock_config_yaml):
    """Tests loading a valid configuration from a YAML file."""
    config = CrickformerConfig.from_file(mock_config_yaml)

    # Assert root-level correctness
    assert isinstance(config, CrickformerConfig)
    assert isinstance(config.model, ModelArchConfig)

    # Assert nested field correctness
    assert config.model.sequence_encoder["hidden_dim"] == 32
    assert config.gnn.embedding_dim == 128
    assert config.agent.value_threshold == 0.08
    assert config.data_paths.video_features_dir == "data/videos/"


def test_to_dict_conversion(mock_config_yaml):
    """Tests the conversion of the config object back to a dictionary."""
    config = CrickformerConfig.from_file(mock_config_yaml)
    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert config_dict["model"]["static_context_encoder"]["input_dim"] == 8
    assert "gnn" in config_dict
    assert "agent" in config_dict


def test_file_not_found_error():
    """Tests that a FileNotFoundError is raised for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        CrickformerConfig.from_file("non_existent_config.yaml")

def test_unsupported_file_format():
    """Tests that a ValueError is raised for an unsupported file format."""
    with open("test.txt", "w") as f:
        f.write("test")
    with pytest.raises(ValueError):
        CrickformerConfig.from_file("test.txt") 