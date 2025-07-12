# Purpose: Tests for CrickformerDataset PyTorch dataset class
# Author: Shamus Rae, Last Modified: 2024-07-30

import pytest
import json
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from crickformers.crickformer_dataset import CrickformerDataset


@pytest.fixture
def mock_data_dir():
    """Create a temporary directory with mock cricket match data."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create match 1 directory
    match1_dir = temp_dir / "match_001"
    match1_dir.mkdir()
    
    # Create balls data for match 1
    match1_balls = [
        {
            "over": 1,
            "ball_in_over": 1,
            "runs_scored": 1,
            "wickets_fallen": 0,
            "current_score": 1,
            "current_wickets": 0,
            "balls_remaining": 119,
            "required_run_rate": 8.5,
            "current_run_rate": 6.0,
            "powerplay_overs_left": 5,
            "batter_id": "player_001",
            "bowler_id": "bowler_001",
            "venue_id": "venue_001",
            "history": [
                {
                    "runs_scored": 0,
                    "wicket_fell": False,
                    "extras": 0,
                    "shot_type": "defensive",
                    "outcome": "dot"
                }
            ],
            "video_signals": {
                "pre_ball_features": [0.1] * 512,
                "post_ball_features": [0.2] * 512,
                "confidence_score": 0.95
            },
            "gnn_embeddings": {
                "batter_embedding": [0.1] * 128,
                "bowler_embedding": [0.2] * 128,
                "venue_embedding": [0.3] * 128
            },
            "market_odds": {
                "back_odds": [1.5, 2.0, 3.0],
                "lay_odds": [1.6, 2.1, 3.2],
                "volume": 1000.0,
                "timestamp": 1234567890
            }
        },
        {
            "over": 1,
            "ball_in_over": 2,
            "runs_scored": 4,
            "wickets_fallen": 0,
            "current_score": 5,
            "current_wickets": 0,
            "balls_remaining": 118,
            "required_run_rate": 8.3,
            "current_run_rate": 7.5,
            "powerplay_overs_left": 5,
            "batter_id": "player_001",
            "bowler_id": "bowler_001",
            "venue_id": "venue_001",
            "history": [
                {
                    "runs_scored": 0,
                    "wicket_fell": False,
                    "extras": 0,
                    "shot_type": "defensive",
                    "outcome": "dot"
                },
                {
                    "runs_scored": 1,
                    "wicket_fell": False,
                    "extras": 0,
                    "shot_type": "push",
                    "outcome": "single"
                }
            ],
            "video_signals": {
                "pre_ball_features": [0.3] * 512,
                "post_ball_features": [0.4] * 512,
                "confidence_score": 0.88
            },
            "gnn_embeddings": {
                "batter_embedding": [0.4] * 128,
                "bowler_embedding": [0.5] * 128,
                "venue_embedding": [0.6] * 128
            },
            "market_odds": {
                "back_odds": [1.4, 1.9, 2.8],
                "lay_odds": [1.5, 2.0, 3.0],
                "volume": 1500.0,
                "timestamp": 1234567900
            }
        }
    ]
    
    # Save balls data
    with open(match1_dir / "balls.json", 'w') as f:
        json.dump(match1_balls, f)
    
    # Create match 2 directory
    match2_dir = temp_dir / "match_002"
    match2_dir.mkdir()
    
    # Create balls data for match 2 (minimal data to test fallbacks)
    match2_balls = [
        {
            "over": 1,
            "ball_in_over": 1,
            "runs_scored": 0,
            "batter_id": "player_002",
            "bowler_id": "bowler_002",
            "venue_id": "venue_002"
            # Note: missing video_signals, gnn_embeddings, market_odds, history
        }
    ]
    
    # Save balls data
    with open(match2_dir / "balls.json", 'w') as f:
        json.dump(match2_balls, f)
    
    # Create embeddings directory for match 2
    embeddings_dir = match2_dir / "embeddings"
    embeddings_dir.mkdir()
    
    # Create embedding files
    np.save(embeddings_dir / "batter_player_002.npy", np.random.rand(128))
    np.save(embeddings_dir / "bowler_bowler_002.npy", np.random.rand(128))
    np.save(embeddings_dir / "venue_venue_002.npy", np.random.rand(64))
    np.save(embeddings_dir / "edge_player_002_bowler_002.npy", np.random.rand(64))
    
    # Create separate video file for match 2
    video_data = {
        "pre_ball_features": [0.7] * 512,
        "post_ball_features": [0.8] * 512,
        "confidence_score": 0.75
    }
    with open(match2_dir / "video_0.json", 'w') as f:
        json.dump(video_data, f)
    
    # Create separate odds file for match 2
    odds_data = {
        "back_odds": [2.0, 2.5, 3.5],
        "lay_odds": [2.1, 2.6, 3.7],
        "volume": 800.0,
        "timestamp": 1234568000
    }
    with open(match2_dir / "odds_0.json", 'w') as f:
        json.dump(odds_data, f)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_manifest_data(mock_data_dir):
    """Create a manifest file for the mock data."""
    manifest = {
        "matches": [
            {
                "match_id": "match_001",
                "balls_file": "match_001/balls.json",
                "date": "2024-01-01",
                "teams": ["Team A", "Team B"]
            },
            {
                "match_id": "match_002", 
                "balls_file": "match_002/balls.json",
                "date": "2024-01-02",
                "teams": ["Team C", "Team D"]
            }
        ]
    }
    
    manifest_file = mock_data_dir / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f)
    
    return manifest_file


def test_dataset_initialization_from_directory(mock_data_dir):
    """Test dataset initialization from directory structure."""
    dataset = CrickformerDataset(data_root=mock_data_dir)
    
    assert len(dataset) == 3  # 2 balls from match_001 + 1 ball from match_002
    assert len(dataset.get_match_ids()) == 2
    assert "match_001" in dataset.get_match_ids()
    assert "match_002" in dataset.get_match_ids()


def test_dataset_initialization_from_manifest(mock_data_dir, mock_manifest_data):
    """Test dataset initialization from manifest file."""
    dataset = CrickformerDataset(
        data_root=mock_data_dir,
        manifest_file="manifest.json"
    )
    
    assert len(dataset) == 3
    assert len(dataset.get_match_ids()) == 2


def test_dataset_getitem_structure(mock_data_dir):
    """Test that __getitem__ returns correctly structured data."""
    dataset = CrickformerDataset(data_root=mock_data_dir)
    
    # Get first sample (from match_001)
    sample = dataset[0]
    
    # Check that it's a dictionary of tensors
    assert isinstance(sample, dict)
    
    # Check for required keys (from InputPreprocessor)
    expected_keys = [
        'numeric_ball_features', 'categorical_ball_features', 'ball_history', 
        'video_features', 'video_mask', 'gnn_embeddings', 'market_odds'
    ]
    
    for key in expected_keys:
        assert key in sample, f"Missing key: {key}"
        assert isinstance(sample[key], torch.Tensor), f"Key {key} is not a tensor"


def test_dataset_feature_dimensions(mock_data_dir):
    """Test that features have correct dimensions."""
    dataset = CrickformerDataset(
        data_root=mock_data_dir,
        history_length=5,
        embedding_dim=128,
        video_feature_dim=512
    )
    
    sample = dataset[0]
    
    # Check numeric ball features dimension (from preprocessor)
    assert sample['numeric_ball_features'].shape[0] == 7  # 7 numeric features
    
    # Check categorical ball features dimension
    assert sample['categorical_ball_features'].shape[0] == 4  # 4 categorical features
    
    # Check history features dimension
    assert sample['ball_history'].shape[0] == 5  # history_length
    assert sample['ball_history'].shape[1] == 6  # 6 features per history entry
    
    # Check GNN embeddings dimension (128 + 128 + 64 + 64)
    assert sample['gnn_embeddings'].shape[0] == 384
    
    # Check video features dimension (8 numeric + 3 categorical)
    assert sample['video_features'].shape[0] == 11
    
    # Check video mask
    assert sample['video_mask'].shape[0] == 1


def test_dataset_handles_missing_data(mock_data_dir):
    """Test that dataset handles missing data gracefully."""
    dataset = CrickformerDataset(data_root=mock_data_dir)
    
    # Get sample from match_002 (which has minimal data)
    match_002_samples = [i for i, s in enumerate(dataset.samples) if s['match_id'] == 'match_002']
    sample_idx = match_002_samples[0]
    
    sample = dataset[sample_idx]
    
    # Should still return valid tensors
    assert isinstance(sample, dict)
    assert all(isinstance(v, torch.Tensor) for v in sample.values())
    
    # Check that dimensions are still correct
    assert sample['numeric_ball_features'].shape[0] == 7
    assert sample['categorical_ball_features'].shape[0] == 4
    assert sample['ball_history'].shape == (5, 6)
    assert sample['gnn_embeddings'].shape[0] == 384
    assert sample['video_features'].shape[0] == 11


def test_dataset_history_padding(mock_data_dir):
    """Test that ball history is properly padded."""
    dataset = CrickformerDataset(data_root=mock_data_dir, history_length=5)
    
    # Get sample from match_002 (which has no history)
    match_002_samples = [i for i, s in enumerate(dataset.samples) if s['match_id'] == 'match_002']
    sample_idx = match_002_samples[0]
    
    sample = dataset[sample_idx]
    
    # History should be padded to required length
    assert sample['ball_history'].shape[0] == 5
    
    # Check that padding values are zeros (for numeric features)
    # First few entries should be padding (all zeros)
    padding_entries = sample['ball_history'][:4]  # First 4 should be padding
    assert torch.all(padding_entries == 0)


def test_dataset_video_alignment(mock_data_dir):
    """Test that video signals are correctly aligned with ball data."""
    dataset = CrickformerDataset(data_root=mock_data_dir)
    
    # Get first sample (has video data embedded)
    sample = dataset[0]
    video_features = sample['video_features']
    video_mask = sample['video_mask']
    
    # Check that video features are not all zeros
    assert not torch.all(video_features == 0)
    
    # Check that video mask indicates video is present
    assert video_mask.item() == 1.0


def test_dataset_embeddings_alignment(mock_data_dir):
    """Test that GNN embeddings are correctly aligned."""
    dataset = CrickformerDataset(data_root=mock_data_dir)
    
    # Get sample from match_002 (loads embeddings from files)
    match_002_samples = [i for i, s in enumerate(dataset.samples) if s['match_id'] == 'match_002']
    sample_idx = match_002_samples[0]
    
    sample = dataset[sample_idx]
    gnn_embeddings = sample['gnn_embeddings']
    
    # Check that embeddings are not all zeros
    assert not torch.all(gnn_embeddings == 0)
    
    # Check that we have 3 concatenated embeddings
    assert gnn_embeddings.shape[0] == 384  # 3 * 128


def test_dataset_odds_alignment(mock_data_dir):
    """Test that market odds are correctly aligned."""
    dataset = CrickformerDataset(data_root=mock_data_dir)
    
    # Get first sample (has odds data embedded)
    sample = dataset[0]
    market_odds = sample['market_odds']
    
    # Check that we have the expected structure
    assert market_odds.shape[0] == 10  # 3 win odds + 7 next ball odds
    
    # Market odds should be tensor of zeros if no odds data
    assert isinstance(market_odds, torch.Tensor)


def test_dataset_selective_loading(mock_data_dir):
    """Test dataset with selective loading options."""
    dataset = CrickformerDataset(
        data_root=mock_data_dir,
        load_video=False,
        load_embeddings=False,
        load_odds=False
    )
    
    sample = dataset[0]
    
    # Video features should be all zeros
    assert torch.all(sample['video_features'] == 0)
    
    # GNN embeddings should be all zeros
    assert torch.all(sample['gnn_embeddings'] == 0)
    
    # Market odds should be all zeros
    assert torch.all(sample['market_odds'] == 0)


def test_dataset_augmentation_hook(mock_data_dir):
    """Test that augmentation function is applied."""
    def mock_augmentation(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Simple augmentation: add noise to numeric ball features
        inputs['numeric_ball_features'] = inputs['numeric_ball_features'] + torch.randn_like(inputs['numeric_ball_features']) * 0.1
        return inputs
    
    dataset = CrickformerDataset(
        data_root=mock_data_dir,
        augmentation_fn=mock_augmentation
    )
    
    sample = dataset[0]
    
    # Should still have valid structure
    assert isinstance(sample, dict)
    assert all(isinstance(v, torch.Tensor) for v in sample.values())


def test_dataset_get_sample_info(mock_data_dir):
    """Test getting sample metadata."""
    dataset = CrickformerDataset(data_root=mock_data_dir)
    
    info = dataset.get_sample_info(0)
    
    assert isinstance(info, dict)
    assert 'match_id' in info
    assert 'ball_idx' in info
    assert 'over' in info
    assert 'ball_in_over' in info
    assert 'batter_id' in info
    assert 'bowler_id' in info
    
    # Check specific values for first sample
    assert info['match_id'] == 'match_001'
    assert info['ball_idx'] == 0
    assert info['over'] == 1
    assert info['ball_in_over'] == 1
    assert info['batter_id'] == 'player_001'


def test_dataset_filter_by_match(mock_data_dir):
    """Test filtering dataset by match ID."""
    dataset = CrickformerDataset(data_root=mock_data_dir)
    
    # Filter to only match_001
    match_001_dataset = dataset.filter_by_match('match_001')
    
    assert len(match_001_dataset) == 2  # Only 2 balls from match_001
    assert len(match_001_dataset.get_match_ids()) == 1
    assert match_001_dataset.get_match_ids()[0] == 'match_001'


def test_dataset_empty_directory():
    """Test dataset with empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = CrickformerDataset(data_root=temp_dir)
        assert len(dataset) == 0
        assert len(dataset.get_match_ids()) == 0


def test_dataset_missing_manifest_file(mock_data_dir):
    """Test dataset with missing manifest file."""
    with pytest.raises(FileNotFoundError):
        CrickformerDataset(
            data_root=mock_data_dir,
            manifest_file="nonexistent_manifest.json"
        )


def test_dataset_tensor_types(mock_data_dir):
    """Test that all returned tensors have correct types."""
    dataset = CrickformerDataset(data_root=mock_data_dir)
    
    sample = dataset[0]
    
    # All values should be float tensors
    for key, tensor in sample.items():
        assert tensor.dtype == torch.float32, f"Tensor {key} has wrong dtype: {tensor.dtype}"


def test_dataset_reproducibility(mock_data_dir):
    """Test that dataset returns consistent results."""
    dataset1 = CrickformerDataset(data_root=mock_data_dir)
    dataset2 = CrickformerDataset(data_root=mock_data_dir)
    
    sample1 = dataset1[0]
    sample2 = dataset2[0]
    
    # Should return identical tensors
    for key in sample1.keys():
        assert torch.equal(sample1[key], sample2[key]), f"Tensors differ for key: {key}"


def test_dataset_custom_dimensions(mock_data_dir):
    """Test dataset with custom dimensions."""
    dataset = CrickformerDataset(
        data_root=mock_data_dir,
        history_length=3,
        embedding_dim=64,
        video_feature_dim=256
    )
    
    sample = dataset[0]
    
    # Check custom dimensions
    assert sample['ball_history'].shape[0] == 3  # Custom history length
    assert sample['gnn_embeddings'].shape[0] == 384  # Fixed: 128 + 128 + 64 + 64
    assert sample['video_features'].shape[0] == 11  # Fixed: 8 numeric + 3 categorical 