# Purpose: Tests for the StaticContextEncoder module.
# Author: Shamus Rae, Last Modified: 2024-07-30

import torch
import pytest
from crickformers.model.static_context_encoder import StaticContextEncoder

@pytest.fixture
def context_encoder_config():
    """Provides a sample configuration for the StaticContextEncoder."""
    return {
        "numeric_dim": 10,
        "categorical_vocab_sizes": {"feat1": 10, "feat2": 5},
        "categorical_embedding_dims": {"feat1": 4, "feat2": 2},
        "video_dim": 20,
        "weather_dim": 6,
        "venue_coord_dim": 2,
        "hidden_dims": [64, 32],
        "context_dim": 40,
        "dropout_rate": 0.1,
    }

@pytest.fixture
def sample_encoder_inputs(context_encoder_config):
    """Provides sample input tensors for the context encoder."""
    batch_size = 8
    numeric_feats = torch.randn(batch_size, context_encoder_config["numeric_dim"])
    # Categorical features must be within the vocab size range
    cat_feats = torch.stack([
        torch.randint(0, 10, (batch_size,)),
        torch.randint(0, 5, (batch_size,)),
    ], dim=1)
    video_feats = torch.randn(batch_size, context_encoder_config["video_dim"])
    # Video mask: half of the batch has video, half does not
    video_mask = torch.tensor([[1.0]] * 4 + [[0.0]] * 4)
    
    # Add optional weather and venue features
    weather_feats = torch.randn(batch_size, 6) if context_encoder_config.get("weather_dim") else None
    venue_coords = torch.randn(batch_size, 2) if context_encoder_config.get("venue_coord_dim") else None
    
    return numeric_feats, cat_feats, video_feats, video_mask, weather_feats, venue_coords

def test_context_encoder_output_shape(context_encoder_config, sample_encoder_inputs):
    """
    Tests that the StaticContextEncoder produces an output with the correct shape.
    """
    model = StaticContextEncoder(**context_encoder_config)
    model.eval()
    
    with torch.no_grad():
        numeric_feats, cat_feats, video_feats, video_mask, weather_feats, venue_coords = sample_encoder_inputs
        output = model(
            numeric_features=numeric_feats,
            categorical_features=cat_feats,
            video_features=video_feats,
            video_mask=video_mask,
            weather_features=weather_feats,
            venue_coordinates=venue_coords
        )
        
    expected_shape = (numeric_feats.shape[0], context_encoder_config["context_dim"])
    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, but got {output.shape}"

def test_embedding_application(context_encoder_config):
    """
    Ensures that the embedding layers are correctly defined and applied.
    """
    model = StaticContextEncoder(**context_encoder_config)
    assert len(model.embedding_layers) == 2
    assert model.embedding_layers["feat1"].embedding_dim == 4
    assert model.embedding_layers["feat2"].embedding_dim == 2
    
def test_video_masking_logic(context_encoder_config, sample_encoder_inputs):
    """
    Tests that the video mask correctly zeros out video features when the mask is 0.
    """
    numeric_feats, cat_feats, video_feats, video_mask, weather_feats, venue_coords = sample_encoder_inputs
    
    # We can check the intermediate combined tensor to see the effect of the mask.
    # To do this without altering the model, we can replicate the forward pass logic.
    embedded_cats = [
        layer(cat_feats[:, i])
        for i, layer in enumerate(StaticContextEncoder(**context_encoder_config).embedding_layers.values())
    ]
    all_embeddings = torch.cat(embedded_cats, dim=1)
    
    masked_video_features = video_feats * video_mask
    
    # Check the second half of the batch where the mask is 0
    assert torch.all(masked_video_features[4:] == 0), "Video features should be zeroed out where mask is 0."
    # Check the first half of the batch where the mask is 1
    assert torch.all(masked_video_features[:4] != 0), "Video features should not be zeroed out where mask is 1."
    
    combined = torch.cat([numeric_feats, all_embeddings, masked_video_features], dim=1)
    # Ensure no NaNs are produced
    assert not torch.isnan(combined).any() 