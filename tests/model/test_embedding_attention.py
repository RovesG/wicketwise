# Purpose: Tests for the MultiHeadGraphAttention model.
# Author: Shamus Rae, Last Modified: 2024-07-30

import torch
import pytest

from crickformers.model.embedding_attention import MultiHeadGraphAttention

@pytest.fixture
def attention_config():
    """Provides a sample configuration for the attention module."""
    return {
        "query_dim": 32,
        "batter_dim": 128,
        "bowler_dim": 128,
        "venue_dim": 64,
        "nhead": 4,
        "attention_dim": 64, # Must be divisible by nhead
    }

def test_multi_head_graph_attention_output_shape(attention_config):
    """
    Tests that the MultiHeadGraphAttention module produces an output
    with the correct shape [batch_size, attention_dim].
    """
    batch_size = 8
    model = MultiHeadGraphAttention(**attention_config)
    model.eval()

    # Create dummy input tensors
    query = torch.randn(batch_size, attention_config["query_dim"])
    batter_emb = torch.randn(batch_size, attention_config["batter_dim"])
    bowler_emb = torch.randn(batch_size, attention_config["bowler_dim"])
    venue_emb = torch.randn(batch_size, attention_config["venue_dim"])

    # Forward pass
    with torch.no_grad():
        output = model(query, batter_emb, bowler_emb, venue_emb)

    # Check output shape
    expected_shape = (batch_size, attention_config["attention_dim"])
    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, but got {output.shape}"

def test_multi_head_graph_attention_invalid_config():
    """
    Tests that the module raises a ValueError if attention_dim is not
    divisible by nhead.
    """
    invalid_config = {
        "query_dim": 32,
        "batter_dim": 128,
        "bowler_dim": 128,
        "venue_dim": 64,
        "nhead": 3,  # Not a divisor of attention_dim
        "attention_dim": 64,
    }
    with pytest.raises(ValueError, match="must be divisible by nhead"):
        MultiHeadGraphAttention(**invalid_config)

def test_multi_head_graph_attention_reproducibility(attention_config):
    """
    Tests for reproducibility of the attention module.
    """
    model = MultiHeadGraphAttention(**attention_config)
    model.eval()

    query = torch.randn(2, attention_config["query_dim"])
    batter_emb = torch.randn(2, attention_config["batter_dim"])
    bowler_emb = torch.randn(2, attention_config["bowler_dim"])
    venue_emb = torch.randn(2, attention_config["venue_dim"])

    with torch.no_grad():
        output1 = model(query, batter_emb, bowler_emb, venue_emb)
        output2 = model(query, batter_emb, bowler_emb, venue_emb)

    assert torch.allclose(output1, output2, atol=1e-6) 