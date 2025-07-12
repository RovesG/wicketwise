# Purpose: Tests for the GNN embedding fetcher.
# Author: Shamus Rae, Last Modified: 2024-07-30

import pytest
import torch
from crickformers.gnn.embedding_fetcher import load_entity_embeddings

@pytest.fixture
def mock_embeddings():
    """Provides a sample dictionary of pre-trained embeddings."""
    return {
        "batter_123": torch.tensor([0.1, 0.2, 0.3]),
        "venue_abc": torch.tensor([0.4, 0.5, 0.6]),
    }

def test_load_existing_embedding(mock_embeddings):
    """Tests loading an embedding that exists in the mock data."""
    embedding_dim = 3
    vector = load_entity_embeddings(
        entity_id="123",
        entity_type="batter",
        embeddings=mock_embeddings,
        embedding_dim=embedding_dim
    )
    assert vector is not None
    assert isinstance(vector, torch.Tensor)
    assert vector.shape == (embedding_dim,)
    assert torch.equal(vector, torch.tensor([0.1, 0.2, 0.3]))

def test_load_missing_embedding(mock_embeddings):
    """Tests loading an embedding for an ID that does not exist."""
    embedding_dim = 3
    vector = load_entity_embeddings(
        entity_id="999",
        entity_type="batter",
        embeddings=mock_embeddings,
        embedding_dim=embedding_dim
    )
    assert vector is not None
    assert isinstance(vector, torch.Tensor)
    assert vector.shape == (embedding_dim,)
    assert torch.equal(vector, torch.zeros(embedding_dim))

def test_load_invalid_entity_type(mock_embeddings):
    """Tests loading an embedding for an entity type that does not exist."""
    embedding_dim = 3
    vector = load_entity_embeddings(
        entity_id="123",
        entity_type="nonexistent_type",
        embeddings=mock_embeddings,
        embedding_dim=embedding_dim
    )
    assert vector is not None
    assert isinstance(vector, torch.Tensor)
    assert vector.shape == (embedding_dim,)
    assert torch.equal(vector, torch.zeros(embedding_dim))

def test_validate_returned_vector_size():
    """Ensures the returned vector matches the specified embedding dimension."""
    embeddings = {}
    embedding_dim = 128
    vector = load_entity_embeddings(
        entity_id="any_id",
        entity_type="any_type",
        embeddings=embeddings,
        embedding_dim=embedding_dim
    )
    assert vector.shape == (embedding_dim,)

    embedding_dim_zero = 0
    vector_zero = load_entity_embeddings(
        entity_id="any_id",
        entity_type="any_type",
        embeddings=embeddings,
        embedding_dim=embedding_dim_zero
    )
    assert vector_zero.shape == (embedding_dim_zero,) 