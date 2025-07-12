# Purpose: Fetches pre-trained GNN entity embeddings.
# Author: Shamus Rae, Last Modified: 2024-07-30

import torch
from typing import Dict

def load_entity_embeddings(
    entity_id: str,
    entity_type: str,
    embeddings: Dict[str, torch.Tensor],
    embedding_dim: int
) -> torch.Tensor:
    """
    Looks up a GNN embedding vector for a given entity.

    Args:
        entity_id: The unique identifier for the entity (e.g., '12345').
        entity_type: The type of entity (e.g., 'batter', 'venue').
        embeddings: A dictionary mapping entity keys to embedding vectors.
        embedding_dim: The dimensionality of the embedding vectors.

    Returns:
        A torch.Tensor representing the entity's embedding, or a zero-vector
        if the entity is not found.
    """
    key = f"{entity_type}_{entity_id}"
    return embeddings.get(key, torch.zeros(embedding_dim)) 