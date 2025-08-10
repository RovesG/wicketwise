# Purpose: Test EmbeddingService minimal behavior
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-10

from crickformers.inference.embedding_service import EmbeddingService


def test_embedding_service_allows_missing():
    svc = EmbeddingService(embeddings_path="models/gnn_embeddings.pt", allow_missing=True)
    assert svc.query("nonexistent") is None
