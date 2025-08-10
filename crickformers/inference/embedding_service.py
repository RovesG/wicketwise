# Purpose: Lightweight service to load/query GNN embeddings
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-10

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch


class EmbeddingService:
    def __init__(self, embeddings_path: str = 'models/gnn_embeddings.pt', allow_missing: bool = False):
        self.embeddings_path = Path(embeddings_path)
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.allow_missing = allow_missing
        if self.embeddings_path.exists():
            try:
                self._load()
            except Exception:
                if not self.allow_missing:
                    raise
                # tolerate incompatible or unsafe checkpoint formats in tests
                self.embeddings = {}
        elif not allow_missing:
            raise FileNotFoundError(f"Embeddings not found at {self.embeddings_path}")

    def _load(self) -> None:
        obj = torch.load(self.embeddings_path, map_location='cpu', weights_only=False)
        if isinstance(obj, dict):
            self.embeddings = {str(k): torch.as_tensor(v) for k, v in obj.items()}
        else:
            self.embeddings = {}

    def query(self, node_id: str) -> Optional[torch.Tensor]:
        return self.embeddings.get(str(node_id))
