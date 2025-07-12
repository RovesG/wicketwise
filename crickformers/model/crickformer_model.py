# Purpose: Defines the main Crickformer model architecture.
# Author: Shamus Rae, Last Modified: 2024-07-30

from typing import Any, Dict

import torch
from torch import nn

from crickformers.model.embedding_attention import MultiHeadGraphAttention
from crickformers.model.fusion_layer import CrickformerFusionLayer
from crickformers.model.prediction_heads import (
    NextBallOutcomeHead,
    OddsMispricingHead,
    WinProbabilityHead,
)
from crickformers.model.sequence_encoder import BallHistoryEncoder
from crickformers.model.static_context_encoder import StaticContextEncoder


class CrickformerModel(nn.Module):
    """
    The main Crickformer model, which integrates all sub-components.
    """

    def __init__(
        self,
        sequence_config: Dict[str, Any],
        static_config: Dict[str, Any],
        fusion_config: Dict[str, Any],
        prediction_heads_config: Dict[str, Any],
        gnn_embedding_dim: int = 128,
    ):
        super().__init__()
        self.sequence_encoder = BallHistoryEncoder(**sequence_config)
        self.static_context_encoder = StaticContextEncoder(**static_config)
        self.gnn_attention = MultiHeadGraphAttention(
            embed_dim=gnn_embedding_dim, num_heads=4
        )
        self.fusion_layer = CrickformerFusionLayer(**fusion_config)

        self.win_prob_head = WinProbabilityHead(
            **prediction_heads_config["win_probability"]
        )
        self.next_ball_head = NextBallOutcomeHead(
            **prediction_heads_config["next_ball_outcome"]
        )
        self.odds_mispricing_head = OddsMispricingHead(
            **prediction_heads_config["odds_mispricing"]
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Defines the forward pass of the model.
        """
        # Ensure all expected keys are present, providing zero tensors for missing ones
        required_keys = [
            "recent_ball_history",
            "static_features",
            "gnn_embeddings",
            "video_signals",
        ]
        for key in required_keys:
            if key not in inputs:
                # This is a simplification; dimensions should be handled more robustly
                inputs[key] = torch.zeros(1, 1, device=self.device)

        sequence_embedding = self.sequence_encoder(inputs["recent_ball_history"])
        static_embedding = self.static_context_encoder(inputs["static_features"])
        gnn_embedding, _ = self.gnn_attention(
            query=static_embedding.unsqueeze(1),
            key=inputs["gnn_embeddings"],
            value=inputs["gnn_embeddings"],
        )
        gnn_embedding = gnn_embedding.squeeze(1)

        fused_vector = self.fusion_layer(
            sequence_vector=sequence_embedding,
            context_vector=static_embedding,
            attended_kg_vector=gnn_embedding,
        )

        return {
            "win_probability": self.win_prob_head(fused_vector),
            "next_ball_outcome": self.next_ball_head(fused_vector),
            "odds_mispricing": self.odds_mispricing_head(fused_vector),
        }

    @property
    def device(self):
        return next(self.parameters()).device 