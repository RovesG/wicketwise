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
            query_dim=128,  # From static context encoder
            batter_dim=128,
            bowler_dim=128,
            venue_dim=64,
            nhead=4,
            attention_dim=128,
            dropout=0.1
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
        # Extract inputs
        recent_ball_history = inputs.get("recent_ball_history")
        numeric_features = inputs.get("numeric_features")
        categorical_features = inputs.get("categorical_features")
        video_features = inputs.get("video_features")
        video_mask = inputs.get("video_mask")
        gnn_embeddings = inputs.get("gnn_embeddings")
        weather_features = inputs.get("weather_features")
        venue_coordinates = inputs.get("venue_coordinates")
        
        # Handle missing inputs with appropriate defaults
        batch_size = recent_ball_history.shape[0] if recent_ball_history is not None else 1
        device = self.device
        
        if recent_ball_history is None:
            recent_ball_history = torch.zeros(batch_size, 5, 64, device=device)  # Updated to 64 dimensions for memory efficiency
        if numeric_features is None:
            numeric_features = torch.zeros(batch_size, 15, device=device)
        if categorical_features is None:
            categorical_features = torch.zeros(batch_size, 4, dtype=torch.long, device=device)
        if video_features is None:
            video_features = torch.zeros(batch_size, 99, device=device)
        if video_mask is None:
            video_mask = torch.zeros(batch_size, 1, device=device)
        if gnn_embeddings is None:
            gnn_embeddings = torch.zeros(batch_size, 1, 320, device=device)  # 128+128+64

        # Forward pass through components
        sequence_embedding = self.sequence_encoder(recent_ball_history)
        
        static_embedding = self.static_context_encoder(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            video_features=video_features,
            video_mask=video_mask,
            weather_features=weather_features,
            venue_coordinates=venue_coordinates
        )
        
        # Split GNN embeddings: 128 + 128 + 64 + 64 = 384
        batter_emb = gnn_embeddings[:, 0, :128]
        bowler_emb = gnn_embeddings[:, 0, 128:256]
        venue_emb = gnn_embeddings[:, 0, 256:320]
        # Note: ignoring edge embeddings for now (320:384)
        
        gnn_embedding = self.gnn_attention(
            query=static_embedding,
            batter_embedding=batter_emb,
            bowler_type_embedding=bowler_emb,
            venue_embedding=venue_emb
        )

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