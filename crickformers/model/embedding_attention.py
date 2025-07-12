# Purpose: Implements a multi-head attention mechanism over GNN embeddings.
# Author: Shamus Rae, Last Modified: 2024-07-30

"""
This module contains the MultiHeadGraphAttention class, which allows a query
vector (e.g., from a context encoder) to attend over a set of graph-derived
embeddings (e.g., batter, bowler, venue).
"""

import torch
from torch import nn

class MultiHeadGraphAttention(nn.Module):
    """
    Performs multi-head attention on a set of GNN embeddings.

    This module takes a query vector and uses it to attend over three
    distinct key-value pairs derived from batter, bowler type, and venue
    embeddings. The embeddings can have different initial dimensions and are
    first projected to a common dimension.
    """
    def __init__(
        self,
        query_dim: int,
        batter_dim: int,
        bowler_dim: int,
        venue_dim: int,
        nhead: int,
        attention_dim: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            query_dim: The dimension of the input query vector.
            batter_dim: The dimension of the batter embedding.
            bowler_dim: The dimension of the bowler type embedding.
            venue_dim: The dimension of the venue embedding.
            nhead: The number of attention heads.
            attention_dim: The common dimension for attention. Must be divisible by nhead.
            dropout: The dropout probability.
        """
        super().__init__()
        if attention_dim % nhead != 0:
            raise ValueError(f"attention_dim ({attention_dim}) must be divisible by nhead ({nhead})")

        self.query_proj = nn.Linear(query_dim, attention_dim)
        self.batter_proj = nn.Linear(batter_dim, attention_dim)
        self.bowler_proj = nn.Linear(bowler_dim, attention_dim)
        self.venue_proj = nn.Linear(venue_dim, attention_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=nhead,  # Corrected parameter name from nhead to num_heads
            dropout=dropout,
            batch_first=True  # Expects [batch, seq, feature]
        )

    def forward(
        self,
        query: torch.Tensor,
        batter_embedding: torch.Tensor,
        bowler_type_embedding: torch.Tensor,
        venue_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the attention mechanism.

        Args:
            query: The query vector. Shape: [batch_size, query_dim].
            batter_embedding: Shape: [batch_size, batter_dim].
            bowler_type_embedding: Shape: [batch_size, bowler_dim].
            venue_embedding: Shape: [batch_size, venue_dim].

        Returns:
            The attended output vector. Shape: [batch_size, attention_dim].
        """
        # Project query and embeddings to the common attention dimension
        # query_proj shape: [batch_size, attention_dim]
        query_proj = self.query_proj(query)

        # Proj shapes: [batch_size, attention_dim]
        batter_proj = self.batter_proj(batter_embedding)
        bowler_proj = self.bowler_proj(bowler_type_embedding)
        venue_proj = self.venue_proj(venue_embedding)

        # Stack embeddings to create the key-value memory bank.
        # This treats the 3 embeddings as a sequence of length 3.
        # kv_memory shape: [batch_size, 3, attention_dim]
        kv_memory = torch.stack([batter_proj, bowler_proj, venue_proj], dim=1)

        # The attention layer expects the query as [batch_size, seq_len, dim].
        # Since we have one query per batch item, we unsqueeze to add a sequence length of 1.
        # query_unsqueezed shape: [batch_size, 1, attention_dim]
        query_unsqueezed = query_proj.unsqueeze(1)

        # Perform attention. The query attends to the key-value memory.
        # attn_output shape: [batch_size, 1, attention_dim]
        attn_output, _ = self.attention(
            query=query_unsqueezed,
            key=kv_memory,
            value=kv_memory,
            need_weights=False
        )

        # Squeeze the output to remove the sequence length of 1.
        # final_output shape: [batch_size, attention_dim]
        return attn_output.squeeze(1) 