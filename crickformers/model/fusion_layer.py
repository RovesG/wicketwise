# Purpose: Implements the fusion layer for the Crickformer model.
# Author: Shamus Rae, Last Modified: 2024-07-30

"""
This module contains the CrickformerFusionLayer, which takes the outputs
from the sequence encoder, context encoder, and KG attention module,
and fuses them into a single latent match state vector.
"""

from typing import List
import torch
from torch import nn

class CrickformerFusionLayer(nn.Module):
    """
    Fuses multiple input vectors into a single latent representation.

    This layer concatenates the sequence, context, and attended KG vectors,
    then processes the result through a series of dense layers with GELU
    activation and dropout to produce the final latent match state.
    """
    def __init__(
        self,
        sequence_dim: int,
        context_dim: int,
        kg_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            sequence_dim: Dimension of the sequence_vector.
            context_dim: Dimension of the context_vector.
            kg_dim: Dimension of the attended_kg_vector.
            hidden_dims: A list of dimensions for the intermediate dense layers.
            latent_dim: The dimension of the final output vector.
            dropout_rate: The dropout probability.
        """
        super().__init__()
        
        input_dim = sequence_dim + context_dim + kg_dim
        
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
            
        layers.append(nn.Linear(current_dim, latent_dim))
        
        self.fusion_network = nn.Sequential(*layers)

    def forward(
        self,
        sequence_vector: torch.Tensor,
        context_vector: torch.Tensor,
        attended_kg_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the fusion layer.

        Args:
            sequence_vector: Output from BallHistoryEncoder.
                             Shape: [batch_size, sequence_dim]
            context_vector: Output from StaticContextEncoder.
                            Shape: [batch_size, context_dim]
            attended_kg_vector: Output from MultiHeadGraphAttention.
                                Shape: [batch_size, kg_dim]

        Returns:
            The final latent match state vector.
            Shape: [batch_size, latent_dim]
        """
        # Concatenate all input vectors along the feature dimension
        # combined_vector shape: [batch_size, sequence_dim + context_dim + kg_dim]
        combined_vector = torch.cat(
            [sequence_vector, context_vector, attended_kg_vector],
            dim=1
        )
        
        # Pass through the fusion network
        latent_vector = self.fusion_network(combined_vector)
        
        return latent_vector 