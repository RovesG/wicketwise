# Purpose: Implements a Transformer-based sequence encoder for ball history.
# Author: Shamus Rae, Last Modified: 2024-07-30

"""
This module contains the BallHistoryEncoder, a PyTorch module that uses a
Transformer architecture to encode a sequence of recent ball events into a
single summary vector.
"""

import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    """
    Injects sinusoidal positional encoding. The input to the forward pass is
    expected to be of shape [seq_len, batch_size, embedding_dim].
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class BallHistoryEncoder(nn.Module):
    """
    A Transformer-based encoder for a fixed-size sequence of ball history.

    This module takes a batch of sequences, applies positional encoding,
    processes them through a series of TransformerEncoder layers, and
    outputs a single summary vector for each sequence by mean pooling.
    """
    def __init__(
        self,
        feature_dim: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            feature_dim: The number of features for each ball in the sequence.
            nhead: The number of heads in the multi-head attention models.
            num_encoder_layers: The number of sub-encoder-layers in the encoder.
            dim_feedforward: The dimension of the feedforward network model.
            dropout: The dropout value.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.pos_encoder = PositionalEncoding(feature_dim, dropout, max_len=5)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important: ensures input is [batch, seq, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the BallHistoryEncoder.

        Args:
            src: Input tensor of ball history.
                 Shape: [batch_size, 5, feature_dim]

        Returns:
            A summary vector for each sequence in the batch.
            Shape: [batch_size, feature_dim]
        """
        # Note: TransformerEncoderLayer with batch_first=True expects [batch, seq, feature]
        # We need to add positional encoding. The created pos_encoder expects
        # [seq, batch, feature], so we'll adjust it for batch_first.
        # Permute self.pos_encoder.pe from [seq, 1, feature] to [1, seq, feature]
        # so it can be broadcasted and added to the input.
        pos_encoded_src = src + self.pos_encoder.pe.permute(1, 0, 2)
        
        # Pass through transformer
        output = self.transformer_encoder(pos_encoded_src)
        # output shape: [batch_size, 5, feature_dim]

        # Mean pool over the sequence dimension
        pooled_output = output.mean(dim=1)
        # pooled_output shape: [batch_size, feature_dim]

        return pooled_output 