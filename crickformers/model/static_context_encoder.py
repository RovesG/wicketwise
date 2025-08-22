# Purpose: Implements an encoder for static, per-ball contextual features.
# Author: Shamus Rae, Last Modified: 2024-07-30

"""
This module contains the StaticContextEncoder, which processes the various
non-sequential features for a single ball (numerical, categorical, video)
and encodes them into a single contextual vector.
"""

import torch
from torch import nn
from typing import List, Dict, Optional

class StaticContextEncoder(nn.Module):
    """
    Encodes numerical, categorical, and video features for a single ball.
    It uses embedding layers for categorical features and an MLP to fuse
    all features into a final context vector.
    """
    def __init__(
        self,
        numeric_dim: int,
        categorical_vocab_sizes: Dict[str, int],
        categorical_embedding_dims: Dict[str, int],
        video_dim: int,
        hidden_dims: List[int],
        context_dim: int,
        weather_dim: int = 6,  # temp, humidity, wind_speed, wind_dir, precip, precip_prob
        venue_coord_dim: int = 2,  # latitude, longitude
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            numeric_dim: Dimension of the numerical features vector.
            categorical_vocab_sizes: A dictionary mapping feature names to their vocabulary size.
            categorical_embedding_dims: A dictionary mapping feature names to their embedding dim.
            video_dim: Dimension of the video signal vector.
            hidden_dims: List of hidden layer dimensions for the MLP.
            context_dim: The dimension of the final output context vector.
            dropout_rate: The dropout probability.
        """
        super().__init__()

        self.embedding_layers = nn.ModuleDict({
            feature: nn.Embedding(num_embeddings, embed_dim)
            for (feature, num_embeddings), embed_dim in zip(
                categorical_vocab_sizes.items(), categorical_embedding_dims.values()
            )
        })

        total_embedding_dim = sum(categorical_embedding_dims.values())
        
        # Enhanced feature encoders for weather and venue data
        self.weather_encoder = nn.Sequential(
            nn.Linear(weather_dim, weather_dim * 2),
            nn.ReLU(),
            nn.Linear(weather_dim * 2, weather_dim)
        ) if weather_dim > 0 else None
        
        self.venue_coord_encoder = nn.Sequential(
            nn.Linear(venue_coord_dim, venue_coord_dim * 4),
            nn.ReLU(),
            nn.Linear(venue_coord_dim * 4, venue_coord_dim * 2)
        ) if venue_coord_dim > 0 else None
        
        # Calculate input dimension including weather and venue features
        weather_encoded_dim = weather_dim if weather_dim > 0 else 0
        venue_encoded_dim = venue_coord_dim * 2 if venue_coord_dim > 0 else 0
        
        mlp_input_dim = numeric_dim + total_embedding_dim + video_dim + weather_encoded_dim + venue_encoded_dim
        
        layers = []
        current_dim = mlp_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, context_dim))
        self.encoder_mlp = nn.Sequential(*layers)

    def forward(
        self,
        numeric_features: torch.Tensor,
        categorical_features: torch.Tensor,
        video_features: torch.Tensor,
        video_mask: torch.Tensor,
        weather_features: Optional[torch.Tensor] = None,
        venue_coordinates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the context encoder.

        Args:
            numeric_features: Tensor of numerical features.
                              Shape: [batch_size, numeric_dim].
            categorical_features: Tensor of categorical feature indices.
                                  Shape: [batch_size, num_categorical_features].
            video_features: Tensor of video signal features.
                            Shape: [batch_size, video_dim].
            video_mask: A mask indicating if video features are present.
                        Shape: [batch_size, 1].

        Returns:
            The context vector. Shape: [batch_size, context_dim].
        """
        embedded_cats = [
            embedding_layer(categorical_features[:, i])
            for i, embedding_layer in enumerate(self.embedding_layers.values())
        ]
        
        all_embeddings = torch.cat(embedded_cats, dim=1)
        
        masked_video_features = video_features * video_mask
        
        # Prepare feature list for concatenation
        feature_list = [numeric_features, all_embeddings, masked_video_features]
        
        # Add weather features if available
        if weather_features is not None and self.weather_encoder is not None:
            weather_encoded = self.weather_encoder(weather_features)
            feature_list.append(weather_encoded)
        
        # Add venue coordinate features if available
        if venue_coordinates is not None and self.venue_coord_encoder is not None:
            venue_encoded = self.venue_coord_encoder(venue_coordinates)
            feature_list.append(venue_encoded)
        
        combined_features = torch.cat(feature_list, dim=1)
        
        context_vector = self.encoder_mlp(combined_features)
        
        return context_vector 