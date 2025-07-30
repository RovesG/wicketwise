# Purpose: Learnable temporal encoding for cricket knowledge graphs
# Author: WicketWise Team, Last Modified: 2024-07-19

"""
This module implements learnable temporal encoding using sinusoidal and linear
components to encode temporal recency in cricket match data. This replaces
simple exponential decay with a more sophisticated learnable representation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def generate_temporal_embedding(days_ago: Union[int, torch.Tensor], 
                               dim: int = 8,
                               max_days: int = 365,
                               device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate temporal embedding using sinusoidal and linear components.
    
    This function creates a learnable temporal representation that combines:
    1. Sinusoidal encodings at different frequencies (like positional encoding)
    2. Linear components for direct temporal distance representation
    3. Normalized outputs in [0, 1] range
    
    Args:
        days_ago: Number of days since the event (int or tensor)
        dim: Dimension of the temporal embedding (default: 8)
        max_days: Maximum number of days to normalize against (default: 365)
        device: Device to place the tensor on (default: None, infers from input)
        
    Returns:
        Temporal embedding tensor of shape [dim] or [batch_size, dim]
        
    Example:
        >>> embedding = generate_temporal_embedding(30, dim=8)
        >>> print(embedding.shape)  # torch.Size([8])
        >>> print(embedding.min(), embedding.max())  # Values in [0, 1]
    """
    # Convert input to tensor if needed
    if isinstance(days_ago, int):
        days_ago = torch.tensor([days_ago], dtype=torch.float32)
    elif isinstance(days_ago, (list, np.ndarray)):
        days_ago = torch.tensor(days_ago, dtype=torch.float32)
    
    # Ensure days_ago is float tensor
    days_ago = days_ago.float()
    
    # Move to specified device
    if device is not None:
        days_ago = days_ago.to(device)
    
    # Normalize days to [0, 1] range
    normalized_days = torch.clamp(days_ago / max_days, 0.0, 1.0)
    
    # Determine batch size
    batch_size = days_ago.shape[0] if len(days_ago.shape) > 0 else 1
    
    # Create embedding tensor
    if device is not None:
        embedding = torch.zeros(batch_size, dim, device=device)
    else:
        embedding = torch.zeros(batch_size, dim)
    
    # Split dimensions between sinusoidal and linear components
    sin_dim = dim // 2
    linear_dim = dim - sin_dim
    
    # 1. Sinusoidal components (first half of dimensions)
    if sin_dim > 0:
        # Create different frequencies for different dimensions
        frequencies = torch.arange(sin_dim, dtype=torch.float32)
        if device is not None:
            frequencies = frequencies.to(device)
        
        # Use different frequency scales with more variation
        frequencies = 1.0 / (10000.0 ** (frequencies / sin_dim)) * (frequencies + 1.0)
        
        # Calculate sinusoidal encodings
        angles = normalized_days.unsqueeze(-1) * frequencies.unsqueeze(0)
        
        # Create sinusoidal components without in-place operations
        sin_components = []
        for i in range(sin_dim):
            if i % 2 == 0:
                sin_components.append(torch.sin(angles[:, i // 2]))
            else:
                sin_components.append(torch.cos(angles[:, i // 2]))
        
        # Assign sinusoidal components
        if sin_components:
            embedding[:, :sin_dim] = torch.stack(sin_components, dim=1)
    
    # 2. Linear components (second half of dimensions)
    if linear_dim > 0:
        # Create linear components without in-place operations
        linear_components = []
        
        for i in range(linear_dim):
            if i == 0:
                # Direct normalized temporal distance
                linear_components.append(normalized_days)
            elif i == 1:
                # Inverse temporal distance (recent = high, old = low)
                linear_components.append(1.0 - normalized_days)
            elif i == 2:
                # Quadratic recency bias (stronger for recent events)
                linear_components.append((1.0 - normalized_days) ** 2)
            elif i == 3:
                # Square root recency (gentler decay)
                linear_components.append(torch.sqrt(torch.clamp(1.0 - normalized_days, min=0.0)))
            else:
                # Polynomial features of different orders with stronger bias
                order = (i - 4) % 4 + 2  # Start from order 2 for stronger differences
                linear_components.append((1.0 - normalized_days) ** order)
        
        # Assign linear components
        if linear_components:
            linear_tensor = torch.stack(linear_components, dim=1)
            embedding[:, sin_dim:] = linear_tensor
    
    # 3. Apply sigmoid normalization to ensure [0, 1] range without flattening
    # This preserves variation better than min-max normalization
    embedding = torch.sigmoid(embedding)
    
    # Return single embedding if input was scalar
    if batch_size == 1 and isinstance(days_ago.item(), (int, float)):
        return embedding.squeeze(0)
    
    return embedding


class LearnableTemporalEncoder(nn.Module):
    """
    Learnable temporal encoder module that can be trained end-to-end.
    
    This module combines fixed sinusoidal encodings with learnable linear
    transformations to create sophisticated temporal representations.
    """
    
    def __init__(self, 
                 dim: int = 8,
                 max_days: int = 365,
                 learnable_components: int = 4):
        """
        Initialize learnable temporal encoder.
        
        Args:
            dim: Output dimension of temporal embeddings
            max_days: Maximum number of days for normalization
            learnable_components: Number of learnable linear components
        """
        super().__init__()
        
        self.dim = dim
        self.max_days = max_days
        self.learnable_components = learnable_components
        
        # Fixed sinusoidal component dimensions
        self.sin_dim = dim // 2
        self.linear_dim = dim - self.sin_dim
        
        # Learnable linear transformations
        if learnable_components > 0:
            self.linear_layers = nn.ModuleList([
                nn.Linear(1, 1, bias=True) for _ in range(min(learnable_components, self.linear_dim))
            ])
        
        # Learnable normalization parameters
        self.norm_scale = nn.Parameter(torch.ones(dim))
        self.norm_bias = nn.Parameter(torch.zeros(dim))
        
        logger.info(f"Initialized LearnableTemporalEncoder: dim={dim}, max_days={max_days}")
    
    def forward(self, days_ago: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through learnable temporal encoder.
        
        Args:
            days_ago: Tensor of days since events [batch_size] or [batch_size, 1]
            
        Returns:
            Temporal embeddings [batch_size, dim]
        """
        # Ensure proper shape
        if len(days_ago.shape) == 1:
            days_ago = days_ago.unsqueeze(-1)
        
        batch_size = days_ago.shape[0]
        
        # Generate base temporal embedding
        base_embedding = generate_temporal_embedding(
            days_ago.squeeze(-1), 
            dim=self.dim, 
            max_days=self.max_days,
            device=days_ago.device
        )
        
        # Ensure batch dimension
        if len(base_embedding.shape) == 1:
            base_embedding = base_embedding.unsqueeze(0)
        
        # Apply learnable transformations to linear components
        if hasattr(self, 'linear_layers') and len(self.linear_layers) > 0:
            # Normalize input for learnable components
            normalized_days = torch.clamp(days_ago / self.max_days, 0.0, 1.0)
            
            # Create new tensor to avoid in-place operations
            enhanced_embedding = base_embedding.clone()
            
            for i, linear_layer in enumerate(self.linear_layers):
                if self.sin_dim + i < self.dim:
                    # Apply learnable transformation
                    learned_component = linear_layer(normalized_days).squeeze(-1)
                    # Combine with base embedding (non-in-place)
                    enhanced_embedding[:, self.sin_dim + i] = (
                        0.7 * base_embedding[:, self.sin_dim + i] + 
                        0.3 * torch.sigmoid(learned_component)
                    )
            
            base_embedding = enhanced_embedding
        
        # Apply learnable normalization
        output = base_embedding * self.norm_scale + self.norm_bias
        
        # Final normalization to [0, 1]
        output = torch.sigmoid(output)
        
        return output
    
    def get_temporal_stats(self, days_range: torch.Tensor) -> dict:
        """
        Get statistics about temporal encodings over a range of days.
        
        Args:
            days_range: Range of days to analyze
            
        Returns:
            Dictionary with encoding statistics
        """
        with torch.no_grad():
            embeddings = self.forward(days_range)
            
            return {
                'mean': embeddings.mean(dim=0).cpu().numpy(),
                'std': embeddings.std(dim=0).cpu().numpy(),
                'min': embeddings.min(dim=0)[0].cpu().numpy(),
                'max': embeddings.max(dim=0)[0].cpu().numpy(),
                'range': (embeddings.max(dim=0)[0] - embeddings.min(dim=0)[0]).cpu().numpy()
            }


def create_temporal_edge_attributes(match_dates: torch.Tensor,
                                   reference_date: torch.Tensor,
                                   embedding_dim: int = 8,
                                   base_edge_attrs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Create edge attributes that include temporal embeddings.
    
    Args:
        match_dates: Tensor of match dates [num_edges]
        reference_date: Reference date for computing days_ago
        embedding_dim: Dimension of temporal embeddings
        base_edge_attrs: Optional base edge attributes to concatenate with
        
    Returns:
        Edge attributes with temporal embeddings [num_edges, total_dim]
    """
    # Calculate days ago
    days_ago = (reference_date - match_dates).float()
    
    # Generate temporal embeddings
    temporal_embeddings = generate_temporal_embedding(
        days_ago, 
        dim=embedding_dim,
        device=match_dates.device
    )
    
    # Ensure batch dimension
    if len(temporal_embeddings.shape) == 1:
        temporal_embeddings = temporal_embeddings.unsqueeze(0)
    
    # Concatenate with base attributes if provided
    if base_edge_attrs is not None:
        if len(base_edge_attrs.shape) == 1:
            base_edge_attrs = base_edge_attrs.unsqueeze(0)
        
        # Ensure same batch size
        if base_edge_attrs.shape[0] != temporal_embeddings.shape[0]:
            base_edge_attrs = base_edge_attrs.expand(temporal_embeddings.shape[0], -1)
        
        edge_attrs = torch.cat([base_edge_attrs, temporal_embeddings], dim=-1)
    else:
        edge_attrs = temporal_embeddings
    
    return edge_attrs