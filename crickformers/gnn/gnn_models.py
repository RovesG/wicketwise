# Purpose: Advanced GNN model implementations for cricket knowledge graphs
# Author: WicketWise Team, Last Modified: 2024-07-19

"""
This module contains advanced GNN model architectures for cricket analytics,
including Graph Attention Networks (GAT) with multi-hop message passing.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MultiHopGATv2(torch.nn.Module):
    """
    Multi-hop Graph Attention Network v2 for cricket player embeddings.
    
    Uses GATv2Conv layers with multi-head attention to learn sophisticated
    representations of cricket players, venues, and their relationships.
    
    Architecture:
    - 3 layers of GATv2Conv with 4 attention heads each
    - Hidden dimension: 64
    - Output dimension: 128
    - Mean pooling across attention heads
    - ReLU activation and dropout between layers
    """
    
    def __init__(self, 
                 in_channels: int = 9,
                 hidden_channels: int = 64,
                 out_channels: int = 128,
                 num_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize Multi-Hop GATv2 model.
        
        Args:
            in_channels: Number of input features per node (default: 9)
            hidden_channels: Number of hidden channels (default: 64)
            out_channels: Number of output embedding dimensions (default: 128)
            num_layers: Number of GATv2 layers (default: 3)
            heads: Number of attention heads per layer (default: 4)
            dropout: Dropout rate for regularization (default: 0.1)
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        
        # Store dimensions for validation
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # Create GATv2 layers
        self.layers = torch.nn.ModuleList()
        
        # First layer: input -> hidden with multi-head attention
        self.layers.append(
            GATv2Conv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                heads=heads,
                dropout=dropout,
                concat=False,  # Mean pooling across heads
                edge_dim=None  # We'll handle edge attributes separately if needed
            )
        )
        
        # Hidden layers: hidden -> hidden with multi-head attention
        for _ in range(num_layers - 2):
            self.layers.append(
                GATv2Conv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    concat=False,  # Mean pooling across heads
                    edge_dim=None
                )
            )
        
        # Final layer: hidden -> output with multi-head attention
        if num_layers > 1:
            self.layers.append(
                GATv2Conv(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    heads=heads,
                    dropout=dropout,
                    concat=False,  # Mean pooling across heads
                    edge_dim=None
                )
            )
        else:
            # Single layer case: input -> output
            self.layers[0] = GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                dropout=dropout,
                concat=False,
                edge_dim=None
            )
        
        # Dropout layer
        self.dropout_layer = torch.nn.Dropout(dropout)
        
        logger.info(f"Initialized MultiHopGATv2: {in_channels}D -> {hidden_channels}D -> {out_channels}D")
        logger.info(f"Architecture: {num_layers} layers, {heads} heads, {dropout} dropout")
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Multi-Hop GATv2 layers.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Optional edge attributes [num_edges, edge_dim] (currently unused)
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        # Validate input dimensions
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input features, got {x.size(1)}")
        
        # Pass through all layers except the last one with ReLU and dropout
        for i in range(self.num_layers - 1):
            # Apply GATv2 layer with attention
            x = self.layers[i](x, edge_index)
            
            # Apply ReLU activation
            x = F.relu(x)
            
            # Apply dropout for regularization
            x = self.dropout_layer(x)
        
        # Final layer without activation (raw embeddings)
        x = self.layers[-1](x, edge_index)
        
        return x
    
    def get_attention_weights(self, 
                            x: torch.Tensor, 
                            edge_index: torch.Tensor,
                            layer_idx: int = 0) -> torch.Tensor:
        """
        Extract attention weights from a specific layer for analysis.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            layer_idx: Which layer to extract attention from (default: 0)
            
        Returns:
            Attention weights [num_edges, heads]
        """
        if layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range (0-{self.num_layers-1})")
        
        # Forward pass up to the specified layer
        current_x = x
        for i in range(layer_idx):
            current_x = self.layers[i](current_x, edge_index)
            if i < self.num_layers - 1:  # Don't apply activation/dropout on final layer
                current_x = F.relu(current_x)
                current_x = self.dropout_layer(current_x)
        
        # Get attention weights from the specified layer
        # Note: GATv2Conv returns (output, attention_weights) when return_attention_weights=True
        # For this implementation, we'll need to modify the layer call
        layer = self.layers[layer_idx]
        
        # Temporarily enable attention weight return
        original_return_attention = getattr(layer, 'return_attention_weights', False)
        layer.return_attention_weights = True
        
        try:
            _, attention_weights = layer(current_x, edge_index, return_attention_weights=True)
            return attention_weights
        except:
            # Fallback if attention weights not available
            logger.warning(f"Could not extract attention weights from layer {layer_idx}")
            return torch.zeros(edge_index.size(1), self.heads)
        finally:
            # Restore original setting
            layer.return_attention_weights = original_return_attention
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"MultiHopGATv2(in_channels={self.in_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"out_channels={self.out_channels}, "
                f"num_layers={self.num_layers}, "
                f"heads={self.heads}, "
                f"dropout={self.dropout})")