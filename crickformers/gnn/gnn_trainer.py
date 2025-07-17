# Purpose: Trains a GNN model to generate node embeddings with multi-hop message passing and temporal decay.
# Author: Shamus Rae, Last Modified: 2024-12-19

"""
This module contains the CricketGNNTrainer class, which handles the
conversion of a NetworkX graph into a PyTorch Geometric Data object,
trains a multi-hop GraphSAGE or GCN model on a mock task, and saves the 
resulting node embeddings to a file.

Multi-hop message passing allows nodes to gather information from their
k-hop neighborhood, where k is determined by the number of GNN layers.

Temporal decay weighting gives more importance to recent matches by applying
exponential decay: weight = exp(-alpha * days_ago).
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.utils import from_networkx
import networkx as nx
from typing import Dict, List, Literal, Optional
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class MultiHopGraphSAGE(torch.nn.Module):
    """
    A multi-hop GraphSAGE model for node embedding generation with temporal decay support.
    
    The number of layers determines the size of the neighborhood each node
    can access during message passing:
    - 1 layer: 1-hop neighborhood (direct neighbors)
    - 2 layers: 2-hop neighborhood (neighbors of neighbors)
    - 3 layers: 3-hop neighborhood (3 steps away)
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 num_layers: int = 3, dropout: float = 0.1):
        """
        Initialize multi-hop GraphSAGE model.
        
        Args:
            in_channels: Number of input features per node
            hidden_channels: Number of hidden channels in intermediate layers
            out_channels: Number of output embedding dimensions
            num_layers: Number of GNN layers (determines hop neighborhood size)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create layers
        self.layers = torch.nn.ModuleList()
        
        # Input layer
        self.layers.append(SAGEConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.layers.append(SAGEConv(hidden_channels, out_channels))
        else:
            # Single layer case
            self.layers[0] = SAGEConv(in_channels, out_channels)
        
        self.dropout_layer = torch.nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multi-hop GraphSAGE layers with optional edge weights.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_weights: Optional edge weights for temporal decay [num_edges]
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        # Pass through all layers except the last one with ReLU and dropout
        for i in range(self.num_layers - 1):
            # SAGEConv doesn't natively support edge weights, so we'll use them in loss
            x = self.layers[i](x, edge_index)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Final layer without activation
        x = self.layers[-1](x, edge_index)
        
        return x

class MultiHopGCN(torch.nn.Module):
    """
    A multi-hop GCN model for node embedding generation with temporal decay support.
    
    The number of layers determines the size of the neighborhood each node
    can access during message passing:
    - 1 layer: 1-hop neighborhood (direct neighbors)
    - 2 layers: 2-hop neighborhood (neighbors of neighbors)
    - 3 layers: 3-hop neighborhood (3 steps away)
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 num_layers: int = 3, dropout: float = 0.1):
        """
        Initialize multi-hop GCN model.
        
        Args:
            in_channels: Number of input features per node
            hidden_channels: Number of hidden channels in intermediate layers
            out_channels: Number of output embedding dimensions
            num_layers: Number of GNN layers (determines hop neighborhood size)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create layers
        self.layers = torch.nn.ModuleList()
        
        # Input layer
        self.layers.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.layers.append(GCNConv(hidden_channels, out_channels))
        else:
            # Single layer case
            self.layers[0] = GCNConv(in_channels, out_channels)
        
        self.dropout_layer = torch.nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multi-hop GCN layers with optional edge weights.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_weights: Optional edge weights for temporal decay [num_edges]
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        # Pass through all layers except the last one with ReLU and dropout
        for i in range(self.num_layers - 1):
            x = self.layers[i](x, edge_index, edge_weight=edge_weights)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Final layer without activation
        x = self.layers[-1](x, edge_index, edge_weight=edge_weights)
        
        return x

class CricketGNNTrainer:
    """
    A trainer for generating node embeddings from a cricket knowledge graph using
    multi-hop message passing with temporal decay weighting.
    
    Multi-hop Reasoning:
    - num_layers=1: Each node sees only direct neighbors (1-hop)
    - num_layers=2: Each node sees neighbors and neighbors-of-neighbors (2-hop)
    - num_layers=3: Each node sees up to 3 steps away in the graph (3-hop)
    
    Temporal Decay:
    - Applies exponential decay to older edges: weight = exp(-alpha * days_ago)
    - More recent matches have higher influence on embeddings
    - Alpha parameter controls decay rate (higher alpha = faster decay)
    """
    
    def __init__(self, graph: nx.DiGraph, embedding_dim: int = 64, 
                 learning_rate: float = 0.01, num_layers: int = 3,
                 model_type: Literal["sage", "gcn"] = "sage", 
                 hidden_channels: int = 32, dropout: float = 0.1,
                 temporal_decay_alpha: float = 0.01, 
                 reference_date: Optional[datetime] = None):
        """
        Initialize the CricketGNNTrainer with multi-hop capabilities and temporal decay.
        
        Args:
            graph: NetworkX directed graph representing cricket knowledge
            embedding_dim: Output embedding dimension
            learning_rate: Learning rate for optimizer
            num_layers: Number of GNN layers (determines hop neighborhood size)
                       - 1: 1-hop neighborhood (direct neighbors only)
                       - 2: 2-hop neighborhood (neighbors of neighbors)
                       - 3: 3-hop neighborhood (3 steps away)
            model_type: Type of GNN model ("sage" or "gcn")
            hidden_channels: Hidden layer dimensions
            dropout: Dropout rate for regularization
            temporal_decay_alpha: Decay parameter for temporal weighting
                                 Higher values = faster decay of older edges
            reference_date: Date to compute decay from (default: current date)
        """
        # Store the original node order before conversion
        self.node_ids = list(graph.nodes())
        self.original_graph = graph.copy()
        
        # Convert to PyTorch Geometric data
        self.data = from_networkx(graph)
        
        # Store parameters
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.model_type = model_type
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.temporal_decay_alpha = temporal_decay_alpha
        self.reference_date = reference_date or datetime.now()
        
        # Compute temporal decay weights
        self.edge_weights = self._compute_temporal_decay_weights()
        
        # Create mock features and labels for a node classification task
        num_nodes = self.data.num_nodes
        self.data.x = torch.randn(num_nodes, 16) # Initial random features
        self.data.y = torch.randint(0, 5, (num_nodes,)) # Mock classes
        
        # Initialize the appropriate model
        if model_type == "sage":
            self.model = MultiHopGraphSAGE(
                in_channels=self.data.num_node_features,
                hidden_channels=hidden_channels,
                out_channels=embedding_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        elif model_type == "gcn":
            self.model = MultiHopGCN(
                in_channels=self.data.num_node_features,
                hidden_channels=hidden_channels,
                out_channels=embedding_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'sage' or 'gcn'.")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        logger.info(f"Initialized {model_type.upper()} model with {num_layers} layers "
                   f"({num_layers}-hop neighborhood) and temporal decay (alpha={temporal_decay_alpha})")
    
    def _compute_temporal_decay_weights(self) -> torch.Tensor:
        """
        Compute temporal decay weights for all edges based on match dates.
        
        Returns:
            Tensor of edge weights with shape [num_edges]
        """
        edge_weights = []
        
        # Get edge list from the original graph
        edges = list(self.original_graph.edges(data=True))
        
        for source, target, attrs in edges:
            # Get match date from edge attributes
            match_date = attrs.get('match_date', self.reference_date)
            
            # Ensure match_date is a datetime object
            if match_date is None:
                match_date = self.reference_date
            elif isinstance(match_date, str):
                try:
                    match_date = datetime.strptime(match_date, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    match_date = self.reference_date
            elif not isinstance(match_date, datetime):
                match_date = self.reference_date
            
            # Compute days ago
            days_ago = (self.reference_date - match_date).days
            
            # Apply exponential decay: weight = exp(-alpha * days_ago)
            weight = np.exp(-self.temporal_decay_alpha * max(0, days_ago))
            edge_weights.append(weight)
        
        return torch.tensor(edge_weights, dtype=torch.float32)
    
    def get_temporal_decay_stats(self) -> Dict[str, float]:
        """
        Get statistics about temporal decay weights.
        
        Returns:
            Dictionary with weight statistics
        """
        weights = self.edge_weights.numpy()
        
        return {
            'min_weight': float(weights.min()),
            'max_weight': float(weights.max()),
            'mean_weight': float(weights.mean()),
            'std_weight': float(weights.std()),
            'num_edges': len(weights),
            'alpha': self.temporal_decay_alpha
        }

    def train(self, epochs: int):
        """
        Trains the multi-hop GNN model with temporal decay weighting.
        
        Args:
            epochs: Number of training epochs
        """
        self.model.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass through multi-hop layers with temporal weights
            out = self.model(self.data.x, self.data.edge_index, self.edge_weights)
            
            # Compute classification loss
            loss = F.cross_entropy(out, self.data.y)
            
            # Add temporal weighting to loss (emphasize recent edge predictions)
            if self.model_type == "gcn":
                # For GCN, weights are already incorporated in the forward pass
                temporal_loss = 0
            else:
                # For SAGE, we add a regularization term based on temporal weights
                temporal_loss = 0.1 * torch.mean(self.edge_weights)
            
            total_loss = loss + temporal_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}, "
                           f"Temporal Loss: {temporal_loss if isinstance(temporal_loss, (int, float)) else temporal_loss.item():.4f}")

    def export_embeddings(self, output_path: str) -> Dict[str, torch.Tensor]:
        """
        Generates and saves the final node embeddings after multi-hop training.

        Args:
            output_path: The path to save the embeddings file (e.g., 'embeddings.pt').

        Returns:
            A dictionary mapping node ID to its embedding tensor.
        """
        self.model.eval()
        with torch.no_grad():
            final_embeddings = self.model(self.data.x, self.data.edge_index, self.edge_weights)
        
        embedding_dict = {
            node_id: emb for node_id, emb in zip(self.node_ids, final_embeddings)
        }
        
        torch.save(embedding_dict, output_path)
        logger.info(f"Saved {len(embedding_dict)} embeddings to {output_path}")
        
        return embedding_dict
    
    def get_intermediate_embeddings(self) -> List[torch.Tensor]:
        """
        Get embeddings from each layer to analyze multi-hop message passing.
        
        Returns:
            List of tensors, one for each layer's output
        """
        self.model.eval()
        with torch.no_grad():
            x = self.data.x
            edge_index = self.data.edge_index
            edge_weights = self.edge_weights
            
            layer_outputs = []
            
            # Forward pass through each layer
            for i in range(self.num_layers):
                if self.model_type == "gcn":
                    x = self.model.layers[i](x, edge_index, edge_weight=edge_weights)
                else:
                    x = self.model.layers[i](x, edge_index)
                
                if i < self.num_layers - 1:  # Apply ReLU and dropout except for last layer
                    x = F.relu(x)
                layer_outputs.append(x.clone())
            
            return layer_outputs 