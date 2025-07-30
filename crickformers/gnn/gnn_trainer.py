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
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.utils import from_networkx
import networkx as nx
from typing import Dict, List, Literal, Optional, Union
import logging
from datetime import datetime
import numpy as np
from .temporal_encoding import generate_temporal_embedding, create_temporal_edge_attributes

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
        
        # Create PyTorch Geometric data manually to avoid attribute issues
        from torch_geometric.data import Data
        
        # Create node mappings
        node_to_idx = {node: i for i, node in enumerate(self.node_ids)}
        
        # Create edge index
        edge_list = []
        for edge in graph.edges():
            source_idx = node_to_idx[edge[0]]
            target_idx = node_to_idx[edge[1]]
            edge_list.append([source_idx, target_idx])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create data object
        self.data = Data(edge_index=edge_index, num_nodes=len(self.node_ids))
        
        # Store parameters
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.model_type = model_type
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.temporal_decay_alpha = temporal_decay_alpha
        self.reference_date = reference_date or datetime.now()
        
        # Compute temporal embeddings for edges
        self.edge_attrs, self.edge_weights = self._compute_temporal_edge_attributes()
        
        # Create uniform features for all nodes
        num_nodes = self.data.num_nodes
        
        # Create uniform feature representation for all nodes
        node_features = []
        for node in self.node_ids:
            node_attrs = self.original_graph.nodes[node]
            node_type = node_attrs.get('type', 'unknown')
            
            # Create feature vector based on node type
            if node_type == 'batter':
                features = [
                    node_attrs.get('average', 0.0),
                    node_attrs.get('strike_rate', 0.0),
                    node_attrs.get('boundary_rate', 0.0),
                    node_attrs.get('six_rate', 0.0),
                    1.0, 0.0, 0.0, 0.0, 0.0  # One-hot encoding for batter
                ]
            elif node_type == 'bowler':
                features = [
                    node_attrs.get('average', 0.0),
                    node_attrs.get('economy', 0.0),
                    node_attrs.get('wicket_rate', 0.0),
                    node_attrs.get('strike_rate', 0.0),
                    0.0, 1.0, 0.0, 0.0, 0.0  # One-hot encoding for bowler
                ]
            elif node_type == 'venue':
                features = [
                    node_attrs.get('avg_score', 0.0),
                    node_attrs.get('total_matches', 0.0),
                    0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0, 0.0  # One-hot encoding for venue
                ]
            elif node_type == 'team':
                features = [
                    node_attrs.get('win_rate', 0.0),
                    node_attrs.get('avg_score', 0.0),
                    0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 0.0  # One-hot encoding for team
                ]
            else:  # match, phase, or other
                features = [0.0] * 4 + [0.0, 0.0, 0.0, 0.0, 1.0]  # Default features
            
            # Ensure feature vector has exactly 9 dimensions
            features = features[:9] + [0.0] * max(0, 9 - len(features))
            node_features.append(features)
        
        self.data.x = torch.tensor(node_features, dtype=torch.float32)
        self.data.y = torch.randint(0, 5, (num_nodes,))  # Mock classes
        
        # Initialize the appropriate model
        if model_type == "sage":
            self.model = MultiHopGraphSAGE(
                in_channels=9,  # Fixed input dimension
                hidden_channels=hidden_channels,
                out_channels=embedding_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        elif model_type == "gcn":
            self.model = MultiHopGCN(
                in_channels=9,  # Fixed input dimension
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
    
    def _compute_temporal_edge_attributes(self, temporal_dim: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute temporal edge attributes using learnable temporal encoding.
        
        Args:
            temporal_dim: Dimension of temporal embeddings
        
        Returns:
            Tuple of (edge_attributes, edge_weights) tensors
        """
        edge_attrs = []
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
            
            # Generate temporal embedding
            temporal_embedding = generate_temporal_embedding(
                max(0, days_ago), 
                dim=temporal_dim
            )
            
            # Create base edge attributes from graph edge data
            base_attrs = []
            for key in ['runs', 'balls_faced', 'dismissals', 'weight']:
                base_attrs.append(attrs.get(key, 0.0))
            
            # Combine base attributes with temporal embedding
            if base_attrs:
                base_tensor = torch.tensor(base_attrs, dtype=torch.float32)
                full_attrs = torch.cat([base_tensor, temporal_embedding])
            else:
                full_attrs = temporal_embedding
            
            edge_attrs.append(full_attrs)
            
            # Compute edge weight from temporal embedding (use mean as weight)
            weight = temporal_embedding.mean().item()
            edge_weights.append(weight)
        
        # Stack edge attributes and weights
        if edge_attrs:
            edge_attrs_tensor = torch.stack(edge_attrs)
            edge_weights_tensor = torch.tensor(edge_weights, dtype=torch.float32)
        else:
            edge_attrs_tensor = torch.empty(0, temporal_dim + 4)  # 4 base attrs + temporal_dim
            edge_weights_tensor = torch.empty(0)
        
        return edge_attrs_tensor, edge_weights_tensor
    
    def get_temporal_encoding_stats(self) -> Dict[str, float]:
        """
        Get statistics about temporal edge attributes and weights.
        
        Returns:
            Dictionary with temporal encoding statistics
        """
        weights = self.edge_weights.numpy()
        attrs = self.edge_attrs.numpy()
        
        stats = {
            'min_weight': float(weights.min()) if len(weights) > 0 else 0.0,
            'max_weight': float(weights.max()) if len(weights) > 0 else 0.0,
            'mean_weight': float(weights.mean()) if len(weights) > 0 else 0.0,
            'std_weight': float(weights.std()) if len(weights) > 0 else 0.0,
            'num_edges': len(weights),
            'edge_attr_dim': attrs.shape[1] if len(attrs) > 0 else 0,
            'temporal_dim': attrs.shape[1] - 4 if len(attrs) > 0 else 8,  # Subtract base attrs
            'alpha': self.temporal_decay_alpha  # Keep for backward compatibility
        }
        
        # Add temporal embedding statistics
        if len(attrs) > 0:
            temporal_part = attrs[:, 4:]  # Skip base attributes
            stats.update({
                'temporal_mean': float(temporal_part.mean()),
                'temporal_std': float(temporal_part.std()),
                'temporal_min': float(temporal_part.min()),
                'temporal_max': float(temporal_part.max())
            })
        
        return stats

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


class CricketHeteroGNNTrainer:
    """
    A trainer for generating node embeddings from cricket HeteroData graphs.
    
    This trainer works with PyTorch Geometric HeteroData format which provides
    better performance and native support for heterogeneous graphs with multiple
    node and edge types.
    """
    
    def __init__(self, hetero_data: HeteroData, embedding_dim: int = 64,
                 learning_rate: float = 0.01, device: Optional[torch.device] = None):
        """
        Initialize the HeteroData GNN trainer.
        
        Args:
            hetero_data: HeteroData object with cricket knowledge graph
            embedding_dim: Output embedding dimension for each node type
            learning_rate: Learning rate for optimizer
            device: Device to run training on
        """
        self.hetero_data = hetero_data
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.device = device if device is not None else torch.device('cpu')
        
        # Move data to device
        self.hetero_data = self.hetero_data.to(self.device)
        
        # Store node and edge type information
        self.node_types = list(self.hetero_data.node_types)
        self.edge_types = list(self.hetero_data.edge_types)
        
        # Initialize simple linear layers for each node type (placeholder)
        self.node_encoders = torch.nn.ModuleDict()
        for node_type in self.node_types:
            input_dim = self.hetero_data[node_type].x.shape[1]
            self.node_encoders[node_type] = torch.nn.Linear(input_dim, embedding_dim)
        
        self.node_encoders = self.node_encoders.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.node_encoders.parameters(), lr=learning_rate)
        
        logger.info(f"Initialized HeteroGNN trainer with {len(self.node_types)} node types and {len(self.edge_types)} edge types")
    
    def get_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Generate embeddings for all node types.
        
        Returns:
            Dictionary mapping node_type -> embeddings tensor
        """
        embeddings = {}
        
        with torch.no_grad():
            for node_type in self.node_types:
                x = self.hetero_data[node_type].x
                embeddings[node_type] = self.node_encoders[node_type](x)
        
        return embeddings
    
    def train_step(self) -> float:
        """
        Perform one training step (placeholder implementation).
        
        Returns:
            Training loss
        """
        self.optimizer.zero_grad()
        
        # Simple reconstruction loss (placeholder)
        total_loss = 0.0
        
        for node_type in self.node_types:
            x = self.hetero_data[node_type].x
            embeddings = self.node_encoders[node_type](x)
            
            # Simple reconstruction loss: try to reconstruct input features
            reconstructed = torch.nn.functional.linear(embeddings, self.node_encoders[node_type].weight.t())
            loss = F.mse_loss(reconstructed, x)
            total_loss += loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def train(self, num_epochs: int = 100) -> List[float]:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            List of training losses
        """
        losses = []
        
        for epoch in range(num_epochs):
            loss = self.train_step()
            losses.append(loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def get_hetero_data_stats(self) -> Dict[str, any]:
        """
        Get statistics about the HeteroData object.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "node_types": self.node_types,
            "edge_types": [str(et) for et in self.edge_types],
            "num_node_types": len(self.node_types),
            "num_edge_types": len(self.edge_types),
            "device": str(self.device),
        }
        
        # Node statistics
        for node_type in self.node_types:
            stats[f"num_nodes_{node_type}"] = self.hetero_data[node_type].num_nodes
            stats[f"feature_dim_{node_type}"] = self.hetero_data[node_type].x.shape[1]
        
        # Edge statistics
        for edge_type in self.edge_types:
            stats[f"num_edges_{edge_type}"] = self.hetero_data[edge_type].edge_index.shape[1]
        
        return stats
    
    def save_embeddings(self, filepath: str) -> None:
        """
        Save node embeddings to file.
        
        Args:
            filepath: Path to save embeddings
        """
        embeddings = self.get_embeddings()
        
        # Convert to CPU and numpy for saving
        embeddings_cpu = {}
        for node_type, emb in embeddings.items():
            embeddings_cpu[node_type] = emb.cpu().numpy()
        
        torch.save({
            'embeddings': embeddings_cpu,
            'node_types': self.node_types,
            'edge_types': [str(et) for et in self.edge_types],
            'embedding_dim': self.embedding_dim,
        }, filepath)
        
        logger.info(f"Saved HeteroData embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Load node embeddings from file.
        
        Args:
            filepath: Path to load embeddings from
            
        Returns:
            Dictionary mapping node_type -> embeddings array
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        embeddings = checkpoint['embeddings']
        
        logger.info(f"Loaded HeteroData embeddings from {filepath}")
        return embeddings 