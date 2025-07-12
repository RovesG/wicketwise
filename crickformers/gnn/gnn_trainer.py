# Purpose: Trains a GNN model to generate node embeddings.
# Author: Shamus Rae, Last Modified: 2024-07-30

"""
This module contains the CricketGNNTrainer class, which handles the
conversion of a NetworkX graph into a PyTorch Geometric Data object,
trains a GraphSAGE model on a mock task, and saves the resulting
node embeddings to a file.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx
import networkx as nx
from typing import Dict

class GraphSAGE(torch.nn.Module):
    """A simple GraphSAGE model for node embedding."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class CricketGNNTrainer:
    """
    A trainer for generating node embeddings from a cricket knowledge graph.
    """
    def __init__(self, graph: nx.DiGraph, embedding_dim: int = 64, learning_rate: float = 0.01):
        # Store the original node order before conversion
        self.node_ids = list(graph.nodes())
        
        self.data = from_networkx(graph)
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        # Create mock features and labels for a node classification task
        num_nodes = self.data.num_nodes
        self.data.x = torch.randn(num_nodes, 16) # Initial random features
        self.data.y = torch.randint(0, 5, (num_nodes,)) # Mock classes

        self.model = GraphSAGE(
            in_channels=self.data.num_node_features,
            hidden_channels=32,
            out_channels=embedding_dim
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, epochs: int):
        """
        Trains the GraphSAGE model for a specified number of epochs.
        """
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            # Use mock labels for a simple classification loss
            loss = F.cross_entropy(out, self.data.y)
            loss.backward()
            self.optimizer.step()

    def export_embeddings(self, output_path: str) -> Dict[int, torch.Tensor]:
        """
        Generates and saves the final node embeddings.

        Args:
            output_path: The path to save the embeddings file (e.g., 'embeddings.pt').

        Returns:
            A dictionary mapping node index to its embedding tensor.
        """
        self.model.eval()
        with torch.no_grad():
            final_embeddings = self.model(self.data.x, self.data.edge_index)
        
        embedding_dict = {
            node_id: emb for node_id, emb in zip(self.node_ids, final_embeddings)
        }
        
        torch.save(embedding_dict, output_path)
        return embedding_dict 