# Purpose: GNN explanation module using PyTorch Geometric's GNNExplainer
# Author: WicketWise Team, Last Modified: 2024-07-19

"""
This module provides interpretable explanations for GNN model predictions
using PyTorch Geometric's GNNExplainer. It highlights which neighbors and
edges most contribute to a player's embedding representation.

Features:
- Load trained GNN models and knowledge graphs
- Explain individual player node embeddings
- Identify top influential neighbors and edges
- Visual explanations with importance scores
- Export explanations to Graphviz format
- Support for different GNN architectures
"""

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.data import Data, HeteroData
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import pickle
import json
from dataclasses import dataclass
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import graphviz for visualization export
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    logger.warning("Graphviz not available. Install with: pip install graphviz")


@dataclass
class ExplanationResult:
    """Container for GNN explanation results."""
    target_node: Union[int, str]
    target_embedding: torch.Tensor
    node_importance: torch.Tensor
    edge_importance: torch.Tensor
    top_neighbors: List[Tuple[Union[int, str], float]]
    top_edges: List[Tuple[Tuple[Union[int, str], Union[int, str]], float]]
    explanation_subgraph: Optional[nx.Graph] = None
    metadata: Optional[Dict[str, Any]] = None


class GNNModelWrapper(torch.nn.Module):
    """Wrapper for different GNN model architectures."""
    
    def __init__(self, model_type: str = "GraphSAGE", input_dim: int = 64, 
                 hidden_dim: int = 128, output_dim: int = 64, num_layers: int = 2):
        """
        Initialize GNN model wrapper.
        
        Args:
            model_type: Type of GNN model ('GraphSAGE', 'GCN', 'GAT')
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of GNN layers
        """
        super().__init__()
        self.model_type = model_type
        self.num_layers = num_layers
        
        if model_type == "GraphSAGE":
            self.convs = torch.nn.ModuleList([
                SAGEConv(input_dim if i == 0 else hidden_dim, 
                        hidden_dim if i < num_layers - 1 else output_dim)
                for i in range(num_layers)
            ])
        elif model_type == "GCN":
            self.convs = torch.nn.ModuleList([
                GCNConv(input_dim if i == 0 else hidden_dim,
                       hidden_dim if i < num_layers - 1 else output_dim)
                for i in range(num_layers)
            ])
        elif model_type == "GAT":
            self.convs = torch.nn.ModuleList([
                GATv2Conv(input_dim if i == 0 else hidden_dim * 4,  # Account for concatenated heads
                         hidden_dim if i < num_layers - 1 else output_dim,
                         heads=4 if i < num_layers - 1 else 1,
                         concat=True if i < num_layers - 1 else False)
                for i in range(num_layers)
            ])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.dropout = torch.nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the GNN."""
        for i, conv in enumerate(self.convs):
            if self.model_type == "GAT" and i < len(self.convs) - 1:
                # GAT with multiple heads needs special handling
                x = conv(x, edge_index)
                x = F.relu(x)
                x = self.dropout(x)
            else:
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = self.dropout(x)
        
        return x


class CricketGNNExplainer:
    """Main class for explaining cricket GNN model predictions."""
    
    def __init__(self, model: Optional[torch.nn.Module] = None, 
                 device: Optional[torch.device] = None):
        """
        Initialize the GNN explainer.
        
        Args:
            model: Trained GNN model (optional, can be loaded later)
            device: Device for computation
        """
        self.model = model
        self.device = device if device is not None else torch.device('cpu')
        self.explainer = None
        self.graph_data = None
        self.node_mapping = {}  # Maps node IDs to indices
        self.reverse_node_mapping = {}  # Maps indices to node IDs
        
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
    
    def load_model(self, model_path: str, model_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Load a trained GNN model from file.
        
        Args:
            model_path: Path to saved model file
            model_config: Model configuration parameters
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if model_config is None:
                # Try to extract config from checkpoint
                model_config = checkpoint.get('config', {
                    'model_type': 'GraphSAGE',
                    'input_dim': 64,
                    'hidden_dim': 128,
                    'output_dim': 64,
                    'num_layers': 2
                })
            
            # Create model with config
            self.model = GNNModelWrapper(**model_config)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded {model_config.get('model_type', 'GNN')} model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def load_graph(self, graph_data: Union[Data, HeteroData, nx.Graph, str]) -> None:
        """
        Load knowledge graph data.
        
        Args:
            graph_data: Graph data in various formats
        """
        if isinstance(graph_data, str):
            # Load from file
            graph_path = Path(graph_data)
            if graph_path.suffix == '.pkl':
                with open(graph_path, 'rb') as f:
                    graph_data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported graph file format: {graph_path.suffix}")
        
        if isinstance(graph_data, nx.Graph):
            # Convert NetworkX to PyTorch Geometric Data
            self.graph_data = self._networkx_to_pyg(graph_data)
        elif isinstance(graph_data, (Data, HeteroData)):
            self.graph_data = graph_data
        else:
            raise ValueError(f"Unsupported graph data type: {type(graph_data)}")
        
        self.graph_data = self.graph_data.to(self.device)
        logger.info(f"Loaded graph with {self.graph_data.num_nodes} nodes and {self.graph_data.num_edges} edges")
    
    def _networkx_to_pyg(self, nx_graph: nx.Graph) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data."""
        # Create node mapping
        nodes = list(nx_graph.nodes())
        self.node_mapping = {node: i for i, node in enumerate(nodes)}
        self.reverse_node_mapping = {i: node for node, i in self.node_mapping.items()}
        
        # Extract node features - first pass to determine max dimension
        max_dim = 64  # Default dimension
        raw_features = []
        
        for node in nodes:
            node_data = nx_graph.nodes[node]
            if 'x' in node_data and isinstance(node_data['x'], torch.Tensor):
                feature = node_data['x']
                max_dim = max(max_dim, feature.shape[0])
                raw_features.append(feature)
            elif 'features' in node_data:
                if isinstance(node_data['features'], torch.Tensor):
                    feature = node_data['features']
                else:
                    feature = torch.tensor(node_data['features'], dtype=torch.float32)
                max_dim = max(max_dim, feature.shape[0])
                raw_features.append(feature)
            else:
                raw_features.append(None)
        
        # Second pass to normalize all features to max_dim
        node_features = []
        for i, feature in enumerate(raw_features):
            if feature is not None:
                if feature.shape[0] < max_dim:
                    # Pad with zeros
                    padded = torch.zeros(max_dim, dtype=torch.float32)
                    padded[:feature.shape[0]] = feature
                    node_features.append(padded)
                elif feature.shape[0] > max_dim:
                    # Truncate
                    node_features.append(feature[:max_dim])
                else:
                    node_features.append(feature)
            else:
                # Default feature vector
                node_features.append(torch.zeros(max_dim, dtype=torch.float32))
        
        x = torch.stack(node_features)
        
        # Extract edges
        edge_list = []
        edge_attrs = []
        
        for u, v, edge_data in nx_graph.edges(data=True):
            u_idx = self.node_mapping[u]
            v_idx = self.node_mapping[v]
            edge_list.append([u_idx, v_idx])
            
            # Extract edge attributes
            if 'edge_attr' in edge_data and isinstance(edge_data['edge_attr'], torch.Tensor):
                edge_attrs.append(edge_data['edge_attr'])
            elif 'weight' in edge_data:
                edge_attrs.append(torch.tensor([edge_data['weight']], dtype=torch.float32))
            else:
                edge_attrs.append(torch.tensor([1.0], dtype=torch.float32))
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attrs) if edge_attrs else None
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def setup_explainer(self, explanation_type: str = "model", **kwargs) -> None:
        """
        Setup the GNN explainer.
        
        Args:
            explanation_type: Type of explanation ('model' or 'phenomenon')
            **kwargs: Additional explainer parameters
        """
        if self.model is None:
            raise ValueError("Model must be loaded before setting up explainer")
        
        if self.graph_data is None:
            raise ValueError("Graph data must be loaded before setting up explainer")
        
        # Default explainer parameters
        explainer_params = {
            'epochs': kwargs.get('epochs', 200),
            'lr': kwargs.get('lr', 0.01),
            'edge_size': kwargs.get('edge_size', 0.005),
            'edge_ent': kwargs.get('edge_ent', 1.0),
            'node_feat_size': kwargs.get('node_feat_size', 0.005),
            'node_feat_ent': kwargs.get('node_feat_ent', 0.1)
        }
        
        # Create explainer
        explainer_algorithm = GNNExplainer(**explainer_params)
        
        self.explainer = Explainer(
            model=self.model,
            algorithm=explainer_algorithm,
            explanation_type=explanation_type,
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='regression',
                task_level='node',
                return_type='raw'
            )
        )
        
        logger.info(f"Setup {explanation_type} explainer with parameters: {explainer_params}")
    
    def explain_player_embedding(self, player_id: Union[str, int], 
                                top_k: int = 5) -> ExplanationResult:
        """
        Explain a player's embedding by identifying influential neighbors.
        
        Args:
            player_id: Player ID or node index to explain
            top_k: Number of top influential neighbors to return
            
        Returns:
            ExplanationResult containing explanation details
        """
        if self.explainer is None:
            self.setup_explainer('model')
        
        # Get node index
        if isinstance(player_id, str):
            if player_id not in self.node_mapping:
                raise ValueError(f"Player {player_id} not found in graph")
            node_idx = self.node_mapping[player_id]
        else:
            node_idx = player_id
            player_id = self.reverse_node_mapping.get(node_idx, str(node_idx))
        
        # Get explanation
        try:
            explanation = self.explainer(
                self.graph_data.x,
                self.graph_data.edge_index,
                index=node_idx,
                edge_attr=getattr(self.graph_data, 'edge_attr', None)
            )
            
            # Extract importance scores
            node_importance = explanation.node_mask
            edge_importance = explanation.edge_mask
            
            # Get target node embedding
            with torch.no_grad():
                embeddings = self.model(
                    self.graph_data.x,
                    self.graph_data.edge_index,
                    getattr(self.graph_data, 'edge_attr', None)
                )
                target_embedding = embeddings[node_idx]
            
            # Find top influential neighbors
            top_neighbors = self._get_top_neighbors(node_idx, node_importance, top_k)
            
            # Find top influential edges
            top_edges = self._get_top_edges(node_idx, edge_importance, top_k)
            
            # Create explanation subgraph
            explanation_subgraph = self._create_explanation_subgraph(
                node_idx, top_neighbors, top_edges
            )
            
            result = ExplanationResult(
                target_node=player_id,
                target_embedding=target_embedding,
                node_importance=node_importance,
                edge_importance=edge_importance,
                top_neighbors=top_neighbors,
                top_edges=top_edges,
                explanation_subgraph=explanation_subgraph,
                metadata={
                    'model_type': getattr(self.model, 'model_type', 'Unknown'),
                    'explanation_params': self.explainer.algorithm.__dict__,
                    'graph_stats': {
                        'num_nodes': self.graph_data.num_nodes,
                        'num_edges': self.graph_data.num_edges
                    }
                }
            )
            
            logger.info(f"Generated explanation for player {player_id} with {len(top_neighbors)} top neighbors")
            return result
            
        except Exception as e:
            logger.error(f"Failed to explain player {player_id}: {str(e)}")
            raise RuntimeError(f"Explanation failed: {str(e)}")
    
    def _get_top_neighbors(self, node_idx: int, node_importance: torch.Tensor, 
                          top_k: int) -> List[Tuple[Union[int, str], float]]:
        """Get top k most influential neighbors."""
        # Get neighbors of the target node
        edge_index = self.graph_data.edge_index
        neighbors = []
        
        # Find all neighbors
        neighbor_mask = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)
        neighbor_edges = edge_index[:, neighbor_mask]
        
        neighbor_nodes = set()
        for i in range(neighbor_edges.shape[1]):
            src, dst = neighbor_edges[0, i].item(), neighbor_edges[1, i].item()
            if src != node_idx:
                neighbor_nodes.add(src)
            if dst != node_idx:
                neighbor_nodes.add(dst)
        
        # Get importance scores for neighbors
        neighbor_scores = []
        for neighbor_idx in neighbor_nodes:
            # Handle both 1D and 2D node importance tensors
            if node_importance.dim() == 2:
                importance_score = node_importance[neighbor_idx].mean().item()
            else:
                importance_score = node_importance[neighbor_idx].item()
            neighbor_id = self.reverse_node_mapping.get(neighbor_idx, str(neighbor_idx))
            neighbor_scores.append((neighbor_id, importance_score))
        
        # Sort by importance and return top k
        neighbor_scores.sort(key=lambda x: x[1], reverse=True)
        return neighbor_scores[:top_k]
    
    def _get_top_edges(self, node_idx: int, edge_importance: torch.Tensor, 
                      top_k: int) -> List[Tuple[Tuple[Union[int, str], Union[int, str]], float]]:
        """Get top k most influential edges."""
        edge_index = self.graph_data.edge_index
        edge_scores = []
        
        # Find edges connected to the target node
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src == node_idx or dst == node_idx:
                importance_score = edge_importance[i].item()
                src_id = self.reverse_node_mapping.get(src, str(src))
                dst_id = self.reverse_node_mapping.get(dst, str(dst))
                edge_scores.append(((src_id, dst_id), importance_score))
        
        # Sort by importance and return top k
        edge_scores.sort(key=lambda x: x[1], reverse=True)
        return edge_scores[:top_k]
    
    def _create_explanation_subgraph(self, node_idx: int, 
                                   top_neighbors: List[Tuple[Union[int, str], float]],
                                   top_edges: List[Tuple[Tuple[Union[int, str], Union[int, str]], float]]) -> nx.Graph:
        """Create a subgraph containing the explanation."""
        subgraph = nx.Graph()
        
        # Add target node
        target_id = self.reverse_node_mapping.get(node_idx, str(node_idx))
        subgraph.add_node(target_id, node_type='target', importance=1.0)
        
        # Add top neighbors
        for neighbor_id, importance in top_neighbors:
            subgraph.add_node(neighbor_id, node_type='neighbor', importance=importance)
        
        # Add top edges
        for (src_id, dst_id), importance in top_edges:
            if src_id in subgraph.nodes and dst_id in subgraph.nodes:
                subgraph.add_edge(src_id, dst_id, importance=importance)
        
        return subgraph
    
    def visualize_explanation(self, result: ExplanationResult, 
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create a visual explanation of the results.
        
        Args:
            result: ExplanationResult to visualize
            save_path: Optional path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'GNN Explanation for Player: {result.target_node}', fontsize=16, fontweight='bold')
        
        # 1. Top neighbors bar chart
        ax1 = axes[0, 0]
        if result.top_neighbors:
            neighbors, scores = zip(*result.top_neighbors)
            y_pos = np.arange(len(neighbors))
            bars = ax1.barh(y_pos, scores, color='skyblue', alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([str(n)[:15] + '...' if len(str(n)) > 15 else str(n) for n in neighbors])
            ax1.set_xlabel('Importance Score')
            ax1.set_title('Top Influential Neighbors')
            ax1.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{score:.3f}', ha='left', va='center', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No neighbors found', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Top Influential Neighbors')
        
        # 2. Top edges bar chart
        ax2 = axes[0, 1]
        if result.top_edges:
            edge_labels = [f"{src[:8]}→{dst[:8]}" for (src, dst), _ in result.top_edges]
            edge_scores = [score for _, score in result.top_edges]
            y_pos = np.arange(len(edge_labels))
            bars = ax2.barh(y_pos, edge_scores, color='lightcoral', alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(edge_labels)
            ax2.set_xlabel('Importance Score')
            ax2.set_title('Top Influential Edges')
            ax2.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, edge_scores)):
                ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{score:.3f}', ha='left', va='center', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No edges found', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Top Influential Edges')
        
        # 3. Importance distribution histogram
        ax3 = axes[1, 0]
        if result.node_importance is not None:
            importance_values = result.node_importance.detach().cpu().numpy()
            # Flatten if 2D tensor
            if importance_values.ndim > 1:
                importance_values = importance_values.flatten()
            ax3.hist(importance_values, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax3.axvline(importance_values.mean(), color='red', linestyle='--', 
                       label=f'Mean: {importance_values.mean():.3f}')
            ax3.set_xlabel('Node Importance Score')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Node Importance Distribution')
            ax3.legend()
            ax3.grid(alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No importance data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Node Importance Distribution')
        
        # 4. Explanation metadata
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if result.metadata:
            metadata_text = []
            metadata_text.append(f"Model Type: {result.metadata.get('model_type', 'Unknown')}")
            
            if 'graph_stats' in result.metadata:
                stats = result.metadata['graph_stats']
                metadata_text.append(f"Graph Nodes: {stats.get('num_nodes', 'Unknown')}")
                metadata_text.append(f"Graph Edges: {stats.get('num_edges', 'Unknown')}")
            
            metadata_text.append(f"Top Neighbors: {len(result.top_neighbors)}")
            metadata_text.append(f"Top Edges: {len(result.top_edges)}")
            
            if result.target_embedding is not None:
                emb_norm = torch.norm(result.target_embedding).item()
                metadata_text.append(f"Embedding Norm: {emb_norm:.3f}")
            
            ax4.text(0.1, 0.9, '\n'.join(metadata_text), transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved explanation visualization to {save_path}")
        
        return fig
    
    def export_to_graphviz(self, result: ExplanationResult, 
                          output_path: str = "explanation.dot",
                          render_format: str = "png") -> Optional[str]:
        """
        Export explanation to Graphviz format.
        
        Args:
            result: ExplanationResult to export
            output_path: Path for output file
            render_format: Output format ('png', 'svg', 'pdf')
            
        Returns:
            Path to rendered file if successful, None otherwise
        """
        if not GRAPHVIZ_AVAILABLE:
            logger.warning("Graphviz not available. Cannot export visualization.")
            return None
        
        if result.explanation_subgraph is None:
            logger.warning("No explanation subgraph available for export.")
            return None
        
        try:
            # Create Graphviz digraph
            dot = graphviz.Digraph(comment=f'GNN Explanation for {result.target_node}')
            dot.attr(rankdir='TB', size='10,8')
            
            # Add nodes
            for node_id, node_data in result.explanation_subgraph.nodes(data=True):
                node_type = node_data.get('node_type', 'unknown')
                importance = node_data.get('importance', 0.0)
                
                if node_type == 'target':
                    # Target node - larger, different color
                    dot.node(str(node_id), str(node_id), 
                            style='filled', fillcolor='lightblue', 
                            shape='box', fontsize='14', fontweight='bold')
                else:
                    # Neighbor nodes - color by importance
                    color_intensity = min(1.0, max(0.1, importance))
                    color = f"gray{int((1 - color_intensity) * 80 + 20)}"
                    dot.node(str(node_id), str(node_id), 
                            style='filled', fillcolor=color,
                            shape='ellipse', fontsize='12')
            
            # Add edges
            for src, dst, edge_data in result.explanation_subgraph.edges(data=True):
                importance = edge_data.get('importance', 0.0)
                
                # Edge thickness based on importance
                penwidth = str(max(1.0, importance * 5))
                
                # Edge color based on importance
                if importance > 0.5:
                    color = 'red'
                elif importance > 0.2:
                    color = 'orange'
                else:
                    color = 'gray'
                
                dot.edge(str(src), str(dst), 
                        penwidth=penwidth, color=color,
                        label=f'{importance:.3f}' if importance > 0.1 else '')
            
            # Save dot file
            dot.save(output_path)
            
            # Render to specified format
            if render_format:
                output_base = Path(output_path).stem
                rendered_path = f"{output_base}.{render_format}"
                dot.render(output_base, format=render_format, cleanup=True)
                logger.info(f"Exported explanation to {rendered_path}")
                return rendered_path
            
            logger.info(f"Saved Graphviz dot file to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export to Graphviz: {str(e)}")
            return None
    
    def create_sample_model_and_graph(self, num_nodes: int = 20, 
                                    num_edges: int = 50) -> Tuple[torch.nn.Module, Data]:
        """
        Create sample model and graph for testing.
        
        Args:
            num_nodes: Number of nodes in sample graph
            num_edges: Number of edges in sample graph
            
        Returns:
            Tuple of (model, graph_data)
        """
        # Create sample model
        model = GNNModelWrapper(
            model_type="GraphSAGE",
            input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_layers=2
        )
        
        # Create sample graph data
        x = torch.randn(num_nodes, 64, dtype=torch.float32)
        
        # Create random edges
        edge_list = []
        for _ in range(num_edges):
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            if src != dst:  # Avoid self-loops
                edge_list.append([src, dst])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.randn(edge_index.shape[1], 1, dtype=torch.float32)
        
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Create node mapping for sample data
        self.node_mapping = {f"player_{i}": i for i in range(num_nodes)}
        self.reverse_node_mapping = {i: f"player_{i}" for i in range(num_nodes)}
        
        logger.info(f"Created sample model and graph with {num_nodes} nodes and {edge_index.shape[1]} edges")
        return model, graph_data


def create_explanation_summary(result: ExplanationResult) -> Dict[str, Any]:
    """
    Create a summary dictionary of the explanation result.
    
    Args:
        result: ExplanationResult to summarize
        
    Returns:
        Dictionary containing explanation summary
    """
    summary = {
        'target_player': result.target_node,
        'num_neighbors_analyzed': len(result.top_neighbors),
        'num_edges_analyzed': len(result.top_edges),
        'top_neighbors': [
            {'neighbor': neighbor, 'importance': float(score)}
            for neighbor, score in result.top_neighbors
        ],
        'top_edges': [
            {'edge': f"{src} -> {dst}", 'importance': float(score)}
            for (src, dst), score in result.top_edges
        ]
    }
    
    if result.metadata:
        summary['metadata'] = result.metadata
    
    if result.target_embedding is not None:
        summary['embedding_stats'] = {
            'norm': float(torch.norm(result.target_embedding).item()),
            'mean': float(result.target_embedding.mean().item()),
            'std': float(result.target_embedding.std().item())
        }
    
    return summary


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample explainer
    explainer = CricketGNNExplainer()
    
    # Create sample data
    model, graph_data = explainer.create_sample_model_and_graph(num_nodes=15, num_edges=30)
    
    # Load the sample data
    explainer.model = model
    explainer.load_graph(graph_data)
    
    # Setup explainer
    explainer.setup_explainer('node')
    
    # Explain a sample player
    result = explainer.explain_player_embedding('player_0', top_k=3)
    
    # Print results
    print(f"Explanation for {result.target_node}:")
    print(f"Top neighbors: {result.top_neighbors}")
    print(f"Top edges: {result.top_edges}")
    
    # Create visualization
    fig = explainer.visualize_explanation(result)
    plt.show()
    
    # Create summary
    summary = create_explanation_summary(result)
    print(f"Summary: {summary}")