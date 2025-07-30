# Purpose: Unit tests for GNN explanation module
# Author: WicketWise Team, Last Modified: 2024-07-19

import pytest
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from unittest.mock import patch, MagicMock
import tempfile
import pickle
from pathlib import Path

# Import the classes and functions from explain_gnn
import sys
sys.path.append('.')
from explain_gnn import (
    CricketGNNExplainer,
    GNNModelWrapper,
    ExplanationResult,
    create_explanation_summary
)


class TestGNNModelWrapper:
    """Test suite for GNN model wrapper."""
    
    def test_graphsage_model_creation(self):
        """Test GraphSAGE model creation."""
        model = GNNModelWrapper(
            model_type="GraphSAGE",
            input_dim=32,
            hidden_dim=64,
            output_dim=32,
            num_layers=2
        )
        
        assert model.model_type == "GraphSAGE", "Should set model type correctly"
        assert model.num_layers == 2, "Should set number of layers correctly"
        assert len(model.convs) == 2, "Should create correct number of conv layers"
    
    def test_gcn_model_creation(self):
        """Test GCN model creation."""
        model = GNNModelWrapper(
            model_type="GCN",
            input_dim=16,
            hidden_dim=32,
            output_dim=16,
            num_layers=3
        )
        
        assert model.model_type == "GCN", "Should set model type correctly"
        assert len(model.convs) == 3, "Should create correct number of conv layers"
    
    def test_gat_model_creation(self):
        """Test GAT model creation."""
        model = GNNModelWrapper(
            model_type="GAT",
            input_dim=24,
            hidden_dim=48,
            output_dim=24,
            num_layers=2
        )
        
        assert model.model_type == "GAT", "Should set model type correctly"
        assert len(model.convs) == 2, "Should create correct number of conv layers"
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            GNNModelWrapper(model_type="InvalidModel")
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = GNNModelWrapper(
            model_type="GraphSAGE",
            input_dim=16,
            hidden_dim=32,
            output_dim=16,
            num_layers=2
        )
        
        # Create sample input
        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x, edge_index)
        
        assert output.shape == (10, 16), "Should output correct shape"
        assert not torch.isnan(output).any(), "Should not contain NaN values"


class TestCricketGNNExplainer:
    """Test suite for Cricket GNN explainer."""
    
    @pytest.fixture
    def sample_explainer(self):
        """Create sample explainer with model and graph."""
        explainer = CricketGNNExplainer()
        model, graph_data = explainer.create_sample_model_and_graph(num_nodes=10, num_edges=15)
        explainer.model = model
        explainer.graph_data = graph_data
        return explainer
    
    def test_explainer_initialization(self):
        """Test explainer initialization."""
        explainer = CricketGNNExplainer()
        
        assert explainer.model is None, "Should initialize without model"
        assert explainer.graph_data is None, "Should initialize without graph data"
        assert explainer.explainer is None, "Should initialize without explainer"
        assert explainer.device is not None, "Should have default device"
    
    def test_explainer_with_model(self):
        """Test explainer initialization with model."""
        model = GNNModelWrapper()
        explainer = CricketGNNExplainer(model=model)
        
        assert explainer.model is not None, "Should initialize with model"
        assert explainer.model.training == False, "Model should be in eval mode"
    
    def test_create_sample_model_and_graph(self):
        """Test sample model and graph creation."""
        explainer = CricketGNNExplainer()
        model, graph_data = explainer.create_sample_model_and_graph(num_nodes=8, num_edges=12)
        
        assert isinstance(model, GNNModelWrapper), "Should return GNN model"
        assert isinstance(graph_data, Data), "Should return PyTorch Geometric Data"
        assert graph_data.x.shape[0] == 8, "Should have correct number of nodes"
        assert graph_data.x.shape[1] == 64, "Should have correct feature dimension"
        assert graph_data.edge_index.shape[1] <= 12, "Should have at most specified edges"
        
        # Check node mappings
        assert len(explainer.node_mapping) == 8, "Should create node mapping"
        assert len(explainer.reverse_node_mapping) == 8, "Should create reverse mapping"
    
    def test_networkx_to_pyg_conversion(self):
        """Test NetworkX to PyTorch Geometric conversion."""
        # Create sample NetworkX graph
        nx_graph = nx.Graph()
        nx_graph.add_node("player1", x=torch.randn(32))
        nx_graph.add_node("player2", x=torch.randn(32))
        nx_graph.add_node("player3", features=[0.1, 0.2, 0.3])
        nx_graph.add_edge("player1", "player2", weight=0.8)
        nx_graph.add_edge("player2", "player3", weight=0.6)
        
        explainer = CricketGNNExplainer()
        pyg_data = explainer._networkx_to_pyg(nx_graph)
        
        assert isinstance(pyg_data, Data), "Should return PyTorch Geometric Data"
        assert pyg_data.num_nodes == 3, "Should have correct number of nodes"
        assert pyg_data.num_edges == 2, "Should have correct number of edges"
        assert pyg_data.x.shape[1] == 64, "Should have normalized feature dimension"
        
        # Check mappings
        assert "player1" in explainer.node_mapping, "Should map player1"
        assert "player2" in explainer.node_mapping, "Should map player2"
        assert "player3" in explainer.node_mapping, "Should map player3"
    
    def test_load_graph_data(self, sample_explainer):
        """Test loading graph data."""
        # Graph data should already be loaded in fixture
        assert sample_explainer.graph_data is not None, "Should have graph data"
        assert sample_explainer.graph_data.num_nodes == 10, "Should have correct nodes"
    
    def test_load_graph_networkx(self):
        """Test loading NetworkX graph."""
        # Create sample NetworkX graph
        nx_graph = nx.Graph()
        for i in range(5):
            nx_graph.add_node(f"player_{i}", x=torch.randn(16))
        nx_graph.add_edge("player_0", "player_1")
        nx_graph.add_edge("player_1", "player_2")
        
        explainer = CricketGNNExplainer()
        explainer.load_graph(nx_graph)
        
        assert explainer.graph_data is not None, "Should load graph data"
        assert explainer.graph_data.num_nodes == 5, "Should have correct nodes"
    
    def test_setup_explainer(self, sample_explainer):
        """Test explainer setup."""
        sample_explainer.setup_explainer(explanation_type='model')
        
        assert sample_explainer.explainer is not None, "Should create explainer"
    
    def test_setup_explainer_without_model(self):
        """Test explainer setup without model raises error."""
        explainer = CricketGNNExplainer()
        
        with pytest.raises(ValueError, match="Model must be loaded"):
            explainer.setup_explainer()
    
    def test_setup_explainer_without_graph(self):
        """Test explainer setup without graph raises error."""
        model = GNNModelWrapper()
        explainer = CricketGNNExplainer(model=model)
        
        with pytest.raises(ValueError, match="Graph data must be loaded"):
            explainer.setup_explainer()
    
    def test_explain_player_embedding(self, sample_explainer):
        """Test explaining player embedding."""
        # Setup explainer
        sample_explainer.setup_explainer('model')
        
        # Explain a player
        result = sample_explainer.explain_player_embedding('player_0', top_k=3)
        
        assert isinstance(result, ExplanationResult), "Should return ExplanationResult"
        assert result.target_node == 'player_0', "Should target correct player"
        assert result.target_embedding is not None, "Should have target embedding"
        assert result.node_importance is not None, "Should have node importance"
        assert result.edge_importance is not None, "Should have edge importance"
        assert len(result.top_neighbors) <= 3, "Should return at most top_k neighbors"
        assert len(result.top_edges) <= 3, "Should return at most top_k edges"
    
    def test_explain_player_embedding_by_index(self, sample_explainer):
        """Test explaining player embedding by node index."""
        sample_explainer.setup_explainer('model')
        
        # Explain by index
        result = sample_explainer.explain_player_embedding(0, top_k=2)
        
        assert isinstance(result, ExplanationResult), "Should return ExplanationResult"
        assert result.target_node == 'player_0', "Should map index to player ID"
        assert len(result.top_neighbors) <= 2, "Should respect top_k parameter"
    
    def test_explain_nonexistent_player(self, sample_explainer):
        """Test explaining nonexistent player raises error."""
        sample_explainer.setup_explainer('model')
        
        with pytest.raises(ValueError, match="Player .* not found"):
            sample_explainer.explain_player_embedding('nonexistent_player')
    
    def test_get_top_neighbors(self, sample_explainer):
        """Test getting top neighbors."""
        sample_explainer.setup_explainer('model')
        
        # Create mock importance scores
        node_importance = torch.rand(sample_explainer.graph_data.num_nodes)
        
        top_neighbors = sample_explainer._get_top_neighbors(0, node_importance, top_k=3)
        
        assert isinstance(top_neighbors, list), "Should return list"
        assert len(top_neighbors) <= 3, "Should return at most top_k neighbors"
        
        # Check that neighbors are sorted by importance
        if len(top_neighbors) > 1:
            scores = [score for _, score in top_neighbors]
            assert scores == sorted(scores, reverse=True), "Should be sorted by importance"
    
    def test_get_top_edges(self, sample_explainer):
        """Test getting top edges."""
        sample_explainer.setup_explainer('model')
        
        # Create mock importance scores
        edge_importance = torch.rand(sample_explainer.graph_data.num_edges)
        
        top_edges = sample_explainer._get_top_edges(0, edge_importance, top_k=3)
        
        assert isinstance(top_edges, list), "Should return list"
        assert len(top_edges) <= 3, "Should return at most top_k edges"
        
        # Check format
        for (src, dst), score in top_edges:
            assert isinstance(src, str), "Source should be string"
            assert isinstance(dst, str), "Destination should be string"
            assert isinstance(score, float), "Score should be float"
    
    def test_create_explanation_subgraph(self, sample_explainer):
        """Test creating explanation subgraph."""
        # Create sample data
        top_neighbors = [('player_1', 0.8), ('player_2', 0.6)]
        top_edges = [(('player_0', 'player_1'), 0.9), (('player_1', 'player_2'), 0.7)]
        
        subgraph = sample_explainer._create_explanation_subgraph(0, top_neighbors, top_edges)
        
        assert isinstance(subgraph, nx.Graph), "Should return NetworkX graph"
        assert 'player_0' in subgraph.nodes, "Should contain target node"
        assert 'player_1' in subgraph.nodes, "Should contain neighbor nodes"
        assert subgraph.nodes['player_0']['node_type'] == 'target', "Should mark target node"
        assert subgraph.nodes['player_1']['node_type'] == 'neighbor', "Should mark neighbor nodes"


class TestExplanationResult:
    """Test suite for ExplanationResult dataclass."""
    
    def test_explanation_result_creation(self):
        """Test creating ExplanationResult."""
        target_embedding = torch.randn(64)
        node_importance = torch.rand(10)
        edge_importance = torch.rand(15)
        top_neighbors = [('player_1', 0.8), ('player_2', 0.6)]
        top_edges = [(('player_0', 'player_1'), 0.9)]
        
        result = ExplanationResult(
            target_node='player_0',
            target_embedding=target_embedding,
            node_importance=node_importance,
            edge_importance=edge_importance,
            top_neighbors=top_neighbors,
            top_edges=top_edges
        )
        
        assert result.target_node == 'player_0', "Should set target node"
        assert torch.equal(result.target_embedding, target_embedding), "Should set embedding"
        assert len(result.top_neighbors) == 2, "Should set neighbors"
        assert len(result.top_edges) == 1, "Should set edges"
    
    def test_explanation_result_with_metadata(self):
        """Test ExplanationResult with metadata."""
        metadata = {'model_type': 'GraphSAGE', 'test_param': 42}
        
        result = ExplanationResult(
            target_node='test_player',
            target_embedding=torch.randn(32),
            node_importance=torch.rand(5),
            edge_importance=torch.rand(8),
            top_neighbors=[],
            top_edges=[],
            metadata=metadata
        )
        
        assert result.metadata == metadata, "Should store metadata"
        assert result.metadata['model_type'] == 'GraphSAGE', "Should access metadata"


class TestVisualizationAndExport:
    """Test suite for visualization and export functionality."""
    
    @pytest.fixture
    def sample_result(self):
        """Create sample ExplanationResult for testing."""
        return ExplanationResult(
            target_node='test_player',
            target_embedding=torch.randn(64),
            node_importance=torch.rand(10),
            edge_importance=torch.rand(15),
            top_neighbors=[('neighbor_1', 0.8), ('neighbor_2', 0.6), ('neighbor_3', 0.4)],
            top_edges=[(('test_player', 'neighbor_1'), 0.9), (('neighbor_1', 'neighbor_2'), 0.7)],
            explanation_subgraph=nx.Graph(),
            metadata={'model_type': 'GraphSAGE', 'graph_stats': {'num_nodes': 10, 'num_edges': 15}}
        )
    
    def test_visualize_explanation(self, sample_result):
        """Test explanation visualization."""
        explainer = CricketGNNExplainer()
        
        # Test visualization creation
        fig = explainer.visualize_explanation(sample_result)
        
        assert fig is not None, "Should create matplotlib figure"
        assert len(fig.axes) == 4, "Should create 4 subplots"
    
    def test_visualize_explanation_empty_result(self):
        """Test visualization with empty result."""
        empty_result = ExplanationResult(
            target_node='empty_player',
            target_embedding=torch.randn(32),
            node_importance=torch.rand(5),
            edge_importance=torch.rand(8),
            top_neighbors=[],
            top_edges=[]
        )
        
        explainer = CricketGNNExplainer()
        fig = explainer.visualize_explanation(empty_result)
        
        assert fig is not None, "Should handle empty results gracefully"
    
    @pytest.mark.skip(reason="Graphviz not installed in test environment")
    def test_export_to_graphviz(self, sample_result):
        """Test Graphviz export."""
        # Add nodes to subgraph
        sample_result.explanation_subgraph.add_node('test_player', node_type='target', importance=1.0)
        sample_result.explanation_subgraph.add_node('neighbor_1', node_type='neighbor', importance=0.8)
        sample_result.explanation_subgraph.add_edge('test_player', 'neighbor_1', importance=0.9)
        
        explainer = CricketGNNExplainer()
        
        # Test when graphviz is available
        with patch('explain_gnn.GRAPHVIZ_AVAILABLE', True):
            with patch('graphviz.Digraph') as mock_digraph:
                mock_dot = MagicMock()
                mock_digraph.return_value = mock_dot
                
                result_path = explainer.export_to_graphviz(sample_result, "test.dot", "png")
                
                # Verify Graphviz was called
                mock_digraph.assert_called_once()
                mock_dot.node.assert_called()
                mock_dot.edge.assert_called()
                mock_dot.save.assert_called_with("test.dot")
    
    @patch('explain_gnn.GRAPHVIZ_AVAILABLE', False)
    def test_export_to_graphviz_unavailable(self, sample_result):
        """Test Graphviz export when not available."""
        explainer = CricketGNNExplainer()
        result = explainer.export_to_graphviz(sample_result)
        
        assert result is None, "Should return None when Graphviz unavailable"
    
    def test_export_to_graphviz_no_subgraph(self):
        """Test Graphviz export with no subgraph."""
        result_no_subgraph = ExplanationResult(
            target_node='test',
            target_embedding=torch.randn(32),
            node_importance=torch.rand(5),
            edge_importance=torch.rand(8),
            top_neighbors=[],
            top_edges=[],
            explanation_subgraph=None
        )
        
        explainer = CricketGNNExplainer()
        result = explainer.export_to_graphviz(result_no_subgraph)
        
        assert result is None, "Should return None with no subgraph"


class TestModelLoading:
    """Test suite for model loading functionality."""
    
    def test_load_model_with_config(self, tmp_path):
        """Test loading model with configuration."""
        # Create sample model and save it
        model = GNNModelWrapper(model_type="GCN", input_dim=32, hidden_dim=64, output_dim=32)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'model_type': 'GCN',
                'input_dim': 32,
                'hidden_dim': 64,
                'output_dim': 32,
                'num_layers': 2
            }
        }
        
        model_path = tmp_path / "test_model.pth"
        torch.save(checkpoint, model_path)
        
        # Load model
        explainer = CricketGNNExplainer()
        explainer.load_model(str(model_path))
        
        assert explainer.model is not None, "Should load model"
        assert explainer.model.model_type == "GCN", "Should load correct model type"
    
    def test_load_model_without_config(self, tmp_path):
        """Test loading model without configuration."""
        # Create and save model state dict only
        model = GNNModelWrapper()
        model_path = tmp_path / "model_only.pth"
        torch.save(model.state_dict(), model_path)
        
        # Provide config manually
        config = {
            'model_type': 'GraphSAGE',
            'input_dim': 64,
            'hidden_dim': 128,
            'output_dim': 64,
            'num_layers': 2
        }
        
        explainer = CricketGNNExplainer()
        explainer.load_model(str(model_path), model_config=config)
        
        assert explainer.model is not None, "Should load model with manual config"
    
    def test_load_model_nonexistent_file(self):
        """Test loading nonexistent model file."""
        explainer = CricketGNNExplainer()
        
        with pytest.raises(RuntimeError, match="Model loading failed"):
            explainer.load_model("nonexistent_model.pth")
    
    def test_load_graph_from_pickle(self, tmp_path):
        """Test loading graph from pickle file."""
        # Create sample NetworkX graph
        nx_graph = nx.Graph()
        nx_graph.add_node("player1", x=torch.randn(16))
        nx_graph.add_node("player2", x=torch.randn(16))
        nx_graph.add_edge("player1", "player2", weight=0.8)
        
        # Save to pickle
        graph_path = tmp_path / "test_graph.pkl"
        with open(graph_path, 'wb') as f:
            pickle.dump(nx_graph, f)
        
        # Load graph
        explainer = CricketGNNExplainer()
        explainer.load_graph(str(graph_path))
        
        assert explainer.graph_data is not None, "Should load graph from pickle"
        assert explainer.graph_data.num_nodes == 2, "Should have correct nodes"
    
    def test_load_graph_invalid_format(self, tmp_path):
        """Test loading graph with invalid format."""
        # Create invalid file
        invalid_path = tmp_path / "invalid.txt"
        with open(invalid_path, 'w') as f:
            f.write("invalid content")
        
        explainer = CricketGNNExplainer()
        
        with pytest.raises(ValueError, match="Unsupported graph file format"):
            explainer.load_graph(str(invalid_path))


class TestExplanationSummary:
    """Test suite for explanation summary functionality."""
    
    def test_create_explanation_summary(self):
        """Test creating explanation summary."""
        result = ExplanationResult(
            target_node='test_player',
            target_embedding=torch.tensor([1.0, 2.0, 3.0]),
            node_importance=torch.rand(5),
            edge_importance=torch.rand(8),
            top_neighbors=[('neighbor_1', 0.8), ('neighbor_2', 0.6)],
            top_edges=[(('test_player', 'neighbor_1'), 0.9)],
            metadata={'model_type': 'GraphSAGE'}
        )
        
        summary = create_explanation_summary(result)
        
        assert isinstance(summary, dict), "Should return dictionary"
        assert summary['target_player'] == 'test_player', "Should include target player"
        assert summary['num_neighbors_analyzed'] == 2, "Should count neighbors"
        assert summary['num_edges_analyzed'] == 1, "Should count edges"
        assert 'top_neighbors' in summary, "Should include top neighbors"
        assert 'top_edges' in summary, "Should include top edges"
        assert 'embedding_stats' in summary, "Should include embedding stats"
        assert 'metadata' in summary, "Should include metadata"
    
    def test_create_explanation_summary_minimal(self):
        """Test creating summary with minimal data."""
        result = ExplanationResult(
            target_node='minimal_player',
            target_embedding=None,
            node_importance=torch.rand(3),
            edge_importance=torch.rand(5),
            top_neighbors=[],
            top_edges=[]
        )
        
        summary = create_explanation_summary(result)
        
        assert summary['target_player'] == 'minimal_player', "Should handle minimal data"
        assert summary['num_neighbors_analyzed'] == 0, "Should handle empty neighbors"
        assert summary['num_edges_analyzed'] == 0, "Should handle empty edges"
        assert 'embedding_stats' not in summary, "Should not include stats for None embedding"


class TestEndToEndWorkflow:
    """Test suite for complete end-to-end explanation workflow."""
    
    def test_complete_explanation_workflow(self):
        """Test complete explanation workflow from start to finish."""
        # 1. Create explainer
        explainer = CricketGNNExplainer()
        
        # 2. Create sample model and graph
        model, graph_data = explainer.create_sample_model_and_graph(num_nodes=12, num_edges=20)
        
        # 3. Load model and graph
        explainer.model = model
        explainer.graph_data = graph_data
        
        # 4. Setup explainer
        explainer.setup_explainer('model', epochs=50)  # Fewer epochs for testing
        
        # 5. Explain a player
        result = explainer.explain_player_embedding('player_0', top_k=3)
        
        # 6. Verify results
        assert isinstance(result, ExplanationResult), "Should return valid result"
        assert result.target_node == 'player_0', "Should target correct player"
        assert len(result.top_neighbors) <= 3, "Should respect top_k limit"
        assert result.target_embedding is not None, "Should have embedding"
        assert result.explanation_subgraph is not None, "Should create subgraph"
        
        # 7. Create visualization
        fig = explainer.visualize_explanation(result)
        assert fig is not None, "Should create visualization"
        
        # 8. Create summary
        summary = create_explanation_summary(result)
        assert isinstance(summary, dict), "Should create summary"
        assert 'target_player' in summary, "Summary should be valid"
        
        print("✅ Complete end-to-end explanation workflow test passed!")
    
    def test_explanation_with_different_models(self):
        """Test explanation with different GNN model types."""
        model_types = ["GraphSAGE", "GCN", "GAT"]
        
        for model_type in model_types:
            explainer = CricketGNNExplainer()
            
            # Create graph first to get correct input dimension
            _, graph_data = explainer.create_sample_model_and_graph(num_nodes=8, num_edges=12)
            input_dim = graph_data.x.shape[1]  # Use actual feature dimension
            
            # Create model of specific type with correct input dimension
            model = GNNModelWrapper(
                model_type=model_type,
                input_dim=input_dim,
                hidden_dim=64,
                output_dim=32,
                num_layers=2
            )
            
            # Load model and graph
            explainer.model = model
            explainer.graph_data = graph_data
            
            # Setup and run explanation
            explainer.setup_explainer('model', epochs=20)  # Quick test
            result = explainer.explain_player_embedding('player_0', top_k=2)
            
            assert isinstance(result, ExplanationResult), f"Should work with {model_type}"
            assert result.target_node == 'player_0', f"Should target correctly for {model_type}"
            
            print(f"✅ {model_type} explanation test passed!")


if __name__ == "__main__":
    # Run specific test for debugging
    pytest.main([__file__ + "::TestEndToEndWorkflow::test_complete_explanation_workflow", "-v"])