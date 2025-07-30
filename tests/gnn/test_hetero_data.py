# Purpose: Unit tests for HeteroData graph functionality
# Author: WicketWise Team, Last Modified: 2024-07-19

import pytest
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import HeteroData
from datetime import datetime

from crickformers.gnn.graph_builder import build_cricket_graph, build_cricket_hetero_graph
from crickformers.gnn.hetero_graph_builder import (
    networkx_to_hetero_data,
    get_hetero_data_stats,
    validate_hetero_data,
    NODE_TYPES,
    EDGE_TYPES,
    _create_node_type_mapping,
    _extract_node_features,
    _determine_hetero_edge_type
)
from crickformers.gnn.gnn_trainer import CricketHeteroGNNTrainer


class TestHeteroDataConversion:
    """Test suite for NetworkX to HeteroData conversion."""
    
    @pytest.fixture
    def sample_match_data(self):
        """Create sample match data for testing."""
        base_date = datetime(2024, 1, 15, 14, 30)
        
        return [
            {
                "match_id": "match_1",
                "innings": 1,
                "over": 1,
                "batter_id": "kohli",
                "non_striker_id": "sharma",
                "bowler_id": "starc",
                "batting_team_name": "india",
                "bowling_team_name": "australia",
                "venue_name": "mcg",
                "bowler_style": "pace",
                "match_date": base_date.strftime("%Y-%m-%d %H:%M:%S"),
                "runs": 4,
                "dismissal_type": ""
            },
            {
                "match_id": "match_1",
                "innings": 1,
                "over": 2,
                "batter_id": "sharma",
                "non_striker_id": "kohli",
                "bowler_id": "cummins",
                "batting_team_name": "india",
                "bowling_team_name": "australia",
                "venue_name": "mcg",
                "bowler_style": "pace",
                "match_date": base_date.strftime("%Y-%m-%d %H:%M:%S"),
                "runs": 0,
                "dismissal_type": "bowled"
            }
        ]
    
    @pytest.fixture
    def nx_graph(self, sample_match_data):
        """Create NetworkX graph from sample data."""
        return build_cricket_graph(sample_match_data)
    
    @pytest.fixture
    def hetero_data(self, nx_graph):
        """Create HeteroData from NetworkX graph."""
        return networkx_to_hetero_data(nx_graph)
    
    def test_node_type_mapping_creation(self, nx_graph):
        """Test creation of node type mappings."""
        mapping = _create_node_type_mapping(nx_graph)
        
        assert isinstance(mapping, dict), "Mapping should be a dictionary"
        assert len(mapping) > 0, "Mapping should not be empty"
        
        # Check that player nodes are mapped correctly
        assert "player" in mapping, "Should have player node type"
        player_mapping = mapping["player"]
        
        # Should have batters and bowlers
        assert "kohli" in player_mapping, "Kohli should be in player mapping"
        assert "starc" in player_mapping, "Starc should be in player mapping"
        
        # Check indices are sequential
        indices = list(player_mapping.values())
        assert indices == list(range(len(indices))), "Indices should be sequential"
    
    def test_hetero_data_creation(self, hetero_data):
        """Test that HeteroData object is created correctly."""
        assert isinstance(hetero_data, HeteroData), "Should return HeteroData object"
        
        # Check that we have expected node types
        node_types = list(hetero_data.node_types)
        assert len(node_types) > 0, "Should have node types"
        
        # Check that we have expected edge types
        edge_types = list(hetero_data.edge_types)
        assert len(edge_types) > 0, "Should have edge types"
        
        # Check that all node types have features
        for node_type in node_types:
            assert hasattr(hetero_data[node_type], 'x'), f"Node type {node_type} should have features"
            assert hetero_data[node_type].x.dtype == torch.float32, "Features should be float32"
    
    def test_node_feature_extraction(self):
        """Test node feature extraction for different node types."""
        # Test player with role embedding
        player_attrs = {
            "type": "batter",
            "role_embedding": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        }
        features = _extract_node_features(player_attrs, "player")
        
        assert isinstance(features, np.ndarray), "Features should be numpy array"
        assert features.dtype == np.float32, "Features should be float32"
        assert len(features) == 4, "Should have 4 features"
        assert features[0] == 1.0, "First feature should be 1.0"
        
        # Test node without specific features
        generic_attrs = {"type": "venue"}
        features = _extract_node_features(generic_attrs, "venue")
        
        assert isinstance(features, np.ndarray), "Features should be numpy array"
        assert len(features) == 4, "Should have default 4 features"
    
    def test_edge_type_determination(self):
        """Test edge type determination logic."""
        # Test batter event edge
        attrs = {"edge_type": "batter_event"}
        edge_type = _determine_hetero_edge_type(attrs, "player", "event")
        assert edge_type == "produced", "Batter event should map to 'produced'"
        
        # Test bowler event edge
        attrs = {"edge_type": "bowler_event"}
        edge_type = _determine_hetero_edge_type(attrs, "player", "event")
        assert edge_type == "conceded", "Bowler event should map to 'conceded'"
        
        # Test player-team edge
        attrs = {"edge_type": "plays_for"}
        edge_type = _determine_hetero_edge_type(attrs, "player", "team")
        assert edge_type == "plays_for", "Should preserve plays_for edge type"
        
        # Test inference from node types
        attrs = {"edge_type": "unknown"}
        edge_type = _determine_hetero_edge_type(attrs, "player", "player")
        assert edge_type == "faced", "Should infer 'faced' for player-player edges"
    
    def test_hetero_data_validation(self, hetero_data):
        """Test HeteroData validation."""
        errors = validate_hetero_data(hetero_data)
        
        if errors:
            pytest.fail(f"HeteroData validation failed with errors: {errors}")
        
        # Should pass validation
        assert len(errors) == 0, "HeteroData should be valid"
    
    def test_hetero_data_stats(self, hetero_data):
        """Test HeteroData statistics extraction."""
        stats = get_hetero_data_stats(hetero_data)
        
        assert isinstance(stats, dict), "Stats should be a dictionary"
        assert "node_types" in stats, "Should include node types"
        assert "edge_types" in stats, "Should include edge types"
        assert "num_node_types" in stats, "Should include number of node types"
        assert "num_edge_types" in stats, "Should include number of edge types"
        
        # Check node type statistics
        for node_type in stats["node_types"]:
            assert f"num_nodes_{node_type}" in stats, f"Should have node count for {node_type}"
            assert f"feature_dim_{node_type}" in stats, f"Should have feature dim for {node_type}"
    
    def test_graph_loading_and_shapes(self, hetero_data):
        """Test that graph loads correctly with proper shapes."""
        # Check node shapes
        for node_type in hetero_data.node_types:
            x = hetero_data[node_type].x
            num_nodes = hetero_data[node_type].num_nodes
            
            assert x.shape[0] == num_nodes, f"Feature tensor size mismatch for {node_type}"
            assert x.shape[1] > 0, f"Feature dimension should be > 0 for {node_type}"
            assert x.dtype == torch.float32, f"Features should be float32 for {node_type}"
        
        # Check edge shapes
        for edge_type in hetero_data.edge_types:
            edge_index = hetero_data[edge_type].edge_index
            
            assert edge_index.shape[0] == 2, f"Edge index should have 2 rows for {edge_type}"
            assert edge_index.dtype == torch.long, f"Edge index should be long for {edge_type}"
            
            # Check edge indices are valid
            src_type, rel_type, dst_type = edge_type
            if src_type in hetero_data.node_types and edge_index.shape[1] > 0:
                max_src_idx = edge_index[0].max().item()
                assert max_src_idx < hetero_data[src_type].num_nodes, \
                    f"Invalid source index for {edge_type}"
            
            if dst_type in hetero_data.node_types and edge_index.shape[1] > 0:
                max_dst_idx = edge_index[1].max().item()
                assert max_dst_idx < hetero_data[dst_type].num_nodes, \
                    f"Invalid target index for {edge_type}"
    
    def test_type_consistency_between_edges_and_nodes(self, hetero_data):
        """Test that edge types are consistent with node types."""
        node_types = set(hetero_data.node_types)
        
        for edge_type in hetero_data.edge_types:
            src_type, rel_type, dst_type = edge_type
            
            # Source and destination types should exist as node types
            assert src_type in node_types, f"Source type {src_type} not in node types"
            assert dst_type in node_types, f"Destination type {dst_type} not in node types"
            
            # Relation type should be valid
            assert isinstance(rel_type, str), "Relation type should be string"
            assert len(rel_type) > 0, "Relation type should not be empty"
    
    def test_feature_integrity_across_node_types(self, hetero_data):
        """Test that features are consistent within each node type."""
        for node_type in hetero_data.node_types:
            x = hetero_data[node_type].x
            
            # All features should have same dimension
            if x.shape[0] > 1:
                first_dim = x.shape[1]
                assert all(row.shape[0] == first_dim for row in x), \
                    f"Inconsistent feature dimensions for {node_type}"
            
            # Features should not contain NaN or infinite values
            assert not torch.isnan(x).any(), f"NaN values in features for {node_type}"
            assert not torch.isinf(x).any(), f"Infinite values in features for {node_type}"
            
            # Features should be in reasonable range
            assert x.abs().max() < 1000, f"Feature values too large for {node_type}"
    
    def test_build_cricket_hetero_graph(self, sample_match_data):
        """Test the main hetero graph building function."""
        hetero_data = build_cricket_hetero_graph(sample_match_data)
        
        assert isinstance(hetero_data, HeteroData), "Should return HeteroData"
        
        # Should have expected node types
        expected_node_types = {"player", "team", "venue", "event", "phase"}
        actual_node_types = set(hetero_data.node_types)
        
        # At least some expected types should be present
        assert len(actual_node_types.intersection(expected_node_types)) > 0, \
            "Should have some expected node types"
        
        # Should have edges
        assert len(hetero_data.edge_types) > 0, "Should have edge types"


class TestHeteroGNNTrainer:
    """Test suite for HeteroData GNN trainer."""
    
    @pytest.fixture
    def sample_hetero_data(self):
        """Create simple HeteroData for testing."""
        data = HeteroData()
        
        # Add player nodes
        data['player'].x = torch.randn(5, 4)
        data['player'].num_nodes = 5
        
        # Add team nodes
        data['team'].x = torch.randn(2, 4)
        data['team'].num_nodes = 2
        
        # Add player-team edges
        data['player', 'plays_for', 'team'].edge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
        
        return data
    
    def test_hetero_trainer_initialization(self, sample_hetero_data):
        """Test HeteroGNN trainer initialization."""
        trainer = CricketHeteroGNNTrainer(sample_hetero_data, embedding_dim=32)
        
        assert trainer.embedding_dim == 32, "Embedding dimension should be set correctly"
        assert len(trainer.node_types) == 2, "Should have 2 node types"
        assert len(trainer.edge_types) == 1, "Should have 1 edge type"
        
        # Check that encoders are created for each node type
        assert "player" in trainer.node_encoders, "Should have player encoder"
        assert "team" in trainer.node_encoders, "Should have team encoder"
    
    def test_hetero_trainer_embeddings(self, sample_hetero_data):
        """Test embedding generation."""
        trainer = CricketHeteroGNNTrainer(sample_hetero_data, embedding_dim=16)
        embeddings = trainer.get_embeddings()
        
        assert isinstance(embeddings, dict), "Embeddings should be dictionary"
        assert "player" in embeddings, "Should have player embeddings"
        assert "team" in embeddings, "Should have team embeddings"
        
        # Check embedding shapes
        player_emb = embeddings["player"]
        team_emb = embeddings["team"]
        
        assert player_emb.shape == (5, 16), "Player embeddings should have correct shape"
        assert team_emb.shape == (2, 16), "Team embeddings should have correct shape"
        assert player_emb.dtype == torch.float32, "Embeddings should be float32"
    
    def test_hetero_trainer_training(self, sample_hetero_data):
        """Test training functionality."""
        trainer = CricketHeteroGNNTrainer(sample_hetero_data, embedding_dim=8)
        
        # Train for a few epochs
        losses = trainer.train(num_epochs=5)
        
        assert len(losses) == 5, "Should have loss for each epoch"
        assert all(isinstance(loss, float) for loss in losses), "Losses should be floats"
        assert all(loss >= 0 for loss in losses), "Losses should be non-negative"
    
    def test_hetero_trainer_stats(self, sample_hetero_data):
        """Test statistics generation."""
        trainer = CricketHeteroGNNTrainer(sample_hetero_data)
        stats = trainer.get_hetero_data_stats()
        
        assert isinstance(stats, dict), "Stats should be dictionary"
        assert "node_types" in stats, "Should include node types"
        assert "edge_types" in stats, "Should include edge types"
        assert "num_nodes_player" in stats, "Should include player count"
        assert "num_nodes_team" in stats, "Should include team count"
        assert "feature_dim_player" in stats, "Should include player feature dim"
        
        # Check values
        assert stats["num_nodes_player"] == 5, "Should have 5 players"
        assert stats["num_nodes_team"] == 2, "Should have 2 teams"
        assert stats["feature_dim_player"] == 4, "Should have 4 player features"
    
    def test_hetero_trainer_save_load(self, sample_hetero_data, tmp_path):
        """Test saving and loading embeddings."""
        trainer = CricketHeteroGNNTrainer(sample_hetero_data, embedding_dim=8)
        
        # Get initial embeddings
        initial_embeddings = trainer.get_embeddings()
        
        # Save embeddings
        save_path = tmp_path / "test_embeddings.pt"
        trainer.save_embeddings(str(save_path))
        
        assert save_path.exists(), "Embeddings file should be created"
        
        # Load embeddings
        loaded_embeddings = trainer.load_embeddings(str(save_path))
        
        assert isinstance(loaded_embeddings, dict), "Loaded embeddings should be dict"
        assert "player" in loaded_embeddings, "Should have player embeddings"
        assert "team" in loaded_embeddings, "Should have team embeddings"
        
        # Check that loaded embeddings match initial ones
        for node_type in ["player", "team"]:
            initial = initial_embeddings[node_type].cpu().numpy()
            loaded = loaded_embeddings[node_type]
            np.testing.assert_array_almost_equal(initial, loaded, decimal=5)


class TestHeteroDataEdgeCases:
    """Test edge cases for HeteroData functionality."""
    
    def test_empty_graph_conversion(self):
        """Test conversion of empty NetworkX graph."""
        G = nx.DiGraph()
        hetero_data = networkx_to_hetero_data(G)
        
        assert isinstance(hetero_data, HeteroData), "Should return HeteroData even for empty graph"
        # May have no node types or empty node types
        
    def test_single_node_graph(self):
        """Test conversion of graph with single node."""
        G = nx.DiGraph()
        G.add_node("player_1", type="batter")
        
        hetero_data = networkx_to_hetero_data(G)
        
        assert isinstance(hetero_data, HeteroData), "Should handle single node graph"
        assert "player" in hetero_data.node_types, "Should have player node type"
        assert hetero_data["player"].num_nodes == 1, "Should have 1 player node"
    
    def test_graph_with_unknown_node_types(self):
        """Test handling of unknown node types."""
        G = nx.DiGraph()
        G.add_node("unknown_1", type="unknown_type")
        G.add_node("unknown_2", type="weird_type")
        
        hetero_data = networkx_to_hetero_data(G)
        
        # Should default unknown types to 'player'
        assert "player" in hetero_data.node_types, "Unknown types should default to player"
        assert hetero_data["player"].num_nodes == 2, "Should have 2 nodes defaulted to player"
    
    def test_graph_with_missing_attributes(self):
        """Test handling of nodes/edges with missing attributes."""
        G = nx.DiGraph()
        G.add_node("node_1")  # No type attribute
        G.add_node("node_2", type="batter")
        G.add_edge("node_1", "node_2")  # No edge_type attribute
        
        hetero_data = networkx_to_hetero_data(G)
        
        # Should handle missing attributes gracefully
        assert isinstance(hetero_data, HeteroData), "Should handle missing attributes"
        assert len(hetero_data.node_types) > 0, "Should have some node types"
    
    def test_validation_with_invalid_data(self):
        """Test validation with invalid HeteroData."""
        data = HeteroData()
        
        # Add node type without features
        data['player'].num_nodes = 5
        # Missing x tensor
        
        errors = validate_hetero_data(data)
        assert len(errors) > 0, "Should detect missing features"
        assert any("missing feature tensor" in error for error in errors), \
            "Should detect missing feature tensor"
    
    @pytest.fixture
    def sample_hetero_data_for_device(self):
        """Create simple HeteroData for device testing."""
        data = HeteroData()
        
        # Add player nodes
        data['player'].x = torch.randn(3, 4)
        data['player'].num_nodes = 3
        
        # Add team nodes
        data['team'].x = torch.randn(2, 4)
        data['team'].num_nodes = 2
        
        return data
    
    def test_device_handling(self, sample_hetero_data_for_device):
        """Test device handling in trainer."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            trainer = CricketHeteroGNNTrainer(sample_hetero_data_for_device, device=device)
            
            assert trainer.device == device, "Should use specified device"
            # Data should be moved to device
            assert trainer.hetero_data['player'].x.device == device, "Data should be on correct device"
        else:
            # Test CPU device
            device = torch.device('cpu')
            trainer = CricketHeteroGNNTrainer(sample_hetero_data_for_device, device=device)
            assert trainer.device == device, "Should use CPU device"


class TestIntegrationHeteroData:
    """Integration tests for HeteroData with full pipeline."""
    
    def test_full_pipeline_integration(self):
        """Test full pipeline from match data to HeteroData training."""
        # Create realistic match data
        match_data = [
            {
                "match_id": "test_match",
                "innings": 1,
                "over": 1,
                "batter_id": "kohli",
                "non_striker_id": "sharma", 
                "bowler_id": "starc",
                "batting_team_name": "india",
                "bowling_team_name": "australia",
                "venue_name": "mcg",
                "bowler_style": "pace",
                "match_date": "2024-01-15 14:30:00",
                "runs": 4,
                "dismissal_type": ""
            },
            {
                "match_id": "test_match",
                "innings": 1,
                "over": 2,
                "batter_id": "sharma",
                "non_striker_id": "kohli",
                "bowler_id": "cummins", 
                "batting_team_name": "india",
                "bowling_team_name": "australia",
                "venue_name": "mcg",
                "bowler_style": "pace",
                "match_date": "2024-01-15 14:30:00",
                "runs": 0,
                "dismissal_type": "bowled"
            }
        ]
        
        # Build HeteroData graph
        hetero_data = build_cricket_hetero_graph(match_data)
        
        # Validate the graph
        errors = validate_hetero_data(hetero_data)
        assert len(errors) == 0, f"Graph validation failed: {errors}"
        
        # Train a model
        trainer = CricketHeteroGNNTrainer(hetero_data, embedding_dim=16)
        losses = trainer.train(num_epochs=3)
        
        assert len(losses) == 3, "Should complete training"
        assert all(isinstance(loss, float) for loss in losses), "Should produce valid losses"
        
        # Get embeddings
        embeddings = trainer.get_embeddings()
        assert len(embeddings) > 0, "Should produce embeddings"
        
        # Check that we have expected node types
        expected_types = {"player", "team", "venue", "event"}
        actual_types = set(embeddings.keys())
        assert len(actual_types.intersection(expected_types)) > 0, \
            "Should have some expected node types in embeddings"