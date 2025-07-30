# Purpose: Unit tests for role embeddings in cricket knowledge graph
# Author: WicketWise Team, Last Modified: 2024-07-19

import pytest
import numpy as np
import networkx as nx
from crickformers.gnn.role_embeddings import (
    get_player_roles,
    create_role_embedding,
    get_role_embedding_for_player,
    add_role_embeddings_to_graph,
    get_role_embedding_stats,
    validate_role_embedding,
    get_role_distribution,
    PLAYER_ROLES,
    ROLE_EMBEDDING_DIM,
    BATTING_ROLES,
    BOWLING_ROLES,
    SPECIALIST_ROLES
)
from crickformers.gnn.graph_builder import build_cricket_graph


class TestRoleEmbeddings:
    """Test suite for role embeddings functionality."""
    
    def test_get_player_roles_known_player(self):
        """Test getting roles for a known player."""
        roles = get_player_roles("kohli")
        assert isinstance(roles, list), "Roles should be returned as a list"
        assert len(roles) > 0, "Known player should have at least one role"
        assert "opener" in roles or "anchor" in roles, "Kohli should have batting roles"
    
    def test_get_player_roles_unknown_player(self):
        """Test getting roles for an unknown player."""
        roles = get_player_roles("unknown_player_xyz")
        assert roles == ["unknown"], "Unknown player should get 'unknown' role"
    
    def test_create_role_embedding_batting_role(self):
        """Test creating embedding for batting roles."""
        embedding = create_role_embedding(["opener"])
        
        assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
        assert embedding.shape == (ROLE_EMBEDDING_DIM,), f"Embedding should have shape ({ROLE_EMBEDDING_DIM},)"
        assert embedding.dtype == np.float32, "Embedding should be float32"
        assert embedding[0] == 1.0, "Batting role should set position 0"
        assert np.sum(embedding) >= 1.0, "Embedding should have at least one non-zero value"
    
    def test_create_role_embedding_bowling_role(self):
        """Test creating embedding for bowling roles."""
        embedding = create_role_embedding(["death_bowler"])
        
        assert embedding[1] == 1.0, "Bowling role should set position 1"
        assert embedding[0] == 0.0, "Non-batting role should not set position 0"
    
    def test_create_role_embedding_specialist_role(self):
        """Test creating embedding for specialist roles."""
        embedding = create_role_embedding(["all_rounder"])
        
        assert embedding[2] == 1.0, "Specialist role should set position 2"
    
    def test_create_role_embedding_unknown_role(self):
        """Test creating embedding for unknown roles."""
        embedding = create_role_embedding(["unknown"])
        
        assert embedding[3] == 1.0, "Unknown role should set position 3"
        assert np.sum(embedding) == 1.0, "Unknown role should only set one position"
    
    def test_create_role_embedding_multiple_roles(self):
        """Test creating embedding for multiple roles."""
        embedding = create_role_embedding(["opener", "death_bowler", "all_rounder"])
        
        assert embedding[0] == 1.0, "Should set batting position"
        assert embedding[1] == 1.0, "Should set bowling position"
        assert embedding[2] == 1.0, "Should set specialist position"
        assert embedding[3] == 0.0, "Should not set unknown position"
        assert np.sum(embedding) == 3.0, "Should have three active positions"
    
    def test_create_role_embedding_empty_roles(self):
        """Test creating embedding for empty role list."""
        embedding = create_role_embedding([])
        
        assert embedding[3] == 1.0, "Empty roles should default to unknown"
        assert np.sum(embedding) == 1.0, "Should only have unknown position set"
    
    def test_get_role_embedding_for_player_known(self):
        """Test getting role embedding for a known player."""
        embedding = get_role_embedding_for_player("kohli")
        
        assert validate_role_embedding(embedding), "Embedding should be valid"
        assert embedding[0] == 1.0, "Kohli should have batting role"
    
    def test_get_role_embedding_for_player_unknown(self):
        """Test getting role embedding for an unknown player."""
        embedding = get_role_embedding_for_player("unknown_player_xyz")
        
        assert validate_role_embedding(embedding), "Embedding should be valid"
        assert embedding[3] == 1.0, "Unknown player should have unknown role"
        assert np.sum(embedding) == 1.0, "Should only have unknown position set"
    
    def test_validate_role_embedding_valid(self):
        """Test validation of a valid role embedding."""
        embedding = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        assert validate_role_embedding(embedding), "Valid embedding should pass validation"
    
    def test_validate_role_embedding_invalid_shape(self):
        """Test validation of embedding with invalid shape."""
        embedding = np.array([1.0, 0.0], dtype=np.float32)
        assert not validate_role_embedding(embedding), "Invalid shape should fail validation"
    
    def test_validate_role_embedding_invalid_dtype(self):
        """Test validation of embedding with invalid dtype."""
        embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        assert not validate_role_embedding(embedding), "Invalid dtype should fail validation"
    
    def test_validate_role_embedding_invalid_values(self):
        """Test validation of embedding with invalid values."""
        embedding = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        assert not validate_role_embedding(embedding), "Non-binary values should fail validation"
    
    def test_validate_role_embedding_all_zeros(self):
        """Test validation of embedding with all zeros."""
        embedding = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert not validate_role_embedding(embedding), "All zeros should fail validation"
    
    def test_add_role_embeddings_to_graph(self):
        """Test adding role embeddings to a knowledge graph."""
        # Create a simple graph with player nodes
        G = nx.DiGraph()
        G.add_node("kohli", type="batter")
        G.add_node("starc", type="bowler")
        G.add_node("venue_1", type="venue")  # Non-player node
        
        # Add role embeddings
        G_updated = add_role_embeddings_to_graph(G)
        
        # Check that player nodes have role embeddings
        assert "role_embedding" in G_updated.nodes["kohli"], "Batter should have role embedding"
        assert "role_embedding" in G_updated.nodes["starc"], "Bowler should have role embedding"
        assert "role_tags" in G_updated.nodes["kohli"], "Batter should have role tags"
        assert "role_tags" in G_updated.nodes["starc"], "Bowler should have role tags"
        
        # Check that non-player nodes don't have role embeddings
        assert "role_embedding" not in G_updated.nodes["venue_1"], "Non-player should not have role embedding"
        
        # Validate the embeddings
        kohli_embedding = G_updated.nodes["kohli"]["role_embedding"]
        starc_embedding = G_updated.nodes["starc"]["role_embedding"]
        
        assert validate_role_embedding(kohli_embedding), "Kohli's embedding should be valid"
        assert validate_role_embedding(starc_embedding), "Starc's embedding should be valid"
    
    def test_add_role_embeddings_with_existing_features(self):
        """Test adding role embeddings to nodes with existing features."""
        G = nx.DiGraph()
        existing_features = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        G.add_node("kohli", type="batter", features=existing_features)
        
        G_updated = add_role_embeddings_to_graph(G)
        
        # Check that features are extended
        updated_features = G_updated.nodes["kohli"]["features"]
        assert len(updated_features) == len(existing_features) + ROLE_EMBEDDING_DIM, \
            "Features should be extended with role embedding"
        
        # Check that original features are preserved
        assert np.array_equal(updated_features[:3], existing_features), \
            "Original features should be preserved"
        
        # Check that role embedding is appended
        role_part = updated_features[3:]
        assert validate_role_embedding(role_part), "Appended role embedding should be valid"
    
    def test_add_role_embeddings_no_existing_features(self):
        """Test adding role embeddings to nodes without existing features."""
        G = nx.DiGraph()
        G.add_node("kohli", type="batter")
        
        G_updated = add_role_embeddings_to_graph(G)
        
        # Check that features are just the role embedding
        features = G_updated.nodes["kohli"]["features"]
        assert len(features) == ROLE_EMBEDDING_DIM, "Features should just be role embedding"
        assert validate_role_embedding(features), "Role embedding should be valid"
    
    def test_get_role_embedding_stats(self):
        """Test getting role embedding statistics."""
        stats = get_role_embedding_stats()
        
        assert "embedding_dim" in stats, "Stats should include embedding dimension"
        assert stats["embedding_dim"] == ROLE_EMBEDDING_DIM, "Dimension should match constant"
        assert "total_players" in stats, "Stats should include total players"
        assert stats["total_players"] == len(PLAYER_ROLES), "Total should match dictionary size"
        assert "batting_roles" in stats, "Stats should include batting roles"
        assert "bowling_roles" in stats, "Stats should include bowling roles"
        assert "specialist_roles" in stats, "Stats should include specialist roles"
    
    def test_get_role_distribution(self):
        """Test getting role distribution."""
        distribution = get_role_distribution()
        
        assert isinstance(distribution, dict), "Distribution should be a dictionary"
        assert len(distribution) > 0, "Distribution should have entries"
        
        # Check that counts are positive integers
        for role, count in distribution.items():
            assert isinstance(count, int), f"Count for {role} should be integer"
            assert count > 0, f"Count for {role} should be positive"
    
    def test_role_embedding_dimension_consistency(self):
        """Test that role embedding dimension is consistent across functions."""
        # Test various players
        test_players = ["kohli", "starc", "unknown_player"]
        
        for player in test_players:
            embedding = get_role_embedding_for_player(player)
            assert embedding.shape == (ROLE_EMBEDDING_DIM,), \
                f"Embedding dimension inconsistent for {player}"
    
    def test_integration_with_graph_builder(self):
        """Test integration with the main graph builder."""
        # Create sample match data
        match_data = [
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
                "match_date": "2024-01-15 14:30:00",
                "runs": 4,
                "dismissal_type": ""
            }
        ]
        
        # Build graph (should automatically include role embeddings)
        G = build_cricket_graph(match_data)
        
        # Check that player nodes have role embeddings
        player_nodes = [node for node, attrs in G.nodes(data=True) 
                       if attrs.get("type") in ["batter", "bowler"]]
        
        assert len(player_nodes) > 0, "Graph should have player nodes"
        
        for player in player_nodes:
            node_attrs = G.nodes[player]
            assert "role_embedding" in node_attrs, f"Player {player} should have role embedding"
            assert "role_tags" in node_attrs, f"Player {player} should have role tags"
            
            embedding = node_attrs["role_embedding"]
            assert validate_role_embedding(embedding), f"Role embedding for {player} should be valid"
    
    def test_extended_feature_dimension_validation(self):
        """Test that extended feature dimensions are correct."""
        G = nx.DiGraph()
        
        # Test with various existing feature dimensions
        test_cases = [
            ([], ROLE_EMBEDDING_DIM),  # No existing features
            ([0.1], 1 + ROLE_EMBEDDING_DIM),  # 1D existing features
            ([0.1, 0.2, 0.3], 3 + ROLE_EMBEDDING_DIM),  # 3D existing features
            (np.random.rand(10), 10 + ROLE_EMBEDDING_DIM),  # 10D existing features
        ]
        
        for i, (existing_features, expected_dim) in enumerate(test_cases):
            node_id = f"player_{i}"
            if len(existing_features) > 0:
                G.add_node(node_id, type="batter", features=np.array(existing_features, dtype=np.float32))
            else:
                G.add_node(node_id, type="batter")
            
            G_updated = add_role_embeddings_to_graph(G)
            
            final_features = G_updated.nodes[node_id]["features"]
            assert len(final_features) == expected_dim, \
                f"Extended features should have dimension {expected_dim}, got {len(final_features)}"
    
    def test_unknown_player_default_role_vector(self):
        """Test that unknown players get the correct default role vector."""
        unknown_players = ["player_not_in_dict", "random_player_123", ""]
        
        for player in unknown_players:
            embedding = get_role_embedding_for_player(player)
            
            # Should have unknown role (position 3 = 1.0)
            assert embedding[3] == 1.0, f"Unknown player {player} should have unknown role"
            assert embedding[0] == 0.0, f"Unknown player {player} should not have batting role"
            assert embedding[1] == 0.0, f"Unknown player {player} should not have bowling role"
            assert embedding[2] == 0.0, f"Unknown player {player} should not have specialist role"
            assert np.sum(embedding) == 1.0, f"Unknown player {player} should have exactly one role"
    
    def test_role_one_hot_encoding_correctness(self):
        """Test that role one-hot encoding is correct for all role types."""
        test_cases = [
            # Batting roles
            (["opener"], [1, 0, 0, 0]),
            (["anchor"], [1, 0, 0, 0]),
            (["finisher"], [1, 0, 0, 0]),
            (["wicketkeeper"], [1, 0, 0, 0]),
            
            # Bowling roles
            (["death_bowler"], [0, 1, 0, 0]),
            (["powerplay_bowler"], [0, 1, 0, 0]),
            (["pace"], [0, 1, 0, 0]),
            (["spin"], [0, 1, 0, 0]),
            
            # Specialist roles
            (["all_rounder"], [0, 0, 1, 0]),
            
            # Unknown roles
            (["unknown"], [0, 0, 0, 1]),
            ([], [0, 0, 0, 1]),  # Empty list should default to unknown
            
            # Multiple roles
            (["opener", "death_bowler"], [1, 1, 0, 0]),
            (["finisher", "all_rounder"], [1, 0, 1, 0]),
            (["anchor", "spin", "all_rounder"], [1, 1, 1, 0]),
        ]
        
        for roles, expected in test_cases:
            embedding = create_role_embedding(roles)
            expected_array = np.array(expected, dtype=np.float32)
            
            assert np.array_equal(embedding, expected_array), \
                f"Roles {roles} should produce embedding {expected}, got {embedding.tolist()}"


class TestRoleEmbeddingEdgeCases:
    """Test edge cases for role embeddings."""
    
    def test_non_string_player_id(self):
        """Test handling of non-string player IDs."""
        # Should not crash, should return unknown role
        embedding = get_role_embedding_for_player(None)
        assert embedding[3] == 1.0, "None player ID should get unknown role"
        
        embedding = get_role_embedding_for_player(123)
        assert embedding[3] == 1.0, "Numeric player ID should get unknown role"
    
    def test_case_sensitivity(self):
        """Test case sensitivity of player IDs."""
        # Assuming our dictionary uses lowercase
        embedding_lower = get_role_embedding_for_player("kohli")
        embedding_upper = get_role_embedding_for_player("KOHLI")
        
        # Upper case should be unknown since our dict uses lowercase
        assert embedding_upper[3] == 1.0, "Case-sensitive lookup should treat KOHLI as unknown"
        assert embedding_lower[0] == 1.0, "Lowercase kohli should have batting role"
    
    def test_graph_with_no_player_nodes(self):
        """Test adding role embeddings to graph with no player nodes."""
        G = nx.DiGraph()
        G.add_node("venue_1", type="venue")
        G.add_node("team_1", type="team")
        
        G_updated = add_role_embeddings_to_graph(G)
        
        # Should not crash, should return same graph
        assert G_updated.number_of_nodes() == 2, "Graph should have same number of nodes"
        assert "role_embedding" not in G_updated.nodes["venue_1"], "Venue should not have role embedding"
        assert "role_embedding" not in G_updated.nodes["team_1"], "Team should not have role embedding"
    
    def test_graph_modification_in_place(self):
        """Test that graph modification works correctly."""
        G = nx.DiGraph()
        G.add_node("kohli", type="batter")
        
        original_id = id(G)
        G_updated = add_role_embeddings_to_graph(G)
        
        # Function should return the same graph object (modified in place)
        assert id(G_updated) == original_id, "Function should modify graph in place"
        assert "role_embedding" in G.nodes["kohli"], "Original graph should be modified"