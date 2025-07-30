# Purpose: Unit tests for video-based style embeddings
# Author: WicketWise Team, Last Modified: 2024-07-19

import pytest
import json
import numpy as np
import networkx as nx
from pathlib import Path
from unittest.mock import patch, mock_open

from crickformers.gnn.style_embeddings import (
    load_style_embeddings_from_json,
    normalize_style_embedding,
    get_style_embedding_for_player,
    add_style_embeddings_to_graph,
    create_sample_style_embeddings,
    get_style_embedding_stats,
    validate_style_embeddings,
    STYLE_EMBEDDING_DIM,
    DEFAULT_STYLE_EMBEDDING
)
from crickformers.gnn.graph_builder import build_cricket_graph, build_cricket_graph_with_style_embeddings


class TestStyleEmbeddingLoading:
    """Test suite for style embedding loading functionality."""
    
    def test_load_style_embeddings_valid_json(self, tmp_path):
        """Test loading valid style embeddings from JSON."""
        # Create test JSON file
        test_embeddings = {
            "kohli": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                     0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "starc": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                     1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
        }
        
        json_file = tmp_path / "test_embeddings.json"
        with open(json_file, 'w') as f:
            json.dump(test_embeddings, f)
        
        # Load embeddings
        loaded_embeddings = load_style_embeddings_from_json(json_file)
        
        assert isinstance(loaded_embeddings, dict), "Should return dictionary"
        assert len(loaded_embeddings) == 2, "Should load 2 embeddings"
        assert "kohli" in loaded_embeddings, "Should contain kohli"
        assert "starc" in loaded_embeddings, "Should contain starc"
        assert loaded_embeddings["kohli"] == test_embeddings["kohli"], "Kohli embedding should match"
        assert loaded_embeddings["starc"] == test_embeddings["starc"], "Starc embedding should match"
    
    def test_load_style_embeddings_nonexistent_file(self):
        """Test loading from nonexistent file."""
        embeddings = load_style_embeddings_from_json("nonexistent.json")
        assert embeddings == {}, "Should return empty dict for nonexistent file"
    
    def test_load_style_embeddings_invalid_json(self, tmp_path):
        """Test loading invalid JSON file."""
        json_file = tmp_path / "invalid.json"
        with open(json_file, 'w') as f:
            f.write("invalid json content {")
        
        embeddings = load_style_embeddings_from_json(json_file)
        assert embeddings == {}, "Should return empty dict for invalid JSON"
    
    def test_load_style_embeddings_invalid_format(self, tmp_path):
        """Test loading JSON with invalid format."""
        # JSON is valid but format is wrong
        invalid_data = ["not", "a", "dict"]
        
        json_file = tmp_path / "invalid_format.json"
        with open(json_file, 'w') as f:
            json.dump(invalid_data, f)
        
        embeddings = load_style_embeddings_from_json(json_file)
        assert embeddings == {}, "Should return empty dict for invalid format"
    
    def test_load_style_embeddings_mixed_validity(self, tmp_path):
        """Test loading JSON with mix of valid and invalid embeddings."""
        test_data = {
            "valid_player": [0.1, 0.2, 0.3, 0.4],
            "invalid_embedding": "not a list",
            "invalid_values": ["not", "numeric", "values"],
            "another_valid": [0.5, 0.6, 0.7, 0.8]
        }
        
        json_file = tmp_path / "mixed.json"
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
        
        embeddings = load_style_embeddings_from_json(json_file)
        
        assert len(embeddings) == 2, "Should load only valid embeddings"
        assert "valid_player" in embeddings, "Should include valid_player"
        assert "another_valid" in embeddings, "Should include another_valid"
        assert "invalid_embedding" not in embeddings, "Should exclude invalid_embedding"
        assert "invalid_values" not in embeddings, "Should exclude invalid_values"


class TestStyleEmbeddingNormalization:
    """Test suite for style embedding normalization."""
    
    def test_normalize_style_embedding_correct_dimension(self):
        """Test normalizing embedding with correct dimension."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                    0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        
        normalized = normalize_style_embedding(embedding, 16)
        
        assert isinstance(normalized, np.ndarray), "Should return numpy array"
        assert normalized.shape == (16,), "Should have correct shape"
        assert normalized.dtype == np.float32, "Should be float32"
        np.testing.assert_array_almost_equal(normalized, embedding, decimal=5)
    
    def test_normalize_style_embedding_truncation(self):
        """Test normalizing embedding that needs truncation."""
        # 20D embedding, target 16D
        embedding = list(range(20))
        
        normalized = normalize_style_embedding(embedding, 16)
        
        assert normalized.shape == (16,), "Should truncate to target dimension"
        np.testing.assert_array_equal(normalized, list(range(16)))
    
    def test_normalize_style_embedding_padding(self):
        """Test normalizing embedding that needs padding."""
        # 10D embedding, target 16D
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        normalized = normalize_style_embedding(embedding, 16)
        
        assert normalized.shape == (16,), "Should pad to target dimension"
        np.testing.assert_array_almost_equal(normalized[:10], embedding, decimal=5)
        np.testing.assert_array_equal(normalized[10:], np.zeros(6))
    
    def test_normalize_style_embedding_empty(self):
        """Test normalizing empty embedding."""
        normalized = normalize_style_embedding([], 16)
        
        assert normalized.shape == (16,), "Should return target dimension"
        np.testing.assert_array_equal(normalized, DEFAULT_STYLE_EMBEDDING)
    
    def test_normalize_style_embedding_custom_dimension(self):
        """Test normalizing with custom target dimension."""
        embedding = [0.1, 0.2, 0.3]
        
        normalized = normalize_style_embedding(embedding, 8)
        
        assert normalized.shape == (8,), "Should use custom target dimension"
        np.testing.assert_array_almost_equal(normalized[:3], embedding, decimal=5)
        np.testing.assert_array_equal(normalized[3:], np.zeros(5))


class TestPlayerStyleEmbeddingRetrieval:
    """Test suite for player style embedding retrieval."""
    
    @pytest.fixture
    def sample_style_embeddings(self):
        """Create sample style embeddings for testing."""
        return {
            "kohli": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                     0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "starc": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                     1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
            "short_embedding": [0.1, 0.2, 0.3, 0.4]  # Will need padding
        }
    
    def test_get_style_embedding_for_existing_player(self, sample_style_embeddings):
        """Test getting style embedding for existing player."""
        embedding = get_style_embedding_for_player("kohli", sample_style_embeddings)
        
        assert isinstance(embedding, np.ndarray), "Should return numpy array"
        assert embedding.shape == (STYLE_EMBEDDING_DIM,), "Should have correct dimension"
        assert embedding.dtype == np.float32, "Should be float32"
        np.testing.assert_array_almost_equal(embedding, sample_style_embeddings["kohli"], decimal=5)
    
    def test_get_style_embedding_for_missing_player(self, sample_style_embeddings):
        """Test getting style embedding for missing player."""
        embedding = get_style_embedding_for_player("unknown_player", sample_style_embeddings)
        
        assert isinstance(embedding, np.ndarray), "Should return numpy array"
        assert embedding.shape == (STYLE_EMBEDDING_DIM,), "Should have correct dimension"
        np.testing.assert_array_equal(embedding, DEFAULT_STYLE_EMBEDDING)
    
    def test_get_style_embedding_with_normalization(self, sample_style_embeddings):
        """Test getting style embedding that requires normalization."""
        embedding = get_style_embedding_for_player("short_embedding", sample_style_embeddings)
        
        assert embedding.shape == (STYLE_EMBEDDING_DIM,), "Should normalize to target dimension"
        np.testing.assert_array_almost_equal(embedding[:4], [0.1, 0.2, 0.3, 0.4], decimal=5)
        np.testing.assert_array_equal(embedding[4:], np.zeros(12))
    
    def test_get_style_embedding_custom_dimension(self, sample_style_embeddings):
        """Test getting style embedding with custom dimension."""
        embedding = get_style_embedding_for_player("kohli", sample_style_embeddings, target_dim=8)
        
        assert embedding.shape == (8,), "Should use custom target dimension"
        np.testing.assert_array_almost_equal(embedding, sample_style_embeddings["kohli"][:8], decimal=5)


class TestGraphStyleEmbeddingIntegration:
    """Test suite for integrating style embeddings into graphs."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create sample NetworkX graph with player nodes."""
        G = nx.DiGraph()
        G.add_node("kohli", type="batter")
        G.add_node("starc", type="bowler")
        G.add_node("venue_1", type="venue")  # Non-player node
        G.add_node("team_1", type="team")    # Non-player node
        return G
    
    @pytest.fixture
    def sample_style_embeddings(self):
        """Create sample style embeddings."""
        return {
            "kohli": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                     0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "starc": [0.2, 0.3, 0.4, 0.5]  # Short embedding to test normalization
        }
    
    def test_add_style_embeddings_to_graph(self, sample_graph, sample_style_embeddings):
        """Test adding style embeddings to graph."""
        updated_graph = add_style_embeddings_to_graph(sample_graph, sample_style_embeddings)
        
        # Check that player nodes have style embeddings
        assert "style_embedding" in updated_graph.nodes["kohli"], "Kohli should have style embedding"
        assert "style_embedding" in updated_graph.nodes["starc"], "Starc should have style embedding"
        
        # Check that non-player nodes don't have style embeddings
        assert "style_embedding" not in updated_graph.nodes["venue_1"], "Venue should not have style embedding"
        assert "style_embedding" not in updated_graph.nodes["team_1"], "Team should not have style embedding"
        
        # Validate the embeddings
        kohli_embedding = updated_graph.nodes["kohli"]["style_embedding"]
        starc_embedding = updated_graph.nodes["starc"]["style_embedding"]
        
        assert isinstance(kohli_embedding, np.ndarray), "Kohli embedding should be numpy array"
        assert isinstance(starc_embedding, np.ndarray), "Starc embedding should be numpy array"
        assert kohli_embedding.shape == (STYLE_EMBEDDING_DIM,), "Kohli embedding should have correct shape"
        assert starc_embedding.shape == (STYLE_EMBEDDING_DIM,), "Starc embedding should have correct shape"
        
        # Check normalization for starc (short embedding)
        np.testing.assert_array_almost_equal(starc_embedding[:4], [0.2, 0.3, 0.4, 0.5], decimal=5)
        np.testing.assert_array_equal(starc_embedding[4:], np.zeros(12))
    
    def test_add_style_embeddings_with_existing_features(self, sample_style_embeddings):
        """Test adding style embeddings to nodes with existing features."""
        G = nx.DiGraph()
        existing_features = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        G.add_node("kohli", type="batter", features=existing_features)
        
        updated_graph = add_style_embeddings_to_graph(G, sample_style_embeddings)
        
        # Check that features are extended
        updated_features = updated_graph.nodes["kohli"]["features"]
        expected_length = len(existing_features) + STYLE_EMBEDDING_DIM
        assert len(updated_features) == expected_length, "Features should be extended with style embedding"
        
        # Check that original features are preserved
        np.testing.assert_array_almost_equal(updated_features[:4], existing_features, decimal=5)
        
        # Check that style embedding is appended
        style_part = updated_features[4:]
        expected_style = sample_style_embeddings["kohli"]
        np.testing.assert_array_almost_equal(style_part, expected_style, decimal=5)
    
    def test_add_style_embeddings_no_existing_features(self, sample_style_embeddings):
        """Test adding style embeddings to nodes without existing features."""
        G = nx.DiGraph()
        G.add_node("kohli", type="batter")
        
        updated_graph = add_style_embeddings_to_graph(G, sample_style_embeddings)
        
        # Check that features are just the style embedding
        features = updated_graph.nodes["kohli"]["features"]
        assert len(features) == STYLE_EMBEDDING_DIM, "Features should just be style embedding"
        np.testing.assert_array_almost_equal(features, sample_style_embeddings["kohli"], decimal=5)
    
    def test_add_style_embeddings_missing_players(self, sample_graph):
        """Test adding style embeddings when players are missing from embeddings dict."""
        # Empty embeddings dict
        empty_embeddings = {}
        
        updated_graph = add_style_embeddings_to_graph(sample_graph, empty_embeddings)
        
        # All players should get default embeddings
        kohli_embedding = updated_graph.nodes["kohli"]["style_embedding"]
        starc_embedding = updated_graph.nodes["starc"]["style_embedding"]
        
        np.testing.assert_array_equal(kohli_embedding, DEFAULT_STYLE_EMBEDDING)
        np.testing.assert_array_equal(starc_embedding, DEFAULT_STYLE_EMBEDDING)


class TestStyleEmbeddingUtilities:
    """Test suite for style embedding utility functions."""
    
    def test_create_sample_style_embeddings(self):
        """Test creating sample style embeddings."""
        player_ids = ["kohli", "starc", "sharma"]
        
        embeddings = create_sample_style_embeddings(player_ids, embedding_dim=8)
        
        assert isinstance(embeddings, dict), "Should return dictionary"
        assert len(embeddings) == 3, "Should create embeddings for all players"
        
        for player_id in player_ids:
            assert player_id in embeddings, f"Should have embedding for {player_id}"
            embedding = embeddings[player_id]
            assert len(embedding) == 8, "Should have correct dimension"
            assert all(isinstance(x, (int, float)) for x in embedding), "All values should be numeric"
    
    def test_create_sample_style_embeddings_with_save(self, tmp_path):
        """Test creating and saving sample style embeddings."""
        player_ids = ["kohli", "starc"]
        save_path = tmp_path / "sample_embeddings.json"
        
        embeddings = create_sample_style_embeddings(player_ids, save_path=save_path)
        
        # Check that file was created
        assert save_path.exists(), "Sample embeddings file should be created"
        
        # Check that file contents match returned embeddings
        with open(save_path, 'r') as f:
            saved_embeddings = json.load(f)
        
        assert saved_embeddings == embeddings, "Saved embeddings should match returned embeddings"
    
    def test_get_style_embedding_stats(self):
        """Test getting style embedding statistics."""
        embeddings = {
            "player1": [0.1, 0.2, 0.3, 0.4],
            "player2": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "player3": [1.1, 1.2]
        }
        
        stats = get_style_embedding_stats(embeddings)
        
        assert isinstance(stats, dict), "Should return dictionary"
        assert stats["num_players"] == 3, "Should count players correctly"
        assert stats["embedding_dims"] == [4, 6, 2], "Should record embedding dimensions"
        assert stats["min_embedding_dim"] == 2, "Should find minimum dimension"
        assert stats["max_embedding_dim"] == 6, "Should find maximum dimension"
        assert abs(stats["avg_embedding_dim"] - 4.0) < 0.01, "Should calculate average dimension"
        assert "value_min" in stats, "Should include value statistics"
        assert "value_max" in stats, "Should include value statistics"
    
    def test_get_style_embedding_stats_empty(self):
        """Test getting statistics for empty embeddings."""
        stats = get_style_embedding_stats({})
        
        assert stats["num_players"] == 0, "Should handle empty embeddings"
        assert stats["embedding_dims"] == [], "Should have empty dimensions list"
    
    def test_validate_style_embeddings_valid(self):
        """Test validating valid style embeddings."""
        embeddings = {
            "player1": [0.1, 0.2, 0.3, 0.4],
            "player2": [0.5, 0.6, 0.7, 0.8]
        }
        
        errors = validate_style_embeddings(embeddings)
        assert errors == [], "Valid embeddings should have no errors"
    
    def test_validate_style_embeddings_invalid(self):
        """Test validating invalid style embeddings."""
        embeddings = {
            123: [0.1, 0.2],  # Invalid player ID type
            "player2": "not a list",  # Invalid embedding type
            "player3": [],  # Empty embedding
            "player4": [0.1, float('nan'), 0.3],  # NaN values
            "player5": [0.1, float('inf'), 0.3],  # Infinite values
        }
        
        errors = validate_style_embeddings(embeddings)
        assert len(errors) > 0, "Invalid embeddings should have errors"
        
        # Check specific error types
        error_text = " ".join(errors)
        assert "Player ID must be string" in error_text, "Should detect invalid player ID type"
        assert "must be list" in error_text, "Should detect invalid embedding type"
        assert "Empty embedding" in error_text, "Should detect empty embeddings"
        assert "NaN or infinite" in error_text, "Should detect NaN/infinite values"


class TestBuildGraphWithStyleEmbeddings:
    """Test suite for building graphs with style embeddings."""
    
    @pytest.fixture
    def sample_match_data(self):
        """Create sample match data."""
        return [
            {
                "match_id": "match_1",
                "innings": 1,
                "over": 1,
                "batter_id": "kohli",
                "bowler_id": "starc",
                "batting_team_name": "india",
                "bowling_team_name": "australia",
                "venue_name": "mcg",
                "match_date": "2024-01-15 14:30:00",
                "runs": 4,
                "dismissal_type": ""
            }
        ]
    
    def test_build_graph_with_style_embeddings_dict(self, sample_match_data):
        """Test building graph with style embeddings from dictionary."""
        style_embeddings = {
            "kohli": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                     0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "starc": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                     1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
        }
        
        G = build_cricket_graph_with_style_embeddings(
            sample_match_data, 
            style_embeddings_dict=style_embeddings
        )
        
        # Check that players have style embeddings
        assert "style_embedding" in G.nodes["kohli"], "Kohli should have style embedding"
        assert "style_embedding" in G.nodes["starc"], "Starc should have style embedding"
        
        # Validate embeddings
        kohli_embedding = G.nodes["kohli"]["style_embedding"]
        np.testing.assert_array_almost_equal(kohli_embedding, style_embeddings["kohli"], decimal=5)
    
    def test_build_graph_with_style_embeddings_file(self, sample_match_data, tmp_path):
        """Test building graph with style embeddings from file."""
        # Create test embeddings file
        style_embeddings = {
            "kohli": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                     0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        }
        
        embeddings_file = tmp_path / "style_embeddings.json"
        with open(embeddings_file, 'w') as f:
            json.dump(style_embeddings, f)
        
        G = build_cricket_graph_with_style_embeddings(
            sample_match_data,
            style_embeddings_path=str(embeddings_file)
        )
        
        # Check that player has style embedding
        assert "style_embedding" in G.nodes["kohli"], "Kohli should have style embedding"
        
        kohli_embedding = G.nodes["kohli"]["style_embedding"]
        np.testing.assert_array_almost_equal(kohli_embedding, style_embeddings["kohli"], decimal=5)
    
    def test_build_graph_with_no_style_embeddings(self, sample_match_data):
        """Test building graph with no style embeddings provided."""
        G = build_cricket_graph_with_style_embeddings(sample_match_data)
        
        # Players should still have style embeddings (default ones)
        assert "style_embedding" in G.nodes["kohli"], "Kohli should have default style embedding"
        assert "style_embedding" in G.nodes["starc"], "Starc should have default style embedding"
        
        # Should be default embeddings
        kohli_embedding = G.nodes["kohli"]["style_embedding"]
        np.testing.assert_array_equal(kohli_embedding, DEFAULT_STYLE_EMBEDDING)
    
    def test_build_graph_dict_overrides_file(self, sample_match_data, tmp_path):
        """Test that dictionary embeddings override file embeddings."""
        # Create file with one embedding
        file_embeddings = {"kohli": [0.1, 0.2, 0.3, 0.4]}
        embeddings_file = tmp_path / "style_embeddings.json"
        with open(embeddings_file, 'w') as f:
            json.dump(file_embeddings, f)
        
        # Provide different dictionary embeddings
        dict_embeddings = {
            "kohli": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
                     1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        }
        
        G = build_cricket_graph_with_style_embeddings(
            sample_match_data,
            style_embeddings_path=str(embeddings_file),
            style_embeddings_dict=dict_embeddings
        )
        
        # Should use dictionary embeddings, not file embeddings
        kohli_embedding = G.nodes["kohli"]["style_embedding"]
        np.testing.assert_array_almost_equal(kohli_embedding, dict_embeddings["kohli"], decimal=5)


class TestStyleEmbeddingFeatureShapes:
    """Test suite for validating feature shapes with style embeddings."""
    
    def test_resulting_feature_shape_with_role_and_style(self):
        """Test that feature shapes are correct when both role and style embeddings are present."""
        from crickformers.gnn.role_embeddings import ROLE_EMBEDDING_DIM, add_role_embeddings_to_graph
        
        # Create a graph and add role embeddings first (as done in the normal pipeline)
        G = nx.DiGraph()
        G.add_node("kohli", type="batter", role_embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        
        # Add role embeddings to features (this is what happens in the normal pipeline)
        G = add_role_embeddings_to_graph(G)
        
        # Now add style embeddings
        style_embeddings = {
            "kohli": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                     0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]  # 16D style embedding
        }
        
        updated_graph = add_style_embeddings_to_graph(G, style_embeddings)
        
        # Check final feature shape
        final_features = updated_graph.nodes["kohli"]["features"]
        expected_shape = ROLE_EMBEDDING_DIM + STYLE_EMBEDDING_DIM  # 4 + 16 = 20
        
        assert len(final_features) == expected_shape, f"Features should have shape {expected_shape}"
        
        # Check that role embedding comes first
        np.testing.assert_array_almost_equal(final_features[:ROLE_EMBEDDING_DIM], [1.0, 0.0, 0.0, 0.0], decimal=5)
        
        # Check that style embedding comes after
        np.testing.assert_array_almost_equal(final_features[ROLE_EMBEDDING_DIM:], style_embeddings["kohli"], decimal=5)
    
    def test_feature_shape_consistency_across_players(self):
        """Test that all players have consistent feature shapes."""
        # Create graph with multiple players
        G = nx.DiGraph()
        G.add_node("kohli", type="batter")
        G.add_node("starc", type="bowler")
        G.add_node("sharma", type="batter")
        
        # Style embeddings with different original dimensions
        style_embeddings = {
            "kohli": list(range(16)),  # Full 16D
            "starc": list(range(8)),   # 8D (will be padded)
            # sharma missing (will get default)
        }
        
        updated_graph = add_style_embeddings_to_graph(G, style_embeddings)
        
        # All players should have same feature dimension
        kohli_features = updated_graph.nodes["kohli"]["features"]
        starc_features = updated_graph.nodes["starc"]["features"]
        sharma_features = updated_graph.nodes["sharma"]["features"]
        
        assert len(kohli_features) == STYLE_EMBEDDING_DIM, "Kohli should have correct feature dim"
        assert len(starc_features) == STYLE_EMBEDDING_DIM, "Starc should have correct feature dim"
        assert len(sharma_features) == STYLE_EMBEDDING_DIM, "Sharma should have correct feature dim"
        
        # Check specific values
        np.testing.assert_array_equal(kohli_features, list(range(16)))
        
        # Starc should be padded
        expected_starc = list(range(8)) + [0] * 8
        np.testing.assert_array_equal(starc_features, expected_starc)
        
        # Sharma should be default
        np.testing.assert_array_equal(sharma_features, DEFAULT_STYLE_EMBEDDING)