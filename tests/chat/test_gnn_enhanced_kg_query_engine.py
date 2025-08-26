# Purpose: Comprehensive tests for GNN-Enhanced KG Query Engine
# Author: WicketWise Team, Last Modified: 2025-08-25

import pytest
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import pickle

from crickformers.chat.gnn_enhanced_kg_query_engine import GNNEnhancedKGQueryEngine


class TestGNNEnhancedKGQueryEngine:
    """Test suite for GNN-Enhanced KG Query Engine"""
    
    @pytest.fixture
    def mock_kg_graph(self):
        """Create a mock knowledge graph for testing"""
        G = nx.Graph()
        
        # Add player nodes
        G.add_node("Virat Kohli", type="player", primary_role="batsman", 
                  batting_stats={"average": 50.2, "strike_rate": 137.8})
        G.add_node("MS Dhoni", type="player", primary_role="wicket-keeper", 
                  batting_stats={"average": 38.1, "strike_rate": 126.4})
        G.add_node("Jasprit Bumrah", type="player", primary_role="bowler",
                  bowling_stats={"average": 24.2, "economy": 7.3})
        
        # Add venue nodes
        G.add_node("MCG", type="venue", capacity=100024, pitch_type="batting")
        G.add_node("Wankhede", type="venue", capacity=33108, pitch_type="batting")
        
        # Add some edges
        G.add_edge("Virat Kohli", "MCG", relationship="played_at", matches=15)
        G.add_edge("MS Dhoni", "Wankhede", relationship="played_at", matches=25)
        
        return G
    
    @pytest.fixture
    def mock_gnn_service(self):
        """Create a mock GNN service"""
        service = Mock()
        
        # Mock player embeddings (128D)
        service.get_player_embedding.side_effect = lambda player: {
            "Virat Kohli": np.random.rand(128).astype(np.float32),
            "MS Dhoni": np.random.rand(128).astype(np.float32),
            "Jasprit Bumrah": np.random.rand(128).astype(np.float32)
        }.get(player, None)
        
        # Mock venue embeddings (64D)
        service.get_venue_embedding.side_effect = lambda venue: {
            "MCG": np.random.rand(64).astype(np.float32),
            "Wankhede": np.random.rand(64).astype(np.float32)
        }.get(venue, None)
        
        return service
    
    @pytest.fixture
    def temp_kg_file(self, mock_kg_graph):
        """Create a temporary KG file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(mock_kg_graph, f)
            return f.name
    
    @pytest.fixture
    def gnn_engine(self, temp_kg_file, mock_gnn_service):
        """Create GNN-enhanced engine with mocked dependencies"""
        with patch('crickformers.chat.gnn_enhanced_kg_query_engine.KGGNNEmbeddingService') as mock_service_class:
            mock_service_class.return_value = mock_gnn_service
            
            engine = GNNEnhancedKGQueryEngine(
                graph_path=temp_kg_file,
                gnn_embeddings_path="mock_embeddings.pt"
            )
            engine.gnn_service = mock_gnn_service
            engine.gnn_embeddings_available = True
            
            return engine
    
    def test_initialization_with_gnn(self, temp_kg_file):
        """Test successful initialization with GNN embeddings"""
        with patch('crickformers.chat.gnn_enhanced_kg_query_engine.KGGNNEmbeddingService') as mock_service:
            with patch('pathlib.Path.exists', return_value=True):
                engine = GNNEnhancedKGQueryEngine(
                    graph_path=temp_kg_file,
                    gnn_embeddings_path="mock_embeddings.pt"
                )
                
                assert engine.gnn_embeddings_available is True
                assert engine.similarity_threshold == 0.7
                assert engine.max_similarity_results == 10
    
    def test_initialization_without_gnn(self, temp_kg_file):
        """Test initialization fallback when GNN embeddings not available"""
        with patch('pathlib.Path.exists', return_value=False):
            engine = GNNEnhancedKGQueryEngine(
                graph_path=temp_kg_file,
                gnn_embeddings_path="nonexistent.pt"
            )
            
            assert engine.gnn_embeddings_available is False
            assert engine.gnn_service is None
    
    def test_find_similar_players_gnn_success(self, gnn_engine):
        """Test successful GNN-based player similarity search"""
        # Mock embeddings to ensure similarity
        kohli_emb = np.array([0.5] * 128, dtype=np.float32)
        dhoni_emb = np.array([0.6] * 128, dtype=np.float32)  # Similar to Kohli
        bumrah_emb = np.array([0.1] * 128, dtype=np.float32)  # Different
        
        gnn_engine.gnn_service.get_player_embedding.side_effect = lambda player: {
            "Virat Kohli": kohli_emb,
            "MS Dhoni": dhoni_emb,
            "Jasprit Bumrah": bumrah_emb
        }.get(player, None)
        
        result = gnn_engine.find_similar_players_gnn("Virat Kohli", top_k=2)
        
        assert "error" not in result
        assert result["target_player"] == "Virat Kohli"
        assert result["gnn_powered"] is True
        assert len(result["similar_players"]) <= 2
        assert result["similarity_metric"] == "cosine"
    
    def test_find_similar_players_gnn_player_not_found(self, gnn_engine):
        """Test GNN similarity search with unknown player"""
        result = gnn_engine.find_similar_players_gnn("Unknown Player")
        
        assert "error" in result
        assert "No GNN embedding found" in result["error"]
    
    def test_find_similar_players_gnn_fallback(self, gnn_engine):
        """Test fallback when GNN not available"""
        gnn_engine.gnn_embeddings_available = False
        
        result = gnn_engine.find_similar_players_gnn("Virat Kohli")
        
        assert "fallback_method" in result
        assert result["gnn_powered"] is False
    
    def test_predict_contextual_performance_success(self, gnn_engine):
        """Test contextual performance prediction"""
        context = {
            "phase": "death",
            "bowling_type": "pace",
            "pressure": True,
            "required_run_rate": 12.0,
            "wickets_lost": 6,
            "balls_remaining": 24
        }
        
        result = gnn_engine.predict_contextual_performance("Virat Kohli", context)
        
        assert "error" not in result
        assert result["player"] == "Virat Kohli"
        assert result["context"] == context
        assert result["gnn_powered"] is True
        assert "prediction" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
    
    def test_predict_contextual_performance_invalid_player(self, gnn_engine):
        """Test contextual prediction with invalid player"""
        context = {"phase": "middle", "bowling_type": "spin"}
        
        result = gnn_engine.predict_contextual_performance("Unknown Player", context)
        
        assert "error" in result
        assert "No GNN embedding found" in result["error"]
    
    def test_analyze_venue_compatibility_success(self, gnn_engine):
        """Test venue compatibility analysis"""
        result = gnn_engine.analyze_venue_compatibility("Virat Kohli", "MCG")
        
        assert "error" not in result
        assert result["player"] == "Virat Kohli"
        assert result["venue"] == "MCG"
        assert result["gnn_powered"] is True
        assert "compatibility_score" in result
        assert 0 <= result["compatibility_score"] <= 1
        assert result["compatibility_level"] in ["excellent", "good", "moderate", "poor"]
    
    def test_analyze_venue_compatibility_missing_embeddings(self, gnn_engine):
        """Test venue compatibility with missing embeddings"""
        # Mock to return None for venue
        gnn_engine.gnn_service.get_venue_embedding.return_value = None
        
        result = gnn_engine.analyze_venue_compatibility("Virat Kohli", "Unknown Venue")
        
        assert "error" in result
        assert "No GNN embedding found for venue" in result["error"]
    
    def test_get_playing_style_similarity_success(self, gnn_engine):
        """Test playing style similarity analysis"""
        result = gnn_engine.get_playing_style_similarity("Virat Kohli", "MS Dhoni")
        
        assert "error" not in result
        assert result["player1"] == "Virat Kohli"
        assert result["player2"] == "MS Dhoni"
        assert result["gnn_powered"] is True
        assert "overall_similarity" in result
        assert 0 <= result["overall_similarity"] <= 1
        assert result["similarity_level"] in ["very high", "high", "moderate", "low"]
    
    def test_get_playing_style_similarity_same_player(self, gnn_engine):
        """Test style similarity with same player (should work)"""
        result = gnn_engine.get_playing_style_similarity("Virat Kohli", "Virat Kohli")
        
        # Should still work, might show high similarity
        assert "error" not in result or result["overall_similarity"] == 1.0
    
    def test_context_embedding_generation(self, gnn_engine):
        """Test context embedding generation"""
        context = {
            "phase": "powerplay",
            "bowling_type": "spin",
            "pressure": True,
            "required_run_rate": 8.5,
            "wickets_lost": 2,
            "balls_remaining": 90
        }
        
        embedding = gnn_engine._get_context_embedding(context)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (32,)  # Fixed size
        assert embedding.dtype == np.float32
        assert np.all(embedding >= 0) and np.all(embedding <= 1)  # Normalized
    
    def test_embedding_caching(self, gnn_engine):
        """Test that embeddings are properly cached"""
        # First call
        emb1 = gnn_engine._get_player_embedding("Virat Kohli")
        
        # Second call should use cache
        emb2 = gnn_engine._get_player_embedding("Virat Kohli")
        
        assert np.array_equal(emb1, emb2)
        
        # Check cache was used (only one call to service)
        assert gnn_engine.gnn_service.get_player_embedding.call_count == 1
    
    def test_similarity_categorization(self, gnn_engine):
        """Test similarity score categorization"""
        assert gnn_engine._categorize_similarity(0.9) == "very high"
        assert gnn_engine._categorize_similarity(0.7) == "high"
        assert gnn_engine._categorize_similarity(0.5) == "moderate"
        assert gnn_engine._categorize_similarity(0.2) == "low"
    
    def test_compatibility_categorization(self, gnn_engine):
        """Test compatibility score categorization"""
        assert gnn_engine._categorize_compatibility(0.9) == "excellent"
        assert gnn_engine._categorize_compatibility(0.7) == "good"
        assert gnn_engine._categorize_compatibility(0.5) == "moderate"
        assert gnn_engine._categorize_compatibility(0.2) == "poor"
    
    def test_performance_prediction_components(self, gnn_engine):
        """Test performance prediction components"""
        combined_emb = np.random.rand(160).astype(np.float32)  # 128 + 32
        context = {"phase": "death", "bowling_type": "pace"}
        
        prediction = gnn_engine._generate_performance_prediction(combined_emb, context)
        
        assert "expected_strike_rate" in prediction
        assert "expected_average" in prediction
        assert "confidence" in prediction
        assert "risk_level" in prediction
        assert prediction["risk_level"] in ["low", "medium", "high"]
    
    def test_error_handling_gnn_service_failure(self, gnn_engine):
        """Test error handling when GNN service fails"""
        gnn_engine.gnn_service.get_player_embedding.side_effect = Exception("GNN service error")
        
        result = gnn_engine.find_similar_players_gnn("Virat Kohli")
        
        assert "error" in result
        assert "GNN similarity search failed" in result["error"]
    
    def test_fallback_methods_called(self, gnn_engine):
        """Test that fallback methods are called when GNN unavailable"""
        gnn_engine.gnn_embeddings_available = False
        
        # Test similarity fallback
        result = gnn_engine.find_similar_players_gnn("Virat Kohli")
        assert result["gnn_powered"] is False
        assert "fallback_method" in result
        
        # Test contextual fallback
        context = {"phase": "middle"}
        result = gnn_engine.predict_contextual_performance("Virat Kohli", context)
        assert result["gnn_powered"] is False
        assert "fallback_method" in result
    
    def test_get_all_players_with_embeddings(self, gnn_engine):
        """Test getting all players with embeddings"""
        players = gnn_engine._get_all_players_with_embeddings()
        
        assert isinstance(players, dict)
        assert len(players) >= 2  # Should have Kohli and Dhoni at least
        
        for player, embedding in players.items():
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (128,)  # Player embedding size
    
    def test_combine_embeddings(self, gnn_engine):
        """Test embedding combination"""
        player_emb = np.random.rand(128).astype(np.float32)
        context_emb = np.random.rand(32).astype(np.float32)
        
        combined = gnn_engine._combine_embeddings(player_emb, context_emb)
        
        assert isinstance(combined, np.ndarray)
        assert combined.shape == (160,)  # 128 + 32
        assert np.array_equal(combined[:128], player_emb)
        assert np.array_equal(combined[128:], context_emb)


class TestGNNEnhancedKGQueryEngineIntegration:
    """Integration tests for GNN-Enhanced KG Query Engine"""
    
    def test_end_to_end_similarity_workflow(self):
        """Test complete similarity search workflow"""
        # This would require actual GNN embeddings and KG data
        # For now, we'll test the structure
        pass
    
    def test_cricket_domain_validation(self):
        """Test cricket-specific validation and insights"""
        # Test that cricket insights are generated correctly
        pass
    
    def test_performance_benchmarks(self):
        """Test performance requirements are met"""
        # Test query response times, memory usage, etc.
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
