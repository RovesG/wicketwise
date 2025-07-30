# Purpose: Unit tests for chat_tools.py - validates routing logic and tool function calls
# Author: Phi1618 Cricket AI, Last Modified: 2025-01-17

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import chat_tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat_tools import (
    handle_chat_query,
    query_kg_player_relationship,
    predict_with_uncertainty,
    get_form_vector
)

class TestChatToolsRouting:
    """Test suite for chat tools routing functionality"""
    
    def test_form_query_routing(self):
        """Test that form-related queries route to form analysis tool"""
        # Test various form query patterns
        form_queries = [
            "What is Virat Kohli's form?",
            "Show me the form analysis for MS Dhoni",
            "How is Rohit Sharma's current form?",
            "form analysis for Kane Williamson"
        ]
        
        for query in form_queries:
            result = handle_chat_query(query)
            assert "**Form Analysis Tool Used**" in result
            assert "Form Score:" in result
            assert "Recent Performance:" in result
            assert "Trend:" in result
            assert "*Analysis generated using form vector analysis*" in result
    
    def test_matchup_query_routing(self):
        """Test that matchup-related queries route to knowledge graph tool"""
        # Test various matchup query patterns
        matchup_queries = [
            "What is the matchup between Virat Kohli and Jasprit Bumrah?",
            "Show me the history of MS Dhoni vs Mitchell Starc",
            "How do these players matchup against each other?",
            "history of confrontations between these players"
        ]
        
        for query in matchup_queries:
            result = handle_chat_query(query)
            assert "**Knowledge Graph Tool Used**" in result
            assert "Head-to-Head Stats:" in result
            assert "Total Matches:" in result
            assert "*Analysis generated using knowledge graph relationship queries*" in result
    
    def test_history_query_routing(self):
        """Test that history-related queries route to knowledge graph tool"""
        # Test various history query patterns
        history_queries = [
            "What's the history of Virat Kohli?",
            "Show me career history for MS Dhoni",
            "batting history analysis",
            "historical performance data"
        ]
        
        for query in history_queries:
            result = handle_chat_query(query)
            assert "**Knowledge Graph Tool Used**" in result
            assert "Relationship Type:" in result
            assert "career_history" in result
    
    def test_prediction_query_routing(self):
        """Test that prediction-related queries route to prediction tool"""
        # Test various prediction query patterns
        prediction_queries = [
            "What is the prediction for next match?",
            "Predict the outcome of this game",
            "Can you predict the winner?",
            "prediction for runs in next over"
        ]
        
        for query in prediction_queries:
            result = handle_chat_query(query)
            assert "**Prediction Tool Used**" in result
            assert "Scenario:" in result
            assert "Prediction:" in result
            assert "Confidence:" in result
            assert "Uncertainty Range:" in result
            assert "*Analysis generated using prediction model with uncertainty quantification*" in result
    
    def test_fallback_response(self):
        """Test fallback response for unmatched queries"""
        # Test various queries that don't match any pattern
        unmatched_queries = [
            "Hello there",
            "How are you?",
            "What's the weather like?",
            "Tell me about cricket rules",
            "Random question about nothing specific"
        ]
        
        for query in unmatched_queries:
            result = handle_chat_query(query)
            assert "**General Query Handler**" in result
            assert "Status: No specific tool matched" in result
            assert "Available tools:" in result
            assert "Form analysis (keywords: 'form')" in result
            assert "Player relationships (keywords: 'matchup', 'history')" in result
            assert "Predictions (keywords: 'prediction', 'predict')" in result
            assert "*General response - no specialized tool used*" in result
    
    def test_context_passing(self):
        """Test that context is properly passed to underlying functions"""
        test_context = {
            "match_id": "match_001",
            "team_a": "India",
            "team_b": "Australia",
            "over": 15
        }
        
        # Test form query with context
        form_result = handle_chat_query("What is Virat Kohli's form?", test_context)
        assert "**Form Analysis Tool Used**" in form_result
        assert "Virat Kohli" in form_result
        
        # Test matchup query with context
        matchup_result = handle_chat_query("Virat Kohli vs Pat Cummins matchup", test_context)
        assert "**Knowledge Graph Tool Used**" in matchup_result
        
        # Test prediction query with context
        prediction_result = handle_chat_query("Predict the match outcome", test_context)
        assert "**Prediction Tool Used**" in prediction_result
    
    def test_case_insensitive_routing(self):
        """Test that routing works regardless of case"""
        # Test uppercase keywords
        assert "**Form Analysis Tool Used**" in handle_chat_query("FORM analysis please")
        assert "**Knowledge Graph Tool Used**" in handle_chat_query("MATCHUP between players")
        assert "**Prediction Tool Used**" in handle_chat_query("PREDICTION for next ball")
        
        # Test mixed case keywords
        assert "**Form Analysis Tool Used**" in handle_chat_query("Form Analysis Request")
        assert "**Knowledge Graph Tool Used**" in handle_chat_query("History Analysis")
        assert "**Prediction Tool Used**" in handle_chat_query("Predict Outcome")

class TestIndividualFunctions:
    """Test suite for individual function behavior"""
    
    def test_get_form_vector_function(self):
        """Test get_form_vector function behavior"""
        result = get_form_vector("Test Player")
        
        assert isinstance(result, dict)
        assert "player" in result
        assert "form_score" in result
        assert "recent_performance" in result
        assert "trend" in result
        assert "vector_features" in result
        
        assert result["player"] == "Test Player"
        assert isinstance(result["form_score"], (int, float))
        assert isinstance(result["vector_features"], list)
    
    def test_query_kg_player_relationship_function(self):
        """Test query_kg_player_relationship function behavior"""
        test_query = "What is the matchup between Virat Kohli and Pat Cummins?"
        result = query_kg_player_relationship(test_query)
        
        assert isinstance(result, dict)
        assert "query" in result
        assert "players_found" in result
        assert "relationship_type" in result
        assert "head_to_head" in result
        assert "context_used" in result
        
        assert result["query"] == test_query
        assert "matchup" in result["relationship_type"]
        assert isinstance(result["head_to_head"], dict)
        assert "matches" in result["head_to_head"]
        assert "wins" in result["head_to_head"]
        assert "avg_runs" in result["head_to_head"]
    
    def test_predict_with_uncertainty_function(self):
        """Test predict_with_uncertainty function behavior"""
        scenarios = [
            ("Who will win the match?", "win"),
            ("How many runs will be scored?", "runs"),
            ("Will there be a wicket?", "wicket"),
            ("What will happen next?", "general")
        ]
        
        for scenario, expected_type in scenarios:
            result = predict_with_uncertainty(scenario)
            
            assert isinstance(result, dict)
            assert "scenario" in result
            assert "prediction" in result
            assert "confidence" in result
            assert "uncertainty_range" in result
            assert "model_used" in result
            assert "context_factors" in result
            
            assert result["scenario"] == scenario
            assert isinstance(result["confidence"], (int, float))
            assert 0 <= result["confidence"] <= 1
            assert "%" in result["uncertainty_range"]

class TestPlayerNameExtraction:
    """Test suite for player name extraction functionality"""
    
    def test_player_name_extraction_in_form_queries(self):
        """Test that player names are correctly extracted from form queries"""
        test_cases = [
            ("What is Virat Kohli's form?", "Virat Kohli"),
            ("Show me MS Dhoni form analysis", "MS Dhoni"),
            ("Kane Williamson form report", "Kane Williamson"),
            ("form analysis", "Unknown Player")  # No player name found
        ]
        
        for query, expected_player in test_cases:
            result = handle_chat_query(query)
            assert expected_player in result
    
    def test_player_name_extraction_in_matchup_queries(self):
        """Test that player names are correctly extracted from matchup queries"""
        test_cases = [
            ("Virat Kohli vs Pat Cummins matchup", ["Virat Kohli", "Pat Cummins"]),
            ("MS Dhoni history against Mitchell Starc", ["MS Dhoni", "Mitchell Starc"]),
            ("matchup analysis", [])  # No player names found
        ]
        
        for query, expected_players in test_cases:
            result = query_kg_player_relationship(query)
            assert result["players_found"] == expected_players

class TestErrorHandling:
    """Test suite for error handling and edge cases"""
    
    def test_empty_query_handling(self):
        """Test handling of empty or whitespace queries"""
        empty_queries = ["", "   ", "\t", "\n"]
        
        for query in empty_queries:
            result = handle_chat_query(query)
            assert "**General Query Handler**" in result
    
    def test_none_context_handling(self):
        """Test handling of None context parameter"""
        result = handle_chat_query("What is Virat Kohli's form?", None)
        assert "**Form Analysis Tool Used**" in result
    
    def test_empty_context_handling(self):
        """Test handling of empty context dictionary"""
        result = handle_chat_query("What is Virat Kohli's form?", {})
        assert "**Form Analysis Tool Used**" in result

class TestIntegrationScenarios:
    """Test suite for integration scenarios"""
    
    def test_multiple_keywords_in_query(self):
        """Test queries with multiple keywords - should route to first match"""
        # Matchup should take precedence over form and prediction
        result = handle_chat_query("Show me the matchup and predict the outcome")
        assert "**Knowledge Graph Tool Used**" in result
        
        # Form should take precedence over prediction
        result = handle_chat_query("What is Virat Kohli's form and can you predict his performance?")
        assert "**Form Analysis Tool Used**" in result
    
    def test_complex_query_scenarios(self):
        """Test complex, realistic query scenarios"""
        complex_queries = [
            ("I need form analysis for Virat Kohli before the big match", "form"),
            ("Can you analyze the historical matchup between these two players?", "matchup"),
            ("What's your prediction for the next over's outcome?", "prediction"),
            ("Tell me about the weather conditions", "fallback")
        ]
        
        for query, expected_routing in complex_queries:
            result = handle_chat_query(query)
            
            if expected_routing == "form":
                assert "**Form Analysis Tool Used**" in result
            elif expected_routing == "matchup":
                assert "**Knowledge Graph Tool Used**" in result
            elif expected_routing == "prediction":
                assert "**Prediction Tool Used**" in result
            elif expected_routing == "fallback":
                assert "**General Query Handler**" in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 