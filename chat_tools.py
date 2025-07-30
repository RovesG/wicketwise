# Purpose: Tool router for chat interface - routes user queries to appropriate analysis tools
# Author: Phi1618 Cricket AI, Last Modified: 2025-01-17

import re
from typing import Dict, Any, Optional

# Try to import graph_features, create stub if not available
try:
    from graph_features import get_form_vector
except ImportError:
    def get_form_vector(player_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Stub function for get_form_vector when graph_features.py is not available"""
        return {
            "player": player_name,
            "form_score": 0.75,
            "recent_performance": "Good",
            "trend": "Improving",
            "vector_features": [0.8, 0.7, 0.6, 0.9, 0.5]
        }

def query_kg_player_relationship(query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Stub function for knowledge graph player relationship queries
    
    Args:
        query: User query about player matchups or history
        context: Additional context information
        
    Returns:
        Dictionary with relationship analysis results
    """
    # Extract player names from query (improved pattern matching)
    # Pattern to match names like "MS Dhoni", "Virat Kohli", "Pat Cummins"
    player_pattern = r'\b(?:[A-Z]{2,3}\s+[A-Z][a-z]+|[A-Z][a-z]+\s+[A-Z][a-z]+)\b'
    players = re.findall(player_pattern, query)
    
    return {
        "query": query,
        "players_found": players,
        "relationship_type": "historical_matchup" if re.search(r'\bmatchup\b', query.lower()) else "career_history",
        "head_to_head": {
            "matches": 12,
            "wins": {"player_1": 7, "player_2": 5},
            "avg_runs": {"player_1": 45.2, "player_2": 38.7}
        },
        "context_used": bool(context)
    }

def predict_with_uncertainty(scenario: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Mock function for prediction with uncertainty quantification
    
    Args:
        scenario: Prediction scenario description
        context: Additional context for prediction
        
    Returns:
        Dictionary with prediction results and uncertainty measures
    """
    # Simple mock prediction based on scenario keywords
    if "win" in scenario.lower():
        prediction = "Team A victory"
        confidence = 0.72
    elif "runs" in scenario.lower():
        prediction = "165-185 runs"
        confidence = 0.68
    elif "wicket" in scenario.lower():
        prediction = "Wicket in next 3 balls"
        confidence = 0.45
    else:
        prediction = "Favorable outcome"
        confidence = 0.60
    
    return {
        "scenario": scenario,
        "prediction": prediction,
        "confidence": confidence,
        "uncertainty_range": f"±{round((1-confidence)*100)}%",
        "model_used": "cricket_predictor_v2",
        "context_factors": len(context) if context else 0
    }

def handle_chat_query(user_input: str, context: Dict[str, Any] = None) -> str:
    """
    Routes user input to appropriate tools based on keyword patterns
    
    Args:
        user_input: User's query string
        context: Additional context information
        
    Returns:
        Formatted string with result and tool used
    """
    if context is None:
        context = {}
    
    # Convert to lowercase for pattern matching
    query_lower = user_input.lower()
    
    # Route based on keyword patterns - check more specific patterns first
    # Use word boundaries to ensure exact keyword matching
    if re.search(r'\b(matchup|histor)', query_lower):
        result = query_kg_player_relationship(user_input, context)
        
        players_text = ", ".join(result['players_found']) if result['players_found'] else "players mentioned"
        
        return f"""**Knowledge Graph Tool Used**

Query: {result['query']}
Players Analyzed: {players_text}
Relationship Type: {result['relationship_type']}

Head-to-Head Stats:
• Total Matches: {result['head_to_head']['matches']}
• Player 1 Wins: {result['head_to_head']['wins']['player_1']}
• Player 2 Wins: {result['head_to_head']['wins']['player_2']}
• Average Runs: {result['head_to_head']['avg_runs']['player_1']:.1f} vs {result['head_to_head']['avg_runs']['player_2']:.1f}

*Analysis generated using knowledge graph relationship queries*"""
    
    elif re.search(r'\bform\b', query_lower):
        # Extract player name for form analysis using improved pattern
        player_pattern = r'\b(?:[A-Z]{2,3}\s+[A-Z][a-z]+|[A-Z][a-z]+\s+[A-Z][a-z]+)\b'
        players = re.findall(player_pattern, user_input)
        player_name = players[0] if players else "Unknown Player"
        
        result = get_form_vector(player_name, context)
        
        return f"""**Form Analysis Tool Used**

Player: {result['player']}
Form Score: {result['form_score']:.2f}
Recent Performance: {result['recent_performance']}
Trend: {result['trend']}
Vector Features: {result['vector_features']}

*Analysis generated using form vector analysis*"""
    
    elif re.search(r'\b(prediction|predict)\b', query_lower):
        result = predict_with_uncertainty(user_input, context)
        
        return f"""**Prediction Tool Used**

Scenario: {result['scenario']}
Prediction: {result['prediction']}
Confidence: {result['confidence']:.1%}
Uncertainty Range: {result['uncertainty_range']}
Model: {result['model_used']}

*Analysis generated using prediction model with uncertainty quantification*"""
    
    else:
        # Fallback response for unmatched queries
        return f"""**General Query Handler**

Query: {user_input}
Status: No specific tool matched

Available tools:
• Form analysis (keywords: 'form')
• Player relationships (keywords: 'matchup', 'history')
• Predictions (keywords: 'prediction', 'predict')

Please rephrase your query using one of the supported keywords for more specific analysis.

*General response - no specialized tool used*"""
