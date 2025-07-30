# Purpose: Tool registry for cricket analysis functions with OpenAI function-calling schemas
# Author: Phi1618 Cricket AI Team, Last Modified: 2024

from typing import Dict, Any, Callable
import pandas as pd

# Import existing functions from chat_tools.py
try:
    from chat_tools import get_form_vector, query_kg_player_relationship, predict_with_uncertainty
except ImportError:
    # Fallback if chat_tools.py is not available
    def get_form_vector(player_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Stub function for get_form_vector"""
        return {
            "player": player_name,
            "form_score": 0.75,
            "recent_performance": "Good",
            "trend": "Improving",
            "vector_features": [0.8, 0.7, 0.6, 0.9, 0.5]
        }
    
    def query_kg_player_relationship(query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Stub function for query_kg_player_relationship"""
        return {
            "query": query,
            "players_found": ["Player A", "Player B"],
            "relationship_type": "matchup",
            "head_to_head": {
                "matches": 10,
                "wins": {"player_1": 6, "player_2": 4},
                "avg_runs": {"player_1": 42.5, "player_2": 38.2}
            }
        }
    
    def predict_with_uncertainty(scenario: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Stub function for predict_with_uncertainty"""
        return {
            "scenario": scenario,
            "prediction": "Favorable outcome",
            "confidence": 0.68,
            "uncertainty_range": "±32%",
            "model_used": "cricket_predictor_v2"
        }

# Create wrapper functions for tool registry (to match expected names)
def predict_ball_outcome(scenario: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Predict ball outcome using uncertainty quantification."""
    return predict_with_uncertainty(scenario, context)

def query_kg_relationship(query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Query knowledge graph for player relationships."""
    return query_kg_player_relationship(query, context)

def get_odds_delta(market_odds: Dict[str, Any], model_odds: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate delta between market and model odds."""
    return {
        "home_delta": abs(market_odds["home_win_prob"] - model_odds["home_win_prob"]),
        "away_delta": abs(market_odds["away_win_prob"] - model_odds["away_win_prob"]),
        "home_value": market_odds["home_win_prob"] < model_odds["home_win_prob"],
        "away_value": market_odds["away_win_prob"] < model_odds["away_win_prob"]
    }

# Tool Registry - maps tool names to functions and schemas
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_form_vector": {
        "function": get_form_vector,
        "schema": {
            "name": "get_form_vector",
            "description": "Analyze player form and performance metrics",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "description": "Name of the cricket player"},
                    "context": {"type": "object", "description": "Additional context"}
                },
                "required": ["player_name"]
            }
        }
    },
    "predict_ball_outcome": {
        "function": predict_ball_outcome,
        "schema": {
            "name": "predict_ball_outcome",
            "description": "Predict cricket ball outcome with uncertainty",
            "parameters": {
                "type": "object",
                "properties": {
                    "scenario": {"type": "string", "description": "Current match scenario"},
                    "context": {"type": "object", "description": "Match context"}
                },
                "required": ["scenario"]
            }
        }
    },
    "query_kg_relationship": {
        "function": query_kg_relationship,
        "schema": {
            "name": "query_kg_relationship",
            "description": "Query knowledge graph for player relationships",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query about player relationships"},
                    "context": {"type": "object", "description": "Additional context"}
                },
                "required": ["query"]
            }
        }
    },
    "get_odds_delta": {
        "function": get_odds_delta,
        "schema": {
            "name": "get_odds_delta",
            "description": "Calculate delta between market and model odds",
            "parameters": {
                "type": "object",
                "properties": {
                    "market_odds": {"type": "object", "description": "Market odds data"},
                    "model_odds": {"type": "object", "description": "Model odds data"}
                },
                "required": ["market_odds", "model_odds"]
            }
        }
    }
}

def get_tool_function(tool_name: str) -> Callable:
    """Get a tool function by name."""
    if tool_name not in TOOL_REGISTRY:
        raise KeyError(f"Tool '{tool_name}' not found in registry. Available tools: {list(TOOL_REGISTRY.keys())}")
    return TOOL_REGISTRY[tool_name]["function"]

def get_tool_schema(tool_name: str) -> Dict[str, Any]:
    """Get a tool schema by name."""
    if tool_name not in TOOL_REGISTRY:
        raise KeyError(f"Tool '{tool_name}' not found in registry. Available tools: {list(TOOL_REGISTRY.keys())}")
    return TOOL_REGISTRY[tool_name]["schema"]

def validate_tool_registry() -> Dict[str, Any]:
    """Validate the tool registry structure."""
    validation_results = {
        "valid": True,
        "total_tools": len(TOOL_REGISTRY),
        "errors": [],
        "warnings": []
    }
    
    for tool_name, tool_data in TOOL_REGISTRY.items():
        if "function" not in tool_data:
            validation_results["errors"].append(f"Tool '{tool_name}' missing function")
            validation_results["valid"] = False
        elif not callable(tool_data["function"]):
            validation_results["errors"].append(f"Tool '{tool_name}' function is not callable")
            validation_results["valid"] = False
        
        if "schema" not in tool_data:
            validation_results["errors"].append(f"Tool '{tool_name}' missing schema")
            validation_results["valid"] = False
        else:
            schema = tool_data["schema"]
            required_fields = ["name", "description", "parameters"]
            for field in required_fields:
                if field not in schema:
                    validation_results["errors"].append(f"Tool '{tool_name}' schema missing '{field}'")
                    validation_results["valid"] = False
    
    return validation_results

def get_all_tool_schemas():
    """Get all tool schemas for OpenAI function calling."""
    return [tool_data["schema"] for tool_data in TOOL_REGISTRY.values()]

def list_available_tools():
    """List all available tool names."""
    return list(TOOL_REGISTRY.keys())
