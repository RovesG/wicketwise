# Purpose: GNN-Enhanced Function Tools for OpenAI LLM Integration
# Author: WicketWise Team, Last Modified: 2025-08-25

"""
GNN-Enhanced Function Tools

This module extends the existing function tools with GNN-powered capabilities,
enabling the LLM to leverage graph neural network embeddings for advanced
cricket analytics and player comparisons.
"""

from typing import List, Dict, Any


def get_gnn_function_tools() -> List[Dict[str, Any]]:
    """
    Get GNN-enhanced function tools for OpenAI LLM
    
    Returns:
        List of GNN function tool definitions in OpenAI format
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "find_similar_players_gnn",
                "description": "Find players with similar playing styles using advanced GNN embeddings that capture complex behavioral patterns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player": {
                            "type": "string",
                            "description": "The target player name to find similarities for (e.g., 'Virat Kohli', 'Jos Buttler')"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of similar players to return",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 15
                        },
                        "similarity_metric": {
                            "type": "string",
                            "description": "Similarity calculation method",
                            "enum": ["cosine", "euclidean"],
                            "default": "cosine"
                        },
                        "min_similarity": {
                            "type": "number",
                            "description": "Minimum similarity threshold (0.0 to 1.0)",
                            "default": 0.6,
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "required": ["player"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "predict_contextual_performance",
                "description": "Predict how a player will perform in specific match contexts using GNN contextual analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player": {
                            "type": "string",
                            "description": "Player name for performance prediction"
                        },
                        "context": {
                            "type": "object",
                            "description": "Match context for prediction",
                            "properties": {
                                "phase": {
                                    "type": "string",
                                    "enum": ["powerplay", "middle", "death"],
                                    "description": "Match phase"
                                },
                                "bowling_type": {
                                    "type": "string",
                                    "enum": ["pace", "spin"],
                                    "description": "Type of bowling being faced"
                                },
                                "pressure": {
                                    "type": "boolean",
                                    "description": "Whether it's a high-pressure situation"
                                },
                                "required_run_rate": {
                                    "type": "number",
                                    "description": "Required run rate for the batting team"
                                },
                                "wickets_lost": {
                                    "type": "integer",
                                    "description": "Number of wickets already lost"
                                },
                                "balls_remaining": {
                                    "type": "integer",
                                    "description": "Balls remaining in the innings"
                                }
                            }
                        }
                    },
                    "required": ["player", "context"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_venue_compatibility",
                "description": "Analyze how well a player's style matches a specific venue using GNN venue-player embeddings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player": {
                            "type": "string",
                            "description": "Player name for venue compatibility analysis"
                        },
                        "venue": {
                            "type": "string",
                            "description": "Venue name (e.g., 'MCG', 'Wankhede Stadium', 'Lord's')"
                        }
                    },
                    "required": ["player", "venue"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_playing_style_similarity",
                "description": "Compare playing styles between two players using detailed GNN style embeddings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player1": {
                            "type": "string",
                            "description": "First player name for style comparison"
                        },
                        "player2": {
                            "type": "string",
                            "description": "Second player name for style comparison"
                        }
                    },
                    "required": ["player1", "player2"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_best_performers_contextual",
                "description": "Find the best performing players in specific contexts using GNN contextual embeddings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "Performance context (e.g., 'death overs against pace', 'powerplay against spin', 'pressure situations')"
                        },
                        "role": {
                            "type": "string",
                            "description": "Player role filter",
                            "enum": ["batsman", "bowler", "all-rounder", "wicket-keeper", "any"],
                            "default": "any"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top performers to return",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 15
                        },
                        "min_matches": {
                            "type": "integer",
                            "description": "Minimum matches played in the context",
                            "default": 10,
                            "minimum": 1
                        }
                    },
                    "required": ["context"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_team_composition_gnn",
                "description": "Analyze team composition and suggest improvements using GNN player compatibility analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "players": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of player names in the team",
                            "minItems": 5,
                            "maxItems": 15
                        },
                        "venue": {
                            "type": "string",
                            "description": "Venue where the team will play"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["T20", "ODI", "Test"],
                            "description": "Cricket format",
                            "default": "T20"
                        }
                    },
                    "required": ["players"]
                }
            }
        }
    ]


def get_enhanced_function_descriptions() -> Dict[str, str]:
    """
    Get enhanced descriptions for all GNN functions
    
    Returns:
        Dictionary mapping function names to detailed descriptions
    """
    return {
        "find_similar_players_gnn": (
            "Uses advanced Graph Neural Network embeddings to find players with similar "
            "playing styles, techniques, and behavioral patterns. More accurate than "
            "traditional statistical comparisons as it captures complex relationships "
            "and contextual performance patterns."
        ),
        "predict_contextual_performance": (
            "Leverages GNN contextual embeddings to predict player performance in "
            "specific match situations. Considers match phase, bowling type, pressure "
            "situations, and game state to provide accurate performance forecasts."
        ),
        "analyze_venue_compatibility": (
            "Uses GNN venue-player embeddings to analyze how well a player's style "
            "and strengths match the characteristics of a specific venue. Considers "
            "pitch conditions, ground dimensions, and historical performance patterns."
        ),
        "get_playing_style_similarity": (
            "Provides detailed playing style comparison between two players using "
            "multi-dimensional GNN embeddings. Analyzes batting technique, bowling "
            "style, fielding patterns, and tactical approaches."
        ),
        "find_best_performers_contextual": (
            "Identifies top performers in specific match contexts using GNN contextual "
            "analysis. Can find specialists for particular situations like death overs, "
            "powerplay, or pressure scenarios."
        ),
        "analyze_team_composition_gnn": (
            "Analyzes team balance and player compatibility using GNN embeddings to "
            "identify synergies, gaps, and optimization opportunities in team selection."
        )
    }


def get_all_enhanced_function_tools() -> List[Dict[str, Any]]:
    """
    Get all function tools (original + GNN-enhanced)
    
    Returns:
        Combined list of all available function tools
    """
    # Import original function tools
    from .function_tools import get_function_tools
    
    # Combine original and GNN tools
    original_tools = get_function_tools()
    gnn_tools = get_gnn_function_tools()
    
    return original_tools + gnn_tools


def get_all_enhanced_function_descriptions() -> Dict[str, str]:
    """
    Get all function descriptions (original + GNN-enhanced)
    
    Returns:
        Combined dictionary of all function descriptions
    """
    # Import original descriptions
    from .function_tools import get_function_descriptions
    
    # Combine original and GNN descriptions
    original_descriptions = get_function_descriptions()
    gnn_descriptions = get_enhanced_function_descriptions()
    
    return {**original_descriptions, **gnn_descriptions}


# Cricket-specific GNN analysis templates
GNN_ANALYSIS_TEMPLATES = {
    "similarity_high": (
        "🎯 **High Similarity Detected**: {player1} and {player2} show {similarity:.1%} "
        "style similarity based on GNN analysis. Key similarities include {aspects}."
    ),
    "similarity_low": (
        "🔄 **Contrasting Styles**: {player1} and {player2} have different approaches "
        "({similarity:.1%} similarity). {player1} excels in {strengths1} while "
        "{player2} is stronger in {strengths2}."
    ),
    "venue_excellent": (
        "🏟️ **Excellent Venue Match**: {player} shows {compatibility:.1%} compatibility "
        "with {venue}. GNN analysis suggests optimal performance due to {factors}."
    ),
    "venue_poor": (
        "⚠️ **Venue Challenge**: {player} may face difficulties at {venue} "
        "({compatibility:.1%} compatibility). Consider tactical adjustments for {challenges}."
    ),
    "contextual_strong": (
        "💪 **Strong Contextual Fit**: {player} is predicted to excel in {context} "
        "with {confidence:.1%} confidence. Expected performance: {metrics}."
    ),
    "contextual_weak": (
        "🤔 **Contextual Uncertainty**: Limited GNN confidence for {player} in {context}. "
        "Recommendation: {recommendation}."
    )
}


def format_gnn_insight(template_key: str, **kwargs) -> str:
    """
    Format GNN analysis insight using predefined templates
    
    Args:
        template_key: Key for the insight template
        **kwargs: Template variables
        
    Returns:
        Formatted insight string
    """
    template = GNN_ANALYSIS_TEMPLATES.get(template_key, "GNN Analysis: {analysis}")
    try:
        return template.format(**kwargs)
    except KeyError as e:
        return f"GNN Analysis: Missing template variable {e}"
