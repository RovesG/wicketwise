# Purpose: OpenAI Function Tool Definitions for Knowledge Graph Chat
# Author: WicketWise Team, Last Modified: 2025-08-16

"""
OpenAI Function Tool Definitions

This module defines the function schemas that the OpenAI LLM can call
to query the cricket knowledge graph. Each function corresponds to a
method in the KGQueryEngine class.
"""

from typing import List, Dict, Any


def get_function_tools() -> List[Dict[str, Any]]:
    """
    Get the list of function tools available to the OpenAI LLM
    
    Returns:
        List of function tool definitions in OpenAI format
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_player_stats",
                "description": "Get comprehensive statistics for a cricket player including career stats, venues played, and teams",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player": {
                            "type": "string",
                            "description": "The player's name (e.g., 'Virat Kohli', 'MS Dhoni')"
                        },
                        "format_filter": {
                            "type": "string",
                            "description": "Optional format filter",
                            "enum": ["T20", "ODI", "Test", "T20I"]
                        },
                        "venue_filter": {
                            "type": "string",
                            "description": "Optional venue filter (e.g., 'MCG', 'Wankhede')"
                        }
                    },
                    "required": ["player"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compare_players",
                "description": "Compare multiple cricket players across specified metrics like strike rate, average, runs, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "players": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of player names to compare (max 10)",
                            "maxItems": 10
                        },
                        "metric": {
                            "type": "string",
                            "description": "Metric to compare players by",
                            "enum": ["strike_rate", "average", "total_runs", "matches", "boundaries", "sixes"],
                            "default": "strike_rate"
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional context filter (venue, format, etc.)"
                        }
                    },
                    "required": ["players"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_venue_history",
                "description": "Get venue history, characteristics, and performance statistics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "venue": {
                            "type": "string",
                            "description": "The venue name (e.g., 'MCG', 'Lord's', 'Eden Gardens')"
                        },
                        "team": {
                            "type": "string",
                            "description": "Optional team filter to see specific team's performance at venue"
                        },
                        "format_filter": {
                            "type": "string",
                            "description": "Optional format filter",
                            "enum": ["T20", "ODI", "Test", "T20I"]
                        },
                        "last_n_matches": {
                            "type": "integer",
                            "description": "Number of recent matches to include",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50
                        }
                    },
                    "required": ["venue"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_head_to_head",
                "description": "Get head-to-head record and statistics between two cricket teams",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "team1": {
                            "type": "string",
                            "description": "First team name (e.g., 'India', 'Australia', 'England')"
                        },
                        "team2": {
                            "type": "string",
                            "description": "Second team name (e.g., 'India', 'Australia', 'England')"
                        },
                        "format_filter": {
                            "type": "string",
                            "description": "Optional format filter",
                            "enum": ["T20", "ODI", "Test", "T20I"]
                        }
                    },
                    "required": ["team1", "team2"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_similar_players",
                "description": "Find players similar to a given player based on specified performance metrics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player": {
                            "type": "string",
                            "description": "Reference player name to find similar players for"
                        },
                        "metric": {
                            "type": "string",
                            "description": "Metric to use for similarity comparison",
                            "enum": ["strike_rate", "average", "total_runs", "boundaries", "sixes"],
                            "default": "strike_rate"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of similar players to return",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10
                        }
                    },
                    "required": ["player"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_graph_summary",
                "description": "Get a summary of the knowledge graph including total nodes, edges, and available entities",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]


def get_function_descriptions() -> Dict[str, str]:
    """
    Get human-readable descriptions of available functions
    
    Returns:
        Dictionary mapping function names to descriptions
    """
    return {
        "get_player_stats": "Get detailed statistics for any cricket player including career performance, venues played, and team history",
        "compare_players": "Compare multiple players side-by-side across various performance metrics",
        "get_venue_history": "Analyze venue characteristics, match history, and team performance at specific grounds",
        "get_head_to_head": "Get head-to-head records and win-loss statistics between any two cricket teams",
        "find_similar_players": "Find players with similar performance profiles based on statistical metrics",
        "get_graph_summary": "Get an overview of the cricket knowledge graph database"
    }
