# Purpose: Role embeddings for cricket players in knowledge graph
# Author: WicketWise Team, Last Modified: 2024-07-19

"""
This module provides role-based embeddings for cricket players.
It defines player roles and converts them to embedding vectors that can be
appended to player node features in the knowledge graph.
"""

import numpy as np
from typing import Dict, List, Union, Optional
import networkx as nx


# Static dictionary mapping player_id to role tag(s)
PLAYER_ROLES = {
    # Batters
    "kohli": ["opener", "anchor"],
    "sharma": ["opener", "powerplay_specialist"],
    "dhoni": ["finisher", "wicketkeeper"],
    "pant": ["finisher", "wicketkeeper"],
    "williamson": ["anchor", "middle_order"],
    "root": ["anchor", "middle_order"],
    "babar": ["opener", "anchor"],
    "smith": ["anchor", "middle_order"],
    "warner": ["opener", "powerplay_specialist"],
    "butler": ["finisher", "wicketkeeper"],
    "stokes": ["finisher", "all_rounder"],
    "pandya": ["finisher", "all_rounder"],
    
    # Bowlers
    "starc": ["death_bowler", "pace"],
    "bumrah": ["death_bowler", "pace"],
    "boult": ["powerplay_bowler", "pace"],
    "archer": ["death_bowler", "pace"],
    "cummins": ["powerplay_bowler", "pace"],
    "rabada": ["death_bowler", "pace"], 
    "nortje": ["powerplay_bowler", "pace"],
    "rashid": ["middle_overs_bowler", "spin"],
    "chahal": ["middle_overs_bowler", "spin"],
    "zampa": ["middle_overs_bowler", "spin"],
    "ashwin": ["powerplay_bowler", "spin"],
    "jadeja": ["middle_overs_bowler", "spin", "all_rounder"],
    
    # All-rounders
    "russell": ["finisher", "death_bowler", "all_rounder"],
    "pollard": ["finisher", "death_bowler", "all_rounder"],
    "maxwell": ["finisher", "middle_overs_bowler", "all_rounder"],
    "shakib": ["anchor", "middle_overs_bowler", "all_rounder"],
}

# Define role categories and their one-hot positions
ROLE_CATEGORIES = {
    # Batting roles (position 0)
    "opener": 0,
    "anchor": 0, 
    "finisher": 0,
    "middle_order": 0,
    "powerplay_specialist": 0,
    "wicketkeeper": 0,
    
    # Bowling roles (position 1)
    "powerplay_bowler": 1,
    "middle_overs_bowler": 1,
    "death_bowler": 1,
    "pace": 1,
    "spin": 1,
    
    # Specialist roles (position 2)
    "all_rounder": 2,
    
    # Unknown role (position 3)
    "unknown": 3,
}

# Detailed role mappings for one-hot encoding
BATTING_ROLES = ["opener", "anchor", "finisher", "middle_order", "powerplay_specialist", "wicketkeeper"]
BOWLING_ROLES = ["powerplay_bowler", "middle_overs_bowler", "death_bowler", "pace", "spin"]
SPECIALIST_ROLES = ["all_rounder"]
UNKNOWN_ROLES = ["unknown"]

ROLE_EMBEDDING_DIM = 4  # [batting, bowling, specialist, unknown]


def get_player_roles(player_id: str) -> List[str]:
    """
    Get the role tags for a specific player.
    
    Args:
        player_id: The player identifier
        
    Returns:
        List of role tags for the player, or ["unknown"] if not found
    """
    return PLAYER_ROLES.get(player_id, ["unknown"])


def create_role_embedding(roles: List[str]) -> np.ndarray:
    """
    Convert role tags to a one-hot embedding vector.
    
    Args:
        roles: List of role tags for a player
        
    Returns:
        One-hot embedding vector of dimension ROLE_EMBEDDING_DIM
    """
    embedding = np.zeros(ROLE_EMBEDDING_DIM, dtype=np.float32)
    
    # Check each role and set corresponding dimension
    for role in roles:
        if role in BATTING_ROLES:
            embedding[0] = 1.0  # Batting role
        elif role in BOWLING_ROLES:
            embedding[1] = 1.0  # Bowling role
        elif role in SPECIALIST_ROLES:
            embedding[2] = 1.0  # Specialist role
        elif role in UNKNOWN_ROLES:
            embedding[3] = 1.0  # Unknown role
    
    # If no known roles found, set unknown
    if np.sum(embedding) == 0:
        embedding[3] = 1.0
    
    return embedding


def get_role_embedding_for_player(player_id: str) -> np.ndarray:
    """
    Get the role embedding vector for a specific player.
    
    Args:
        player_id: The player identifier
        
    Returns:
        Role embedding vector of dimension ROLE_EMBEDDING_DIM
    """
    roles = get_player_roles(player_id)
    return create_role_embedding(roles)


def add_role_embeddings_to_graph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Add role embeddings to player nodes in the knowledge graph.
    
    This function extends the feature vectors of player nodes (batters, bowlers)
    by appending role embedding vectors.
    
    Args:
        G: NetworkX DiGraph representing the cricket knowledge graph
        
    Returns:
        Updated graph with role embeddings added to player nodes
    """
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "")
        
        # Only add role embeddings to player nodes
        if node_type in ["batter", "bowler"]:
            # Get role embedding for this player
            role_embedding = get_role_embedding_for_player(node)
            
            # Add role embedding to node attributes
            attrs["role_embedding"] = role_embedding
            attrs["role_tags"] = get_player_roles(node)
            
            # If node already has features, extend them
            if "features" in attrs:
                existing_features = attrs["features"]
                if isinstance(existing_features, np.ndarray):
                    # Concatenate existing features with role embedding
                    attrs["features"] = np.concatenate([existing_features, role_embedding])
                else:
                    # Convert to numpy array and concatenate
                    existing_array = np.array(existing_features, dtype=np.float32)
                    attrs["features"] = np.concatenate([existing_array, role_embedding])
            else:
                # No existing features, just use role embedding
                attrs["features"] = role_embedding
    
    return G


def get_role_embedding_stats() -> Dict[str, Union[int, List[str]]]:
    """
    Get statistics about the role embedding system.
    
    Returns:
        Dictionary containing embedding statistics
    """
    return {
        "embedding_dim": ROLE_EMBEDDING_DIM,
        "total_players": len(PLAYER_ROLES),
        "batting_roles": BATTING_ROLES,
        "bowling_roles": BOWLING_ROLES,
        "specialist_roles": SPECIALIST_ROLES,
        "role_categories": list(ROLE_CATEGORIES.keys()),
        "players_with_roles": list(PLAYER_ROLES.keys())
    }


def validate_role_embedding(embedding: np.ndarray) -> bool:
    """
    Validate that a role embedding vector is correctly formatted.
    
    Args:
        embedding: Role embedding vector to validate
        
    Returns:
        True if embedding is valid, False otherwise
    """
    if not isinstance(embedding, np.ndarray):
        return False
    
    if embedding.shape != (ROLE_EMBEDDING_DIM,):
        return False
    
    if embedding.dtype != np.float32:
        return False
    
    # Should have at least one non-zero value
    if np.sum(embedding) == 0:
        return False
    
    # All values should be 0 or 1
    if not np.all((embedding == 0) | (embedding == 1)):
        return False
    
    return True


def get_role_distribution() -> Dict[str, int]:
    """
    Get the distribution of roles across all players.
    
    Returns:
        Dictionary mapping role tags to their frequency
    """
    role_counts = {}
    
    for player_id, roles in PLAYER_ROLES.items():
        for role in roles:
            role_counts[role] = role_counts.get(role, 0) + 1
    
    return role_counts