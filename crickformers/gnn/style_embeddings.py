# Purpose: Video-based player style embeddings for cricket knowledge graph
# Author: WicketWise Team, Last Modified: 2024-07-19

"""
This module provides video-based style embeddings for cricket players.
It loads pre-computed style embeddings from JSON files and integrates them
into player node features in the knowledge graph.

Style embeddings capture visual playing patterns like:
- Batting stance and technique
- Shot selection preferences  
- Bowling action and variations
- Fielding positioning and movement
"""

import json
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default style embedding dimension
STYLE_EMBEDDING_DIM = 16

# Default style embedding (all zeros) for players without video analysis
DEFAULT_STYLE_EMBEDDING = np.zeros(STYLE_EMBEDDING_DIM, dtype=np.float32)


def load_style_embeddings_from_json(json_path: Union[str, Path]) -> Dict[str, List[float]]:
    """
    Load player style embeddings from a JSON file.
    
    Expected JSON format:
    {
        "player_id_1": [0.1, 0.2, ..., 0.16],  # 16D embedding
        "player_id_2": [0.3, 0.4, ..., 0.18],
        ...
    }
    
    Args:
        json_path: Path to JSON file containing style embeddings
        
    Returns:
        Dictionary mapping player_id -> style embedding vector
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        logger.warning(f"Style embeddings file not found: {json_path}")
        return {}
    
    try:
        with open(json_path, 'r') as f:
            embeddings_data = json.load(f)
        
        # Validate format
        if not isinstance(embeddings_data, dict):
            logger.error(f"Invalid JSON format: expected dict, got {type(embeddings_data)}")
            return {}
        
        # Convert to proper format and validate
        style_embeddings = {}
        for player_id, embedding in embeddings_data.items():
            if not isinstance(embedding, list):
                logger.warning(f"Invalid embedding for {player_id}: expected list, got {type(embedding)}")
                continue
            
            # Convert to float and validate
            try:
                embedding_array = [float(x) for x in embedding]
                style_embeddings[str(player_id)] = embedding_array
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid embedding values for {player_id}: {e}")
                continue
        
        logger.info(f"Loaded style embeddings for {len(style_embeddings)} players from {json_path}")
        return style_embeddings
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file {json_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading style embeddings from {json_path}: {e}")
        return {}


def normalize_style_embedding(embedding: List[float], target_dim: int = STYLE_EMBEDDING_DIM) -> np.ndarray:
    """
    Normalize a style embedding to the target dimension.
    
    Args:
        embedding: Raw style embedding vector
        target_dim: Target dimension (default: 16)
        
    Returns:
        Normalized embedding vector of target dimension
    """
    if not embedding:
        return DEFAULT_STYLE_EMBEDDING.copy()
    
    embedding_array = np.array(embedding, dtype=np.float32)
    current_dim = len(embedding_array)
    
    if current_dim == target_dim:
        # Already correct dimension
        return embedding_array
    elif current_dim > target_dim:
        # Truncate to target dimension
        logger.debug(f"Truncating embedding from {current_dim}D to {target_dim}D")
        return embedding_array[:target_dim]
    else:
        # Pad with zeros to target dimension
        logger.debug(f"Padding embedding from {current_dim}D to {target_dim}D")
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[:current_dim] = embedding_array
        return padded


def get_style_embedding_for_player(player_id: str, 
                                  style_embeddings: Dict[str, List[float]],
                                  target_dim: int = STYLE_EMBEDDING_DIM) -> np.ndarray:
    """
    Get the normalized style embedding for a specific player.
    
    Args:
        player_id: Player identifier
        style_embeddings: Dictionary of player style embeddings
        target_dim: Target embedding dimension
        
    Returns:
        Normalized style embedding vector (default zeros if player not found)
    """
    if player_id in style_embeddings:
        raw_embedding = style_embeddings[player_id]
        return normalize_style_embedding(raw_embedding, target_dim)
    else:
        logger.debug(f"No style embedding found for player {player_id}, using default")
        return DEFAULT_STYLE_EMBEDDING.copy()


def add_style_embeddings_to_graph(G: nx.DiGraph, 
                                 style_embeddings: Dict[str, List[float]],
                                 target_dim: int = STYLE_EMBEDDING_DIM) -> nx.DiGraph:
    """
    Add style embeddings to player nodes in the knowledge graph.
    
    This function extends the feature vectors of player nodes (batters, bowlers)
    by appending normalized style embedding vectors.
    
    Args:
        G: NetworkX DiGraph representing the cricket knowledge graph
        style_embeddings: Dictionary mapping player_id -> style embedding
        target_dim: Target dimension for style embeddings
        
    Returns:
        Updated graph with style embeddings added to player nodes
    """
    players_updated = 0
    players_with_style = 0
    
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "")
        
        # Only add style embeddings to player nodes
        if node_type in ["batter", "bowler"]:
            # Get style embedding for this player
            style_embedding = get_style_embedding_for_player(node, style_embeddings, target_dim)
            
            # Track statistics
            players_updated += 1
            if node in style_embeddings:
                players_with_style += 1
            
            # Add style embedding to node attributes
            attrs["style_embedding"] = style_embedding
            
            # If node already has features, extend them
            if "features" in attrs:
                existing_features = attrs["features"]
                if isinstance(existing_features, np.ndarray):
                    # Concatenate existing features with style embedding
                    attrs["features"] = np.concatenate([existing_features, style_embedding])
                else:
                    # Convert to numpy array and concatenate
                    existing_array = np.array(existing_features, dtype=np.float32)
                    attrs["features"] = np.concatenate([existing_array, style_embedding])
            else:
                # No existing features, create features with style embedding
                attrs["features"] = style_embedding.copy()
    
    logger.info(f"Added style embeddings to {players_updated} players "
                f"({players_with_style} with custom embeddings, "
                f"{players_updated - players_with_style} with default)")
    
    return G


def create_sample_style_embeddings(player_ids: List[str], 
                                  embedding_dim: int = STYLE_EMBEDDING_DIM,
                                  save_path: Optional[Union[str, Path]] = None) -> Dict[str, List[float]]:
    """
    Create sample style embeddings for testing purposes.
    
    Args:
        player_ids: List of player IDs to create embeddings for
        embedding_dim: Dimension of embeddings to create
        save_path: Optional path to save the sample embeddings as JSON
        
    Returns:
        Dictionary of sample style embeddings
    """
    np.random.seed(42)  # For reproducible sample embeddings
    
    sample_embeddings = {}
    
    for player_id in player_ids:
        # Create a unique but deterministic embedding for each player
        # Use player_id hash to seed random generation
        player_seed = hash(player_id) % (2**32)
        np.random.seed(player_seed)
        
        # Generate random embedding with some structure
        embedding = np.random.normal(0, 0.3, embedding_dim).astype(np.float32)
        
        # Add some player-specific patterns
        if "batter" in player_id.lower() or any(name in player_id.lower() for name in ["kohli", "sharma", "root"]):
            # Batting-focused embeddings - emphasize first half
            embedding[:embedding_dim//2] *= 1.5
        elif "bowler" in player_id.lower() or any(name in player_id.lower() for name in ["starc", "bumrah", "archer"]):
            # Bowling-focused embeddings - emphasize second half
            embedding[embedding_dim//2:] *= 1.5
        
        # Normalize to reasonable range
        embedding = np.clip(embedding, -1.0, 1.0)
        
        sample_embeddings[player_id] = embedding.tolist()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(sample_embeddings, f, indent=2)
        
        logger.info(f"Saved sample style embeddings to {save_path}")
    
    return sample_embeddings


def get_style_embedding_stats(style_embeddings: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Get statistics about style embeddings.
    
    Args:
        style_embeddings: Dictionary of style embeddings
        
    Returns:
        Dictionary containing embedding statistics
    """
    if not style_embeddings:
        return {
            "num_players": 0,
            "embedding_dims": [],
            "avg_embedding_dim": 0,
            "min_embedding_dim": 0,
            "max_embedding_dim": 0
        }
    
    embedding_dims = [len(emb) for emb in style_embeddings.values()]
    
    stats = {
        "num_players": len(style_embeddings),
        "embedding_dims": embedding_dims,
        "avg_embedding_dim": np.mean(embedding_dims),
        "min_embedding_dim": min(embedding_dims),
        "max_embedding_dim": max(embedding_dims),
        "players_with_embeddings": list(style_embeddings.keys())
    }
    
    # Analyze embedding value ranges
    all_values = []
    for embedding in style_embeddings.values():
        all_values.extend(embedding)
    
    if all_values:
        stats.update({
            "value_min": min(all_values),
            "value_max": max(all_values),
            "value_mean": np.mean(all_values),
            "value_std": np.std(all_values)
        })
    
    return stats


def validate_style_embeddings(style_embeddings: Dict[str, List[float]], 
                             expected_dim: int = STYLE_EMBEDDING_DIM) -> List[str]:
    """
    Validate style embeddings for consistency and correctness.
    
    Args:
        style_embeddings: Dictionary of style embeddings to validate
        expected_dim: Expected embedding dimension
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not isinstance(style_embeddings, dict):
        errors.append("Style embeddings must be a dictionary")
        return errors
    
    for player_id, embedding in style_embeddings.items():
        if not isinstance(player_id, str):
            errors.append(f"Player ID must be string, got {type(player_id)}")
        
        if not isinstance(embedding, list):
            errors.append(f"Embedding for {player_id} must be list, got {type(embedding)}")
            continue
        
        if len(embedding) == 0:
            errors.append(f"Empty embedding for player {player_id}")
            continue
        
        # Check if all values are numeric
        try:
            float_values = [float(x) for x in embedding]
        except (ValueError, TypeError):
            errors.append(f"Non-numeric values in embedding for {player_id}")
            continue
        
        # Check for NaN or infinite values
        if any(np.isnan(float_values)) or any(np.isinf(float_values)):
            errors.append(f"NaN or infinite values in embedding for {player_id}")
        
        # Warn about dimension mismatch (not an error, will be normalized)
        if len(embedding) != expected_dim:
            logger.warning(f"Dimension mismatch for {player_id}: got {len(embedding)}, expected {expected_dim}")
    
    return errors