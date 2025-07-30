# Purpose: HeteroData graph builder for cricket knowledge graph
# Author: WicketWise Team, Last Modified: 2024-07-19

"""
This module converts NetworkX cricket knowledge graphs to PyTorch Geometric HeteroData format.
It defines node types, edge types, and handles feature conversion for efficient GNN training.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Union
from torch_geometric.data import HeteroData
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# Define node types for HeteroData
NODE_TYPES = {
    "player",      # Batters and bowlers
    "team", 
    "venue",
    "match",
    "phase",       # Match phases (powerplay, middle, death)
    "event",       # Cricket events (four, six, dot, wicket)
    "bowler_type"  # Bowling styles
}

# Define edge types as (source_node_type, relation_type, target_node_type)
EDGE_TYPES = [
    # Player relationships
    ("player", "faced", "player"),           # batter faced bowler
    ("player", "dismissed_by", "player"),    # batter dismissed by bowler
    ("player", "partnered_with", "player"),  # batting partnerships
    ("player", "teammate_of", "player"),     # team relationships
    
    # Player-team relationships
    ("player", "plays_for", "team"),         # player plays for team
    
    # Player-venue relationships  
    ("player", "played_at", "venue"),        # player played at venue
    
    # Player-phase relationships
    ("player", "bowled_at", "phase"),        # bowler bowled in phase
    
    # Player-event relationships
    ("player", "produced", "event"),         # batter produced event
    ("player", "conceded", "event"),         # bowler conceded event
    
    # Player-style relationships
    ("player", "excels_against", "bowler_type"), # batter excels against style
    
    # Team-venue relationships
    ("team", "played_at", "venue"),          # team played at venue
    
    # Match relationships (if we add match nodes in future)
    # ("match", "played_at", "venue"),
    # ("team", "played_in", "match"),
]


def networkx_to_hetero_data(G: nx.DiGraph) -> HeteroData:
    """
    Convert a NetworkX cricket knowledge graph to PyTorch Geometric HeteroData.
    
    Args:
        G: NetworkX DiGraph with cricket knowledge graph
        
    Returns:
        HeteroData object ready for GNN training
    """
    data = HeteroData()
    
    # Step 1: Group nodes by type and create node mappings
    node_type_mapping = _create_node_type_mapping(G)
    
    # Step 2: Extract and convert node features
    _add_node_features_to_hetero_data(data, G, node_type_mapping)
    
    # Step 3: Convert edges to HeteroData format
    _add_edges_to_hetero_data(data, G, node_type_mapping)
    
    # Step 4: Add metadata
    _add_metadata_to_hetero_data(data, G, node_type_mapping)
    
    return data


def _create_node_type_mapping(G: nx.DiGraph) -> Dict[str, Dict[str, int]]:
    """
    Create mapping from node IDs to indices within each node type.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary mapping node_type -> {node_id: index}
    """
    node_type_mapping = defaultdict(dict)
    node_type_counts = defaultdict(int)
    
    for node, attrs in G.nodes(data=True):
        # Determine node type
        nx_node_type = attrs.get("type", "unknown")
        
        # Map NetworkX node types to HeteroData node types
        if nx_node_type in ["batter", "bowler"]:
            hetero_node_type = "player"
        elif nx_node_type == "bowler_type":
            hetero_node_type = "bowler_type"
        elif nx_node_type in ["team"]:
            hetero_node_type = "team"
        elif nx_node_type in ["venue"]:
            hetero_node_type = "venue"
        elif nx_node_type in ["phase"]:
            hetero_node_type = "phase"
        elif nx_node_type in ["event"]:
            hetero_node_type = "event"
        else:
            hetero_node_type = "player"  # Default fallback
            logger.warning(f"Unknown node type '{nx_node_type}' for node '{node}', defaulting to 'player'")
        
        # Assign index within node type
        node_type_mapping[hetero_node_type][node] = node_type_counts[hetero_node_type]
        node_type_counts[hetero_node_type] += 1
    
    return dict(node_type_mapping)


def _add_node_features_to_hetero_data(data: HeteroData, G: nx.DiGraph, 
                                     node_type_mapping: Dict[str, Dict[str, int]]) -> None:
    """
    Extract node features from NetworkX graph and add to HeteroData.
    
    Args:
        data: HeteroData object to populate
        G: NetworkX graph
        node_type_mapping: Mapping of node types to indices
    """
    for node_type, node_mapping in node_type_mapping.items():
        num_nodes = len(node_mapping)
        
        # Collect features for all nodes of this type
        feature_list = []
        node_ids = []
        
        for node_id, idx in node_mapping.items():
            node_attrs = G.nodes[node_id]
            
            # Extract features
            features = _extract_node_features(node_attrs, node_type)
            feature_list.append(features)
            node_ids.append(node_id)
        
        # Convert to tensor
        if feature_list:
            if all(isinstance(f, np.ndarray) for f in feature_list):
                # All features are numpy arrays
                feature_tensor = torch.tensor(np.stack(feature_list), dtype=torch.float32)
            elif all(isinstance(f, (int, float)) for f in feature_list):
                # All features are scalars
                feature_tensor = torch.tensor(feature_list, dtype=torch.float32).unsqueeze(1)
            else:
                # Mixed types - convert to consistent format
                processed_features = []
                for f in feature_list:
                    if isinstance(f, np.ndarray):
                        processed_features.append(f)
                    elif isinstance(f, (int, float)):
                        processed_features.append(np.array([f], dtype=np.float32))
                    else:
                        # Fallback - create zero vector
                        processed_features.append(np.zeros(1, dtype=np.float32))
                
                feature_tensor = torch.tensor(np.stack(processed_features), dtype=torch.float32)
        else:
            # No nodes of this type
            feature_tensor = torch.empty((0, 1), dtype=torch.float32)
        
        # Add to HeteroData
        data[node_type].x = feature_tensor
        data[node_type].node_ids = node_ids  # Keep track of original node IDs
        data[node_type].num_nodes = num_nodes


def _extract_node_features(node_attrs: Dict[str, Any], node_type: str) -> np.ndarray:
    """
    Extract features from a node's attributes.
    
    Args:
        node_attrs: Node attributes from NetworkX
        node_type: Type of the node
        
    Returns:
        Feature vector as numpy array
    """
    # Check if node already has processed features
    if "features" in node_attrs:
        features = node_attrs["features"]
        if isinstance(features, np.ndarray):
            return features.astype(np.float32)
        elif isinstance(features, (list, tuple)):
            return np.array(features, dtype=np.float32)
    
    # Check for role embeddings (for players)
    if node_type == "player" and "role_embedding" in node_attrs:
        role_embedding = node_attrs["role_embedding"]
        if isinstance(role_embedding, np.ndarray):
            return role_embedding.astype(np.float32)
    
    # Check for style embeddings (for players)
    if node_type == "player" and "style_embedding" in node_attrs:
        style_embedding = node_attrs["style_embedding"]
        if isinstance(style_embedding, np.ndarray):
            return style_embedding.astype(np.float32)
    
    # Default: create simple feature based on node type
    if node_type == "player":
        # For players, create a simple feature vector
        # Could be enhanced with player statistics
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    elif node_type == "team":
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    elif node_type == "venue":
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    elif node_type == "event":
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        # Default feature vector
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _add_edges_to_hetero_data(data: HeteroData, G: nx.DiGraph, 
                             node_type_mapping: Dict[str, Dict[str, int]]) -> None:
    """
    Convert NetworkX edges to HeteroData edge format.
    
    Args:
        data: HeteroData object to populate
        G: NetworkX graph
        node_type_mapping: Mapping of node types to indices
    """
    # Group edges by type
    edge_groups = defaultdict(list)
    edge_attrs_groups = defaultdict(list)
    
    for source, target, attrs in G.edges(data=True):
        # Determine source and target node types
        source_type = _get_node_type_from_mapping(source, node_type_mapping)
        target_type = _get_node_type_from_mapping(target, node_type_mapping)
        
        if source_type is None or target_type is None:
            logger.warning(f"Could not determine node types for edge {source} -> {target}")
            continue
        
        # Determine edge type based on NetworkX edge attributes
        edge_type = _determine_hetero_edge_type(attrs, source_type, target_type)
        
        if edge_type is None:
            logger.warning(f"Could not determine edge type for {source} -> {target}")
            continue
        
        # Get node indices
        source_idx = node_type_mapping[source_type][source]
        target_idx = node_type_mapping[target_type][target]
        
        # Store edge
        hetero_edge_type = (source_type, edge_type, target_type)
        edge_groups[hetero_edge_type].append([source_idx, target_idx])
        edge_attrs_groups[hetero_edge_type].append(attrs)
    
    # Convert edge groups to tensors
    for hetero_edge_type, edges in edge_groups.items():
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            data[hetero_edge_type].edge_index = edge_index
            
            # Add edge attributes
            edge_attrs = edge_attrs_groups[hetero_edge_type]
            _add_edge_attributes(data, hetero_edge_type, edge_attrs)


def _get_node_type_from_mapping(node_id: str, node_type_mapping: Dict[str, Dict[str, int]]) -> Optional[str]:
    """Get the node type for a given node ID."""
    for node_type, mapping in node_type_mapping.items():
        if node_id in mapping:
            return node_type
    return None


def _determine_hetero_edge_type(attrs: Dict[str, Any], source_type: str, target_type: str) -> Optional[str]:
    """
    Determine the HeteroData edge type based on NetworkX edge attributes and node types.
    
    Args:
        attrs: Edge attributes from NetworkX
        source_type: Source node type
        target_type: Target node type
        
    Returns:
        Edge type string or None if cannot be determined
    """
    nx_edge_type = attrs.get("edge_type", "")
    
    # Map NetworkX edge types to HeteroData edge types
    edge_type_mapping = {
        "faced": "faced",
        "dismissed_by": "dismissed_by",
        "partnered_with": "partnered_with",
        "teammate_of": "teammate_of",
        "plays_for": "plays_for",
        "match_played_at": "played_at",
        "bowled_at": "bowled_at",
        "excels_against": "excels_against",
        "batter_event": "produced",
        "bowler_event": "conceded",
    }
    
    hetero_edge_type = edge_type_mapping.get(nx_edge_type)
    
    if hetero_edge_type is None:
        # Try to infer from node types
        if source_type == "player" and target_type == "player":
            hetero_edge_type = "faced"  # Default player-player relationship
        elif source_type == "player" and target_type == "team":
            hetero_edge_type = "plays_for"
        elif source_type == "player" and target_type == "venue":
            hetero_edge_type = "played_at"
        elif source_type == "team" and target_type == "venue":
            hetero_edge_type = "played_at"
        else:
            logger.warning(f"Cannot infer edge type for {source_type} -> {target_type} with attrs: {nx_edge_type}")
    
    return hetero_edge_type


def _add_edge_attributes(data: HeteroData, hetero_edge_type: Tuple[str, str, str], 
                        edge_attrs: List[Dict[str, Any]]) -> None:
    """
    Add edge attributes to HeteroData.
    
    Args:
        data: HeteroData object
        hetero_edge_type: Edge type tuple
        edge_attrs: List of edge attribute dictionaries
    """
    if not edge_attrs:
        return
    
    # Extract common attributes
    weights = []
    runs = []
    event_counts = []
    
    for attrs in edge_attrs:
        weights.append(attrs.get("weight", 1.0))
        runs.append(attrs.get("runs", 0))
        event_counts.append(attrs.get("event_count", 1))
    
    # Add as edge attributes
    data[hetero_edge_type].edge_weight = torch.tensor(weights, dtype=torch.float32)
    data[hetero_edge_type].edge_runs = torch.tensor(runs, dtype=torch.float32)
    data[hetero_edge_type].edge_event_counts = torch.tensor(event_counts, dtype=torch.float32)


def _add_metadata_to_hetero_data(data: HeteroData, G: nx.DiGraph, 
                                node_type_mapping: Dict[str, Dict[str, int]]) -> None:
    """
    Add metadata to HeteroData for debugging and analysis.
    
    Args:
        data: HeteroData object
        G: NetworkX graph
        node_type_mapping: Node type mapping
    """
    # Store original graph statistics
    data.metadata = {
        "num_nodes_nx": G.number_of_nodes(),
        "num_edges_nx": G.number_of_edges(),
        "node_type_counts": {nt: len(mapping) for nt, mapping in node_type_mapping.items()},
        "conversion_timestamp": torch.tensor([0.0])  # Placeholder
    }


def get_hetero_data_stats(data: HeteroData) -> Dict[str, Any]:
    """
    Get statistics about a HeteroData object.
    
    Args:
        data: HeteroData object
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        "node_types": list(data.node_types),
        "edge_types": list(data.edge_types),
        "num_node_types": len(data.node_types),
        "num_edge_types": len(data.edge_types),
    }
    
    # Node statistics
    for node_type in data.node_types:
        stats[f"num_nodes_{node_type}"] = data[node_type].num_nodes
        if hasattr(data[node_type], 'x'):
            stats[f"feature_dim_{node_type}"] = data[node_type].x.shape[1]
    
    # Edge statistics
    for edge_type in data.edge_types:
        stats[f"num_edges_{edge_type}"] = data[edge_type].edge_index.shape[1]
    
    return stats


def validate_hetero_data(data: HeteroData) -> List[str]:
    """
    Validate a HeteroData object for consistency.
    
    Args:
        data: HeteroData object to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check node types
    for node_type in data.node_types:
        if not hasattr(data[node_type], 'x'):
            errors.append(f"Node type '{node_type}' missing feature tensor 'x'")
        elif not hasattr(data[node_type], 'num_nodes'):
            errors.append(f"Node type '{node_type}' missing 'num_nodes'")
        elif data[node_type].x.shape[0] != data[node_type].num_nodes:
            errors.append(f"Node type '{node_type}': feature tensor size mismatch")
    
    # Check edge types
    for edge_type in data.edge_types:
        src_type, rel_type, dst_type = edge_type
        
        if not hasattr(data[edge_type], 'edge_index'):
            errors.append(f"Edge type '{edge_type}' missing 'edge_index'")
            continue
        
        edge_index = data[edge_type].edge_index
        
        # Check edge index shape
        if edge_index.shape[0] != 2:
            errors.append(f"Edge type '{edge_type}': edge_index should have shape [2, num_edges]")
        
        # Check node indices are valid
        if src_type in data.node_types:
            max_src_idx = edge_index[0].max().item() if edge_index.shape[1] > 0 else -1
            if max_src_idx >= data[src_type].num_nodes:
                errors.append(f"Edge type '{edge_type}': invalid source node index")
        
        if dst_type in data.node_types:
            max_dst_idx = edge_index[1].max().item() if edge_index.shape[1] > 0 else -1
            if max_dst_idx >= data[dst_type].num_nodes:
                errors.append(f"Edge type '{edge_type}': invalid target node index")
    
    return errors