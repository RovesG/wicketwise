# Purpose: Processes video-derived biomechanical signals for cricket players
# Author: Shamus Rae, Last Modified: 2024-01-15

"""
This module handles the extraction, aggregation, and integration of biomechanical
signals derived from video analysis into player node features. It supports
technique analysis for batters, bowlers, and fielders.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque
from datetime import datetime, timedelta
import networkx as nx
import logging

logger = logging.getLogger(__name__)


@dataclass
class BiomechanicalSignalSchema:
    """Schema definition for biomechanical signals per delivery."""
    
    # Batter biomechanical signals
    BATTER_SIGNALS = [
        'head_stability',           # 0.0-1.0: Head position consistency during shot
        'backlift_type',           # 0.0-1.0: Backlift technique score (straight=1.0, across=0.0)
        'footwork_direction',      # 0.0-1.0: Footwork alignment score (forward=1.0, back=0.0)
        'shot_commitment'          # 0.0-1.0: Decision commitment level
    ]
    
    # Bowler biomechanical signals
    BOWLER_SIGNALS = [
        'release_point_consistency',  # 0.0-1.0: Release point repeatability
        'arm_path',                  # 0.0-1.0: Arm action smoothness
        'follow_through_momentum'    # 0.0-1.0: Follow-through completion
    ]
    
    # Fielder biomechanical signals
    FIELDER_SIGNALS = [
        'closing_speed',            # 0.0-1.0: Speed of approach to ball
        'reaction_time',            # 0.0-1.0: Initial reaction speed (inverted, 1.0=fastest)
        'interception_type'         # 0.0-1.0: Interception technique quality
    ]
    
    @classmethod
    def get_all_signals(cls) -> List[str]:
        """Get all biomechanical signal names."""
        return cls.BATTER_SIGNALS + cls.BOWLER_SIGNALS + cls.FIELDER_SIGNALS
    
    @classmethod
    def get_signals_for_role(cls, role: str) -> List[str]:
        """Get biomechanical signals for a specific player role."""
        role_mapping = {
            'batter': cls.BATTER_SIGNALS,
            'bowler': cls.BOWLER_SIGNALS,
            'fielder': cls.FIELDER_SIGNALS
        }
        return role_mapping.get(role.lower(), [])


@dataclass
class BiomechanicalConfig:
    """Configuration for biomechanical feature processing."""
    
    rolling_window: int = 100              # Number of recent deliveries to consider
    min_deliveries_required: int = 10      # Minimum deliveries needed for reliable stats
    missing_value_threshold: float = 0.3   # Max proportion of missing values allowed
    default_signal_value: float = 0.5      # Default value for missing signals
    feature_prefix: str = "biomech_"       # Prefix for biomechanical features
    
    # Signal validation ranges
    signal_ranges: Dict[str, Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.signal_ranges is None:
            # All signals should be normalized to [0.0, 1.0]
            all_signals = BiomechanicalSignalSchema.get_all_signals()
            self.signal_ranges = {signal: (0.0, 1.0) for signal in all_signals}


class BiomechanicalSignalLoader:
    """Loads and validates biomechanical signals from JSON data."""
    
    def __init__(self, config: BiomechanicalConfig = None):
        self.config = config or BiomechanicalConfig()
        self.schema = BiomechanicalSignalSchema()
    
    def load_from_json(self, json_path: str) -> Dict[str, Any]:
        """Load biomechanical signals from JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return self._validate_and_normalize(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading biomechanical data from {json_path}: {e}")
            return {}
    
    def load_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load biomechanical signals from dictionary."""
        return self._validate_and_normalize(data)
    
    def _validate_and_normalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize biomechanical signal data."""
        validated_data = {}
        
        for delivery_id, delivery_data in data.items():
            validated_delivery = {}
            
            # Validate each signal
            for signal_name in self.schema.get_all_signals():
                if signal_name in delivery_data:
                    value = delivery_data[signal_name]
                    validated_value = self._validate_signal_value(signal_name, value)
                    if validated_value is not None:
                        validated_delivery[signal_name] = validated_value
                else:
                    # Use default value for missing signals
                    validated_delivery[signal_name] = self.config.default_signal_value
            
            # Only include delivery if it has some valid signals
            if validated_delivery:
                validated_data[delivery_id] = validated_delivery
        
        return validated_data
    
    def _validate_signal_value(self, signal_name: str, value: Any) -> Optional[float]:
        """Validate and normalize a single signal value."""
        try:
            float_value = float(value)
            
            # Check if value is within expected range
            min_val, max_val = self.config.signal_ranges[signal_name]
            if min_val <= float_value <= max_val:
                return float_value
            else:
                logger.warning(f"Signal {signal_name} value {float_value} outside range [{min_val}, {max_val}]")
                # Clamp to valid range
                return max(min_val, min(max_val, float_value))
                
        except (ValueError, TypeError):
            logger.warning(f"Invalid signal value for {signal_name}: {value}")
            return None


class BiomechanicalAggregator:
    """Aggregates biomechanical signals into rolling statistics per player."""
    
    def __init__(self, config: BiomechanicalConfig = None):
        self.config = config or BiomechanicalConfig()
        self.schema = BiomechanicalSignalSchema()
        
        # Rolling data storage per player
        self.player_signal_history: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.config.rolling_window))
        )
    
    def add_delivery_signals(
        self, 
        player_id: str, 
        delivery_signals: Dict[str, float],
        player_role: str = None
    ) -> None:
        """Add biomechanical signals for a single delivery."""
        
        # Determine relevant signals based on player role
        if player_role:
            relevant_signals = self.schema.get_signals_for_role(player_role)
        else:
            relevant_signals = self.schema.get_all_signals()
        
        # Add signals to rolling history
        for signal_name in relevant_signals:
            if signal_name in delivery_signals:
                value = delivery_signals[signal_name]
                self.player_signal_history[player_id][signal_name].append(value)
            else:
                # Add default value for missing signals in the role
                self.player_signal_history[player_id][signal_name].append(self.config.default_signal_value)
    
    def get_player_aggregated_features(self, player_id: str) -> Dict[str, float]:
        """Get aggregated biomechanical features for a player."""
        if player_id not in self.player_signal_history:
            return {}
        
        aggregated_features = {}
        player_signals = self.player_signal_history[player_id]
        
        for signal_name, signal_values in player_signals.items():
            if len(signal_values) < self.config.min_deliveries_required:
                # Not enough data for reliable statistics
                continue
            
            # Calculate various aggregation statistics
            values_array = np.array(list(signal_values))
            
            # Check for missing value threshold
            non_missing_ratio = np.sum(~np.isnan(values_array)) / len(values_array)
            if non_missing_ratio < (1 - self.config.missing_value_threshold):
                continue
            
            # Remove NaN values for calculation
            clean_values = values_array[~np.isnan(values_array)]
            
            if len(clean_values) > 0:
                # Generate multiple aggregation features
                feature_base = f"{self.config.feature_prefix}{signal_name}"
                
                aggregated_features[f"{feature_base}_mean"] = float(np.mean(clean_values))
                aggregated_features[f"{feature_base}_std"] = float(np.std(clean_values))
                aggregated_features[f"{feature_base}_recent"] = float(clean_values[-1])  # Most recent value
                
                # Trend analysis (last 20% vs first 80%)
                if len(clean_values) >= 10:  # Reduced requirement for trend analysis
                    split_point = max(1, int(len(clean_values) * 0.8))  # Ensure at least 1 value in early part
                    early_mean = np.mean(clean_values[:split_point])
                    recent_mean = np.mean(clean_values[split_point:])
                    trend = (recent_mean - early_mean) / (early_mean + 1e-8)  # Avoid division by zero
                    aggregated_features[f"{feature_base}_trend"] = float(trend)
        
        return aggregated_features
    
    def get_all_players_features(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated features for all players."""
        all_features = {}
        for player_id in self.player_signal_history:
            features = self.get_player_aggregated_features(player_id)
            if features:  # Only include players with valid features
                all_features[player_id] = features
        return all_features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all possible biomechanical feature names."""
        feature_names = []
        
        for signal_name in self.schema.get_all_signals():
            base_name = f"{self.config.feature_prefix}{signal_name}"
            feature_names.extend([
                f"{base_name}_mean",
                f"{base_name}_std", 
                f"{base_name}_recent",
                f"{base_name}_trend"
            ])
        
        return feature_names
    
    def reset_player_history(self, player_id: str) -> None:
        """Reset biomechanical history for a specific player."""
        if player_id in self.player_signal_history:
            del self.player_signal_history[player_id]
    
    def get_player_delivery_count(self, player_id: str) -> int:
        """Get number of deliveries recorded for a player."""
        if player_id not in self.player_signal_history:
            return 0
        
        # Return maximum count across all signals for this player
        max_count = 0
        for signal_values in self.player_signal_history[player_id].values():
            max_count = max(max_count, len(signal_values))
        
        return max_count


def process_match_biomechanical_data(
    match_data: pd.DataFrame,
    biomechanical_data: Dict[str, Any],
    config: BiomechanicalConfig = None
) -> Dict[str, Dict[str, float]]:
    """
    Process biomechanical data for an entire match and return aggregated features.
    
    Args:
        match_data: DataFrame with ball-by-ball match data
        biomechanical_data: Dictionary with biomechanical signals per delivery
        config: Configuration for processing
    
    Returns:
        Dictionary mapping player_id to aggregated biomechanical features
    """
    config = config or BiomechanicalConfig()
    aggregator = BiomechanicalAggregator(config)
    loader = BiomechanicalSignalLoader(config)
    
    # Validate and load biomechanical data
    validated_signals = loader.load_from_dict(biomechanical_data)
    
    # Process each ball in the match
    for _, ball_row in match_data.iterrows():
        ball_id = f"{ball_row.get('match_id', 'unknown')}_{ball_row.get('innings', 0)}_{ball_row.get('over', 0)}_{ball_row.get('ball', 0)}"
        
        if ball_id in validated_signals:
            delivery_signals = validated_signals[ball_id]
            
            # Add signals for batter
            batter_id = ball_row.get('batter')
            if batter_id:
                aggregator.add_delivery_signals(batter_id, delivery_signals, 'batter')
            
            # Add signals for bowler
            bowler_id = ball_row.get('bowler')
            if bowler_id:
                aggregator.add_delivery_signals(bowler_id, delivery_signals, 'bowler')
            
            # Add signals for fielders (if fielder info available)
            fielder_id = ball_row.get('fielder')
            if fielder_id and fielder_id not in ['', 'N/A', None]:
                aggregator.add_delivery_signals(fielder_id, delivery_signals, 'fielder')
    
    return aggregator.get_all_players_features()


def add_biomechanical_features_to_graph(
    graph: nx.Graph,
    biomechanical_features: Dict[str, Dict[str, float]],
    config: BiomechanicalConfig = None
) -> nx.Graph:
    """
    Add biomechanical features to player nodes in the cricket knowledge graph.
    
    Args:
        graph: NetworkX graph with cricket data
        biomechanical_features: Dictionary mapping player_id to features
        config: Configuration for processing
    
    Returns:
        Updated graph with biomechanical features added to player nodes
    """
    config = config or BiomechanicalConfig()
    updated_count = 0
    
    for node_id, node_data in graph.nodes(data=True):
        # Check if this is a player node
        if node_data.get('node_type') == 'player':
            player_id = node_id
            
            if player_id in biomechanical_features:
                player_biomech_features = biomechanical_features[player_id]
                
                # Get existing features or create new array
                existing_features = node_data.get('features', np.array([]))
                if isinstance(existing_features, list):
                    existing_features = np.array(existing_features)
                
                # Convert biomechanical features to array
                feature_names = sorted(player_biomech_features.keys())  # Ensure consistent ordering
                biomech_array = np.array([player_biomech_features[name] for name in feature_names])
                
                # Concatenate with existing features
                if existing_features.size > 0:
                    combined_features = np.concatenate([existing_features, biomech_array])
                else:
                    combined_features = biomech_array
                
                # Update node with combined features
                graph.nodes[node_id]['features'] = combined_features
                graph.nodes[node_id]['biomechanical_features'] = player_biomech_features
                graph.nodes[node_id]['feature_names'] = node_data.get('feature_names', []) + feature_names
                
                updated_count += 1
                logger.debug(f"Added {len(biomech_array)} biomechanical features to player {player_id}")
    
    logger.info(f"Added biomechanical features to {updated_count} player nodes")
    return graph


def get_biomechanical_feature_dimension() -> int:
    """Get the total dimension of biomechanical features per player."""
    # Each signal generates 4 aggregated features (mean, std, recent, trend)
    num_signals = len(BiomechanicalSignalSchema.get_all_signals())
    return num_signals * 4


@dataclass
class BiomechanicalEventMetadata:
    """Metadata for biomechanical event nodes."""
    
    delivery_id: str
    match_id: str
    innings: int
    over: float
    ball: int
    timestamp: Optional[datetime] = None
    batter: Optional[str] = None
    bowler: Optional[str] = None
    fielder: Optional[str] = None
    runs_scored: Optional[int] = None
    wicket_type: Optional[str] = None
    video_frame_start: Optional[int] = None
    video_frame_end: Optional[int] = None
    confidence_score: Optional[float] = None
    processing_version: str = "1.0"


def create_biomechanical_event_nodes(
    graph: nx.Graph,
    match_data: pd.DataFrame,
    biomechanical_data: Dict[str, Dict[str, float]],
    config: BiomechanicalConfig = None
) -> nx.Graph:
    """
    Create BiomechanicalEvent nodes and connect them to delivery events.
    
    Args:
        graph: NetworkX graph with cricket data
        match_data: DataFrame with ball-by-ball match data
        biomechanical_data: Dictionary with biomechanical signals per delivery
        config: Configuration for processing
    
    Returns:
        Updated graph with BiomechanicalEvent nodes and has_biomechanics edges
    """
    config = config or BiomechanicalConfig()
    loader = BiomechanicalSignalLoader(config)
    
    # Validate biomechanical data
    validated_signals = loader.load_from_dict(biomechanical_data)
    
    biomech_nodes_created = 0
    biomech_edges_created = 0
    
    # Process each ball in the match
    for _, ball_row in match_data.iterrows():
        # Create delivery identifier
        delivery_id = f"{ball_row.get('match_id', 'unknown')}_{ball_row.get('innings', 0)}_{ball_row.get('over', 0)}_{ball_row.get('ball', 0)}"
        
        if delivery_id in validated_signals:
            delivery_signals = validated_signals[delivery_id]
            
            # Create BiomechanicalEvent node
            biomech_event_id = f"biomech_event_{delivery_id}"
            
            # Create metadata
            metadata = BiomechanicalEventMetadata(
                delivery_id=delivery_id,
                match_id=ball_row.get('match_id', 'unknown'),
                innings=ball_row.get('innings', 0),
                over=ball_row.get('over', 0.0),
                ball=ball_row.get('ball', 0),
                batter=ball_row.get('batter'),
                bowler=ball_row.get('bowler'),
                fielder=ball_row.get('fielder'),
                runs_scored=ball_row.get('runs_scored'),
                wicket_type=ball_row.get('wicket_type'),
                confidence_score=1.0  # Default confidence
            )
            
            # Add timestamp if date information is available
            if 'date' in ball_row and pd.notna(ball_row['date']):
                try:
                    if isinstance(ball_row['date'], str):
                        metadata.timestamp = datetime.fromisoformat(ball_row['date'])
                    elif isinstance(ball_row['date'], datetime):
                        metadata.timestamp = ball_row['date']
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse date for delivery {delivery_id}: {ball_row['date']}")
            
            # Add BiomechanicalEvent node to graph
            graph.add_node(
                biomech_event_id,
                node_type='biomechanical_event',
                delivery_id=delivery_id,
                biomechanical_signals=delivery_signals,
                metadata=metadata,
                signal_count=len(delivery_signals),
                signal_names=list(delivery_signals.keys()),
                features=np.array(list(delivery_signals.values())),  # Raw signal values as features
                timestamp=metadata.timestamp,
                match_id=metadata.match_id,
                innings=metadata.innings,
                over=metadata.over,
                ball=metadata.ball
            )
            
            biomech_nodes_created += 1
            
            # Find corresponding delivery event nodes to link to
            delivery_event_nodes = [
                node_id for node_id, node_data in graph.nodes(data=True)
                if node_data.get('node_type') == 'event' and 
                node_data.get('delivery_id') == delivery_id
            ]
            
            # If no specific delivery event nodes exist, create connections to general event nodes
            if not delivery_event_nodes:
                # Determine event type based on ball outcome
                runs = ball_row.get('runs_scored', 0)
                is_wicket = ball_row.get('wicket_type') not in [None, '', 'not_out']
                
                if is_wicket:
                    event_type = 'wicket'
                elif runs >= 6:
                    event_type = 'six'
                elif runs >= 4:
                    event_type = 'four'
                else:
                    event_type = 'dot'
                
                # Find event node of this type
                event_nodes = [
                    node_id for node_id, node_data in graph.nodes(data=True)
                    if node_data.get('node_type') == 'event' and node_id == event_type
                ]
                
                delivery_event_nodes = event_nodes
            
            # Create has_biomechanics edges
            for event_node_id in delivery_event_nodes:
                # Add temporal information to edge
                edge_attrs = {
                    'edge_type': 'has_biomechanics',
                    'weight': 1.0,
                    'delivery_id': delivery_id,
                    'signal_count': len(delivery_signals),
                    'timestamp': metadata.timestamp,
                    'confidence': metadata.confidence_score or 1.0
                }
                
                # Add temporal encoding if timestamp is available
                if metadata.timestamp:
                    # Calculate days ago from a reference date (e.g., most recent match)
                    reference_date = datetime.now()
                    days_ago = (reference_date - metadata.timestamp).days
                    edge_attrs['days_ago'] = days_ago
                
                graph.add_edge(
                    event_node_id,
                    biomech_event_id,
                    **edge_attrs
                )
                
                biomech_edges_created += 1
                logger.debug(f"Created has_biomechanics edge: {event_node_id} -> {biomech_event_id}")
    
    logger.info(f"Created {biomech_nodes_created} BiomechanicalEvent nodes and {biomech_edges_created} has_biomechanics edges")
    return graph


def get_biomechanical_events_for_delivery(
    graph: nx.Graph,
    delivery_id: str
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Get all biomechanical event nodes for a specific delivery.
    
    Args:
        graph: NetworkX graph with cricket data
        delivery_id: Delivery identifier
    
    Returns:
        List of (node_id, node_data) tuples for biomechanical events
    """
    biomech_events = []
    
    for node_id, node_data in graph.nodes(data=True):
        if (node_data.get('node_type') == 'biomechanical_event' and 
            node_data.get('delivery_id') == delivery_id):
            biomech_events.append((node_id, node_data))
    
    return biomech_events


def get_biomechanical_events_by_match(
    graph: nx.Graph,
    match_id: str
) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    """
    Get all biomechanical events grouped by delivery for a match.
    
    Args:
        graph: NetworkX graph with cricket data
        match_id: Match identifier
    
    Returns:
        Dictionary mapping delivery_id to list of biomechanical events
    """
    match_events = defaultdict(list)
    
    for node_id, node_data in graph.nodes(data=True):
        if (node_data.get('node_type') == 'biomechanical_event' and 
            node_data.get('match_id') == match_id):
            delivery_id = node_data.get('delivery_id')
            if delivery_id:
                match_events[delivery_id].append((node_id, node_data))
    
    return dict(match_events)


def analyze_biomechanical_event_patterns(
    graph: nx.Graph,
    player_id: str,
    event_type: str = None
) -> Dict[str, Any]:
    """
    Analyze biomechanical patterns for a specific player across events.
    
    Args:
        graph: NetworkX graph with cricket data
        player_id: Player identifier
        event_type: Optional filter for specific event types
    
    Returns:
        Dictionary with pattern analysis results
    """
    player_biomech_events = []
    
    # Find all biomechanical events involving this player
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get('node_type') == 'biomechanical_event':
            metadata = node_data.get('metadata')
            if metadata and (metadata.batter == player_id or 
                           metadata.bowler == player_id or 
                           metadata.fielder == player_id):
                
                # Filter by event type if specified
                if event_type:
                    # Check connected event nodes
                    connected_events = [
                        neighbor for neighbor in graph.neighbors(node_id)
                        if graph.nodes[neighbor].get('node_type') == 'event'
                    ]
                    if event_type not in connected_events:
                        continue
                
                player_biomech_events.append((node_id, node_data))
    
    if not player_biomech_events:
        return {'player_id': player_id, 'event_count': 0, 'patterns': {}}
    
    # Analyze patterns
    all_signals = defaultdict(list)
    event_outcomes = []
    timestamps = []
    
    for node_id, node_data in player_biomech_events:
        signals = node_data.get('biomechanical_signals', {})
        metadata = node_data.get('metadata')
        
        # Collect signal values
        for signal_name, value in signals.items():
            all_signals[signal_name].append(value)
        
        # Collect outcomes and timestamps
        if metadata:
            event_outcomes.append({
                'runs_scored': metadata.runs_scored,
                'wicket_type': metadata.wicket_type,
                'delivery_id': metadata.delivery_id
            })
            if metadata.timestamp:
                timestamps.append(metadata.timestamp)
    
    # Calculate pattern statistics
    pattern_analysis = {
        'player_id': player_id,
        'event_count': len(player_biomech_events),
        'patterns': {},
        'temporal_span': None,
        'outcome_distribution': {}
    }
    
    # Signal pattern analysis
    for signal_name, values in all_signals.items():
        if values:
            pattern_analysis['patterns'][signal_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'trend': float(np.corrcoef(range(len(values)), values)[0, 1]) if len(values) > 1 else 0.0
            }
    
    # Temporal analysis
    if timestamps:
        timestamps.sort()
        pattern_analysis['temporal_span'] = {
            'start': timestamps[0].isoformat() if timestamps[0] else None,
            'end': timestamps[-1].isoformat() if timestamps[-1] else None,
            'duration_days': (timestamps[-1] - timestamps[0]).days if len(timestamps) > 1 else 0
        }
    
    # Outcome distribution
    runs_distribution = defaultdict(int)
    wicket_types = defaultdict(int)
    
    for outcome in event_outcomes:
        runs = outcome.get('runs_scored', 0)
        runs_distribution[runs] += 1
        
        wicket = outcome.get('wicket_type')
        if wicket and wicket not in [None, '', 'not_out']:
            wicket_types[wicket] += 1
    
    pattern_analysis['outcome_distribution'] = {
        'runs': dict(runs_distribution),
        'wickets': dict(wicket_types)
    }
    
    return pattern_analysis


def create_sample_biomechanical_data(
    match_data: pd.DataFrame,
    noise_level: float = 0.1
) -> Dict[str, Dict[str, float]]:
    """
    Create sample biomechanical data for testing purposes.
    
    Args:
        match_data: DataFrame with ball-by-ball match data
        noise_level: Amount of random noise to add to signals
    
    Returns:
        Dictionary with sample biomechanical signals
    """
    np.random.seed(42)  # For reproducible results
    sample_data = {}
    schema = BiomechanicalSignalSchema()
    
    for _, ball_row in match_data.iterrows():
        ball_id = f"{ball_row.get('match_id', 'test')}_{ball_row.get('innings', 1)}_{ball_row.get('over', 1)}_{ball_row.get('ball', 1)}"
        
        # Generate sample signals with some realistic patterns
        signals = {}
        
        # Batter signals - vary based on runs scored
        runs = ball_row.get('runs_scored', 0)
        if runs >= 4:  # Boundary - good technique
            signals['head_stability'] = np.clip(0.8 + np.random.normal(0, noise_level), 0, 1)
            signals['shot_commitment'] = np.clip(0.9 + np.random.normal(0, noise_level), 0, 1)
        else:  # Regular delivery
            signals['head_stability'] = np.clip(0.6 + np.random.normal(0, noise_level), 0, 1)
            signals['shot_commitment'] = np.clip(0.7 + np.random.normal(0, noise_level), 0, 1)
        
        signals['backlift_type'] = np.clip(0.7 + np.random.normal(0, noise_level), 0, 1)
        signals['footwork_direction'] = np.clip(0.6 + np.random.normal(0, noise_level), 0, 1)
        
        # Bowler signals - vary based on wicket
        is_wicket = ball_row.get('wicket_type') not in [None, '', 'not_out']
        if is_wicket:  # Good bowling
            signals['release_point_consistency'] = np.clip(0.9 + np.random.normal(0, noise_level), 0, 1)
            signals['arm_path'] = np.clip(0.8 + np.random.normal(0, noise_level), 0, 1)
        else:  # Regular delivery
            signals['release_point_consistency'] = np.clip(0.7 + np.random.normal(0, noise_level), 0, 1)
            signals['arm_path'] = np.clip(0.7 + np.random.normal(0, noise_level), 0, 1)
        
        signals['follow_through_momentum'] = np.clip(0.75 + np.random.normal(0, noise_level), 0, 1)
        
        # Fielder signals - random but realistic
        signals['closing_speed'] = np.clip(0.6 + np.random.normal(0, noise_level), 0, 1)
        signals['reaction_time'] = np.clip(0.7 + np.random.normal(0, noise_level), 0, 1)
        signals['interception_type'] = np.clip(0.65 + np.random.normal(0, noise_level), 0, 1)
        
        sample_data[ball_id] = signals
    
    return sample_data