# Purpose: Graph builder with integrated learnable temporal decay
# Author: Shamus Rae, Last Modified: 2024-01-15

"""
This module integrates learnable temporal decay into the graph building process,
replacing fixed exponential decay with adaptive, feature-specific temporal weighting
for edges and node features.
"""

import torch
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging

from crickformers.model.learnable_temporal_decay import (
    LearnableTemporalDecay,
    AdaptiveTemporalEncoder,
    create_learnable_temporal_decay
)

logger = logging.getLogger(__name__)


class TemporalGraphBuilder:
    """Graph builder with integrated learnable temporal decay."""
    
    def __init__(
        self,
        feature_names: List[str],
        temporal_config: Optional[Dict[str, Any]] = None,
        use_adaptive_encoder: bool = False,
        encoder_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize temporal graph builder.
        
        Args:
            feature_names: List of feature names for temporal decay
            temporal_config: Configuration for learnable temporal decay
            use_adaptive_encoder: Whether to use adaptive temporal encoder
            encoder_config: Configuration for adaptive encoder
        """
        self.feature_names = feature_names
        self.temporal_config = temporal_config or {}
        self.use_adaptive_encoder = use_adaptive_encoder
        
        # Create learnable temporal decay module
        self.temporal_decay = create_learnable_temporal_decay(
            feature_names, temporal_config
        )
        
        # Create adaptive encoder if requested
        if use_adaptive_encoder:
            encoder_config = encoder_config or {}
            self.adaptive_encoder = AdaptiveTemporalEncoder(
                feature_names, **encoder_config
            )
        else:
            self.adaptive_encoder = None
        
        # Feature statistics for monitoring
        self.feature_stats = {}
        self.edge_stats = {}
    
    def add_temporal_edges(
        self,
        graph: nx.Graph,
        match_data: pd.DataFrame,
        reference_date: Optional[datetime] = None
    ) -> nx.Graph:
        """
        Add edges with learnable temporal decay weights.
        
        Args:
            graph: NetworkX graph
            match_data: Ball-by-ball match data with date information
            reference_date: Reference date for temporal calculations
        
        Returns:
            Graph with temporal edges
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        logger.info("Adding temporal edges with learnable decay")
        
        # Process each match
        for match_id, match_group in match_data.groupby('match_id'):
            match_date = self._extract_match_date(match_group)
            if match_date is None:
                continue
            
            days_ago = (reference_date - match_date).days
            
            # Add player-player edges (partnerships, dismissals, etc.)
            self._add_player_edges(graph, match_group, days_ago)
            
            # Add player-event edges
            self._add_event_edges(graph, match_group, days_ago)
            
            # Add contextual edges (venue, conditions, etc.)
            self._add_contextual_edges(graph, match_group, days_ago)
        
        logger.info(f"Added temporal edges for {len(match_data['match_id'].unique())} matches")
        return graph
    
    def compute_form_vectors(
        self,
        player_history: Dict[str, pd.DataFrame],
        reference_date: Optional[datetime] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute form vectors using learnable temporal decay.
        
        Args:
            player_history: Dictionary mapping player_id to historical performance data
            reference_date: Reference date for temporal calculations
        
        Returns:
            Dictionary mapping player_id to form vector
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        form_vectors = {}
        
        for player_id, history_df in player_history.items():
            if len(history_df) == 0:
                continue
            
            # Extract temporal information
            days_ago_list = []
            feature_matrix = []
            
            for _, row in history_df.iterrows():
                match_date = pd.to_datetime(row.get('match_date', reference_date))
                days_ago = (reference_date - match_date).days
                days_ago_list.append(days_ago)
                
                # Extract feature values
                feature_values = []
                for feature_name in self.feature_names:
                    value = row.get(feature_name, 0.0)
                    feature_values.append(float(value))
                
                feature_matrix.append(feature_values)
            
            if not feature_matrix:
                continue
            
            # Convert to tensors
            days_ago_tensor = torch.tensor(days_ago_list, dtype=torch.float32)
            feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
            
            # Compute form vector using learnable decay
            form_vector = self.temporal_decay.get_aggregated_form_vector(
                feature_tensor, days_ago_tensor, self.feature_names
            )
            
            form_vectors[player_id] = form_vector
            
            # Update statistics
            self._update_player_stats(player_id, form_vector, days_ago_tensor)
        
        logger.info(f"Computed form vectors for {len(form_vectors)} players")
        return form_vectors
    
    def update_node_features(
        self,
        graph: nx.Graph,
        form_vectors: Dict[str, torch.Tensor],
        feature_dim: int = 64
    ) -> nx.Graph:
        """
        Update node features with temporal form vectors.
        
        Args:
            graph: NetworkX graph
            form_vectors: Dictionary of player form vectors
            feature_dim: Target feature dimension
        
        Returns:
            Graph with updated node features
        """
        nodes_updated = 0
        
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get('node_type') == 'player' and node_id in form_vectors:
                form_vector = form_vectors[node_id]
                
                # Use adaptive encoder if available
                if self.adaptive_encoder is not None:
                    # Create dummy days_ago (use 0 for current state)
                    days_ago = torch.tensor([0.0])
                    encoded_features = self.adaptive_encoder(
                        days_ago, form_vector.unsqueeze(0), self.feature_names
                    ).squeeze(0)
                else:
                    encoded_features = form_vector
                
                # Pad or truncate to target dimension
                if len(encoded_features) < feature_dim:
                    padding = torch.zeros(feature_dim - len(encoded_features))
                    encoded_features = torch.cat([encoded_features, padding])
                elif len(encoded_features) > feature_dim:
                    encoded_features = encoded_features[:feature_dim]
                
                # Update node features
                node_data['features'] = encoded_features.detach().numpy()
                node_data['temporal_form'] = form_vector.detach().numpy()
                node_data['form_updated'] = True
                
                nodes_updated += 1
        
        logger.info(f"Updated features for {nodes_updated} player nodes")
        return graph
    
    def _extract_match_date(self, match_group: pd.DataFrame) -> Optional[datetime]:
        """Extract match date from match data."""
        date_columns = ['match_date', 'date', 'timestamp']
        
        for col in date_columns:
            if col in match_group.columns:
                date_value = match_group[col].iloc[0]
                if pd.notna(date_value):
                    try:
                        return pd.to_datetime(date_value)
                    except:
                        continue
        
        return None
    
    def _add_player_edges(
        self,
        graph: nx.Graph,
        match_group: pd.DataFrame,
        days_ago: int
    ):
        """Add player-player edges with temporal weights."""
        # Partnership edges
        partnerships = self._extract_partnerships(match_group)
        
        for partnership in partnerships:
            player1, player2 = partnership['players']
            
            if not (graph.has_node(player1) and graph.has_node(player2)):
                continue
            
            # Extract partnership features
            partnership_features = self._extract_partnership_features(partnership)
            
            # Compute temporal weight
            base_weight = partnership['runs'] / max(partnership['balls'], 1)
            
            # Use feature-specific decay if features available
            if partnership_features:
                feature_values = torch.tensor([partnership_features.get(name, 0.0) 
                                             for name in self.feature_names], dtype=torch.float32)
                days_ago_tensor = torch.tensor([days_ago], dtype=torch.float32)
                
                temporal_weight = self.temporal_decay.compute_feature_weights(
                    days_ago_tensor, feature_values.unsqueeze(0), self.feature_names
                ).mean().item()
            else:
                days_ago_tensor = torch.tensor([days_ago], dtype=torch.float32)
                temporal_weight = self.temporal_decay.compute_temporal_weights(days_ago_tensor).item()
            
            final_weight = base_weight * temporal_weight
            
            # Add or update edge
            if graph.has_edge(player1, player2):
                existing_weight = graph[player1][player2].get('weight', 0.0)
                graph[player1][player2]['weight'] = existing_weight + final_weight
            else:
                graph.add_edge(player1, player2, 
                             edge_type='partnership',
                             weight=final_weight,
                             temporal_weight=temporal_weight,
                             days_ago=days_ago,
                             partnership_data=partnership)
    
    def _add_event_edges(
        self,
        graph: nx.Graph,
        match_group: pd.DataFrame,
        days_ago: int
    ):
        """Add player-event edges with temporal weights."""
        # Create event nodes if they don't exist
        event_types = ['boundary', 'wicket', 'dot', 'single']
        for event_type in event_types:
            if not graph.has_node(event_type):
                graph.add_node(event_type, node_type='event', event_type=event_type)
        
        # Process each ball
        for _, ball_row in match_group.iterrows():
            runs = ball_row.get('runs_scored', 0)
            wicket = ball_row.get('wicket_type')
            batter = ball_row.get('batter')
            bowler = ball_row.get('bowler')
            
            if not batter or not bowler:
                continue
            
            # Determine event type
            if wicket:
                event_type = 'wicket'
            elif runs >= 4:
                event_type = 'boundary'
            elif runs == 0:
                event_type = 'dot'
            else:
                event_type = 'single'
            
            # Extract ball features
            ball_features = self._extract_ball_features(ball_row)
            
            # Compute temporal weight
            if ball_features:
                feature_values = torch.tensor([ball_features.get(name, 0.0) 
                                             for name in self.feature_names], dtype=torch.float32)
                days_ago_tensor = torch.tensor([days_ago], dtype=torch.float32)
                
                temporal_weight = self.temporal_decay.compute_feature_weights(
                    days_ago_tensor, feature_values.unsqueeze(0), self.feature_names
                ).mean().item()
            else:
                days_ago_tensor = torch.tensor([days_ago], dtype=torch.float32)
                temporal_weight = self.temporal_decay.compute_temporal_weights(days_ago_tensor).item()
            
            # Add batter-event edge
            if graph.has_node(batter) and graph.has_node(event_type):
                base_weight = 1.0
                final_weight = base_weight * temporal_weight
                
                if graph.has_edge(batter, event_type):
                    existing_weight = graph[batter][event_type].get('weight', 0.0)
                    graph[batter][event_type]['weight'] = existing_weight + final_weight
                else:
                    graph.add_edge(batter, event_type,
                                 edge_type='batter_event',
                                 weight=final_weight,
                                 temporal_weight=temporal_weight,
                                 days_ago=days_ago)
            
            # Add bowler-event edge
            if graph.has_node(bowler) and graph.has_node(event_type):
                base_weight = 1.0
                final_weight = base_weight * temporal_weight
                
                if graph.has_edge(bowler, event_type):
                    existing_weight = graph[bowler][event_type].get('weight', 0.0)
                    graph[bowler][event_type]['weight'] = existing_weight + final_weight
                else:
                    graph.add_edge(bowler, event_type,
                                 edge_type='bowler_event',
                                 weight=final_weight,
                                 temporal_weight=temporal_weight,
                                 days_ago=days_ago)
    
    def _add_contextual_edges(
        self,
        graph: nx.Graph,
        match_group: pd.DataFrame,
        days_ago: int
    ):
        """Add contextual edges (venue, conditions) with temporal weights."""
        venue = match_group.get('venue', pd.Series()).iloc[0] if len(match_group) > 0 else None
        conditions = match_group.get('conditions', pd.Series()).iloc[0] if len(match_group) > 0 else None
        
        # Add venue node if not exists
        if venue and not graph.has_node(venue):
            graph.add_node(venue, node_type='venue', venue_name=venue)
        
        # Add conditions node if not exists
        if conditions and not graph.has_node(f"conditions_{conditions}"):
            graph.add_node(f"conditions_{conditions}", node_type='conditions', condition_type=conditions)
        
        # Connect players to venue and conditions
        players = set(match_group['batter'].dropna()) | set(match_group['bowler'].dropna())
        
        days_ago_tensor = torch.tensor([days_ago], dtype=torch.float32)
        temporal_weight = self.temporal_decay.compute_temporal_weights(days_ago_tensor).item()
        
        for player in players:
            if not graph.has_node(player):
                continue
            
            # Player-venue edge
            if venue and graph.has_node(venue):
                base_weight = 1.0
                final_weight = base_weight * temporal_weight
                
                if graph.has_edge(player, venue):
                    existing_weight = graph[player][venue].get('weight', 0.0)
                    graph[player][venue]['weight'] = existing_weight + final_weight
                else:
                    graph.add_edge(player, venue,
                                 edge_type='player_venue',
                                 weight=final_weight,
                                 temporal_weight=temporal_weight,
                                 days_ago=days_ago)
            
            # Player-conditions edge
            if conditions and graph.has_node(f"conditions_{conditions}"):
                base_weight = 1.0
                final_weight = base_weight * temporal_weight
                
                condition_node = f"conditions_{conditions}"
                if graph.has_edge(player, condition_node):
                    existing_weight = graph[player][condition_node].get('weight', 0.0)
                    graph[player][condition_node]['weight'] = existing_weight + final_weight
                else:
                    graph.add_edge(player, condition_node,
                                 edge_type='player_conditions',
                                 weight=final_weight,
                                 temporal_weight=temporal_weight,
                                 days_ago=days_ago)
    
    def _extract_partnerships(self, match_group: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract partnership information from match data."""
        partnerships = []
        
        for (batter, non_striker), partnership_group in match_group.groupby(['batter', 'non_striker']):
            if pd.isna(batter) or pd.isna(non_striker):
                continue
            
            partnership = {
                'players': (batter, non_striker),
                'runs': partnership_group['runs_scored'].sum(),
                'balls': len(partnership_group),
                'boundaries': len(partnership_group[partnership_group['runs_scored'] >= 4]),
                'wickets': len(partnership_group[partnership_group['wicket_type'].notna()]),
                'data': partnership_group
            }
            
            partnerships.append(partnership)
        
        return partnerships
    
    def _extract_partnership_features(self, partnership: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from partnership data."""
        features = {}
        
        # Basic features
        features['partnership_runs'] = partnership['runs']
        features['partnership_balls'] = partnership['balls']
        features['partnership_run_rate'] = partnership['runs'] / max(partnership['balls'], 1) * 6
        features['partnership_boundaries'] = partnership['boundaries']
        features['partnership_boundary_rate'] = partnership['boundaries'] / max(partnership['balls'], 1)
        
        # Advanced features
        partnership_data = partnership['data']
        features['strike_rotation'] = len(partnership_data[partnership_data['runs_scored'] % 2 == 1]) / max(partnership['balls'], 1)
        features['dot_ball_rate'] = len(partnership_data[partnership_data['runs_scored'] == 0]) / max(partnership['balls'], 1)
        
        return features
    
    def _extract_ball_features(self, ball_row: pd.Series) -> Dict[str, float]:
        """Extract features from ball data."""
        features = {}
        
        # Basic features
        features['runs_scored'] = float(ball_row.get('runs_scored', 0))
        features['is_boundary'] = float(ball_row.get('runs_scored', 0) >= 4)
        features['is_wicket'] = float(pd.notna(ball_row.get('wicket_type')))
        features['is_dot'] = float(ball_row.get('runs_scored', 0) == 0)
        
        # Contextual features
        features['over'] = float(ball_row.get('over', 0))
        features['ball_in_over'] = float(ball_row.get('ball', 0))
        
        return features
    
    def _update_player_stats(
        self,
        player_id: str,
        form_vector: torch.Tensor,
        days_ago_tensor: torch.Tensor
    ):
        """Update player statistics for monitoring."""
        if player_id not in self.feature_stats:
            self.feature_stats[player_id] = {
                'form_vector_history': [],
                'days_ago_history': [],
                'update_count': 0
            }
        
        stats = self.feature_stats[player_id]
        stats['form_vector_history'].append(form_vector.detach().numpy())
        stats['days_ago_history'].append(days_ago_tensor.detach().numpy())
        stats['update_count'] += 1
        
        # Keep only recent history
        max_history = 100
        if len(stats['form_vector_history']) > max_history:
            stats['form_vector_history'] = stats['form_vector_history'][-max_history:]
            stats['days_ago_history'] = stats['days_ago_history'][-max_history:]
    
    def get_temporal_statistics(self) -> Dict[str, Any]:
        """Get temporal decay statistics."""
        stats = {}
        
        # Learnable decay statistics
        decay_stats = self.temporal_decay.get_statistics()
        stats.update(decay_stats)
        
        # Feature statistics
        stats['num_players_with_features'] = len(self.feature_stats)
        
        if self.feature_stats:
            avg_form_magnitude = np.mean([
                np.linalg.norm(np.array(player_stats['form_vector_history'][-1]))
                for player_stats in self.feature_stats.values()
                if player_stats['form_vector_history']
            ])
            stats['avg_form_vector_magnitude'] = avg_form_magnitude
        
        return stats
    
    def save_temporal_state(self, filepath: str):
        """Save temporal decay state."""
        state = {
            'temporal_decay_state': self.temporal_decay.state_dict(),
            'feature_names': self.feature_names,
            'temporal_config': self.temporal_config
        }
        
        if self.adaptive_encoder is not None:
            state['adaptive_encoder_state'] = self.adaptive_encoder.state_dict()
        
        torch.save(state, filepath)
        logger.info(f"Saved temporal state to {filepath}")
    
    def load_temporal_state(self, filepath: str):
        """Load temporal decay state."""
        state = torch.load(filepath, map_location='cpu')
        
        self.temporal_decay.load_state_dict(state['temporal_decay_state'])
        
        if 'adaptive_encoder_state' in state and self.adaptive_encoder is not None:
            self.adaptive_encoder.load_state_dict(state['adaptive_encoder_state'])
        
        logger.info(f"Loaded temporal state from {filepath}")


def create_temporal_graph_builder(
    feature_names: List[str],
    config: Optional[Dict[str, Any]] = None
) -> TemporalGraphBuilder:
    """
    Factory function to create temporal graph builder.
    
    Args:
        feature_names: List of feature names
        config: Optional configuration dictionary
    
    Returns:
        Configured TemporalGraphBuilder
    """
    default_config = {
        'temporal_config': {
            'initial_half_life': 30.0,
            'min_half_life': 1.0,
            'max_half_life': 365.0,
            'learnable': True
        },
        'use_adaptive_encoder': False,
        'encoder_config': {
            'embed_dim': 64,
            'max_days': 365,
            'use_positional_encoding': True
        }
    }
    
    if config:
        # Deep merge configuration
        for key, value in config.items():
            if key in default_config and isinstance(value, dict):
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    return TemporalGraphBuilder(
        feature_names=feature_names,
        temporal_config=default_config['temporal_config'],
        use_adaptive_encoder=default_config['use_adaptive_encoder'],
        encoder_config=default_config['encoder_config']
    )