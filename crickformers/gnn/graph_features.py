# Purpose: Generates recent form vectors for cricket players to enhance GNN node features
# Author: Assistant, Last Modified: 2024

"""
This module computes rolling statistics for cricket players based on their recent match
performance and generates feature vectors that can be attached to NetworkX nodes
before GNN training.

The form features capture recent performance trends including:
- Batting: runs, strike rate, dismissals, dot balls, boundaries
- Bowling: runs conceded, economy rate, wickets, dot balls, bowling average
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FormFeatureConfig:
    """Configuration for form feature computation."""
    
    def __init__(self, 
                 lookback_matches: int = 5,
                 min_balls_per_match: int = 6,  # Minimum balls for valid match
                 default_values: Optional[Dict[str, float]] = None):
        """
        Initialize form feature configuration.
        
        Args:
            lookback_matches: Number of recent matches to consider
            min_balls_per_match: Minimum balls required for a valid match
            default_values: Default values for padding missing data
        """
        self.lookback_matches = lookback_matches
        self.min_balls_per_match = min_balls_per_match
        self.default_values = default_values or {
            # Batting defaults
            'avg_runs': 20.0,
            'strike_rate': 120.0,
            'dismissal_rate': 0.2,
            'dot_ball_pct': 0.4,
            'boundary_pct': 0.15,
            # Bowling defaults
            'avg_runs_conceded': 25.0,
            'economy_rate': 7.5,
            'wicket_rate': 0.05,
            'dot_ball_pct_bowler': 0.3,
            'bowling_average': 25.0
        }

def _extract_match_data(ball_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert ball-by-ball data to a pandas DataFrame for easier processing.
    
    Args:
        ball_data: List of ball-by-ball records
        
    Returns:
        DataFrame with standardized columns
    """
    # Convert to DataFrame
    df = pd.DataFrame(ball_data)
    
    if df.empty:
        return df
    
    # Handle runs_scored column mapping
    if 'runs_scored' not in df.columns:
        if 'runs' in df.columns:
            df['runs_scored'] = df['runs']
        else:
            df['runs_scored'] = 0
    else:
        # If runs_scored exists but has NaN values, try to fill from 'runs'
        if 'runs' in df.columns:
            df['runs_scored'] = df['runs_scored'].fillna(df['runs'])
        df['runs_scored'] = df['runs_scored'].fillna(0)
    
    # Handle other column mappings
    column_mapping = {
        'dot': 'is_dot',
        'four': 'is_four',
        'six': 'is_six',
        'bowler_runs_ball': 'runs_conceded',
        'ball': 'ball_in_over'
    }
    
    # Apply column mapping where columns exist and target doesn't exist
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Create derived columns
    if 'runs_scored' in df.columns:
        if 'is_dot' not in df.columns:
            df['is_dot'] = (df['runs_scored'] == 0).astype(int)
        if 'is_four' not in df.columns:
            df['is_four'] = (df['runs_scored'] == 4).astype(int)
        if 'is_six' not in df.columns:
            df['is_six'] = (df['runs_scored'] == 6).astype(int)
        df['is_boundary'] = ((df['runs_scored'] == 4) | (df['runs_scored'] == 6)).astype(int)
    
    # Handle dismissals
    if 'is_wicket' not in df.columns:
        if 'dismissal_type' in df.columns:
            df['is_wicket'] = (df['dismissal_type'] != '').astype(int)
        else:
            df['is_wicket'] = 0
    else:
        # If is_wicket exists but has NaN values, try to fill from dismissal_type
        if 'dismissal_type' in df.columns:
            dismissal_derived = (df['dismissal_type'] != '').astype(int)
            df['is_wicket'] = df['is_wicket'].fillna(dismissal_derived)
        df['is_wicket'] = df['is_wicket'].fillna(0)
    
    # Ensure required columns exist
    required_columns = ['match_id', 'batter_id', 'bowler_id', 'runs_scored', 'is_wicket']
    for col in required_columns:
        if col not in df.columns:
            if col == 'runs_scored':
                df[col] = 0
            elif col == 'is_wicket':
                df[col] = 0
            else:
                df[col] = ''
    
    return df

def _compute_batter_match_stats(match_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute batting statistics for a single match.
    
    Args:
        match_data: DataFrame containing balls faced by batter in one match
        
    Returns:
        Dictionary of batting statistics
    """
    if len(match_data) == 0:
        return {}
    
    balls_faced = len(match_data)
    runs_scored = match_data['runs_scored'].sum()
    dots = match_data.get('is_dot', match_data['runs_scored'] == 0).sum()
    boundaries = match_data.get('is_boundary', 
                              (match_data['runs_scored'] == 4) | 
                              (match_data['runs_scored'] == 6)).sum()
    is_dismissed = match_data['is_wicket'].sum() > 0
    
    # Calculate rates
    strike_rate = (runs_scored / balls_faced) * 100 if balls_faced > 0 else 0
    dot_ball_pct = (dots / balls_faced) if balls_faced > 0 else 0
    boundary_pct = (boundaries / balls_faced) if balls_faced > 0 else 0
    
    return {
        'balls_faced': balls_faced,
        'runs_scored': runs_scored,
        'strike_rate': strike_rate,
        'dot_ball_pct': dot_ball_pct,
        'boundary_pct': boundary_pct,
        'is_dismissed': int(is_dismissed)
    }

def _compute_bowler_match_stats(match_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute bowling statistics for a single match.
    
    Args:
        match_data: DataFrame containing balls bowled by bowler in one match
        
    Returns:
        Dictionary of bowling statistics
    """
    if len(match_data) == 0:
        return {}
    
    balls_bowled = len(match_data)
    runs_conceded = match_data['runs_scored'].sum()
    wickets = match_data['is_wicket'].sum()
    dots = match_data.get('is_dot', match_data['runs_scored'] == 0).sum()
    
    # Calculate rates
    economy_rate = (runs_conceded / balls_bowled) * 6 if balls_bowled > 0 else 0
    wicket_rate = (wickets / balls_bowled) if balls_bowled > 0 else 0
    dot_ball_pct = (dots / balls_bowled) if balls_bowled > 0 else 0
    bowling_average = (runs_conceded / wickets) if wickets > 0 else 100.0
    
    return {
        'balls_bowled': balls_bowled,
        'runs_conceded': runs_conceded,
        'wickets': wickets,
        'economy_rate': economy_rate,
        'wicket_rate': wicket_rate,
        'dot_ball_pct': dot_ball_pct,
        'bowling_average': bowling_average
    }

def _compute_rolling_stats(match_stats: List[Dict[str, Any]], 
                          stat_keys: List[str],
                          config: FormFeatureConfig) -> List[float]:
    """
    Compute rolling averages for the specified statistics.
    
    Args:
        match_stats: List of match statistics dictionaries
        stat_keys: Keys to compute rolling averages for
        config: Configuration object
        
    Returns:
        List of rolling averages
    """
    if not match_stats:
        # Return default values if no match data
        return [config.default_values.get(key, 0.0) for key in stat_keys]
    
    # Take only the most recent matches
    recent_matches = match_stats[-config.lookback_matches:]
    
    rolling_stats = []
    for key in stat_keys:
        values = [match.get(key, 0) for match in recent_matches]
        
        if values:
            # For rates, use weighted average by balls faced/bowled
            if key in ['strike_rate', 'dot_ball_pct', 'boundary_pct']:
                # Weight by balls faced for batting stats
                weights = [match.get('balls_faced', 1) for match in recent_matches]
                if sum(weights) > 0:
                    avg_value = sum(v * w for v, w in zip(values, weights)) / sum(weights)
                else:
                    avg_value = np.mean(values)
            elif key in ['economy_rate', 'wicket_rate', 'dot_ball_pct', 'bowling_average']:
                # Weight by balls bowled for bowling stats
                weights = [match.get('balls_bowled', 1) for match in recent_matches]
                if sum(weights) > 0:
                    avg_value = sum(v * w for v, w in zip(values, weights)) / sum(weights)
                else:
                    avg_value = np.mean(values)
            else:
                # Simple average for counting stats
                avg_value = np.mean(values)
            
            rolling_stats.append(avg_value)
        else:
            rolling_stats.append(config.default_values.get(key, 0.0))
    
    return rolling_stats

def generate_form_features(match_data: List[Dict[str, Any]], 
                         config: Optional[FormFeatureConfig] = None) -> Dict[str, Dict[str, List[float]]]:
    """
    Generate recent form features for all players in the match data.
    
    Args:
        match_data: List of ball-by-ball records
        config: Configuration for form feature computation
        
    Returns:
        Dictionary mapping player_id to feature type to feature vector:
        {
            'player_id': {
                'batter_features': [avg_runs, strike_rate, dismissal_rate, dot_pct, boundary_pct],
                'bowler_features': [avg_runs_conceded, economy_rate, wicket_rate, dot_pct, bowling_avg]
            }
        }
    """
    if config is None:
        config = FormFeatureConfig()
    
    # Convert to DataFrame for easier processing
    df = _extract_match_data(match_data)
    
    if len(df) == 0:
        logger.warning("No match data provided for form feature generation")
        return {}
    
    # Group by player and match to compute match-level statistics
    player_features = defaultdict(lambda: {'batter_features': [], 'bowler_features': []})
    
    # Process batting statistics
    logger.info("Computing batting form features...")
    batter_groups = df.groupby(['batter_id', 'match_id'])
    
    batter_match_stats = defaultdict(list)
    for (batter_id, match_id), group in batter_groups:
        if len(group) >= config.min_balls_per_match:
            match_stats = _compute_batter_match_stats(group)
            if match_stats:
                batter_match_stats[batter_id].append(match_stats)
    
    # Generate batting form features
    for batter_id, match_stats in batter_match_stats.items():
        stat_keys = ['runs_scored', 'strike_rate', 'is_dismissed', 'dot_ball_pct', 'boundary_pct']
        
        # Sort by match chronologically if match_date is available
        # For now, assume they're already in order
        
        rolling_stats = _compute_rolling_stats(match_stats, stat_keys, config)
        
        # Convert dismissal count to dismissal rate
        if len(rolling_stats) >= 3:
            dismissal_rate = rolling_stats[2] / max(len(match_stats), 1)
            rolling_stats[2] = dismissal_rate
        
        player_features[batter_id]['batter_features'] = rolling_stats
    
    # Process bowling statistics
    logger.info("Computing bowling form features...")
    bowler_groups = df.groupby(['bowler_id', 'match_id'])
    
    bowler_match_stats = defaultdict(list)
    for (bowler_id, match_id), group in bowler_groups:
        if len(group) >= config.min_balls_per_match:
            match_stats = _compute_bowler_match_stats(group)
            if match_stats:
                bowler_match_stats[bowler_id].append(match_stats)
    
    # Generate bowling form features
    for bowler_id, match_stats in bowler_match_stats.items():
        stat_keys = ['runs_conceded', 'economy_rate', 'wicket_rate', 'dot_ball_pct', 'bowling_average']
        
        rolling_stats = _compute_rolling_stats(match_stats, stat_keys, config)
        
        player_features[bowler_id]['bowler_features'] = rolling_stats
    
    # Fill in missing features with defaults
    all_player_ids = set(df['batter_id'].unique()) | set(df['bowler_id'].unique())
    
    for player_id in all_player_ids:
        if not player_features[player_id]['batter_features']:
            default_batter_features = [
                config.default_values['avg_runs'],
                config.default_values['strike_rate'],
                config.default_values['dismissal_rate'],
                config.default_values['dot_ball_pct'],
                config.default_values['boundary_pct']
            ]
            player_features[player_id]['batter_features'] = default_batter_features
        
        if not player_features[player_id]['bowler_features']:
            default_bowler_features = [
                config.default_values['avg_runs_conceded'],
                config.default_values['economy_rate'],
                config.default_values['wicket_rate'],
                config.default_values['dot_ball_pct_bowler'],
                config.default_values['bowling_average']
            ]
            player_features[player_id]['bowler_features'] = default_bowler_features
    
    logger.info(f"Generated form features for {len(player_features)} players")
    
    return dict(player_features)

def attach_form_features_to_graph(graph, form_features: Dict[str, Dict[str, List[float]]]):
    """
    Attach form features to NetworkX graph nodes.
    
    Args:
        graph: NetworkX graph with cricket player nodes
        form_features: Dictionary of form features from generate_form_features()
        
    Returns:
        None (modifies graph in place)
    """
    logger.info("Attaching form features to graph nodes...")
    
    nodes_updated = 0
    for node_id in graph.nodes():
        node_data = graph.nodes[node_id]
        node_type = node_data.get('type', '')
        
        if node_id in form_features:
            features = form_features[node_id]
            
            if node_type == 'batter' and 'batter_features' in features:
                node_data['form_features'] = features['batter_features']
                nodes_updated += 1
            elif node_type == 'bowler' and 'bowler_features' in features:
                node_data['form_features'] = features['bowler_features']
                nodes_updated += 1
            elif node_type in ['player', 'all_rounder']:
                # For generic players, combine both batting and bowling features
                combined_features = features.get('batter_features', [0.0] * 5) + \
                                  features.get('bowler_features', [0.0] * 5)
                node_data['form_features'] = combined_features
                nodes_updated += 1
    
    logger.info(f"Updated {nodes_updated} nodes with form features")

def get_form_feature_names() -> Tuple[List[str], List[str]]:
    """
    Get the names of form features for documentation and debugging.
    
    Returns:
        Tuple of (batter_feature_names, bowler_feature_names)
    """
    batter_features = [
        'avg_runs_last_5',
        'strike_rate_last_5', 
        'dismissal_rate_last_5',
        'dot_ball_pct_last_5',
        'boundary_pct_last_5'
    ]
    
    bowler_features = [
        'avg_runs_conceded_last_5',
        'economy_rate_last_5',
        'wicket_rate_last_5',
        'dot_ball_pct_last_5',
        'bowling_average_last_5'
    ]
    
    return batter_features, bowler_features 