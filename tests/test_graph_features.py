# Purpose: Tests for graph features module that generates recent form vectors
# Author: Assistant, Last Modified: 2024

import pytest
import numpy as np
import networkx as nx
from unittest.mock import patch

from crickformers.gnn.graph_features import (
    generate_form_features,
    attach_form_features_to_graph,
    get_form_feature_names,
    FormFeatureConfig,
    _extract_match_data,
    _compute_batter_match_stats,
    _compute_bowler_match_stats,
    _compute_rolling_stats
)

@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return FormFeatureConfig(
        lookback_matches=5,
        min_balls_per_match=6,
        default_values={
            'avg_runs': 20.0,
            'strike_rate': 120.0,
            'dismissal_rate': 0.2,
            'dot_ball_pct': 0.4,
            'boundary_pct': 0.15,
            'avg_runs_conceded': 25.0,
            'economy_rate': 7.5,
            'wicket_rate': 0.05,
            'dot_ball_pct_bowler': 0.3,
            'bowling_average': 25.0
        }
    )

@pytest.fixture
def complete_match_data():
    """Create mock match data for a player with exactly 5 matches."""
    match_data = []
    
    # Create data for batter "kohli_v" across 5 matches
    matches = ["match_1", "match_2", "match_3", "match_4", "match_5"]
    
    for match_idx, match_id in enumerate(matches):
        # Create 12 balls per match (2 overs)
        for ball in range(12):
            # Simulate different scoring patterns per match
            if match_idx == 0:  # Good match: 24 runs in 12 balls
                runs = 2 if ball % 2 == 0 else 0
                is_wicket = 1 if ball == 11 else 0
            elif match_idx == 1:  # Excellent match: 36 runs in 12 balls
                runs = 4 if ball % 3 == 0 else 2 if ball % 3 == 1 else 0
                is_wicket = 0
            elif match_idx == 2:  # Poor match: 6 runs in 12 balls, dismissed
                runs = 1 if ball % 6 == 0 else 0
                is_wicket = 1 if ball == 5 else 0  # Dismissed early
            elif match_idx == 3:  # Average match: 18 runs in 12 balls
                runs = 3 if ball % 4 == 0 else 0
                is_wicket = 0
            else:  # match_idx == 4, Recent excellent match: 42 runs in 12 balls
                runs = 6 if ball % 4 == 0 else 1 if ball % 4 == 1 else 0
                is_wicket = 0
            
            match_data.append({
                'match_id': match_id,
                'match_date': f'2024-0{match_idx + 1}-01',
                'batter_id': 'kohli_v',
                'bowler_id': f'bowler_{ball % 3}',
                'runs_scored': runs,
                'is_wicket': is_wicket,
                'runs': runs,  # Alternative field name
                'dot': 1 if runs == 0 else 0,
                'four': 1 if runs == 4 else 0,
                'six': 1 if runs == 6 else 0,
                'dismissal_type': 'caught' if is_wicket else ''
            })
    
    # Add bowling data for "ashwin_r" across 3 matches
    bowling_matches = ["match_1", "match_2", "match_3"]
    
    for match_idx, match_id in enumerate(bowling_matches):
        # Create 18 balls per match (3 overs)
        for ball in range(18):
            # Simulate different bowling performances
            if match_idx == 0:  # Good bowling: 18 runs, 2 wickets
                runs = 1 if ball % 3 == 0 else 0
                is_wicket = 1 if ball in [5, 11] else 0
            elif match_idx == 1:  # Average bowling: 24 runs, 1 wicket
                runs = 2 if ball % 4 == 0 else 1 if ball % 4 == 1 else 0
                is_wicket = 1 if ball == 8 else 0
            else:  # Poor bowling: 36 runs, 0 wickets
                runs = 2 if ball % 2 == 0 else 0
                is_wicket = 0
            
            match_data.append({
                'match_id': match_id,
                'match_date': f'2024-0{match_idx + 1}-01',
                'batter_id': f'batter_{ball % 4}',
                'bowler_id': 'ashwin_r',
                'runs_scored': runs,
                'is_wicket': is_wicket,
                'runs': runs,
                'dot': 1 if runs == 0 else 0,
                'four': 1 if runs == 4 else 0,
                'six': 1 if runs == 6 else 0,
                'dismissal_type': 'bowled' if is_wicket else ''
            })
    
    return match_data

@pytest.fixture
def incomplete_match_data():
    """Create mock match data for a player with fewer than 5 matches."""
    match_data = []
    
    # Create data for batter "root_j" across only 2 matches
    matches = ["match_1", "match_2"]
    
    for match_idx, match_id in enumerate(matches):
        # Create 10 balls per match
        for ball in range(10):
            runs = 2 if ball % 2 == 0 else 0
            is_wicket = 1 if ball == 9 else 0
            
            match_data.append({
                'match_id': match_id,
                'match_date': f'2024-0{match_idx + 1}-01',
                'batter_id': 'root_j',
                'bowler_id': f'bowler_{ball % 2}',
                'runs_scored': runs,
                'is_wicket': is_wicket,
                'runs': runs,
                'dot': 1 if runs == 0 else 0,
                'four': 1 if runs == 4 else 0,
                'six': 1 if runs == 6 else 0,
                'dismissal_type': 'caught' if is_wicket else ''
            })
    
    return match_data

@pytest.fixture
def cricket_graph():
    """Create a sample cricket graph for testing feature attachment."""
    G = nx.DiGraph()
    
    # Add nodes with different types
    G.add_nodes_from([
        ('kohli_v', {'type': 'batter'}),
        ('root_j', {'type': 'batter'}),
        ('ashwin_r', {'type': 'bowler'}),
        ('jadeja_r', {'type': 'all_rounder'}),
        ('India', {'type': 'team'}),
        ('England', {'type': 'team'}),
    ])
    
    # Add some edges
    G.add_edges_from([
        ('kohli_v', 'ashwin_r', {'edge_type': 'faced'}),
        ('root_j', 'ashwin_r', {'edge_type': 'faced'}),
        ('kohli_v', 'India', {'edge_type': 'plays_for'}),
        ('root_j', 'England', {'edge_type': 'plays_for'}),
    ])
    
    return G

def test_form_feature_config():
    """Test the FormFeatureConfig class initialization."""
    config = FormFeatureConfig()
    
    # Test default values
    assert config.lookback_matches == 5
    assert config.min_balls_per_match == 6
    assert 'avg_runs' in config.default_values
    assert config.default_values['avg_runs'] == 20.0
    
    # Test custom values
    custom_config = FormFeatureConfig(
        lookback_matches=3,
        min_balls_per_match=4,
        default_values={'avg_runs': 15.0}
    )
    
    assert custom_config.lookback_matches == 3
    assert custom_config.min_balls_per_match == 4
    assert custom_config.default_values['avg_runs'] == 15.0

def test_extract_match_data():
    """Test the _extract_match_data function."""
    raw_data = [
        {
            'batter_id': 'player_1',
            'bowler_id': 'bowler_1',
            'runs': 4,
            'match_id': 'match_1',
            'dismissal_type': 'caught'
        },
        {
            'batter_id': 'player_2',
            'bowler_id': 'bowler_2',
            'runs_scored': 2,
            'match_id': 'match_2',
            'is_wicket': 0
        }
    ]
    
    df = _extract_match_data(raw_data)
    
    # Check that column mapping worked
    assert 'runs_scored' in df.columns
    assert 'is_wicket' in df.columns
    assert 'batter_id' in df.columns
    assert 'bowler_id' in df.columns
    assert 'match_id' in df.columns
    
    # Check derived columns
    assert 'is_dot' in df.columns
    assert 'is_four' in df.columns
    assert 'is_six' in df.columns
    assert 'is_boundary' in df.columns
    
    # Check values
    assert df.iloc[0]['runs_scored'] == 4
    assert df.iloc[0]['is_four'] == 1
    assert df.iloc[0]['is_boundary'] == 1
    assert df.iloc[0]['is_wicket'] == 1  # Derived from dismissal_type

def test_compute_batter_match_stats():
    """Test the _compute_batter_match_stats function."""
    import pandas as pd
    
    # Create sample match data for a batter
    match_data = pd.DataFrame([
        {'runs_scored': 4, 'is_wicket': 0},
        {'runs_scored': 0, 'is_wicket': 0},
        {'runs_scored': 2, 'is_wicket': 0},
        {'runs_scored': 6, 'is_wicket': 0},
        {'runs_scored': 1, 'is_wicket': 1},  # Dismissed
        {'runs_scored': 0, 'is_wicket': 0},
    ])
    
    # Add derived columns
    match_data['is_dot'] = (match_data['runs_scored'] == 0).astype(int)
    match_data['is_boundary'] = ((match_data['runs_scored'] == 4) | 
                                (match_data['runs_scored'] == 6)).astype(int)
    
    stats = _compute_batter_match_stats(match_data)
    
    # Check basic stats
    assert stats['balls_faced'] == 6
    assert stats['runs_scored'] == 13  # 4 + 0 + 2 + 6 + 1 + 0
    assert stats['is_dismissed'] == 1
    
    # Check calculated rates
    expected_strike_rate = (13 / 6) * 100  # ≈ 216.67
    assert abs(stats['strike_rate'] - expected_strike_rate) < 0.01
    
    expected_dot_pct = 2 / 6  # 2 dots out of 6 balls
    assert abs(stats['dot_ball_pct'] - expected_dot_pct) < 0.01
    
    expected_boundary_pct = 2 / 6  # 2 boundaries out of 6 balls
    assert abs(stats['boundary_pct'] - expected_boundary_pct) < 0.01

def test_compute_bowler_match_stats():
    """Test the _compute_bowler_match_stats function."""
    import pandas as pd
    
    # Create sample match data for a bowler
    match_data = pd.DataFrame([
        {'runs_scored': 1, 'is_wicket': 0},
        {'runs_scored': 0, 'is_wicket': 0},
        {'runs_scored': 4, 'is_wicket': 0},
        {'runs_scored': 0, 'is_wicket': 1},  # Wicket
        {'runs_scored': 2, 'is_wicket': 0},
        {'runs_scored': 0, 'is_wicket': 0},
    ])
    
    # Add derived columns
    match_data['is_dot'] = (match_data['runs_scored'] == 0).astype(int)
    
    stats = _compute_bowler_match_stats(match_data)
    
    # Check basic stats
    assert stats['balls_bowled'] == 6
    assert stats['runs_conceded'] == 7  # 1 + 0 + 4 + 0 + 2 + 0
    assert stats['wickets'] == 1
    
    # Check calculated rates
    expected_economy = (7 / 6) * 6  # ≈ 7.0
    assert abs(stats['economy_rate'] - expected_economy) < 0.01
    
    expected_wicket_rate = 1 / 6  # 1 wicket in 6 balls
    assert abs(stats['wicket_rate'] - expected_wicket_rate) < 0.01
    
    expected_dot_pct = 3 / 6  # 3 dots out of 6 balls
    assert abs(stats['dot_ball_pct'] - expected_dot_pct) < 0.01
    
    expected_bowling_avg = 7 / 1  # 7 runs per wicket
    assert abs(stats['bowling_average'] - expected_bowling_avg) < 0.01

def test_compute_rolling_stats(sample_config):
    """Test the _compute_rolling_stats function."""
    # Create mock match stats
    match_stats = [
        {'runs_scored': 20, 'balls_faced': 15, 'strike_rate': 133.33},
        {'runs_scored': 30, 'balls_faced': 18, 'strike_rate': 166.67},
        {'runs_scored': 10, 'balls_faced': 12, 'strike_rate': 83.33},
        {'runs_scored': 25, 'balls_faced': 20, 'strike_rate': 125.0},
        {'runs_scored': 35, 'balls_faced': 24, 'strike_rate': 145.83}
    ]
    
    stat_keys = ['runs_scored', 'strike_rate']
    
    rolling_stats = _compute_rolling_stats(match_stats, stat_keys, sample_config)
    
    # Check that we get the right number of stats
    assert len(rolling_stats) == 2
    
    # Check runs_scored (simple average)
    expected_runs_avg = (20 + 30 + 10 + 25 + 35) / 5
    assert abs(rolling_stats[0] - expected_runs_avg) < 0.01
    
    # Check strike_rate (weighted by balls_faced)
    total_runs = sum(match['runs_scored'] for match in match_stats)
    total_balls = sum(match['balls_faced'] for match in match_stats)
    expected_weighted_sr = (total_runs / total_balls) * 100
    assert abs(rolling_stats[1] - expected_weighted_sr) < 0.01

def test_generate_form_features_complete_data(complete_match_data, sample_config):
    """Test generate_form_features with complete match data."""
    form_features = generate_form_features(complete_match_data, sample_config)
    
    # Check that features were generated for both players
    assert 'kohli_v' in form_features
    assert 'ashwin_r' in form_features
    
    # Check batter features for kohli_v
    kohli_features = form_features['kohli_v']
    assert 'batter_features' in kohli_features
    assert len(kohli_features['batter_features']) == 5
    
    # Check that features are reasonable
    batter_features = kohli_features['batter_features']
    assert batter_features[0] > 0  # avg_runs should be positive
    assert batter_features[1] > 0  # strike_rate should be positive
    assert 0 <= batter_features[2] <= 1  # dismissal_rate should be between 0 and 1
    assert 0 <= batter_features[3] <= 1  # dot_ball_pct should be between 0 and 1
    assert 0 <= batter_features[4] <= 1  # boundary_pct should be between 0 and 1
    
    # Check bowler features for ashwin_r
    ashwin_features = form_features['ashwin_r']
    assert 'bowler_features' in ashwin_features
    assert len(ashwin_features['bowler_features']) == 5
    
    # Check that features are reasonable
    bowler_features = ashwin_features['bowler_features']
    assert bowler_features[0] > 0  # avg_runs_conceded should be positive
    assert bowler_features[1] > 0  # economy_rate should be positive
    assert bowler_features[2] >= 0  # wicket_rate should be non-negative
    assert 0 <= bowler_features[3] <= 1  # dot_ball_pct should be between 0 and 1
    assert bowler_features[4] > 0  # bowling_average should be positive

def test_generate_form_features_incomplete_data(incomplete_match_data, sample_config):
    """Test generate_form_features with incomplete match data (padding test)."""
    form_features = generate_form_features(incomplete_match_data, sample_config)
    
    # Check that features were generated for the player
    assert 'root_j' in form_features
    
    # Check batter features
    root_features = form_features['root_j']
    assert 'batter_features' in root_features
    assert len(root_features['batter_features']) == 5
    
    # Features should be computed from available data (2 matches instead of 5)
    batter_features = root_features['batter_features']
    assert batter_features[0] > 0  # avg_runs should be positive
    assert batter_features[1] > 0  # strike_rate should be positive
    
    # Since we only have 2 matches, the rolling stats should be based on those
    # The player scored 10 runs in each match (2 runs on 5 balls, 0 on 5 balls)
    expected_avg_runs = 10.0  # 10 runs per match average
    assert abs(batter_features[0] - expected_avg_runs) < 0.01

def test_generate_form_features_empty_data(sample_config):
    """Test generate_form_features with empty data."""
    form_features = generate_form_features([], sample_config)
    
    # Should return empty dictionary for empty input
    assert form_features == {}

def test_generate_form_features_insufficient_balls(sample_config):
    """Test generate_form_features with insufficient balls per match."""
    # Create data with only 3 balls per match (less than min_balls_per_match=6)
    match_data = [
        {'match_id': 'match_1', 'batter_id': 'player_1', 'bowler_id': 'bowler_1', 
         'runs_scored': 2, 'is_wicket': 0},
        {'match_id': 'match_1', 'batter_id': 'player_1', 'bowler_id': 'bowler_1', 
         'runs_scored': 0, 'is_wicket': 0},
        {'match_id': 'match_1', 'batter_id': 'player_1', 'bowler_id': 'bowler_1', 
         'runs_scored': 1, 'is_wicket': 1},
    ]
    
    form_features = generate_form_features(match_data, sample_config)
    
    # Player should still have features (defaults)
    assert 'player_1' in form_features
    
    # Should use default values since match doesn't meet minimum balls requirement
    player_features = form_features['player_1']
    assert player_features['batter_features'][0] == sample_config.default_values['avg_runs']
    assert player_features['batter_features'][1] == sample_config.default_values['strike_rate']

def test_attach_form_features_to_graph(cricket_graph, complete_match_data, sample_config):
    """Test attaching form features to a NetworkX graph."""
    # Generate form features
    form_features = generate_form_features(complete_match_data, sample_config)
    
    # Attach to graph
    attach_form_features_to_graph(cricket_graph, form_features)
    
    # Check that features were attached to the correct nodes
    kohli_node = cricket_graph.nodes['kohli_v']
    assert 'form_features' in kohli_node
    assert len(kohli_node['form_features']) == 5  # Batter features
    
    ashwin_node = cricket_graph.nodes['ashwin_r']
    assert 'form_features' in ashwin_node
    assert len(ashwin_node['form_features']) == 5  # Bowler features
    
    # Check that nodes without features don't have them
    india_node = cricket_graph.nodes['India']
    assert 'form_features' not in india_node

def test_attach_form_features_all_rounder(cricket_graph, complete_match_data, sample_config):
    """Test attaching form features to an all-rounder node."""
    # Generate form features
    form_features = generate_form_features(complete_match_data, sample_config)
    
    # Add all-rounder features manually for testing
    form_features['jadeja_r'] = {
        'batter_features': [25.0, 130.0, 0.15, 0.35, 0.20],
        'bowler_features': [20.0, 6.5, 0.08, 0.35, 22.0]
    }
    
    # Attach to graph
    attach_form_features_to_graph(cricket_graph, form_features)
    
    # Check that all-rounder gets combined features
    jadeja_node = cricket_graph.nodes['jadeja_r']
    assert 'form_features' in jadeja_node
    assert len(jadeja_node['form_features']) == 10  # 5 batting + 5 bowling

def test_get_form_feature_names():
    """Test getting form feature names."""
    batter_names, bowler_names = get_form_feature_names()
    
    # Check batter feature names
    assert len(batter_names) == 5
    assert 'avg_runs_last_5' in batter_names
    assert 'strike_rate_last_5' in batter_names
    assert 'dismissal_rate_last_5' in batter_names
    assert 'dot_ball_pct_last_5' in batter_names
    assert 'boundary_pct_last_5' in batter_names
    
    # Check bowler feature names
    assert len(bowler_names) == 5
    assert 'avg_runs_conceded_last_5' in bowler_names
    assert 'economy_rate_last_5' in bowler_names
    assert 'wicket_rate_last_5' in bowler_names
    assert 'dot_ball_pct_last_5' in bowler_names
    assert 'bowling_average_last_5' in bowler_names

def test_feature_vector_consistency():
    """Test that feature vectors are consistent across different scenarios."""
    # Test with different configurations
    configs = [
        FormFeatureConfig(lookback_matches=3),
        FormFeatureConfig(lookback_matches=5),
        FormFeatureConfig(lookback_matches=7)
    ]
    
    match_data = [
        {'match_id': 'match_1', 'batter_id': 'player_1', 'bowler_id': 'bowler_1', 
         'runs_scored': 20, 'is_wicket': 0},
        {'match_id': 'match_1', 'batter_id': 'player_1', 'bowler_id': 'bowler_1', 
         'runs_scored': 0, 'is_wicket': 0},
        {'match_id': 'match_1', 'batter_id': 'player_1', 'bowler_id': 'bowler_1', 
         'runs_scored': 4, 'is_wicket': 0},
        {'match_id': 'match_1', 'batter_id': 'player_1', 'bowler_id': 'bowler_1', 
         'runs_scored': 6, 'is_wicket': 0},
        {'match_id': 'match_1', 'batter_id': 'player_1', 'bowler_id': 'bowler_1', 
         'runs_scored': 2, 'is_wicket': 0},
        {'match_id': 'match_1', 'batter_id': 'player_1', 'bowler_id': 'bowler_1', 
         'runs_scored': 1, 'is_wicket': 1},
    ]
    
    for config in configs:
        form_features = generate_form_features(match_data, config)
        
        # All should produce the same structure
        assert 'player_1' in form_features
        assert len(form_features['player_1']['batter_features']) == 5
        assert len(form_features['player_1']['bowler_features']) == 5
        
        # Features should be finite and reasonable
        batter_features = form_features['player_1']['batter_features']
        for feature in batter_features:
            assert np.isfinite(feature)
            assert feature >= 0  # All features should be non-negative

def test_edge_cases_missing_fields():
    """Test handling of missing fields in match data."""
    # Data with missing optional fields
    match_data = [
        {'match_id': 'match_1', 'batter_id': 'player_1', 'bowler_id': 'bowler_1'},
        {'match_id': 'match_1', 'batter_id': 'player_1', 'bowler_id': 'bowler_1', 'runs_scored': 4},
        {'match_id': 'match_1', 'batter_id': 'player_1', 'bowler_id': 'bowler_1', 'runs': 2},
    ]
    
    # Should not raise errors
    form_features = generate_form_features(match_data)
    
    # Should still generate features using defaults
    assert 'player_1' in form_features
    assert len(form_features['player_1']['batter_features']) == 5
    assert len(form_features['player_1']['bowler_features']) == 5 