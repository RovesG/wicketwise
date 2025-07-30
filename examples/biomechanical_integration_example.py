#!/usr/bin/env python3
# Purpose: Example of integrating biomechanical features with existing graph builder
# Author: Shamus Rae, Last Modified: 2024-01-15

"""
This example shows how to integrate biomechanical features into the existing
cricket knowledge graph building pipeline.
"""

import pandas as pd
import numpy as np
from crickformers.gnn.graph_builder import build_cricket_graph
from crickformers.gnn.biomechanical_features import (
    process_match_biomechanical_data,
    add_biomechanical_features_to_graph,
    create_sample_biomechanical_data,
    BiomechanicalConfig
)

def integrate_biomechanical_features_example():
    """Example of integrating biomechanical features into the graph building pipeline."""
    
    print("🏏 Biomechanical Features Integration Example")
    print("=" * 50)
    
    # Step 1: Create sample match data (same format as existing pipeline)
    match_data = pd.DataFrame([
        {
            'match_id': 'integration_test',
            'innings': 1,
            'over': 1,
            'ball': 1,
            'batter': 'kohli',
            'bowler': 'bumrah',
            'runs_scored': 4,
            'wicket_type': None,
            'venue': 'wankhede',
            'date': '2024-01-15',
            'team_batting': 'india',
            'team_bowling': 'australia'
        },
        {
            'match_id': 'integration_test',
            'innings': 1,
            'over': 1,
            'ball': 2,
            'batter': 'kohli',
            'bowler': 'bumrah',
            'runs_scored': 1,
            'wicket_type': None,
            'venue': 'wankhede',
            'date': '2024-01-15',
            'team_batting': 'india',
            'team_bowling': 'australia'
        },
        {
            'match_id': 'integration_test',
            'innings': 1,
            'over': 1,
            'ball': 3,
            'batter': 'rohit',
            'bowler': 'bumrah',
            'runs_scored': 0,
            'wicket_type': 'bowled',
            'venue': 'wankhede',
            'date': '2024-01-15',
            'team_batting': 'india',
            'team_bowling': 'australia'
        }
    ])
    
    print(f"✅ Created sample match data with {len(match_data)} deliveries")
    
    # Step 2: Build standard cricket knowledge graph
    print("\n📊 Building standard cricket knowledge graph...")
    
    try:
        # Build the graph using existing pipeline
        cricket_graph = build_cricket_graph(match_data)
        print(f"✅ Built graph with {cricket_graph.number_of_nodes()} nodes and {cricket_graph.number_of_edges()} edges")
        
        # Show original player node features
        player_nodes = [n for n, d in cricket_graph.nodes(data=True) if d.get('node_type') == 'player']
        print(f"📋 Found {len(player_nodes)} player nodes")
        
        if player_nodes:
            sample_player = player_nodes[0]
            original_features = cricket_graph.nodes[sample_player].get('features', np.array([]))
            print(f"   Original feature dimension for {sample_player}: {len(original_features)}")
        
    except Exception as e:
        print(f"⚠️ Could not build full graph (missing dependencies): {e}")
        print("   Creating simplified graph for demonstration...")
        
        # Create a simplified graph for demonstration
        import networkx as nx
        cricket_graph = nx.Graph()
        
        # Add player nodes with mock features
        for player in ['kohli', 'rohit', 'bumrah']:
            cricket_graph.add_node(
                player,
                node_type='player',
                features=np.random.random(16)  # Mock existing features
            )
        
        print(f"✅ Created simplified graph with {cricket_graph.number_of_nodes()} nodes")
    
    # Step 3: Generate biomechanical data
    print("\n🎯 Generating biomechanical data...")
    
    # Create extended match data for better aggregation
    extended_data = []
    for i in range(20):  # 20 deliveries for better statistics
        extended_data.append({
            'match_id': f'extended_test_{i//10}',
            'innings': 1,
            'over': (i // 6) + 1,
            'ball': (i % 6) + 1,
            'batter': ['kohli', 'rohit'][i % 2],
            'bowler': 'bumrah',
            'runs_scored': np.random.choice([0, 1, 4, 6]),
            'wicket_type': None
        })
    
    extended_match_df = pd.DataFrame(extended_data)
    biomech_data = create_sample_biomechanical_data(extended_match_df, noise_level=0.08)
    
    print(f"✅ Generated biomechanical data for {len(biomech_data)} deliveries")
    
    # Step 4: Process biomechanical features
    print("\n⚙️ Processing biomechanical features...")
    
    config = BiomechanicalConfig(
        rolling_window=30,
        min_deliveries_required=5,
        missing_value_threshold=0.4
    )
    
    player_biomech_features = process_match_biomechanical_data(
        extended_match_df,
        biomech_data,
        config
    )
    
    print(f"✅ Processed features for {len(player_biomech_features)} players")
    
    for player_id, features in player_biomech_features.items():
        print(f"   {player_id}: {len(features)} biomechanical features")
    
    # Step 5: Integrate biomechanical features into graph
    print("\n🔗 Integrating biomechanical features into knowledge graph...")
    
    # Get original feature dimensions
    original_dimensions = {}
    for node_id, node_data in cricket_graph.nodes(data=True):
        if node_data.get('node_type') == 'player':
            features = node_data.get('features', np.array([]))
            original_dimensions[node_id] = len(features)
    
    # Add biomechanical features
    enhanced_graph = add_biomechanical_features_to_graph(
        cricket_graph,
        player_biomech_features,
        config
    )
    
    print(f"✅ Enhanced graph with biomechanical features")
    
    # Step 6: Show feature dimension changes
    print("\n📈 Feature Dimension Changes:")
    
    for node_id, node_data in enhanced_graph.nodes(data=True):
        if node_data.get('node_type') == 'player' and node_id in player_biomech_features:
            original_dim = original_dimensions.get(node_id, 0)
            new_features = node_data.get('features', np.array([]))
            new_dim = len(new_features)
            biomech_features = node_data.get('biomechanical_features', {})
            biomech_dim = len(biomech_features)
            
            print(f"   👤 {node_id.upper()}:")
            print(f"      Original: {original_dim} features")
            print(f"      Biomechanical: {biomech_dim} features")
            print(f"      Total: {new_dim} features")
            print(f"      Increase: +{new_dim - original_dim} features ({((new_dim - original_dim) / original_dim * 100):.1f}% increase)")
    
    # Step 7: Show sample biomechanical insights
    print("\n🎯 Sample Biomechanical Insights:")
    
    for player_id in ['kohli', 'rohit']:
        if player_id in player_biomech_features:
            features = player_biomech_features[player_id]
            
            print(f"\n   👤 {player_id.upper()}:")
            
            # Batting technique analysis
            head_stability = features.get('biomech_head_stability_mean', 0)
            shot_commitment = features.get('biomech_shot_commitment_mean', 0)
            footwork = features.get('biomech_footwork_direction_mean', 0)
            
            technique_score = (head_stability + shot_commitment + footwork) / 3
            
            print(f"      🎯 Head Stability: {head_stability:.3f}")
            print(f"      💪 Shot Commitment: {shot_commitment:.3f}")
            print(f"      🦶 Footwork Quality: {footwork:.3f}")
            print(f"      📊 Overall Technique: {technique_score:.3f}")
            
            # Consistency analysis
            head_stability_std = features.get('biomech_head_stability_std', 0)
            consistency = 1 / (1 + head_stability_std)
            print(f"      📈 Consistency Score: {consistency:.3f}")
            
            # Trend analysis
            trend = features.get('biomech_head_stability_trend', 0)
            if trend > 0.01:
                print(f"      📈 Improving trend: +{trend:.4f}")
            elif trend < -0.01:
                print(f"      📉 Declining trend: {trend:.4f}")
            else:
                print(f"      ➡️ Stable performance: {trend:.4f}")
    
    print("\n" + "=" * 50)
    print("🎯 Integration Example Completed!")
    print("\nKey Integration Points:")
    print("✅ Biomechanical data can be added to existing match data")
    print("✅ Features integrate seamlessly with current graph structure")
    print("✅ Player nodes get enhanced with technique analysis")
    print("✅ No changes needed to existing GNN model architecture")
    print("✅ Feature dimensions scale automatically")
    
    return enhanced_graph, player_biomech_features


def usage_in_training_pipeline():
    """Show how to use biomechanical features in training pipeline."""
    
    print("\n" + "=" * 50)
    print("📚 Usage in Training Pipeline:")
    print("=" * 50)
    
    code_example = '''
# In your training pipeline:

from crickformers.gnn.biomechanical_features import (
    process_match_biomechanical_data,
    add_biomechanical_features_to_graph,
    BiomechanicalConfig
)

def enhanced_graph_building_pipeline(match_data, biomech_data_path=None):
    """Enhanced graph building with biomechanical features."""
    
    # Step 1: Build standard graph
    graph = build_cricket_graph(match_data)
    
    # Step 2: Add biomechanical features if available
    if biomech_data_path:
        # Load biomechanical data
        with open(biomech_data_path, 'r') as f:
            biomech_data = json.load(f)
        
        # Process biomechanical features
        config = BiomechanicalConfig(
            rolling_window=100,
            min_deliveries_required=10
        )
        
        player_features = process_match_biomechanical_data(
            match_data, biomech_data, config
        )
        
        # Integrate into graph
        graph = add_biomechanical_features_to_graph(
            graph, player_features, config
        )
    
    return graph

# Usage:
enhanced_graph = enhanced_graph_building_pipeline(
    match_data, 
    biomech_data_path="biomech_signals.json"
)
'''
    
    print(code_example)
    
    print("\n🔧 Configuration Options:")
    print("   rolling_window: Number of recent deliveries to consider (default: 100)")
    print("   min_deliveries_required: Minimum deliveries for reliable stats (default: 10)")
    print("   missing_value_threshold: Max proportion of missing values (default: 0.3)")
    print("   feature_prefix: Prefix for biomechanical feature names (default: 'biomech_')")
    
    print("\n📊 Feature Types Generated:")
    print("   _mean: Average value over rolling window")
    print("   _std: Standard deviation (consistency measure)")
    print("   _recent: Most recent value")
    print("   _trend: Recent vs early performance comparison")


if __name__ == "__main__":
    enhanced_graph, features = integrate_biomechanical_features_example()
    usage_in_training_pipeline()