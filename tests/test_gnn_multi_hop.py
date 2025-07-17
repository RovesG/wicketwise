# Purpose: Tests for multi-hop GNN training functionality
# Author: Assistant, Last Modified: 2024

import torch
import pytest
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx

from crickformers.gnn.gnn_trainer import CricketGNNTrainer, MultiHopGraphSAGE, MultiHopGCN

@pytest.fixture
def three_hop_graph():
    """
    Creates a cricket graph with clear 3-hop paths for testing multi-hop message passing.
    
    Graph structure:
    batter_A -> bowler_X -> team_Y -> venue_Z
    
    This creates a 3-hop path: batter_A can reach venue_Z through 3 hops.
    """
    G = nx.DiGraph()
    
    # Add nodes with types
    G.add_nodes_from([
        ("batter_A", {"type": "batter"}),
        ("batter_B", {"type": "batter"}),
        ("bowler_X", {"type": "bowler"}),
        ("bowler_Y", {"type": "bowler"}),
        ("team_Alpha", {"type": "team"}),
        ("team_Beta", {"type": "team"}),
        ("venue_Lords", {"type": "venue"}),
        ("venue_Oval", {"type": "venue"}),
        ("fast_bowling", {"type": "bowler_type"}),
        ("spin_bowling", {"type": "bowler_type"}),
    ])
    
    # Add edges to create 3-hop paths
    G.add_edges_from([
        # 1-hop connections
        ("batter_A", "bowler_X", {"edge_type": "faced"}),
        ("batter_B", "bowler_Y", {"edge_type": "faced"}),
        
        # 2-hop connections
        ("bowler_X", "team_Alpha", {"edge_type": "plays_for"}),
        ("bowler_Y", "team_Beta", {"edge_type": "plays_for"}),
        ("batter_A", "team_Alpha", {"edge_type": "plays_for"}),
        ("batter_B", "team_Beta", {"edge_type": "plays_for"}),
        
        # 3-hop connections
        ("team_Alpha", "venue_Lords", {"edge_type": "match_played_at"}),
        ("team_Beta", "venue_Oval", {"edge_type": "match_played_at"}),
        
        # Excellence connections (create additional paths)
        ("batter_A", "fast_bowling", {"edge_type": "excels_against"}),
        ("batter_B", "spin_bowling", {"edge_type": "excels_against"}),
        ("bowler_X", "fast_bowling", {"edge_type": "bowls_style"}),
        ("bowler_Y", "spin_bowling", {"edge_type": "bowls_style"}),
    ])
    
    return G

@pytest.fixture
def linear_chain_graph():
    """
    Creates a simple linear chain graph for testing multi-hop propagation.
    
    node_0 -> node_1 -> node_2 -> node_3 -> node_4
    
    This allows testing how information propagates through the chain.
    """
    G = nx.DiGraph()
    
    # Create a linear chain
    nodes = [f"node_{i}" for i in range(5)]
    G.add_nodes_from([(node, {"type": "player"}) for node in nodes])
    
    # Add edges in sequence
    for i in range(4):
        G.add_edge(f"node_{i}", f"node_{i+1}", edge_type="connected")
    
    return G

def test_multi_hop_initialization():
    """Test that multi-hop models initialize with correct layer structure."""
    
    # Test GraphSAGE with different layer counts
    for num_layers in [1, 2, 3]:
        model = MultiHopGraphSAGE(
            in_channels=16,
            hidden_channels=32,
            out_channels=64,
            num_layers=num_layers
        )
        
        assert len(model.layers) == num_layers
        assert model.num_layers == num_layers
        
        # Check input layer
        assert model.layers[0].in_channels == 16
        
        # Check output layer
        assert model.layers[-1].out_channels == 64
        
        # Check hidden layers dimensions
        if num_layers > 1:
            assert model.layers[0].out_channels == 32
            if num_layers > 2:
                assert model.layers[1].in_channels == 32
                assert model.layers[1].out_channels == 32

def test_multi_hop_gcn_initialization():
    """Test that multi-hop GCN models initialize correctly."""
    
    model = MultiHopGCN(
        in_channels=16,
        hidden_channels=32,
        out_channels=64,
        num_layers=3
    )
    
    assert len(model.layers) == 3
    assert model.num_layers == 3
    
    # Check layer types
    from torch_geometric.nn import GCNConv
    for layer in model.layers:
        assert isinstance(layer, GCNConv)

def test_trainer_multi_hop_configuration(three_hop_graph):
    """Test that trainer correctly configures multi-hop models."""
    
    # Test different configurations
    configs = [
        {"num_layers": 1, "model_type": "sage"},
        {"num_layers": 2, "model_type": "sage"},
        {"num_layers": 3, "model_type": "sage"},
        {"num_layers": 2, "model_type": "gcn"},
        {"num_layers": 3, "model_type": "gcn"},
    ]
    
    for config in configs:
        trainer = CricketGNNTrainer(
            three_hop_graph,
            embedding_dim=32,
            **config
        )
        
        assert trainer.num_layers == config["num_layers"]
        assert trainer.model_type == config["model_type"]
        assert len(trainer.model.layers) == config["num_layers"]

def test_embeddings_change_with_depth(three_hop_graph):
    """
    Test that embeddings change as the number of layers (depth) increases.
    This verifies that multi-hop message passing is actually working.
    """
    
    embedding_dim = 32
    
    # Create trainers with different depths
    trainers = {}
    for num_layers in [1, 2, 3]:
        trainer = CricketGNNTrainer(
            three_hop_graph,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            learning_rate=0.01
        )
        
        # Train for a few epochs
        trainer.train(epochs=5)
        trainers[num_layers] = trainer
    
    # Get final embeddings for comparison
    embeddings = {}
    for num_layers, trainer in trainers.items():
        trainer.model.eval()
        with torch.no_grad():
            emb = trainer.model(trainer.data.x, trainer.data.edge_index)
        embeddings[num_layers] = emb
    
    # Check that embeddings are different between depths
    # (They should be different due to different receptive fields)
    
    # Compare 1-hop vs 2-hop
    diff_1_2 = torch.norm(embeddings[1] - embeddings[2]).item()
    assert diff_1_2 > 0.01, f"1-hop and 2-hop embeddings are too similar: {diff_1_2}"
    
    # Compare 2-hop vs 3-hop
    diff_2_3 = torch.norm(embeddings[2] - embeddings[3]).item()
    assert diff_2_3 > 0.01, f"2-hop and 3-hop embeddings are too similar: {diff_2_3}"
    
    # Compare 1-hop vs 3-hop (should be most different)
    diff_1_3 = torch.norm(embeddings[1] - embeddings[3]).item()
    assert diff_1_3 > 0.01, f"1-hop and 3-hop embeddings are too similar: {diff_1_3}"

def test_intermediate_embeddings_analysis(linear_chain_graph):
    """
    Test the intermediate embeddings functionality to analyze multi-hop propagation.
    """
    
    trainer = CricketGNNTrainer(
        linear_chain_graph,
        embedding_dim=16,
        num_layers=3,
        learning_rate=0.01
    )
    
    # Train briefly
    trainer.train(epochs=3)
    
    # Get intermediate embeddings
    layer_outputs = trainer.get_intermediate_embeddings()
    
    # Should have 3 layers of output
    assert len(layer_outputs) == 3
    
    # Each layer should have the right shape
    num_nodes = linear_chain_graph.number_of_nodes()
    
    # First layer output should be hidden_channels
    assert layer_outputs[0].shape == (num_nodes, trainer.hidden_channels)
    
    # Last layer output should be embedding_dim
    assert layer_outputs[-1].shape == (num_nodes, trainer.embedding_dim)
    
    # Check that layers produce different outputs
    for i in range(len(layer_outputs) - 1):
        # Different layers have different dimensions, so we can't directly compare
        # but we can check that they're producing different information
        assert layer_outputs[i].shape[0] == layer_outputs[i+1].shape[0]  # Same number of nodes
        
        # Check that each layer produces non-zero outputs
        assert torch.norm(layer_outputs[i]).item() > 0
        assert torch.norm(layer_outputs[i+1]).item() > 0
        
        # Check that the outputs are not all the same (some variation)
        assert torch.std(layer_outputs[i]).item() > 0
        assert torch.std(layer_outputs[i+1]).item() > 0

def test_multi_hop_with_different_graph_sizes():
    """Test multi-hop training with different graph sizes."""
    
    # Create graphs of different sizes
    graph_sizes = [5, 10, 20]
    
    for size in graph_sizes:
        # Create a random graph
        G = nx.erdos_renyi_graph(size, 0.3, directed=True)
        
        # Add node attributes
        for node in G.nodes():
            G.nodes[node]['type'] = 'player'
        
        # Add edge attributes
        for u, v in G.edges():
            G.edges[u, v]['edge_type'] = 'connected'
        
        # Test with different layer counts
        for num_layers in [1, 2, 3]:
            trainer = CricketGNNTrainer(
                G,
                embedding_dim=16,
                num_layers=num_layers,
                learning_rate=0.01
            )
            
            # Should initialize without error
            assert trainer.data.num_nodes == size
            assert len(trainer.model.layers) == num_layers
            
            # Should train without error
            trainer.train(epochs=2)
            
            # Should produce embeddings
            trainer.model.eval()
            with torch.no_grad():
                embeddings = trainer.model(trainer.data.x, trainer.data.edge_index)
            
            assert embeddings.shape == (size, 16)

def test_hop_neighborhood_reasoning(three_hop_graph):
    """
    Test that different hop counts actually capture different neighborhood information.
    This is a more sophisticated test of multi-hop reasoning.
    """
    
    # Get the node index for batter_A
    node_list = list(three_hop_graph.nodes())
    batter_a_idx = node_list.index("batter_A")
    
    # Train models with different hop counts
    models = {}
    for num_layers in [1, 2, 3]:
        trainer = CricketGNNTrainer(
            three_hop_graph,
            embedding_dim=32,
            num_layers=num_layers,
            learning_rate=0.01
        )
        trainer.train(epochs=10)
        models[num_layers] = trainer
    
    # Get embeddings for batter_A from each model
    batter_a_embeddings = {}
    for num_layers, trainer in models.items():
        trainer.model.eval()
        with torch.no_grad():
            embeddings = trainer.model(trainer.data.x, trainer.data.edge_index)
        batter_a_embeddings[num_layers] = embeddings[batter_a_idx]
    
    # The embeddings should be sufficiently different
    # because they're seeing different sized neighborhoods
    
    # 1-hop vs 3-hop should be most different
    similarity_1_3 = torch.cosine_similarity(
        batter_a_embeddings[1].unsqueeze(0),
        batter_a_embeddings[3].unsqueeze(0)
    ).item()
    
    # Should not be perfectly similar (cosine similarity < 0.95)
    assert similarity_1_3 < 0.95, f"1-hop and 3-hop are too similar: {similarity_1_3}"
    
    # All embeddings should have the same dimension
    for emb in batter_a_embeddings.values():
        assert emb.shape == (32,)

def test_model_types_produce_different_results(three_hop_graph):
    """Test that SAGE and GCN models produce different results."""
    
    # Train both model types
    sage_trainer = CricketGNNTrainer(
        three_hop_graph,
        embedding_dim=32,
        num_layers=3,
        model_type="sage",
        learning_rate=0.01
    )
    
    gcn_trainer = CricketGNNTrainer(
        three_hop_graph,
        embedding_dim=32,
        num_layers=3,
        model_type="gcn",
        learning_rate=0.01
    )
    
    # Set the same random seed for fair comparison
    torch.manual_seed(42)
    sage_trainer.train(epochs=5)
    
    torch.manual_seed(42)
    gcn_trainer.train(epochs=5)
    
    # Get embeddings
    sage_trainer.model.eval()
    gcn_trainer.model.eval()
    
    with torch.no_grad():
        sage_embeddings = sage_trainer.model(sage_trainer.data.x, sage_trainer.data.edge_index)
        gcn_embeddings = gcn_trainer.model(gcn_trainer.data.x, gcn_trainer.data.edge_index)
    
    # Should produce different results
    diff = torch.norm(sage_embeddings - gcn_embeddings).item()
    assert diff > 0.01, f"SAGE and GCN produce too similar results: {diff}"

def test_invalid_model_type_raises_error(three_hop_graph):
    """Test that invalid model types raise appropriate errors."""
    
    with pytest.raises(ValueError, match="Unknown model_type"):
        CricketGNNTrainer(
            three_hop_graph,
            embedding_dim=32,
            model_type="invalid_type"
        ) 