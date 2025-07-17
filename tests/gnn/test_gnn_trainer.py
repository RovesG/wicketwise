# Purpose: Tests for the CricketGNNTrainer with multi-hop capabilities.
# Author: Shamus Rae, Last Modified: 2024-07-30

import torch
import pytest
import networkx as nx
import os

from crickformers.gnn.gnn_trainer import CricketGNNTrainer

@pytest.fixture
def tiny_mock_graph():
    """Creates a small, simple NetworkX graph for testing."""
    G = nx.DiGraph()
    G.add_nodes_from([
        ("player_A", {"type": "batter"}),
        ("player_B", {"type": "bowler"}),
        ("Team_X", {"type": "team"}),
    ])
    G.add_edges_from([
        ("player_A", "player_B", {"edge_type": "faced"}),
        ("player_A", "Team_X", {"edge_type": "plays_for"}),
        ("player_B", "Team_X", {"edge_type": "plays_for"}),
    ])
    return G

def test_gnn_trainer_initialization(tiny_mock_graph):
    """
    Tests that the CricketGNNTrainer initializes correctly and converts the graph.
    """
    try:
        trainer = CricketGNNTrainer(tiny_mock_graph, embedding_dim=32)
        # Check if the data object was created
        assert trainer.data is not None
        assert trainer.data.num_nodes == 3
        assert trainer.data.num_edges == 3
        assert trainer.model.layers[-1].out_channels == 32
    except Exception as e:
        pytest.fail(f"CricketGNNTrainer initialization failed: {e}")

def test_gnn_training_completes(tiny_mock_graph):
    """
    Validates that the training process runs for a few epochs without errors.
    """
    trainer = CricketGNNTrainer(tiny_mock_graph)
    try:
        trainer.train(epochs=3)
    except Exception as e:
        pytest.fail(f"GNN training failed with an exception: {e}")

def test_embedding_export(tiny_mock_graph, tmp_path):
    """
    Tests that node embeddings are exported correctly, checking dimensions and file creation.
    """
    embedding_dim = 16
    trainer = CricketGNNTrainer(tiny_mock_graph, embedding_dim=embedding_dim)
    trainer.train(epochs=2)
    
    # Use a temporary path for the output file
    output_file = tmp_path / "test_embeddings.pt"
    
    exported_embeddings = trainer.export_embeddings(str(output_file))
    
    # 1. Check if the file was created
    assert os.path.exists(output_file)
    
    # 2. Check the exported dictionary
    assert isinstance(exported_embeddings, dict)
    assert len(exported_embeddings) == tiny_mock_graph.number_of_nodes()
    
    # 3. Check the dimensions of the embeddings
    for node_id, embedding in exported_embeddings.items():
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (embedding_dim,)
        
    # 4. Load the file and verify its contents
    loaded_embeddings = torch.load(output_file)
    assert len(loaded_embeddings) == len(exported_embeddings)
    assert torch.allclose(loaded_embeddings["player_A"], exported_embeddings["player_A"]) 