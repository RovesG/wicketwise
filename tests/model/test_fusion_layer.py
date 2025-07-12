# Purpose: Tests for the CrickformerFusionLayer.
# Author: Shamus Rae, Last Modified: 2024-07-30

import torch
import pytest
from crickformers.model.fusion_layer import CrickformerFusionLayer

@pytest.fixture
def fusion_config():
    """Provides a sample configuration for the fusion layer."""
    return {
        "sequence_dim": 32,
        "context_dim": 16,
        "kg_dim": 24,
        "hidden_dims": [64, 48],
        "latent_dim": 40,
        "dropout_rate": 0.5,
    }

@pytest.fixture
def sample_input_tensors(fusion_config):
    """Provides a tuple of sample input tensors for the fusion layer."""
    batch_size = 4
    seq_vec = torch.randn(batch_size, fusion_config["sequence_dim"])
    ctx_vec = torch.randn(batch_size, fusion_config["context_dim"])
    kg_vec = torch.randn(batch_size, fusion_config["kg_dim"])
    return seq_vec, ctx_vec, kg_vec

def test_fusion_layer_output_shape(fusion_config, sample_input_tensors):
    """Tests that the fusion layer produces an output with the correct shape."""
    model = CrickformerFusionLayer(**fusion_config)
    model.eval()
    
    seq_vec, ctx_vec, kg_vec = sample_input_tensors
    
    with torch.no_grad():
        output = model(seq_vec, ctx_vec, kg_vec)
        
    expected_shape = (seq_vec.shape[0], fusion_config["latent_dim"])
    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, but got {output.shape}"

def test_dropout_and_activation_logic(fusion_config, sample_input_tensors):
    """
    Checks that dropout is active during training and inactive during evaluation.
    """
    model = CrickformerFusionLayer(**fusion_config)
    seq_vec, ctx_vec, kg_vec = sample_input_tensors

    # In training mode, dropout should be active, so outputs should differ
    model.train()
    with torch.no_grad():
        output1_train = model(seq_vec, ctx_vec, kg_vec)
        output2_train = model(seq_vec, ctx_vec, kg_vec)
    assert not torch.allclose(output1_train, output2_train), \
        "Outputs should be different in train mode due to dropout."

    # In evaluation mode, dropout is disabled, so outputs should be identical
    model.eval()
    with torch.no_grad():
        output1_eval = model(seq_vec, ctx_vec, kg_vec)
        output2_eval = model(seq_vec, ctx_vec, kg_vec)
    assert torch.allclose(output1_eval, output2_eval), \
        "Outputs should be identical in eval mode."

def test_fusion_layer_handles_zero_tensors(fusion_config, sample_input_tensors):
    """
    Ensures the layer can process tensors containing all zeros without error.
    """
    model = CrickformerFusionLayer(**fusion_config)
    model.eval()
    
    seq_vec, _, kg_vec = sample_input_tensors
    # Create a zero tensor for the context vector
    zero_ctx_vec = torch.zeros_like(sample_input_tensors[1])
    
    try:
        with torch.no_grad():
            output = model(seq_vec, zero_ctx_vec, kg_vec)
    except Exception as e:
        pytest.fail(f"Fusion layer failed to handle zero tensors: {e}")
        
    expected_shape = (seq_vec.shape[0], fusion_config["latent_dim"])
    assert output.shape == expected_shape 