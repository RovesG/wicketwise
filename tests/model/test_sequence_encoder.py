# Purpose: Tests for the BallHistoryEncoder model.
# Author: Shamus Rae, Last Modified: 2024-07-30

import torch
import pytest

from crickformers.model.sequence_encoder import BallHistoryEncoder

@pytest.mark.parametrize("num_encoder_layers", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_ball_history_encoder_output_shape(num_encoder_layers, batch_size):
    """
    Tests that the BallHistoryEncoder produces an output tensor with the
    correct shape [batch_size, feature_dim].
    """
    # Model configuration
    feature_dim = 16  # Must be divisible by nhead
    nhead = 4
    dim_feedforward = 64
    seq_length = 5

    # Instantiate the model
    encoder = BallHistoryEncoder(
        feature_dim=feature_dim,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
    )
    encoder.eval() # Set to evaluation mode

    # Create a dummy input tensor
    # Shape: [batch_size, seq_length, feature_dim]
    dummy_input = torch.randn(batch_size, seq_length, feature_dim)

    # Forward pass
    with torch.no_grad():
        output = encoder(dummy_input)

    # Check output shape
    expected_shape = (batch_size, feature_dim)
    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, but got {output.shape}"

def test_ball_history_encoder_reproducibility():
    """
    Tests that the model produces the same output for the same input
    when in evaluation mode.
    """
    feature_dim = 16
    nhead = 4
    dim_feedforward = 64
    
    encoder = BallHistoryEncoder(
        feature_dim=feature_dim,
        nhead=nhead,
        num_encoder_layers=1,
        dim_feedforward=dim_feedforward,
    )
    encoder.eval()

    dummy_input = torch.randn(2, 5, feature_dim)

    with torch.no_grad():
        output1 = encoder(dummy_input)
        output2 = encoder(dummy_input)

    assert torch.allclose(output1, output2, atol=1e-6) 