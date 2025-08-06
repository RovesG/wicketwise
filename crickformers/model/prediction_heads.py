# Purpose: Defines the prediction heads for the Crickformer model.
# Author: Shamus Rae, Last Modified: 2024-07-30

"""
This module contains the individual prediction head classes. Each head takes
the final latent vector from the fusion layer and produces raw logits for a
specific prediction task.
"""

from torch import nn
import torch

class NextBallOutcomeHead(nn.Module):
    """
    Predicts the outcome of the next ball (e.g., 0, 1, 2, 4, 6, Wicket).
    Outputs raw logits for classification.
    """
    def __init__(self, latent_dim: int, num_outcomes: int, dropout_rate: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(latent_dim, num_outcomes)
        )

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_vector: The latent state from the fusion layer.
                           Shape: [batch_size, latent_dim]
        Returns:
            Raw logits for each outcome class.
            Shape: [batch_size, num_outcomes]
        """
        return self.network(latent_vector)


class WinProbabilityHead(nn.Module):
    """
    Predicts the win probability for the batting team.
    Outputs a single raw logit for regression. A sigmoid function should be
    applied to this output to get a probability between 0 and 1.
    """
    def __init__(self, latent_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(latent_dim, 1)
        )

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_vector: The latent state from the fusion layer.
                           Shape: [batch_size, latent_dim]
        Returns:
            Win probability between 0 and 1 (sigmoid applied to logits).
            Shape: [batch_size, 1]
        """
        logits = self.network(latent_vector)
        return torch.sigmoid(logits)  # Convert logits to probabilities [0,1] for BCELoss


class OddsMispricingHead(nn.Module):
    """
    Predicts whether the current betting odds are mispriced.
    Outputs a single raw logit for binary classification. A sigmoid function
    should be applied to this output for a probability.
    """
    def __init__(self, latent_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(latent_dim, 1)
        )

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_vector: The latent state from the fusion layer.
                           Shape: [batch_size, latent_dim]
        Returns:
            Mispricing probability between 0 and 1 (sigmoid applied to logits).
            Shape: [batch_size, 1]
        """
        logits = self.network(latent_vector)
        return torch.sigmoid(logits)  # Convert logits to probabilities [0,1] for BCELoss 