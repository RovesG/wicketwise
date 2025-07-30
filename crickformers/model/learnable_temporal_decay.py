# Purpose: Learnable temporal decay mechanism with feature-specific half-life parameters
# Author: Shamus Rae, Last Modified: 2024-01-15

"""
This module implements a learnable temporal decay mechanism that replaces fixed
exponential decay with adaptive, feature-specific temporal weighting. Each feature
learns its own optimal half-life parameter for temporal decay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import math

logger = logging.getLogger(__name__)


class LearnableTemporalDecay(nn.Module):
    """
    Learnable temporal decay mechanism with feature-specific half-life parameters.
    
    For each feature, learns a half-life parameter that controls decay speed:
    weight = exp(-ln(2) * (days_ago / half_life_feature))
    """
    
    def __init__(
        self,
        feature_names: List[str],
        initial_half_life: float = 30.0,
        min_half_life: float = 1.0,
        max_half_life: float = 365.0,
        learnable: bool = True,
        temperature: float = 1.0
    ):
        """
        Initialize learnable temporal decay.
        
        Args:
            feature_names: List of feature names to learn decay for
            initial_half_life: Initial half-life in days for all features
            min_half_life: Minimum allowed half-life (days)
            max_half_life: Maximum allowed half-life (days)
            learnable: Whether half-life parameters are learnable
            temperature: Temperature for softmax-like weighting
        """
        super().__init__()
        
        self.feature_names = feature_names
        self.num_features = len(feature_names)
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.learnable = learnable
        self.temperature = temperature
        
        # Initialize half-life parameters
        # Use log-space for better optimization dynamics
        initial_log_half_life = math.log(initial_half_life)
        
        if learnable:
            self.log_half_lives = nn.Parameter(
                torch.full((self.num_features,), initial_log_half_life, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'log_half_lives',
                torch.full((self.num_features,), initial_log_half_life, dtype=torch.float32)
            )
        
        # Feature name to index mapping
        self.feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        # Statistics tracking
        self.register_buffer('decay_stats', torch.zeros(self.num_features, 4))  # min, max, mean, std
        self.register_buffer('update_count', torch.zeros(1, dtype=torch.long))
    
    def get_half_lives(self) -> torch.Tensor:
        """
        Get current half-life parameters (clamped to valid range).
        
        Returns:
            Tensor of shape (num_features,) with half-life values
        """
        half_lives = torch.exp(self.log_half_lives)
        return torch.clamp(half_lives, self.min_half_life, self.max_half_life)
    
    def compute_temporal_weights(
        self,
        days_ago: torch.Tensor,
        feature_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute temporal weights using learnable decay.
        
        Args:
            days_ago: Tensor of days ago values, shape (...,)
            feature_indices: Feature indices for each weight, shape (...,)
                           If None, uses first feature's half-life for all
        
        Returns:
            Temporal weights, same shape as days_ago
        """
        half_lives = self.get_half_lives()
        
        if feature_indices is None:
            # Use first feature's half-life for all weights
            half_life = half_lives[0]
        else:
            # Use feature-specific half-lives
            half_life = half_lives[feature_indices]
        
        # Compute exponential decay: exp(-ln(2) * (days_ago / half_life))
        # Equivalent to: (1/2) ^ (days_ago / half_life)
        ln2 = math.log(2.0)
        decay_exponent = -ln2 * (days_ago / half_life)
        weights = torch.exp(decay_exponent)
        
        # Apply temperature scaling for smoother gradients
        if self.temperature != 1.0:
            weights = torch.pow(weights, 1.0 / self.temperature)
        
        return weights
    
    def compute_feature_weights(
        self,
        days_ago: torch.Tensor,
        feature_values: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Compute temporal weights for feature values.
        
        Args:
            days_ago: Days ago for each sample, shape (batch_size,)
            feature_values: Feature values, shape (batch_size, num_features)
            feature_names: Optional feature names, uses self.feature_names if None
        
        Returns:
            Weighted feature values, shape (batch_size, num_features)
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        batch_size, num_feats = feature_values.shape
        
        # Get feature indices
        feature_indices = torch.tensor([
            self.feature_to_idx.get(name, 0) for name in feature_names
        ], device=feature_values.device, dtype=torch.long)
        
        # Expand for broadcasting
        days_ago_expanded = days_ago.unsqueeze(1)  # (batch_size, 1)
        feature_indices_expanded = feature_indices.unsqueeze(0)  # (1, num_features)
        
        # Compute weights
        weights = self.compute_temporal_weights(
            days_ago_expanded, feature_indices_expanded
        )  # (batch_size, num_features)
        
        # Apply weights
        weighted_features = feature_values * weights
        
        # Update statistics
        self._update_statistics(weights)
        
        return weighted_features
    
    def compute_edge_weights(
        self,
        days_ago: torch.Tensor,
        base_weights: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        feature_types: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Compute temporal weights for graph edges.
        
        Args:
            days_ago: Days ago for each edge, shape (num_edges,)
            base_weights: Base edge weights, shape (num_edges,)
            edge_features: Optional edge features, shape (num_edges, num_edge_features)
            feature_types: Optional feature type names for edge features
        
        Returns:
            Temporally weighted edge weights, shape (num_edges,)
        """
        if edge_features is not None and feature_types is not None:
            # Use feature-specific decay
            feature_indices = torch.tensor([
                self.feature_to_idx.get(name, 0) for name in feature_types
            ], device=days_ago.device, dtype=torch.long)
            
            # Average decay across edge features
            weights = self.compute_temporal_weights(days_ago.unsqueeze(1), feature_indices.unsqueeze(0))
            temporal_weight = weights.mean(dim=1)
        else:
            # Use default (first feature) decay
            temporal_weight = self.compute_temporal_weights(days_ago)
        
        return base_weights * temporal_weight
    
    def get_aggregated_form_vector(
        self,
        feature_history: torch.Tensor,
        days_ago_history: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Aggregate historical features into form vector using learnable decay.
        
        Args:
            feature_history: Historical features, shape (history_length, num_features)
            days_ago_history: Days ago for each historical point, shape (history_length,)
            feature_names: Feature names, uses self.feature_names if None
        
        Returns:
            Aggregated form vector, shape (num_features,)
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        # Compute temporal weights for each historical point
        weighted_features = self.compute_feature_weights(
            days_ago_history, feature_history, feature_names
        )
        
        # Normalize by sum of weights to maintain scale
        temporal_weights = self.compute_temporal_weights(
            days_ago_history.unsqueeze(1),
            torch.arange(len(feature_names), device=feature_history.device).unsqueeze(0)
        )
        
        weight_sums = temporal_weights.sum(dim=0, keepdim=True)
        weight_sums = torch.clamp(weight_sums, min=1e-8)  # Avoid division by zero
        
        # Weighted average
        form_vector = weighted_features.sum(dim=0) / weight_sums.squeeze(0)
        
        return form_vector
    
    def _update_statistics(self, weights: torch.Tensor):
        """Update decay weight statistics for monitoring."""
        if not self.training:
            return
        
        with torch.no_grad():
            # Compute statistics per feature
            weight_stats = torch.stack([
                weights.min(dim=0)[0],  # min
                weights.max(dim=0)[0],  # max
                weights.mean(dim=0),    # mean
                weights.std(dim=0)      # std
            ], dim=1)  # (num_features, 4)
            
            # Exponential moving average
            alpha = 0.01
            self.decay_stats = (1 - alpha) * self.decay_stats + alpha * weight_stats
            self.update_count += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current decay statistics."""
        if self.update_count == 0:
            return {}
        
        half_lives = self.get_half_lives()
        stats = {}
        
        for i, feature_name in enumerate(self.feature_names):
            stats[f"{feature_name}_half_life"] = half_lives[i].item()
            stats[f"{feature_name}_weight_min"] = self.decay_stats[i, 0].item()
            stats[f"{feature_name}_weight_max"] = self.decay_stats[i, 1].item()
            stats[f"{feature_name}_weight_mean"] = self.decay_stats[i, 2].item()
            stats[f"{feature_name}_weight_std"] = self.decay_stats[i, 3].item()
        
        return stats
    
    def regularization_loss(self, l1_weight: float = 0.01, l2_weight: float = 0.01) -> torch.Tensor:
        """
        Compute regularization loss for half-life parameters.
        
        Args:
            l1_weight: L1 regularization weight
            l2_weight: L2 regularization weight
        
        Returns:
            Regularization loss
        """
        if not self.learnable:
            return torch.tensor(0.0, device=self.log_half_lives.device)
        
        half_lives = self.get_half_lives()
        
        # L1 loss (sparsity)
        l1_loss = l1_weight * half_lives.abs().mean()
        
        # L2 loss (smoothness)
        l2_loss = l2_weight * (half_lives ** 2).mean()
        
        return l1_loss + l2_loss
    
    def forward(
        self,
        days_ago: torch.Tensor,
        feature_values: Optional[torch.Tensor] = None,
        feature_names: Optional[List[str]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for temporal decay.
        
        Args:
            days_ago: Days ago values, shape (...,)
            feature_values: Optional feature values for weighting
            feature_names: Optional feature names
        
        Returns:
            If feature_values is None: temporal weights
            If feature_values is provided: (temporal_weights, weighted_features)
        """
        if feature_values is None:
            # Just compute temporal weights
            return self.compute_temporal_weights(days_ago)
        else:
            # Compute weighted features
            weighted_features = self.compute_feature_weights(
                days_ago, feature_values, feature_names
            )
            # Compute temporal weights for the feature dimensions
            if days_ago.dim() == 1:
                days_ago_expanded = days_ago.unsqueeze(1).expand(-1, len(feature_names or self.feature_names))
                feature_indices = torch.arange(len(feature_names or self.feature_names), device=days_ago.device).unsqueeze(0).expand(len(days_ago), -1)
                weights = self.compute_temporal_weights(days_ago_expanded, feature_indices)
            else:
                weights = self.compute_temporal_weights(days_ago)
            return weights, weighted_features


class AdaptiveTemporalEncoder(nn.Module):
    """
    Adaptive temporal encoder that combines learnable decay with positional encoding.
    """
    
    def __init__(
        self,
        feature_names: List[str],
        embed_dim: int = 64,
        max_days: int = 365,
        use_positional_encoding: bool = True,
        **decay_kwargs
    ):
        """
        Initialize adaptive temporal encoder.
        
        Args:
            feature_names: List of feature names
            embed_dim: Embedding dimension
            max_days: Maximum days for positional encoding
            use_positional_encoding: Whether to use positional encoding
            **decay_kwargs: Arguments for LearnableTemporalDecay
        """
        super().__init__()
        
        self.feature_names = feature_names
        self.embed_dim = embed_dim
        self.max_days = max_days
        self.use_positional_encoding = use_positional_encoding
        
        # Learnable temporal decay
        self.temporal_decay = LearnableTemporalDecay(feature_names, **decay_kwargs)
        
        # Positional encoding for temporal information
        if use_positional_encoding:
            self.temporal_embedding = nn.Embedding(max_days + 1, embed_dim)
            self.temporal_projection = nn.Linear(embed_dim, len(feature_names))
        
        # Feature-specific temporal encoders
        self.feature_encoders = nn.ModuleDict({
            name: nn.Linear(1, embed_dim) for name in feature_names
        })
        
        # Output projection
        self.output_projection = nn.Linear(len(feature_names) * embed_dim, len(feature_names))
    
    def forward(
        self,
        days_ago: torch.Tensor,
        feature_values: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Forward pass for adaptive temporal encoding.
        
        Args:
            days_ago: Days ago values, shape (batch_size,)
            feature_values: Feature values, shape (batch_size, num_features)
            feature_names: Feature names, uses self.feature_names if None
        
        Returns:
            Temporally encoded features, shape (batch_size, num_features)
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        batch_size, num_features = feature_values.shape
        
        # Apply learnable temporal decay
        decay_weights, weighted_features = self.temporal_decay(
            days_ago, feature_values, feature_names
        )
        
        # Positional encoding
        if self.use_positional_encoding:
            # Clamp days to valid range
            days_clamped = torch.clamp(days_ago.long(), 0, self.max_days)
            temporal_embeds = self.temporal_embedding(days_clamped)  # (batch_size, embed_dim)
            temporal_weights = torch.sigmoid(self.temporal_projection(temporal_embeds))  # (batch_size, num_features)
            
            # Combine with decay weights
            combined_weights = decay_weights * temporal_weights
        else:
            combined_weights = decay_weights
        
        # Feature-specific encoding
        encoded_features = []
        for i, feature_name in enumerate(feature_names):
            if feature_name in self.feature_encoders:
                feature_val = weighted_features[:, i:i+1]  # (batch_size, 1)
                encoded = self.feature_encoders[feature_name](feature_val)  # (batch_size, embed_dim)
                encoded_features.append(encoded)
        
        if encoded_features:
            # Concatenate and project
            concat_features = torch.cat(encoded_features, dim=1)  # (batch_size, num_features * embed_dim)
            output = self.output_projection(concat_features)  # (batch_size, num_features)
        else:
            # Fallback to weighted features
            output = weighted_features
        
        return output


class TemporalDecayLoss(nn.Module):
    """
    Loss function for training temporal decay parameters.
    """
    
    def __init__(
        self,
        decay_module: LearnableTemporalDecay,
        target_half_lives: Optional[Dict[str, float]] = None,
        consistency_weight: float = 0.1,
        smoothness_weight: float = 0.05
    ):
        """
        Initialize temporal decay loss.
        
        Args:
            decay_module: Learnable temporal decay module
            target_half_lives: Optional target half-lives for supervised learning
            consistency_weight: Weight for temporal consistency loss
            smoothness_weight: Weight for smoothness regularization
        """
        super().__init__()
        
        self.decay_module = decay_module
        self.target_half_lives = target_half_lives or {}
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        days_ago: torch.Tensor,
        feature_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute temporal decay loss.
        
        Args:
            predictions: Model predictions
            targets: Target values
            days_ago: Days ago for each sample
            feature_values: Feature values used in predictions
        
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Main prediction loss
        main_loss = F.mse_loss(predictions, targets)
        losses['main_loss'] = main_loss
        
        # Temporal consistency loss
        if self.consistency_weight > 0:
            consistency_loss = self._compute_consistency_loss(days_ago, feature_values)
            losses['consistency_loss'] = consistency_loss
        
        # Smoothness regularization
        if self.smoothness_weight > 0:
            smoothness_loss = self._compute_smoothness_loss()
            losses['smoothness_loss'] = smoothness_loss
        
        # Supervised half-life loss
        if self.target_half_lives:
            supervised_loss = self._compute_supervised_loss()
            losses['supervised_loss'] = supervised_loss
        
        # Total loss
        total_loss = main_loss
        if 'consistency_loss' in losses:
            total_loss += self.consistency_weight * losses['consistency_loss']
        if 'smoothness_loss' in losses:
            total_loss += self.smoothness_weight * losses['smoothness_loss']
        if 'supervised_loss' in losses:
            total_loss += losses['supervised_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_consistency_loss(self, days_ago: torch.Tensor, feature_values: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency loss."""
        # Sort by days_ago
        sorted_indices = torch.argsort(days_ago)
        sorted_days = days_ago[sorted_indices]
        sorted_features = feature_values[sorted_indices]
        
        # Compute weights for consecutive time points
        weights_current = self.decay_module.compute_feature_weights(
            sorted_days[:-1], sorted_features[:-1]
        )
        weights_next = self.decay_module.compute_feature_weights(
            sorted_days[1:], sorted_features[1:]
        )
        
        # Consistency: closer time points should have more similar weights
        time_diffs = sorted_days[1:] - sorted_days[:-1]
        weight_diffs = torch.abs(weights_current - weights_next)
        
        # Penalize large weight differences for small time differences
        consistency_loss = (weight_diffs / (time_diffs.unsqueeze(1) + 1e-8)).mean()
        
        return consistency_loss
    
    def _compute_smoothness_loss(self) -> torch.Tensor:
        """Compute smoothness regularization loss."""
        half_lives = self.decay_module.get_half_lives()
        
        # Penalize large variations in half-lives across features
        if len(half_lives) > 1:
            smoothness_loss = torch.var(half_lives)
        else:
            smoothness_loss = torch.tensor(0.0, device=half_lives.device)
        
        return smoothness_loss
    
    def _compute_supervised_loss(self) -> torch.Tensor:
        """Compute supervised loss for target half-lives."""
        current_half_lives = self.decay_module.get_half_lives()
        supervised_loss = torch.tensor(0.0, device=current_half_lives.device)
        
        for feature_name, target_half_life in self.target_half_lives.items():
            if feature_name in self.decay_module.feature_to_idx:
                idx = self.decay_module.feature_to_idx[feature_name]
                current_half_life = current_half_lives[idx]
                target_tensor = torch.tensor(target_half_life, device=current_half_life.device)
                supervised_loss += F.mse_loss(current_half_life, target_tensor)
        
        return supervised_loss


def create_learnable_temporal_decay(
    feature_names: List[str],
    config: Optional[Dict[str, Any]] = None
) -> LearnableTemporalDecay:
    """
    Factory function to create learnable temporal decay module.
    
    Args:
        feature_names: List of feature names
        config: Optional configuration dictionary
    
    Returns:
        Configured LearnableTemporalDecay module
    """
    default_config = {
        'initial_half_life': 30.0,
        'min_half_life': 1.0,
        'max_half_life': 365.0,
        'learnable': True,
        'temperature': 1.0
    }
    
    if config:
        default_config.update(config)
    
    return LearnableTemporalDecay(feature_names, **default_config)