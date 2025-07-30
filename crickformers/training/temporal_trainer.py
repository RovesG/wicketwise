# Purpose: Training wrapper for learnable temporal decay with gradient flow
# Author: Shamus Rae, Last Modified: 2024-01-15

"""
This module provides a training wrapper that ensures proper gradient flow
for learnable temporal decay parameters, with specialized optimizers and
learning rate schedules.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from collections import defaultdict
import time

from crickformers.model.learnable_temporal_decay import (
    LearnableTemporalDecay,
    AdaptiveTemporalEncoder,
    TemporalDecayLoss
)

logger = logging.getLogger(__name__)


class TemporalDecayTrainer:
    """
    Training wrapper for learnable temporal decay parameters.
    Ensures proper gradient flow and optimization of half-life parameters.
    """
    
    def __init__(
        self,
        temporal_decay: LearnableTemporalDecay,
        main_model: nn.Module,
        temporal_lr: float = 0.001,
        main_lr: float = 0.0001,
        temporal_weight_decay: float = 0.01,
        main_weight_decay: float = 0.01,
        use_separate_optimizers: bool = True,
        scheduler_type: str = "plateau",
        scheduler_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize temporal decay trainer.
        
        Args:
            temporal_decay: Learnable temporal decay module
            main_model: Main model (e.g., CrickFormer)
            temporal_lr: Learning rate for temporal parameters
            main_lr: Learning rate for main model parameters
            temporal_weight_decay: Weight decay for temporal parameters
            main_weight_decay: Weight decay for main model parameters
            use_separate_optimizers: Whether to use separate optimizers
            scheduler_type: Type of learning rate scheduler
            scheduler_config: Configuration for scheduler
        """
        self.temporal_decay = temporal_decay
        self.main_model = main_model
        self.temporal_lr = temporal_lr
        self.main_lr = main_lr
        self.use_separate_optimizers = use_separate_optimizers
        
        # Create optimizers
        self._create_optimizers(temporal_weight_decay, main_weight_decay)
        
        # Create schedulers
        self._create_schedulers(scheduler_type, scheduler_config or {})
        
        # Create temporal loss function
        self.temporal_loss = TemporalDecayLoss(temporal_decay)
        
        # Training statistics
        self.training_stats = defaultdict(list)
        self.gradient_stats = defaultdict(list)
        self.half_life_history = defaultdict(list)
        
        # Gradient clipping
        self.max_grad_norm = 1.0
        self.temporal_max_grad_norm = 0.1
    
    def _create_optimizers(self, temporal_weight_decay: float, main_weight_decay: float):
        """Create optimizers for temporal and main parameters."""
        if self.use_separate_optimizers:
            # Separate optimizers for temporal and main parameters
            temporal_params = list(self.temporal_decay.parameters())
            main_params = list(self.main_model.parameters())
            
            self.temporal_optimizer = optim.AdamW(
                temporal_params,
                lr=self.temporal_lr,
                weight_decay=temporal_weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            self.main_optimizer = optim.AdamW(
                main_params,
                lr=self.main_lr,
                weight_decay=main_weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            logger.info(f"Created separate optimizers: temporal_params={len(temporal_params)}, main_params={len(main_params)}")
        else:
            # Single optimizer for all parameters
            all_params = list(self.temporal_decay.parameters()) + list(self.main_model.parameters())
            
            # Use different learning rates for different parameter groups
            param_groups = [
                {'params': list(self.temporal_decay.parameters()), 'lr': self.temporal_lr, 'weight_decay': temporal_weight_decay},
                {'params': list(self.main_model.parameters()), 'lr': self.main_lr, 'weight_decay': main_weight_decay}
            ]
            
            self.optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
            self.temporal_optimizer = None
            self.main_optimizer = None
            
            logger.info(f"Created single optimizer with {len(param_groups)} parameter groups")
    
    def _create_schedulers(self, scheduler_type: str, scheduler_config: Dict[str, Any]):
        """Create learning rate schedulers."""
        default_config = {
            'plateau': {'mode': 'min', 'factor': 0.5, 'patience': 10, 'min_lr': 1e-6},
            'cosine': {'T_max': 100, 'eta_min': 1e-6}
        }
        
        config = default_config.get(scheduler_type, {})
        config.update(scheduler_config)
        
        if self.use_separate_optimizers:
            if scheduler_type == "plateau":
                self.temporal_scheduler = ReduceLROnPlateau(self.temporal_optimizer, **config)
                self.main_scheduler = ReduceLROnPlateau(self.main_optimizer, **config)
            elif scheduler_type == "cosine":
                self.temporal_scheduler = CosineAnnealingLR(self.temporal_optimizer, **config)
                self.main_scheduler = CosineAnnealingLR(self.main_optimizer, **config)
            else:
                self.temporal_scheduler = None
                self.main_scheduler = None
        else:
            if scheduler_type == "plateau":
                self.scheduler = ReduceLROnPlateau(self.optimizer, **config)
            elif scheduler_type == "cosine":
                self.scheduler = CosineAnnealingLR(self.optimizer, **config)
            else:
                self.scheduler = None
            
            self.temporal_scheduler = None
            self.main_scheduler = None
    
    def train_step(
        self,
        batch_data: Dict[str, torch.Tensor],
        compute_temporal_loss: bool = True,
        temporal_loss_weight: float = 0.1
    ) -> Dict[str, float]:
        """
        Perform a single training step with temporal decay.
        
        Args:
            batch_data: Batch of training data
            compute_temporal_loss: Whether to compute temporal decay loss
            temporal_loss_weight: Weight for temporal loss component
        
        Returns:
            Dictionary of loss values and metrics
        """
        # Extract data
        days_ago = batch_data.get('days_ago')
        feature_values = batch_data.get('feature_values')
        targets = batch_data.get('targets')
        
        if days_ago is None or feature_values is None or targets is None:
            raise ValueError("Missing required batch data: days_ago, feature_values, targets")
        
        # Forward pass with temporal decay
        temporal_weights, weighted_features = self.temporal_decay(days_ago, feature_values)
        
        # Main model forward pass
        predictions = self.main_model(weighted_features)
        
        # Compute losses
        losses = {}
        
        # Main prediction loss
        main_loss = nn.functional.mse_loss(predictions, targets)
        losses['main_loss'] = main_loss.item()
        
        # Temporal decay loss
        if compute_temporal_loss:
            temporal_losses = self.temporal_loss(predictions, targets, days_ago, feature_values)
            for key, value in temporal_losses.items():
                losses[f'temporal_{key}'] = value.item()
            
            total_temporal_loss = temporal_losses['total_loss']
        else:
            total_temporal_loss = torch.tensor(0.0, device=main_loss.device)
        
        # Total loss
        total_loss = main_loss + temporal_loss_weight * total_temporal_loss
        losses['total_loss'] = total_loss.item()
        
        # Backward pass
        if self.use_separate_optimizers:
            # Clear gradients
            self.temporal_optimizer.zero_grad()
            self.main_optimizer.zero_grad()
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            temporal_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.temporal_decay.parameters(), self.temporal_max_grad_norm
            )
            main_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.main_model.parameters(), self.max_grad_norm
            )
            
            # Optimizer steps
            self.temporal_optimizer.step()
            self.main_optimizer.step()
            
            losses['temporal_grad_norm'] = temporal_grad_norm.item()
            losses['main_grad_norm'] = main_grad_norm.item()
        else:
            # Single optimizer
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.temporal_decay.parameters()) + list(self.main_model.parameters()),
                self.max_grad_norm
            )
            
            self.optimizer.step()
            losses['total_grad_norm'] = total_grad_norm.item()
        
        # Update statistics
        self._update_training_stats(losses)
        self._update_gradient_stats(losses)
        self._update_half_life_history()
        
        return losses
    
    def validate_step(
        self,
        batch_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Perform a validation step.
        
        Args:
            batch_data: Batch of validation data
        
        Returns:
            Dictionary of validation metrics
        """
        with torch.no_grad():
            # Extract data
            days_ago = batch_data['days_ago']
            feature_values = batch_data['feature_values']
            targets = batch_data['targets']
            
            # Forward pass
            temporal_weights, weighted_features = self.temporal_decay(days_ago, feature_values)
            predictions = self.main_model(weighted_features)
            
            # Compute losses
            main_loss = nn.functional.mse_loss(predictions, targets)
            temporal_losses = self.temporal_loss(predictions, targets, days_ago, feature_values)
            
            losses = {
                'val_main_loss': main_loss.item(),
                'val_total_loss': temporal_losses['total_loss'].item()
            }
            
            # Add temporal loss components
            for key, value in temporal_losses.items():
                if key != 'total_loss':
                    losses[f'val_temporal_{key}'] = value.item()
            
            return losses
    
    def step_schedulers(self, validation_loss: Optional[float] = None):
        """Step learning rate schedulers."""
        if self.use_separate_optimizers:
            if self.temporal_scheduler is not None:
                if isinstance(self.temporal_scheduler, ReduceLROnPlateau):
                    if validation_loss is not None:
                        self.temporal_scheduler.step(validation_loss)
                else:
                    self.temporal_scheduler.step()
            
            if self.main_scheduler is not None:
                if isinstance(self.main_scheduler, ReduceLROnPlateau):
                    if validation_loss is not None:
                        self.main_scheduler.step(validation_loss)
                else:
                    self.main_scheduler.step()
        else:
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if validation_loss is not None:
                        self.scheduler.step(validation_loss)
                else:
                    self.scheduler.step()
    
    def get_learning_rates(self) -> Dict[str, float]:
        """Get current learning rates."""
        if self.use_separate_optimizers:
            return {
                'temporal_lr': self.temporal_optimizer.param_groups[0]['lr'],
                'main_lr': self.main_optimizer.param_groups[0]['lr']
            }
        else:
            return {
                'temporal_lr': self.optimizer.param_groups[0]['lr'],
                'main_lr': self.optimizer.param_groups[1]['lr']
            }
    
    def get_half_lives(self) -> Dict[str, float]:
        """Get current half-life parameters."""
        half_lives = self.temporal_decay.get_half_lives()
        return {
            feature_name: half_lives[i].item()
            for i, feature_name in enumerate(self.temporal_decay.feature_names)
        }
    
    def _update_training_stats(self, losses: Dict[str, float]):
        """Update training statistics."""
        for key, value in losses.items():
            self.training_stats[key].append(value)
    
    def _update_gradient_stats(self, losses: Dict[str, float]):
        """Update gradient statistics."""
        grad_keys = [k for k in losses.keys() if 'grad_norm' in k]
        for key in grad_keys:
            self.gradient_stats[key].append(losses[key])
    
    def _update_half_life_history(self):
        """Update half-life parameter history."""
        current_half_lives = self.get_half_lives()
        for feature_name, half_life in current_half_lives.items():
            self.half_life_history[feature_name].append(half_life)
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = {}
        
        # Loss statistics
        for key, values in self.training_stats.items():
            if values:
                stats[f'{key}_mean'] = np.mean(values[-100:])  # Last 100 steps
                stats[f'{key}_std'] = np.std(values[-100:])
        
        # Gradient statistics
        for key, values in self.gradient_stats.items():
            if values:
                stats[f'{key}_mean'] = np.mean(values[-100:])
                stats[f'{key}_max'] = np.max(values[-100:])
        
        # Half-life statistics
        for feature_name, values in self.half_life_history.items():
            if values:
                stats[f'{feature_name}_half_life_current'] = values[-1]
                if len(values) > 1:
                    stats[f'{feature_name}_half_life_change'] = values[-1] - values[0]
        
        # Learning rates
        stats.update(self.get_learning_rates())
        
        # Temporal decay module statistics
        temporal_stats = self.temporal_decay.get_statistics()
        stats.update(temporal_stats)
        
        return stats
    
    def save_checkpoint(self, filepath: str, epoch: int, additional_info: Optional[Dict] = None):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'temporal_decay_state': self.temporal_decay.state_dict(),
            'main_model_state': self.main_model.state_dict(),
            'training_stats': dict(self.training_stats),
            'gradient_stats': dict(self.gradient_stats),
            'half_life_history': dict(self.half_life_history)
        }
        
        # Save optimizer states
        if self.use_separate_optimizers:
            checkpoint['temporal_optimizer_state'] = self.temporal_optimizer.state_dict()
            checkpoint['main_optimizer_state'] = self.main_optimizer.state_dict()
            
            if self.temporal_scheduler is not None:
                checkpoint['temporal_scheduler_state'] = self.temporal_scheduler.state_dict()
            if self.main_scheduler is not None:
                checkpoint['main_scheduler_state'] = self.main_scheduler.state_dict()
        else:
            checkpoint['optimizer_state'] = self.optimizer.state_dict()
            if self.scheduler is not None:
                checkpoint['scheduler_state'] = self.scheduler.state_dict()
        
        # Add additional info
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved training checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Load model states
        self.temporal_decay.load_state_dict(checkpoint['temporal_decay_state'])
        self.main_model.load_state_dict(checkpoint['main_model_state'])
        
        # Load optimizer states
        if self.use_separate_optimizers:
            if 'temporal_optimizer_state' in checkpoint:
                self.temporal_optimizer.load_state_dict(checkpoint['temporal_optimizer_state'])
            if 'main_optimizer_state' in checkpoint:
                self.main_optimizer.load_state_dict(checkpoint['main_optimizer_state'])
            
            if 'temporal_scheduler_state' in checkpoint and self.temporal_scheduler is not None:
                self.temporal_scheduler.load_state_dict(checkpoint['temporal_scheduler_state'])
            if 'main_scheduler_state' in checkpoint and self.main_scheduler is not None:
                self.main_scheduler.load_state_dict(checkpoint['main_scheduler_state'])
        else:
            if 'optimizer_state' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if 'scheduler_state' in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        # Load statistics
        if 'training_stats' in checkpoint:
            self.training_stats.update(checkpoint['training_stats'])
        if 'gradient_stats' in checkpoint:
            self.gradient_stats.update(checkpoint['gradient_stats'])
        if 'half_life_history' in checkpoint:
            self.half_life_history.update(checkpoint['half_life_history'])
        
        logger.info(f"Loaded training checkpoint from {filepath}")
        return checkpoint
    
    def analyze_gradient_flow(self) -> Dict[str, Any]:
        """Analyze gradient flow for temporal parameters."""
        analysis = {}
        
        # Check if temporal parameters have gradients
        temporal_params_with_grad = []
        temporal_params_without_grad = []
        
        for name, param in self.temporal_decay.named_parameters():
            if param.grad is not None:
                temporal_params_with_grad.append(name)
                grad_norm = param.grad.norm().item()
                analysis[f'{name}_grad_norm'] = grad_norm
            else:
                temporal_params_without_grad.append(name)
        
        analysis['temporal_params_with_grad'] = temporal_params_with_grad
        analysis['temporal_params_without_grad'] = temporal_params_without_grad
        analysis['temporal_gradient_flow_health'] = len(temporal_params_with_grad) / max(len(list(self.temporal_decay.parameters())), 1)
        
        # Gradient statistics
        if self.gradient_stats:
            for key, values in self.gradient_stats.items():
                if 'temporal' in key and values:
                    analysis[f'{key}_recent_mean'] = np.mean(values[-10:])
                    analysis[f'{key}_recent_std'] = np.std(values[-10:])
        
        return analysis


def create_temporal_trainer(
    temporal_decay: LearnableTemporalDecay,
    main_model: nn.Module,
    config: Optional[Dict[str, Any]] = None
) -> TemporalDecayTrainer:
    """
    Factory function to create temporal decay trainer.
    
    Args:
        temporal_decay: Learnable temporal decay module
        main_model: Main model
        config: Optional configuration dictionary
    
    Returns:
        Configured TemporalDecayTrainer
    """
    default_config = {
        'temporal_lr': 0.001,
        'main_lr': 0.0001,
        'temporal_weight_decay': 0.01,
        'main_weight_decay': 0.01,
        'use_separate_optimizers': True,
        'scheduler_type': 'plateau',
        'scheduler_config': {}
    }
    
    if config:
        default_config.update(config)
    
    return TemporalDecayTrainer(temporal_decay, main_model, **default_config)