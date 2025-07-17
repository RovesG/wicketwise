# Purpose: Enhanced training pipeline with drift detection, confidence estimation, and monitoring
# Author: Shamus Rae, Last Modified: 2024

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import pandas as pd

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

from crickformers.crickformer_dataset import CrickformerDataset
from crickformers.csv_data_adapter import CSVDataConfig
from crickformers.model.crickformer_model import CrickformerModel
from crickformers.drift_detector import DriftDetector
from crickformers.confidence_utils import predict_with_uncertainty, calculate_confidence_score
from crickformers.train import collate_fn, load_config  # Import from original trainer

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class EnhancedTrainer:
    """Enhanced trainer with drift detection, confidence estimation, and monitoring."""
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        """Initialize the enhanced trainer."""
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Training parameters
        self.batch_size = config.get("batch_size", 32)
        self.num_epochs = config.get("num_epochs", 10)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.log_interval = config.get("log_interval", 100)
        self.validation_interval = config.get("validation_interval", 500)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        
        # Monitoring components
        self.drift_detector = DriftDetector(
            feature_dim=config.get("model", {}).get("hidden_dim", 256),
            threshold=config.get("drift_threshold", 0.1),
            window_size=config.get("drift_window_size", 1000)
        )
        
        # Training state and metrics
        self.running_loss = 0.0
        self.step_count = 0
        self.epoch_metrics = defaultdict(list)
        self.confidence_scores = deque(maxlen=1000)
        self.drift_alerts = []
        
        # Performance tracking
        self.loss_history = []
        self.validation_metrics = []
        self.confidence_history = []
        
    def setup_model(self):
        """Initialize the model."""
        model_config = self.config.get("model", {})
        
        # Create config dictionaries expected by CrickformerModel
        sequence_config = {
            "sequence_length": model_config.get("sequence_length", 5),
            "feature_dim": model_config.get("hidden_dim", 256),
            "num_layers": model_config.get("num_layers", 4),
            "num_heads": model_config.get("num_heads", 8),
            "dropout_rate": model_config.get("dropout_rate", 0.1)
        }
        
        static_config = {
            "numeric_dim": model_config.get("numeric_dim", 20),
            "categorical_vocab_sizes": model_config.get("categorical_vocab_sizes", {}),
            "categorical_embedding_dims": model_config.get("categorical_embedding_dims", {}),
            "video_dim": model_config.get("video_dim", 512),
            "hidden_dims": [model_config.get("hidden_dim", 256)],
            "context_dim": model_config.get("context_dim", 128),
            "dropout_rate": model_config.get("dropout_rate", 0.1)
        }
        
        fusion_config = {
            "sequence_dim": model_config.get("hidden_dim", 256),
            "context_dim": model_config.get("context_dim", 128),
            "gnn_dim": 128,
            "fusion_dim": model_config.get("hidden_dim", 256),
            "dropout_rate": model_config.get("dropout_rate", 0.1)
        }
        
        prediction_heads_config = {
            "win_probability": {
                "input_dim": model_config.get("hidden_dim", 256),
                "hidden_dims": [128, 64],
                "dropout_rate": model_config.get("dropout_rate", 0.1)
            },
            "next_ball_outcome": {
                "input_dim": model_config.get("hidden_dim", 256),
                "num_classes": 10,
                "hidden_dims": [128, 64],
                "dropout_rate": model_config.get("dropout_rate", 0.1)
            },
            "mispricing": {
                "input_dim": model_config.get("hidden_dim", 256),
                "hidden_dims": [128, 64],
                "dropout_rate": model_config.get("dropout_rate", 0.1)
            }
        }
        
        self.model = CrickformerModel(
            sequence_config=sequence_config,
            static_config=static_config,
            fusion_config=fusion_config,
            prediction_heads_config=prediction_heads_config,
            gnn_embedding_dim=128
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.get("weight_decay", 0.01)
        )
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
    def setup_dataset(self, data_path: str, use_csv: bool = True, 
                     train_matches_path: Optional[str] = None, 
                     val_matches_path: Optional[str] = None):
        """Setup dataset and data loaders with optional match-level filtering."""
        logger.info(f"Loading dataset from: {data_path}")
        
        if train_matches_path and val_matches_path:
            # Use match-level splits
            logger.info("Using match-level train/validation splits")
            
            if use_csv:
                train_dataset = CrickformerDataset(
                    data_root=data_path,
                    use_csv_adapter=True,
                    csv_config=CSVDataConfig(),
                    history_length=5,
                    load_video=True,
                    load_embeddings=True,
                    load_market_odds=True,
                    match_id_list_path=train_matches_path
                )
                
                val_dataset = CrickformerDataset(
                    data_root=data_path,
                    use_csv_adapter=True,
                    csv_config=CSVDataConfig(),
                    history_length=5,
                    load_video=True,
                    load_embeddings=True,
                    load_market_odds=True,
                    match_id_list_path=val_matches_path
                )
            else:
                train_dataset = CrickformerDataset(
                    data_root=data_path,
                    use_csv_adapter=False,
                    history_length=5,
                    match_id_list_path=train_matches_path
                )
                
                val_dataset = CrickformerDataset(
                    data_root=data_path,
                    use_csv_adapter=False,
                    history_length=5,
                    match_id_list_path=val_matches_path
                )
            
            logger.info(f"Training dataset: {len(train_dataset):,} samples from {len(train_dataset.get_match_ids())} matches")
            logger.info(f"Validation dataset: {len(val_dataset):,} samples from {len(val_dataset.get_match_ids())} matches")
            
        else:
            # Use random splitting with warning
            logger.warning("⚠️  Using random splitting may cause data leakage. Consider using match-level splits.")
            
            if use_csv:
                dataset = CrickformerDataset(
                    data_root=data_path,
                    use_csv_adapter=True,
                    csv_config=CSVDataConfig(),
                    history_length=5,
                    load_video=True,
                    load_embeddings=True,
                    load_market_odds=True
                )
            else:
                dataset = CrickformerDataset(
                    data_root=data_path,
                    use_csv_adapter=False,
                    history_length=5
                )
            
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=2,
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=2,
            collate_fn=collate_fn
        )
        
        logger.info(f"Data loaders created - Train: {len(self.train_loader)} batches, Val: {len(self.val_loader)} batches")
        
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss and individual loss components."""
        loss_dict = {}
        
        # Win probability loss
        win_prob_loss = nn.BCELoss()(outputs["win_prob"], targets["win_prob"])
        loss_dict["win_prob"] = win_prob_loss.item()
        
        # Next ball outcome loss
        outcome_loss = nn.CrossEntropyLoss()(outputs["next_ball_outcome"], targets["next_ball_outcome"])
        loss_dict["outcome"] = outcome_loss.item()
        
        # Mispricing loss (if available)
        mispricing_loss = torch.tensor(0.0, device=self.device)
        if "mispricing" in outputs and "mispricing" in targets:
            mispricing_loss = nn.MSELoss()(outputs["mispricing"], targets["mispricing"])
            loss_dict["mispricing"] = mispricing_loss.item()
        
        # Combine losses
        total_loss = win_prob_loss + outcome_loss + 0.5 * mispricing_loss
        loss_dict["total"] = total_loss.item()
        
        return total_loss, loss_dict
        
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a single training step with drift detection."""
        self.model.train()
        
        # Move batch to device
        inputs = {k: v.to(self.device) for k, v in batch["inputs"].items()}
        targets = {k: v.to(self.device) for k, v in batch["targets"].items()}
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        
        # Drift detection on model representations
        if hasattr(self.model, 'get_intermediate_representations'):
            representations = self.model.get_intermediate_representations(inputs)
            drift_detected = self.drift_detector.detect_drift(representations)
            if drift_detected:
                self.drift_alerts.append({
                    'step': self.step_count,
                    'epoch': len(self.epoch_metrics['loss']),
                    'drift_score': self.drift_detector.get_last_drift_score()
                })
                logger.warning(f"🚨 Drift detected at step {self.step_count}")
        
        # Compute loss
        total_loss, loss_dict = self.compute_loss(outputs, targets)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Update metrics
        self.running_loss += loss_dict["total"]
        self.step_count += 1
        self.loss_history.append(loss_dict["total"])
        
        return loss_dict
        
    def validate_with_confidence(self) -> Dict[str, float]:
        """Perform validation with confidence estimation."""
        self.model.eval()
        val_loss = 0.0
        confidence_scores = []
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = {k: v.to(self.device) for k, v in batch["inputs"].items()}
                targets = {k: v.to(self.device) for k, v in batch["targets"].items()}
                
                # Standard prediction
                outputs = self.model(inputs)
                total_loss, _ = self.compute_loss(outputs, targets)
                val_loss += total_loss.item()
                
                # Confidence estimation using Monte Carlo dropout
                mean_pred, std_pred, conf_interval = predict_with_uncertainty(
                    self.model, inputs, n_samples=20
                )
                
                # Calculate confidence scores
                for i in range(len(mean_pred["win_prob"])):
                    conf_score = calculate_confidence_score(
                        mean_pred["win_prob"][i].item(),
                        std_pred["win_prob"][i].item()
                    )
                    confidence_scores.append(conf_score)
                
                predictions.extend(outputs["win_prob"].cpu().numpy())
                targets_list.extend(targets["win_prob"].cpu().numpy())
        
        # Calculate metrics
        val_loss /= len(self.val_loader)
        avg_confidence = np.mean(confidence_scores)
        
        # Store confidence scores
        self.confidence_scores.extend(confidence_scores)
        self.confidence_history.append(avg_confidence)
        
        metrics = {
            "val_loss": val_loss,
            "avg_confidence": avg_confidence,
            "min_confidence": np.min(confidence_scores),
            "max_confidence": np.max(confidence_scores),
            "std_confidence": np.std(confidence_scores)
        }
        
        self.validation_metrics.append(metrics)
        
        return metrics
        
    def train_epoch(self, epoch: int):
        """Train for one epoch with enhanced monitoring."""
        logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
        
        epoch_start_time = time.time()
        total_batches = len(self.train_loader)
        epoch_losses = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Training step
            loss_dict = self.train_step(batch)
            epoch_losses.append(loss_dict["total"])
            
            # Validation and monitoring
            if (batch_idx + 1) % self.validation_interval == 0:
                val_metrics = self.validate_with_confidence()
                
                logger.info(
                    f"Validation | Step {self.step_count} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Avg Confidence: {val_metrics['avg_confidence']:.3f} | "
                    f"Drift Alerts: {len(self.drift_alerts)}"
                )
            
            # Regular logging
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = self.running_loss / self.log_interval
                
                logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs} | "
                    f"Batch {batch_idx + 1}/{total_batches} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Win Prob: {loss_dict['win_prob']:.4f} | "
                    f"Outcome: {loss_dict['outcome']:.4f} | "
                    f"Mispricing: {loss_dict['mispricing']:.4f}"
                )
                
                self.running_loss = 0.0
        
        # Record epoch metrics
        epoch_time = time.time() - epoch_start_time
        self.epoch_metrics['loss'].append(np.mean(epoch_losses))
        self.epoch_metrics['time'].append(epoch_time)
        
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        
        # End of epoch validation
        final_val_metrics = self.validate_with_confidence()
        logger.info(
            f"End of Epoch {epoch + 1} | "
            f"Val Loss: {final_val_metrics['val_loss']:.4f} | "
            f"Avg Confidence: {final_val_metrics['avg_confidence']:.3f}"
        )
        
    def create_monitoring_plots(self, save_dir: str = "monitoring_plots"):
        """Create comprehensive monitoring plots."""
        Path(save_dir).mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Loss curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training loss
        ax1.plot(self.loss_history, alpha=0.7, label='Training Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Validation loss
        val_losses = [m['val_loss'] for m in self.validation_metrics]
        ax2.plot(val_losses, 'o-', label='Validation Loss')
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Validation Step')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/loss_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confidence distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confidence histogram
        ax1.hist(list(self.confidence_scores), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Confidence Score Distribution')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Confidence over time
        ax2.plot(self.confidence_history, 'g-', label='Average Confidence')
        ax2.set_title('Confidence Over Time')
        ax2.set_xlabel('Validation Step')
        ax2.set_ylabel('Average Confidence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confidence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Drift detection alerts
        if self.drift_alerts:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            alert_steps = [alert['step'] for alert in self.drift_alerts]
            alert_scores = [alert['drift_score'] for alert in self.drift_alerts]
            
            ax.scatter(alert_steps, alert_scores, c='red', s=100, alpha=0.7, label='Drift Alerts')
            ax.set_title('Drift Detection Alerts')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Drift Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/drift_alerts.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Training summary
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Epoch loss
        ax1.plot(self.epoch_metrics['loss'], 'b-o', label='Epoch Loss')
        ax1.set_title('Loss per Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Epoch time
        ax2.plot(self.epoch_metrics['time'], 'r-o', label='Epoch Time')
        ax2.set_title('Training Time per Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Time (seconds)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Validation metrics
        if self.validation_metrics:
            val_conf = [m['avg_confidence'] for m in self.validation_metrics]
            ax3.plot(val_conf, 'g-o', label='Validation Confidence')
            ax3.set_title('Validation Confidence')
            ax3.set_xlabel('Validation Step')
            ax3.set_ylabel('Confidence')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Drift summary
        ax4.text(0.1, 0.8, f"Total Drift Alerts: {len(self.drift_alerts)}", fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f"Total Training Steps: {self.step_count:,}", fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.4, f"Average Confidence: {np.mean(list(self.confidence_scores)):.3f}", fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.2, f"Final Validation Loss: {self.validation_metrics[-1]['val_loss']:.4f}" if self.validation_metrics else "No validation data", fontsize=12, transform=ax4.transAxes)
        ax4.set_title('Training Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Monitoring plots saved to {save_dir}/")
        
    def generate_training_report(self, save_path: str = "training_report.json"):
        """Generate a comprehensive training report."""
        report = {
            "training_config": self.config,
            "training_summary": {
                "total_epochs": self.num_epochs,
                "total_steps": self.step_count,
                "total_training_time": sum(self.epoch_metrics['time']) if self.epoch_metrics['time'] else 0,
                "final_training_loss": self.epoch_metrics['loss'][-1] if self.epoch_metrics['loss'] else None,
                "final_validation_loss": self.validation_metrics[-1]['val_loss'] if self.validation_metrics else None
            },
            "drift_detection": {
                "total_drift_alerts": len(self.drift_alerts),
                "drift_alert_rate": len(self.drift_alerts) / self.step_count if self.step_count > 0 else 0,
                "drift_alerts": self.drift_alerts
            },
            "confidence_analysis": {
                "average_confidence": np.mean(list(self.confidence_scores)) if self.confidence_scores else 0.0,
                "confidence_std": np.std(list(self.confidence_scores)) if self.confidence_scores else 0.0,
                "min_confidence": np.min(list(self.confidence_scores)) if self.confidence_scores else 0.0,
                "max_confidence": np.max(list(self.confidence_scores)) if self.confidence_scores else 0.0,
                "confidence_trend": self.confidence_history
            },
            "performance_metrics": {
                "loss_history": self.loss_history[-1000:],  # Last 1000 steps
                "epoch_metrics": dict(self.epoch_metrics),
                "validation_metrics": self.validation_metrics
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Training report saved to {save_path}")
        
    def train(self):
        """Main training loop with enhanced monitoring."""
        logger.info("Starting enhanced training with monitoring...")
        training_start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)
            
        total_training_time = time.time() - training_start_time
        logger.info(f"Training completed in {total_training_time:.2f}s")
        logger.info(f"Total steps: {self.step_count:,}")
        logger.info(f"Total drift alerts: {len(self.drift_alerts)}")
        logger.info(f"Average confidence: {np.mean(list(self.confidence_scores)):.3f}")
        
        # Generate final reports and plots
        self.create_monitoring_plots()
        self.generate_training_report()
        
    def save_model(self, save_path: str):
        """Save the trained model with monitoring data."""
        logger.info(f"Saving enhanced model to {save_path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'step_count': self.step_count,
            'drift_detector_state': self.drift_detector.get_state(),
            'training_metrics': {
                'loss_history': self.loss_history,
                'epoch_metrics': dict(self.epoch_metrics),
                'validation_metrics': self.validation_metrics,
                'confidence_scores': list(self.confidence_scores),
                'drift_alerts': self.drift_alerts
            }
        }, save_path)
        logger.info("Enhanced model saved successfully")


def main():
    """Main function to parse arguments and launch enhanced training."""
    parser = argparse.ArgumentParser(description="Enhanced training with monitoring for Crickformer model.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/train_config.yaml",
        help="Path to the training config file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data",
        help="Path to the cricket data directory"
    )
    parser.add_argument(
        "--train-matches",
        type=str,
        help="Path to file containing training match IDs"
    )
    parser.add_argument(
        "--val-matches",
        type=str,
        help="Path to file containing validation match IDs"
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default="checkpoints/enhanced_crickformer.pth",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--use-csv",
        action="store_true",
        default=True,
        help="Use CSV data adapter (default: True)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create enhanced trainer
    trainer = EnhancedTrainer(config, device=args.device)
    
    # Setup components
    trainer.setup_model()
    trainer.setup_dataset(
        args.data_path,
        use_csv=args.use_csv,
        train_matches_path=args.train_matches,
        val_matches_path=args.val_matches
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(args.save_model)
    
    logger.info("✅ Enhanced training completed successfully!")


if __name__ == "__main__":
    main() 