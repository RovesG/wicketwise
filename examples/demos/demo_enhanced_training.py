# Purpose: Demo of enhanced training pipeline with drift detection, confidence estimation, and monitoring
# Author: Shamus Rae, Last Modified: 2024

"""
This demo showcases the complete enhanced training pipeline for cricket prediction,
featuring:
- Drift detection with real-time monitoring
- Confidence estimation using Monte Carlo dropout
- Comprehensive training metrics and visualizations
- Performance monitoring and alerting
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Import our enhanced components
from crickformers.enhanced_trainer import EnhancedTrainer
from crickformers.drift_detector import DriftDetector
from crickformers.confidence_utils import predict_with_uncertainty, calculate_confidence_score
from crickformers.train import load_config


def create_demo_config():
    """Create a demonstration configuration for enhanced training."""
    config = {
        "batch_size": 16,
        "num_epochs": 5,
        "learning_rate": 0.001,
        "log_interval": 50,
        "validation_interval": 100,
        "drift_threshold": 0.08,
        "drift_window_size": 200,
        "weight_decay": 0.01,
        "model": {
            "numeric_dim": 20,
            "categorical_vocab_sizes": {
                "player": 100,
                "team": 20,
                "venue": 50,
                "bowler_type": 10,
                "batsman_type": 8
            },
            "categorical_embedding_dims": {
                "player": 32,
                "team": 16,
                "venue": 24,
                "bowler_type": 8,
                "batsman_type": 8
            },
            "video_dim": 512,
            "sequence_length": 5,
            "hidden_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "context_dim": 128,
            "dropout_rate": 0.15,
            "gnn_config": {
                "num_layers": 3,
                "hidden_dim": 128,
                "num_heads": 4,
                "dropout_rate": 0.1
            },
            "enable_temporal_decay": True,
            "temporal_decay_factor": 0.05
        }
    }
    return config


def simulate_cricket_data_batch(batch_size=16):
    """Simulate a batch of cricket data for demonstration."""
    return {
        "inputs": {
            "numeric_features": torch.randn(batch_size, 20),
            "categorical_features": torch.randint(0, 50, (batch_size, 5)),
            "video_features": torch.randn(batch_size, 512),
            "video_mask": torch.ones(batch_size, 1),
            "sequence_features": torch.randn(batch_size, 5, 256),
            "sequence_mask": torch.ones(batch_size, 5)
        },
        "targets": {
            "win_prob": torch.rand(batch_size, 1),
            "next_ball_outcome": torch.randint(0, 10, (batch_size,)),
            "mispricing": torch.randn(batch_size, 1) * 0.1
        }
    }


class MockDataLoader:
    """Mock data loader for demonstration purposes."""
    
    def __init__(self, num_batches=50, batch_size=16):
        self.num_batches = num_batches
        self.batch_size = batch_size
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        for i in range(self.num_batches):
            # Simulate some data drift after batch 30
            if i > 30:
                # Introduce slight distribution shift
                batch = simulate_cricket_data_batch(self.batch_size)
                batch["inputs"]["numeric_features"] += torch.randn_like(batch["inputs"]["numeric_features"]) * 0.3
            else:
                batch = simulate_cricket_data_batch(self.batch_size)
            yield batch


def demonstrate_drift_detection():
    """Demonstrate drift detection capabilities."""
    print("=" * 60)
    print("🔍 DRIFT DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Create drift detector
    drift_detector = DriftDetector(
        feature_dim=128,
        threshold=0.1,
        window_size=100
    )
    
    # Simulate normal data
    print("📊 Processing normal data...")
    normal_data = []
    for i in range(150):
        features = torch.randn(16, 128)
        drift_detected = drift_detector.detect_drift(features)
        normal_data.append(drift_detector.get_last_drift_score())
        
        if drift_detected:
            print(f"   ⚠️  Drift detected at step {i+1} (score: {drift_detector.get_last_drift_score():.4f})")
    
    # Simulate data with drift
    print("\n📊 Processing data with drift...")
    drift_data = []
    for i in range(150, 250):
        # Introduce distribution shift
        features = torch.randn(16, 128) + torch.randn(16, 128) * 0.5
        drift_detected = drift_detector.detect_drift(features)
        drift_data.append(drift_detector.get_last_drift_score())
        
        if drift_detected:
            print(f"   🚨 Drift detected at step {i+1} (score: {drift_detector.get_last_drift_score():.4f})")
    
    # Plot drift scores
    fig, ax = plt.subplots(figsize=(12, 6))
    all_scores = normal_data + drift_data
    ax.plot(all_scores, label='Drift Score', color='blue', alpha=0.7)
    ax.axhline(y=0.1, color='red', linestyle='--', label='Threshold')
    ax.axvline(x=150, color='orange', linestyle='--', label='Distribution Shift')
    ax.set_xlabel('Step')
    ax.set_ylabel('Drift Score')
    ax.set_title('Drift Detection Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_drift_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Drift detection demonstration complete!")
    print(f"   📈 Total drift alerts: {len([s for s in all_scores if s > 0.1])}")
    print(f"   📊 Average drift score: {np.mean(all_scores):.4f}")


def demonstrate_confidence_estimation():
    """Demonstrate confidence estimation with Monte Carlo dropout."""
    print("\n" + "=" * 60)
    print("🎯 CONFIDENCE ESTIMATION DEMONSTRATION")
    print("=" * 60)
    
    # Create a simple model for demonstration
    from crickformers.model.crickformer_model import CrickformerModel
    
    config = create_demo_config()
    model_config = config["model"]
    
    # Create config dictionaries expected by CrickformerModel
    sequence_config = {
        "sequence_length": model_config["sequence_length"],
        "feature_dim": model_config["hidden_dim"],
        "num_layers": model_config["num_layers"],
        "num_heads": model_config["num_heads"],
        "dropout_rate": model_config["dropout_rate"]
    }
    
    static_config = {
        "numeric_dim": model_config["numeric_dim"],
        "categorical_vocab_sizes": model_config["categorical_vocab_sizes"],
        "categorical_embedding_dims": model_config["categorical_embedding_dims"],
        "video_dim": model_config["video_dim"],
        "hidden_dims": [model_config["hidden_dim"]],
        "context_dim": model_config["context_dim"],
        "dropout_rate": model_config["dropout_rate"]
    }
    
    fusion_config = {
        "sequence_dim": model_config["hidden_dim"],
        "context_dim": model_config["context_dim"],
        "gnn_dim": 128,
        "fusion_dim": model_config["hidden_dim"],
        "dropout_rate": model_config["dropout_rate"]
    }
    
    prediction_heads_config = {
        "win_probability": {
            "input_dim": model_config["hidden_dim"],
            "hidden_dims": [128, 64],
            "dropout_rate": model_config["dropout_rate"]
        },
        "next_ball_outcome": {
            "input_dim": model_config["hidden_dim"],
            "num_classes": 10,
            "hidden_dims": [128, 64],
            "dropout_rate": model_config["dropout_rate"]
        },
        "mispricing": {
            "input_dim": model_config["hidden_dim"],
            "hidden_dims": [128, 64],
            "dropout_rate": model_config["dropout_rate"]
        }
    }
    
    model = CrickformerModel(
        sequence_config=sequence_config,
        static_config=static_config,
        fusion_config=fusion_config,
        prediction_heads_config=prediction_heads_config,
        gnn_embedding_dim=128
    )
    
    # Test confidence estimation
    print("📊 Testing confidence estimation...")
    confidences = []
    predictions = []
    
    for i in range(10):
        # Generate test data
        batch = simulate_cricket_data_batch(8)
        
        # Get prediction with uncertainty
        mean_pred, std_pred, conf_interval = predict_with_uncertainty(
            model, batch["inputs"], n_samples=30
        )
        
        # Calculate confidence scores
        batch_confidences = []
        for j in range(len(mean_pred["win_prob"])):
            conf_score = calculate_confidence_score(
                mean_pred["win_prob"][j].item(),
                std_pred["win_prob"][j].item()
            )
            batch_confidences.append(conf_score)
        
        confidences.extend(batch_confidences)
        predictions.extend(mean_pred["win_prob"].detach().numpy())
        
        print(f"   Batch {i+1}: Avg confidence = {np.mean(batch_confidences):.3f}, "
              f"Avg prediction = {np.mean(mean_pred['win_prob'].detach().numpy()):.3f}")
    
    # Plot confidence analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confidence distribution
    ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Confidence Score Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Prediction vs Confidence
    ax2.scatter(predictions, confidences, alpha=0.6, color='orange')
    ax2.set_xlabel('Prediction (Win Probability)')
    ax2.set_ylabel('Confidence Score')
    ax2.set_title('Prediction vs Confidence')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_confidence_estimation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Confidence estimation demonstration complete!")
    print(f"   📊 Average confidence: {np.mean(confidences):.3f}")
    print(f"   📈 Confidence range: {np.min(confidences):.3f} - {np.max(confidences):.3f}")


def demonstrate_enhanced_training():
    """Demonstrate the complete enhanced training pipeline."""
    print("\n" + "=" * 60)
    print("🚀 ENHANCED TRAINING DEMONSTRATION")
    print("=" * 60)
    
    # Create configuration
    config = create_demo_config()
    
    # Initialize enhanced trainer
    trainer = EnhancedTrainer(config, device="cpu")
    
    # Setup model
    print("🔧 Setting up model...")
    trainer.setup_model()
    
    # Create mock data loaders
    trainer.train_loader = MockDataLoader(num_batches=100, batch_size=16)
    trainer.val_loader = MockDataLoader(num_batches=20, batch_size=16)
    
    print(f"✅ Model setup complete! Total parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    # Run training
    print("\n🎯 Starting enhanced training...")
    start_time = time.time()
    
    try:
        # Run a shortened training for demo
        trainer.num_epochs = 3
        trainer.train()
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.2f}s")
        
        # Display final metrics
        print("\n📊 FINAL TRAINING METRICS:")
        print(f"   Total steps: {trainer.step_count:,}")
        print(f"   Final loss: {trainer.loss_history[-1]:.4f}")
        print(f"   Drift alerts: {len(trainer.drift_alerts)}")
        print(f"   Average confidence: {np.mean(list(trainer.confidence_scores)):.3f}")
        
        # Show drift alerts if any
        if trainer.drift_alerts:
            print("\n🚨 DRIFT ALERTS:")
            for alert in trainer.drift_alerts:
                print(f"   Step {alert['step']}: Drift score = {alert['drift_score']:.4f}")
        
        # Display validation metrics
        if trainer.validation_metrics:
            final_val = trainer.validation_metrics[-1]
            print(f"\n🎯 FINAL VALIDATION METRICS:")
            print(f"   Validation loss: {final_val['val_loss']:.4f}")
            print(f"   Average confidence: {final_val['avg_confidence']:.3f}")
            print(f"   Confidence range: {final_val['min_confidence']:.3f} - {final_val['max_confidence']:.3f}")
    
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Save model and generate reports
    print("\n💾 Saving model and generating reports...")
    trainer.save_model("demo_enhanced_model.pth")
    
    print("\n✅ Enhanced training demonstration complete!")
    print(f"   📈 Check 'monitoring_plots/' for visualizations")
    print(f"   📋 Check 'training_report.json' for detailed metrics")


def create_comprehensive_demo():
    """Create a comprehensive demonstration of all features."""
    print("🎯 WICKETWISE ENHANCED TRAINING SYSTEM DEMO")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Feature 1: Drift Detection
    demonstrate_drift_detection()
    
    # Feature 2: Confidence Estimation
    demonstrate_confidence_estimation()
    
    # Feature 3: Enhanced Training Pipeline
    demonstrate_enhanced_training()
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("🎯 Features demonstrated:")
    print("   1. Real-time drift detection with alerting")
    print("   2. Monte Carlo dropout confidence estimation")
    print("   3. Comprehensive training monitoring")
    print("   4. Automated visualization and reporting")
    print("   5. Enhanced model checkpointing")
    print("\n🔧 Generated files:")
    print("   • demo_drift_detection.png - Drift detection visualization")
    print("   • demo_confidence_estimation.png - Confidence analysis")
    print("   • monitoring_plots/ - Training monitoring plots")
    print("   • training_report.json - Detailed training metrics")
    print("   • demo_enhanced_model.pth - Enhanced model checkpoint")
    print("\n🚀 Ready for production cricket prediction!")


def display_system_architecture():
    """Display the enhanced training system architecture."""
    print("\n" + "=" * 60)
    print("🏗️  ENHANCED TRAINING SYSTEM ARCHITECTURE")
    print("=" * 60)
    
    architecture = """
    ┌─────────────────────────────────────────────────────────────────┐
    │                    ENHANCED CRICKET TRAINING PIPELINE            │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
    │  │   DATA LOADER   │    │   CRICKFORMER   │    │   TRAINING   │ │
    │  │                 │    │     MODEL       │    │   METRICS    │ │
    │  │ • CSV Adapter   │ -> │ • Transformer   │ -> │ • Loss Track │ │
    │  │ • Match Split   │    │ • GNN Enhanced  │    │ • Validation │ │
    │  │ • Preprocessing │    │ • Temporal      │    │ • Confidence │ │
    │  └─────────────────┘    │   Decay         │    └──────────────┘ │
    │                         └─────────────────┘                     │
    │                                                                 │
    │  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
    │  │   DRIFT         │    │   CONFIDENCE    │    │   MONITORING │ │
    │  │   DETECTOR      │    │   ESTIMATOR     │    │   DASHBOARD  │ │
    │  │                 │    │                 │    │              │ │
    │  │ • Feature Drift │    │ • MC Dropout    │    │ • Real-time  │ │
    │  │ • Distribution  │    │ • Uncertainty   │    │ • Plots      │ │
    │  │ • Alerts        │    │ • Intervals     │    │ • Reports    │ │
    │  └─────────────────┘    └─────────────────┘    └──────────────┘ │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    print(architecture)
    
    components = {
        "Core Components": [
            "EnhancedTrainer - Main training orchestrator",
            "DriftDetector - Real-time drift monitoring",
            "ConfidenceUtils - Monte Carlo uncertainty estimation",
            "CrickformerModel - Enhanced transformer architecture"
        ],
        "Monitoring Features": [
            "Real-time loss tracking and visualization",
            "Confidence score distribution analysis",
            "Drift detection with alerting system",
            "Comprehensive training reports"
        ],
        "Data Pipeline": [
            "CSV data adapter for real cricket data",
            "Match-level train/validation splitting",
            "Temporal decay weighting for GNN",
            "Automated preprocessing and validation"
        ],
        "Output Systems": [
            "Automated plot generation",
            "JSON training reports",
            "Enhanced model checkpoints",
            "Performance benchmarking"
        ]
    }
    
    for category, items in components.items():
        print(f"\n📋 {category}:")
        for item in items:
            print(f"   • {item}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Display system architecture
    display_system_architecture()
    
    # Run comprehensive demo
    create_comprehensive_demo()
    
    print(f"\n🎯 Demo completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🚀 Enhanced training system ready for production use!") 