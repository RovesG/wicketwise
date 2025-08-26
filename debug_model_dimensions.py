#!/usr/bin/env python3
"""
Debug Model Dimensions - Test actual tensor shapes during forward pass
"""

import torch
import yaml
from pathlib import Path
import sys
sys.path.append('.')

from crickformers.model.crickformer_model import CrickformerModel

def debug_model_dimensions():
    """Debug actual tensor dimensions during model forward pass"""
    
    print("🔧 DEBUGGING MODEL DIMENSIONS")
    print("=" * 40)
    
    # Load config
    config_path = Path("config/train_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    try:
        # Create model
        print("📦 Creating Crickformer model...")
        model = CrickformerModel(
            sequence_config=model_config['sequence_encoder'],
            static_config=model_config['static_context_encoder'],
            fusion_config=model_config['fusion_layer'],
            prediction_heads_config=model_config['prediction_heads']
        )
        model.eval()
        
        print("✅ Model created successfully")
        
        # Create test inputs matching our dataset format
        batch_size = 2
        print(f"\n🧪 Creating test inputs (batch_size={batch_size})...")
        
        test_inputs = {
            'recent_ball_history': torch.randn(batch_size, 5, 64),  # [batch, seq, features]
            'numeric_features': torch.randn(batch_size, 15),        # 15 numeric features
            'categorical_features': torch.randint(0, 10, (batch_size, 4)),  # 4 categorical features
            'video_features': torch.randn(batch_size, 99),          # 99 video features
            'video_mask': torch.ones(batch_size, 1),                # Video mask
            'gnn_embeddings': torch.randn(batch_size, 1, 320),      # GNN embeddings
            'weather_features': torch.randn(batch_size, 6),         # 6 weather features
            'venue_coordinates': torch.randn(batch_size, 2),        # 2 venue coordinates
            'market_odds': torch.randn(batch_size, 3),              # 3 market odds
        }
        
        print("📊 Input tensor shapes:")
        for key, tensor in test_inputs.items():
            print(f"   {key}: {tensor.shape}")
        
        # Test forward pass
        print(f"\n🚀 Testing forward pass...")
        
        with torch.no_grad():
            outputs = model(test_inputs)
        
        print("✅ Forward pass successful!")
        print("📊 Output tensor shapes:")
        for key, tensor in outputs.items():
            print(f"   {key}: {tensor.shape}")
            
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        import traceback
        print("📋 Full traceback:")
        traceback.print_exc()
        
        # Try to identify the exact layer where it fails
        print(f"\n🔍 Analyzing error location...")
        error_str = str(e)
        if "dimension" in error_str.lower():
            print(f"   Dimension mismatch detected: {error_str}")
        if "embedding" in error_str.lower():
            print(f"   Embedding dimension issue: {error_str}")

if __name__ == "__main__":
    debug_model_dimensions()
