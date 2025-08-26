#!/usr/bin/env python3
"""
Comprehensive Dimension Analysis for Crickformer Model
Checks all layer dimensions for mismatches before training
"""

import yaml
import torch
from pathlib import Path

def analyze_dimensions():
    """Analyze all model dimensions for potential mismatches"""
    
    print("🔍 CRICKFORMER DIMENSION ANALYSIS")
    print("=" * 50)
    
    # Load config
    config_path = Path("config/train_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    print("\n📋 CONFIGURATION SUMMARY:")
    print("-" * 30)
    
    # 1. Sequence Encoder Analysis
    seq_config = model_config['sequence_encoder']
    print(f"🔄 SEQUENCE ENCODER:")
    print(f"   - feature_dim: {seq_config['feature_dim']}")
    print(f"   - nhead: {seq_config['nhead']}")
    print(f"   - dims per head: {seq_config['feature_dim'] // seq_config['nhead']}")
    
    if seq_config['feature_dim'] % seq_config['nhead'] != 0:
        print(f"   ❌ ERROR: feature_dim ({seq_config['feature_dim']}) not divisible by nhead ({seq_config['nhead']})")
    else:
        print(f"   ✅ OK: Attention heads properly configured")
    
    # 2. Static Context Encoder Analysis
    static_config = model_config['static_context_encoder']
    print(f"\n🏗️  STATIC CONTEXT ENCODER:")
    print(f"   - numeric_dim: {static_config['numeric_dim']}")
    print(f"   - video_dim: {static_config['video_dim']}")
    print(f"   - weather_dim: {static_config['weather_dim']}")
    print(f"   - venue_coord_dim: {static_config['venue_coord_dim']}")
    print(f"   - context_dim (output): {static_config['context_dim']}")
    
    # Calculate total categorical embedding size
    cat_embed_total = sum(static_config['categorical_embedding_dims'].values())
    print(f"   - categorical embeddings total: {cat_embed_total}")
    
    # Calculate expected input dimension to MLP (including encoded weather/venue)
    weather_encoded_dim = static_config['weather_dim'] if static_config['weather_dim'] > 0 else 0
    venue_encoded_dim = static_config['venue_coord_dim'] * 2 if static_config['venue_coord_dim'] > 0 else 0
    
    expected_mlp_input = (
        static_config['numeric_dim'] + 
        cat_embed_total + 
        static_config['video_dim'] + 
        weather_encoded_dim + 
        venue_encoded_dim
    )
    print(f"   - weather encoded dim: {weather_encoded_dim}")
    print(f"   - venue encoded dim: {venue_encoded_dim}")
    print(f"   - expected MLP input dim: {expected_mlp_input}")
    
    # 3. Fusion Layer Analysis
    fusion_config = model_config['fusion_layer']
    print(f"\n🔗 FUSION LAYER:")
    print(f"   - sequence_dim (input): {fusion_config['sequence_dim']}")
    print(f"   - context_dim (input): {fusion_config['context_dim']}")
    print(f"   - kg_dim (input): {fusion_config['kg_dim']}")
    print(f"   - latent_dim (output): {fusion_config['latent_dim']}")
    
    fusion_input_dim = fusion_config['sequence_dim'] + fusion_config['context_dim'] + fusion_config['kg_dim']
    print(f"   - total input dim: {fusion_input_dim}")
    
    # Check dimension compatibility
    print(f"\n🔍 DIMENSION COMPATIBILITY CHECK:")
    print("-" * 40)
    
    # Check sequence encoder output matches fusion input
    if seq_config['feature_dim'] != fusion_config['sequence_dim']:
        print(f"❌ MISMATCH: Sequence encoder output ({seq_config['feature_dim']}) != Fusion sequence input ({fusion_config['sequence_dim']})")
    else:
        print(f"✅ OK: Sequence encoder → Fusion layer")
    
    # Check static context encoder output matches fusion input
    if static_config['context_dim'] != fusion_config['context_dim']:
        print(f"❌ MISMATCH: Static encoder output ({static_config['context_dim']}) != Fusion context input ({fusion_config['context_dim']})")
    else:
        print(f"✅ OK: Static context encoder → Fusion layer")
    
    # 4. Prediction Heads Analysis
    pred_config = model_config['prediction_heads']
    print(f"\n🎯 PREDICTION HEADS:")
    print(f"   - win_probability input: {pred_config['win_probability']['latent_dim']}")
    print(f"   - next_ball_outcome input: {pred_config['next_ball_outcome']['latent_dim']}")
    print(f"   - odds_mispricing input: {pred_config['odds_mispricing']['latent_dim']}")
    
    # Check fusion output matches prediction head inputs
    for head_name, head_config in pred_config.items():
        if head_config['latent_dim'] != fusion_config['latent_dim']:
            print(f"❌ MISMATCH: Fusion output ({fusion_config['latent_dim']}) != {head_name} input ({head_config['latent_dim']})")
        else:
            print(f"✅ OK: Fusion layer → {head_name}")
    
    print(f"\n📊 SUMMARY:")
    print("-" * 20)
    print(f"Data Flow: Ball History ({seq_config['feature_dim']}) + Static Context ({static_config['context_dim']}) + KG ({fusion_config['kg_dim']}) → Fusion ({fusion_input_dim}) → Latent ({fusion_config['latent_dim']}) → Predictions")
    
    return {
        'sequence_encoder': seq_config,
        'static_context_encoder': static_config,
        'fusion_layer': fusion_config,
        'prediction_heads': pred_config,
        'expected_mlp_input': expected_mlp_input
    }

if __name__ == "__main__":
    analysis = analyze_dimensions()
