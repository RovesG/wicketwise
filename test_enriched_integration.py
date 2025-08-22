#!/usr/bin/env python3
"""
Test script for enriched data integration with Knowledge Graph and ML model

Author: WicketWise Team, Last Modified: 2025-01-21
"""

import os
import sys
import json
import pandas as pd
import torch
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_enriched_data_matcher():
    """Test the enriched data matching system"""
    logger.info("🔍 Testing Enriched Data Matcher...")
    
    try:
        from enriched_data_matcher import EnrichedDataMatcher
        
        matcher = EnrichedDataMatcher()
        
        # Test paths
        enriched_data_path = "./enriched_data/enriched_betting_matches.json"
        kg_data_path = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/joined_ball_by_ball_data.csv"
        ml_data_path = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/decimal_data_v3.csv"
        
        if not os.path.exists(enriched_data_path):
            logger.warning(f"❌ Enriched data not found: {enriched_data_path}")
            return False
        
        # Run matching
        results = matcher.match_enriched_data_to_datasets(
            enriched_data_path, kg_data_path, ml_data_path
        )
        
        # Print results
        stats = results['statistics']
        logger.info(f"✅ Matching completed!")
        logger.info(f"📊 KG Match rates: Teams {stats['match_rates']['kg_teams']:.1f}%, Venues {stats['match_rates']['kg_venues']:.1f}%, Players {stats['match_rates']['kg_players']:.1f}%")
        logger.info(f"📊 ML Match rates: Teams {stats['match_rates']['ml_teams']:.1f}%, Venues {stats['match_rates']['ml_venues']:.1f}%, Players {stats['match_rates']['ml_players']:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enriched data matcher test failed: {e}")
        return False

def test_enhanced_kg_builder():
    """Test the enhanced Knowledge Graph builder with enrichment integration"""
    logger.info("🏗️ Testing Enhanced KG Builder...")
    
    try:
        from crickformers.gnn.unified_kg_builder import UnifiedKGBuilder
        
        # Test with enrichment data
        enriched_data_path = "./enriched_data/enriched_betting_matches.json"
        
        builder = UnifiedKGBuilder(
            data_dir="./models",
            enriched_data_path=enriched_data_path if os.path.exists(enriched_data_path) else None
        )
        
        # Test enrichment loading
        enrichments = builder._load_enrichment_data()
        logger.info(f"✅ Loaded {len(enrichments)} enrichments")
        
        # Test venue enrichment finding
        if enrichments:
            test_venue = "M Chinnaswamy Stadium"
            venue_enrichment = builder._find_venue_enrichment(test_venue)
            if venue_enrichment:
                logger.info(f"✅ Found enrichment for {test_venue}: {venue_enrichment.get('city', 'Unknown city')}")
            else:
                logger.info(f"⚠️ No enrichment found for {test_venue}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced KG builder test failed: {e}")
        return False

def test_weather_aware_gnn():
    """Test the weather-aware GNN"""
    logger.info("🌦️ Testing Weather-Aware GNN...")
    
    try:
        from crickformers.gnn.weather_aware_gnn import WeatherAwareGNN, create_weather_aware_gnn
        
        # Create test config
        config = {
            'player_feature_dim': 128,
            'venue_feature_dim': 64,
            'match_feature_dim': 96,
            'weather_dim': 64,
            'coord_dim': 32,
            'squad_dim': 48,
            'hidden_dim': 256,
            'output_dim': 128,
            'num_layers': 2,  # Reduced for testing
            'dropout': 0.1
        }
        
        # Create GNN
        gnn = create_weather_aware_gnn(config)
        
        # Test individual encoders
        batch_size = 16
        
        # Weather features: [temp, humidity, wind_speed, wind_dir, precip, precip_prob]
        weather_features = torch.randn(batch_size, 6)
        weather_emb = gnn.weather_encoder(weather_features)
        logger.info(f"✅ Weather encoder output shape: {weather_emb.shape}")
        
        # Venue coordinates: [lat, lon]
        venue_coordinates = torch.randn(batch_size, 2) * 90
        coord_emb = gnn.coord_encoder(venue_coordinates)
        logger.info(f"✅ Coordinate encoder output shape: {coord_emb.shape}")
        
        # Squad features: [role_id, batting_style_id, bowling_style_id]
        squad_features = torch.randint(0, 5, (batch_size, 3))
        squad_emb = gnn.squad_encoder(squad_features)
        logger.info(f"✅ Squad encoder output shape: {squad_emb.shape}")
        
        # Test weather advantage prediction
        weather_advantage = gnn.predict_weather_advantage(weather_features, venue_coordinates)
        logger.info(f"✅ Weather advantage shape: {weather_advantage.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Weather-aware GNN test failed: {e}")
        return False

def test_enhanced_static_context_encoder():
    """Test the enhanced static context encoder with weather and venue features"""
    logger.info("🎯 Testing Enhanced Static Context Encoder...")
    
    try:
        from crickformers.model.static_context_encoder import StaticContextEncoder
        
        # Create encoder with weather and venue support
        encoder = StaticContextEncoder(
            numeric_dim=15,
            categorical_vocab_sizes={'team': 20, 'player_role': 5},
            categorical_embedding_dims={'team': 8, 'player_role': 4},
            video_dim=99,
            weather_dim=6,
            venue_coord_dim=2,
            hidden_dims=[128, 64],
            context_dim=128
        )
        
        batch_size = 32
        
        # Standard features
        numeric_features = torch.randn(batch_size, 15)
        categorical_features = torch.randint(0, 20, (batch_size, 2))
        video_features = torch.randn(batch_size, 99)
        video_mask = torch.ones(batch_size, 1)
        
        # Enhanced features
        weather_features = torch.randn(batch_size, 6)
        venue_coordinates = torch.randn(batch_size, 2)
        
        # Test forward pass
        context_vector = encoder(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            video_features=video_features,
            video_mask=video_mask,
            weather_features=weather_features,
            venue_coordinates=venue_coordinates
        )
        
        logger.info(f"✅ Enhanced context encoder output shape: {context_vector.shape}")
        
        # Test without weather/venue features
        context_vector_basic = encoder(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            video_features=video_features,
            video_mask=video_mask
        )
        
        logger.info(f"✅ Basic context encoder output shape: {context_vector_basic.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced static context encoder test failed: {e}")
        return False

def test_data_integration_pipeline():
    """Test the complete data integration pipeline"""
    logger.info("🔗 Testing Complete Data Integration Pipeline...")
    
    try:
        # Check if we have the necessary data files
        enriched_data_path = "./enriched_data/enriched_betting_matches.json"
        kg_data_path = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/joined_ball_by_ball_data.csv"
        ml_data_path = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/decimal_data_v3.csv"
        
        missing_files = []
        for name, path in [("Enriched", enriched_data_path), ("KG", kg_data_path), ("ML", ml_data_path)]:
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            logger.warning(f"⚠️ Missing data files for full integration test:")
            for missing in missing_files:
                logger.warning(f"   - {missing}")
            return False
        
        # Load and analyze data
        with open(enriched_data_path, 'r') as f:
            enriched_data = json.load(f)
        
        kg_data = pd.read_csv(kg_data_path)
        ml_data = pd.read_csv(ml_data_path)
        
        logger.info(f"📊 Data loaded:")
        logger.info(f"   - Enriched matches: {len(enriched_data)}")
        logger.info(f"   - KG balls: {len(kg_data):,}")
        logger.info(f"   - ML betting data: {len(ml_data):,}")
        
        # Analyze enrichment coverage
        enriched_venues = set()
        enriched_teams = set()
        
        for match in enriched_data:
            if 'venue' in match:
                enriched_venues.add(match['venue'].get('name', ''))
            if 'teams' in match:
                for team in match['teams']:
                    enriched_teams.add(team.get('name', ''))
        
        kg_venues = set(kg_data['venue'].unique()) if 'venue' in kg_data.columns else set()
        ml_venues = set(ml_data['venue'].unique()) if 'venue' in ml_data.columns else set()
        
        venue_overlap_kg = len(enriched_venues.intersection(kg_venues))
        venue_overlap_ml = len(enriched_venues.intersection(ml_venues))
        
        logger.info(f"🏟️ Venue overlap:")
        logger.info(f"   - Enriched venues: {len(enriched_venues)}")
        logger.info(f"   - KG venue overlap: {venue_overlap_kg}/{len(kg_venues)} ({venue_overlap_kg/len(kg_venues)*100:.1f}%)" if kg_venues else "   - KG: No venue data")
        logger.info(f"   - ML venue overlap: {venue_overlap_ml}/{len(ml_venues)} ({venue_overlap_ml/len(ml_venues)*100:.1f}%)" if ml_venues else "   - ML: No venue data")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data integration pipeline test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    logger.info("🚀 Starting Enriched Data Integration Tests...")
    
    tests = [
        ("Enriched Data Matcher", test_enriched_data_matcher),
        ("Enhanced KG Builder", test_enhanced_kg_builder),
        ("Weather-Aware GNN", test_weather_aware_gnn),
        ("Enhanced Static Context Encoder", test_enhanced_static_context_encoder),
        ("Data Integration Pipeline", test_data_integration_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"🧪 Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{status}: {test_name}")
        except Exception as e:
            logger.error(f"❌ FAILED: {test_name} - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All integration tests passed! Enriched data integration is ready!")
    else:
        logger.warning("⚠️ Some tests failed. Please check the logs and fix issues before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
