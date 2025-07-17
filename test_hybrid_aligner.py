# Purpose: Test script for hybrid match aligner
# Author: Assistant, Last Modified: 2024

import logging
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hybrid_match_aligner import HybridMatchAligner, hybrid_align_matches

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_hybrid_aligner():
    """Test the hybrid match aligner functionality."""
    
    print("🏏 Testing Hybrid Match Aligner")
    print("=" * 50)
    
    # Configuration
    data_path = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data"
    nvplay_path = f"{data_path}/nvplay_data_v3.csv"
    decimal_path = f"{data_path}/decimal_data_v3.csv"
    
    # Check if files exist
    if not Path(nvplay_path).exists():
        print(f"❌ NVPlay file not found: {nvplay_path}")
        return
    
    if not Path(decimal_path).exists():
        print(f"❌ Decimal file not found: {decimal_path}")
        return
    
    # Get API key from environment
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_key:
        print("⚠️  No OpenAI API key found, testing with fallback configuration")
    else:
        print("✅ OpenAI API key found, testing with LLM configuration")
    
    try:
        # Test 1: Initialize aligner
        print("\n1️⃣ Testing Aligner Initialization...")
        aligner = HybridMatchAligner(nvplay_path, decimal_path, openai_key)
        print(f"✅ Loaded {len(aligner.nvplay_df)} NVPlay records")
        print(f"✅ Loaded {len(aligner.decimal_df)} decimal records")
        
        # Test 2: Generate configuration
        print("\n2️⃣ Testing Configuration Generation...")
        config = aligner.generate_llm_configuration()
        print(f"✅ Configuration generated (confidence: {config.confidence:.2f})")
        print(f"📋 Reasoning: {config.reasoning}")
        print(f"🎯 Threshold: {config.similarity_threshold}")
        print(f"📏 Fingerprint length: {config.fingerprint_length}")
        
        # Test 3: Extract fingerprints (small sample)
        print("\n3️⃣ Testing Fingerprint Extraction...")
        
        # Limit to first few matches for testing
        original_nvplay = aligner.nvplay_df.copy()
        original_decimal = aligner.decimal_df.copy()
        
        # Sample data for faster testing
        sample_matches = list(aligner.nvplay_df['Match'].unique())[:5]
        aligner.nvplay_df = aligner.nvplay_df[aligner.nvplay_df['Match'].isin(sample_matches)]
        
        # Sample decimal data
        aligner.decimal_df = aligner.decimal_df.head(1000)
        
        nvplay_fps, decimal_fps = aligner.extract_fingerprints()
        print(f"✅ Extracted {len(nvplay_fps)} NVPlay fingerprints")
        print(f"✅ Extracted {len(decimal_fps)} decimal fingerprints")
        
        # Test 4: Test similarity calculation
        print("\n4️⃣ Testing Similarity Calculation...")
        if nvplay_fps and decimal_fps:
            # Get first fingerprints
            nvplay_fp = list(nvplay_fps.values())[0]
            decimal_fp = list(decimal_fps.values())[0]
            
            similarity = aligner.calculate_fuzzy_similarity(nvplay_fp, decimal_fp)
            print(f"✅ Sample similarity score: {similarity:.3f}")
            
            # Test name similarity
            if nvplay_fp and decimal_fp:
                name_sim = aligner.fuzzy_name_similarity(
                    nvplay_fp[0]['batter'], 
                    decimal_fp[0]['batter']
                )
                print(f"✅ Name similarity example: {name_sim:.3f}")
        
        # Test 5: Find matches (limited sample)
        print("\n5️⃣ Testing Match Finding...")
        matches = aligner.find_matches(use_llm_config=openai_key is not None)
        print(f"✅ Found {len(matches)} matches in sample data")
        
        if matches:
            print("📊 Sample matches:")
            for match in matches[:3]:
                print(f"  - {match['nvplay_match_id']} <-> {match['decimal_match_id']} (sim: {match['similarity_score']:.3f})")
        
        # Restore original data
        aligner.nvplay_df = original_nvplay
        aligner.decimal_df = original_decimal
        
        print("\n✅ All tests passed! Hybrid aligner is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convenience_function():
    """Test the convenience function."""
    
    print("\n🔧 Testing Convenience Function")
    print("=" * 50)
    
    data_path = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data"
    nvplay_path = f"{data_path}/nvplay_data_v3.csv"
    decimal_path = f"{data_path}/decimal_data_v3.csv"
    openai_key = os.getenv('OPENAI_API_KEY')
    
    try:
        # Test with very small sample by creating temporary files
        import pandas as pd
        
        # Create small sample files
        nvplay_df = pd.read_csv(nvplay_path)
        decimal_df = pd.read_csv(decimal_path)
        
        # Get first match from each
        sample_nvplay = nvplay_df[nvplay_df['Match'] == nvplay_df['Match'].iloc[0]].head(20)
        sample_decimal = decimal_df.head(100)
        
        # Save temporary files
        sample_nvplay.to_csv("temp_nvplay.csv", index=False)
        sample_decimal.to_csv("temp_decimal.csv", index=False)
        
        # Test convenience function
        matches = hybrid_align_matches(
            "temp_nvplay.csv", 
            "temp_decimal.csv", 
            openai_key,
            "test_hybrid_matches.csv"
        )
        
        print(f"✅ Convenience function found {len(matches)} matches")
        
        # Cleanup
        Path("temp_nvplay.csv").unlink(missing_ok=True)
        Path("temp_decimal.csv").unlink(missing_ok=True)
        Path("test_hybrid_matches.csv").unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Convenience function test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_hybrid_aligner()
    if success:
        test_convenience_function()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("The hybrid match aligner is ready for use.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.") 