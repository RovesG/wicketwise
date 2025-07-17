# Purpose: Demonstration of hybrid match aligner with real cricket data
# Author: Assistant, Last Modified: 2024

import os
import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hybrid_match_aligner import HybridMatchAligner, hybrid_align_matches

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """Demonstrate the hybrid match aligner with real cricket data."""
    
    print("🏏 WicketWise Hybrid Match Aligner Demo")
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
    
    # Get API key
    openai_key = os.getenv('OPENAI_API_KEY')
    
    print(f"📊 Data files found")
    print(f"🔑 OpenAI API key: {'✅ Available' if openai_key else '❌ Not found'}")
    
    # Demo 1: Show configuration differences
    print("\n1️⃣ CONFIGURATION COMPARISON")
    print("-" * 30)
    
    # Test fallback configuration
    print("🔧 Fallback Configuration:")
    aligner_fallback = HybridMatchAligner(nvplay_path, decimal_path, None)
    fallback_config = aligner_fallback.generate_llm_configuration()
    print(f"   Threshold: {fallback_config.similarity_threshold}")
    print(f"   Fingerprint length: {fallback_config.fingerprint_length}")
    print(f"   Confidence: {fallback_config.confidence}")
    print(f"   Column mapping: {fallback_config.column_mapping['decimal']['batter']}")
    
    if openai_key:
        print("\n🤖 LLM Configuration:")
        aligner_llm = HybridMatchAligner(nvplay_path, decimal_path, openai_key)
        llm_config = aligner_llm.generate_llm_configuration()
        print(f"   Threshold: {llm_config.similarity_threshold}")
        print(f"   Fingerprint length: {llm_config.fingerprint_length}")
        print(f"   Confidence: {llm_config.confidence}")
        print(f"   Column mapping: {llm_config.column_mapping['decimal']['batter']}")
        print(f"   Reasoning: {llm_config.reasoning[:100]}...")
    
    # Demo 2: Small-scale matching test
    print("\n2️⃣ SMALL-SCALE MATCHING TEST")
    print("-" * 30)
    
    # Use convenience function with limited data
    print("🔍 Testing with sample data...")
    
    try:
        # Create sample files for testing
        import pandas as pd
        
        nvplay_df = pd.read_csv(nvplay_path)
        decimal_df = pd.read_csv(decimal_path)
        
        # Get Big Bash League matches only
        bbl_nvplay = nvplay_df[nvplay_df['Competition'] == 'Big Bash League']
        bbl_decimal = decimal_df[decimal_df['competition'] == 'Big Bash League']
        
        # Sample first few matches
        sample_matches = list(bbl_nvplay['Match'].unique())[:3]
        sample_nvplay = bbl_nvplay[bbl_nvplay['Match'].isin(sample_matches)]
        sample_decimal = bbl_decimal.head(500)  # First 500 balls
        
        print(f"📊 Sample data: {len(sample_nvplay)} NVPlay balls, {len(sample_decimal)} decimal balls")
        
        # Save sample files
        sample_nvplay.to_csv("sample_nvplay.csv", index=False)
        sample_decimal.to_csv("sample_decimal.csv", index=False)
        
        # Test with fallback configuration
        print("\n🔧 Testing with fallback configuration...")
        matches_fallback = hybrid_align_matches(
            "sample_nvplay.csv",
            "sample_decimal.csv", 
            None,  # No API key
            "matches_fallback.csv"
        )
        print(f"✅ Fallback found: {len(matches_fallback)} matches")
        
        # Test with LLM configuration (if available)
        if openai_key:
            print("\n🤖 Testing with LLM configuration...")
            matches_llm = hybrid_align_matches(
                "sample_nvplay.csv",
                "sample_decimal.csv",
                openai_key,
                "matches_llm.csv"
            )
            print(f"✅ LLM found: {len(matches_llm)} matches")
            
            # Compare results
            if matches_llm and matches_fallback:
                print(f"📈 Improvement: {len(matches_llm) - len(matches_fallback)} additional matches")
        
        # Show sample matches
        if matches_fallback:
            print("\n📋 Sample matches found:")
            for i, match in enumerate(matches_fallback[:3]):
                print(f"   {i+1}. {match['nvplay_match_id']} <-> {match['decimal_match_id']}")
                print(f"      Similarity: {match['similarity_score']:.3f} ({match['confidence']})")
        
        # Cleanup
        Path("sample_nvplay.csv").unlink(missing_ok=True)
        Path("sample_decimal.csv").unlink(missing_ok=True)
        Path("matches_fallback.csv").unlink(missing_ok=True)
        Path("matches_llm.csv").unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Sample test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Demo 3: Usage recommendations
    print("\n3️⃣ USAGE RECOMMENDATIONS")
    print("-" * 30)
    
    print("🎯 For best results:")
    print("   1. Use OpenAI API key for intelligent configuration")
    print("   2. Start with Big Bash League data (most overlap)")
    print("   3. Monitor similarity scores (>0.8 = high confidence)")
    print("   4. Validate matches manually for critical applications")
    
    print("\n💡 Integration options:")
    print("   1. UI Launcher: Use 'Hybrid LLM-Enhanced Alignment' checkbox")
    print("   2. Command line: python3 hybrid_match_aligner.py nvplay.csv decimal.csv --api-key YOUR_KEY")
    print("   3. Python API: from hybrid_match_aligner import hybrid_align_matches")
    
    print("\n🔧 Configuration tips:")
    if openai_key:
        print("   ✅ LLM configuration will analyze your data and suggest optimal settings")
        print("   ✅ One-time cost (~$0.02) for intelligent configuration")
        print("   ✅ Automatic column mapping and similarity tuning")
    else:
        print("   ⚠️  Add OPENAI_API_KEY to environment for enhanced features")
        print("   ⚠️  Fallback configuration may need manual tuning")
        print("   ⚠️  Consider testing with sample data first")
    
    print("\n🎉 Hybrid Match Aligner Demo Complete!")
    print("Ready to integrate with your cricket analytics workflow.")

if __name__ == "__main__":
    main() 