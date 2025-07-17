# Purpose: Comprehensive test to find optimal similarity threshold and locate matches
# Author: Assistant, Last Modified: 2024

import sys
sys.path.append('.')
import pandas as pd
from hybrid_match_aligner import HybridMatchAligner
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
data_path = '/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data'
nvplay_path = f'{data_path}/nvplay_data_v3.csv'
decimal_path = f'{data_path}/decimal_data_v3.csv'

print("🎯 COMPREHENSIVE MATCH FINDER TEST")
print("=" * 60)

# Create aligner
aligner = HybridMatchAligner(nvplay_path, decimal_path, None)

# Force fallback configuration with lower threshold
aligner.openai_api_key = None
config = aligner.generate_llm_configuration()
config.similarity_threshold = 0.4  # Lower threshold for testing
config.fingerprint_length = 15  # Shorter fingerprint for faster testing
aligner.config = config

print(f"Using threshold: {config.similarity_threshold}")
print(f"Fingerprint length: {config.fingerprint_length}")

# Test with different sample sizes
sample_sizes = [50, 100, 200, 500]

for sample_size in sample_sizes:
    print(f"\n📊 Testing with sample size: {sample_size}")
    
    # Create sample data
    nvplay_sample = aligner.nvplay_df.head(sample_size)
    decimal_sample = aligner.decimal_df.head(sample_size * 2)  # More decimal data
    
    # Temporarily set sample data
    original_nvplay = aligner.nvplay_df
    original_decimal = aligner.decimal_df
    
    aligner.nvplay_df = nvplay_sample
    aligner.decimal_df = decimal_sample
    
    # Extract fingerprints
    nvplay_fps, decimal_fps = aligner.extract_fingerprints()
    print(f"  NVPlay fingerprints: {len(nvplay_fps)}")
    print(f"  Decimal fingerprints: {len(decimal_fps)}")
    
    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for threshold in thresholds:
        matches = []
        similarities = []
        
        # Test a subset of combinations (first 10 vs first 20)
        nvplay_items = list(nvplay_fps.items())[:10]
        decimal_items = list(decimal_fps.items())[:20]
        
        for nvplay_id, nvplay_fp in nvplay_items:
            for decimal_id, decimal_fp in decimal_items:
                similarity = aligner.calculate_fuzzy_similarity(nvplay_fp, decimal_fp)
                similarities.append(similarity)
                
                if similarity >= threshold:
                    matches.append({
                        'nvplay_id': nvplay_id,
                        'decimal_id': decimal_id,
                        'similarity': similarity
                    })
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        max_similarity = max(similarities) if similarities else 0
        
        print(f"    Threshold {threshold}: {len(matches)} matches, avg_sim: {avg_similarity:.3f}, max_sim: {max_similarity:.3f}")
        
        # Show top matches
        if matches:
            top_matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)[:3]
            for match in top_matches:
                print(f"      {match['similarity']:.3f}: {match['nvplay_id']} <-> {match['decimal_id']}")
    
    # Restore original data
    aligner.nvplay_df = original_nvplay
    aligner.decimal_df = original_decimal

print(f"\n🏆 FINDING BEST MATCHES ACROSS FULL DATASET")
print("Testing with optimized threshold...")

# Set optimized threshold based on results
optimized_threshold = 0.4
config.similarity_threshold = optimized_threshold
aligner.config = config

# Test with larger sample
large_sample_size = 1000
nvplay_large = aligner.nvplay_df.head(large_sample_size)
decimal_large = aligner.decimal_df.head(large_sample_size)

aligner.nvplay_df = nvplay_large
aligner.decimal_df = decimal_large

print(f"Large sample testing: {len(nvplay_large)} NVPlay, {len(decimal_large)} decimal rows")

# Find matches
matches = aligner.find_matches(use_llm_config=False)
print(f"Found {len(matches)} matches with threshold {optimized_threshold}")

# Show results
for i, match in enumerate(matches[:10]):  # Show first 10
    print(f"{i+1}. {match['similarity_score']:.3f}: {match['nvplay_match_id']} <-> {match['decimal_match_id']}")

# Calculate statistics
if matches:
    similarities = [m['similarity_score'] for m in matches]
    print(f"\nMatch Statistics:")
    print(f"  Total matches: {len(matches)}")
    print(f"  Average similarity: {sum(similarities)/len(similarities):.3f}")
    print(f"  Min similarity: {min(similarities):.3f}")
    print(f"  Max similarity: {max(similarities):.3f}")
    print(f"  High confidence matches (>0.8): {sum(1 for s in similarities if s > 0.8)}")
    print(f"  Medium confidence matches (0.6-0.8): {sum(1 for s in similarities if 0.6 <= s <= 0.8)}")
    print(f"  Low confidence matches (<0.6): {sum(1 for s in similarities if s < 0.6)}")

print(f"\n✅ Comprehensive test complete") 