# Purpose: Focused test of hybrid aligner with specific matches
# Author: Assistant, Last Modified: 2024

import sys
sys.path.append('.')
import pandas as pd
from hybrid_match_aligner import HybridMatchAligner

# Configuration
data_path = '/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data'
nvplay_path = f'{data_path}/nvplay_data_v3.csv'
decimal_path = f'{data_path}/decimal_data_v3.csv'

print("🎯 FOCUSED HYBRID ALIGNER TEST")
print("=" * 50)

# Load data
nvplay_df = pd.read_csv(nvplay_path)
decimal_df = pd.read_csv(decimal_path)

# Focus on Big Bash League and specific date
target_date = '2021-12-05'
nvplay_bbl = nvplay_df[
    (nvplay_df['Competition'] == 'Big Bash League') & 
    (nvplay_df['Date'] == target_date)
]
decimal_bbl = decimal_df[
    (decimal_df['competition'] == 'Big Bash League') & 
    (decimal_df['date'] == target_date)
]

print(f"Target date: {target_date}")
print(f"NVPlay data: {len(nvplay_bbl)} rows")
print(f"Decimal data: {len(decimal_bbl)} rows")

# Show match details
print("\nNVPlay match:")
nvplay_match = nvplay_bbl['Match'].iloc[0]
print(f"  {nvplay_match}")

print("\nDecimal match:")
decimal_match = decimal_bbl.groupby(['home', 'away']).first().reset_index()
for _, row in decimal_match.iterrows():
    print(f"  {row['home']} vs {row['away']}")

# Save focused datasets
nvplay_bbl.to_csv('focused_nvplay.csv', index=False)
decimal_bbl.to_csv('focused_decimal.csv', index=False)

# Test aligner with focused data
print("\n🔍 Testing with focused data...")
aligner = HybridMatchAligner('focused_nvplay.csv', 'focused_decimal.csv', None)

# Force fallback configuration
print("\nForcing fallback configuration...")
aligner.openai_api_key = None
config = aligner.generate_llm_configuration()
print(f"Configuration: {config.reasoning}")
print(f"Threshold: {config.similarity_threshold}")

# Test fingerprint extraction
print("\n📊 Extracting fingerprints...")
nvplay_fps, decimal_fps = aligner.extract_fingerprints()
print(f"NVPlay fingerprints: {len(nvplay_fps)}")
print(f"Decimal fingerprints: {len(decimal_fps)}")

if nvplay_fps and decimal_fps:
    nvplay_fp = list(nvplay_fps.values())[0]
    decimal_fp = list(decimal_fps.values())[0]
    
    print(f"\nNVPlay fingerprint (first 5 balls):")
    for i, ball in enumerate(nvplay_fp[:5]):
        print(f"  {i+1}. Over {ball['over']}.{ball['ball']}: {ball['batter']} vs {ball['bowler']}, {ball['runs']} runs")
    
    print(f"\nDecimal fingerprint (first 5 balls):")
    for i, ball in enumerate(decimal_fp[:5]):
        print(f"  {i+1}. Over {ball['over']}.{ball['ball']}: {ball['batter']} vs {ball['bowler']}, {ball['runs']} runs")
    
    # Test similarity
    print(f"\n🎯 Testing similarity...")
    similarity = aligner.calculate_fuzzy_similarity(nvplay_fp, decimal_fp)
    print(f"Similarity score: {similarity:.3f}")
    
    # Test individual components
    if len(nvplay_fp) > 0 and len(decimal_fp) > 0:
        ball1, ball2 = nvplay_fp[0], decimal_fp[0]
        
        # Test the enhanced name matching
        print(f"\n🔍 ENHANCED NAME MATCHING DEBUG:")
        print(f"Testing 'JR Philippe' vs 'Josh Philippe':")
        
        # Test the new cricket name similarity
        cricket_sim = aligner._calculate_cricket_name_similarity('JR Philippe', 'Josh Philippe')
        print(f"  Cricket-specific similarity: {cricket_sim:.3f}")
        
        # Test name parsing
        parts1 = aligner._parse_cricket_name('JR Philippe')
        parts2 = aligner._parse_cricket_name('Josh Philippe')
        print(f"  Parsed 'JR Philippe': {parts1}")
        print(f"  Parsed 'Josh Philippe': {parts2}")
        
        # Test initial matching
        initial_match = aligner._match_initials_to_names(parts1, parts2)
        print(f"  Initial matching score: {initial_match:.3f}")
        
        # Test bowler names too
        print(f"\nTesting 'SM Elliott' vs 'Sam Elliott':")
        bowler_cricket_sim = aligner._calculate_cricket_name_similarity('SM Elliott', 'Sam Elliott')
        print(f"  Bowler cricket-specific similarity: {bowler_cricket_sim:.3f}")
        
        batter_sim = aligner.fuzzy_name_similarity(ball1['batter'], ball2['batter'])
        bowler_sim = aligner.fuzzy_name_similarity(ball1['bowler'], ball2['bowler'])
        
        print(f"\nFinal similarity scores:")
        print(f"Batter similarity: {batter_sim:.3f} ('{ball1['batter']}' vs '{ball2['batter']}')")
        print(f"Bowler similarity: {bowler_sim:.3f} ('{ball1['bowler']}' vs '{ball2['bowler']}')")
        print(f"Runs match: {ball1['runs'] == ball2['runs']} ({ball1['runs']} vs {ball2['runs']})")
        print(f"Over match: {ball1['over'] == ball2['over']} ({ball1['over']} vs {ball2['over']})")

# Test full matching
print(f"\n🏏 Testing full matching...")
matches = aligner.find_matches(use_llm_config=False)
print(f"Found {len(matches)} matches")

if matches:
    for match in matches:
        print(f"  Match: {match['nvplay_match_id']} <-> {match['decimal_match_id']}")
        print(f"  Similarity: {match['similarity_score']:.3f} ({match['confidence']})")

# Test different thresholds
print(f"\n🎯 THRESHOLD TESTING:")
test_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

for threshold in test_thresholds:
    test_matches = []
    if nvplay_fps and decimal_fps:
        for nvplay_id, nvplay_fp in nvplay_fps.items():
            for decimal_id, decimal_fp in decimal_fps.items():
                similarity = aligner.calculate_fuzzy_similarity(nvplay_fp, decimal_fp)
                if similarity >= threshold:
                    test_matches.append({
                        'nvplay_match_id': nvplay_id,
                        'decimal_match_id': decimal_id,
                        'similarity_score': similarity
                    })
    
    print(f"  Threshold {threshold:.2f}: {len(test_matches)} matches")
    if test_matches:
        print(f"    Best match: {test_matches[0]['similarity_score']:.3f}")

# Test with larger dataset sample
print(f"\n📊 LARGE DATASET TESTING:")
print("Testing with first 1000 rows from each dataset...")

# Restore original data and test with larger sample
aligner.nvplay_df = aligner.nvplay_df.head(1000)
aligner.decimal_df = aligner.decimal_df.head(1000)

larger_nvplay_fps, larger_decimal_fps = aligner.extract_fingerprints()
print(f"Larger sample: {len(larger_nvplay_fps)} NVPlay, {len(larger_decimal_fps)} decimal fingerprints")

# Test with lower threshold
if larger_nvplay_fps and larger_decimal_fps:
    sample_matches = []
    for nvplay_id, nvplay_fp in list(larger_nvplay_fps.items())[:10]:  # Test first 10
        for decimal_id, decimal_fp in list(larger_decimal_fps.items())[:20]:  # Against first 20
            similarity = aligner.calculate_fuzzy_similarity(nvplay_fp, decimal_fp)
            if similarity >= 0.5:  # Lower threshold
                sample_matches.append({
                    'nvplay_match_id': nvplay_id,
                    'decimal_match_id': decimal_id,
                    'similarity_score': similarity
                })
    
    print(f"Found {len(sample_matches)} matches with threshold 0.5")
    for match in sample_matches[:5]:  # Show first 5
        print(f"  {match['similarity_score']:.3f}: {match['nvplay_match_id']} <-> {match['decimal_match_id']}")

print(f"\n✅ Focused test complete")

# Cleanup
import os
os.remove('focused_nvplay.csv')
os.remove('focused_decimal.csv') 