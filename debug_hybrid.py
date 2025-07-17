# Purpose: Debug script for hybrid match aligner
# Author: Assistant, Last Modified: 2024

import sys
sys.path.append('.')
import pandas as pd
from hybrid_match_aligner import HybridMatchAligner

# Configuration
data_path = '/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data'
nvplay_path = f'{data_path}/nvplay_data_v3.csv'
decimal_path = f'{data_path}/decimal_data_v3.csv'

print("🔍 DEBUGGING HYBRID MATCH ALIGNER")
print("=" * 50)

# Load data
print("1. Loading data...")
nvplay_df = pd.read_csv(nvplay_path)
decimal_df = pd.read_csv(decimal_path)

print(f"NVPlay data: {len(nvplay_df)} rows, {nvplay_df['Match'].nunique()} unique matches")
print(f"Decimal data: {len(decimal_df)} rows")

# Check columns
print("\n2. Checking columns...")
print(f"NVPlay columns: {list(nvplay_df.columns)}")
print(f"Decimal columns: {list(decimal_df.columns)}")

# Check sample data
print("\n3. Sample data...")
print("NVPlay sample:")
print(nvplay_df[['Match', 'Over', 'Ball', 'Batter', 'Bowler', 'Runs', 'Innings']].head(3))

print("\nDecimal sample:")
print(decimal_df[['date', 'home', 'away', 'over', 'delivery', 'batsman', 'bowler', 'runs', 'innings']].head(3))

# Test aligner initialization
print("\n4. Testing aligner initialization...")
aligner = HybridMatchAligner(nvplay_path, decimal_path, None)
print(f"Aligner created successfully")

# Test configuration
print("\n5. Testing configuration...")
config = aligner.generate_llm_configuration()
print(f"Configuration: {config.column_mapping}")
print(f"Threshold: {config.similarity_threshold}")

# Test fingerprint extraction with limited data
print("\n6. Testing fingerprint extraction...")
# Limit to first few matches for testing
original_nvplay = aligner.nvplay_df.copy()
original_decimal = aligner.decimal_df.copy()

# Sample data for testing
sample_matches = list(aligner.nvplay_df['Match'].unique())[:2]
aligner.nvplay_df = aligner.nvplay_df[aligner.nvplay_df['Match'].isin(sample_matches)]
aligner.decimal_df = aligner.decimal_df.head(200)

print(f"Testing with {len(aligner.nvplay_df)} NVPlay rows, {len(aligner.decimal_df)} decimal rows")

try:
    nvplay_fps, decimal_fps = aligner.extract_fingerprints()
    print(f"NVPlay fingerprints: {len(nvplay_fps)}")
    print(f"Decimal fingerprints: {len(decimal_fps)}")
    
    if nvplay_fps:
        sample_nvplay_fp = list(nvplay_fps.values())[0]
        print(f"Sample NVPlay fingerprint (first 3 balls): {sample_nvplay_fp[:3]}")
    
    if decimal_fps:
        sample_decimal_fp = list(decimal_fps.values())[0]
        print(f"Sample decimal fingerprint (first 3 balls): {sample_decimal_fp[:3]}")
        
    # Test similarity calculation
    if nvplay_fps and decimal_fps:
        print("\n7. Testing similarity calculation...")
        nvplay_fp = list(nvplay_fps.values())[0]
        decimal_fp = list(decimal_fps.values())[0]
        
        similarity = aligner.calculate_fuzzy_similarity(nvplay_fp, decimal_fp)
        print(f"Sample similarity: {similarity:.3f}")
        
        # Test name similarity
        if nvplay_fp and decimal_fp:
            name_sim = aligner.fuzzy_name_similarity(nvplay_fp[0]['batter'], decimal_fp[0]['batter'])
            print(f"Name similarity: {name_sim:.3f} ('{nvplay_fp[0]['batter']}' vs '{decimal_fp[0]['batter']}')")
        
except Exception as e:
    print(f"Error in fingerprint extraction: {e}")
    import traceback
    traceback.print_exc()

# Restore original data
aligner.nvplay_df = original_nvplay
aligner.decimal_df = original_decimal

print("\n✅ Debug complete") 