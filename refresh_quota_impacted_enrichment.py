# Purpose: Refresh enrichment data for matches impacted by API quota limits
# Author: WicketWise Team, Last Modified: 2025-08-26

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_quota_impacted_matches() -> List[int]:
    """Load the list of quota-impacted match indices"""
    quota_file = Path('enriched_data/quota_impacted_matches.json')
    if not quota_file.exists():
        logger.error("❌ Quota impacted matches file not found. Run analysis first.")
        return []
    
    with open(quota_file, 'r') as f:
        return json.load(f)

def load_enriched_data() -> List[Dict[str, Any]]:
    """Load current enriched data"""
    enriched_path = Path('enriched_data/enriched_betting_matches.json')
    with open(enriched_path, 'r') as f:
        return json.load(f)

def create_refresh_dataset(enriched_data: List[Dict[str, Any]], impacted_indices: List[int]) -> List[Dict[str, Any]]:
    """Create a dataset for re-enrichment containing only impacted matches"""
    
    refresh_matches = []
    
    for idx in impacted_indices:
        if idx < len(enriched_data):
            match = enriched_data[idx]
            
            # Extract core match data for re-enrichment
            refresh_match = {
                'date': match.get('date'),
                'competition': match.get('competition'),
                'venue': match.get('venue', {}).get('name', ''),
                'home': None,
                'away': None,
                'original_index': idx  # Track original position
            }
            
            # Extract team names
            teams = match.get('teams', [])
            if teams and len(teams) >= 2:
                home_team = next((t for t in teams if t.get('is_home')), teams[0])
                away_team = next((t for t in teams if not t.get('is_home')), teams[1])
                refresh_match['home'] = home_team.get('name')
                refresh_match['away'] = away_team.get('name')
            
            refresh_matches.append(refresh_match)
    
    return refresh_matches

def refresh_enrichment_data(refresh_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Re-run enrichment for quota-impacted matches"""
    
    try:
        from openai_match_enrichment_pipeline import MatchEnrichmentPipeline
        import pandas as pd
        import tempfile
        import os
        
        logger.info(f"🚀 Starting enrichment refresh for {len(refresh_matches)} matches...")
        
        # Create temporary CSV file for enrichment
        temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_csv_path = temp_csv.name
        
        try:
            # Convert refresh matches to DataFrame and save as CSV
            refresh_df = pd.DataFrame(refresh_matches)
            refresh_df.to_csv(temp_csv_path, index=False)
            temp_csv.close()
            
            logger.info(f"📄 Created temporary CSV with {len(refresh_matches)} matches: {temp_csv_path}")
            
            # Initialize enrichment pipeline
            pipeline = MatchEnrichmentPipeline()
            
            # Clear cache to force re-enrichment
            logger.info("🗑️ Clearing enrichment cache to force refresh...")
            pipeline.clear_cache()
            
            # Run enrichment on temporary file
            logger.info("🚀 Running enrichment pipeline...")
            output_path = pipeline.enrich_betting_dataset(
                temp_csv_path, 
                additional_matches=len(refresh_matches),
                force_refresh=True
            )
            
            # Load enriched results
            with open(output_path, 'r') as f:
                import json
                refreshed_data = json.load(f)
            
            logger.info(f"✅ Enriched {len(refreshed_data)} matches")
            
            # Create mapping from original index to refreshed data
            refresh_mapping = {}
            
            # Match refreshed data back to original indices
            for i, refresh_match in enumerate(refresh_matches):
                original_idx = refresh_match.get('original_index')
                if original_idx is not None and i < len(refreshed_data):
                    refresh_mapping[original_idx] = refreshed_data[i]
            
            return refresh_mapping
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)
                logger.info(f"🗑️ Cleaned up temporary file: {temp_csv_path}")
        
    except Exception as e:
        logger.error(f"❌ Error during enrichment refresh: {e}")
        import traceback
        traceback.print_exc()
        return {}

def merge_refreshed_data(original_data: List[Dict[str, Any]], refresh_mapping: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge refreshed data back into the original dataset"""
    
    updated_data = original_data.copy()
    update_count = 0
    
    for original_idx, refreshed_match in refresh_mapping.items():
        if original_idx < len(updated_data):
            # Preserve original structure but update with refreshed data
            original_match = updated_data[original_idx]
            
            # Update key fields with refreshed data
            if 'weather_hourly' in refreshed_match:
                original_match['weather_hourly'] = refreshed_match['weather_hourly']
            
            if 'venue' in refreshed_match:
                # Merge venue data
                original_venue = original_match.get('venue', {})
                refreshed_venue = refreshed_match['venue']
                
                # Update coordinates if they're valid
                if (refreshed_venue.get('latitude', 0.0) != 0.0 and 
                    refreshed_venue.get('longitude', 0.0) != 0.0):
                    original_venue.update(refreshed_venue)
            
            # Update enrichment metadata
            original_match['enrichment_status'] = refreshed_match.get('enrichment_status', 'refreshed')
            original_match['confidence_score'] = refreshed_match.get('confidence_score', 1.0)
            original_match['data_sources'] = refreshed_match.get('data_sources', {})
            original_match['generated_at'] = datetime.now().isoformat()
            original_match['refresh_timestamp'] = datetime.now().isoformat()
            
            update_count += 1
    
    logger.info(f"✅ Successfully merged {update_count} refreshed matches")
    return updated_data

def backup_original_data():
    """Create backup of original enriched data"""
    original_path = Path('enriched_data/enriched_betting_matches.json')
    backup_path = Path(f'enriched_data/enriched_betting_matches_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    if original_path.exists():
        import shutil
        shutil.copy2(original_path, backup_path)
        logger.info(f"💾 Created backup: {backup_path}")
        return backup_path
    return None

def save_refreshed_data(updated_data: List[Dict[str, Any]]):
    """Save the updated enriched data"""
    output_path = Path('enriched_data/enriched_betting_matches.json')
    
    with open(output_path, 'w') as f:
        json.dump(updated_data, f, indent=2)
    
    logger.info(f"💾 Saved refreshed data to {output_path}")

def validate_refresh_quality(updated_data: List[Dict[str, Any]], original_impacted_count: int):
    """Validate the quality of the refresh"""
    
    stats = {
        'total_matches': len(updated_data),
        'complete_weather': 0,
        'valid_coordinates': 0,
        'high_confidence': 0,
        'successful_enrichment': 0
    }
    
    for match in updated_data:
        # Check weather data
        weather_hourly = match.get('weather_hourly', [])
        if weather_hourly and len(weather_hourly) > 0:
            stats['complete_weather'] += 1
        
        # Check coordinates
        venue = match.get('venue', {})
        lat = venue.get('latitude', 0.0)
        lng = venue.get('longitude', 0.0)
        if lat != 0.0 and lng != 0.0:
            stats['valid_coordinates'] += 1
        
        # Check confidence
        confidence = match.get('confidence_score', 0.0)
        if confidence >= 0.8:
            stats['high_confidence'] += 1
        
        # Check enrichment status
        status = match.get('enrichment_status', '')
        if status not in ['fallback', 'error']:
            stats['successful_enrichment'] += 1
    
    print(f"\\n📊 Refresh Quality Validation:")
    print(f"   Total matches: {stats['total_matches']}")
    print(f"   🌤️ Complete weather: {stats['complete_weather']} ({stats['complete_weather']/stats['total_matches']*100:.1f}%)")
    print(f"   📍 Valid coordinates: {stats['valid_coordinates']} ({stats['valid_coordinates']/stats['total_matches']*100:.1f}%)")
    print(f"   📈 High confidence: {stats['high_confidence']} ({stats['high_confidence']/stats['total_matches']*100:.1f}%)")
    print(f"   ✅ Successful enrichment: {stats['successful_enrichment']} ({stats['successful_enrichment']/stats['total_matches']*100:.1f}%)")
    
    improvement = stats['complete_weather'] - (stats['total_matches'] - original_impacted_count)
    print(f"\\n🎯 Weather Data Improvement: +{improvement} matches with weather data")
    
    return stats

def main():
    """Main refresh process"""
    
    print("🔄 WicketWise Enrichment Data Refresh")
    print("=" * 60)
    
    try:
        # Step 1: Load quota-impacted matches
        print("\\n📋 Step 1: Loading quota-impacted matches...")
        impacted_indices = load_quota_impacted_matches()
        
        if not impacted_indices:
            print("❌ No quota-impacted matches found. Exiting.")
            return False
        
        print(f"✅ Found {len(impacted_indices)} matches to refresh")
        
        # Step 2: Load current enriched data
        print("\\n📊 Step 2: Loading current enriched data...")
        original_data = load_enriched_data()
        print(f"✅ Loaded {len(original_data)} total matches")
        
        # Step 3: Create backup
        print("\\n💾 Step 3: Creating backup...")
        backup_path = backup_original_data()
        
        # Step 4: Create refresh dataset
        print("\\n🎯 Step 4: Preparing refresh dataset...")
        refresh_matches = create_refresh_dataset(original_data, impacted_indices)
        print(f"✅ Prepared {len(refresh_matches)} matches for re-enrichment")
        
        # Step 5: Run enrichment refresh
        print("\\n🚀 Step 5: Running enrichment refresh...")
        print("⚠️ This may take several minutes with API calls...")
        
        refresh_mapping = refresh_enrichment_data(refresh_matches)
        
        if not refresh_mapping:
            print("❌ Enrichment refresh failed. Check API access and try again.")
            return False
        
        print(f"✅ Successfully refreshed {len(refresh_mapping)} matches")
        
        # Step 6: Merge refreshed data
        print("\\n🔄 Step 6: Merging refreshed data...")
        updated_data = merge_refreshed_data(original_data, refresh_mapping)
        
        # Step 7: Save updated data
        print("\\n💾 Step 7: Saving updated data...")
        save_refreshed_data(updated_data)
        
        # Step 8: Validate refresh quality
        print("\\n📊 Step 8: Validating refresh quality...")
        validate_refresh_quality(updated_data, len(impacted_indices))
        
        print("\\n🎉 Enrichment refresh completed successfully!")
        print("\\n🚀 Next steps:")
        print("   1. Rebuild Knowledge Graph with enhanced weather data")
        print("   2. Retrain GNN with weather-aware embeddings")
        print("   3. Test weather-based cricket intelligence queries")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during refresh process: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
