#!/usr/bin/env python3
"""
Quick test of match enrichment with a small batch
"""

import os
import json
from pathlib import Path
from openai_match_enrichment_pipeline import MatchEnrichmentPipeline

def main():
    print("🧪 TESTING MATCH ENRICHMENT")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OpenAI API key found!")
        return
    
    print(f"✅ OpenAI API key configured (ends with: ...{api_key[-8:]})")
    
    # Initialize pipeline
    print("\n📊 Initializing enrichment pipeline...")
    pipeline = MatchEnrichmentPipeline(api_key=api_key, output_dir="test_enrichment_output")
    
    # Test with very small batch
    betting_data_path = '/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/decimal_data_v3.csv'
    
    print(f"📁 Loading betting data from: {betting_data_path}")
    
    # Priority competitions for testing
    priority_competitions = [
        'Indian Premier League',
        'Big Bash League'
    ]
    
    print(f"🎯 Testing with {priority_competitions}")
    print(f"📈 Max matches: 5 (for testing)")
    print(f"💰 Estimated cost: ~$0.10")
    
    try:
        # Run enrichment on small batch
        print("\n🚀 Starting enrichment...")
        enriched_file = pipeline.enrich_betting_dataset(
            betting_data_path=betting_data_path,
            max_matches=5,  # Very small test
            priority_competitions=priority_competitions
        )
        
        print(f"\n✅ Enrichment complete!")
        print(f"📄 Output file: {enriched_file}")
        
        # Generate summary
        print("\n📋 Generating summary report...")
        report_file = pipeline.generate_summary_report(enriched_file)
        print(f"📊 Report file: {report_file}")
        
        # Show cache statistics
        print("\n📦 Cache statistics:")
        cache_stats = pipeline.get_cache_statistics()
        for key, value in cache_stats.items():
            if key not in ['competitions_cached', 'venues_cached']:
                print(f"  {key}: {value}")
        
        # Show sample enriched data
        print("\n🔍 SAMPLE ENRICHED DATA:")
        print("=" * 30)
        
        with open(enriched_file, 'r') as f:
            enriched_data = json.load(f)
        
        if enriched_data:
            sample_match = enriched_data[0]
            
            print(f"Match: {sample_match.get('competition', 'Unknown')}")
            print(f"Format: {sample_match.get('format', 'Unknown')}")
            print(f"Date: {sample_match.get('date', 'Unknown')}")
            
            # Venue info
            venue = sample_match.get('venue', {})
            print(f"Venue: {venue.get('name', 'Unknown')}")
            print(f"Location: {venue.get('city', 'Unknown')}, {venue.get('country', 'Unknown')}")
            print(f"Coordinates: ({venue.get('latitude', 0)}, {venue.get('longitude', 0)})")
            
            # Teams
            teams = sample_match.get('teams', [])
            print(f"Teams: {len(teams)} teams")
            for team in teams:
                team_name = team.get('name', 'Unknown')
                players = team.get('players', [])
                print(f"  {team_name}: {len(players)} players")
            
            # Weather
            weather = sample_match.get('weather_hourly', [])
            print(f"Weather data: {len(weather)} hourly entries")
            if weather:
                first_weather = weather[0]
                print(f"  Temperature: {first_weather.get('temperature_c', 0)}°C")
                print(f"  Humidity: {first_weather.get('humidity_pct', 0)}%")
                print(f"  Wind: {first_weather.get('wind_speed_kph', 0)} kph")
            
            # Confidence
            confidence = sample_match.get('confidence_score', 0)
            status = sample_match.get('enrichment_status', 'unknown')
            print(f"Confidence: {confidence:.2f} ({status})")
        
        print(f"\n🎉 Test completed successfully!")
        print(f"📁 All files saved to: test_enrichment_output/")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
