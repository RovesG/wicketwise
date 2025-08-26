# Purpose: Improved enrichment refresh with better prompts and error handling
# Author: WicketWise Team, Last Modified: 2025-08-26

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime
import os
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enrich_single_match_improved(match_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich a single match with improved prompting and error handling"""
    
    try:
        from openai import OpenAI
        
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Create more specific prompt
        prompt = f"""
Please provide detailed information about this cricket match. Return ONLY valid JSON.

Match Details:
- Date: {match_data.get('date')}
- Competition: {match_data.get('competition')}
- Venue: {match_data.get('venue')}
- Home Team: {match_data.get('home')}
- Away Team: {match_data.get('away')}

Please research and provide:
1. Exact venue coordinates (latitude/longitude)
2. Venue details (city, country, capacity, pitch type)
3. Weather conditions for that date and location (temperature, humidity, wind)
4. Match context (day/night, toss details if known)

Return this exact JSON structure:
{{
    "competition": "{match_data.get('competition')}",
    "format": "T20",
    "date": "{match_data.get('date')}",
    "venue": {{
        "name": "{match_data.get('venue')}",
        "city": "actual_city_name",
        "country": "actual_country_name",
        "latitude": actual_latitude_number,
        "longitude": actual_longitude_number,
        "capacity": actual_capacity_number,
        "pitch_type": "grass/turf/etc"
    }},
    "teams": [
        {{"name": "{match_data.get('home')}", "is_home": true}},
        {{"name": "{match_data.get('away')}", "is_home": false}}
    ],
    "weather_hourly": [
        {{
            "time": "match_start_time",
            "temperature": actual_temp_celsius,
            "humidity": actual_humidity_percent,
            "wind_speed": actual_wind_kmh,
            "conditions": "actual_weather_description"
        }}
    ],
    "enrichment_status": "success",
    "confidence_score": 0.85
}}

If you cannot find specific weather or venue data, use "unknown" for strings and 0 for numbers, but try to provide as much real data as possible.
        """
        
        # Try multiple models in order of preference
        models_to_try = ['gpt-4o', 'gpt-4o-mini']
        
        for model in models_to_try:
            try:
                logger.info(f"Trying {model} for match enrichment...")
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a cricket data expert with access to historical match and venue information. Provide accurate, detailed data in valid JSON format only."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=1500
                )
                
                # Parse response
                response_text = response.choices[0].message.content.strip()
                
                if not response_text:
                    logger.warning(f"{model} returned empty response")
                    continue
                
                # Clean up JSON (remove markdown formatting if present)
                cleaned_text = response_text
                if response_text.startswith('```'):
                    lines = response_text.split('\n')
                    json_lines = []
                    in_json = False
                    
                    for line in lines:
                        if line.strip().startswith('```'):
                            in_json = not in_json
                            continue
                        if in_json:
                            json_lines.append(line)
                    
                    cleaned_text = '\n'.join(json_lines)
                
                # Parse JSON
                enriched_data = json.loads(cleaned_text)
                
                # Add metadata
                enriched_data['original_index'] = match_data.get('original_index')
                enriched_data['refresh_timestamp'] = datetime.now().isoformat()
                enriched_data['enrichment_model'] = model
                
                # Validate enrichment quality
                quality_score = 0
                venue = enriched_data.get('venue', {})
                weather = enriched_data.get('weather_hourly', [])
                
                # Check venue coordinates
                if (venue.get('latitude', 0) != 0 and venue.get('longitude', 0) != 0):
                    quality_score += 0.3
                
                # Check venue details
                if (venue.get('city', 'unknown') != 'unknown' and 
                    venue.get('country', 'unknown') != 'unknown'):
                    quality_score += 0.2
                
                # Check weather data
                if weather and len(weather) > 0:
                    weather_entry = weather[0]
                    if (weather_entry.get('temperature', 0) != 0 or 
                        weather_entry.get('humidity', 0) != 0):
                        quality_score += 0.5
                
                enriched_data['confidence_score'] = quality_score
                
                logger.info(f"✅ {model} enrichment successful, quality: {quality_score:.2f}")
                return enriched_data
                
            except json.JSONDecodeError as e:
                logger.warning(f"{model} returned invalid JSON: {e}")
                logger.debug(f"Response was: {response_text[:200]}...")
                continue
                
            except Exception as e:
                logger.warning(f"{model} failed: {e}")
                continue
        
        # If all models failed, return fallback
        logger.warning("All models failed, returning fallback data")
        raise Exception("All enrichment models failed")
        
    except Exception as e:
        logger.error(f"❌ Error enriching match: {e}")
        
        # Return fallback data
        return {
            "competition": match_data.get('competition'),
            "format": "T20",
            "date": match_data.get('date'),
            "venue": {
                "name": match_data.get('venue'),
                "city": "unknown",
                "country": "unknown",
                "latitude": 0.0,
                "longitude": 0.0,
                "capacity": 0,
                "pitch_type": "unknown"
            },
            "teams": [
                {"name": match_data.get('home'), "is_home": True},
                {"name": match_data.get('away'), "is_home": False}
            ],
            "weather_hourly": [],
            "enrichment_status": "fallback",
            "confidence_score": 0.1,
            "original_index": match_data.get('original_index'),
            "refresh_timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

def refresh_with_improved_approach(max_matches: int = 20):
    """Refresh quota-impacted matches with improved approach"""
    
    print(f"🔄 Improved Enrichment Refresh (Max {max_matches} matches)")
    print("=" * 60)
    
    try:
        # Load quota-impacted matches
        with open('enriched_data/quota_impacted_matches.json', 'r') as f:
            impacted_indices = json.load(f)
        
        # Load current enriched data
        with open('enriched_data/enriched_betting_matches.json', 'r') as f:
            enriched_data = json.load(f)
        
        print(f"📊 Total impacted matches: {len(impacted_indices)}")
        print(f"🎯 Processing {max_matches} matches with improved approach...")
        
        # Process subset
        refresh_indices = impacted_indices[:max_matches]
        results = {
            'total_processed': 0,
            'successful_enrichments': 0,
            'weather_improvements': 0,
            'coordinate_improvements': 0,
            'venue_improvements': 0
        }
        
        for i, idx in enumerate(refresh_indices):
            if idx >= len(enriched_data):
                continue
                
            match = enriched_data[idx]
            
            print(f"\n[{i+1}/{len(refresh_indices)}] Processing match {idx}:")
            print(f"   📅 {match.get('date')} - {match.get('competition')}")
            
            # Extract match info for enrichment
            match_info = {
                'date': match.get('date'),
                'competition': match.get('competition'),
                'venue': match.get('venue', {}).get('name', ''),
                'home': None,
                'away': None,
                'original_index': idx
            }
            
            # Extract team names
            teams = match.get('teams', [])
            if teams and len(teams) >= 2:
                home_team = next((t for t in teams if t.get('is_home')), teams[0])
                away_team = next((t for t in teams if not t.get('is_home')), teams[1])
                match_info['home'] = home_team.get('name')
                match_info['away'] = away_team.get('name')
            
            print(f"   🏏 {match_info['home']} vs {match_info['away']} at {match_info['venue']}")
            
            # Store original data for comparison
            old_weather = match.get('weather_hourly', [])
            old_venue = match.get('venue', {})
            old_confidence = match.get('confidence_score', 0.0)
            
            # Enrich the match
            print("   🚀 Enriching with improved approach...")
            enriched_match = enrich_single_match_improved(match_info)
            
            # Analyze improvements
            new_weather = enriched_match.get('weather_hourly', [])
            new_venue = enriched_match.get('venue', {})
            new_confidence = enriched_match.get('confidence_score', 0.0)
            
            improvements = []
            
            # Check weather improvement
            if len(new_weather) > len(old_weather):
                results['weather_improvements'] += 1
                improvements.append("🌤️ Weather")
            
            # Check coordinate improvement
            old_lat = old_venue.get('latitude', 0.0) if old_venue else 0.0
            new_lat = new_venue.get('latitude', 0.0) if new_venue else 0.0
            if new_lat != 0.0 and old_lat == 0.0:
                results['coordinate_improvements'] += 1
                improvements.append("📍 Coordinates")
            
            # Check venue detail improvement
            old_city = old_venue.get('city', 'unknown') if old_venue else 'unknown'
            new_city = new_venue.get('city', 'unknown') if new_venue else 'unknown'
            if new_city != 'unknown' and old_city == 'unknown':
                results['venue_improvements'] += 1
                improvements.append("🏟️ Venue details")
            
            # Check overall improvement
            if new_confidence > old_confidence:
                results['successful_enrichments'] += 1
            
            # Update the match data
            enriched_data[idx] = enriched_match
            results['total_processed'] += 1
            
            # Display results
            status = "✅ IMPROVED" if improvements else "🔄 REFRESHED"
            improvement_text = ", ".join(improvements) if improvements else "No improvements"
            
            print(f"   {status} - Confidence: {new_confidence:.2f}")
            print(f"   📈 Improvements: {improvement_text}")
            
            # Add small delay to avoid rate limiting
            time.sleep(1)
        
        # Save updated data
        backup_path = f"enriched_data/enriched_betting_matches_backup_improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_path, 'w') as f:
            json.dump(enriched_data, f, indent=2)
        
        with open('enriched_data/enriched_betting_matches.json', 'w') as f:
            json.dump(enriched_data, f, indent=2)
        
        # Display final results
        print(f"\n📊 Improved Refresh Results:")
        print(f"   🔄 Total processed: {results['total_processed']}")
        print(f"   ✅ Successful enrichments: {results['successful_enrichments']}")
        print(f"   🌤️ Weather improvements: {results['weather_improvements']}")
        print(f"   📍 Coordinate improvements: {results['coordinate_improvements']}")
        print(f"   🏟️ Venue improvements: {results['venue_improvements']}")
        
        success_rate = (results['successful_enrichments'] / results['total_processed']) * 100 if results['total_processed'] > 0 else 0
        print(f"   📈 Success rate: {success_rate:.1f}%")
        print(f"   💾 Backup saved: {backup_path}")
        
        if results['successful_enrichments'] > 0:
            print(f"\n🎉 Success! {results['successful_enrichments']} matches improved!")
            print("\n🚀 Next steps:")
            print("   1. Run with higher max_matches for more improvements")
            print("   2. Rebuild Knowledge Graph with enhanced data")
            print("   3. Test weather-aware cricket intelligence")
            return True
        else:
            print("\n⚠️ No improvements achieved. Consider checking API access or model availability.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Refresh failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Improved enrichment refresh")
    parser.add_argument("--max-matches", type=int, default=10, help="Maximum number of matches to refresh")
    args = parser.parse_args()
    
    success = refresh_with_improved_approach(args.max_matches)
    sys.exit(0 if success else 1)
