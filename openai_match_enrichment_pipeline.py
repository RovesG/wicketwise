#!/usr/bin/env python3
"""
OpenAI Match Enrichment Pipeline
Enriches cricket match data with comprehensive information using OpenAI API
"""

import json
import os
import pandas as pd
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import openai
from dataclasses import dataclass, asdict
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VenueData:
    name: str
    city: str
    country: str
    latitude: float
    longitude: float
    timezone: str

@dataclass
class PlayerData:
    name: str
    role: str  # batter|bowler|allrounder|wk
    batting_style: str  # RHB|LHB|unknown
    bowling_style: str  # RF|RM|LF|LM|OB|LB|SLA|SLC|unknown
    captain: bool = False
    wicket_keeper: bool = False
    playing_xi: bool = True

@dataclass
class TeamData:
    name: str
    short_name: str
    is_home: bool
    players: List[PlayerData]

@dataclass
class TossData:
    won_by: str
    decision: str  # bat|bowl|unknown

@dataclass
class WeatherData:
    time_local: str
    time_utc: str
    temperature_c: float
    feels_like_c: float
    humidity_pct: int
    precip_mm: float
    precip_prob_pct: int
    wind_speed_kph: float
    wind_gust_kph: float
    wind_dir_deg: int
    cloud_cover_pct: int
    pressure_hpa: int
    uv_index: float
    weather_code: str

@dataclass
class EnrichedMatchData:
    competition: str
    format: str  # T20|ODI|Test|Other
    date: str
    start_time_local: str
    end_time_local: str
    timezone: str
    venue: VenueData
    teams: List[TeamData]
    toss: TossData
    weather_hourly: List[WeatherData]
    data_sources: Dict[str, str]
    generated_at: str
    confidence_score: float
    enrichment_status: str

class OpenAIMatchEnricher:
    """
    Enriches cricket match data using OpenAI API with structured JSON responses
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the enricher with OpenAI API key"""
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv('OPENAI_API_KEY')
        )
        self.cache = {}  # Simple in-memory cache
        self.rate_limit_delay = 1.0  # seconds between API calls
        
    def create_enrichment_prompt(self, match_info: Dict[str, Any]) -> str:
        """Create a structured prompt for match enrichment"""
        
        prompt = f"""You are a cricket data expert. I need comprehensive information about this cricket match in a specific JSON format.

MATCH DETAILS:
- Home Team: {match_info['home']}
- Away Team: {match_info['away']}
- Venue: {match_info['venue']}
- Date: {match_info['date']}
- Competition: {match_info['competition']}

Please provide a complete JSON response with the following structure:

{{
  "match": {{
    "competition": "{match_info['competition']}",
    "format": "T20|ODI|Test|Other",
    "date": "{match_info['date']}",
    "start_time_local": "HH:MM",
    "end_time_local": "HH:MM", 
    "timezone": "IANA TZ string",
    "venue": {{
      "name": "Official venue name",
      "city": "City name",
      "country": "Country name",
      "latitude": 0.0,
      "longitude": 0.0
    }},
    "teams": [
      {{
        "name": "Official team name",
        "short_name": "3-4 letter code",
        "is_home": true,
        "players": [
          {{
            "name": "Full player name",
            "role": "batter|bowler|allrounder|wk",
            "batting_style": "RHB|LHB|unknown",
            "bowling_style": "RF|RM|LF|LM|OB|LB|SLA|SLC|unknown",
            "captain": false,
            "wicket_keeper": false,
            "playing_xi": true
          }}
        ]
      }},
      {{
        "name": "Official team name", 
        "short_name": "3-4 letter code",
        "is_home": false,
        "players": [
          // Same player structure
        ]
      }}
    ],
    "toss": {{ "won_by": "Team name", "decision": "bat|bowl|unknown" }}
  }},
  "weather_hourly": [
    {{
      "time_local": "{match_info['date']}T14:00:00",
      "time_utc": "{match_info['date']}T09:00:00Z",
      "temperature_c": 0.0,
      "feels_like_c": 0.0,
      "humidity_pct": 0,
      "precip_mm": 0.0,
      "precip_prob_pct": 0,
      "wind_speed_kph": 0.0,
      "wind_gust_kph": 0.0,
      "wind_dir_deg": 0,
      "cloud_cover_pct": 0,
      "pressure_hpa": 0,
      "uv_index": 0.0,
      "weather_code": "clear|cloudy|rain|etc"
    }}
  ],
  "data_sources": {{
    "players": "ESPNCricinfo|Cricbuzz|Official",
    "weather": "Historical weather service"
  }},
  "generated_at": "{datetime.now().isoformat()}Z"
}}

IMPORTANT:
1. Use official team and player names
2. Include complete playing XIs if available
3. Provide accurate venue coordinates
4. Include hourly weather for match duration
5. Mark unknown fields as "unknown" rather than guessing
6. Ensure all JSON is valid and complete
"""
        return prompt

    def enrich_match(self, match_info: Dict[str, Any]) -> Optional[EnrichedMatchData]:
        """Enrich a single match using OpenAI API"""
        
        # Check cache first
        cache_key = f"{match_info['home']}_{match_info['away']}_{match_info['venue']}_{match_info['date']}"
        if cache_key in self.cache:
            logger.info(f"📦 Using cached data for {cache_key}")
            return self.cache[cache_key]
        
        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            prompt = self.create_enrichment_prompt(match_info)
            
            logger.info(f"🤖 Enriching: {match_info['home']} vs {match_info['away']} at {match_info['venue']}")
            
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use gpt-4o for better structured output
                messages=[
                    {"role": "system", "content": "You are a cricket data expert who provides accurate, structured JSON responses about cricket matches."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for factual accuracy
                max_tokens=4000,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            # Parse the JSON response
            raw_json = response.choices[0].message.content
            enriched_data = json.loads(raw_json)
            
            # Validate and convert to our data structure
            validated_data = self.validate_and_convert(enriched_data, match_info)
            
            # Cache the result
            self.cache[cache_key] = validated_data
            
            logger.info(f"✅ Successfully enriched match: {validated_data.confidence_score:.2f} confidence")
            return validated_data
            
        except Exception as e:
            logger.error(f"❌ Failed to enrich match {cache_key}: {e}")
            return self.create_fallback_data(match_info, str(e))
    
    def validate_and_convert(self, raw_data: Dict[str, Any], original_match: Dict[str, Any]) -> EnrichedMatchData:
        """Validate and convert raw OpenAI response to structured data"""
        
        try:
            match_data = raw_data.get('match', {})
            weather_data = raw_data.get('weather_hourly', [])
            
            # Convert venue data
            venue_raw = match_data.get('venue', {})
            venue = VenueData(
                name=venue_raw.get('name', original_match['venue']),
                city=venue_raw.get('city', 'unknown'),
                country=venue_raw.get('country', 'unknown'),
                latitude=float(venue_raw.get('latitude', 0.0)),
                longitude=float(venue_raw.get('longitude', 0.0)),
                timezone=match_data.get('timezone', 'UTC')
            )
            
            # Convert team data
            teams = []
            for team_raw in match_data.get('teams', []):
                players = []
                for player_raw in team_raw.get('players', []):
                    player = PlayerData(
                        name=player_raw.get('name', 'unknown'),
                        role=player_raw.get('role', 'unknown'),
                        batting_style=player_raw.get('batting_style', 'unknown'),
                        bowling_style=player_raw.get('bowling_style', 'unknown'),
                        captain=player_raw.get('captain', False),
                        wicket_keeper=player_raw.get('wicket_keeper', False),
                        playing_xi=player_raw.get('playing_xi', True)
                    )
                    players.append(player)
                
                team = TeamData(
                    name=team_raw.get('name', 'unknown'),
                    short_name=team_raw.get('short_name', 'UNK'),
                    is_home=team_raw.get('is_home', False),
                    players=players
                )
                teams.append(team)
            
            # Convert toss data
            toss_raw = match_data.get('toss', {})
            toss = TossData(
                won_by=toss_raw.get('won_by', 'unknown'),
                decision=toss_raw.get('decision', 'unknown')
            )
            
            # Convert weather data
            weather_hourly = []
            for weather_raw in weather_data:
                weather = WeatherData(
                    time_local=weather_raw.get('time_local', ''),
                    time_utc=weather_raw.get('time_utc', ''),
                    temperature_c=float(weather_raw.get('temperature_c', 0.0)),
                    feels_like_c=float(weather_raw.get('feels_like_c', 0.0)),
                    humidity_pct=int(weather_raw.get('humidity_pct', 0)),
                    precip_mm=float(weather_raw.get('precip_mm', 0.0)),
                    precip_prob_pct=int(weather_raw.get('precip_prob_pct', 0)),
                    wind_speed_kph=float(weather_raw.get('wind_speed_kph', 0.0)),
                    wind_gust_kph=float(weather_raw.get('wind_gust_kph', 0.0)),
                    wind_dir_deg=int(weather_raw.get('wind_dir_deg', 0)),
                    cloud_cover_pct=int(weather_raw.get('cloud_cover_pct', 0)),
                    pressure_hpa=int(weather_raw.get('pressure_hpa', 1013)),
                    uv_index=float(weather_raw.get('uv_index', 0.0)),
                    weather_code=weather_raw.get('weather_code', 'unknown')
                )
                weather_hourly.append(weather)
            
            # Calculate confidence score based on data completeness
            confidence_score = self.calculate_confidence(match_data, weather_data, teams)
            
            return EnrichedMatchData(
                competition=match_data.get('competition', original_match['competition']),
                format=match_data.get('format', 'T20'),
                date=match_data.get('date', original_match['date']),
                start_time_local=match_data.get('start_time_local', '14:00'),
                end_time_local=match_data.get('end_time_local', '18:00'),
                timezone=match_data.get('timezone', 'UTC'),
                venue=venue,
                teams=teams,
                toss=toss,
                weather_hourly=weather_hourly,
                data_sources=raw_data.get('data_sources', {}),
                generated_at=raw_data.get('generated_at', datetime.now().isoformat()),
                confidence_score=confidence_score,
                enrichment_status='success'
            )
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return self.create_fallback_data(original_match, f"Validation error: {e}")
    
    def calculate_confidence(self, match_data: Dict, weather_data: List, teams: List[TeamData]) -> float:
        """Calculate confidence score based on data completeness"""
        
        score = 0.0
        max_score = 10.0
        
        # Venue coordinates (2 points)
        if match_data.get('venue', {}).get('latitude', 0) != 0:
            score += 2.0
        
        # Team data completeness (3 points)
        if len(teams) >= 2:
            score += 1.0
            total_players = sum(len(team.players) for team in teams)
            if total_players >= 22:  # Full XIs
                score += 2.0
        
        # Weather data (2 points)
        if len(weather_data) >= 4:  # At least 4 hours of data
            score += 2.0
        
        # Toss information (1 point)
        if match_data.get('toss', {}).get('won_by', 'unknown') != 'unknown':
            score += 1.0
        
        # Format and timing (2 points)
        if match_data.get('format', 'unknown') != 'unknown':
            score += 1.0
        if match_data.get('start_time_local', 'unknown') != 'unknown':
            score += 1.0
        
        return score / max_score
    
    def create_fallback_data(self, match_info: Dict[str, Any], error_msg: str) -> EnrichedMatchData:
        """Create minimal fallback data when enrichment fails"""
        
        return EnrichedMatchData(
            competition=match_info['competition'],
            format='T20',  # Default assumption
            date=match_info['date'],
            start_time_local='14:00',
            end_time_local='18:00',
            timezone='UTC',
            venue=VenueData(
                name=match_info['venue'],
                city='unknown',
                country='unknown',
                latitude=0.0,
                longitude=0.0,
                timezone='UTC'
            ),
            teams=[
                TeamData(name=match_info['home'], short_name='HOME', is_home=True, players=[]),
                TeamData(name=match_info['away'], short_name='AWAY', is_home=False, players=[])
            ],
            toss=TossData(won_by='unknown', decision='unknown'),
            weather_hourly=[],
            data_sources={'error': error_msg},
            generated_at=datetime.now().isoformat(),
            confidence_score=0.1,
            enrichment_status='fallback'
        )

class MatchEnrichmentPipeline:
    """
    Main pipeline for enriching cricket match datasets with intelligent caching
    """
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "enriched_data"):
        self.enricher = OpenAIMatchEnricher(api_key)
        self.output_dir = output_dir
        self.cache_file = f"{output_dir}/enrichment_cache.json"
        self.master_file = f"{output_dir}/enriched_betting_matches.json"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load existing enrichment cache
        self.enrichment_cache = self._load_enrichment_cache()
        logger.info(f"📦 Loaded {len(self.enrichment_cache)} cached enrichments")
    
    def _load_enrichment_cache(self) -> Dict[str, Dict]:
        """Load existing enrichment cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                return cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Starting fresh.")
                return {}
        return {}
    
    def _save_enrichment_cache(self):
        """Save enrichment cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.enrichment_cache, f, indent=2)
            logger.info(f"💾 Saved {len(self.enrichment_cache)} enrichments to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _create_match_key(self, match_info: Dict[str, Any]) -> str:
        """Create a unique key for a match (for caching)"""
        # Use home, away, venue, date, competition as unique identifier
        key_parts = [
            str(match_info.get('home', '')),
            str(match_info.get('away', '')), 
            str(match_info.get('venue', '')),
            str(match_info.get('date', '')),
            str(match_info.get('competition', ''))
        ]
        # Create a hash for consistent key generation
        import hashlib
        key_string = '|'.join(key_parts).lower()
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_match_cached(self, match_info: Dict[str, Any]) -> bool:
        """Check if a match has already been enriched"""
        match_key = self._create_match_key(match_info)
        return match_key in self.enrichment_cache
    
    def _get_cached_match(self, match_info: Dict[str, Any]) -> Optional[Dict]:
        """Get cached enrichment for a match"""
        match_key = self._create_match_key(match_info)
        return self.enrichment_cache.get(match_key)
    
    def _cache_match(self, match_info: Dict[str, Any], enriched_data: EnrichedMatchData):
        """Cache an enriched match"""
        match_key = self._create_match_key(match_info)
        self.enrichment_cache[match_key] = {
            'match_info': match_info,
            'enriched_data': asdict(enriched_data),
            'cached_at': datetime.now().isoformat(),
            'cache_version': '1.0'
        }
        
    def enrich_betting_dataset(self, betting_data_path: str, max_matches: Optional[int] = None, 
                              priority_competitions: Optional[List[str]] = None, force_refresh: bool = False) -> str:
        """Enrich the betting dataset with OpenAI match data (with intelligent caching)"""
        
        logger.info(f"🚀 Starting betting dataset enrichment from {betting_data_path}")
        
        # Load betting data
        betting_data = pd.read_csv(betting_data_path)
        
        # Extract unique matches
        matches = betting_data.groupby(['date', 'competition', 'venue', 'home', 'away']).agg({
            'ball': 'count'
        }).rename(columns={'ball': 'total_balls'}).reset_index()
        
        # Prioritize matches
        if priority_competitions:
            matches = matches[matches['competition'].isin(priority_competitions)]
        
        # Sort by total balls (more complete matches first)
        matches = matches.sort_values('total_balls', ascending=False)
        
        logger.info(f"📊 Analyzing {len(matches)} matches...")
        
        # Separate cached vs new matches FIRST
        cached_matches = []
        new_matches = []
        
        for idx, match in matches.iterrows():
            match_info = {
                'home': match['home'],
                'away': match['away'],
                'venue': match['venue'],
                'date': match['date'],
                'competition': match['competition']
            }
            
            if not force_refresh and self._is_match_cached(match_info):
                cached_match = self._get_cached_match(match_info)
                cached_matches.append(cached_match['enriched_data'])
            else:
                new_matches.append(match_info)
        
        # NOW apply max_matches limit to NEW matches only
        if max_matches and len(new_matches) > max_matches:
            logger.info(f"🎯 Limiting to {max_matches} new matches (from {len(new_matches)} available)")
            new_matches = new_matches[:max_matches]
        
        logger.info(f"📦 Found {len(cached_matches)} cached matches")
        logger.info(f"🆕 Need to enrich {len(new_matches)} new matches")
        
        # Enrich new matches only
        newly_enriched = []
        api_calls_made = 0
        
        for idx, match_info in enumerate(new_matches):
            logger.info(f"🤖 Enriching: {match_info['home']} vs {match_info['away']} ({idx+1}/{len(new_matches)})")
            
            enriched_data = self.enricher.enrich_match(match_info)
            if enriched_data:
                newly_enriched.append(asdict(enriched_data))
                # Cache the enrichment
                self._cache_match(match_info, enriched_data)
                api_calls_made += 1
                
                # Save cache every 10 matches to prevent data loss
                if api_calls_made % 10 == 0:
                    self._save_enrichment_cache()
                    logger.info(f"💾 Cache saved at {api_calls_made} API calls")
            
            # Progress logging
            if (idx + 1) % 5 == 0:
                logger.info(f"📈 Progress: {idx + 1}/{len(new_matches)} new matches processed")
        
        # Save final cache
        self._save_enrichment_cache()
        
        # Combine cached and newly enriched matches
        all_enriched_matches = cached_matches + newly_enriched
        
        # Save combined results
        output_file = self.master_file
        with open(output_file, 'w') as f:
            json.dump(all_enriched_matches, f, indent=2)
        
        # Log summary
        total_cost = api_calls_made * 0.02
        logger.info(f"✅ Enrichment complete!")
        logger.info(f"📊 Total matches: {len(all_enriched_matches)}")
        logger.info(f"📦 From cache: {len(cached_matches)}")
        logger.info(f"🆕 Newly enriched: {len(newly_enriched)}")
        logger.info(f"💰 API calls made: {api_calls_made} (~${total_cost:.2f})")
        logger.info(f"💾 Saved to: {output_file}")
        
        return output_file
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get statistics about the enrichment cache"""
        cache_stats = {
            'total_cached_matches': len(self.enrichment_cache),
            'cache_file_exists': os.path.exists(self.cache_file),
            'cache_file_size_mb': 0,
            'oldest_cache_entry': None,
            'newest_cache_entry': None,
            'competitions_cached': set(),
            'venues_cached': set()
        }
        
        if os.path.exists(self.cache_file):
            cache_stats['cache_file_size_mb'] = round(os.path.getsize(self.cache_file) / 1024 / 1024, 2)
        
        if self.enrichment_cache:
            # Analyze cache entries
            cache_dates = []
            for entry in self.enrichment_cache.values():
                if 'cached_at' in entry:
                    cache_dates.append(entry['cached_at'])
                
                match_info = entry.get('match_info', {})
                if 'competition' in match_info:
                    cache_stats['competitions_cached'].add(match_info['competition'])
                if 'venue' in match_info:
                    cache_stats['venues_cached'].add(match_info['venue'])
            
            if cache_dates:
                cache_stats['oldest_cache_entry'] = min(cache_dates)
                cache_stats['newest_cache_entry'] = max(cache_dates)
        
        # Convert sets to lists for JSON serialization (always, even if empty)
        cache_stats['competitions_cached'] = list(cache_stats['competitions_cached'])
        cache_stats['venues_cached'] = list(cache_stats['venues_cached'])
        
        return cache_stats
    
    def clear_cache(self, confirm: bool = False) -> bool:
        """Clear the enrichment cache (use with caution!)"""
        if not confirm:
            logger.warning("⚠️  Cache clear requested but not confirmed. Use confirm=True to proceed.")
            return False
        
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                logger.info(f"🗑️ Deleted cache file: {self.cache_file}")
            
            self.enrichment_cache = {}
            logger.info("✅ Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear cache: {e}")
            return False
    
    def generate_summary_report(self, enriched_data_path: str) -> str:
        """Generate a summary report of the enrichment results"""
        
        with open(enriched_data_path, 'r') as f:
            enriched_data = json.load(f)
        
        total_matches = len(enriched_data)
        successful_enrichments = len([m for m in enriched_data if m['enrichment_status'] == 'success'])
        avg_confidence = sum(m['confidence_score'] for m in enriched_data) / total_matches
        
        # Venue analysis
        venues_with_coords = len([m for m in enriched_data if m['venue']['latitude'] != 0])
        
        # Weather analysis
        matches_with_weather = len([m for m in enriched_data if len(m['weather_hourly']) > 0])
        
        # Team analysis
        matches_with_full_squads = len([m for m in enriched_data 
                                      if len(m['teams']) == 2 and 
                                      all(len(team['players']) >= 11 for team in m['teams'])])
        
        report = f"""
🎯 MATCH ENRICHMENT SUMMARY REPORT
{'='*50}

📊 OVERALL STATISTICS:
• Total matches processed: {total_matches:,}
• Successful enrichments: {successful_enrichments:,} ({successful_enrichments/total_matches*100:.1f}%)
• Average confidence score: {avg_confidence:.2f}

🌍 VENUE ENRICHMENT:
• Venues with coordinates: {venues_with_coords:,} ({venues_with_coords/total_matches*100:.1f}%)

🌤️ WEATHER ENRICHMENT:
• Matches with weather data: {matches_with_weather:,} ({matches_with_weather/total_matches*100:.1f}%)

👥 TEAM ENRICHMENT:
• Matches with full squads: {matches_with_full_squads:,} ({matches_with_full_squads/total_matches*100:.1f}%)

💰 COST ANALYSIS:
• Estimated API cost: ${total_matches * 0.02:.2f}
• Cost per successful enrichment: ${(total_matches * 0.02) / successful_enrichments:.3f}

✅ NEXT STEPS:
1. Integrate enriched data into Knowledge Graph
2. Update betting model with weather features
3. Validate team-player mappings
4. Build venue-based performance models
"""
        
        report_file = f"{self.output_dir}/enrichment_summary.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        return report_file

def main():
    """Example usage of the match enrichment pipeline"""
    
    # Initialize pipeline
    pipeline = MatchEnrichmentPipeline()
    
    # Test with a small sample first
    priority_competitions = [
        'Indian Premier League',
        'Big Bash League', 
        'Pakistan Super League',
        'T20I'
    ]
    
    # Enrich top 50 matches from priority competitions
    enriched_file = pipeline.enrich_betting_dataset(
        betting_data_path='/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/decimal_data_v3.csv',
        max_matches=50,
        priority_competitions=priority_competitions
    )
    
    # Generate summary report
    pipeline.generate_summary_report(enriched_file)

if __name__ == "__main__":
    main()
