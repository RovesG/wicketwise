#!/usr/bin/env python3
"""
Test script for OpenAI Match Enrichment Pipeline
Tests the pipeline without making actual API calls
"""

import json
from datetime import datetime
from dataclasses import asdict
from openai_match_enrichment_pipeline import (
    OpenAIMatchEnricher, 
    MatchEnrichmentPipeline,
    EnrichedMatchData,
    VenueData,
    TeamData, 
    PlayerData,
    TossData,
    WeatherData
)

def test_data_structures():
    """Test that all data structures work correctly"""
    print("🧪 Testing data structures...")
    
    # Test player data
    player = PlayerData(
        name="Virat Kohli",
        role="batter",
        batting_style="RHB",
        bowling_style="RM",
        captain=True,
        wicket_keeper=False,
        playing_xi=True
    )
    
    # Test team data
    team = TeamData(
        name="Royal Challengers Bangalore",
        short_name="RCB",
        is_home=True,
        players=[player]
    )
    
    # Test venue data
    venue = VenueData(
        name="M. Chinnaswamy Stadium",
        city="Bangalore",
        country="India",
        latitude=12.9784,
        longitude=77.5946,
        timezone="Asia/Kolkata"
    )
    
    # Test weather data
    weather = WeatherData(
        time_local="2024-04-15T19:30:00",
        time_utc="2024-04-15T14:00:00Z",
        temperature_c=28.5,
        feels_like_c=32.1,
        humidity_pct=65,
        precip_mm=0.0,
        precip_prob_pct=10,
        wind_speed_kph=12.3,
        wind_gust_kph=18.7,
        wind_dir_deg=180,
        cloud_cover_pct=25,
        pressure_hpa=1013,
        uv_index=6.5,
        weather_code="partly_cloudy"
    )
    
    # Test toss data
    toss = TossData(
        won_by="Royal Challengers Bangalore",
        decision="bat"
    )
    
    # Test complete match data
    enriched_match = EnrichedMatchData(
        competition="Indian Premier League",
        format="T20",
        date="2024-04-15",
        start_time_local="19:30",
        end_time_local="23:00",
        timezone="Asia/Kolkata",
        venue=venue,
        teams=[team],
        toss=toss,
        weather_hourly=[weather],
        data_sources={"players": "ESPNCricinfo", "weather": "OpenWeather"},
        generated_at=datetime.now().isoformat(),
        confidence_score=0.85,
        enrichment_status="success"
    )
    
    print("✅ All data structures created successfully!")
    return enriched_match

def test_prompt_generation():
    """Test prompt generation for OpenAI"""
    print("\n🤖 Testing prompt generation...")
    
    # Mock enricher (without API key to avoid actual calls)
    enricher = OpenAIMatchEnricher(api_key="test_key")
    
    match_info = {
        'home': 'Royal Challengers Bangalore',
        'away': 'Mumbai Indians',
        'venue': 'M. Chinnaswamy Stadium',
        'date': '2024-04-15',
        'competition': 'Indian Premier League'
    }
    
    prompt = enricher.create_enrichment_prompt(match_info)
    
    print("📝 Generated prompt preview (first 500 chars):")
    print(prompt[:500] + "...")
    
    # Validate prompt contains key elements
    assert "Royal Challengers Bangalore" in prompt
    assert "Mumbai Indians" in prompt
    assert "M. Chinnaswamy Stadium" in prompt
    assert "Indian Premier League" in prompt
    assert "JSON" in prompt
    
    print("✅ Prompt generation working correctly!")

def test_confidence_calculation():
    """Test confidence score calculation"""
    print("\n📊 Testing confidence calculation...")
    
    enricher = OpenAIMatchEnricher(api_key="test_key")
    
    # Test high-confidence match data
    high_confidence_match = {
        'venue': {'latitude': 12.9784, 'longitude': 77.5946},
        'format': 'T20',
        'start_time_local': '19:30',
        'toss': {'won_by': 'RCB', 'decision': 'bat'}
    }
    
    high_confidence_weather = [
        {'time_local': '2024-04-15T19:00:00'} for _ in range(6)
    ]
    
    high_confidence_teams = [
        TeamData("RCB", "RCB", True, [PlayerData("Player" + str(i), "batter", "RHB", "unknown") for i in range(11)]),
        TeamData("MI", "MI", False, [PlayerData("Player" + str(i), "batter", "RHB", "unknown") for i in range(11)])
    ]
    
    high_score = enricher.calculate_confidence(high_confidence_match, high_confidence_weather, high_confidence_teams)
    
    # Test low-confidence match data
    low_confidence_match = {
        'venue': {'latitude': 0, 'longitude': 0},
        'format': 'unknown',
        'start_time_local': 'unknown',
        'toss': {'won_by': 'unknown', 'decision': 'unknown'}
    }
    
    low_confidence_weather = []
    low_confidence_teams = []
    
    low_score = enricher.calculate_confidence(low_confidence_match, low_confidence_weather, low_confidence_teams)
    
    print(f"High confidence score: {high_score:.2f}")
    print(f"Low confidence score: {low_score:.2f}")
    
    assert high_score > low_score
    assert 0 <= high_score <= 1
    assert 0 <= low_score <= 1
    
    print("✅ Confidence calculation working correctly!")

def test_fallback_data():
    """Test fallback data creation"""
    print("\n🛡️ Testing fallback data creation...")
    
    enricher = OpenAIMatchEnricher(api_key="test_key")
    
    match_info = {
        'home': 'Team A',
        'away': 'Team B', 
        'venue': 'Test Stadium',
        'date': '2024-04-15',
        'competition': 'Test League'
    }
    
    fallback_data = enricher.create_fallback_data(match_info, "Test error")
    
    assert fallback_data.competition == 'Test League'
    assert fallback_data.venue.name == 'Test Stadium'
    assert len(fallback_data.teams) == 2
    assert fallback_data.teams[0].name == 'Team A'
    assert fallback_data.teams[1].name == 'Team B'
    assert fallback_data.confidence_score == 0.1
    assert fallback_data.enrichment_status == 'fallback'
    
    print("✅ Fallback data creation working correctly!")

def test_json_serialization():
    """Test that enriched data can be serialized to JSON"""
    print("\n💾 Testing JSON serialization...")
    
    enriched_match = test_data_structures()
    
    # Convert to dict (as done in the pipeline)
    match_dict = asdict(enriched_match)
    
    # Serialize to JSON
    json_str = json.dumps(match_dict, indent=2)
    
    # Deserialize back
    loaded_dict = json.loads(json_str)
    
    assert loaded_dict['competition'] == 'Indian Premier League'
    assert loaded_dict['venue']['name'] == 'M. Chinnaswamy Stadium'
    assert loaded_dict['teams'][0]['players'][0]['name'] == 'Virat Kohli'
    
    print("✅ JSON serialization working correctly!")

def create_sample_enriched_data():
    """Create sample enriched data for testing integration"""
    print("\n🎯 Creating sample enriched data...")
    
    sample_matches = []
    
    # Create 5 sample matches with varying confidence levels
    matches_data = [
        {
            'home': 'Royal Challengers Bangalore',
            'away': 'Mumbai Indians',
            'venue': 'M. Chinnaswamy Stadium',
            'competition': 'Indian Premier League',
            'confidence': 0.9
        },
        {
            'home': 'Chennai Super Kings',
            'away': 'Delhi Capitals',
            'venue': 'M.A. Chidambaram Stadium',
            'competition': 'Indian Premier League',
            'confidence': 0.8
        },
        {
            'home': 'Sydney Sixers',
            'away': 'Melbourne Stars',
            'venue': 'Sydney Cricket Ground',
            'competition': 'Big Bash League',
            'confidence': 0.7
        },
        {
            'home': 'Karachi Kings',
            'away': 'Lahore Qalandars',
            'venue': 'National Stadium',
            'competition': 'Pakistan Super League',
            'confidence': 0.6
        },
        {
            'home': 'Test Team A',
            'away': 'Test Team B',
            'venue': 'Unknown Stadium',
            'competition': 'Unknown League',
            'confidence': 0.2
        }
    ]
    
    for i, match_data in enumerate(matches_data):
        # Create sample players
        players_team1 = [
            PlayerData(f"{match_data['home']} Player {j}", "batter", "RHB", "unknown")
            for j in range(11)
        ]
        players_team2 = [
            PlayerData(f"{match_data['away']} Player {j}", "batter", "RHB", "unknown")
            for j in range(11)
        ]
        
        # Create teams
        teams = [
            TeamData(match_data['home'], match_data['home'][:3].upper(), True, players_team1),
            TeamData(match_data['away'], match_data['away'][:3].upper(), False, players_team2)
        ]
        
        # Create venue
        venue = VenueData(
            name=match_data['venue'],
            city="Test City",
            country="Test Country",
            latitude=20.0 + i,  # Vary coordinates
            longitude=70.0 + i,
            timezone="UTC"
        )
        
        # Create weather (more weather for higher confidence)
        weather_hours = int(match_data['confidence'] * 6)  # 0-6 hours of weather data
        weather_hourly = [
            WeatherData(
                time_local=f"2024-04-{15+i}T{14+h}:00:00",
                time_utc=f"2024-04-{15+i}T{14+h}:00:00Z",
                temperature_c=25.0 + h,
                feels_like_c=27.0 + h,
                humidity_pct=60,
                precip_mm=0.0,
                precip_prob_pct=10,
                wind_speed_kph=10.0,
                wind_gust_kph=15.0,
                wind_dir_deg=180,
                cloud_cover_pct=20,
                pressure_hpa=1013,
                uv_index=5.0,
                weather_code="clear"
            ) for h in range(weather_hours)
        ]
        
        # Create enriched match
        enriched_match = EnrichedMatchData(
            competition=match_data['competition'],
            format="T20",
            date=f"2024-04-{15+i}",
            start_time_local="19:30",
            end_time_local="23:00",
            timezone="UTC",
            venue=venue,
            teams=teams,
            toss=TossData(match_data['home'], "bat"),
            weather_hourly=weather_hourly,
            data_sources={"players": "Test", "weather": "Test"},
            generated_at=datetime.now().isoformat(),
            confidence_score=match_data['confidence'],
            enrichment_status="success" if match_data['confidence'] > 0.5 else "fallback"
        )
        
        sample_matches.append(asdict(enriched_match))
    
    # Save sample data
    with open('sample_enriched_matches.json', 'w') as f:
        json.dump(sample_matches, f, indent=2)
    
    print(f"✅ Created {len(sample_matches)} sample enriched matches!")
    print("📁 Saved to: sample_enriched_matches.json")
    
    return sample_matches

def main():
    """Run all tests"""
    print("🚀 TESTING OPENAI MATCH ENRICHMENT PIPELINE")
    print("=" * 60)
    
    try:
        test_data_structures()
        test_prompt_generation()
        test_confidence_calculation()
        test_fallback_data()
        test_json_serialization()
        create_sample_enriched_data()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ PIPELINE READY FOR DEPLOYMENT")
        print("\n📋 NEXT STEPS:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Run pipeline on small sample (50 matches)")
        print("3. Validate results and adjust confidence scoring")
        print("4. Scale to full dataset (~4,000 matches)")
        print("5. Integrate enriched data into Knowledge Graph")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    main()
