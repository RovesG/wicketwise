#!/usr/bin/env python3
"""
Enriched Knowledge Graph Builder
Extends the existing KG with OpenAI-enriched match data including:
- Weather conditions (temperature, humidity, wind, precipitation)
- Team squad information (playing XIs, roles, captaincy)
- Venue coordinates and timezone data
- Match context (toss details, format, timing)

Author: WicketWise Team, Last Modified: 2025-01-19
"""

import json
import pickle
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class WeatherConditions:
    """Weather conditions during a match"""
    temperature_c: float
    feels_like_c: float
    humidity_pct: int
    wind_speed_kph: float
    wind_gust_kph: float
    wind_dir_deg: int
    precip_mm: float
    precip_prob_pct: int
    cloud_cover_pct: int
    pressure_hpa: int
    uv_index: float
    weather_code: str

@dataclass
class VenueEnrichment:
    """Enhanced venue information"""
    latitude: float
    longitude: float
    timezone: str
    city: str
    country: str
    weather_conditions: List[WeatherConditions]

@dataclass
class PlayerSquadInfo:
    """Enhanced player information from team squads"""
    role: str  # batter|bowler|allrounder|wk
    batting_style: str  # RHB|LHB|unknown
    bowling_style: str  # RF|RM|LF|LM|OB|LB|SLA|SLC|unknown
    captain: bool
    wicket_keeper: bool
    playing_xi: bool

@dataclass
class TeamEnrichment:
    """Enhanced team information"""
    official_name: str
    short_name: str
    is_home: bool
    squad: Dict[str, PlayerSquadInfo]  # player_name -> PlayerSquadInfo

@dataclass
class MatchEnrichment:
    """Enhanced match information"""
    format: str  # T20|ODI|Test|Other
    start_time_local: str
    end_time_local: str
    timezone: str
    toss_won_by: str
    toss_decision: str  # bat|bowl|unknown
    venue_enrichment: VenueEnrichment
    team_enrichments: Dict[str, TeamEnrichment]  # team_name -> TeamEnrichment
    confidence_score: float
    enrichment_status: str

class EnrichedKGBuilder:
    """
    Builds an enriched cricket knowledge graph by combining:
    1. Existing unified KG (players, matches, venues, balls)
    2. OpenAI-enriched match data (weather, squads, coordinates)
    """
    
    def __init__(self, base_kg_path: str, enriched_data_path: str):
        self.base_kg_path = Path(base_kg_path)
        self.enriched_data_path = Path(enriched_data_path)
        self.graph = None
        self.enrichments = {}  # match_key -> MatchEnrichment
        
    def load_base_kg(self) -> nx.Graph:
        """Load the existing unified cricket KG"""
        logger.info(f"📊 Loading base KG from {self.base_kg_path}")
        
        if not self.base_kg_path.exists():
            raise FileNotFoundError(f"Base KG not found: {self.base_kg_path}")
        
        with open(self.base_kg_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        logger.info(f"✅ Loaded base KG: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        return self.graph
    
    def load_enriched_data(self) -> Dict[str, MatchEnrichment]:
        """Load OpenAI-enriched match data"""
        logger.info(f"🌤️ Loading enriched data from {self.enriched_data_path}")
        
        if not self.enriched_data_path.exists():
            logger.warning(f"Enriched data not found: {self.enriched_data_path}")
            return {}
        
        with open(self.enriched_data_path, 'r') as f:
            enriched_matches = json.load(f)
        
        # Convert to MatchEnrichment objects
        for match_data in enriched_matches:
            try:
                match_key = self._create_match_key(match_data)
                enrichment = self._parse_match_enrichment(match_data)
                self.enrichments[match_key] = enrichment
            except Exception as e:
                logger.warning(f"Failed to parse match enrichment: {e}")
        
        logger.info(f"✅ Loaded {len(self.enrichments)} enriched matches")
        return self.enrichments
    
    def _create_match_key(self, match_data: Dict) -> str:
        """Create a unique key for a match"""
        import hashlib
        
        # Extract teams and venue from match data
        teams = match_data.get('teams', [])
        venue_name = match_data.get('venue', {}).get('name', '')
        date = match_data.get('date', '')
        competition = match_data.get('competition', '')
        
        home_team = ''
        away_team = ''
        for team in teams:
            if team.get('is_home'):
                home_team = team.get('name', '')
            else:
                away_team = team.get('name', '')
        
        key_string = f"{home_team}|{away_team}|{venue_name}|{date}|{competition}".lower()
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _parse_match_enrichment(self, match_data: Dict) -> MatchEnrichment:
        """Parse enriched match data into structured format"""
        
        # Parse venue enrichment
        venue_data = match_data.get('venue', {})
        weather_data = match_data.get('weather_hourly', [])
        
        weather_conditions = []
        for weather in weather_data:
            weather_conditions.append(WeatherConditions(
                temperature_c=weather.get('temperature_c', 0.0),
                feels_like_c=weather.get('feels_like_c', 0.0),
                humidity_pct=weather.get('humidity_pct', 0),
                wind_speed_kph=weather.get('wind_speed_kph', 0.0),
                wind_gust_kph=weather.get('wind_gust_kph', 0.0),
                wind_dir_deg=weather.get('wind_dir_deg', 0),
                precip_mm=weather.get('precip_mm', 0.0),
                precip_prob_pct=weather.get('precip_prob_pct', 0),
                cloud_cover_pct=weather.get('cloud_cover_pct', 0),
                pressure_hpa=weather.get('pressure_hpa', 1013),
                uv_index=weather.get('uv_index', 0.0),
                weather_code=weather.get('weather_code', 'unknown')
            ))
        
        venue_enrichment = VenueEnrichment(
            latitude=venue_data.get('latitude', 0.0),
            longitude=venue_data.get('longitude', 0.0),
            timezone=match_data.get('timezone', 'UTC'),
            city=venue_data.get('city', 'unknown'),
            country=venue_data.get('country', 'unknown'),
            weather_conditions=weather_conditions
        )
        
        # Parse team enrichments
        team_enrichments = {}
        for team_data in match_data.get('teams', []):
            team_name = team_data.get('name', '')
            
            # Parse squad information
            squad = {}
            for player_data in team_data.get('players', []):
                player_name = player_data.get('name', '')
                squad[player_name] = PlayerSquadInfo(
                    role=player_data.get('role', 'unknown'),
                    batting_style=player_data.get('batting_style', 'unknown'),
                    bowling_style=player_data.get('bowling_style', 'unknown'),
                    captain=player_data.get('captain', False),
                    wicket_keeper=player_data.get('wicket_keeper', False),
                    playing_xi=player_data.get('playing_xi', True)
                )
            
            team_enrichments[team_name] = TeamEnrichment(
                official_name=team_data.get('name', ''),
                short_name=team_data.get('short_name', ''),
                is_home=team_data.get('is_home', False),
                squad=squad
            )
        
        # Parse toss information
        toss_data = match_data.get('toss', {})
        
        return MatchEnrichment(
            format=match_data.get('format', 'T20'),
            start_time_local=match_data.get('start_time_local', ''),
            end_time_local=match_data.get('end_time_local', ''),
            timezone=match_data.get('timezone', 'UTC'),
            toss_won_by=toss_data.get('won_by', 'unknown'),
            toss_decision=toss_data.get('decision', 'unknown'),
            venue_enrichment=venue_enrichment,
            team_enrichments=team_enrichments,
            confidence_score=match_data.get('confidence_score', 0.0),
            enrichment_status=match_data.get('enrichment_status', 'unknown')
        )
    
    def enrich_knowledge_graph(self, 
                              progress_callback: Optional[Callable[[str, str, int], None]] = None) -> nx.Graph:
        """Enrich the base KG with weather, team, and venue data"""
        
        if not self.graph:
            raise ValueError("Base KG not loaded. Call load_base_kg() first.")
        
        logger.info("🚀 Starting knowledge graph enrichment...")
        
        # Track enrichment statistics
        stats = {
            'venues_enriched': 0,
            'weather_nodes_added': 0,
            'team_nodes_added': 0,
            'player_roles_updated': 0,
            'coordinates_added': 0
        }
        
        total_enrichments = len(self.enrichments)
        
        for idx, (match_key, enrichment) in enumerate(self.enrichments.items()):
            try:
                # Update progress
                if progress_callback:
                    progress = int((idx / total_enrichments) * 100)
                    progress_callback("enriching_kg", f"Processing match {idx+1}/{total_enrichments}", progress)
                
                # Enrich venue nodes
                self._enrich_venue_nodes(enrichment, stats)
                
                # Add weather nodes
                self._add_weather_nodes(match_key, enrichment, stats)
                
                # Enrich team information
                self._enrich_team_nodes(enrichment, stats)
                
                # Update player roles and information
                self._update_player_information(enrichment, stats)
                
            except Exception as e:
                logger.warning(f"Failed to enrich match {match_key}: {e}")
        
        # Add summary statistics to graph
        self.graph.graph['enrichment_stats'] = stats
        self.graph.graph['enrichment_timestamp'] = datetime.now().isoformat()
        
        logger.info("✅ Knowledge graph enrichment complete!")
        logger.info(f"📊 Enrichment statistics: {stats}")
        
        return self.graph
    
    def _enrich_venue_nodes(self, enrichment: MatchEnrichment, stats: Dict[str, int]):
        """Add coordinates and timezone to venue nodes"""
        venue_enrichment = enrichment.venue_enrichment
        venue_name = None
        
        # Find venue node in graph
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('type') == 'venue':
                # Try to match venue name (this could be improved with fuzzy matching)
                if venue_enrichment.city.lower() in node_id.lower() or node_id.lower() in venue_enrichment.city.lower():
                    venue_name = node_id
                    break
        
        if venue_name:
            # Update venue node with enriched data
            self.graph.nodes[venue_name].update({
                'latitude': venue_enrichment.latitude,
                'longitude': venue_enrichment.longitude,
                'timezone': venue_enrichment.timezone,
                'city': venue_enrichment.city,
                'country': venue_enrichment.country,
                'coordinates_available': venue_enrichment.latitude != 0.0
            })
            
            if venue_enrichment.latitude != 0.0:
                stats['coordinates_added'] += 1
            stats['venues_enriched'] += 1
    
    def _add_weather_nodes(self, match_key: str, enrichment: MatchEnrichment, stats: Dict[str, int]):
        """Add weather condition nodes for each match"""
        weather_conditions = enrichment.venue_enrichment.weather_conditions
        
        for idx, weather in enumerate(weather_conditions):
            weather_node_id = f"weather_{match_key}_{idx}"
            
            self.graph.add_node(weather_node_id, 
                type='weather',
                match_key=match_key,
                temperature_c=weather.temperature_c,
                feels_like_c=weather.feels_like_c,
                humidity_pct=weather.humidity_pct,
                wind_speed_kph=weather.wind_speed_kph,
                wind_gust_kph=weather.wind_gust_kph,
                wind_dir_deg=weather.wind_dir_deg,
                precip_mm=weather.precip_mm,
                precip_prob_pct=weather.precip_prob_pct,
                cloud_cover_pct=weather.cloud_cover_pct,
                pressure_hpa=weather.pressure_hpa,
                uv_index=weather.uv_index,
                weather_code=weather.weather_code
            )
            
            stats['weather_nodes_added'] += 1
    
    def _enrich_team_nodes(self, enrichment: MatchEnrichment, stats: Dict[str, int]):
        """Add or update team nodes with squad information"""
        for team_name, team_enrichment in enrichment.team_enrichments.items():
            # Add team node if it doesn't exist
            if team_name not in self.graph.nodes:
                self.graph.add_node(team_name, type='team')
                stats['team_nodes_added'] += 1
            
            # Update team node with enriched data
            self.graph.nodes[team_name].update({
                'official_name': team_enrichment.official_name,
                'short_name': team_enrichment.short_name,
                'squad_size': len(team_enrichment.squad),
                'captains': [name for name, info in team_enrichment.squad.items() if info.captain],
                'wicket_keepers': [name for name, info in team_enrichment.squad.items() if info.wicket_keeper]
            })
    
    def _update_player_information(self, enrichment: MatchEnrichment, stats: Dict[str, int]):
        """Update player nodes with role and style information"""
        for team_name, team_enrichment in enrichment.team_enrichments.items():
            for player_name, player_info in team_enrichment.squad.items():
                # Find player node in graph (could be improved with fuzzy matching)
                player_node = None
                for node_id, node_data in self.graph.nodes(data=True):
                    if (node_data.get('type') == 'player' and 
                        (player_name.lower() in node_id.lower() or node_id.lower() in player_name.lower())):
                        player_node = node_id
                        break
                
                if player_node:
                    # Update player node with enriched data
                    current_data = self.graph.nodes[player_node]
                    
                    # Only update if we have better information
                    updates = {}
                    if player_info.role != 'unknown' and current_data.get('role', 'unknown') == 'unknown':
                        updates['role'] = player_info.role
                    if player_info.batting_style != 'unknown' and current_data.get('batting_style', 'unknown') == 'unknown':
                        updates['batting_style'] = player_info.batting_style
                    if player_info.bowling_style != 'unknown' and current_data.get('bowling_style', 'unknown') == 'unknown':
                        updates['bowling_style'] = player_info.bowling_style
                    
                    updates.update({
                        'captain_experience': current_data.get('captain_experience', 0) + (1 if player_info.captain else 0),
                        'wicket_keeper_experience': current_data.get('wicket_keeper_experience', 0) + (1 if player_info.wicket_keeper else 0)
                    })
                    
                    self.graph.nodes[player_node].update(updates)
                    
                    if updates:
                        stats['player_roles_updated'] += 1
    
    def save_enriched_kg(self, output_path: str):
        """Save the enriched knowledge graph"""
        logger.info(f"💾 Saving enriched KG to {output_path}")
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f)
        
        logger.info("✅ Enriched KG saved successfully")
    
    def get_enrichment_summary(self) -> Dict[str, Any]:
        """Get summary of enrichment process"""
        if not self.graph:
            return {"error": "No graph loaded"}
        
        # Count different node types
        node_counts = {}
        weather_nodes = 0
        venues_with_coords = 0
        players_with_roles = 0
        
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('type', 'unknown')
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
            
            if node_type == 'weather':
                weather_nodes += 1
            elif node_type == 'venue' and node_data.get('coordinates_available', False):
                venues_with_coords += 1
            elif node_type == 'player' and node_data.get('role', 'unknown') != 'unknown':
                players_with_roles += 1
        
        return {
            'total_nodes': len(self.graph.nodes),
            'total_edges': len(self.graph.edges),
            'node_type_counts': node_counts,
            'weather_nodes': weather_nodes,
            'venues_with_coordinates': venues_with_coords,
            'players_with_roles': players_with_roles,
            'enrichments_loaded': len(self.enrichments),
            'enrichment_stats': self.graph.graph.get('enrichment_stats', {}),
            'enrichment_timestamp': self.graph.graph.get('enrichment_timestamp')
        }

def main():
    """Example usage of the enriched KG builder"""
    
    # Paths
    base_kg_path = "models/unified_cricket_kg.pkl"
    enriched_data_path = "enriched_data/enriched_betting_matches.json"
    output_path = "models/enriched_cricket_kg.pkl"
    
    # Build enriched KG
    builder = EnrichedKGBuilder(base_kg_path, enriched_data_path)
    
    try:
        # Load base KG and enriched data
        builder.load_base_kg()
        builder.load_enriched_data()
        
        # Enrich the knowledge graph
        enriched_kg = builder.enrich_knowledge_graph()
        
        # Save enriched KG
        builder.save_enriched_kg(output_path)
        
        # Print summary
        summary = builder.get_enrichment_summary()
        print("\n🎯 ENRICHMENT SUMMARY:")
        print("=" * 50)
        for key, value in summary.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Failed to build enriched KG: {e}")
        raise

if __name__ == "__main__":
    main()
