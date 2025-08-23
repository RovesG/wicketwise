# Purpose: Unified entity harmonization system for players, teams, and venues across all models
# Author: WicketWise Team, Last Modified: 2025-01-22

"""
Unified Entity Harmonization System

This module provides consistent entity resolution across:
- Knowledge Graph (KG) 
- Graph Neural Network (GNN)
- T20 Model (Crickformer)
- Betting Intelligence System
- UI Components

Key Features:
- Master player registry from people.csv
- Team name standardization with aliases
- Venue name normalization with coordinates
- Fuzzy matching with confidence scoring
- Cache-based performance optimization
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from difflib import SequenceMatcher
import pickle
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class EntityMapping:
    """Represents a mapping between different entity names"""
    canonical_name: str
    aliases: List[str]
    confidence: float
    source: str  # 'exact', 'fuzzy', 'manual'
    metadata: Dict[str, Any] = None

@dataclass 
class PlayerEntity:
    """Standardized player entity"""
    identifier: str  # Unique ID from people.csv
    canonical_name: str
    unique_name: str
    aliases: List[str]
    birth_date: Optional[str] = None
    country: Optional[str] = None
    batting_style: Optional[str] = None
    bowling_style: Optional[str] = None

@dataclass
class TeamEntity:
    """Standardized team entity"""
    identifier: str
    canonical_name: str
    short_name: str
    aliases: List[str]
    country: Optional[str] = None
    competition: Optional[str] = None
    founded: Optional[str] = None

@dataclass
class VenueEntity:
    """Standardized venue entity"""
    identifier: str
    canonical_name: str
    city: str
    country: str
    aliases: List[str]
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    capacity: Optional[int] = None


class EntityHarmonizer:
    """
    Unified entity harmonization system that provides consistent
    entity resolution across all WicketWise components.
    """
    
    def __init__(self, 
                 people_csv_path: str = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/people.csv",
                 cache_dir: str = "cache/entity_harmonizer"):
        
        self.people_csv_path = people_csv_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Entity registries
        self.players: Dict[str, PlayerEntity] = {}
        self.teams: Dict[str, TeamEntity] = {}
        self.venues: Dict[str, VenueEntity] = {}
        
        # Mapping caches for performance
        self.player_mappings: Dict[str, str] = {}  # variant -> canonical
        self.team_mappings: Dict[str, str] = {}
        self.venue_mappings: Dict[str, str] = {}
        
        # Configuration
        self.fuzzy_threshold = 0.85
        self.cache_file = self.cache_dir / "entity_mappings.pkl"
        
        # Initialize registries
        self._initialize_registries()
        
    def _initialize_registries(self):
        """Initialize all entity registries"""
        logger.info("🔧 Initializing Entity Harmonization System...")
        
        # Load from cache if available
        if self._load_from_cache():
            logger.info("✅ Loaded entity mappings from cache")
            return
            
        # Build registries from scratch
        self._build_player_registry()
        self._build_team_registry()
        self._build_venue_registry()
        
        # Save to cache
        self._save_to_cache()
        logger.info("✅ Entity Harmonization System initialized")
    
    def _build_player_registry(self):
        """Build player registry from people.csv"""
        try:
            df = pd.read_csv(self.people_csv_path)
            logger.info(f"📊 Loading {len(df)} players from {self.people_csv_path}")
            
            for _, row in df.iterrows():
                identifier = str(row.get('identifier', ''))
                name = str(row.get('name', ''))
                unique_name = str(row.get('unique_name', name))
                
                if identifier and name:
                    # Create aliases from various name formats
                    aliases = self._generate_player_aliases(name, unique_name)
                    
                    player = PlayerEntity(
                        identifier=identifier,
                        canonical_name=name,
                        unique_name=unique_name,
                        aliases=aliases,
                        birth_date=row.get('birth_date'),
                        country=row.get('country'),
                        batting_style=row.get('batting_style'),
                        bowling_style=row.get('bowling_style')
                    )
                    
                    self.players[identifier] = player
                    
                    # Build mapping cache
                    self.player_mappings[name.lower()] = identifier
                    self.player_mappings[unique_name.lower()] = identifier
                    for alias in aliases:
                        self.player_mappings[alias.lower()] = identifier
                        
            logger.info(f"✅ Built player registry: {len(self.players)} players, {len(self.player_mappings)} mappings")
            
        except Exception as e:
            logger.error(f"❌ Error building player registry: {e}")
            self._build_fallback_player_registry()
    
    def _generate_player_aliases(self, name: str, unique_name: str) -> List[str]:
        """Generate common aliases for a player name"""
        aliases = set()
        
        # Add the names themselves
        aliases.add(name)
        aliases.add(unique_name)
        
        # Common variations
        parts = name.split()
        if len(parts) >= 2:
            # Last name only
            aliases.add(parts[-1])
            # First + Last initial
            aliases.add(f"{parts[0]} {parts[-1][0]}")
            # Initials + Last name
            if len(parts) >= 3:
                initials = "".join([p[0] for p in parts[:-1]])
                aliases.add(f"{initials} {parts[-1]}")
        
        # Common cricket nicknames
        nickname_map = {
            'Virat Kohli': ['Kohli', 'VK', 'Kohl'],
            'MS Dhoni': ['Dhoni', 'MSD', 'Captain Cool'],
            'Rohit Sharma': ['Rohit', 'Hitman'],
            'AB de Villiers': ['ABD', 'AB', 'Mr 360'],
            'Suryakumar Yadav': ['SKY', 'Surya'],
            'KL Rahul': ['Rahul', 'KL'],
            'Hardik Pandya': ['Hardik', 'HP'],
            'Jasprit Bumrah': ['Bumrah', 'Boom Boom'],
            'Rishabh Pant': ['Pant', 'RP']
        }
        
        if name in nickname_map:
            aliases.update(nickname_map[name])
            
        return list(aliases)
    
    def _build_team_registry(self):
        """Build team registry with common cricket teams"""
        teams_data = [
            {
                'identifier': 'csk',
                'canonical_name': 'Chennai Super Kings',
                'short_name': 'CSK',
                'aliases': ['Chennai Super Kings', 'CSK', 'Chennai', 'Super Kings'],
                'country': 'India',
                'competition': 'IPL'
            },
            {
                'identifier': 'mi',
                'canonical_name': 'Mumbai Indians',
                'short_name': 'MI',
                'aliases': ['Mumbai Indians', 'MI', 'Mumbai'],
                'country': 'India',
                'competition': 'IPL'
            },
            {
                'identifier': 'rcb',
                'canonical_name': 'Royal Challengers Bangalore',
                'short_name': 'RCB',
                'aliases': ['Royal Challengers Bangalore', 'RCB', 'Royal Challengers', 'Bangalore'],
                'country': 'India',
                'competition': 'IPL'
            },
            {
                'identifier': 'dc',
                'canonical_name': 'Delhi Capitals',
                'short_name': 'DC',
                'aliases': ['Delhi Capitals', 'DC', 'Delhi', 'Delhi Daredevils', 'DD'],
                'country': 'India',
                'competition': 'IPL'
            },
            {
                'identifier': 'kkr',
                'canonical_name': 'Kolkata Knight Riders',
                'short_name': 'KKR',
                'aliases': ['Kolkata Knight Riders', 'KKR', 'Kolkata', 'Knight Riders'],
                'country': 'India',
                'competition': 'IPL'
            },
            {
                'identifier': 'pbks',
                'canonical_name': 'Punjab Kings',
                'short_name': 'PBKS',
                'aliases': ['Punjab Kings', 'PBKS', 'Punjab', 'Kings XI Punjab', 'KXIP'],
                'country': 'India',
                'competition': 'IPL'
            },
            {
                'identifier': 'rr',
                'canonical_name': 'Rajasthan Royals',
                'short_name': 'RR',
                'aliases': ['Rajasthan Royals', 'RR', 'Rajasthan', 'Royals'],
                'country': 'India',
                'competition': 'IPL'
            },
            {
                'identifier': 'srh',
                'canonical_name': 'Sunrisers Hyderabad',
                'short_name': 'SRH',
                'aliases': ['Sunrisers Hyderabad', 'SRH', 'Sunrisers', 'Hyderabad'],
                'country': 'India',
                'competition': 'IPL'
            },
            # International teams
            {
                'identifier': 'ind',
                'canonical_name': 'India',
                'short_name': 'IND',
                'aliases': ['India', 'IND', 'Team India'],
                'country': 'India',
                'competition': 'International'
            },
            {
                'identifier': 'aus',
                'canonical_name': 'Australia',
                'short_name': 'AUS',
                'aliases': ['Australia', 'AUS', 'Aussies'],
                'country': 'Australia',
                'competition': 'International'
            },
            {
                'identifier': 'eng',
                'canonical_name': 'England',
                'short_name': 'ENG',
                'aliases': ['England', 'ENG'],
                'country': 'England',
                'competition': 'International'
            },
            # BBL teams
            {
                'identifier': 'ms',
                'canonical_name': 'Melbourne Stars',
                'short_name': 'MS',
                'aliases': ['Melbourne Stars', 'MS', 'Stars'],
                'country': 'Australia',
                'competition': 'BBL'
            },
            {
                'identifier': 'mr',
                'canonical_name': 'Melbourne Renegades',
                'short_name': 'MR',
                'aliases': ['Melbourne Renegades', 'MR', 'Renegades'],
                'country': 'Australia',
                'competition': 'BBL'
            }
        ]
        
        for team_data in teams_data:
            team = TeamEntity(**team_data)
            self.teams[team.identifier] = team
            
            # Build mapping cache
            for alias in team.aliases:
                self.team_mappings[alias.lower()] = team.identifier
                
        logger.info(f"✅ Built team registry: {len(self.teams)} teams, {len(self.team_mappings)} mappings")
    
    def _build_venue_registry(self):
        """Build venue registry with common cricket venues"""
        venues_data = [
            {
                'identifier': 'wankhede',
                'canonical_name': 'Wankhede Stadium',
                'city': 'Mumbai',
                'country': 'India',
                'aliases': ['Wankhede Stadium', 'Wankhede', 'Mumbai'],
                'latitude': 18.9388,
                'longitude': 72.8258,
                'capacity': 33108
            },
            {
                'identifier': 'mcg',
                'canonical_name': 'Melbourne Cricket Ground',
                'city': 'Melbourne',
                'country': 'Australia',
                'aliases': ['Melbourne Cricket Ground', 'MCG', 'Melbourne'],
                'latitude': -37.8200,
                'longitude': 144.9834,
                'capacity': 100024
            },
            {
                'identifier': 'lords',
                'canonical_name': "Lord's Cricket Ground",
                'city': 'London',
                'country': 'England',
                'aliases': ["Lord's Cricket Ground", "Lord's", "Lords", "London"],
                'latitude': 51.5294,
                'longitude': -0.1728,
                'capacity': 31100
            },
            {
                'identifier': 'eden_gardens',
                'canonical_name': 'Eden Gardens',
                'city': 'Kolkata',
                'country': 'India',
                'aliases': ['Eden Gardens', 'Eden', 'Kolkata'],
                'latitude': 22.5645,
                'longitude': 88.3433,
                'capacity': 66000
            },
            {
                'identifier': 'chinnaswamy',
                'canonical_name': 'M. Chinnaswamy Stadium',
                'city': 'Bangalore',
                'country': 'India',
                'aliases': ['M. Chinnaswamy Stadium', 'Chinnaswamy Stadium', 'Chinnaswamy', 'Bangalore'],
                'latitude': 12.9788,
                'longitude': 77.5996,
                'capacity': 40000
            }
        ]
        
        for venue_data in venues_data:
            venue = VenueEntity(**venue_data)
            self.venues[venue.identifier] = venue
            
            # Build mapping cache
            for alias in venue.aliases:
                self.venue_mappings[alias.lower()] = venue.identifier
                
        logger.info(f"✅ Built venue registry: {len(self.venues)} venues, {len(self.venue_mappings)} mappings")
    
    def _build_fallback_player_registry(self):
        """Build minimal player registry as fallback"""
        fallback_players = [
            {'identifier': 'virat_kohli', 'name': 'Virat Kohli', 'unique_name': 'Virat Kohli'},
            {'identifier': 'ms_dhoni', 'name': 'MS Dhoni', 'unique_name': 'MS Dhoni'},
            {'identifier': 'rohit_sharma', 'name': 'Rohit Sharma', 'unique_name': 'Rohit Sharma'},
            {'identifier': 'kl_rahul', 'name': 'KL Rahul', 'unique_name': 'KL Rahul'},
            {'identifier': 'hardik_pandya', 'name': 'Hardik Pandya', 'unique_name': 'Hardik Pandya'}
        ]
        
        for player_data in fallback_players:
            aliases = self._generate_player_aliases(player_data['name'], player_data['unique_name'])
            player = PlayerEntity(
                identifier=player_data['identifier'],
                canonical_name=player_data['name'],
                unique_name=player_data['unique_name'],
                aliases=aliases
            )
            
            self.players[player.identifier] = player
            
            # Build mapping cache
            for alias in aliases:
                self.player_mappings[alias.lower()] = player.identifier
                
        logger.info(f"✅ Built fallback player registry: {len(self.players)} players")
    
    def resolve_player(self, name: str) -> Optional[PlayerEntity]:
        """Resolve a player name to canonical entity"""
        if not name:
            return None
            
        name_lower = name.lower().strip()
        
        # Exact match first
        if name_lower in self.player_mappings:
            identifier = self.player_mappings[name_lower]
            return self.players.get(identifier)
        
        # Fuzzy matching
        best_match = None
        best_score = 0.0
        
        for mapped_name, identifier in self.player_mappings.items():
            score = SequenceMatcher(None, name_lower, mapped_name).ratio()
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = identifier
        
        if best_match:
            # Cache the fuzzy match for future use
            self.player_mappings[name_lower] = best_match
            return self.players.get(best_match)
            
        return None
    
    def resolve_team(self, name: str) -> Optional[TeamEntity]:
        """Resolve a team name to canonical entity"""
        if not name:
            return None
            
        name_lower = name.lower().strip()
        
        # Exact match first
        if name_lower in self.team_mappings:
            identifier = self.team_mappings[name_lower]
            return self.teams.get(identifier)
        
        # Fuzzy matching
        best_match = None
        best_score = 0.0
        
        for mapped_name, identifier in self.team_mappings.items():
            score = SequenceMatcher(None, name_lower, mapped_name).ratio()
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = identifier
        
        if best_match:
            # Cache the fuzzy match for future use
            self.team_mappings[name_lower] = best_match
            return self.teams.get(best_match)
            
        return None
    
    def resolve_venue(self, name: str) -> Optional[VenueEntity]:
        """Resolve a venue name to canonical entity"""
        if not name:
            return None
            
        name_lower = name.lower().strip()
        
        # Exact match first
        if name_lower in self.venue_mappings:
            identifier = self.venue_mappings[name_lower]
            return self.venues.get(identifier)
        
        # Fuzzy matching
        best_match = None
        best_score = 0.0
        
        for mapped_name, identifier in self.venue_mappings.items():
            score = SequenceMatcher(None, name_lower, mapped_name).ratio()
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = identifier
        
        if best_match:
            # Cache the fuzzy match for future use
            self.venue_mappings[name_lower] = best_match
            return self.venues.get(best_match)
            
        return None
    
    def get_canonical_name(self, entity_type: str, name: str) -> Optional[str]:
        """Get canonical name for any entity type"""
        if entity_type == 'player':
            entity = self.resolve_player(name)
            return entity.canonical_name if entity else None
        elif entity_type == 'team':
            entity = self.resolve_team(name)
            return entity.canonical_name if entity else None
        elif entity_type == 'venue':
            entity = self.resolve_venue(name)
            return entity.canonical_name if entity else None
        else:
            return None
    
    def harmonize_dataset(self, df: pd.DataFrame, 
                         player_columns: List[str] = None,
                         team_columns: List[str] = None,
                         venue_columns: List[str] = None) -> pd.DataFrame:
        """Harmonize entity names in a dataset"""
        df_harmonized = df.copy()
        
        # Default column mappings
        if player_columns is None:
            player_columns = ['batter_name', 'bowler_name', 'player', 'batsman', 'bowler']
        if team_columns is None:
            team_columns = ['team_batting', 'team_bowling', 'team', 'batting_team', 'bowling_team']
        if venue_columns is None:
            venue_columns = ['venue', 'ground', 'stadium']
        
        # Harmonize player names
        for col in player_columns:
            if col in df_harmonized.columns:
                logger.info(f"🔧 Harmonizing player names in column: {col}")
                df_harmonized[col] = df_harmonized[col].apply(
                    lambda x: self.get_canonical_name('player', str(x)) if pd.notna(x) else x
                )
        
        # Harmonize team names
        for col in team_columns:
            if col in df_harmonized.columns:
                logger.info(f"🔧 Harmonizing team names in column: {col}")
                df_harmonized[col] = df_harmonized[col].apply(
                    lambda x: self.get_canonical_name('team', str(x)) if pd.notna(x) else x
                )
        
        # Harmonize venue names
        for col in venue_columns:
            if col in df_harmonized.columns:
                logger.info(f"🔧 Harmonizing venue names in column: {col}")
                df_harmonized[col] = df_harmonized[col].apply(
                    lambda x: self.get_canonical_name('venue', str(x)) if pd.notna(x) else x
                )
        
        return df_harmonized
    
    def get_entity_stats(self) -> Dict[str, Any]:
        """Get statistics about entity registries"""
        return {
            'players': {
                'total': len(self.players),
                'mappings': len(self.player_mappings)
            },
            'teams': {
                'total': len(self.teams),
                'mappings': len(self.team_mappings)
            },
            'venues': {
                'total': len(self.venues),
                'mappings': len(self.venue_mappings)
            }
        }
    
    def _save_to_cache(self):
        """Save entity mappings to cache"""
        try:
            cache_data = {
                'players': self.players,
                'teams': self.teams,
                'venues': self.venues,
                'player_mappings': self.player_mappings,
                'team_mappings': self.team_mappings,
                'venue_mappings': self.venue_mappings
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            logger.warning(f"Failed to save entity cache: {e}")
    
    def _load_from_cache(self) -> bool:
        """Load entity mappings from cache"""
        try:
            if not self.cache_file.exists():
                return False
                
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.players = cache_data.get('players', {})
            self.teams = cache_data.get('teams', {})
            self.venues = cache_data.get('venues', {})
            self.player_mappings = cache_data.get('player_mappings', {})
            self.team_mappings = cache_data.get('team_mappings', {})
            self.venue_mappings = cache_data.get('venue_mappings', {})
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load entity cache: {e}")
            return False


# Global instance for easy access
_global_harmonizer = None

def get_entity_harmonizer() -> EntityHarmonizer:
    """Get global EntityHarmonizer instance"""
    global _global_harmonizer
    if _global_harmonizer is None:
        _global_harmonizer = EntityHarmonizer()
    return _global_harmonizer


if __name__ == "__main__":
    # Test the harmonizer
    harmonizer = EntityHarmonizer()
    
    # Test player resolution
    test_names = ['Kohli', 'Kohl', 'VK', 'MS Dhoni', 'SKY', 'ABD']
    print("\n🧪 Testing Player Resolution:")
    for name in test_names:
        player = harmonizer.resolve_player(name)
        if player:
            print(f"  '{name}' → {player.canonical_name} ({player.identifier})")
        else:
            print(f"  '{name}' → Not found")
    
    # Test team resolution
    test_teams = ['CSK', 'Mumbai Indians', 'RCB', 'Australia']
    print("\n🧪 Testing Team Resolution:")
    for name in test_teams:
        team = harmonizer.resolve_team(name)
        if team:
            print(f"  '{name}' → {team.canonical_name} ({team.identifier})")
        else:
            print(f"  '{name}' → Not found")
    
    # Print stats
    stats = harmonizer.get_entity_stats()
    print(f"\n📊 Entity Registry Stats:")
    print(f"  Players: {stats['players']['total']} entities, {stats['players']['mappings']} mappings")
    print(f"  Teams: {stats['teams']['total']} entities, {stats['teams']['mappings']} mappings")  
    print(f"  Venues: {stats['venues']['total']} entities, {stats['venues']['mappings']} mappings")
