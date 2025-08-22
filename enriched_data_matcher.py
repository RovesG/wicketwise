#!/usr/bin/env python3
"""
Enhanced Fuzzy Matching System for Cricket Data Integration
Matches enriched data to KG and ML datasets with high accuracy

Author: WicketWise Team, Last Modified: 2025-01-21
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class MatchCandidate:
    """Represents a potential match between datasets"""
    enriched_key: str
    dataset_key: str
    similarity_score: float
    match_type: str  # 'exact', 'fuzzy', 'venue_alias'
    confidence: str  # 'high', 'medium', 'low'

class EnrichedDataMatcher:
    """
    Advanced matching system for integrating enriched match data with 
    Knowledge Graph and ML datasets using multiple strategies
    """
    
    def __init__(self):
        self.team_aliases = self._load_team_aliases()
        self.venue_aliases = self._load_venue_aliases()
        self.player_aliases = self._load_player_aliases()
        
    def _load_team_aliases(self) -> Dict[str, List[str]]:
        """Load known team name variations"""
        return {
            'Royal Challengers Bangalore': [
                'Royal Challengers Bengaluru', 'RCB', 'Bangalore', 'Bengaluru'
            ],
            'Chennai Super Kings': [
                'CSK', 'Chennai', 'Super Kings'
            ],
            'Mumbai Indians': [
                'MI', 'Mumbai'
            ],
            'Kolkata Knight Riders': [
                'KKR', 'Kolkata', 'Knight Riders'
            ],
            'Delhi Capitals': [
                'DC', 'Delhi Daredevils', 'DD', 'Delhi'
            ],
            'Rajasthan Royals': [
                'RR', 'Rajasthan'
            ],
            'Punjab Kings': [
                'PBKS', 'Kings XI Punjab', 'KXIP', 'Punjab'
            ],
            'Sunrisers Hyderabad': [
                'SRH', 'Hyderabad', 'Sunrisers'
            ],
            'Australia': ['AUS'],
            'India': ['IND'],
            'England': ['ENG'],
            'South Africa': ['SA', 'RSA'],
            'New Zealand': ['NZ'],
            'Pakistan': ['PAK'],
            'Sri Lanka': ['SL', 'SLA'],
            'Bangladesh': ['BAN'],
            'West Indies': ['WI']
        }
    
    def _load_venue_aliases(self) -> Dict[str, List[str]]:
        """Load known venue name variations"""
        return {
            'M Chinnaswamy Stadium': [
                'M. Chinnaswamy Stadium', 'Chinnaswamy Stadium', 'Chinnaswamy', 'Bangalore'
            ],
            'Wankhede Stadium': [
                'Wankhede', 'Mumbai'
            ],
            'Eden Gardens': [
                'Eden', 'Kolkata'
            ],
            'Arun Jaitley Stadium': [
                'Feroz Shah Kotla', 'Kotla', 'Delhi'
            ],
            'Rajiv Gandhi International Stadium': [
                'Rajiv Gandhi Stadium', 'Hyderabad', 'Uppal'
            ],
            'Sawai Mansingh Stadium': [
                'SMS Stadium', 'Jaipur'
            ],
            'PCA Stadium': [
                'IS Bindra Stadium', 'Mohali'
            ],
            'Dubai International Cricket Stadium': [
                'Dubai Cricket Stadium', 'Dubai'
            ],
            'Melbourne Cricket Ground': [
                'MCG', 'Melbourne'
            ],
            'Sydney Cricket Ground': [
                'SCG', 'Sydney'
            ],
            'Lord\'s': [
                'Lords', 'Lord\'s Cricket Ground'
            ]
        }
    
    def _load_player_aliases(self) -> Dict[str, List[str]]:
        """Load common player name variations"""
        return {
            'Virat Kohli': ['V Kohli', 'Kohli', 'Kohl'],
            'MS Dhoni': ['M Dhoni', 'Dhoni', 'Mahendra Singh Dhoni'],
            'Rohit Sharma': ['R Sharma', 'Rohit'],
            'AB de Villiers': ['ABD', 'de Villiers', 'A de Villiers'],
            'Suryakumar Yadav': ['SKY', 'S Yadav', 'Surya'],
            'KL Rahul': ['Rahul', 'K Rahul'],
            'Hardik Pandya': ['H Pandya', 'Hardik'],
            'Jasprit Bumrah': ['J Bumrah', 'Bumrah'],
            'Rishabh Pant': ['R Pant', 'Pant']
        }
    
    def normalize_name(self, name: str) -> str:
        """Normalize names for better matching"""
        if not name:
            return ""
        
        # Basic normalization
        normalized = str(name).strip().lower()
        
        # Remove common prefixes/suffixes
        prefixes = ['captain', 'mr', 'dr', 'sir']
        suffixes = ['jr', 'sr', 'ii', 'iii']
        
        for prefix in prefixes:
            if normalized.startswith(prefix + ' '):
                normalized = normalized[len(prefix):].strip()
        
        for suffix in suffixes:
            if normalized.endswith(' ' + suffix):
                normalized = normalized[:-len(suffix)].strip()
        
        # Remove extra spaces
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def calculate_fuzzy_similarity(self, name1: str, name2: str) -> float:
        """Calculate fuzzy similarity with cricket-specific enhancements"""
        if not name1 or not name2:
            return 0.0
        
        norm1 = self.normalize_name(name1)
        norm2 = self.normalize_name(name2)
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Sequence matching
        seq_sim = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Word-based matching for multi-word names
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if words1 and words2:
            word_overlap = len(words1.intersection(words2))
            word_union = len(words1.union(words2))
            word_sim = word_overlap / word_union if word_union > 0 else 0.0
        else:
            word_sim = 0.0
        
        # Combine similarities
        combined_sim = 0.6 * seq_sim + 0.4 * word_sim
        
        return combined_sim
    
    def find_team_matches(self, enriched_teams: List[str], dataset_teams: List[str]) -> List[MatchCandidate]:
        """Find matches between enriched team names and dataset team names"""
        matches = []
        
        for enriched_team in enriched_teams:
            best_match = None
            best_score = 0.0
            match_type = 'fuzzy'
            
            for dataset_team in dataset_teams:
                # Exact match
                if enriched_team == dataset_team:
                    best_match = dataset_team
                    best_score = 1.0
                    match_type = 'exact'
                    break
                
                # Alias matching
                alias_score = self._check_team_aliases(enriched_team, dataset_team)
                if alias_score > best_score:
                    best_score = alias_score
                    best_match = dataset_team
                    match_type = 'alias' if alias_score == 1.0 else 'fuzzy'
                
                # Fuzzy matching
                fuzzy_score = self.calculate_fuzzy_similarity(enriched_team, dataset_team)
                if fuzzy_score > best_score and fuzzy_score > 0.7:  # 70% threshold
                    best_score = fuzzy_score
                    best_match = dataset_team
                    match_type = 'fuzzy'
            
            if best_match and best_score > 0.7:
                confidence = 'high' if best_score > 0.9 else 'medium' if best_score > 0.8 else 'low'
                matches.append(MatchCandidate(
                    enriched_key=enriched_team,
                    dataset_key=best_match,
                    similarity_score=best_score,
                    match_type=match_type,
                    confidence=confidence
                ))
        
        return matches
    
    def find_venue_matches(self, enriched_venues: List[str], dataset_venues: List[str]) -> List[MatchCandidate]:
        """Find matches between enriched venue names and dataset venue names"""
        matches = []
        
        for enriched_venue in enriched_venues:
            best_match = None
            best_score = 0.0
            match_type = 'fuzzy'
            
            for dataset_venue in dataset_venues:
                # Exact match
                if enriched_venue == dataset_venue:
                    best_match = dataset_venue
                    best_score = 1.0
                    match_type = 'exact'
                    break
                
                # Alias matching
                alias_score = self._check_venue_aliases(enriched_venue, dataset_venue)
                if alias_score > best_score:
                    best_score = alias_score
                    best_match = dataset_venue
                    match_type = 'venue_alias' if alias_score == 1.0 else 'fuzzy'
                
                # Fuzzy matching
                fuzzy_score = self.calculate_fuzzy_similarity(enriched_venue, dataset_venue)
                if fuzzy_score > best_score and fuzzy_score > 0.6:  # 60% threshold for venues
                    best_score = fuzzy_score
                    best_match = dataset_venue
                    match_type = 'fuzzy'
            
            if best_match and best_score > 0.6:
                confidence = 'high' if best_score > 0.9 else 'medium' if best_score > 0.75 else 'low'
                matches.append(MatchCandidate(
                    enriched_key=enriched_venue,
                    dataset_key=best_match,
                    similarity_score=best_score,
                    match_type=match_type,
                    confidence=confidence
                ))
        
        return matches
    
    def find_player_matches(self, enriched_players: List[str], dataset_players: List[str]) -> List[MatchCandidate]:
        """Find matches between enriched player names and dataset player names"""
        matches = []
        
        for enriched_player in enriched_players:
            best_match = None
            best_score = 0.0
            match_type = 'fuzzy'
            
            for dataset_player in dataset_players:
                # Exact match
                if enriched_player == dataset_player:
                    best_match = dataset_player
                    best_score = 1.0
                    match_type = 'exact'
                    break
                
                # Alias matching
                alias_score = self._check_player_aliases(enriched_player, dataset_player)
                if alias_score > best_score:
                    best_score = alias_score
                    best_match = dataset_player
                    match_type = 'alias' if alias_score == 1.0 else 'fuzzy'
                
                # Fuzzy matching
                fuzzy_score = self.calculate_fuzzy_similarity(enriched_player, dataset_player)
                if fuzzy_score > best_score and fuzzy_score > 0.8:  # 80% threshold for players
                    best_score = fuzzy_score
                    best_match = dataset_player
                    match_type = 'fuzzy'
            
            if best_match and best_score > 0.8:
                confidence = 'high' if best_score > 0.95 else 'medium' if best_score > 0.85 else 'low'
                matches.append(MatchCandidate(
                    enriched_key=enriched_player,
                    dataset_key=best_match,
                    similarity_score=best_score,
                    match_type=match_type,
                    confidence=confidence
                ))
        
        return matches
    
    def _check_team_aliases(self, team1: str, team2: str) -> float:
        """Check if teams match via known aliases"""
        for canonical_name, aliases in self.team_aliases.items():
            if (team1 == canonical_name and team2 in aliases) or \
               (team2 == canonical_name and team1 in aliases) or \
               (team1 in aliases and team2 in aliases):
                return 1.0
        return 0.0
    
    def _check_venue_aliases(self, venue1: str, venue2: str) -> float:
        """Check if venues match via known aliases"""
        for canonical_name, aliases in self.venue_aliases.items():
            if (venue1 == canonical_name and venue2 in aliases) or \
               (venue2 == canonical_name and venue1 in aliases) or \
               (venue1 in aliases and venue2 in aliases):
                return 1.0
        return 0.0
    
    def _check_player_aliases(self, player1: str, player2: str) -> float:
        """Check if players match via known aliases"""
        for canonical_name, aliases in self.player_aliases.items():
            if (player1 == canonical_name and player2 in aliases) or \
               (player2 == canonical_name and player1 in aliases) or \
               (player1 in aliases and player2 in aliases):
                return 1.0
        return 0.0
    
    def create_match_key(self, date: str, team1: str, team2: str, venue: str) -> str:
        """Create a standardized match key for comparison"""
        # Sort teams alphabetically for consistent keys
        teams = sorted([self.normalize_name(team1), self.normalize_name(team2)])
        venue_norm = self.normalize_name(venue)
        return f"{date}_{teams[0]}_{teams[1]}_{venue_norm}"
    
    def match_enriched_data_to_datasets(self, enriched_data_path: str, 
                                      kg_data_path: str, 
                                      ml_data_path: str) -> Dict[str, List[MatchCandidate]]:
        """
        Match enriched data to both KG and ML datasets
        
        Returns:
            Dict with 'kg_matches' and 'ml_matches' keys containing match candidates
        """
        logger.info("🔍 Starting enriched data matching process...")
        
        # Load enriched data
        with open(enriched_data_path, 'r') as f:
            enriched_data = json.load(f)
        
        # Load KG data
        kg_data = pd.read_csv(kg_data_path)
        
        # Load ML data  
        ml_data = pd.read_csv(ml_data_path)
        
        results = {
            'kg_matches': [],
            'ml_matches': [],
            'statistics': {}
        }
        
        # Extract entities from each dataset
        enriched_teams = set()
        enriched_venues = set()
        enriched_players = set()
        
        for match in enriched_data:
            enriched_venues.add(match['venue']['name'])
            for team in match.get('teams', []):
                enriched_teams.add(team['name'])
                for player in team.get('players', []):
                    enriched_players.add(player['name'])
        
        # Extract KG entities
        kg_teams = set(kg_data['batting_team'].unique()) | set(kg_data['bowling_team'].unique()) if 'batting_team' in kg_data.columns else set()
        kg_venues = set(kg_data['venue'].unique()) if 'venue' in kg_data.columns else set()
        kg_players = set(kg_data['batter'].unique()) | set(kg_data['bowler'].unique()) if 'batter' in kg_data.columns else set()
        
        # Extract ML entities
        ml_teams = set(ml_data['home'].unique()) | set(ml_data['away'].unique()) if 'home' in ml_data.columns else set()
        ml_venues = set(ml_data['venue'].unique()) if 'venue' in ml_data.columns else set()
        ml_players = set()  # ML dataset may not have individual player names
        
        # Perform matching
        logger.info("🎯 Matching teams...")
        kg_team_matches = self.find_team_matches(list(enriched_teams), list(kg_teams))
        ml_team_matches = self.find_team_matches(list(enriched_teams), list(ml_teams))
        
        logger.info("🏟️ Matching venues...")
        kg_venue_matches = self.find_venue_matches(list(enriched_venues), list(kg_venues))
        ml_venue_matches = self.find_venue_matches(list(enriched_venues), list(ml_venues))
        
        logger.info("👥 Matching players...")
        kg_player_matches = self.find_player_matches(list(enriched_players), list(kg_players))
        ml_player_matches = self.find_player_matches(list(enriched_players), list(ml_players))
        
        # Compile results
        results['kg_matches'] = {
            'teams': kg_team_matches,
            'venues': kg_venue_matches,
            'players': kg_player_matches
        }
        
        results['ml_matches'] = {
            'teams': ml_team_matches,
            'venues': ml_venue_matches,
            'players': ml_player_matches
        }
        
        # Calculate statistics
        results['statistics'] = {
            'enriched_data': {
                'teams': len(enriched_teams),
                'venues': len(enriched_venues),
                'players': len(enriched_players)
            },
            'kg_data': {
                'teams': len(kg_teams),
                'venues': len(kg_venues),
                'players': len(kg_players)
            },
            'ml_data': {
                'teams': len(ml_teams),
                'venues': len(ml_venues),
                'players': len(ml_players)
            },
            'match_rates': {
                'kg_teams': len(kg_team_matches) / len(enriched_teams) * 100 if enriched_teams else 0,
                'kg_venues': len(kg_venue_matches) / len(enriched_venues) * 100 if enriched_venues else 0,
                'kg_players': len(kg_player_matches) / len(enriched_players) * 100 if enriched_players else 0,
                'ml_teams': len(ml_team_matches) / len(enriched_teams) * 100 if enriched_teams else 0,
                'ml_venues': len(ml_venue_matches) / len(enriched_venues) * 100 if enriched_venues else 0,
                'ml_players': len(ml_player_matches) / len(enriched_players) * 100 if enriched_players else 0
            }
        }
        
        logger.info("✅ Matching complete!")
        logger.info(f"📊 KG Match rates: Teams {results['statistics']['match_rates']['kg_teams']:.1f}%, Venues {results['statistics']['match_rates']['kg_venues']:.1f}%, Players {results['statistics']['match_rates']['kg_players']:.1f}%")
        logger.info(f"📊 ML Match rates: Teams {results['statistics']['match_rates']['ml_teams']:.1f}%, Venues {results['statistics']['match_rates']['ml_venues']:.1f}%, Players {results['statistics']['match_rates']['ml_players']:.1f}%")
        
        return results

if __name__ == "__main__":
    # Test the matcher
    matcher = EnrichedDataMatcher()
    
    # Test paths
    enriched_data_path = "./enriched_data/enriched_betting_matches.json"
    kg_data_path = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/joined_ball_by_ball_data.csv"
    ml_data_path = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/decimal_data_v3.csv"
    
    results = matcher.match_enriched_data_to_datasets(enriched_data_path, kg_data_path, ml_data_path)
    
    print(f"🎯 Matching Results:")
    print(f"KG Matches: {len(results['kg_matches']['teams'])} teams, {len(results['kg_matches']['venues'])} venues, {len(results['kg_matches']['players'])} players")
    print(f"ML Matches: {len(results['ml_matches']['teams'])} teams, {len(results['ml_matches']['venues'])} venues, {len(results['ml_matches']['players'])} players")
