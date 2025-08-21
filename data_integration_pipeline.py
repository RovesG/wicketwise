# Purpose: Data integration pipeline for betting model training
# Author: Assistant, Last Modified: 2025-01-21

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
import logging
from datetime import datetime
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CricketDataIntegrator:
    """
    Integrates betting dataset with Knowledge Graph for model training
    """
    
    def __init__(self, betting_data_path: str, kg_path: str):
        self.betting_data_path = betting_data_path
        self.kg_path = kg_path
        self.betting_data = None
        self.kg = None
        self.entity_mappings = {}
        
    def load_data(self):
        """Load betting data and knowledge graph"""
        logger.info("Loading betting dataset...")
        self.betting_data = pd.read_csv(self.betting_data_path)
        logger.info(f"✅ Loaded {len(self.betting_data):,} betting records")
        
        logger.info("Loading knowledge graph...")
        with open(self.kg_path, 'rb') as f:
            self.kg = pickle.load(f)
        logger.info(f"✅ Loaded KG with {len(self.kg.nodes):,} nodes")
    
    def create_match_level_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create 80/20 train/test split preserving whole matches
        Stratified by competition and temporal distribution
        """
        logger.info("Creating match-level train/test split...")
        
        # Create match identifiers
        self.betting_data['match_id'] = (
            self.betting_data['date'].astype(str) + '_' +
            self.betting_data['venue'].astype(str) + '_' +
            self.betting_data['home'].astype(str) + '_' +
            self.betting_data['away'].astype(str)
        )
        
        # Get match-level metadata
        match_metadata = self.betting_data.groupby('match_id').agg({
            'date': 'first',
            'competition': 'first', 
            'venue': 'first',
            'home': 'first',
            'away': 'first',
            'ball': 'count'  # Number of balls in match
        }).reset_index()
        
        match_metadata['date'] = pd.to_datetime(match_metadata['date'])
        match_metadata['year'] = match_metadata['date'].dt.year
        
        logger.info(f"Total matches: {len(match_metadata):,}")
        logger.info(f"Date range: {match_metadata['date'].min()} to {match_metadata['date'].max()}")
        
        # Stratified split by competition and year
        match_metadata['strata'] = (
            match_metadata['competition'].astype(str) + '_' +
            match_metadata['year'].astype(str)
        )
        
        # Handle strata with only 1 match
        strata_counts = match_metadata['strata'].value_counts()
        single_match_strata = strata_counts[strata_counts == 1].index
        match_metadata.loc[match_metadata['strata'].isin(single_match_strata), 'strata'] = 'single_matches'
        
        try:
            train_matches, test_matches = train_test_split(
                match_metadata,
                test_size=test_size,
                random_state=random_state,
                stratify=match_metadata['strata']
            )
        except ValueError as e:
            logger.warning(f"Stratification failed: {e}. Using random split.")
            train_matches, test_matches = train_test_split(
                match_metadata,
                test_size=test_size,
                random_state=random_state
            )
        
        # Split the betting data
        train_data = self.betting_data[self.betting_data['match_id'].isin(train_matches['match_id'])]
        test_data = self.betting_data[self.betting_data['match_id'].isin(test_matches['match_id'])]
        
        logger.info(f"✅ Train: {len(train_matches):,} matches ({len(train_data):,} balls)")
        logger.info(f"✅ Test: {len(test_matches):,} matches ({len(test_data):,} balls)")
        
        # Verify no data leakage
        assert len(set(train_matches['match_id']) & set(test_matches['match_id'])) == 0
        
        return train_data, test_data
    
    def extract_entities(self) -> Dict[str, Set[str]]:
        """Extract all unique entities from betting dataset"""
        logger.info("Extracting entities from betting dataset...")
        
        # Players
        all_batsmen = set(self.betting_data['batsman'].dropna().unique())
        all_nonstrikers = set(self.betting_data['nonstriker'].dropna().unique())
        all_bowlers = set(self.betting_data['bowler'].dropna().unique())
        betting_players = all_batsmen | all_nonstrikers | all_bowlers
        
        # Teams
        betting_teams = set(pd.concat([
            self.betting_data['home'],
            self.betting_data['away'],
            self.betting_data['battingteam']
        ]).dropna().unique())
        
        # Venues
        betting_venues = set(self.betting_data['venue'].dropna().unique())
        
        entities = {
            'players': betting_players,
            'teams': betting_teams,
            'venues': betting_venues
        }
        
        logger.info(f"Extracted entities: {len(betting_players)} players, {len(betting_teams)} teams, {len(betting_venues)} venues")
        return entities
    
    def extract_kg_entities(self) -> Dict[str, Set[str]]:
        """Extract entities from Knowledge Graph"""
        logger.info("Extracting entities from Knowledge Graph...")
        
        kg_players = set()
        kg_teams = set()
        kg_venues = set()
        
        for node_id, node_data in self.kg.nodes(data=True):
            node_type = node_data.get('type', '')
            name = node_data.get('name', node_id)
            
            if node_type == 'player':
                kg_players.add(name)
            elif node_type == 'team':
                kg_teams.add(name)
            elif node_type == 'venue':
                kg_venues.add(name)
        
        kg_entities = {
            'players': kg_players,
            'teams': kg_teams,
            'venues': kg_venues
        }
        
        logger.info(f"KG entities: {len(kg_players)} players, {len(kg_teams)} teams, {len(kg_venues)} venues")
        return kg_entities
    
    def create_entity_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        Create mappings between betting dataset entities and KG entities
        Uses fuzzy matching for player names
        """
        from difflib import SequenceMatcher
        
        logger.info("Creating entity mappings...")
        
        betting_entities = self.extract_entities()
        kg_entities = self.extract_kg_entities()
        
        mappings = {
            'players': {},
            'teams': {},
            'venues': {}
        }
        
        # Player mapping with fuzzy matching
        logger.info("Mapping players...")
        for betting_player in betting_entities['players']:
            best_match = None
            best_score = 0.0
            
            for kg_player in kg_entities['players']:
                # Calculate similarity
                similarity = SequenceMatcher(None, betting_player.lower(), kg_player.lower()).ratio()
                
                if similarity > best_score and similarity > 0.8:  # 80% threshold
                    best_score = similarity
                    best_match = kg_player
            
            if best_match:
                mappings['players'][betting_player] = best_match
        
        logger.info(f"✅ Mapped {len(mappings['players']):,}/{len(betting_entities['players']):,} players ({len(mappings['players'])/len(betting_entities['players'])*100:.1f}%)")
        
        # Team and venue mapping (exact match for now)
        # Note: KG has very limited team/venue data
        for betting_team in betting_entities['teams']:
            if betting_team in kg_entities['teams']:
                mappings['teams'][betting_team] = betting_team
        
        for betting_venue in betting_entities['venues']:
            if betting_venue in kg_entities['venues']:
                mappings['venues'][betting_venue] = betting_venue
        
        logger.info(f"✅ Mapped {len(mappings['teams']):,}/{len(betting_entities['teams']):,} teams")
        logger.info(f"✅ Mapped {len(mappings['venues']):,}/{len(betting_entities['venues']):,} venues")
        
        self.entity_mappings = mappings
        return mappings
    
    def save_mappings(self, output_path: str):
        """Save entity mappings for later use"""
        with open(output_path, 'wb') as f:
            pickle.dump(self.entity_mappings, f)
        logger.info(f"✅ Saved entity mappings to {output_path}")
    
    def create_simulation_matches(self, test_data: pd.DataFrame, num_matches: int = 10) -> List[Dict]:
        """
        Create simulation-ready matches from test data for UI demonstration
        """
        logger.info(f"Creating {num_matches} simulation matches...")
        
        # Get diverse matches for simulation - simplified approach
        test_matches = test_data.groupby('match_id').agg({
            'date': 'first',
            'competition': 'first',
            'venue': 'first', 
            'home': 'first',
            'away': 'first',
            'ball': 'count',
            'win_prob': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        test_matches.columns = ['match_id', 'date', 'competition', 'venue', 'home', 'away', 'ball', 'win_prob_min', 'win_prob_max']
        
        # Calculate excitement score (win probability swings)
        test_matches['excitement'] = test_matches['win_prob_max'] - test_matches['win_prob_min']
        
        # Select most exciting matches
        exciting_matches = test_matches.nlargest(num_matches, 'excitement')
        
        simulation_matches = []
        for _, match in exciting_matches.iterrows():
            simulation_matches.append({
                'match_id': match['match_id'],
                'date': match['date'],
                'competition': match['competition'],
                'venue': match['venue'],
                'teams': f"{match['home']} vs {match['away']}",
                'balls': match['ball'],
                'excitement_score': match['excitement']
            })
        
        logger.info(f"✅ Created {len(simulation_matches)} simulation matches")
        return simulation_matches
    
    def run_full_integration(self, output_dir: str = "integrated_data"):
        """Run the complete integration pipeline"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info("🚀 Starting full data integration pipeline...")
        
        # Load data
        self.load_data()
        
        # Create train/test split
        train_data, test_data = self.create_match_level_split()
        
        # Save splits
        train_data.to_parquet(output_path / "train_data.parquet")
        test_data.to_parquet(output_path / "test_data.parquet")
        logger.info(f"✅ Saved train/test splits to {output_path}")
        
        # Create entity mappings
        mappings = self.create_entity_mappings()
        self.save_mappings(output_path / "entity_mappings.pkl")
        
        # Create simulation matches
        simulation_matches = self.create_simulation_matches(test_data)
        with open(output_path / "simulation_matches.pkl", 'wb') as f:
            pickle.dump(simulation_matches, f)
        
        # Create summary report
        summary = {
            'train_matches': len(train_data['match_id'].unique()),
            'test_matches': len(test_data['match_id'].unique()),
            'train_balls': len(train_data),
            'test_balls': len(test_data),
            'player_mappings': len(mappings['players']),
            'team_mappings': len(mappings['teams']),
            'venue_mappings': len(mappings['venues']),
            'simulation_matches': len(simulation_matches),
            'integration_date': datetime.now().isoformat()
        }
        
        with open(output_path / "integration_summary.json", 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        logger.info("🎉 Integration pipeline complete!")
        logger.info(f"📊 Summary: {summary}")
        
        return summary

if __name__ == "__main__":
    # Configuration
    BETTING_DATA_PATH = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/decimal_data_v3.csv"
    KG_PATH = "models/unified_cricket_kg.pkl"
    
    # Run integration
    integrator = CricketDataIntegrator(BETTING_DATA_PATH, KG_PATH)
    summary = integrator.run_full_integration()
    
    print("\n🎯 INTEGRATION COMPLETE!")
    print("=" * 50)
    for key, value in summary.items():
        print(f"{key}: {value}")
