# Purpose: Adapter to convert CSV cricket data to CrickformerDataset format
# Author: Assistant, Last Modified: 2024

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from .csv_data_schema import (
    CurrentBallFeatures, RecentBallHistoryEntry, VideoSignals,
    GNNEmbeddings, MarketOdds
)

logger = logging.getLogger(__name__)


@dataclass
class CSVDataConfig:
    """Configuration for CSV data loading"""
    nvplay_file: str = "nvplay_data_v3.csv"
    decimal_file: str = "decimal_data_v3.csv"
    max_history_length: int = 5
    default_embedding_dim: int = 128


class CSVDataAdapter:
    """
    Adapter to convert CSV cricket data format to CrickformerDataset compatible format.
    
    This adapter handles the real data structure with two main CSV files:
    - nvplay_data_v3.csv: Ball-by-ball tracking and event data
    - decimal_data_v3.csv: Betting odds and win probability data
    """
    
    def __init__(self, data_root: str, config: Optional[CSVDataConfig] = None):
        self.data_root = Path(data_root)
        self.config = config or CSVDataConfig()
        
        # Load the CSV files
        self.nvplay_df = None
        self.decimal_df = None
        self._load_data()
        
        # Create match index
        self._create_match_index()
    
    def _load_data(self):
        """Load the CSV files into pandas DataFrames"""
        nvplay_path = self.data_root / self.config.nvplay_file
        decimal_path = self.data_root / self.config.decimal_file
        
        if not nvplay_path.exists():
            raise FileNotFoundError(f"NVPlay data file not found: {nvplay_path}")
        if not decimal_path.exists():
            raise FileNotFoundError(f"Decimal data file not found: {decimal_path}")
        
        logger.info(f"Loading NVPlay data from {nvplay_path}")
        self.nvplay_df = pd.read_csv(nvplay_path)
        
        logger.info(f"Loading decimal data from {decimal_path}")
        self.decimal_df = pd.read_csv(decimal_path)
        
        # Basic data validation
        logger.info(f"Loaded {len(self.nvplay_df)} NVPlay records")
        logger.info(f"Loaded {len(self.decimal_df)} decimal records")
    
    def _create_match_index(self):
        """Create an index of all matches and balls for efficient lookup"""
        # For nvplay data, use Match column as match identifier
        self.nvplay_matches = self.nvplay_df['Match'].unique()
        
        # For decimal data, create match identifier from competition, home, away, date
        self.decimal_df['match_id'] = (
            self.decimal_df['competition'] + '_' + 
            self.decimal_df['home'] + '_vs_' + 
            self.decimal_df['away'] + '_' + 
            self.decimal_df['date'].astype(str)
        )
        self.decimal_matches = self.decimal_df['match_id'].unique()
        
        # Create ball-level index for nvplay data
        self.nvplay_df['ball_id'] = (
            self.nvplay_df['Match'] + '_' + 
            self.nvplay_df['Innings'].astype(str) + '_' + 
            self.nvplay_df['Innings Ball'].astype(str)
        )
        
        # Create ball-level index for decimal data  
        self.decimal_df['ball_id'] = (
            self.decimal_df['match_id'] + '_' + 
            self.decimal_df['innings'].astype(str) + '_' + 
            self.decimal_df['ball'].astype(str)
        )
        
        logger.info(f"Found {len(self.nvplay_matches)} matches in NVPlay data")
        logger.info(f"Found {len(self.decimal_matches)} matches in decimal data")
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float, handling 'Unknown' and NaN values"""
        if pd.isna(value) or value == 'Unknown' or value == '':
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def get_match_ids(self) -> List[str]:
        """Get list of all available match IDs"""
        # Use nvplay matches as primary source
        return list(self.nvplay_matches)
    
    def get_balls_for_match(self, match_id: str) -> List[str]:
        """Get all ball IDs for a specific match"""
        match_balls = self.nvplay_df[self.nvplay_df['Match'] == match_id]
        return list(match_balls['ball_id'].values)
    
    def get_current_ball_features(self, ball_id: str) -> CurrentBallFeatures:
        """Extract current ball features from the data"""
        # Get nvplay data for this ball
        nvplay_row = self.nvplay_df[self.nvplay_df['ball_id'] == ball_id].iloc[0]
        
        # Extract match and ball information
        match_id = nvplay_row['Match']
        competition_name = nvplay_row['Competition']
        venue = nvplay_row['Venue']
        
        return CurrentBallFeatures(
            match_id=match_id,
            competition_name=competition_name,
            venue=venue,
            venue_city=venue,  # Using venue as city for now
            venue_country="Unknown",  # Not in current data
            
            innings=int(nvplay_row['Innings']),
            over=float(nvplay_row['Over']),
            ball_in_over=int(nvplay_row['Ball']),
            innings_ball=int(nvplay_row['Innings Ball']),
            
            batter_name=nvplay_row['Batter'],
            batter_id=nvplay_row['Batter ID'],
            bowler_name=nvplay_row['Bowler'], 
            bowler_id=nvplay_row['Bowler ID'],
            
            runs_scored=int(nvplay_row['Runs']),
            extras=int(nvplay_row['Extra Runs']),
            is_wicket=nvplay_row['Wicket'] != 'No Wicket',
            
            team_score=int(nvplay_row['Team Runs']),
            team_wickets=int(nvplay_row['Team Wickets']),
            
            batting_team=nvplay_row['Batting Team'],
            bowling_team=nvplay_row['Bowling Team'],
            
            batter_hand=nvplay_row['Batting Hand'],
            bowler_type=nvplay_row['Bowler Type'],
            
            # Ball tracking data
            field_x=self._safe_float(nvplay_row['FieldX']),
            field_y=self._safe_float(nvplay_row['FieldY']),
            pitch_x=self._safe_float(nvplay_row['PitchX']),
            pitch_y=self._safe_float(nvplay_row['PitchY']),
            
            # Additional features
            powerplay=self._safe_float(nvplay_row['Power Play']),
            run_rate=self._safe_float(nvplay_row['Run Rate After']),
            req_run_rate=self._safe_float(nvplay_row['Req Run Rate After'])
        )
    
    def get_ball_history(self, ball_id: str, history_length: int = 5) -> List[RecentBallHistoryEntry]:
        """Get recent ball history for the current ball"""
        # Get current ball info
        current_ball = self.nvplay_df[self.nvplay_df['ball_id'] == ball_id].iloc[0]
        match_id = current_ball['Match']
        innings = current_ball['Innings']
        current_ball_num = current_ball['Innings Ball']
        
        # Get previous balls in same innings
        match_innings = self.nvplay_df[
            (self.nvplay_df['Match'] == match_id) & 
            (self.nvplay_df['Innings'] == innings) &
            (self.nvplay_df['Innings Ball'] < current_ball_num)
        ].sort_values('Innings Ball', ascending=False)
        
        history = []
        for i, (_, row) in enumerate(match_innings.head(history_length).iterrows()):
            history.append(RecentBallHistoryEntry(
                runs_scored=int(row['Runs']),
                extras=int(row['Extra Runs']),
                is_wicket=row['Wicket'] != 'No Wicket',
                batter_name=row['Batter'],
                bowler_name=row['Bowler'],
                batter_hand=row['Batting Hand'],
                bowler_type=row['Bowler Type']
            ))
        
        # Pad with zeros if needed
        while len(history) < history_length:
            history.append(RecentBallHistoryEntry(
                runs_scored=0,
                extras=0,
                is_wicket=False,
                batter_name="PADDING",
                bowler_name="PADDING", 
                batter_hand="PADDING",
                bowler_type="PADDING"
            ))
        
        return history[:history_length]
    
    def get_video_signals(self, ball_id: str) -> VideoSignals:
        """Get video signals for the ball (mock implementation)"""
        # Real implementation would extract video features
        # For now, return mock data
        return VideoSignals(
            ball_tracking_confidence=0.8,
            player_detection_confidence=0.9,
            scene_classification="cricket_match",
            motion_vectors=np.random.randn(32).astype(np.float32),
            optical_flow=np.random.randn(64).astype(np.float32)
        )
    
    def get_gnn_embeddings(self, ball_id: str) -> GNNEmbeddings:
        """Get GNN embeddings for the ball (mock implementation)"""
        # Real implementation would use pre-computed embeddings
        # For now, return mock embeddings
        return GNNEmbeddings(
            batter_embedding=np.random.randn(self.config.default_embedding_dim).astype(np.float32),
            bowler_embedding=np.random.randn(self.config.default_embedding_dim).astype(np.float32),
            venue_embedding=np.random.randn(64).astype(np.float32),
            edge_embeddings=np.random.randn(0).astype(np.float32)  # Empty for now
        )
    
    def get_market_odds(self, ball_id: str) -> Optional[MarketOdds]:
        """Get market odds for the ball from decimal data"""
        # Try to find matching ball in decimal data
        current_ball = self.nvplay_df[self.nvplay_df['ball_id'] == ball_id].iloc[0]
        
        # Create approximate match for decimal data
        # This is tricky as the data formats don't perfectly align
        decimal_match = self.decimal_df[
            (self.decimal_df['date'] == current_ball['Date']) &
            (self.decimal_df['innings'] == current_ball['Innings']) &
            (self.decimal_df['ball'] == current_ball['Innings Ball'])
        ]
        
        if len(decimal_match) == 0:
            return None
        
        row = decimal_match.iloc[0]
        
        return MarketOdds(
            win_probability=self._safe_float(row['win_prob']) if 'win_prob' in row else 0.5,
            total_runs_over=0.0,  # Not in current data
            total_runs_under=0.0,  # Not in current data
            next_wicket_over=0.0,  # Not in current data
            next_wicket_under=0.0,  # Not in current data
            match_odds_home=0.0,   # Not in current data
            match_odds_away=0.0    # Not in current data
        )
    
    def __len__(self) -> int:
        """Total number of balls in the dataset"""
        return len(self.nvplay_df)
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get sample information for a given index"""
        row = self.nvplay_df.iloc[idx]
        return {
            'ball_id': row['ball_id'],
            'match_id': row['Match'],
            'innings': row['Innings'],
            'ball_number': row['Innings Ball'],
            'over': row['Over'],
            'ball_in_over': row['Ball']
        } 