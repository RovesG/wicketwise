# Purpose: Generates features from cricket ball-by-ball data for ML models
# Author: WicketWise Team, Last Modified: 2024-12-07

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    history_length: int = 5
    include_player_stats: bool = True
    include_match_context: bool = True
    include_situational_features: bool = True
    normalize_features: bool = True
    handle_missing_values: bool = True


class FeatureGenerator:
    """
    Generates comprehensive features from cricket ball-by-ball data.
    
    Creates features for machine learning models including:
    - Current ball features (runs, wickets, players)
    - Historical ball sequences
    - Match context and situation
    - Player statistics and performance
    - Venue and competition features
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the feature generator.
        
        Args:
            config: Configuration for feature generation
        """
        self.config = config or FeatureConfig()
        self.scalers = {}
        self.encoders = {}
        self.player_stats = {}
        self.venue_stats = {}
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame) -> 'FeatureGenerator':
        """
        Fit the feature generator on training data.
        
        Args:
            data: Training data DataFrame
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting feature generator on training data...")
        
        # Compute player statistics
        if self.config.include_player_stats:
            self._compute_player_stats(data)
        
        # Compute venue statistics
        self._compute_venue_stats(data)
        
        # Fit scalers and encoders
        features_df = self._extract_features(data, is_training=True)
        self._fit_preprocessors(features_df)
        
        self.is_fitted = True
        logger.info("Feature generator fitted successfully")
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data into features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with generated features
        """
        if not self.is_fitted:
            raise ValueError("Feature generator not fitted. Call fit() first.")
        
        logger.info(f"Transforming {len(data):,} rows into features...")
        
        # Extract features
        features_df = self._extract_features(data, is_training=False)
        
        # Apply preprocessing
        features_df = self._apply_preprocessing(features_df)
        
        logger.info(f"Generated {features_df.shape[1]} features")
        
        return features_df
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data in one step.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with generated features
        """
        return self.fit(data).transform(data)
    
    def _extract_features(self, data: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Extract all features from the data."""
        features = []
        
        # Sort data by match and ball order
        data = data.sort_values(['match_id', 'innings', 'over', 'ball']).reset_index(drop=True)
        
        for idx, row in data.iterrows():
            ball_features = self._extract_ball_features(data, idx, row)
            features.append(ball_features)
            
            if len(features) % 10000 == 0:
                logger.info(f"Processed {len(features):,} balls...")
        
        return pd.DataFrame(features)
    
    def _extract_ball_features(self, data: pd.DataFrame, idx: int, row: pd.Series) -> Dict:
        """Extract features for a single ball."""
        features = {}
        
        # Current ball features
        features.update(self._get_current_ball_features(row))
        
        # Historical features
        if self.config.history_length > 0:
            features.update(self._get_historical_features(data, idx, row))
        
        # Match context features
        if self.config.include_match_context:
            features.update(self._get_match_context_features(data, idx, row))
        
        # Situational features
        if self.config.include_situational_features:
            features.update(self._get_situational_features(data, idx, row))
        
        # Player features
        if self.config.include_player_stats:
            features.update(self._get_player_features(row))
        
        # Venue features
        features.update(self._get_venue_features(row))
        
        return features
    
    def _get_current_ball_features(self, row: pd.Series) -> Dict:
        """Extract current ball features."""
        features = {}
        
        # Basic ball info
        features['over'] = row.get('over', 0)
        features['ball_in_over'] = row.get('ball', 1)
        features['innings'] = row.get('innings', 1)
        features['runs_scored'] = row.get('runs_scored', 0)
        features['extras'] = row.get('extras', 0)
        features['is_wicket'] = int(row.get('is_wicket', False))
        
        # Derived features
        features['ball_number'] = features['over'] * 6 + features['ball_in_over']
        features['is_boundary'] = int(features['runs_scored'] >= 4)
        features['is_six'] = int(features['runs_scored'] == 6)
        features['is_dot_ball'] = int(features['runs_scored'] == 0 and features['extras'] == 0)
        
        # Phase of play
        if features['over'] < 6:
            features['phase'] = 'powerplay'
        elif features['over'] < 16:
            features['phase'] = 'middle'
        else:
            features['phase'] = 'death'
        
        return features
    
    def _get_historical_features(self, data: pd.DataFrame, idx: int, row: pd.Series) -> Dict:
        """Extract historical ball features."""
        features = {}
        
        # Get previous balls in same match
        match_data = data[data['match_id'] == row['match_id']].iloc[:idx]
        
        if len(match_data) == 0:
            # No history available
            for i in range(self.config.history_length):
                features[f'hist_{i}_runs'] = 0
                features[f'hist_{i}_wicket'] = 0
                features[f'hist_{i}_extras'] = 0
        else:
            # Get last N balls
            recent_balls = match_data.tail(self.config.history_length)
            
            for i in range(self.config.history_length):
                if i < len(recent_balls):
                    ball_data = recent_balls.iloc[-(i+1)]
                    features[f'hist_{i}_runs'] = ball_data.get('runs_scored', 0)
                    features[f'hist_{i}_wicket'] = int(ball_data.get('is_wicket', False))
                    features[f'hist_{i}_extras'] = ball_data.get('extras', 0)
                else:
                    features[f'hist_{i}_runs'] = 0
                    features[f'hist_{i}_wicket'] = 0
                    features[f'hist_{i}_extras'] = 0
        
        return features
    
    def _get_match_context_features(self, data: pd.DataFrame, idx: int, row: pd.Series) -> Dict:
        """Extract match context features."""
        features = {}
        
        # Get match progress
        match_data = data[data['match_id'] == row['match_id']].iloc[:idx+1]
        
        # Team scores
        innings_data = match_data[match_data['innings'] == row['innings']]
        features['team_score'] = innings_data['runs_scored'].sum()
        features['team_wickets'] = innings_data['is_wicket'].sum()
        features['balls_faced'] = len(innings_data)
        
        # Run rate
        overs_completed = features['balls_faced'] / 6.0
        features['current_run_rate'] = features['team_score'] / max(overs_completed, 0.1)
        
        # Remaining resources
        max_overs = 20  # T20 format
        features['overs_remaining'] = max_overs - overs_completed
        features['balls_remaining'] = (max_overs * 6) - features['balls_faced']
        features['wickets_remaining'] = 10 - features['team_wickets']
        
        # Required run rate (if chasing)
        if row['innings'] == 2:
            # Get first innings total
            first_innings = data[
                (data['match_id'] == row['match_id']) & 
                (data['innings'] == 1)
            ]
            target = first_innings['runs_scored'].sum() + 1
            runs_needed = target - features['team_score']
            features['target'] = target
            features['runs_needed'] = max(runs_needed, 0)
            features['required_run_rate'] = runs_needed / max(features['overs_remaining'], 0.1)
        else:
            features['target'] = 0
            features['runs_needed'] = 0
            features['required_run_rate'] = 0
        
        return features
    
    def _get_situational_features(self, data: pd.DataFrame, idx: int, row: pd.Series) -> Dict:
        """Extract situational features."""
        features = {}
        
        # Get recent performance (last 2 overs)
        match_data = data[data['match_id'] == row['match_id']].iloc[:idx]
        recent_data = match_data.tail(12)  # Last 2 overs
        
        if len(recent_data) > 0:
            features['recent_run_rate'] = recent_data['runs_scored'].sum() / max(len(recent_data) / 6.0, 0.1)
            features['recent_boundaries'] = (recent_data['runs_scored'] >= 4).sum()
            features['recent_wickets'] = recent_data['is_wicket'].sum()
            features['recent_dot_balls'] = ((recent_data['runs_scored'] == 0) & 
                                          (recent_data['extras'] == 0)).sum()
        else:
            features['recent_run_rate'] = 0
            features['recent_boundaries'] = 0
            features['recent_wickets'] = 0
            features['recent_dot_balls'] = 0
        
        # Partnership features
        current_partnership = self._get_current_partnership(match_data, row)
        features['partnership_runs'] = current_partnership['runs']
        features['partnership_balls'] = current_partnership['balls']
        
        return features
    
    def _get_current_partnership(self, match_data: pd.DataFrame, row: pd.Series) -> Dict:
        """Get current partnership statistics."""
        # Find last wicket in current innings
        innings_data = match_data[match_data['innings'] == row['innings']]
        
        if len(innings_data) == 0:
            return {'runs': 0, 'balls': 0}
        
        wicket_balls = innings_data[innings_data['is_wicket'] == True]
        
        if len(wicket_balls) == 0:
            # No wickets yet, partnership from start
            partnership_data = innings_data
        else:
            # Partnership from last wicket
            last_wicket_idx = wicket_balls.index[-1]
            partnership_data = innings_data[innings_data.index > last_wicket_idx]
        
        return {
            'runs': partnership_data['runs_scored'].sum(),
            'balls': len(partnership_data)
        }
    
    def _get_player_features(self, row: pd.Series) -> Dict:
        """Extract player-based features."""
        features = {}
        
        batter = row.get('batter', 'unknown')
        bowler = row.get('bowler', 'unknown')
        
        # Batter stats
        batter_stats = self.player_stats.get(batter, {})
        features['batter_avg'] = batter_stats.get('avg', 25.0)
        features['batter_strike_rate'] = batter_stats.get('strike_rate', 120.0)
        features['batter_boundary_rate'] = batter_stats.get('boundary_rate', 0.15)
        
        # Bowler stats
        bowler_stats = self.player_stats.get(bowler, {})
        features['bowler_avg'] = bowler_stats.get('avg', 25.0)
        features['bowler_economy'] = bowler_stats.get('economy', 7.5)
        features['bowler_wicket_rate'] = bowler_stats.get('wicket_rate', 0.05)
        
        return features
    
    def _get_venue_features(self, row: pd.Series) -> Dict:
        """Extract venue-based features."""
        features = {}
        
        venue = row.get('venue', 'unknown')
        venue_stats = self.venue_stats.get(venue, {})
        
        features['venue_avg_score'] = venue_stats.get('avg_score', 160.0)
        features['venue_boundary_rate'] = venue_stats.get('boundary_rate', 0.15)
        features['venue_wicket_rate'] = venue_stats.get('wicket_rate', 0.05)
        
        return features
    
    def _compute_player_stats(self, data: pd.DataFrame):
        """Compute player statistics from historical data."""
        logger.info("Computing player statistics...")
        
        # Batter statistics
        batter_stats = data.groupby('batter').agg({
            'runs_scored': ['sum', 'count'],
            'is_wicket': 'sum'
        }).round(2)
        
        batter_stats.columns = ['total_runs', 'balls_faced', 'dismissals']
        batter_stats['avg'] = batter_stats['total_runs'] / np.maximum(batter_stats['dismissals'], 1)
        batter_stats['strike_rate'] = (batter_stats['total_runs'] / batter_stats['balls_faced']) * 100
        
        # Boundary rate
        boundary_data = data[data['runs_scored'] >= 4].groupby('batter').size()
        batter_stats['boundary_rate'] = boundary_data / batter_stats['balls_faced']
        batter_stats['boundary_rate'] = batter_stats['boundary_rate'].fillna(0)
        
        # Bowler statistics
        bowler_stats = data.groupby('bowler').agg({
            'runs_scored': ['sum', 'count'],
            'is_wicket': 'sum'
        }).round(2)
        
        bowler_stats.columns = ['runs_conceded', 'balls_bowled', 'wickets']
        bowler_stats['avg'] = bowler_stats['runs_conceded'] / np.maximum(bowler_stats['wickets'], 1)
        bowler_stats['economy'] = (bowler_stats['runs_conceded'] / bowler_stats['balls_bowled']) * 6
        bowler_stats['wicket_rate'] = bowler_stats['wickets'] / bowler_stats['balls_bowled']
        
        # Store stats
        self.player_stats = {}
        for player in batter_stats.index:
            self.player_stats[player] = batter_stats.loc[player].to_dict()
        
        for player in bowler_stats.index:
            if player in self.player_stats:
                self.player_stats[player].update(bowler_stats.loc[player].to_dict())
            else:
                self.player_stats[player] = bowler_stats.loc[player].to_dict()
        
        logger.info(f"Computed stats for {len(self.player_stats)} players")
    
    def _compute_venue_stats(self, data: pd.DataFrame):
        """Compute venue statistics from historical data."""
        logger.info("Computing venue statistics...")
        
        # Group by match and venue to get match totals
        match_totals = data.groupby(['match_id', 'venue', 'innings']).agg({
            'runs_scored': 'sum',
            'is_wicket': 'sum'
        }).reset_index()
        
        # Venue statistics
        venue_stats = match_totals.groupby('venue').agg({
            'runs_scored': 'mean',
            'is_wicket': 'mean'
        }).round(2)
        
        venue_stats.columns = ['avg_score', 'avg_wickets']
        
        # Boundary rate by venue
        boundary_data = data[data['runs_scored'] >= 4].groupby('venue').size()
        total_balls = data.groupby('venue').size()
        venue_stats['boundary_rate'] = boundary_data / total_balls
        venue_stats['boundary_rate'] = venue_stats['boundary_rate'].fillna(0)
        
        # Wicket rate by venue
        venue_stats['wicket_rate'] = venue_stats['avg_wickets'] / 120  # Per ball
        
        # Store stats
        self.venue_stats = venue_stats.to_dict('index')
        
        logger.info(f"Computed stats for {len(self.venue_stats)} venues")
    
    def _fit_preprocessors(self, features_df: pd.DataFrame):
        """Fit scalers and encoders on features."""
        logger.info("Fitting preprocessors...")
        
        # Identify numeric and categorical columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        categorical_cols = features_df.select_dtypes(include=['object']).columns
        
        # Fit scalers for numeric features
        if self.config.normalize_features and len(numeric_cols) > 0:
            scaler = StandardScaler()
            scaler.fit(features_df[numeric_cols])
            self.scalers['numeric'] = scaler
        
        # Fit encoders for categorical features
        for col in categorical_cols:
            encoder = LabelEncoder()
            encoder.fit(features_df[col].astype(str))
            self.encoders[col] = encoder
        
        logger.info(f"Fitted preprocessors for {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")
    
    def _apply_preprocessing(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to features."""
        features_df = features_df.copy()
        
        # Handle missing values
        if self.config.handle_missing_values:
            features_df = features_df.fillna(0)
        
        # Apply scaling
        if self.config.normalize_features and 'numeric' in self.scalers:
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                features_df[numeric_cols] = self.scalers['numeric'].transform(features_df[numeric_cols])
        
        # Apply encoding
        for col, encoder in self.encoders.items():
            if col in features_df.columns:
                # Handle unseen categories
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        features_df[col] = encoder.transform(features_df[col].astype(str))
                    except ValueError:
                        # Handle unseen categories by assigning 0
                        features_df[col] = 0
        
        return features_df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        if not self.is_fitted:
            raise ValueError("Feature generator not fitted. Call fit() first.")
        
        # This would be populated during fitting
        # For now, return a basic list
        feature_names = []
        
        # Current ball features
        feature_names.extend([
            'over', 'ball_in_over', 'innings', 'runs_scored', 'extras', 'is_wicket',
            'ball_number', 'is_boundary', 'is_six', 'is_dot_ball'
        ])
        
        # Historical features
        for i in range(self.config.history_length):
            feature_names.extend([
                f'hist_{i}_runs', f'hist_{i}_wicket', f'hist_{i}_extras'
            ])
        
        # Match context features
        if self.config.include_match_context:
            feature_names.extend([
                'team_score', 'team_wickets', 'balls_faced', 'current_run_rate',
                'overs_remaining', 'balls_remaining', 'wickets_remaining',
                'target', 'runs_needed', 'required_run_rate'
            ])
        
        # Situational features
        if self.config.include_situational_features:
            feature_names.extend([
                'recent_run_rate', 'recent_boundaries', 'recent_wickets', 'recent_dot_balls',
                'partnership_runs', 'partnership_balls'
            ])
        
        # Player features
        if self.config.include_player_stats:
            feature_names.extend([
                'batter_avg', 'batter_strike_rate', 'batter_boundary_rate',
                'bowler_avg', 'bowler_economy', 'bowler_wicket_rate'
            ])
        
        # Venue features
        feature_names.extend([
            'venue_avg_score', 'venue_boundary_rate', 'venue_wicket_rate'
        ])
        
        return feature_names 