# Purpose: Ingests and validates cricket CSV data from multiple sources
# Author: WicketWise Team, Last Modified: 2024-12-07

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class DataIngestor:
    """
    Handles ingestion and validation of cricket data from CSV files.
    
    Supports multiple data sources including NVPlay ball-by-ball data
    and decimal betting odds data with comprehensive validation and
    preprocessing capabilities.
    """
    
    def __init__(self, data_root: Union[str, Path]):
        """
        Initialize the data ingestor.
        
        Args:
            data_root: Path to directory containing CSV files
        """
        self.data_root = Path(data_root)
        self.nvplay_data: Optional[pd.DataFrame] = None
        self.decimal_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        
        # Expected column mappings
        self.nvplay_columns = {
            'required': [
                'match_id', 'competition_name', 'venue', 'innings', 
                'over', 'ball', 'batter', 'bowler', 'runs_scored',
                'extras', 'is_wicket', 'team_batting', 'team_bowling'
            ],
            'optional': [
                'ball_type', 'shot_type', 'fielder', 'dismissal_type',
                'boundary_type', 'pitch_x', 'pitch_y', 'impact_x', 'impact_y'
            ]
        }
        
        self.decimal_columns = {
            'required': [
                'match_id', 'team_a', 'team_b', 'win_probability',
                'timestamp', 'market_odds'
            ],
            'optional': [
                'volume', 'liquidity', 'back_odds', 'lay_odds'
            ]
        }
        
    def load_nvplay_data(self, filename: str = "nvplay_data_v3.csv") -> pd.DataFrame:
        """
        Load NVPlay ball-by-ball data.
        
        Args:
            filename: Name of NVPlay CSV file
            
        Returns:
            Loaded and validated DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        filepath = self.data_root / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"NVPlay data file not found: {filepath}")
            
        logger.info(f"Loading NVPlay data from {filepath}")
        
        try:
            # Load with error handling for encoding issues
            df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='latin-1', low_memory=False)
            
        # Validate required columns
        missing_cols = set(self.nvplay_columns['required']) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in NVPlay data: {missing_cols}")
            
        # Clean and validate data
        df = self._clean_nvplay_data(df)
        
        self.nvplay_data = df
        logger.info(f"Loaded {len(df):,} balls from {df['match_id'].nunique()} matches")
        
        return df
    
    def load_decimal_data(self, filename: str = "decimal_data_v3.csv") -> pd.DataFrame:
        """
        Load decimal betting odds data.
        
        Args:
            filename: Name of decimal CSV file
            
        Returns:
            Loaded and validated DataFrame
        """
        filepath = self.data_root / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Decimal data file not found: {filepath}")
            
        logger.info(f"Loading decimal data from {filepath}")
        
        try:
            df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='latin-1', low_memory=False)
            
        # Validate required columns (flexible for betting data)
        available_cols = set(df.columns)
        required_cols = set(self.decimal_columns['required'])
        
        # Check for at least match_id and some odds data
        if 'match_id' not in available_cols:
            raise ValueError("Decimal data must contain 'match_id' column")
            
        # Clean and validate data
        df = self._clean_decimal_data(df)
        
        self.decimal_data = df
        logger.info(f"Loaded {len(df):,} records from {df['match_id'].nunique()} matches")
        
        return df
    
    def _clean_nvplay_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate NVPlay data."""
        logger.info("Cleaning NVPlay data...")
        
        # Convert numeric columns
        numeric_cols = ['over', 'ball', 'runs_scored', 'extras', 'innings']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert boolean columns
        bool_cols = ['is_wicket']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Clean string columns
        string_cols = ['batter', 'bowler', 'team_batting', 'team_bowling']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Remove invalid rows
        initial_count = len(df)
        df = df.dropna(subset=['match_id', 'over', 'ball'])
        df = df[df['over'] >= 0]  # Valid overs
        df = df[df['ball'] >= 1]  # Valid balls
        
        logger.info(f"Cleaned data: {initial_count:,} → {len(df):,} rows")
        
        return df
    
    def _clean_decimal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate decimal betting data."""
        logger.info("Cleaning decimal data...")
        
        # Convert numeric columns if they exist
        numeric_cols = ['win_probability', 'market_odds', 'volume', 'liquidity']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean string columns
        string_cols = ['team_a', 'team_b']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Remove invalid rows
        initial_count = len(df)
        df = df.dropna(subset=['match_id'])
        
        logger.info(f"Cleaned data: {initial_count:,} → {len(df):,} rows")
        
        return df
    
    def merge_data_sources(self) -> pd.DataFrame:
        """
        Merge NVPlay and decimal data sources.
        
        Returns:
            Merged DataFrame with ball-by-ball and betting data
        """
        if self.nvplay_data is None:
            raise ValueError("NVPlay data not loaded. Call load_nvplay_data() first.")
        
        if self.decimal_data is None:
            logger.warning("Decimal data not loaded. Proceeding with NVPlay data only.")
            self.processed_data = self.nvplay_data.copy()
            return self.processed_data
        
        logger.info("Merging NVPlay and decimal data...")
        
        # Merge on match_id
        merged_df = pd.merge(
            self.nvplay_data,
            self.decimal_data,
            on='match_id',
            how='left',
            suffixes=('', '_decimal')
        )
        
        logger.info(f"Merged data: {len(merged_df):,} rows from {merged_df['match_id'].nunique()} matches")
        
        self.processed_data = merged_df
        return merged_df
    
    def get_match_summary(self) -> Dict[str, any]:
        """
        Get summary statistics of loaded data.
        
        Returns:
            Dictionary with data summary statistics
        """
        summary = {}
        
        if self.nvplay_data is not None:
            summary['nvplay'] = {
                'total_balls': len(self.nvplay_data),
                'total_matches': self.nvplay_data['match_id'].nunique(),
                'competitions': list(self.nvplay_data['competition_name'].unique()),
                'date_range': {
                    'start': self.nvplay_data['match_id'].min(),
                    'end': self.nvplay_data['match_id'].max()
                }
            }
        
        if self.decimal_data is not None:
            summary['decimal'] = {
                'total_records': len(self.decimal_data),
                'total_matches': self.decimal_data['match_id'].nunique(),
            }
        
        if self.processed_data is not None:
            summary['processed'] = {
                'total_records': len(self.processed_data),
                'total_matches': self.processed_data['match_id'].nunique(),
                'merge_success_rate': len(self.processed_data) / len(self.nvplay_data) if self.nvplay_data is not None else 0
            }
        
        return summary
    
    def filter_matches(self, 
                      competition: Optional[str] = None,
                      venue: Optional[str] = None,
                      min_balls: int = 10,
                      max_balls: int = 240) -> pd.DataFrame:
        """
        Filter matches based on criteria.
        
        Args:
            competition: Filter by competition name
            venue: Filter by venue
            min_balls: Minimum balls per match
            max_balls: Maximum balls per match
            
        Returns:
            Filtered DataFrame
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call merge_data_sources() first.")
        
        df = self.processed_data.copy()
        
        # Filter by competition
        if competition:
            df = df[df['competition_name'] == competition]
        
        # Filter by venue
        if venue:
            df = df[df['venue'] == venue]
        
        # Filter by match length
        match_lengths = df.groupby('match_id').size()
        valid_matches = match_lengths[
            (match_lengths >= min_balls) & (match_lengths <= max_balls)
        ].index
        
        df = df[df['match_id'].isin(valid_matches)]
        
        logger.info(f"Filtered to {len(df):,} balls from {df['match_id'].nunique()} matches")
        
        return df
    
    def export_processed_data(self, filename: str = "processed_cricket_data.csv") -> Path:
        """
        Export processed data to CSV.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        if self.processed_data is None:
            raise ValueError("No processed data to export. Call merge_data_sources() first.")
        
        output_path = self.data_root / filename
        self.processed_data.to_csv(output_path, index=False)
        
        logger.info(f"Exported processed data to {output_path}")
        
        return output_path
    
    def validate_data_quality(self) -> Dict[str, any]:
        """
        Validate data quality and return quality metrics.
        
        Returns:
            Dictionary with quality metrics
        """
        if self.processed_data is None:
            raise ValueError("No processed data to validate. Call merge_data_sources() first.")
        
        df = self.processed_data
        
        quality_metrics = {
            'completeness': {
                'total_records': len(df),
                'missing_values': df.isnull().sum().to_dict(),
                'completeness_rate': (1 - df.isnull().sum() / len(df)).to_dict()
            },
            'consistency': {
                'duplicate_records': df.duplicated().sum(),
                'invalid_overs': (df['over'] < 0).sum() if 'over' in df.columns else 0,
                'invalid_balls': (df['ball'] < 1).sum() if 'ball' in df.columns else 0
            },
            'coverage': {
                'matches_covered': df['match_id'].nunique(),
                'competitions_covered': df['competition_name'].nunique() if 'competition_name' in df.columns else 0,
                'venues_covered': df['venue'].nunique() if 'venue' in df.columns else 0
            }
        }
        
        return quality_metrics 