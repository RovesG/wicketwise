# Purpose: Cricket match simulator engine for ball-by-ball replay
# Author: Phi1618 Cricket AI Team, Last Modified: 2024

import pandas as pd
from typing import Dict, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchSimulator:
    """
    Cricket match simulator for ball-by-ball replay functionality.
    
    Provides methods to load match data, filter by match_id, and retrieve
    ball-by-ball information with caching for performance.
    """
    
    def __init__(self, csv_path: str, match_id: Optional[str] = None):
        """
        Initialize the MatchSimulator.
        
        Args:
            csv_path: Path to the CSV file containing match data
            match_id: Optional match identifier to filter data
        """
        self.csv_path = csv_path
        self.match_id = match_id
        self._raw_data: Optional[pd.DataFrame] = None
        self._match_data: Optional[pd.DataFrame] = None
        self._cached = False
        
        logger.info(f"MatchSimulator initialized with CSV: {csv_path}")
        if match_id:
            logger.info(f"Match ID filter: {match_id}")
    
    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw data from CSV file."""
        if self._raw_data is None:
            try:
                self._raw_data = pd.read_csv(self.csv_path)
                logger.info(f"Loaded {len(self._raw_data)} rows from {self.csv_path}")
            except FileNotFoundError:
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            except Exception as e:
                raise RuntimeError(f"Error loading CSV file: {str(e)}")
        
        return self._raw_data
    
    def _filter_match_data(self) -> pd.DataFrame:
        """Filter data for the specified match_id."""
        raw_data = self._load_raw_data()
        
        if self.match_id is None:
            # If no match_id specified, use all data
            filtered_data = raw_data
        else:
            # Filter by match_id if column exists
            if 'match_id' in raw_data.columns:
                filtered_data = raw_data[raw_data['match_id'] == self.match_id]
                if len(filtered_data) == 0:
                    raise ValueError(f"No data found for match_id: {self.match_id}")
            else:
                logger.warning("No match_id column found, using all data")
                filtered_data = raw_data
        
        # Sort by ball_id to ensure correct order
        if 'ball_id' in filtered_data.columns:
            filtered_data = filtered_data.sort_values('ball_id')
        
        logger.info(f"Filtered data contains {len(filtered_data)} balls")
        return filtered_data
    
    def _ensure_cached(self) -> None:
        """Ensure match data is loaded and cached."""
        if not self._cached:
            self._match_data = self._filter_match_data()
            self._cached = True
    
    def get_match_ball_count(self) -> int:
        """
        Get the total number of balls in the match.
        
        Returns:
            Total number of balls in the match
        """
        self._ensure_cached()
        return len(self._match_data)
    
    def get_ball(self, ball_index: int) -> Dict[str, Any]:
        """
        Get ball information for a specific ball index.
        
        Args:
            ball_index: Zero-based index of the ball to retrieve
            
        Returns:
            Dictionary containing ball information
            
        Raises:
            IndexError: If ball_index is out of range
        """
        self._ensure_cached()
        
        if ball_index < 0 or ball_index >= len(self._match_data):
            raise IndexError(f"Ball index {ball_index} out of range (0-{len(self._match_data)-1})")
        
        # Get the row for this ball
        row = self._match_data.iloc[ball_index]
        
        # Build ball information dictionary with flexible column handling
        ball_info = {
            'ball_index': ball_index,
            'ball_number': ball_index + 1,  # 1-based ball number
        }
        
        # Core ball information
        ball_info['batter'] = self._get_column_value(row, ['batter', 'batsman', 'striker'])
        ball_info['bowler'] = self._get_column_value(row, ['bowler'])
        ball_info['over'] = self._get_column_value(row, ['over'])
        ball_info['ball'] = self._get_column_value(row, ['ball', 'ball_in_over'])
        ball_info['runs'] = self._get_column_value(row, ['runs', 'runs_off_bat'], default=0)
        
        # Additional information
        ball_info['extras'] = self._get_column_value(row, ['extras'], default=0)
        ball_info['total_runs'] = self._get_column_value(row, ['total_runs'], default=ball_info['runs'] + ball_info['extras'])
        ball_info['wicket'] = self._get_column_value(row, ['wicket', 'is_wicket'], default=0)
        
        # Match state information
        ball_info['phase'] = self._get_column_value(row, ['phase', 'innings'], default='Unknown')
        ball_info['dismissal'] = self._get_column_value(row, ['dismissal', 'wicket_type'], default=None)
        ball_info['match_state'] = self._build_match_state(row, ball_index)
        
        # Raw data for debugging
        ball_info['raw_data'] = row.to_dict()
        
        return ball_info
    
    def _get_column_value(self, row: pd.Series, column_names: list, default: Any = None) -> Any:
        """
        Get value from row using multiple possible column names.
        
        Args:
            row: Pandas Series (row of data)
            column_names: List of possible column names to try
            default: Default value if none of the columns exist
            
        Returns:
            Value from the first matching column, or default
        """
        for col_name in column_names:
            if col_name in row.index and pd.notna(row[col_name]):
                return row[col_name]
        return default
    
    def _build_match_state(self, row: pd.Series, ball_index: int) -> Dict[str, Any]:
        """
        Build match state information for the current ball.
        
        Args:
            row: Current ball data
            ball_index: Index of current ball
            
        Returns:
            Dictionary containing match state information
        """
        match_state = {
            'balls_bowled': ball_index + 1,
            'total_balls': len(self._match_data),
            'progress_percentage': round(((ball_index + 1) / len(self._match_data)) * 100, 1)
        }
        
        # Add cumulative statistics if available
        if ball_index >= 0:
            balls_so_far = self._match_data.iloc[:ball_index + 1]
            
            # Calculate cumulative runs
            runs_col = self._find_column(balls_so_far, ['runs', 'runs_off_bat'])
            if runs_col:
                match_state['total_runs'] = balls_so_far[runs_col].sum()
            
            # Calculate cumulative wickets
            wickets_col = self._find_column(balls_so_far, ['wicket', 'is_wicket'])
            if wickets_col:
                match_state['total_wickets'] = balls_so_far[wickets_col].sum()
            
            # Calculate run rate
            if 'total_runs' in match_state and ball_index >= 0:
                overs_bowled = (ball_index + 1) / 6.0
                if overs_bowled > 0:
                    match_state['run_rate'] = round(match_state['total_runs'] / overs_bowled, 2)
        
        return match_state
    
    def _find_column(self, df: pd.DataFrame, column_names: list) -> Optional[str]:
        """
        Find the first existing column from a list of possible names.
        
        Args:
            df: DataFrame to search
            column_names: List of possible column names
            
        Returns:
            First matching column name, or None if none found
        """
        for col_name in column_names:
            if col_name in df.columns:
                return col_name
        return None
    
    def get_match_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the match.
        
        Returns:
            Dictionary containing match summary
        """
        self._ensure_cached()
        
        summary = {
            'match_id': self.match_id,
            'total_balls': len(self._match_data),
            'total_overs': round(len(self._match_data) / 6.0, 1),
            'data_source': self.csv_path,
            'columns': list(self._match_data.columns)
        }
        
        # Add statistical summaries
        runs_col = self._find_column(self._match_data, ['runs', 'runs_off_bat'])
        if runs_col:
            summary['total_runs'] = self._match_data[runs_col].sum()
            summary['boundaries'] = len(self._match_data[self._match_data[runs_col] >= 4])
            summary['sixes'] = len(self._match_data[self._match_data[runs_col] == 6])
        
        wickets_col = self._find_column(self._match_data, ['wicket', 'is_wicket'])
        if wickets_col:
            summary['total_wickets'] = self._match_data[wickets_col].sum()
        
        # Get unique players
        batter_col = self._find_column(self._match_data, ['batter', 'batsman', 'striker'])
        if batter_col:
            summary['batters'] = list(self._match_data[batter_col].unique())
        
        bowler_col = self._find_column(self._match_data, ['bowler'])
        if bowler_col:
            summary['bowlers'] = list(self._match_data[bowler_col].unique())
        
        return summary
    
    def reset_cache(self) -> None:
        """Reset the internal cache, forcing data to be reloaded."""
        self._raw_data = None
        self._match_data = None
        self._cached = False
        logger.info("Cache reset")
    
    def set_match_id(self, match_id: str) -> None:
        """
        Set a new match_id and reset cache.
        
        Args:
            match_id: New match identifier
        """
        self.match_id = match_id
        self.reset_cache()
        logger.info(f"Match ID updated to: {match_id}")


# Convenience function for quick access
def create_simulator(csv_path: str, match_id: Optional[str] = None) -> MatchSimulator:
    """
    Create a MatchSimulator instance.
    
    Args:
        csv_path: Path to the CSV file containing match data
        match_id: Optional match identifier to filter data
        
    Returns:
        MatchSimulator instance
    """
    return MatchSimulator(csv_path, match_id) 