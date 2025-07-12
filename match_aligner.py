# Purpose: Identifies overlapping matches across two cricket data sources using ball-by-ball sequence fingerprints
# Author: Assistant, Last Modified: 2024

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MatchAligner:
    """
    Identifies overlapping matches between nvplay.csv and decimal.csv data sources.
    
    Instead of relying on match IDs, this aligner uses ball-by-ball sequence fingerprints
    to find matches that represent the same cricket match across different data sources.
    """
    
    def __init__(self, nvplay_path: str, decimal_path: str, fingerprint_length: int = 50):
        """
        Initialize the MatchAligner.
        
        Args:
            nvplay_path: Path to nvplay.csv file
            decimal_path: Path to decimal.csv file
            fingerprint_length: Number of balls to use for fingerprint (default: 50)
        """
        self.nvplay_path = Path(nvplay_path)
        self.decimal_path = Path(decimal_path)
        self.fingerprint_length = fingerprint_length
        
        # Load data
        self.nvplay_df = None
        self.decimal_df = None
        self._load_data()
        
        # Extract match fingerprints
        self.nvplay_fingerprints = {}
        self.decimal_fingerprints = {}
        self._extract_fingerprints()
    
    def _load_data(self):
        """Load CSV files into pandas DataFrames."""
        if not self.nvplay_path.exists():
            raise FileNotFoundError(f"NVPlay data file not found: {self.nvplay_path}")
        if not self.decimal_path.exists():
            raise FileNotFoundError(f"Decimal data file not found: {self.decimal_path}")
        
        logger.info(f"Loading NVPlay data from {self.nvplay_path}")
        self.nvplay_df = pd.read_csv(self.nvplay_path)
        
        logger.info(f"Loading decimal data from {self.decimal_path}")
        self.decimal_df = pd.read_csv(self.decimal_path)
        
        logger.info(f"Loaded {len(self.nvplay_df)} NVPlay records")
        logger.info(f"Loaded {len(self.decimal_df)} decimal records")
    
    def _extract_fingerprints(self):
        """Extract match fingerprints from both data sources."""
        logger.info("Extracting fingerprints from NVPlay data...")
        self.nvplay_fingerprints = self._extract_nvplay_fingerprints()
        
        logger.info("Extracting fingerprints from decimal data...")
        self.decimal_fingerprints = self._extract_decimal_fingerprints()
        
        logger.info(f"Extracted {len(self.nvplay_fingerprints)} NVPlay fingerprints")
        logger.info(f"Extracted {len(self.decimal_fingerprints)} decimal fingerprints")
    
    def _extract_nvplay_fingerprints(self) -> Dict[str, List[Tuple]]:
        """
        Extract fingerprints from NVPlay data.
        
        Returns:
            Dictionary mapping match_id to list of (over, ball, batter, bowler, runs) tuples
        """
        fingerprints = {}
        
        # Group by match
        for match_id in self.nvplay_df['Match'].unique():
            match_data = self.nvplay_df[self.nvplay_df['Match'] == match_id]
            
            # Sort by innings and ball number
            match_data = match_data.sort_values(['Innings', 'Innings Ball'])
            
            # Extract first N balls
            first_balls = match_data.head(self.fingerprint_length)
            
            # Create fingerprint: sequence of (over, ball, batter, bowler, runs)
            fingerprint = []
            for _, row in first_balls.iterrows():
                fingerprint.append((
                    float(row['Over']),
                    int(row['Ball']),
                    str(row['Batter']),
                    str(row['Bowler']),
                    int(row['Runs'])
                ))
            
            fingerprints[match_id] = fingerprint
        
        return fingerprints
    
    def _extract_decimal_fingerprints(self) -> Dict[str, List[Tuple]]:
        """
        Extract fingerprints from decimal data.
        
        Returns:
            Dictionary mapping match_id to list of (over, ball, batter, bowler, runs) tuples
        """
        fingerprints = {}
        
        # Create match identifier from competition, home, away, date
        self.decimal_df['match_id'] = (
            self.decimal_df['competition'].astype(str) + '_' + 
            self.decimal_df['home'].astype(str) + '_vs_' + 
            self.decimal_df['away'].astype(str) + '_' + 
            self.decimal_df['date'].astype(str)
        )
        
        # Group by match
        for match_id in self.decimal_df['match_id'].unique():
            match_data = self.decimal_df[self.decimal_df['match_id'] == match_id]
            
            # Sort by innings and ball number
            match_data = match_data.sort_values(['innings', 'ball'])
            
            # Extract first N balls
            first_balls = match_data.head(self.fingerprint_length)
            
            # Create fingerprint: sequence of (over, ball, batter, bowler, runs)
            fingerprint = []
            for _, row in first_balls.iterrows():
                # Calculate over from ball number (assuming 6 balls per over)
                ball_num = int(row['ball'])
                over = (ball_num - 1) // 6 + 1
                ball_in_over = ((ball_num - 1) % 6) + 1
                
                fingerprint.append((
                    float(over),
                    int(ball_in_over),
                    str(row.get('batter', 'Unknown')),
                    str(row.get('bowler', 'Unknown')),
                    int(row.get('runs', 0))
                ))
            
            fingerprints[match_id] = fingerprint
        
        return fingerprints
    
    def _calculate_similarity(self, fingerprint1: List[Tuple], fingerprint2: List[Tuple]) -> float:
        """
        Calculate similarity score between two fingerprints.
        
        Args:
            fingerprint1: First fingerprint
            fingerprint2: Second fingerprint
            
        Returns:
            Similarity score as percentage (0.0 to 1.0)
        """
        if not fingerprint1 or not fingerprint2:
            return 0.0
        
        # Take minimum length to avoid index errors
        min_length = min(len(fingerprint1), len(fingerprint2))
        
        matches = 0
        for i in range(min_length):
            # Compare (batter, bowler, runs) tuples - ignore over/ball as they might differ
            tuple1 = fingerprint1[i]
            tuple2 = fingerprint2[i]
            
            # Extract (batter, bowler, runs) from each tuple
            batter1, bowler1, runs1 = tuple1[2], tuple1[3], tuple1[4]
            batter2, bowler2, runs2 = tuple2[2], tuple2[3], tuple2[4]
            
            if batter1 == batter2 and bowler1 == bowler2 and runs1 == runs2:
                matches += 1
        
        return matches / min_length if min_length > 0 else 0.0
    
    def find_matches(self, similarity_threshold: float = 0.9) -> List[Dict[str, any]]:
        """
        Find overlapping matches between the two data sources.
        
        Args:
            similarity_threshold: Minimum similarity score to consider a match (default: 0.9)
            
        Returns:
            List of match dictionaries with nvplay_match_id, decimal_match_id, and similarity_score
        """
        matches = []
        
        logger.info(f"Finding matches with similarity threshold: {similarity_threshold}")
        
        # Compare each nvplay match with each decimal match
        for nvplay_match_id, nvplay_fingerprint in self.nvplay_fingerprints.items():
            for decimal_match_id, decimal_fingerprint in self.decimal_fingerprints.items():
                similarity = self._calculate_similarity(nvplay_fingerprint, decimal_fingerprint)
                
                if similarity >= similarity_threshold:
                    matches.append({
                        'nvplay_match_id': nvplay_match_id,
                        'decimal_match_id': decimal_match_id,
                        'similarity_score': similarity
                    })
                    logger.info(f"Match found: {nvplay_match_id} <-> {decimal_match_id} (similarity: {similarity:.3f})")
        
        logger.info(f"Found {len(matches)} matching pairs")
        return matches
    
    def save_matches(self, matches: List[Dict[str, any]], output_path: str):
        """
        Save matched matches to CSV file.
        
        Args:
            matches: List of match dictionaries
            output_path: Path to save the CSV file
        """
        if not matches:
            logger.warning("No matches to save")
            return
        
        # Convert to DataFrame
        matches_df = pd.DataFrame(matches)
        
        # Save to CSV
        output_path = Path(output_path)
        matches_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(matches)} matches to {output_path}")


def align_matches(nvplay_path: str, decimal_path: str, output_path: str = "matched_matches.csv",
                  fingerprint_length: int = 50, similarity_threshold: float = 0.9):
    """
    Main function to align matches between two cricket data sources.
    
    Args:
        nvplay_path: Path to nvplay.csv file
        decimal_path: Path to decimal.csv file
        output_path: Path to save matched results (default: "matched_matches.csv")
        fingerprint_length: Number of balls to use for fingerprint (default: 50)
        similarity_threshold: Minimum similarity score to consider a match (default: 0.9)
    """
    # Create aligner
    aligner = MatchAligner(nvplay_path, decimal_path, fingerprint_length)
    
    # Find matches
    matches = aligner.find_matches(similarity_threshold)
    
    # Save results
    aligner.save_matches(matches, output_path)
    
    return matches


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Align cricket matches between two data sources")
    parser.add_argument("nvplay_path", help="Path to nvplay.csv file")
    parser.add_argument("decimal_path", help="Path to decimal.csv file")
    parser.add_argument("--output", "-o", default="matched_matches.csv", help="Output CSV file path")
    parser.add_argument("--fingerprint-length", "-f", type=int, default=50, help="Number of balls for fingerprint")
    parser.add_argument("--threshold", "-t", type=float, default=0.9, help="Similarity threshold")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run alignment
    matches = align_matches(
        args.nvplay_path,
        args.decimal_path,
        args.output,
        args.fingerprint_length,
        args.threshold
    )
    
    print(f"✅ Found {len(matches)} matching pairs")
    print(f"📄 Results saved to {args.output}") 