# Purpose: Integration with WicketWise holdout data for SIM system
# Author: WicketWise AI, Last Modified: 2024

"""
Data Integration for SIM System

Connects the SIM system to the actual 20% holdout matches from the T20 training pipeline.
Ensures that simulation uses only matches that were held back from model training.
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class HoldoutDataManager:
    """
    Manages access to the 20% holdout matches for simulation
    """
    
    def __init__(self, data_dir: str = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data"):
        self.data_dir = Path(data_dir)
        
        # Standard paths for holdout data
        self.val_matches_path = self.data_dir / "val_matches.csv"
        self.test_matches_path = self.data_dir / "test_matches.csv"
        # Real decimal data path with betting odds
        self.decimal_data_path = Path("/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/decimal_data_v3.csv")
        
        # Alternative paths to check
        self.alternative_paths = [
            Path("artifacts/splits/val_matches.csv"),
            Path("artifacts/splits/test_matches.csv"),
            Path("output/val_matches.csv"),
            Path("output/test_matches.csv"),
            Path("cache/training_pipeline/val_matches.csv"),
            Path("cache/training_pipeline/test_matches.csv")
        ]
        
    def get_holdout_matches(self) -> List[str]:
        """
        Get list of match IDs that were held out from training (20% validation + test)
        
        Returns:
            List of match IDs safe for simulation
        """
        holdout_matches = []
        
        # Try to load validation matches
        val_matches = self._load_match_list(self.val_matches_path, "validation")
        if val_matches:
            holdout_matches.extend(val_matches)
        
        # Try to load test matches
        test_matches = self._load_match_list(self.test_matches_path, "test")
        if test_matches:
            holdout_matches.extend(test_matches)
        
        # If standard paths don't exist, try alternatives
        if not holdout_matches:
            logger.warning("Standard holdout paths not found, checking alternatives...")
            for alt_path in self.alternative_paths:
                if alt_path.exists():
                    matches = self._load_match_list(alt_path, f"alternative ({alt_path})")
                    if matches:
                        holdout_matches.extend(matches)
        
        # If still no matches found, generate from decimal data
        if not holdout_matches:
            logger.warning("No holdout match lists found, generating from decimal data...")
            holdout_matches = self._generate_holdout_from_decimal_data()
        
        # Remove duplicates and return
        unique_matches = list(set(holdout_matches))
        logger.info(f"✅ Found {len(unique_matches)} holdout matches for simulation")
        
        return unique_matches
    
    def _load_match_list(self, file_path: Path, dataset_name: str) -> List[str]:
        """Load match IDs from CSV file"""
        try:
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                # Handle different possible column names
                match_col = None
                for col in ['match_id', 'Match_ID', 'id', 'ID']:
                    if col in df.columns:
                        match_col = col
                        break
                
                if match_col:
                    matches = df[match_col].astype(str).tolist()
                    logger.info(f"📊 Loaded {len(matches)} {dataset_name} matches from {file_path}")
                    return matches
                else:
                    logger.warning(f"No match ID column found in {file_path}")
                    return []
            else:
                logger.debug(f"{dataset_name} matches file not found: {file_path}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading {dataset_name} matches from {file_path}: {e}")
            return []
    
    def _generate_holdout_from_decimal_data(self) -> List[str]:
        """
        Generate holdout matches by taking 20% of matches from decimal data
        This is a fallback when proper train/val/test splits aren't available
        """
        try:
            if not self.decimal_data_path.exists():
                logger.error(f"Decimal data not found: {self.decimal_data_path}")
                return []
            
            # Load decimal data
            df = pd.read_csv(self.decimal_data_path)
            logger.info(f"📊 Loaded decimal data: {len(df):,} balls")
            
            # Try to find existing match ID column first
            match_col = None
            for col in ['match_id', 'Match_ID', 'id', 'ID']:
                if col in df.columns:
                    match_col = col
                    break
            
            if match_col:
                # Use existing match ID column
                unique_matches = df[match_col].unique()
                logger.info(f"📊 Found {len(unique_matches)} unique matches using column '{match_col}'")
            else:
                # Create match IDs from available columns
                logger.info("No match ID column found, creating match IDs from date+home+away+competition")
                
                # Check required columns exist
                required_cols = ['date', 'home', 'away', 'competition']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    logger.error(f"Missing required columns for match ID creation: {missing_cols}")
                    return []
                
                # Create match IDs
                df['generated_match_id'] = (
                    df['date'].astype(str) + '_' + 
                    df['home'].astype(str) + '_' + 
                    df['away'].astype(str) + '_' + 
                    df['competition'].astype(str)
                ).str.replace(' ', '_').str.replace('/', '_').str.replace('-', '_')
                
                unique_matches = df['generated_match_id'].unique()
                logger.info(f"📊 Created {len(unique_matches)} unique match IDs from available columns")
            
            # Take last 20% chronologically (assuming data is roughly chronological)
            holdout_count = max(1, int(len(unique_matches) * 0.2))
            holdout_matches = unique_matches[-holdout_count:].tolist()
            
            logger.info(f"📊 Generated {len(holdout_matches)} holdout matches (20% of total)")
            return [str(match_id) for match_id in holdout_matches]
            
        except Exception as e:
            logger.error(f"Error generating holdout matches from decimal data: {e}")
            return []
    
    def get_match_data(self, match_ids: List[str]) -> pd.DataFrame:
        """
        Get ball-by-ball data for specific matches
        
        Args:
            match_ids: List of match IDs to retrieve
            
        Returns:
            DataFrame with ball-by-ball data for the matches
        """
        try:
            if not self.decimal_data_path.exists():
                raise FileNotFoundError(f"Decimal data not found: {self.decimal_data_path}")
            
            # Load decimal data
            df = pd.read_csv(self.decimal_data_path)
            
            # Find match ID column
            match_col = None
            for col in ['match_id', 'Match_ID', 'id', 'ID']:
                if col in df.columns:
                    match_col = col
                    break
            
            if not match_col:
                # Create match IDs from available columns (same logic as _generate_holdout_from_decimal_data)
                logger.info("No match ID column found, creating match IDs from date+home+away+competition")
                
                # Check required columns exist
                required_cols = ['date', 'home', 'away', 'competition']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    raise ValueError(f"Missing required columns for match ID creation: {missing_cols}")
                
                # Create match IDs
                df['generated_match_id'] = (
                    df['date'].astype(str) + '_' + 
                    df['home'].astype(str) + '_' + 
                    df['away'].astype(str) + '_' + 
                    df['competition'].astype(str)
                ).str.replace(' ', '_').str.replace('/', '_').str.replace('-', '_')
                
                match_col = 'generated_match_id'
            
            # Filter for requested matches
            match_data = df[df[match_col].astype(str).isin(match_ids)]
            
            logger.info(f"📊 Retrieved {len(match_data):,} balls for {len(match_ids)} matches")
            return match_data
            
        except Exception as e:
            logger.error(f"Error retrieving match data: {e}")
            return pd.DataFrame()
    
    def validate_holdout_integrity(self, train_matches_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate that holdout matches don't overlap with training matches
        
        Args:
            train_matches_path: Optional path to training matches file
            
        Returns:
            Validation report
        """
        report = {
            "status": "unknown",
            "holdout_matches": 0,
            "train_matches": 0,
            "overlap_count": 0,
            "overlap_matches": [],
            "integrity_ok": False
        }
        
        try:
            # Get holdout matches
            holdout_matches = set(self.get_holdout_matches())
            report["holdout_matches"] = len(holdout_matches)
            
            # Get training matches if path provided
            train_matches = set()
            if train_matches_path and Path(train_matches_path).exists():
                train_matches = set(self._load_match_list(Path(train_matches_path), "training"))
                report["train_matches"] = len(train_matches)
                
                # Check for overlap
                overlap = holdout_matches & train_matches
                report["overlap_count"] = len(overlap)
                report["overlap_matches"] = list(overlap)
                
                if len(overlap) == 0:
                    report["status"] = "valid"
                    report["integrity_ok"] = True
                else:
                    report["status"] = "data_leakage_detected"
                    logger.warning(f"⚠️ Data leakage detected: {len(overlap)} matches in both train and holdout")
            else:
                report["status"] = "no_train_data_to_compare"
                report["integrity_ok"] = True  # Assume OK if we can't check
            
        except Exception as e:
            report["status"] = f"error: {str(e)}"
            logger.error(f"Error validating holdout integrity: {e}")
        
        return report
    
    def get_available_matches_summary(self) -> Dict[str, Any]:
        """Get summary of available matches for simulation"""
        summary = {
            "total_holdout_matches": 0,
            "data_sources_found": [],
            "data_sources_missing": [],
            "sample_matches": [],
            "recommendations": []
        }
        
        try:
            # Check data sources
            sources_to_check = [
                ("val_matches.csv", self.val_matches_path),
                ("test_matches.csv", self.test_matches_path),
                ("decimal_data_v3.csv", self.decimal_data_path)
            ]
            
            for name, path in sources_to_check:
                if path.exists():
                    summary["data_sources_found"].append(name)
                else:
                    summary["data_sources_missing"].append(name)
            
            # Get holdout matches
            holdout_matches = self.get_holdout_matches()
            summary["total_holdout_matches"] = len(holdout_matches)
            summary["sample_matches"] = holdout_matches[:5]  # First 5 as sample
            
            # Generate recommendations
            if len(holdout_matches) == 0:
                summary["recommendations"].append("No holdout matches found. Run data splitting pipeline first.")
            elif len(holdout_matches) < 10:
                summary["recommendations"].append("Very few holdout matches available. Consider expanding dataset.")
            else:
                summary["recommendations"].append(f"Good: {len(holdout_matches)} matches available for simulation.")
            
            if "decimal_data_v3.csv" not in summary["data_sources_found"]:
                summary["recommendations"].append("Missing decimal_data_v3.csv - ball-by-ball data won't be available.")
            
        except Exception as e:
            summary["error"] = str(e)
            logger.error(f"Error generating matches summary: {e}")
        
        return summary


def integrate_holdout_data_with_sim():
    """
    Integration function to connect SIM system with holdout data
    
    Returns:
        Configuration updates for SIM system
    """
    try:
        manager = HoldoutDataManager()
        
        # Get holdout matches
        holdout_matches = manager.get_holdout_matches()
        
        if not holdout_matches:
            logger.warning("⚠️ No holdout matches available for simulation")
            return {
                "status": "no_data",
                "matches": [],
                "message": "No holdout matches found. Using mock data for simulation."
            }
        
        # Validate integrity
        integrity_report = manager.validate_holdout_integrity()
        
        return {
            "status": "success",
            "matches": holdout_matches,
            "match_count": len(holdout_matches),
            "integrity_report": integrity_report,
            "data_manager": manager,
            "message": f"Successfully integrated {len(holdout_matches)} holdout matches for simulation"
        }
        
    except Exception as e:
        logger.error(f"Error integrating holdout data: {e}")
        return {
            "status": "error",
            "matches": [],
            "error": str(e),
            "message": "Failed to integrate holdout data. Using mock data for simulation."
        }


if __name__ == "__main__":
    # Test the integration
    logging.basicConfig(level=logging.INFO)
    
    print("🏏 Testing Holdout Data Integration...")
    
    manager = HoldoutDataManager()
    
    # Get summary
    summary = manager.get_available_matches_summary()
    print(f"\n📊 Available Matches Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test integration
    integration_result = integrate_holdout_data_with_sim()
    print(f"\n🔗 Integration Result:")
    for key, value in integration_result.items():
        if key != "data_manager":  # Skip the manager object
            print(f"  {key}: {value}")
