# Purpose: Enhanced training pipeline that automatically integrates enriched data
# Author: WicketWise Team, Last Modified: 2025-01-22

"""
Enhanced Training Pipeline with Automatic Enrichment Integration

This module provides:
- Automatic enriched data integration during model training
- Entity harmonization across all data sources
- Incremental enrichment for new matches
- Unified data pipeline for KG and T20 model training
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import os
from datetime import datetime

from entity_harmonizer import get_entity_harmonizer
from openai_match_enrichment_pipeline import MatchEnrichmentPipeline

logger = logging.getLogger(__name__)

class EnrichedTrainingPipeline:
    """
    Enhanced training pipeline that automatically integrates enriched data
    and provides unified data processing for both KG and T20 model training.
    """
    
    def __init__(self, 
                 data_dir: str = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data",
                 enriched_data_dir: str = "enriched_data",
                 cache_dir: str = "cache/training_pipeline"):
        
        self.data_dir = Path(data_dir)
        self.enriched_data_dir = Path(enriched_data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Core data paths - Use Parquet for efficiency with large datasets
        self.json_data_path = Path("artifacts/kg_background/events/events.parquet")  # For KG (10M+ balls)
        self.decimal_data_path = self.data_dir / "decimal_data_v3.csv"  # For T20 model
        self.enriched_matches_path = self.enriched_data_dir / "enriched_betting_matches.json"
        
        # Initialize components
        self.entity_harmonizer = get_entity_harmonizer()
        
        # Initialize enrichment pipeline only if OpenAI API key is available
        try:
            if os.getenv('OPENAI_API_KEY'):
                self.enrichment_pipeline = MatchEnrichmentPipeline()
            else:
                logger.warning("⚠️ OpenAI API key not found, enrichment will be skipped")
                self.enrichment_pipeline = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize enrichment pipeline: {e}")
            self.enrichment_pipeline = None
        
        logger.info("🚀 Enhanced Training Pipeline initialized")
    
    def prepare_kg_training_data(self, 
                                auto_enrich: bool = True,
                                max_new_matches: int = 100) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare harmonized training data for Knowledge Graph building
        
        Args:
            auto_enrich: Whether to automatically enrich new matches
            max_new_matches: Maximum new matches to enrich
            
        Returns:
            Tuple of (harmonized_dataframe, enrichment_metadata)
        """
        logger.info("🔧 Preparing KG training data with enrichment integration...")
        
        # Load JSON dataset (10M+ balls for KG)
        if not self.json_data_path.exists():
            raise FileNotFoundError(f"JSON dataset not found: {self.json_data_path}")
            
        # Load from Parquet for better performance with large datasets
        if self.json_data_path.suffix == '.parquet':
            df = pd.read_parquet(self.json_data_path)
        else:
            df = pd.read_csv(self.json_data_path)
        logger.info(f"📊 Loaded {len(df):,} balls from JSON dataset")
        
        # Auto-enrich if requested
        enrichment_metadata = {}
        if auto_enrich:
            enrichment_metadata = self._auto_enrich_matches(
                df, max_new_matches, data_source="json"
            )
        
        # Apply entity harmonization
        df_harmonized = self.entity_harmonizer.harmonize_dataset(
            df,
            player_columns=['batsman', 'bowler'],
            team_columns=['battingteam', 'home', 'away'],
            venue_columns=['venue']
        )
        
        # Integrate enriched data if available
        if self.enriched_matches_path.exists():
            df_harmonized = self._integrate_enriched_data(df_harmonized, "kg")
        
        logger.info(f"✅ KG training data prepared: {len(df_harmonized):,} balls")
        return df_harmonized, enrichment_metadata
    
    def prepare_t20_training_data(self, 
                                 auto_enrich: bool = True,
                                 max_new_matches: int = 100) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare harmonized training data for T20 Model (Crickformer) training
        
        Args:
            auto_enrich: Whether to automatically enrich new matches
            max_new_matches: Maximum new matches to enrich
            
        Returns:
            Tuple of (harmonized_dataframe, enrichment_metadata)
        """
        logger.info("🔧 Preparing T20 Model training data with enrichment integration...")
        
        # Load Decimal dataset (betting data for T20 model)
        if not self.decimal_data_path.exists():
            raise FileNotFoundError(f"Decimal dataset not found: {self.decimal_data_path}")
            
        df = pd.read_csv(self.decimal_data_path)
        logger.info(f"📊 Loaded {len(df):,} balls from Decimal dataset")
        
        # Auto-enrich if requested
        enrichment_metadata = {}
        if auto_enrich:
            enrichment_metadata = self._auto_enrich_matches(
                df, max_new_matches, data_source="decimal"
            )
        
        # Apply entity harmonization
        df_harmonized = self.entity_harmonizer.harmonize_dataset(
            df,
            player_columns=['batsman', 'bowler'],
            team_columns=['battingteam', 'home', 'away'],
            venue_columns=['venue']
        )
        
        # Integrate enriched data if available
        if self.enriched_matches_path.exists():
            df_harmonized = self._integrate_enriched_data(df_harmonized, "t20")
        
        logger.info(f"✅ T20 Model training data prepared: {len(df_harmonized):,} balls")
        return df_harmonized, enrichment_metadata
    
    def _auto_enrich_matches(self, df: pd.DataFrame, max_new_matches: int, 
                           data_source: str) -> Dict[str, Any]:
        """Automatically enrich new matches in the dataset"""
        try:
            logger.info(f"🤖 Auto-enriching new matches from {data_source} dataset...")
            
            # Check if enrichment pipeline is available
            if not self.enrichment_pipeline:
                logger.warning("⚠️ Enrichment pipeline not available, skipping auto-enrichment")
                return {'status': 'skipped', 'reason': 'no_enrichment_pipeline'}
            
            # Identify unique matches in dataset
            match_columns = ['match_id'] if 'match_id' in df.columns else []
            if not match_columns:
                # Try to identify matches by other means
                if all(col in df.columns for col in ['home', 'away', 'venue', 'date']):
                    match_columns = ['home', 'away', 'venue', 'date']
                else:
                    logger.warning("⚠️ Cannot identify unique matches, skipping auto-enrichment")
                    return {'status': 'skipped', 'reason': 'no_match_identifiers'}
            
            # Get unique matches
            unique_matches = df[match_columns].drop_duplicates()
            logger.info(f"📊 Found {len(unique_matches)} unique matches in {data_source} dataset")
            
            # Check which matches are already enriched
            enriched_count = 0
            if self.enriched_matches_path.exists():
                with open(self.enriched_matches_path, 'r') as f:
                    enriched_matches = json.load(f)
                enriched_count = len(enriched_matches)
            
            new_matches_needed = min(max_new_matches, len(unique_matches) - enriched_count)
            
            if new_matches_needed <= 0:
                logger.info("✅ All matches already enriched")
                return {'status': 'complete', 'enriched_count': enriched_count, 'new_matches': 0}
            
            # Run enrichment
            logger.info(f"🚀 Enriching {new_matches_needed} new matches...")
            
            # For now, use the betting dataset path as default
            # TODO: Make this more flexible based on data_source
            betting_data_path = str(self.decimal_data_path)
            
            result = self.enrichment_pipeline.enrich_betting_dataset(
                betting_data_path=betting_data_path,
                additional_matches=new_matches_needed,
                priority_competitions=[]  # All competitions
            )
            
            return {
                'status': 'success',
                'enriched_count': enriched_count,
                'new_matches': new_matches_needed,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"❌ Auto-enrichment failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _integrate_enriched_data(self, df: pd.DataFrame, model_type: str) -> pd.DataFrame:
        """Integrate enriched match data into the training dataset"""
        try:
            logger.info(f"🔗 Integrating enriched data for {model_type} model...")
            
            # Load enriched matches
            with open(self.enriched_matches_path, 'r') as f:
                enriched_matches = json.load(f)
            
            logger.info(f"📊 Found {len(enriched_matches)} enriched matches")
            
            # Create enrichment lookup
            enrichment_lookup = {}
            for match in enriched_matches:
                # Create match key for lookup
                key = self._create_match_key(match)
                enrichment_lookup[key] = match
            
            # Add enriched features to dataframe
            enriched_features = []
            
            for idx, row in df.iterrows():
                # Try to find enriched data for this match
                match_key = self._create_match_key_from_row(row)
                enriched_data = enrichment_lookup.get(match_key, {})
                
                # Extract enriched features
                features = {
                    'weather_temperature': enriched_data.get('weather', {}).get('temperature'),
                    'weather_humidity': enriched_data.get('weather', {}).get('humidity'),
                    'weather_conditions': enriched_data.get('weather', {}).get('conditions'),
                    'toss_winner': enriched_data.get('toss', {}).get('winner'),
                    'toss_decision': enriched_data.get('toss', {}).get('decision'),
                    'venue_latitude': enriched_data.get('venue', {}).get('latitude'),
                    'venue_longitude': enriched_data.get('venue', {}).get('longitude'),
                    'enriched': 1 if enriched_data else 0
                }
                
                enriched_features.append(features)
            
            # Add enriched features to dataframe
            enriched_df = pd.DataFrame(enriched_features)
            df_with_enriched = pd.concat([df, enriched_df], axis=1)
            
            enriched_count = df_with_enriched['enriched'].sum()
            logger.info(f"✅ Integrated enriched data: {enriched_count:,}/{len(df):,} balls ({enriched_count/len(df)*100:.1f}%)")
            
            return df_with_enriched
            
        except Exception as e:
            logger.error(f"❌ Failed to integrate enriched data: {e}")
            return df  # Return original dataframe if integration fails
    
    def _create_match_key(self, match_data: Dict[str, Any]) -> str:
        """Create a unique key for a match from enriched data"""
        match_info = match_data.get('match', {})
        home = match_info.get('home_team', '')
        away = match_info.get('away_team', '')
        venue = match_info.get('venue', '')
        date = match_info.get('date', '')
        
        return f"{home}_{away}_{venue}_{date}".lower().replace(' ', '_')
    
    def _create_match_key_from_row(self, row: pd.Series) -> str:
        """Create a unique key for a match from dataframe row"""
        # Try different column combinations
        if 'home' in row and 'away' in row:
            home = str(row.get('home', ''))
            away = str(row.get('away', ''))
            venue = str(row.get('venue', ''))
            date = str(row.get('date', ''))
        else:
            # Fallback to team names if available
            home = str(row.get('team_batting', ''))
            away = str(row.get('team_bowling', ''))
            venue = str(row.get('venue', ''))
            date = str(row.get('date', ''))
        
        return f"{home}_{away}_{venue}_{date}".lower().replace(' ', '_')
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get statistics about available training data"""
        stats = {
            'json_dataset': {
                'exists': self.json_data_path.exists(),
                'size_mb': 0,
                'rows': 0
            },
            'decimal_dataset': {
                'exists': self.decimal_data_path.exists(),
                'size_mb': 0,
                'rows': 0
            },
            'enriched_matches': {
                'exists': self.enriched_matches_path.exists(),
                'count': 0
            },
            'entity_harmonizer': self.entity_harmonizer.get_entity_stats()
        }
        
        # JSON dataset stats
        if self.json_data_path.exists():
            size_bytes = self.json_data_path.stat().st_size
            stats['json_dataset']['size_mb'] = round(size_bytes / (1024 * 1024), 1)
            try:
                if self.json_data_path.suffix == '.parquet':
                    # For Parquet, read just the metadata to get row count efficiently
                    import pyarrow.parquet as pq
                    table = pq.read_table(self.json_data_path)
                    stats['json_dataset']['rows'] = table.num_rows
                else:
                    # For CSV, count lines
                    total_rows = sum(1 for _ in open(self.json_data_path)) - 1  # Subtract header
                    stats['json_dataset']['rows'] = total_rows
            except Exception as e:
                logger.warning(f"Could not get row count for JSON dataset: {e}")
                pass
        
        # Decimal dataset stats
        if self.decimal_data_path.exists():
            size_bytes = self.decimal_data_path.stat().st_size
            stats['decimal_dataset']['size_mb'] = round(size_bytes / (1024 * 1024), 1)
            try:
                df = pd.read_csv(self.decimal_data_path, nrows=1)
                total_rows = sum(1 for _ in open(self.decimal_data_path)) - 1  # Subtract header
                stats['decimal_dataset']['rows'] = total_rows
            except:
                pass
        
        # Enriched matches stats
        if self.enriched_matches_path.exists():
            try:
                with open(self.enriched_matches_path, 'r') as f:
                    enriched_matches = json.load(f)
                stats['enriched_matches']['count'] = len(enriched_matches)
            except:
                pass
        
        return stats
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity across all sources"""
        validation_results = {
            'json_dataset': {'valid': False, 'issues': []},
            'decimal_dataset': {'valid': False, 'issues': []},
            'enriched_data': {'valid': False, 'issues': []},
            'entity_harmonization': {'valid': False, 'issues': []}
        }
        
        # Validate JSON dataset
        try:
            if self.json_data_path.exists():
                df = pd.read_csv(self.json_data_path, nrows=1000)  # Sample validation
                required_cols = ['batsman', 'bowler', 'venue']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    validation_results['json_dataset']['issues'].append(f"Missing columns: {missing_cols}")
                else:
                    validation_results['json_dataset']['valid'] = True
            else:
                validation_results['json_dataset']['issues'].append("File not found")
        except Exception as e:
            validation_results['json_dataset']['issues'].append(f"Read error: {e}")
        
        # Validate Decimal dataset
        try:
            if self.decimal_data_path.exists():
                df = pd.read_csv(self.decimal_data_path, nrows=1000)  # Sample validation
                required_cols = ['batsman', 'bowler']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    validation_results['decimal_dataset']['issues'].append(f"Missing columns: {missing_cols}")
                else:
                    validation_results['decimal_dataset']['valid'] = True
            else:
                validation_results['decimal_dataset']['issues'].append("File not found")
        except Exception as e:
            validation_results['decimal_dataset']['issues'].append(f"Read error: {e}")
        
        # Validate enriched data
        try:
            if self.enriched_matches_path.exists():
                with open(self.enriched_matches_path, 'r') as f:
                    enriched_matches = json.load(f)
                
                if len(enriched_matches) > 0:
                    # Check structure of first match
                    first_match = enriched_matches[0]
                    if 'match' in first_match and 'weather' in first_match:
                        validation_results['enriched_data']['valid'] = True
                    else:
                        validation_results['enriched_data']['issues'].append("Invalid structure")
                else:
                    validation_results['enriched_data']['issues'].append("Empty file")
            else:
                validation_results['enriched_data']['issues'].append("File not found")
        except Exception as e:
            validation_results['enriched_data']['issues'].append(f"Read error: {e}")
        
        # Validate entity harmonization
        try:
            stats = self.entity_harmonizer.get_entity_stats()
            if stats['players']['total'] > 0:
                validation_results['entity_harmonization']['valid'] = True
            else:
                validation_results['entity_harmonization']['issues'].append("No players loaded")
        except Exception as e:
            validation_results['entity_harmonization']['issues'].append(f"Harmonizer error: {e}")
        
        return validation_results


# Global instance for easy access
_global_pipeline = None

def get_enriched_training_pipeline() -> EnrichedTrainingPipeline:
    """Get global EnrichedTrainingPipeline instance"""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = EnrichedTrainingPipeline()
    return _global_pipeline


if __name__ == "__main__":
    # Test the pipeline
    pipeline = EnrichedTrainingPipeline()
    
    # Get statistics
    stats = pipeline.get_training_statistics()
    print("\n📊 Training Data Statistics:")
    print(f"  JSON Dataset: {stats['json_dataset']['rows']:,} rows, {stats['json_dataset']['size_mb']} MB")
    print(f"  Decimal Dataset: {stats['decimal_dataset']['rows']:,} rows, {stats['decimal_dataset']['size_mb']} MB")
    print(f"  Enriched Matches: {stats['enriched_matches']['count']:,} matches")
    
    # Validate data integrity
    validation = pipeline.validate_data_integrity()
    print("\n🔍 Data Integrity Validation:")
    for dataset, result in validation.items():
        status = "✅ Valid" if result['valid'] else "❌ Issues"
        print(f"  {dataset}: {status}")
        if result['issues']:
            for issue in result['issues']:
                print(f"    - {issue}")
