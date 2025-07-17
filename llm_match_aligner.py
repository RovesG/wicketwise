# Purpose: LLM-enhanced match alignment for cricket data sources
# Author: Assistant, Last Modified: 2024

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import openai
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ColumnMapping:
    """Column mapping configuration from LLM analysis."""
    nvplay_columns: Dict[str, str]
    decimal_columns: Dict[str, str]
    confidence: float
    reasoning: str

@dataclass
class SimilarityConfig:
    """Similarity configuration from LLM analysis."""
    recommended_threshold: float
    fingerprint_length: int
    weight_factors: Dict[str, float]
    reasoning: str

class LLMMatchAligner:
    """
    Enhanced match aligner using LLM for intelligent column mapping and configuration.
    
    This version uses OpenAI GPT-4 to:
    1. Analyze column structures and suggest optimal mappings
    2. Recommend similarity thresholds based on data characteristics
    3. Provide intelligent name normalization strategies
    4. Suggest match validation criteria
    """
    
    def __init__(self, nvplay_path: str, decimal_path: str, openai_api_key: str):
        self.nvplay_path = Path(nvplay_path)
        self.decimal_path = Path(decimal_path)
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Load data
        self.nvplay_df = pd.read_csv(self.nvplay_path)
        self.decimal_df = pd.read_csv(self.decimal_path)
        
        # LLM-generated configurations
        self.column_mapping: Optional[ColumnMapping] = None
        self.similarity_config: Optional[SimilarityConfig] = None
        
        logger.info(f"Loaded NVPlay data: {len(self.nvplay_df)} records")
        logger.info(f"Loaded Decimal data: {len(self.decimal_df)} records")
    
    def analyze_column_structure(self) -> ColumnMapping:
        """Use LLM to analyze column structures and suggest optimal mappings."""
        
        # Prepare column information for LLM
        nvplay_columns = list(self.nvplay_df.columns)
        decimal_columns = list(self.decimal_df.columns)
        
        # Sample data for context
        nvplay_sample = self.nvplay_df.head(3).to_dict('records')
        decimal_sample = self.decimal_df.head(3).to_dict('records')
        
        prompt = f"""
        Analyze these two cricket datasets and provide optimal column mappings for match fingerprinting.
        
        NVPLAY DATASET:
        Columns: {nvplay_columns}
        Sample data: {nvplay_sample}
        
        DECIMAL DATASET:
        Columns: {decimal_columns}
        Sample data: {decimal_sample}
        
        TASK: Create column mappings for match fingerprinting that uses:
        1. Match identifier (match name/ID)
        2. Ball sequence (over, ball number)
        3. Player information (batter, bowler)
        4. Ball outcome (runs scored)
        5. Additional context (innings, team info)
        
        REQUIREMENTS:
        - Map equivalent columns between datasets
        - Handle missing or differently named columns
        - Suggest data transformations if needed
        - Assess confidence level (0-1)
        
        Return JSON format:
        {{
            "nvplay_columns": {{
                "match_id": "column_name",
                "over": "column_name", 
                "ball": "column_name",
                "batter": "column_name",
                "bowler": "column_name",
                "runs": "column_name",
                "innings": "column_name"
            }},
            "decimal_columns": {{
                "match_id": "column_name or transformation",
                "over": "column_name or transformation",
                "ball": "column_name or transformation", 
                "batter": "column_name",
                "bowler": "column_name",
                "runs": "column_name",
                "innings": "column_name"
            }},
            "confidence": 0.85,
            "reasoning": "Detailed explanation of mappings and any concerns"
        }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            self.column_mapping = ColumnMapping(
                nvplay_columns=result["nvplay_columns"],
                decimal_columns=result["decimal_columns"],
                confidence=result["confidence"],
                reasoning=result["reasoning"]
            )
            
            logger.info(f"LLM Column Mapping (confidence: {self.column_mapping.confidence:.2f})")
            logger.info(f"Reasoning: {self.column_mapping.reasoning}")
            
            return self.column_mapping
            
        except Exception as e:
            logger.error(f"LLM column analysis failed: {e}")
            # Fallback to manual mapping
            return self._fallback_column_mapping()
    
    def analyze_similarity_requirements(self) -> SimilarityConfig:
        """Use LLM to analyze data characteristics and suggest similarity configuration."""
        
        # Analyze data characteristics
        nvplay_stats = {
            "total_matches": self.nvplay_df['Match'].nunique() if 'Match' in self.nvplay_df.columns else 0,
            "total_balls": len(self.nvplay_df),
            "competitions": self.nvplay_df['Competition'].unique().tolist() if 'Competition' in self.nvplay_df.columns else [],
            "sample_players": self.nvplay_df['Batter'].unique()[:10].tolist() if 'Batter' in self.nvplay_df.columns else []
        }
        
        decimal_stats = {
            "total_matches": len(self.decimal_df.groupby(['home', 'away', 'date'])) if all(col in self.decimal_df.columns for col in ['home', 'away', 'date']) else 0,
            "total_balls": len(self.decimal_df),
            "competitions": self.decimal_df['competition'].unique().tolist() if 'competition' in self.decimal_df.columns else [],
            "sample_players": self.decimal_df['batsman'].unique()[:10].tolist() if 'batsman' in self.decimal_df.columns else []
        }
        
        prompt = f"""
        Analyze these cricket datasets and recommend optimal similarity configuration for match fingerprinting.
        
        NVPLAY DATASET STATS:
        {json.dumps(nvplay_stats, indent=2)}
        
        DECIMAL DATASET STATS:
        {json.dumps(decimal_stats, indent=2)}
        
        CONSIDERATIONS:
        1. Player name variations (e.g., "JR Philippe" vs "J Philippe")
        2. Data quality and completeness
        3. Match overlap likelihood
        4. Optimal fingerprint length (number of balls to compare)
        5. Similarity threshold for reliable matching
        
        TASK: Recommend configuration for:
        - Similarity threshold (0.5-1.0)
        - Fingerprint length (5-50 balls)
        - Weight factors for different components
        - Expected match rate
        
        Return JSON format:
        {{
            "recommended_threshold": 0.75,
            "fingerprint_length": 20,
            "weight_factors": {{
                "player_names": 0.4,
                "ball_sequence": 0.3,
                "runs_pattern": 0.2,
                "match_context": 0.1
            }},
            "expected_match_rate": 0.65,
            "reasoning": "Detailed explanation of recommendations"
        }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            self.similarity_config = SimilarityConfig(
                recommended_threshold=result["recommended_threshold"],
                fingerprint_length=result["fingerprint_length"],
                weight_factors=result["weight_factors"],
                reasoning=result["reasoning"]
            )
            
            logger.info(f"LLM Similarity Config (threshold: {self.similarity_config.recommended_threshold})")
            logger.info(f"Reasoning: {self.similarity_config.reasoning}")
            
            return self.similarity_config
            
        except Exception as e:
            logger.error(f"LLM similarity analysis failed: {e}")
            # Fallback configuration
            return SimilarityConfig(
                recommended_threshold=0.8,
                fingerprint_length=15,
                weight_factors={"player_names": 0.5, "runs_pattern": 0.3, "ball_sequence": 0.2},
                reasoning="Fallback configuration due to LLM analysis failure"
            )
    
    def extract_fingerprints_with_llm_mapping(self) -> Tuple[Dict[str, List], Dict[str, List]]:
        """Extract fingerprints using LLM-determined column mappings."""
        
        if not self.column_mapping:
            self.analyze_column_structure()
        
        nvplay_fingerprints = {}
        decimal_fingerprints = {}
        
        # Extract NVPlay fingerprints
        nvplay_mapping = self.column_mapping.nvplay_columns
        for match_id in self.nvplay_df[nvplay_mapping['match_id']].unique():
            match_data = self.nvplay_df[self.nvplay_df[nvplay_mapping['match_id']] == match_id]
            match_data = match_data.sort_values([nvplay_mapping['innings'], nvplay_mapping['over'], nvplay_mapping['ball']])
            
            fingerprint = []
            for _, row in match_data.head(self.similarity_config.fingerprint_length).iterrows():
                fingerprint.append({
                    'over': row[nvplay_mapping['over']],
                    'ball': row[nvplay_mapping['ball']],
                    'batter': self._normalize_name(str(row[nvplay_mapping['batter']])),
                    'bowler': self._normalize_name(str(row[nvplay_mapping['bowler']])),
                    'runs': row[nvplay_mapping['runs']],
                    'innings': row[nvplay_mapping['innings']]
                })
            
            nvplay_fingerprints[match_id] = fingerprint
        
        # Extract Decimal fingerprints with transformations
        decimal_mapping = self.column_mapping.decimal_columns
        
        # Handle match_id creation if it's a transformation
        if 'transformation' in decimal_mapping['match_id']:
            # Create composite match ID
            self.decimal_df['match_id'] = (
                self.decimal_df['competition'].astype(str) + '_' +
                self.decimal_df['home'].astype(str) + '_vs_' +
                self.decimal_df['away'].astype(str) + '_' +
                self.decimal_df['date'].astype(str)
            )
            match_id_col = 'match_id'
        else:
            match_id_col = decimal_mapping['match_id']
        
        for match_id in self.decimal_df[match_id_col].unique():
            match_data = self.decimal_df[self.decimal_df[match_id_col] == match_id]
            match_data = match_data.sort_values([decimal_mapping['innings'], decimal_mapping['ball']])
            
            fingerprint = []
            for _, row in match_data.head(self.similarity_config.fingerprint_length).iterrows():
                # Handle ball number transformation if needed
                ball_info = self._parse_ball_number(row[decimal_mapping['ball']])
                
                fingerprint.append({
                    'over': ball_info['over'],
                    'ball': ball_info['ball'],
                    'batter': self._normalize_name(str(row[decimal_mapping['batter']])),
                    'bowler': self._normalize_name(str(row[decimal_mapping['bowler']])),
                    'runs': row[decimal_mapping['runs']],
                    'innings': row[decimal_mapping['innings']]
                })
            
            decimal_fingerprints[match_id] = fingerprint
        
        return nvplay_fingerprints, decimal_fingerprints
    
    def calculate_weighted_similarity(self, fp1: List[Dict], fp2: List[Dict]) -> float:
        """Calculate weighted similarity using LLM-suggested factors."""
        
        if not fp1 or not fp2:
            return 0.0
        
        min_length = min(len(fp1), len(fp2))
        weights = self.similarity_config.weight_factors
        
        total_score = 0.0
        for i in range(min_length):
            ball1, ball2 = fp1[i], fp2[i]
            
            # Player name similarity
            player_score = 0.0
            if ball1['batter'] == ball2['batter']:
                player_score += 0.5
            if ball1['bowler'] == ball2['bowler']:
                player_score += 0.5
            
            # Ball sequence similarity
            sequence_score = 0.0
            if ball1['over'] == ball2['over']:
                sequence_score += 0.5
            if ball1['ball'] == ball2['ball']:
                sequence_score += 0.5
            
            # Runs pattern similarity
            runs_score = 1.0 if ball1['runs'] == ball2['runs'] else 0.0
            
            # Match context similarity
            context_score = 1.0 if ball1['innings'] == ball2['innings'] else 0.0
            
            # Weighted combination
            ball_similarity = (
                weights['player_names'] * player_score +
                weights['ball_sequence'] * sequence_score +
                weights['runs_pattern'] * runs_score +
                weights['match_context'] * context_score
            )
            
            total_score += ball_similarity
        
        return total_score / min_length
    
    def find_matches_with_llm_config(self) -> List[Dict]:
        """Find matches using LLM-optimized configuration."""
        
        # Ensure configurations are loaded
        if not self.column_mapping:
            self.analyze_column_structure()
        if not self.similarity_config:
            self.analyze_similarity_requirements()
        
        # Extract fingerprints
        nvplay_fps, decimal_fps = self.extract_fingerprints_with_llm_mapping()
        
        matches = []
        threshold = self.similarity_config.recommended_threshold
        
        logger.info(f"Finding matches with LLM threshold: {threshold}")
        
        for nvplay_id, nvplay_fp in nvplay_fps.items():
            for decimal_id, decimal_fp in decimal_fps.items():
                similarity = self.calculate_weighted_similarity(nvplay_fp, decimal_fp)
                
                if similarity >= threshold:
                    matches.append({
                        'nvplay_match_id': nvplay_id,
                        'decimal_match_id': decimal_id,
                        'similarity_score': similarity,
                        'confidence': 'high' if similarity > 0.9 else 'medium'
                    })
                    
                    logger.info(f"Match found: {nvplay_id} <-> {decimal_id} (similarity: {similarity:.3f})")
        
        logger.info(f"Found {len(matches)} matches using LLM configuration")
        return matches
    
    def _normalize_name(self, name: str) -> str:
        """Normalize player names for better matching."""
        return name.strip().upper().replace('.', '').replace(' ', '')
    
    def _parse_ball_number(self, ball_str: str) -> Dict[str, int]:
        """Parse ball number format (e.g., '13.2' -> over 13, ball 2)."""
        try:
            if '.' in str(ball_str):
                over, ball = str(ball_str).split('.')
                return {'over': int(float(over)), 'ball': int(ball)}
            else:
                # Assume continuous ball numbering
                ball_num = int(ball_str)
                over = (ball_num - 1) // 6 + 1
                ball = ((ball_num - 1) % 6) + 1
                return {'over': over, 'ball': ball}
        except:
            return {'over': 0, 'ball': 0}
    
    def _fallback_column_mapping(self) -> ColumnMapping:
        """Fallback column mapping if LLM analysis fails."""
        return ColumnMapping(
            nvplay_columns={
                "match_id": "Match",
                "over": "Over",
                "ball": "Ball", 
                "batter": "Batter",
                "bowler": "Bowler",
                "runs": "Runs",
                "innings": "Innings"
            },
            decimal_columns={
                "match_id": "transformation",
                "over": "transformation",
                "ball": "ball",
                "batter": "batsman",
                "bowler": "bowler", 
                "runs": "runs",
                "innings": "innings"
            },
            confidence=0.6,
            reasoning="Fallback mapping based on manual analysis"
        )


def align_matches_with_llm(nvplay_path: str, decimal_path: str, openai_api_key: str, 
                          output_path: str = "llm_aligned_matches.csv") -> List[Dict]:
    """
    Convenience function to align matches using LLM enhancement.
    
    Args:
        nvplay_path: Path to NVPlay CSV file
        decimal_path: Path to decimal CSV file
        openai_api_key: OpenAI API key
        output_path: Output CSV file path
        
    Returns:
        List of matched pairs
    """
    aligner = LLMMatchAligner(nvplay_path, decimal_path, openai_api_key)
    matches = aligner.find_matches_with_llm_config()
    
    # Save results
    if matches:
        df = pd.DataFrame(matches)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(matches)} matches to {output_path}")
    
    return matches


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-enhanced cricket match alignment")
    parser.add_argument("nvplay_file", help="NVPlay CSV file path")
    parser.add_argument("decimal_file", help="Decimal CSV file path")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--output", default="llm_aligned_matches.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    align_matches_with_llm(args.nvplay_file, args.decimal_file, args.api_key, args.output) 