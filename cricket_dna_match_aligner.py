# Purpose: Hash-based match alignment using cricket data patterns instead of metadata
# Author: Shamus Rae, Last Modified: 2025-01-30

"""
CricketDNA Match Aligner - A robust approach to matching cricket datasets
by creating unique fingerprints from immutable cricket events rather than
unreliable metadata like team names or player names.

Core Philosophy:
- Every cricket match has a unique "DNA" - the sequence of runs, wickets, and events
- This DNA is consistent across different scorekeeping systems
- Hash-based matching is much more reliable than string matching on names
"""

import pandas as pd
import numpy as np
import hashlib
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import logging
from difflib import SequenceMatcher
import re

logger = logging.getLogger(__name__)

@dataclass
class CricketDNAHash:
    """Represents a match's unique cricket DNA fingerprint"""
    primary_hash: str      # Core runs/overs pattern
    secondary_hash: str    # Wickets/innings pattern  
    tertiary_hash: str     # Detailed event sequence
    match_signature: str   # Human-readable summary
    confidence: float      # Quality score (0-1)
    
    @property
    def full_hash(self) -> str:
        return f"{self.primary_hash}_{self.secondary_hash}_{self.tertiary_hash}"

@dataclass 
class MatchCandidate:
    """Potential match between two datasets"""
    decimal_match_id: str
    nvplay_match_id: str
    similarity_score: float
    hash_match_type: str  # 'exact', 'fuzzy', 'statistical'
    confidence: float
    details: Dict

class CricketDNAHasher:
    """Creates robust match fingerprints from cricket data patterns"""
    
    def __init__(self):
        self.hash_cache = {}
        
    def create_match_hash(self, match_data: pd.DataFrame, dataset_type: str) -> CricketDNAHash:
        """
        Create a comprehensive hash for a cricket match based on immutable patterns
        
        Args:
            match_data: Ball-by-ball data for the match
            dataset_type: 'decimal' or 'nvplay' for column mapping
            
        Returns:
            CricketDNAHash object with multiple hash components
        """
        try:
            # Extract core patterns
            runs_sequence = self._extract_runs_sequence(match_data, dataset_type)
            over_totals = self._extract_over_totals(match_data, dataset_type) 
            wicket_pattern = self._extract_wicket_pattern(match_data, dataset_type)
            innings_structure = self._extract_innings_structure(match_data, dataset_type)
            
            # Create hierarchical hashes
            primary_hash = self._create_primary_hash(runs_sequence, over_totals)
            secondary_hash = self._create_secondary_hash(wicket_pattern, innings_structure)
            tertiary_hash = self._create_tertiary_hash(match_data, dataset_type)
            
            # Generate human-readable signature
            match_signature = self._create_match_signature(
                runs_sequence, over_totals, wicket_pattern, innings_structure
            )
            
            # Calculate confidence based on data completeness
            confidence = self._calculate_confidence(match_data, dataset_type)
            
            return CricketDNAHash(
                primary_hash=primary_hash,
                secondary_hash=secondary_hash, 
                tertiary_hash=tertiary_hash,
                match_signature=match_signature,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error creating match hash: {e}")
            return self._create_fallback_hash(match_data, dataset_type)
    
    def _extract_runs_sequence(self, match_data: pd.DataFrame, dataset_type: str) -> str:
        """Extract ball-by-ball runs sequence"""
        if dataset_type == 'decimal':
            # Try multiple possible runs columns for decimal data
            possible_runs_cols = ['batsmanruns', 'runs', 'batsmanrunsball']
            runs_col = None
            for col in possible_runs_cols:
                if col in match_data.columns:
                    runs_col = col
                    break
            if runs_col is None:
                logger.warning(f"No runs column found in decimal data. Available columns: {list(match_data.columns)}")
                return "0"
        else:  # nvplay
            runs_col = 'Runs'
            
        # Get clean runs sequence (exclude extras for consistency)
        try:
            runs_sequence = match_data[runs_col].fillna(0).astype(int).tolist()
        except Exception as e:
            logger.warning(f"Error extracting runs sequence from {runs_col}: {e}")
            return "0"
        
        # Convert to compressed string representation
        return ','.join(map(str, runs_sequence))
    
    def _extract_over_totals(self, match_data: pd.DataFrame, dataset_type: str) -> str:
        """Extract over-by-over cumulative totals"""
        if dataset_type == 'decimal':
            over_col = 'over'
            # Use same logic as runs sequence extraction
            possible_runs_cols = ['batsmanruns', 'runs', 'batsmanrunsball']
            runs_col = None
            for col in possible_runs_cols:
                if col in match_data.columns:
                    runs_col = col
                    break
            if runs_col is None:
                logger.warning(f"No runs column found for over totals in decimal data")
                return "0"
            innings_col = 'innings'
        else:  # nvplay
            over_col = 'Over'
            runs_col = 'Runs'
            innings_col = 'Innings'
        
        over_totals = []
        
        # Group by innings and over to get cumulative totals
        for innings in sorted(match_data[innings_col].unique()):
            innings_data = match_data[match_data[innings_col] == innings]
            
            cumulative_runs = 0
            for over in sorted(innings_data[over_col].unique()):
                over_data = innings_data[innings_data[over_col] == over]
                over_runs = over_data[runs_col].sum()
                cumulative_runs += over_runs
                over_totals.append(str(cumulative_runs))
        
        return ','.join(over_totals)
    
    def _extract_wicket_pattern(self, match_data: pd.DataFrame, dataset_type: str) -> str:
        """Extract wicket timings and types"""
        if dataset_type == 'decimal':
            wicket_col = 'wicket'
            over_col = 'over'
            ball_col = 'delivery'
            wicket_type_col = 'wickettype'
        else:  # nvplay
            wicket_col = 'Wicket'
            over_col = 'Over'
            ball_col = 'Ball'
            wicket_type_col = 'Ball Outcome'  # May need adjustment
        
        wickets = []
        
        # Find all wicket events
        wicket_data = match_data[match_data[wicket_col].notna() & (match_data[wicket_col] != 0)]
        
        for _, wicket in wicket_data.iterrows():
            over = wicket[over_col]
            ball = wicket[ball_col]
            wicket_timing = f"{over}.{ball}"
            
            # Normalize wicket type
            wicket_type = str(wicket.get(wicket_type_col, 'unknown')).lower()
            wicket_type = self._normalize_wicket_type(wicket_type)
            
            wickets.append(f"{wicket_timing}:{wicket_type}")
        
        return '|'.join(wickets)
    
    def _extract_innings_structure(self, match_data: pd.DataFrame, dataset_type: str) -> str:
        """Extract innings totals and structure"""
        if dataset_type == 'decimal':
            innings_col = 'innings'
            # Use same logic as runs sequence extraction
            possible_runs_cols = ['batsmanruns', 'runs', 'batsmanrunsball']
            runs_col = None
            for col in possible_runs_cols:
                if col in match_data.columns:
                    runs_col = col
                    break
            if runs_col is None:
                logger.warning(f"No runs column found for innings structure in decimal data")
                return "0/0"
            wicket_col = 'wicket'
        else:  # nvplay
            innings_col = 'Innings'
            runs_col = 'Runs'
            wicket_col = 'Wicket'
        
        innings_totals = []
        
        for innings in sorted(match_data[innings_col].unique()):
            innings_data = match_data[match_data[innings_col] == innings]
            
            total_runs = innings_data[runs_col].sum()
            total_wickets = len(innings_data[innings_data[wicket_col].notna() & (innings_data[wicket_col] != 0)])
            
            innings_totals.append(f"{total_runs}/{total_wickets}")
        
        return '|'.join(innings_totals)
    
    def _create_primary_hash(self, runs_sequence: str, over_totals: str) -> str:
        """Create primary hash from core scoring patterns"""
        combined = f"{runs_sequence}#{over_totals}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _create_secondary_hash(self, wicket_pattern: str, innings_structure: str) -> str:
        """Create secondary hash from wicket and innings patterns"""
        combined = f"{wicket_pattern}#{innings_structure}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _create_tertiary_hash(self, match_data: pd.DataFrame, dataset_type: str) -> str:
        """Create detailed hash from additional patterns"""
        # Extract additional patterns for fine-grained matching
        if dataset_type == 'decimal':
            extras_col = 'extras'
        else:
            extras_col = 'Extra Runs'
        
        # Create pattern from extras, boundaries, etc.
        extras_pattern = match_data.get(extras_col, pd.Series()).fillna(0).sum()
        boundaries = len(match_data[match_data.get('batsmanruns', match_data.get('Runs', pd.Series())) == 4])
        sixes = len(match_data[match_data.get('batsmanruns', match_data.get('Runs', pd.Series())) == 6])
        
        pattern = f"{extras_pattern}:{boundaries}:{sixes}"
        return hashlib.md5(pattern.encode()).hexdigest()[:8]
    
    def _create_match_signature(self, runs_seq: str, over_totals: str, 
                               wickets: str, innings: str) -> str:
        """Create human-readable match signature"""
        # Extract key stats for signature
        total_runs = sum(int(x) for x in runs_seq.split(',') if x.isdigit())
        total_wickets = len(wickets.split('|')) if wickets else 0
        total_overs = len(over_totals.split(','))
        
        return f"R{total_runs}_W{total_wickets}_O{total_overs}"
    
    def _calculate_confidence(self, match_data: pd.DataFrame, dataset_type: str) -> float:
        """Calculate confidence score based on data completeness"""
        total_rows = len(match_data)
        if total_rows == 0:
            return 0.0
        
        # Check key columns availability using same logic as extraction
        if dataset_type == 'decimal':
            # Find available runs column
            possible_runs_cols = ['batsmanruns', 'runs', 'batsmanrunsball']
            runs_col = None
            for col in possible_runs_cols:
                if col in match_data.columns:
                    runs_col = col
                    break
            
            key_columns = [runs_col, 'over', 'innings'] if runs_col else ['over', 'innings']
        else:  # nvplay
            key_columns = ['Runs', 'Over', 'Innings']
        
        # Filter out None columns
        key_columns = [col for col in key_columns if col is not None]
        
        if not key_columns:
            return 0.1  # Minimal confidence if no key columns found
        
        available_columns = sum(1 for col in key_columns if col in match_data.columns)
        
        # Check data completeness
        try:
            available_cols = [col for col in key_columns if col in match_data.columns]
            if available_cols:
                non_null_ratio = match_data[available_cols].notna().mean().mean()
            else:
                non_null_ratio = 0.0
        except Exception:
            non_null_ratio = 0.0
        
        # Base confidence on availability and completeness
        confidence = (available_columns / len(key_columns)) * non_null_ratio
        
        # Bonus for having wicket data
        if dataset_type == 'decimal' and 'wicket' in match_data.columns:
            confidence += 0.1
        elif dataset_type == 'nvplay' and 'Wicket' in match_data.columns:
            confidence += 0.1
            
        return min(confidence, 1.0)
    
    def _normalize_wicket_type(self, wicket_type: str) -> str:
        """Normalize wicket type strings for consistency"""
        wicket_type = wicket_type.lower().strip()
        
        # Map variations to standard types
        if 'bowled' in wicket_type or 'b ' in wicket_type:
            return 'bowled'
        elif 'caught' in wicket_type or 'c ' in wicket_type:
            return 'caught'
        elif 'lbw' in wicket_type:
            return 'lbw'
        elif 'run' in wicket_type and 'out' in wicket_type:
            return 'runout'
        elif 'stump' in wicket_type:
            return 'stumped'
        else:
            return 'other'
    
    def _create_fallback_hash(self, match_data: pd.DataFrame, dataset_type: str) -> CricketDNAHash:
        """Create basic hash when detailed extraction fails"""
        basic_pattern = f"{len(match_data)}_{match_data.iloc[0] if len(match_data) > 0 else 'empty'}"
        basic_hash = hashlib.md5(basic_pattern.encode()).hexdigest()[:8]
        
        return CricketDNAHash(
            primary_hash=basic_hash,
            secondary_hash=basic_hash,
            tertiary_hash=basic_hash,
            match_signature="FALLBACK",
            confidence=0.1
        )

class CricketDNAMatcher:
    """Matches cricket datasets using DNA hash comparison"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.hasher = CricketDNAHasher()
        self.similarity_threshold = similarity_threshold
        
    def find_matches(self, decimal_data: pd.DataFrame, nvplay_data: pd.DataFrame) -> List[MatchCandidate]:
        """
        Find matches between decimal and nvplay datasets using cricket DNA hashes
        
        Args:
            decimal_data: Decimal cricket dataset
            nvplay_data: NVPlay cricket dataset
            
        Returns:
            List of MatchCandidate objects with similarity scores
        """
        logger.info("Creating cricket DNA hashes for datasets...")
        
        # Group data by match and create hashes
        decimal_hashes = self._create_dataset_hashes(decimal_data, 'decimal')
        nvplay_hashes = self._create_dataset_hashes(nvplay_data, 'nvplay')
        
        logger.info(f"Created {len(decimal_hashes)} decimal hashes and {len(nvplay_hashes)} nvplay hashes")
        
        # Find matches using different strategies
        matches = []
        
        # Strategy 1: Exact hash matches
        exact_matches = self._find_exact_matches(decimal_hashes, nvplay_hashes)
        matches.extend(exact_matches)
        logger.info(f"Found {len(exact_matches)} exact hash matches")
        
        # Strategy 2: Fuzzy hash matches for remaining data
        used_decimal = {m.decimal_match_id for m in exact_matches}
        used_nvplay = {m.nvplay_match_id for m in exact_matches}
        
        fuzzy_matches = self._find_fuzzy_matches(
            decimal_hashes, nvplay_hashes, used_decimal, used_nvplay
        )
        matches.extend(fuzzy_matches)
        logger.info(f"Found {len(fuzzy_matches)} fuzzy hash matches")
        
        # Strategy 3: Statistical similarity for remaining data
        used_decimal.update(m.decimal_match_id for m in fuzzy_matches)
        used_nvplay.update(m.nvplay_match_id for m in fuzzy_matches)
        
        statistical_matches = self._find_statistical_matches(
            decimal_hashes, nvplay_hashes, used_decimal, used_nvplay
        )
        matches.extend(statistical_matches)
        logger.info(f"Found {len(statistical_matches)} statistical matches")
        
        # Sort by confidence and similarity
        matches.sort(key=lambda x: (x.confidence, x.similarity_score), reverse=True)
        
        return matches
    
    def _create_dataset_hashes(self, data: pd.DataFrame, dataset_type: str) -> Dict[str, CricketDNAHash]:
        """Create hashes for all matches in a dataset"""
        hashes = {}
        
        # Group by match - need to determine match grouping column
        if dataset_type == 'decimal':
            # Try different possible match ID columns
            match_cols = ['match_id', 'Match', 'date', 'competition']
            match_col = None
            for col in match_cols:
                if col in data.columns:
                    match_col = col
                    break
            
            if match_col is None:
                # Fallback: create synthetic match groups by date + teams
                if 'date' in data.columns and 'battingteam' in data.columns:
                    data['synthetic_match_id'] = data['date'].astype(str) + '_' + data['battingteam'].astype(str)
                    match_col = 'synthetic_match_id'
                else:
                    logger.error("Cannot determine match grouping for decimal data")
                    return hashes
        else:  # nvplay
            match_col = 'Match' if 'Match' in data.columns else None
            if match_col is None:
                logger.error("Cannot find Match column in nvplay data")
                return hashes
        
        # Create hash for each match
        for match_id in data[match_col].unique():
            if pd.isna(match_id):
                continue
                
            match_data = data[data[match_col] == match_id]
            
            # Skip matches with too little data
            if len(match_data) < 10:  # At least 10 balls
                continue
                
            hash_obj = self.hasher.create_match_hash(match_data, dataset_type)
            hashes[str(match_id)] = hash_obj
        
        return hashes
    
    def _find_exact_matches(self, decimal_hashes: Dict[str, CricketDNAHash], 
                           nvplay_hashes: Dict[str, CricketDNAHash]) -> List[MatchCandidate]:
        """Find exact hash matches"""
        matches = []
        
        # Create reverse lookup for exact matching
        nvplay_lookup = {hash_obj.full_hash: match_id for match_id, hash_obj in nvplay_hashes.items()}
        
        for decimal_id, decimal_hash in decimal_hashes.items():
            if decimal_hash.full_hash in nvplay_lookup:
                nvplay_id = nvplay_lookup[decimal_hash.full_hash]
                nvplay_hash = nvplay_hashes[nvplay_id]
                
                matches.append(MatchCandidate(
                    decimal_match_id=decimal_id,
                    nvplay_match_id=nvplay_id,
                    similarity_score=1.0,
                    hash_match_type='exact',
                    confidence=min(decimal_hash.confidence, nvplay_hash.confidence),
                    details={
                        'decimal_signature': decimal_hash.match_signature,
                        'nvplay_signature': nvplay_hash.match_signature,
                        'hash_match': 'full'
                    }
                ))
        
        return matches
    
    def _find_fuzzy_matches(self, decimal_hashes: Dict[str, CricketDNAHash],
                           nvplay_hashes: Dict[str, CricketDNAHash],
                           used_decimal: Set[str], used_nvplay: Set[str]) -> List[MatchCandidate]:
        """Find fuzzy hash matches using partial hash similarity"""
        matches = []
        
        for decimal_id, decimal_hash in decimal_hashes.items():
            if decimal_id in used_decimal:
                continue
                
            best_match = None
            best_score = 0
            
            for nvplay_id, nvplay_hash in nvplay_hashes.items():
                if nvplay_id in used_nvplay:
                    continue
                
                # Calculate similarity score based on hash components
                similarity = self._calculate_hash_similarity(decimal_hash, nvplay_hash)
                
                if similarity > best_score and similarity >= self.similarity_threshold:
                    best_score = similarity
                    best_match = (nvplay_id, nvplay_hash)
            
            if best_match:
                nvplay_id, nvplay_hash = best_match
                matches.append(MatchCandidate(
                    decimal_match_id=decimal_id,
                    nvplay_match_id=nvplay_id,
                    similarity_score=best_score,
                    hash_match_type='fuzzy',
                    confidence=min(decimal_hash.confidence, nvplay_hash.confidence) * best_score,
                    details={
                        'decimal_signature': decimal_hash.match_signature,
                        'nvplay_signature': nvplay_hash.match_signature,
                        'similarity_breakdown': self._get_similarity_breakdown(decimal_hash, nvplay_hash)
                    }
                ))
        
        return matches
    
    def _find_statistical_matches(self, decimal_hashes: Dict[str, CricketDNAHash],
                                 nvplay_hashes: Dict[str, CricketDNAHash],
                                 used_decimal: Set[str], used_nvplay: Set[str]) -> List[MatchCandidate]:
        """Find matches using statistical similarity of match signatures"""
        matches = []
        
        # Lower threshold for statistical matching
        stat_threshold = max(0.7, self.similarity_threshold - 0.15)
        
        for decimal_id, decimal_hash in decimal_hashes.items():
            if decimal_id in used_decimal:
                continue
                
            best_match = None
            best_score = 0
            
            for nvplay_id, nvplay_hash in nvplay_hashes.items():
                if nvplay_id in used_nvplay:
                    continue
                
                # Calculate similarity based on match signatures
                similarity = self._calculate_signature_similarity(
                    decimal_hash.match_signature, nvplay_hash.match_signature
                )
                
                if similarity > best_score and similarity >= stat_threshold:
                    best_score = similarity
                    best_match = (nvplay_id, nvplay_hash)
            
            if best_match:
                nvplay_id, nvplay_hash = best_match
                matches.append(MatchCandidate(
                    decimal_match_id=decimal_id,
                    nvplay_match_id=nvplay_id,
                    similarity_score=best_score,
                    hash_match_type='statistical',
                    confidence=min(decimal_hash.confidence, nvplay_hash.confidence) * best_score * 0.8,  # Lower confidence
                    details={
                        'decimal_signature': decimal_hash.match_signature,
                        'nvplay_signature': nvplay_hash.match_signature,
                        'match_type': 'statistical_similarity'
                    }
                ))
        
        return matches
    
    def _calculate_hash_similarity(self, hash1: CricketDNAHash, hash2: CricketDNAHash) -> float:
        """Calculate similarity between two cricket DNA hashes"""
        # Primary hash similarity (most important)
        primary_sim = 1.0 if hash1.primary_hash == hash2.primary_hash else 0.0
        
        # Secondary hash similarity
        secondary_sim = 1.0 if hash1.secondary_hash == hash2.secondary_hash else 0.0
        
        # Tertiary hash similarity
        tertiary_sim = 1.0 if hash1.tertiary_hash == hash2.tertiary_hash else 0.0
        
        # Signature similarity (fuzzy)
        signature_sim = self._calculate_signature_similarity(hash1.match_signature, hash2.match_signature)
        
        # Weighted combination
        similarity = (
            primary_sim * 0.5 +      # Core runs pattern is most important
            secondary_sim * 0.3 +    # Wicket pattern is important
            tertiary_sim * 0.1 +     # Additional details
            signature_sim * 0.1      # Overall match stats
        )
        
        return similarity
    
    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between match signatures"""
        if sig1 == sig2:
            return 1.0
        
        # Extract numbers from signatures (R123_W8_O20 format)
        def extract_numbers(sig):
            numbers = re.findall(r'\d+', sig)
            return [int(n) for n in numbers]
        
        nums1 = extract_numbers(sig1)
        nums2 = extract_numbers(sig2)
        
        if len(nums1) != len(nums2):
            return 0.0
        
        # Calculate similarity based on numerical closeness
        similarities = []
        for n1, n2 in zip(nums1, nums2):
            if n1 == 0 and n2 == 0:
                similarities.append(1.0)
            elif n1 == 0 or n2 == 0:
                similarities.append(0.0)
            else:
                # Percentage difference
                diff = abs(n1 - n2) / max(n1, n2)
                similarities.append(max(0.0, 1.0 - diff))
        
        return sum(similarities) / len(similarities)
    
    def _get_similarity_breakdown(self, hash1: CricketDNAHash, hash2: CricketDNAHash) -> Dict:
        """Get detailed breakdown of similarity components"""
        return {
            'primary_match': hash1.primary_hash == hash2.primary_hash,
            'secondary_match': hash1.secondary_hash == hash2.secondary_hash,
            'tertiary_match': hash1.tertiary_hash == hash2.tertiary_hash,
            'signature_similarity': self._calculate_signature_similarity(
                hash1.match_signature, hash2.match_signature
            )
        }

def align_matches_with_cricket_dna(decimal_csv_path: str, nvplay_csv_path: str, 
                                  output_path: str = "workflow_output/dna_aligned_matches.csv",
                                  similarity_threshold: float = 0.85) -> pd.DataFrame:
    """
    Main function to align matches using cricket DNA hash approach
    
    Args:
        decimal_csv_path: Path to decimal cricket data
        nvplay_csv_path: Path to nvplay cricket data  
        output_path: Where to save aligned matches
        similarity_threshold: Minimum similarity for matching (0-1)
        
    Returns:
        DataFrame with aligned matches
    """
    logger.info("Starting Cricket DNA match alignment...")
    
    # Load datasets
    logger.info("Loading datasets...")
    decimal_data = pd.read_csv(decimal_csv_path)
    nvplay_data = pd.read_csv(nvplay_csv_path)
    
    logger.info(f"Loaded {len(decimal_data)} decimal records and {len(nvplay_data)} nvplay records")
    
    # Create matcher and find matches
    matcher = CricketDNAMatcher(similarity_threshold=similarity_threshold)
    matches = matcher.find_matches(decimal_data, nvplay_data)
    
    logger.info(f"Found {len(matches)} total matches")
    
    # Create aligned dataset
    aligned_matches = []
    
    for match in matches:
        # Get match data from both datasets
        decimal_match_data = decimal_data[
            decimal_data.get('match_id', decimal_data.get('synthetic_match_id', pd.Series())) == match.decimal_match_id
        ]
        nvplay_match_data = nvplay_data[
            nvplay_data.get('Match', pd.Series()) == match.nvplay_match_id
        ]
        
        # Align ball-by-ball data (simplified for now)
        min_balls = min(len(decimal_match_data), len(nvplay_match_data))
        
        for i in range(min_balls):
            decimal_ball = decimal_match_data.iloc[i]
            nvplay_ball = nvplay_match_data.iloc[i]
            
            # Create aligned record
            aligned_record = {
                'match_id': f"DNA_{match.decimal_match_id}_{match.nvplay_match_id}",
                'similarity_score': match.similarity_score,
                'match_type': match.hash_match_type,
                'confidence': match.confidence,
                
                # Core cricket data (prioritize nvplay for detailed data)
                'batter': nvplay_ball.get('Batter', decimal_ball.get('batsman', '')),
                'bowler': nvplay_ball.get('Bowler', decimal_ball.get('bowler', '')),
                'runs_scored': nvplay_ball.get('Runs', decimal_ball.get('batsmanruns', 0)),
                'over': nvplay_ball.get('Over', decimal_ball.get('over', 0)),
                'ball': nvplay_ball.get('Ball', decimal_ball.get('delivery', 0)),
                'innings': nvplay_ball.get('Innings', decimal_ball.get('innings', 0)),
                'is_wicket': 1 if (nvplay_ball.get('Wicket') or decimal_ball.get('wicket')) else 0,
                
                # Additional nvplay data
                'ball_outcome': nvplay_ball.get('Ball Outcome', ''),
                'line': nvplay_ball.get('Line', ''),
                'length': nvplay_ball.get('Length', ''),
                'shot': nvplay_ball.get('Shot', ''),
                'speed': nvplay_ball.get('Speed', ''),
                
                # Additional decimal data  
                'venue': decimal_ball.get('venue', ''),
                'team_batting': decimal_ball.get('battingteam', ''),
                'team_bowling': decimal_ball.get('bowler', ''),  # Approximate
                'win_prob': decimal_ball.get('win_prob', ''),
            }
            
            aligned_matches.append(aligned_record)
    
    # Convert to DataFrame and save
    aligned_df = pd.DataFrame(aligned_matches)
    
    if not aligned_df.empty:
        aligned_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(aligned_df)} aligned records to {output_path}")
        
        # Print summary statistics
        logger.info("Match alignment summary:")
        logger.info(f"  Total matches found: {len(matches)}")
        logger.info(f"  Exact matches: {len([m for m in matches if m.hash_match_type == 'exact'])}")
        logger.info(f"  Fuzzy matches: {len([m for m in matches if m.hash_match_type == 'fuzzy'])}")
        logger.info(f"  Statistical matches: {len([m for m in matches if m.hash_match_type == 'statistical'])}")
        logger.info(f"  Average confidence: {np.mean([m.confidence for m in matches]):.3f}")
        logger.info(f"  Average similarity: {np.mean([m.similarity_score for m in matches]):.3f}")
    else:
        logger.warning("No matches found - check data format and thresholds")
        # Add diagnostic information
        if len(decimal_hashes) > 0 and len(nvplay_hashes) > 0:
            sample_decimal = list(decimal_hashes.values())[0]
            sample_nvplay = list(nvplay_hashes.values())[0]
            logger.info(f"Sample decimal hash confidence: {sample_decimal.confidence:.3f}")
            logger.info(f"Sample nvplay hash confidence: {sample_nvplay.confidence:.3f}")
            logger.info(f"Sample decimal primary hash: {sample_decimal.primary_hash[:50]}...")
            logger.info(f"Sample nvplay primary hash: {sample_nvplay.primary_hash[:50]}...")
    
    return aligned_df

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    aligned_df = align_matches_with_cricket_dna(
        decimal_csv_path="data/decimal_sample.csv",
        nvplay_csv_path="data/nvplay_sample.csv",
        similarity_threshold=0.8
    )
    
    print(f"Successfully aligned {len(aligned_df)} records using Cricket DNA matching!")