#!/usr/bin/env python3
"""
Unified Match Aligner - Consolidates 4 duplicate aligners into one configurable system
Supports fingerprint, DNA hash, LLM-enhanced, and hybrid strategies

Author: WicketWise Team, Last Modified: 2025-01-21
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Protocol
from dataclasses import dataclass
from enum import Enum
import logging
import hashlib
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
import Levenshtein
import json
import time

from unified_configuration import get_config

logger = logging.getLogger(__name__)
config = get_config()

class AlignmentStrategy(Enum):
    """Available alignment strategies"""
    FINGERPRINT = "fingerprint"
    DNA_HASH = "dna_hash"
    LLM_ENHANCED = "llm_enhanced"
    HYBRID = "hybrid"

@dataclass
class MatchCandidate:
    """Represents a potential match between datasets"""
    dataset1_id: str
    dataset2_id: str
    similarity_score: float
    match_type: str
    confidence: str
    metadata: Dict[str, Any] = None

@dataclass
class AlignmentConfig:
    """Configuration for match alignment"""
    strategy: AlignmentStrategy = AlignmentStrategy.HYBRID
    similarity_threshold: float = 0.8
    fingerprint_length: int = 50
    use_llm_fallback: bool = True
    team_aliases: Dict[str, List[str]] = None
    venue_aliases: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.team_aliases is None:
            self.team_aliases = config.matching.team_aliases
        if self.venue_aliases is None:
            self.venue_aliases = {}

class AlignmentStrategy(Protocol):
    """Protocol for alignment strategies"""
    
    def find_matches(
        self,
        dataset1: pd.DataFrame,
        dataset2: pd.DataFrame,
        config: AlignmentConfig
    ) -> List[MatchCandidate]:
        """Find matches between two datasets"""
        ...

class FingerprintStrategy:
    """Ball-by-ball sequence fingerprinting strategy"""
    
    def find_matches(
        self,
        dataset1: pd.DataFrame,
        dataset2: pd.DataFrame,
        config: AlignmentConfig
    ) -> List[MatchCandidate]:
        """Find matches using ball-by-ball fingerprints"""
        
        logger.info("🔍 Using fingerprint strategy for alignment")
        
        # Extract fingerprints from both datasets
        fp1 = self._extract_fingerprints(dataset1, config.fingerprint_length)
        fp2 = self._extract_fingerprints(dataset2, config.fingerprint_length)
        
        matches = []
        
        for match1_id, fingerprint1 in fp1.items():
            best_match = None
            best_score = 0.0
            
            for match2_id, fingerprint2 in fp2.items():
                similarity = self._calculate_fingerprint_similarity(fingerprint1, fingerprint2)
                
                if similarity > best_score and similarity >= config.similarity_threshold:
                    best_score = similarity
                    best_match = match2_id
            
            if best_match:
                matches.append(MatchCandidate(
                    dataset1_id=match1_id,
                    dataset2_id=best_match,
                    similarity_score=best_score,
                    match_type="fingerprint",
                    confidence="high" if best_score > 0.9 else "medium",
                    metadata={"fingerprint_length": config.fingerprint_length}
                ))
        
        logger.info(f"✅ Fingerprint strategy found {len(matches)} matches")
        return matches
    
    def _extract_fingerprints(self, df: pd.DataFrame, length: int) -> Dict[str, List[Tuple]]:
        """Extract ball-by-ball fingerprints from dataset"""
        
        fingerprints = {}
        
        # Group by match
        for match_id, match_data in df.groupby('match_id'):
            if pd.isna(match_id):
                continue
            
            # Sort by ball sequence
            match_data = match_data.sort_values(['over', 'ball'])
            
            # Create fingerprint tuples
            fingerprint = []
            for _, ball in match_data.head(length).iterrows():
                fingerprint.append((
                    ball.get('over', 0),
                    ball.get('ball', 0),
                    ball.get('batter', ''),
                    ball.get('bowler', ''),
                    ball.get('runs_scored', 0)
                ))
            
            if len(fingerprint) >= 10:  # Minimum fingerprint length
                fingerprints[str(match_id)] = fingerprint
        
        return fingerprints
    
    def _calculate_fingerprint_similarity(self, fp1: List[Tuple], fp2: List[Tuple]) -> float:
        """Calculate similarity between two fingerprints"""
        
        if not fp1 or not fp2:
            return 0.0
        
        # Compare sequence similarity
        min_length = min(len(fp1), len(fp2))
        matches = 0
        
        for i in range(min_length):
            if fp1[i] == fp2[i]:
                matches += 1
        
        return matches / min_length

class DNAHashStrategy:
    """Cricket DNA hash-based matching strategy"""
    
    def find_matches(
        self,
        dataset1: pd.DataFrame,
        dataset2: pd.DataFrame,
        config: AlignmentConfig
    ) -> List[MatchCandidate]:
        """Find matches using cricket DNA hashes"""
        
        logger.info("🧬 Using DNA hash strategy for alignment")
        
        # Create DNA hashes for both datasets
        hashes1 = self._create_dna_hashes(dataset1)
        hashes2 = self._create_dna_hashes(dataset2)
        
        matches = []
        
        # Find exact hash matches first
        exact_matches = self._find_exact_hash_matches(hashes1, hashes2)
        matches.extend(exact_matches)
        
        # Find fuzzy hash matches for remaining data
        used_ids1 = {m.dataset1_id for m in exact_matches}
        used_ids2 = {m.dataset2_id for m in exact_matches}
        
        fuzzy_matches = self._find_fuzzy_hash_matches(
            hashes1, hashes2, used_ids1, used_ids2, config.similarity_threshold
        )
        matches.extend(fuzzy_matches)
        
        logger.info(f"✅ DNA hash strategy found {len(matches)} matches ({len(exact_matches)} exact, {len(fuzzy_matches)} fuzzy)")
        return matches
    
    def _create_dna_hashes(self, df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        """Create DNA hashes for each match"""
        
        dna_hashes = {}
        
        for match_id, match_data in df.groupby('match_id'):
            if pd.isna(match_id):
                continue
            
            # Create primary hash (runs pattern)
            runs_sequence = match_data['runs_scored'].astype(str).str.cat()
            primary_hash = hashlib.md5(runs_sequence.encode()).hexdigest()[:16]
            
            # Create secondary hash (wickets pattern)
            wickets_sequence = match_data['is_wicket'].astype(str).str.cat()
            secondary_hash = hashlib.md5(wickets_sequence.encode()).hexdigest()[:16]
            
            # Create tertiary hash (overs pattern)
            overs_sequence = match_data['over'].astype(str).str.cat()
            tertiary_hash = hashlib.md5(overs_sequence.encode()).hexdigest()[:16]
            
            dna_hashes[str(match_id)] = {
                'primary': primary_hash,
                'secondary': secondary_hash,
                'tertiary': tertiary_hash,
                'full': f"{primary_hash}_{secondary_hash}_{tertiary_hash}"
            }
        
        return dna_hashes
    
    def _find_exact_hash_matches(
        self,
        hashes1: Dict[str, Dict[str, str]],
        hashes2: Dict[str, Dict[str, str]]
    ) -> List[MatchCandidate]:
        """Find exact hash matches"""
        
        matches = []
        
        for id1, hash1 in hashes1.items():
            for id2, hash2 in hashes2.items():
                if hash1['full'] == hash2['full']:
                    matches.append(MatchCandidate(
                        dataset1_id=id1,
                        dataset2_id=id2,
                        similarity_score=1.0,
                        match_type="dna_exact",
                        confidence="high",
                        metadata={"hash_match": "full"}
                    ))
                    break
        
        return matches
    
    def _find_fuzzy_hash_matches(
        self,
        hashes1: Dict[str, Dict[str, str]],
        hashes2: Dict[str, Dict[str, str]],
        used_ids1: set,
        used_ids2: set,
        threshold: float
    ) -> List[MatchCandidate]:
        """Find fuzzy hash matches"""
        
        matches = []
        
        for id1, hash1 in hashes1.items():
            if id1 in used_ids1:
                continue
            
            best_match = None
            best_score = 0.0
            
            for id2, hash2 in hashes2.items():
                if id2 in used_ids2:
                    continue
                
                # Calculate partial hash similarity
                similarity = 0.0
                if hash1['primary'] == hash2['primary']:
                    similarity += 0.5
                if hash1['secondary'] == hash2['secondary']:
                    similarity += 0.3
                if hash1['tertiary'] == hash2['tertiary']:
                    similarity += 0.2
                
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = id2
            
            if best_match:
                matches.append(MatchCandidate(
                    dataset1_id=id1,
                    dataset2_id=best_match,
                    similarity_score=best_score,
                    match_type="dna_fuzzy",
                    confidence="medium" if best_score > 0.7 else "low",
                    metadata={"hash_match": "partial"}
                ))
        
        return matches

class HybridStrategy:
    """Hybrid strategy combining multiple approaches"""
    
    def __init__(self):
        self.fingerprint_strategy = FingerprintStrategy()
        self.dna_strategy = DNAHashStrategy()
    
    def find_matches(
        self,
        dataset1: pd.DataFrame,
        dataset2: pd.DataFrame,
        config: AlignmentConfig
    ) -> List[MatchCandidate]:
        """Find matches using hybrid approach"""
        
        logger.info("🔀 Using hybrid strategy for alignment")
        
        # Try DNA hash first (fastest and most accurate)
        dna_matches = self.dna_strategy.find_matches(dataset1, dataset2, config)
        
        # Get unmatched records
        matched_ids1 = {m.dataset1_id for m in dna_matches}
        matched_ids2 = {m.dataset2_id for m in dna_matches}
        
        unmatched_df1 = dataset1[~dataset1['match_id'].astype(str).isin(matched_ids1)]
        unmatched_df2 = dataset2[~dataset2['match_id'].astype(str).isin(matched_ids2)]
        
        # Try fingerprint matching on remaining data
        fingerprint_matches = []
        if not unmatched_df1.empty and not unmatched_df2.empty:
            # Lower threshold for fingerprint matching
            fingerprint_config = AlignmentConfig(
                strategy=AlignmentStrategy.FINGERPRINT,
                similarity_threshold=max(0.6, config.similarity_threshold - 0.2),
                fingerprint_length=config.fingerprint_length
            )
            fingerprint_matches = self.fingerprint_strategy.find_matches(
                unmatched_df1, unmatched_df2, fingerprint_config
            )
        
        # Combine results
        all_matches = dna_matches + fingerprint_matches
        
        # Sort by confidence and similarity
        all_matches.sort(key=lambda m: (m.confidence == "high", m.similarity_score), reverse=True)
        
        logger.info(f"✅ Hybrid strategy found {len(all_matches)} matches ({len(dna_matches)} DNA, {len(fingerprint_matches)} fingerprint)")
        return all_matches

class UnifiedMatchAligner:
    """Unified match aligner supporting multiple strategies"""
    
    def __init__(self, config: AlignmentConfig = None):
        self.config = config or AlignmentConfig()
        
        # Initialize strategies
        self.strategies = {
            AlignmentStrategy.FINGERPRINT: FingerprintStrategy(),
            AlignmentStrategy.DNA_HASH: DNAHashStrategy(),
            AlignmentStrategy.HYBRID: HybridStrategy()
        }
        
        logger.info(f"🔧 Unified Match Aligner initialized with {self.config.strategy.value} strategy")
    
    def align_datasets(
        self,
        dataset1_path: str,
        dataset2_path: str,
        output_path: Optional[str] = None
    ) -> List[MatchCandidate]:
        """Align two cricket datasets"""
        
        logger.info(f"🏏 Starting dataset alignment: {dataset1_path} <-> {dataset2_path}")
        start_time = time.time()
        
        # Load datasets
        df1 = self._load_dataset(dataset1_path, "dataset1")
        df2 = self._load_dataset(dataset2_path, "dataset2")
        
        # Get strategy
        strategy = self.strategies.get(self.config.strategy)
        if not strategy:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
        
        # Find matches
        matches = strategy.find_matches(df1, df2, self.config)
        
        # Post-process matches
        matches = self._post_process_matches(matches)
        
        # Save results if requested
        if output_path:
            self._save_matches(matches, output_path)
        
        processing_time = time.time() - start_time
        
        # Log summary
        self._log_alignment_summary(matches, processing_time)
        
        return matches
    
    def _load_dataset(self, path: str, dataset_name: str) -> pd.DataFrame:
        """Load and preprocess dataset"""
        
        logger.info(f"📊 Loading {dataset_name} from {path}")
        
        df = pd.read_csv(path)
        
        # Basic preprocessing
        df = df.dropna(subset=['match_id'])
        
        # Standardize column names
        column_mapping = {
            'Match': 'match_id',
            'Over': 'over',
            'Ball': 'ball',
            'Batsman': 'batter',
            'Bowler': 'bowler',
            'Runs': 'runs_scored',
            'Wicket': 'is_wicket'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Fill missing values
        df['runs_scored'] = df['runs_scored'].fillna(0)
        df['is_wicket'] = df['is_wicket'].fillna(False)
        
        logger.info(f"✅ Loaded {len(df):,} records from {dataset_name}")
        return df
    
    def _post_process_matches(self, matches: List[MatchCandidate]) -> List[MatchCandidate]:
        """Post-process matches for quality improvement"""
        
        # Remove duplicate matches
        seen_pairs = set()
        unique_matches = []
        
        for match in matches:
            pair = (match.dataset1_id, match.dataset2_id)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_matches.append(match)
        
        # Sort by confidence and similarity
        unique_matches.sort(
            key=lambda m: (
                m.confidence == "high",
                m.confidence == "medium", 
                m.similarity_score
            ),
            reverse=True
        )
        
        logger.info(f"📋 Post-processing: {len(matches)} -> {len(unique_matches)} unique matches")
        return unique_matches
    
    def _save_matches(self, matches: List[MatchCandidate], output_path: str):
        """Save matches to file"""
        
        # Convert to DataFrame for easy saving
        match_data = []
        for match in matches:
            match_data.append({
                'dataset1_id': match.dataset1_id,
                'dataset2_id': match.dataset2_id,
                'similarity_score': match.similarity_score,
                'match_type': match.match_type,
                'confidence': match.confidence,
                'metadata': json.dumps(match.metadata) if match.metadata else None
            })
        
        df = pd.DataFrame(match_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"💾 Saved {len(matches)} matches to {output_path}")
    
    def _log_alignment_summary(self, matches: List[MatchCandidate], processing_time: float):
        """Log alignment summary"""
        
        if not matches:
            logger.warning("⚠️ No matches found!")
            return
        
        # Calculate statistics
        confidence_counts = {}
        match_type_counts = {}
        
        for match in matches:
            confidence_counts[match.confidence] = confidence_counts.get(match.confidence, 0) + 1
            match_type_counts[match.match_type] = match_type_counts.get(match.match_type, 0) + 1
        
        avg_similarity = sum(m.similarity_score for m in matches) / len(matches)
        
        logger.info("🎯 Alignment Summary:")
        logger.info(f"   Total matches: {len(matches)}")
        logger.info(f"   Average similarity: {avg_similarity:.3f}")
        logger.info(f"   Processing time: {processing_time:.2f}s")
        logger.info(f"   Confidence distribution: {confidence_counts}")
        logger.info(f"   Match type distribution: {match_type_counts}")

# Factory function for easy instantiation
def create_aligner(
    strategy: str = "hybrid",
    similarity_threshold: float = 0.8,
    **kwargs
) -> UnifiedMatchAligner:
    """Create a unified match aligner with specified configuration"""
    
    strategy_enum = AlignmentStrategy(strategy.lower())
    
    config = AlignmentConfig(
        strategy=strategy_enum,
        similarity_threshold=similarity_threshold,
        **kwargs
    )
    
    return UnifiedMatchAligner(config)

# Example usage and testing
if __name__ == "__main__":
    # Test the unified aligner
    
    # Create aligner with hybrid strategy
    aligner = create_aligner(
        strategy="hybrid",
        similarity_threshold=0.8,
        fingerprint_length=50
    )
    
    # Mock data paths for testing
    dataset1_path = "/path/to/nvplay_data.csv"
    dataset2_path = "/path/to/decimal_data.csv"
    
    if Path(dataset1_path).exists() and Path(dataset2_path).exists():
        # Run alignment
        matches = aligner.align_datasets(
            dataset1_path,
            dataset2_path,
            output_path="aligned_matches.csv"
        )
        
        print(f"🎉 Found {len(matches)} matches!")
        
        # Show top matches
        for i, match in enumerate(matches[:5]):
            print(f"  {i+1}. {match.dataset1_id} <-> {match.dataset2_id} "
                  f"(similarity: {match.similarity_score:.3f}, confidence: {match.confidence})")
    
    else:
        print("⚠️ Test data not found")
        print("🔧 Unified Match Aligner ready for real data!")
        print(f"📊 Available strategies: {[s.value for s in AlignmentStrategy]}")
        print(f"🎯 Current strategy: {aligner.config.strategy.value}")
        print(f"🔍 Similarity threshold: {aligner.config.similarity_threshold}")
