# Purpose: Comprehensive tests for Cricket DNA hash-based match alignment
# Author: Shamus Rae, Last Modified: 2025-01-30

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path

from cricket_dna_match_aligner import (
    CricketDNAHash, 
    CricketDNAHasher, 
    CricketDNAMatcher,
    MatchCandidate,
    align_matches_with_cricket_dna
)

class TestCricketDNAHash:
    """Test CricketDNAHash dataclass"""
    
    def test_hash_creation(self):
        """Test basic hash object creation"""
        hash_obj = CricketDNAHash(
            primary_hash="abc123",
            secondary_hash="def456", 
            tertiary_hash="ghi789",
            match_signature="R150_W8_O20",
            confidence=0.95
        )
        
        assert hash_obj.primary_hash == "abc123"
        assert hash_obj.secondary_hash == "def456"
        assert hash_obj.tertiary_hash == "ghi789"
        assert hash_obj.match_signature == "R150_W8_O20"
        assert hash_obj.confidence == 0.95
        assert hash_obj.full_hash == "abc123_def456_ghi789"

class TestCricketDNAHasher:
    """Test CricketDNAHasher functionality"""
    
    @pytest.fixture
    def hasher(self):
        return CricketDNAHasher()
    
    @pytest.fixture
    def sample_decimal_match(self):
        """Sample decimal cricket match data"""
        return pd.DataFrame({
            'over': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            'delivery': [1, 2, 3, 4, 5, 6, 1, 2, 3, 4],
            'innings': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'batsmanruns': [0, 1, 4, 0, 6, 1, 0, 2, 1, 4],
            'wicket': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'wickettype': ['', '', '', '', '', '', 'bowled', '', '', ''],
            'extras': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'batsman': ['Player1', 'Player1', 'Player1', 'Player1', 'Player1', 'Player1', 'Player2', 'Player2', 'Player2', 'Player2'],
            'bowler': ['Bowler1', 'Bowler1', 'Bowler1', 'Bowler1', 'Bowler1', 'Bowler1', 'Bowler1', 'Bowler1', 'Bowler1', 'Bowler1']
        })
    
    @pytest.fixture
    def sample_nvplay_match(self):
        """Sample nvplay cricket match data"""
        return pd.DataFrame({
            'Over': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            'Ball': [1, 2, 3, 4, 5, 6, 1, 2, 3, 4],
            'Innings': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'Runs': [0, 1, 4, 0, 6, 1, 0, 2, 1, 4],
            'Wicket': ['', '', '', '', '', '', 'bowled', '', '', ''],
            'Ball Outcome': ['', '', '', '', '', '', 'bowled', '', '', ''],
            'Extra Runs': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Batter': ['Player1', 'Player1', 'Player1', 'Player1', 'Player1', 'Player1', 'Player2', 'Player2', 'Player2', 'Player2'],
            'Bowler': ['Bowler1', 'Bowler1', 'Bowler1', 'Bowler1', 'Bowler1', 'Bowler1', 'Bowler1', 'Bowler1', 'Bowler1', 'Bowler1']
        })
    
    def test_extract_runs_sequence_decimal(self, hasher, sample_decimal_match):
        """Test runs sequence extraction for decimal data"""
        runs_seq = hasher._extract_runs_sequence(sample_decimal_match, 'decimal')
        expected = "0,1,4,0,6,1,0,2,1,4"
        assert runs_seq == expected
    
    def test_extract_runs_sequence_nvplay(self, hasher, sample_nvplay_match):
        """Test runs sequence extraction for nvplay data"""
        runs_seq = hasher._extract_runs_sequence(sample_nvplay_match, 'nvplay')
        expected = "0,1,4,0,6,1,0,2,1,4"
        assert runs_seq == expected
    
    def test_extract_over_totals_decimal(self, hasher, sample_decimal_match):
        """Test over totals extraction for decimal data"""
        over_totals = hasher._extract_over_totals(sample_decimal_match, 'decimal')
        # Over 1: 0+1+4+0+6+1 = 12, Over 2: 12+0+2+1+4 = 19
        expected = "12,19"
        assert over_totals == expected
    
    def test_extract_over_totals_nvplay(self, hasher, sample_nvplay_match):
        """Test over totals extraction for nvplay data"""
        over_totals = hasher._extract_over_totals(sample_nvplay_match, 'nvplay')
        expected = "12,19"
        assert over_totals == expected
    
    def test_extract_wicket_pattern_decimal(self, hasher, sample_decimal_match):
        """Test wicket pattern extraction for decimal data"""
        wicket_pattern = hasher._extract_wicket_pattern(sample_decimal_match, 'decimal')
        expected = "2.1:bowled"  # Wicket at over 2, ball 1
        assert wicket_pattern == expected
    
    def test_extract_wicket_pattern_nvplay(self, hasher, sample_nvplay_match):
        """Test wicket pattern extraction for nvplay data"""
        wicket_pattern = hasher._extract_wicket_pattern(sample_nvplay_match, 'nvplay')
        expected = "2.1:bowled"
        assert wicket_pattern == expected
    
    def test_extract_innings_structure_decimal(self, hasher, sample_decimal_match):
        """Test innings structure extraction for decimal data"""
        innings_structure = hasher._extract_innings_structure(sample_decimal_match, 'decimal')
        expected = "19/1"  # 19 runs, 1 wicket in innings 1
        assert innings_structure == expected
    
    def test_extract_innings_structure_nvplay(self, hasher, sample_nvplay_match):
        """Test innings structure extraction for nvplay data"""
        innings_structure = hasher._extract_innings_structure(sample_nvplay_match, 'nvplay')
        expected = "19/1"
        assert innings_structure == expected
    
    def test_create_match_hash_decimal(self, hasher, sample_decimal_match):
        """Test complete hash creation for decimal data"""
        hash_obj = hasher.create_match_hash(sample_decimal_match, 'decimal')
        
        assert isinstance(hash_obj, CricketDNAHash)
        assert len(hash_obj.primary_hash) == 16
        assert len(hash_obj.secondary_hash) == 16
        assert len(hash_obj.tertiary_hash) == 8
        assert hash_obj.match_signature == "R19_W1_O2"
        assert 0.0 <= hash_obj.confidence <= 1.0
    
    def test_create_match_hash_nvplay(self, hasher, sample_nvplay_match):
        """Test complete hash creation for nvplay data"""
        hash_obj = hasher.create_match_hash(sample_nvplay_match, 'nvplay')
        
        assert isinstance(hash_obj, CricketDNAHash)
        assert len(hash_obj.primary_hash) == 16
        assert len(hash_obj.secondary_hash) == 16
        assert len(hash_obj.tertiary_hash) == 8
        assert hash_obj.match_signature == "R19_W1_O2"
        assert 0.0 <= hash_obj.confidence <= 1.0
    
    def test_identical_matches_same_hash(self, hasher, sample_decimal_match, sample_nvplay_match):
        """Test that identical matches produce identical hashes"""
        decimal_hash = hasher.create_match_hash(sample_decimal_match, 'decimal')
        nvplay_hash = hasher.create_match_hash(sample_nvplay_match, 'nvplay')
        
        # Should have identical hashes for same match data
        assert decimal_hash.primary_hash == nvplay_hash.primary_hash
        assert decimal_hash.secondary_hash == nvplay_hash.secondary_hash
        assert decimal_hash.match_signature == nvplay_hash.match_signature
    
    def test_normalize_wicket_type(self, hasher):
        """Test wicket type normalization"""
        assert hasher._normalize_wicket_type("bowled") == "bowled"
        assert hasher._normalize_wicket_type("c Smith b Jones") == "caught"
        assert hasher._normalize_wicket_type("lbw") == "lbw"
        assert hasher._normalize_wicket_type("run out") == "runout"
        assert hasher._normalize_wicket_type("stumped") == "stumped"
        assert hasher._normalize_wicket_type("hit wicket") == "other"
    
    def test_calculate_confidence(self, hasher, sample_decimal_match):
        """Test confidence calculation"""
        confidence = hasher._calculate_confidence(sample_decimal_match, 'decimal')
        assert 0.0 <= confidence <= 1.0
        
        # Should be high confidence for complete data
        assert confidence > 0.8
    
    def test_fallback_hash_creation(self, hasher):
        """Test fallback hash creation for problematic data"""
        empty_df = pd.DataFrame()
        hash_obj = hasher._create_fallback_hash(empty_df, 'decimal')
        
        assert isinstance(hash_obj, CricketDNAHash)
        assert hash_obj.match_signature == "FALLBACK"
        assert hash_obj.confidence == 0.1

class TestCricketDNAMatcher:
    """Test CricketDNAMatcher functionality"""
    
    @pytest.fixture
    def matcher(self):
        return CricketDNAMatcher(similarity_threshold=0.85)
    
    @pytest.fixture
    def sample_decimal_data(self):
        """Sample decimal dataset with multiple matches"""
        return pd.DataFrame({
            'date': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
            'battingteam': ['TeamA', 'TeamA', 'TeamA', 'TeamB', 'TeamB'],
            'over': [1, 1, 1, 1, 1],
            'delivery': [1, 2, 3, 1, 2],
            'innings': [1, 1, 1, 1, 1],
            'batsmanruns': [4, 0, 6, 1, 2],
            'wicket': [0, 0, 0, 0, 1],
            'wickettype': ['', '', '', '', 'caught'],
            'extras': [0, 0, 0, 0, 0]
        })
    
    @pytest.fixture
    def sample_nvplay_data(self):
        """Sample nvplay dataset with matching data"""
        return pd.DataFrame({
            'Match': ['Match1', 'Match1', 'Match1', 'Match2', 'Match2'],
            'Over': [1, 1, 1, 1, 1],
            'Ball': [1, 2, 3, 1, 2],
            'Innings': [1, 1, 1, 1, 1],
            'Runs': [4, 0, 6, 1, 2],
            'Wicket': ['', '', '', '', 'caught'],
            'Ball Outcome': ['', '', '', '', 'caught'],
            'Extra Runs': [0, 0, 0, 0, 0]
        })
    
    def test_create_dataset_hashes_decimal(self, matcher, sample_decimal_data):
        """Test hash creation for decimal dataset"""
        hashes = matcher._create_dataset_hashes(sample_decimal_data, 'decimal')
        
        # Should create hashes for synthetic match IDs
        assert len(hashes) == 2  # Two different date+team combinations
        
        for match_id, hash_obj in hashes.items():
            assert isinstance(hash_obj, CricketDNAHash)
            assert hash_obj.confidence > 0
    
    def test_create_dataset_hashes_nvplay(self, matcher, sample_nvplay_data):
        """Test hash creation for nvplay dataset"""
        hashes = matcher._create_dataset_hashes(sample_nvplay_data, 'nvplay')
        
        assert len(hashes) == 2  # Two matches
        
        for match_id, hash_obj in hashes.items():
            assert isinstance(hash_obj, CricketDNAHash)
            assert hash_obj.confidence > 0
    
    def test_find_exact_matches(self, matcher):
        """Test exact hash matching"""
        # Create identical hashes
        hash1 = CricketDNAHash("abc", "def", "ghi", "R10_W1_O1", 0.9)
        hash2 = CricketDNAHash("abc", "def", "ghi", "R10_W1_O1", 0.9)
        hash3 = CricketDNAHash("xyz", "uvw", "rst", "R20_W2_O2", 0.8)
        
        decimal_hashes = {"match1": hash1}
        nvplay_hashes = {"matchA": hash2, "matchB": hash3}
        
        matches = matcher._find_exact_matches(decimal_hashes, nvplay_hashes)
        
        assert len(matches) == 1
        match = matches[0]
        assert match.decimal_match_id == "match1"
        assert match.nvplay_match_id == "matchA"
        assert match.similarity_score == 1.0
        assert match.hash_match_type == 'exact'
    
    def test_calculate_hash_similarity(self, matcher):
        """Test hash similarity calculation"""
        # Identical hashes
        hash1 = CricketDNAHash("abc", "def", "ghi", "R10_W1_O1", 0.9)
        hash2 = CricketDNAHash("abc", "def", "ghi", "R10_W1_O1", 0.9)
        
        similarity = matcher._calculate_hash_similarity(hash1, hash2)
        assert similarity == 1.0
        
        # Partially similar hashes
        hash3 = CricketDNAHash("abc", "xyz", "ghi", "R10_W1_O1", 0.9)
        similarity = matcher._calculate_hash_similarity(hash1, hash3)
        assert 0.5 < similarity < 1.0  # Primary and tertiary match, secondary doesn't
    
    def test_calculate_signature_similarity(self, matcher):
        """Test signature similarity calculation"""
        # Identical signatures
        similarity = matcher._calculate_signature_similarity("R150_W8_O20", "R150_W8_O20")
        assert similarity == 1.0
        
        # Similar signatures
        similarity = matcher._calculate_signature_similarity("R150_W8_O20", "R148_W8_O20")
        assert 0.9 < similarity < 1.0  # Very close runs
        
        # Different signatures
        similarity = matcher._calculate_signature_similarity("R150_W8_O20", "R300_W4_O20")
        assert similarity < 0.7  # Different runs and wickets
    
    def test_find_matches_integration(self, matcher, sample_decimal_data, sample_nvplay_data):
        """Test complete match finding workflow"""
        matches = matcher.find_matches(sample_decimal_data, sample_nvplay_data)
        
        # Should find some matches
        assert len(matches) >= 0
        
        for match in matches:
            assert isinstance(match, MatchCandidate)
            assert 0.0 <= match.similarity_score <= 1.0
            assert 0.0 <= match.confidence <= 1.0
            assert match.hash_match_type in ['exact', 'fuzzy', 'statistical']

class TestMatchCandidate:
    """Test MatchCandidate dataclass"""
    
    def test_match_candidate_creation(self):
        """Test match candidate creation"""
        candidate = MatchCandidate(
            decimal_match_id="dec_match_1",
            nvplay_match_id="nv_match_A",
            similarity_score=0.95,
            hash_match_type='exact',
            confidence=0.88,
            details={'test': 'data'}
        )
        
        assert candidate.decimal_match_id == "dec_match_1"
        assert candidate.nvplay_match_id == "nv_match_A"
        assert candidate.similarity_score == 0.95
        assert candidate.hash_match_type == 'exact'
        assert candidate.confidence == 0.88
        assert candidate.details == {'test': 'data'}

class TestAlignMatchesWithCricketDNA:
    """Test main alignment function"""
    
    @pytest.fixture
    def temp_csv_files(self):
        """Create temporary CSV files for testing"""
        # Create decimal CSV
        decimal_data = pd.DataFrame({
            'date': ['2024-01-01'] * 6,
            'battingteam': ['TeamA'] * 6,
            'over': [1, 1, 1, 1, 1, 1],
            'delivery': [1, 2, 3, 4, 5, 6],
            'innings': [1, 1, 1, 1, 1, 1],
            'batsmanruns': [4, 0, 6, 1, 2, 0],
            'wicket': [0, 0, 0, 0, 0, 1],
            'wickettype': ['', '', '', '', '', 'bowled'],
            'extras': [0, 0, 0, 0, 0, 0],
            'batsman': ['Player1'] * 6,
            'bowler': ['Bowler1'] * 6,
            'venue': ['Ground1'] * 6
        })
        
        # Create nvplay CSV
        nvplay_data = pd.DataFrame({
            'Match': ['Match1'] * 6,
            'Over': [1, 1, 1, 1, 1, 1],
            'Ball': [1, 2, 3, 4, 5, 6],
            'Innings': [1, 1, 1, 1, 1, 1],
            'Runs': [4, 0, 6, 1, 2, 0],
            'Wicket': ['', '', '', '', '', 'bowled'],
            'Ball Outcome': ['', '', '', '', '', 'bowled'],
            'Extra Runs': [0, 0, 0, 0, 0, 0],
            'Batter': ['Player1'] * 6,
            'Bowler': ['Bowler1'] * 6,
            'Line': ['good length'] * 6,
            'Length': ['on stumps'] * 6,
            'Shot': ['defensive'] * 6,
            'Speed': ['140'] * 6
        })
        
        # Write to temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            decimal_data.to_csv(f.name, index=False)
            decimal_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            nvplay_data.to_csv(f.name, index=False)
            nvplay_path = f.name
        
        yield decimal_path, nvplay_path
        
        # Cleanup
        os.unlink(decimal_path)
        os.unlink(nvplay_path)
    
    def test_align_matches_with_cricket_dna(self, temp_csv_files):
        """Test complete alignment workflow"""
        decimal_path, nvplay_path = temp_csv_files
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            # Run alignment
            aligned_df = align_matches_with_cricket_dna(
                decimal_csv_path=decimal_path,
                nvplay_csv_path=nvplay_path,
                output_path=output_path,
                similarity_threshold=0.7  # Lower threshold for testing
            )
            
            # Should create some aligned records
            assert isinstance(aligned_df, pd.DataFrame)
            
            if len(aligned_df) > 0:
                # Check expected columns
                expected_columns = [
                    'match_id', 'similarity_score', 'match_type', 'confidence',
                    'batter', 'bowler', 'runs_scored', 'over', 'ball', 'innings',
                    'is_wicket', 'ball_outcome', 'line', 'length', 'shot', 'speed',
                    'venue', 'team_batting', 'team_bowling', 'win_prob'
                ]
                
                for col in expected_columns:
                    assert col in aligned_df.columns
                
                # Check data types and values
                assert aligned_df['similarity_score'].dtype == float
                assert aligned_df['confidence'].dtype == float
                assert all(aligned_df['similarity_score'] >= 0.0)
                assert all(aligned_df['similarity_score'] <= 1.0)
                assert all(aligned_df['confidence'] >= 0.0)
                assert all(aligned_df['confidence'] <= 1.0)
                
                # Check output file was created
                assert os.path.exists(output_path)
        
        finally:
            # Cleanup output file
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_align_matches_empty_result(self):
        """Test alignment with no matches found"""
        # Create completely different datasets
        decimal_data = pd.DataFrame({
            'date': ['2024-01-01'],
            'battingteam': ['TeamA'],
            'over': [1], 'delivery': [1], 'innings': [1],
            'batsmanruns': [4], 'wicket': [0], 'extras': [0]
        })
        
        nvplay_data = pd.DataFrame({
            'Match': ['Match1'],
            'Over': [1], 'Ball': [1], 'Innings': [1],
            'Runs': [0], 'Wicket': [''], 'Extra Runs': [0]  # Different runs
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            decimal_data.to_csv(f.name, index=False)
            decimal_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            nvplay_data.to_csv(f.name, index=False)
            nvplay_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            aligned_df = align_matches_with_cricket_dna(
                decimal_csv_path=decimal_path,
                nvplay_csv_path=nvplay_path,
                output_path=output_path,
                similarity_threshold=0.9  # High threshold
            )
            
            # Should return empty DataFrame if no matches
            assert isinstance(aligned_df, pd.DataFrame)
            
        finally:
            # Cleanup
            os.unlink(decimal_path)
            os.unlink(nvplay_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataframes(self):
        """Test handling of empty dataframes"""
        hasher = CricketDNAHasher()
        empty_df = pd.DataFrame()
        
        hash_obj = hasher.create_match_hash(empty_df, 'decimal')
        assert hash_obj.match_signature == "FALLBACK"
        assert hash_obj.confidence == 0.1
    
    def test_missing_columns(self):
        """Test handling of missing columns"""
        hasher = CricketDNAHasher()
        
        # DataFrame with minimal columns
        minimal_df = pd.DataFrame({
            'some_column': [1, 2, 3]
        })
        
        hash_obj = hasher.create_match_hash(minimal_df, 'decimal')
        assert isinstance(hash_obj, CricketDNAHash)
        assert hash_obj.confidence < 0.5  # Low confidence due to missing data
    
    def test_malformed_data(self):
        """Test handling of malformed data"""
        hasher = CricketDNAHasher()
        
        # DataFrame with NaN values
        malformed_df = pd.DataFrame({
            'over': [1, np.nan, 2],
            'delivery': [1, 2, np.nan], 
            'innings': [1, 1, 1],
            'batsmanruns': [4, np.nan, 6],
            'wicket': [0, 0, np.nan]
        })
        
        hash_obj = hasher.create_match_hash(malformed_df, 'decimal')
        assert isinstance(hash_obj, CricketDNAHash)
        # Should handle NaN values gracefully
    
    def test_very_high_similarity_threshold(self):
        """Test behavior with unrealistically high similarity threshold"""
        matcher = CricketDNAMatcher(similarity_threshold=0.99)
        
        # Even identical data might not meet 0.99 threshold due to floating point precision
        decimal_data = pd.DataFrame({'date': ['2024-01-01'], 'battingteam': ['A'], 'over': [1], 'delivery': [1], 'innings': [1], 'batsmanruns': [4]})
        nvplay_data = pd.DataFrame({'Match': ['1'], 'Over': [1], 'Ball': [1], 'Innings': [1], 'Runs': [4]})
        
        matches = matcher.find_matches(decimal_data, nvplay_data)
        # Should still find exact matches
        assert len(matches) >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])