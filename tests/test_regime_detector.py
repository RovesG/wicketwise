# Purpose: Unit tests for regime detection system
# Author: Shamus Rae, Last Modified: 2024-01-15

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from crickformers.regime_detector import (
    RegimeConfig,
    RegimeLabels,
    RegimeFeatures,
    RegimeFeatureExtractor,
    RegimeDetector
)


class TestRegimeConfig:
    """Test regime configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RegimeConfig()
        
        assert config.window_size == 12
        assert config.min_window_size == 6
        assert config.n_regimes == 5
        assert config.clustering_method == 'gmm'
        assert config.fallback_mode is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RegimeConfig(
            window_size=8,
            n_regimes=3,
            clustering_method='kmeans',
            fallback_mode=False
        )
        
        assert config.window_size == 8
        assert config.n_regimes == 3
        assert config.clustering_method == 'kmeans'
        assert config.fallback_mode is False


class TestRegimeLabels:
    """Test regime label utilities."""
    
    def test_all_labels(self):
        """Test getting all regime labels."""
        labels = RegimeLabels.get_all_labels()
        
        assert len(labels) == 5
        assert RegimeLabels.POST_WICKET_CONSOLIDATION in labels
        assert RegimeLabels.SPIN_SQUEEZE in labels
        assert RegimeLabels.ALL_OUT_ATTACK in labels
        assert RegimeLabels.STEADY_ACCUMULATION in labels
        assert RegimeLabels.PRESSURE_BUILDING in labels
    
    def test_label_encoding(self):
        """Test label encoding and reverse encoding."""
        encoding = RegimeLabels.get_label_encoding()
        reverse_encoding = RegimeLabels.get_reverse_encoding()
        
        assert len(encoding) == 5
        assert len(reverse_encoding) == 5
        
        # Test round-trip
        for label, idx in encoding.items():
            assert reverse_encoding[idx] == label


class TestRegimeFeatures:
    """Test regime features container."""
    
    def test_feature_creation(self):
        """Test creating regime features."""
        features = RegimeFeatures(
            run_rate=8.5,
            boundary_percentage=0.25,
            wicket_count=1,
            wicket_density=0.1,
            biomech_volatility=0.3,
            balls_since_wicket=5,
            balls_since_boundary=2,
            phase='middle',
            over_number=12.0,
            required_run_rate=9.0
        )
        
        assert features.run_rate == 8.5
        assert features.boundary_percentage == 0.25
        assert features.wicket_count == 1
        assert features.phase == 'middle'
    
    def test_to_array(self):
        """Test converting features to numpy array."""
        features = RegimeFeatures(
            run_rate=8.5,
            boundary_percentage=0.25,
            wicket_count=1,
            wicket_density=0.1,
            biomech_volatility=0.3,
            balls_since_wicket=5,
            balls_since_boundary=2,
            phase='middle',
            over_number=12.0,
            required_run_rate=9.0
        )
        
        array = features.to_array()
        
        assert len(array) == 10
        assert array[0] == 8.5  # run_rate
        assert array[1] == 0.25  # boundary_percentage
        assert array[7] == 1  # phase (middle = 1)
        assert array[9] == 9.0  # required_run_rate
    
    def test_to_array_with_none_values(self):
        """Test array conversion with None values."""
        features = RegimeFeatures(
            run_rate=8.5,
            boundary_percentage=0.25,
            wicket_count=1,
            wicket_density=0.1,
            biomech_volatility=0.3,
            balls_since_wicket=5,
            balls_since_boundary=2,
            phase='unknown',
            over_number=12.0,
            required_run_rate=None
        )
        
        array = features.to_array()
        
        assert array[7] == 3  # unknown phase = 3
        assert array[9] == 0.0  # None required_run_rate = 0.0
    
    def test_feature_names(self):
        """Test getting feature names."""
        names = RegimeFeatures.get_feature_names()
        
        assert len(names) == 10
        assert 'run_rate' in names
        assert 'boundary_percentage' in names
        assert 'biomech_volatility' in names


class TestRegimeFeatureExtractor:
    """Test regime feature extraction."""
    
    @pytest.fixture
    def sample_ball_data(self):
        """Sample ball-by-ball data for testing."""
        return pd.DataFrame([
            {
                'match_id': 'test_match',
                'innings': 1,
                'over': 10,
                'ball': 1,
                'runs_scored': 1,
                'wicket_type': None
            },
            {
                'match_id': 'test_match',
                'innings': 1,
                'over': 10,
                'ball': 2,
                'runs_scored': 4,
                'wicket_type': None
            },
            {
                'match_id': 'test_match',
                'innings': 1,
                'over': 10,
                'ball': 3,
                'runs_scored': 0,
                'wicket_type': 'bowled'
            },
            {
                'match_id': 'test_match',
                'innings': 1,
                'over': 10,
                'ball': 4,
                'runs_scored': 1,
                'wicket_type': None
            },
            {
                'match_id': 'test_match',
                'innings': 1,
                'over': 10,
                'ball': 5,
                'runs_scored': 6,
                'wicket_type': None
            },
            {
                'match_id': 'test_match',
                'innings': 1,
                'over': 10,
                'ball': 6,
                'runs_scored': 0,
                'wicket_type': None
            }
        ])
    
    @pytest.fixture
    def sample_biomech_data(self):
        """Sample biomechanical data for testing."""
        return {
            'test_match_1_10_1': {'head_stability': 0.8, 'shot_commitment': 0.7},
            'test_match_1_10_2': {'head_stability': 0.9, 'shot_commitment': 0.9},
            'test_match_1_10_3': {'head_stability': 0.5, 'shot_commitment': 0.4},
            'test_match_1_10_4': {'head_stability': 0.7, 'shot_commitment': 0.6},
            'test_match_1_10_5': {'head_stability': 0.95, 'shot_commitment': 0.95},
            'test_match_1_10_6': {'head_stability': 0.6, 'shot_commitment': 0.5}
        }
    
    def test_extract_basic_features(self, sample_ball_data):
        """Test basic feature extraction."""
        extractor = RegimeFeatureExtractor()
        features = extractor.extract_features(sample_ball_data)
        
        assert isinstance(features, RegimeFeatures)
        assert features.run_rate > 0  # Should have positive run rate
        assert 0 <= features.boundary_percentage <= 1
        assert features.wicket_count >= 0
        assert features.phase in ['powerplay', 'middle', 'death', 'unknown']
    
    def test_run_rate_calculation(self, sample_ball_data):
        """Test run rate calculation."""
        extractor = RegimeFeatureExtractor()
        
        # Total runs in sample: 1+4+0+1+6+0 = 12 runs in 6 balls
        # Run rate = (12/6) * 6 = 12 runs per over
        features = extractor.extract_features(sample_ball_data)
        
        assert features.run_rate == 12.0
    
    def test_boundary_percentage_calculation(self, sample_ball_data):
        """Test boundary percentage calculation."""
        extractor = RegimeFeatureExtractor()
        features = extractor.extract_features(sample_ball_data)
        
        # Boundaries in sample: 4 and 6 = 2 boundaries out of 6 balls
        expected_boundary_pct = 2.0 / 6.0
        assert abs(features.boundary_percentage - expected_boundary_pct) < 0.01
    
    def test_wicket_metrics_calculation(self, sample_ball_data):
        """Test wicket count and density calculation."""
        extractor = RegimeFeatureExtractor()
        features = extractor.extract_features(sample_ball_data)
        
        # One wicket in sample data
        assert features.wicket_count == 1
        expected_density = 1.0 / 6.0
        assert abs(features.wicket_density - expected_density) < 0.01
    
    def test_biomech_volatility_calculation(self, sample_ball_data, sample_biomech_data):
        """Test biomechanical volatility calculation."""
        extractor = RegimeFeatureExtractor()
        features = extractor.extract_features(sample_ball_data, sample_biomech_data)
        
        # Should have non-zero volatility due to varying signals
        assert features.biomech_volatility > 0
        assert features.biomech_volatility < 1  # Should be reasonable
    
    def test_balls_since_wicket(self, sample_ball_data):
        """Test balls since wicket calculation."""
        extractor = RegimeFeatureExtractor()
        features = extractor.extract_features(sample_ball_data)
        
        # Wicket on ball 3, so balls since = 3 (balls 4, 5, 6)
        assert features.balls_since_wicket == 3
    
    def test_balls_since_boundary(self, sample_ball_data):
        """Test balls since boundary calculation."""
        extractor = RegimeFeatureExtractor()
        features = extractor.extract_features(sample_ball_data)
        
        # Last boundary on ball 5, so balls since = 1 (ball 6)
        assert features.balls_since_boundary == 1
    
    def test_phase_determination(self, sample_ball_data):
        """Test match phase determination."""
        extractor = RegimeFeatureExtractor()
        features = extractor.extract_features(sample_ball_data)
        
        # Over 10 should be middle phase
        assert features.phase == 'middle'
    
    def test_required_run_rate_calculation(self, sample_ball_data):
        """Test required run rate calculation."""
        extractor = RegimeFeatureExtractor()
        
        target_info = {
            'target_runs': 180,
            'current_score': 120,
            'balls_remaining': 60
        }
        
        features = extractor.extract_features(sample_ball_data, target_info=target_info)
        
        # Need 60 runs in 10 overs = 6 runs per over
        assert features.required_run_rate == 6.0
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        extractor = RegimeFeatureExtractor()
        empty_data = pd.DataFrame()
        
        features = extractor.extract_features(empty_data)
        
        # Should return default features
        assert isinstance(features, RegimeFeatures)
        assert features.run_rate > 0
        assert features.phase == 'middle'
    
    def test_missing_biomech_data(self, sample_ball_data):
        """Test handling missing biomechanical data."""
        extractor = RegimeFeatureExtractor()
        features = extractor.extract_features(sample_ball_data, biomech_data=None)
        
        # Should default to 0 volatility
        assert features.biomech_volatility == 0.0


class TestRegimeDetector:
    """Test regime detection system."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Sample training data for multiple matches."""
        matches = []
        
        # Match 1: Post-wicket consolidation scenario
        match1 = pd.DataFrame([
            {'match_id': 'match1', 'innings': 1, 'over': i//6 + 1, 'ball': i%6 + 1, 
             'runs_scored': 1 if i > 2 else 0, 'wicket_type': 'bowled' if i == 2 else None}
            for i in range(12)
        ])
        matches.append(match1)
        
        # Match 2: All-out attack scenario
        match2 = pd.DataFrame([
            {'match_id': 'match2', 'innings': 1, 'over': i//6 + 1, 'ball': i%6 + 1,
             'runs_scored': 4 if i % 2 == 0 else 6, 'wicket_type': None}
            for i in range(12)
        ])
        matches.append(match2)
        
        # Match 3: Spin squeeze scenario
        match3 = pd.DataFrame([
            {'match_id': 'match3', 'innings': 1, 'over': i//6 + 8, 'ball': i%6 + 1,
             'runs_scored': 1 if i % 3 == 0 else 0, 'wicket_type': None}
            for i in range(12)
        ])
        matches.append(match3)
        
        return matches
    
    def test_detector_initialization(self):
        """Test regime detector initialization."""
        detector = RegimeDetector()
        
        assert detector.config is not None
        assert detector.feature_extractor is not None
        assert not detector.is_trained
        assert len(detector.regime_mapping) == 5
    
    def test_training_basic(self, sample_training_data):
        """Test basic training functionality."""
        detector = RegimeDetector()
        
        # Should not raise exception
        detector.train(sample_training_data)
        
        assert detector.is_trained
        assert detector.clustering_model is not None
        assert detector.scaler is not None
    
    def test_regime_detection_untrained(self):
        """Test regime detection without training (fallback mode)."""
        detector = RegimeDetector()
        
        # Post-wicket scenario
        recent_balls = pd.DataFrame([
            {'match_id': 'test', 'innings': 1, 'over': 5, 'ball': i+1,
             'runs_scored': 1, 'wicket_type': 'bowled' if i == 0 else None}
            for i in range(6)
        ])
        
        regime, confidence = detector.detect_regime(recent_balls)
        
        assert regime in RegimeLabels.get_all_labels()
        assert 0 <= confidence <= 1
    
    def test_rule_based_post_wicket_detection(self):
        """Test rule-based post-wicket consolidation detection."""
        detector = RegimeDetector()
        
        # Create post-wicket scenario: recent wicket (last 3 balls), low run rate
        recent_balls = pd.DataFrame([
            {'match_id': 'test', 'innings': 1, 'over': 5, 'ball': 1,
             'runs_scored': 1, 'wicket_type': None},
            {'match_id': 'test', 'innings': 1, 'over': 5, 'ball': 2,
             'runs_scored': 0, 'wicket_type': None},
            {'match_id': 'test', 'innings': 1, 'over': 5, 'ball': 3,
             'runs_scored': 0, 'wicket_type': None},
            {'match_id': 'test', 'innings': 1, 'over': 5, 'ball': 4,
             'runs_scored': 0, 'wicket_type': 'bowled'},  # Wicket on ball 4
            {'match_id': 'test', 'innings': 1, 'over': 5, 'ball': 5,
             'runs_scored': 1, 'wicket_type': None},      # 1 ball since wicket
            {'match_id': 'test', 'innings': 1, 'over': 5, 'ball': 6,
             'runs_scored': 0, 'wicket_type': None}       # 2 balls since wicket
        ])
        
        regime, confidence = detector.detect_regime(recent_balls)
        
        assert regime == RegimeLabels.POST_WICKET_CONSOLIDATION
        assert confidence > 0
    
    def test_rule_based_all_out_attack_detection(self):
        """Test rule-based all-out attack detection."""
        detector = RegimeDetector()
        
        # Create all-out attack scenario: high run rate, many boundaries
        recent_balls = pd.DataFrame([
            {'match_id': 'test', 'innings': 1, 'over': 18, 'ball': i+1,
             'runs_scored': 6 if i % 2 == 0 else 4, 'wicket_type': None}
            for i in range(6)
        ])
        
        regime, confidence = detector.detect_regime(recent_balls)
        
        assert regime == RegimeLabels.ALL_OUT_ATTACK
        assert confidence > 0
    
    def test_rule_based_spin_squeeze_detection(self):
        """Test rule-based spin squeeze detection."""
        detector = RegimeDetector()
        
        # Create spin squeeze scenario: low run rate, no boundaries, middle overs
        recent_balls = pd.DataFrame([
            {'match_id': 'test', 'innings': 1, 'over': 10, 'ball': i+1,
             'runs_scored': 1 if i % 3 == 0 else 0, 'wicket_type': None}
            for i in range(12)
        ])
        
        regime, confidence = detector.detect_regime(recent_balls)
        
        assert regime == RegimeLabels.SPIN_SQUEEZE
        assert confidence > 0
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        detector = RegimeDetector()
        
        # Too few balls
        recent_balls = pd.DataFrame([
            {'match_id': 'test', 'innings': 1, 'over': 5, 'ball': 1,
             'runs_scored': 4, 'wicket_type': None}
        ])
        
        regime, confidence = detector.detect_regime(recent_balls)
        
        assert regime == RegimeLabels.UNKNOWN
        assert confidence == 0.0
    
    def test_regime_probabilities_untrained(self):
        """Test getting regime probabilities without training."""
        detector = RegimeDetector()
        
        recent_balls = pd.DataFrame([
            {'match_id': 'test', 'innings': 1, 'over': 5, 'ball': i+1,
             'runs_scored': 1, 'wicket_type': None}
            for i in range(6)
        ])
        
        probabilities = detector.get_regime_probabilities(recent_balls)
        
        assert len(probabilities) == 5
        assert all(0 <= prob <= 1 for prob in probabilities.values())
        assert abs(sum(probabilities.values()) - 1.0) < 0.01
    
    def test_regime_encoding(self):
        """Test regime label encoding."""
        detector = RegimeDetector()
        
        encoding = detector.get_regime_encoding(RegimeLabels.ALL_OUT_ATTACK)
        assert isinstance(encoding, int)
        assert 0 <= encoding <= 5
        
        # Unknown regime should get highest value
        unknown_encoding = detector.get_regime_encoding("Unknown Regime")
        assert unknown_encoding == 5
    
    def test_model_persistence(self, sample_training_data):
        """Test saving and loading model."""
        detector = RegimeDetector()
        detector.train(sample_training_data)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            detector.save_model(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Load model in new detector
            new_detector = RegimeDetector()
            success = new_detector.load_model(tmp_path)
            
            assert success
            assert new_detector.is_trained
            assert new_detector.clustering_model is not None
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_model_loading_failure(self):
        """Test handling of model loading failure."""
        detector = RegimeDetector()
        
        # Try to load non-existent file
        success = detector.load_model("nonexistent_file.pkl")
        
        assert not success
        assert not detector.is_trained
    
    def test_feature_importance(self, sample_training_data):
        """Test feature importance calculation."""
        detector = RegimeDetector()
        detector.train(sample_training_data)
        
        importance = detector.get_feature_importance()
        
        if importance:  # May be empty if clustering doesn't support it
            assert len(importance) == 10  # Number of features
            assert all(0 <= imp <= 1 for imp in importance.values())
            # Should sum to approximately 1
            assert abs(sum(importance.values()) - 1.0) < 0.1


class TestRegimeDetectorWithSyntheticSequences:
    """Test regime detector with synthetic sequences to validate specific behaviors."""
    
    def create_synthetic_sequence(self, regime_type: str, length: int = 20) -> pd.DataFrame:
        """Create synthetic ball sequence for specific regime type."""
        balls = []
        
        for i in range(length):
            over = (i // 6) + 1
            ball = (i % 6) + 1
            
            if regime_type == "post_wicket":
                # Recent wicket, low scoring
                runs = 1 if i % 4 == 0 else 0  # Very low scoring
                wicket = 'bowled' if i == length - 3 else None  # Wicket near the end
            elif regime_type == "all_out_attack":
                # High scoring, many boundaries
                runs = 6 if i % 2 == 0 else 4
                wicket = None
            elif regime_type == "spin_squeeze":
                # Low scoring, few boundaries, middle overs
                runs = 1 if i % 4 == 0 else 0
                wicket = None
                over += 8  # Middle overs
            elif regime_type == "steady_accumulation":
                # Moderate scoring, occasional boundaries
                runs = 4 if i % 6 == 0 else (2 if i % 3 == 0 else 1)
                wicket = None
            else:  # pressure_building
                # Moderate scoring with pressure (death overs)
                runs = 2 if i % 3 == 0 else 1
                wicket = None
                over += 16  # Death overs
            
            balls.append({
                'match_id': f'synthetic_{regime_type}',
                'innings': 1,
                'over': over,
                'ball': ball,
                'runs_scored': runs,
                'wicket_type': wicket
            })
        
        return pd.DataFrame(balls)
    
    def test_synthetic_post_wicket_sequence(self):
        """Test detection on synthetic post-wicket sequence."""
        detector = RegimeDetector()
        sequence = self.create_synthetic_sequence("post_wicket")
        
        # Test on recent balls
        recent_balls = sequence.tail(12)
        regime, confidence = detector.detect_regime(recent_balls)
        
        # Should detect post-wicket consolidation
        assert regime == RegimeLabels.POST_WICKET_CONSOLIDATION
        assert confidence > 0.5
    
    def test_synthetic_all_out_attack_sequence(self):
        """Test detection on synthetic all-out attack sequence."""
        detector = RegimeDetector()
        sequence = self.create_synthetic_sequence("all_out_attack")
        
        recent_balls = sequence.tail(12)
        regime, confidence = detector.detect_regime(recent_balls)
        
        # Should detect all-out attack
        assert regime == RegimeLabels.ALL_OUT_ATTACK
        assert confidence > 0.5
    
    def test_synthetic_spin_squeeze_sequence(self):
        """Test detection on synthetic spin squeeze sequence."""
        detector = RegimeDetector()
        sequence = self.create_synthetic_sequence("spin_squeeze")
        
        recent_balls = sequence.tail(12)
        regime, confidence = detector.detect_regime(recent_balls)
        
        # Should detect spin squeeze
        assert regime == RegimeLabels.SPIN_SQUEEZE
        assert confidence > 0.5
    
    def test_state_transition_consistency(self):
        """Test that state transitions are consistent."""
        detector = RegimeDetector()
        
        # Create sequence that transitions from post-wicket to attack
        post_wicket_seq = self.create_synthetic_sequence("post_wicket", 10)
        attack_seq = self.create_synthetic_sequence("all_out_attack", 10)
        
        # Test detection at different points
        regimes = []
        
        # Test post-wicket phase
        recent_balls = post_wicket_seq.tail(8)
        regime, _ = detector.detect_regime(recent_balls)
        regimes.append(regime)
        
        # Test transition phase (mix of both)
        transition_seq = pd.concat([post_wicket_seq.tail(6), attack_seq.head(6)])
        regime, _ = detector.detect_regime(transition_seq)
        regimes.append(regime)
        
        # Test attack phase
        recent_balls = attack_seq.tail(8)
        regime, _ = detector.detect_regime(recent_balls)
        regimes.append(regime)
        
        # Should show logical progression
        assert regimes[0] == RegimeLabels.POST_WICKET_CONSOLIDATION
        assert regimes[2] == RegimeLabels.ALL_OUT_ATTACK
        # Transition regime could be either, depending on which signals dominate
    
    def test_trained_vs_untrained_consistency(self):
        """Test consistency between trained and untrained detection."""
        # Create training data
        training_sequences = [
            self.create_synthetic_sequence("post_wicket", 30),
            self.create_synthetic_sequence("all_out_attack", 30),
            self.create_synthetic_sequence("spin_squeeze", 30)
        ]
        
        # Test sequence
        test_sequence = self.create_synthetic_sequence("all_out_attack", 15)
        recent_balls = test_sequence.tail(12)
        
        # Untrained detection
        detector_untrained = RegimeDetector()
        regime_untrained, conf_untrained = detector_untrained.detect_regime(recent_balls)
        
        # Trained detection
        detector_trained = RegimeDetector()
        detector_trained.train(training_sequences)
        regime_trained, conf_trained = detector_trained.detect_regime(recent_balls)
        
        # Both should detect attack regime (though confidence may differ)
        assert regime_untrained == RegimeLabels.ALL_OUT_ATTACK
        # Trained detector should also detect attack or at least not unknown
        assert regime_trained != RegimeLabels.UNKNOWN
    
    def test_fallback_mode_functionality(self):
        """Test fallback mode when model fails."""
        detector = RegimeDetector()
        
        # Mock model failure
        with patch.object(detector, '_model_based_detection', side_effect=Exception("Model failed")):
            # Train detector so it thinks it has a model
            detector.is_trained = True
            
            sequence = self.create_synthetic_sequence("all_out_attack")
            recent_balls = sequence.tail(12)
            
            regime, confidence = detector.detect_regime(recent_balls)
            
            # Should fall back to rule-based detection
            assert regime == RegimeLabels.ALL_OUT_ATTACK
            assert confidence > 0
    
    def test_fallback_mode_disabled(self):
        """Test behavior when fallback mode is disabled."""
        config = RegimeConfig(fallback_mode=False)
        detector = RegimeDetector(config)
        
        # Mock model failure
        with patch.object(detector, '_model_based_detection', side_effect=Exception("Model failed")):
            detector.is_trained = True
            
            sequence = self.create_synthetic_sequence("all_out_attack")
            recent_balls = sequence.tail(12)
            
            regime, confidence = detector.detect_regime(recent_balls)
            
            # Should return unknown when fallback is disabled
            assert regime == RegimeLabels.UNKNOWN
            assert confidence == 0.0


class TestRegimeDetectorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_training_data(self):
        """Test training with empty data."""
        detector = RegimeDetector()
        
        # Should not crash
        detector.train([])
        
        # Should not be marked as trained
        assert not detector.is_trained
    
    def test_single_ball_training_data(self):
        """Test training with insufficient data."""
        detector = RegimeDetector()
        
        single_ball_match = pd.DataFrame([
            {'match_id': 'test', 'innings': 1, 'over': 1, 'ball': 1,
             'runs_scored': 4, 'wicket_type': None}
        ])
        
        # Should handle gracefully
        detector.train([single_ball_match])
        
        # May or may not be trained depending on implementation
        # Should not crash
    
    def test_malformed_ball_data(self):
        """Test handling malformed ball data."""
        detector = RegimeDetector()
        
        # Missing required columns
        malformed_data = pd.DataFrame([
            {'some_column': 'value'}
        ])
        
        regime, confidence = detector.detect_regime(malformed_data)
        
        # Should handle gracefully
        assert regime == RegimeLabels.UNKNOWN
        assert confidence == 0.0
    
    def test_extreme_values(self):
        """Test handling extreme values."""
        detector = RegimeDetector()
        
        # Extreme run scoring
        extreme_data = pd.DataFrame([
            {'match_id': 'test', 'innings': 1, 'over': 1, 'ball': i+1,
             'runs_scored': 100, 'wicket_type': None}  # Impossible runs
            for i in range(6)
        ])
        
        # Should handle without crashing
        regime, confidence = detector.detect_regime(extreme_data)
        
        assert regime in RegimeLabels.get_all_labels() + [RegimeLabels.UNKNOWN]
        assert 0 <= confidence <= 1
    
    def test_mixed_data_types(self):
        """Test handling mixed data types."""
        detector = RegimeDetector()
        
        mixed_data = pd.DataFrame([
            {'match_id': 'test', 'innings': 1, 'over': 1.0, 'ball': 1,
             'runs_scored': '4', 'wicket_type': None},  # String runs
            {'match_id': 'test', 'innings': 1, 'over': 1.0, 'ball': 2,
             'runs_scored': 2.5, 'wicket_type': None},  # Float runs
            {'match_id': 'test', 'innings': 1, 'over': 1.0, 'ball': 3,
             'runs_scored': None, 'wicket_type': None}   # None runs
        ])
        
        # Should handle type conversion issues gracefully
        try:
            regime, confidence = detector.detect_regime(mixed_data)
            assert regime is not None
            assert confidence is not None
        except Exception as e:
            # If it fails, it should fail gracefully
            assert isinstance(e, (ValueError, TypeError))
    
    def test_concurrent_access(self):
        """Test thread safety (basic check)."""
        detector = RegimeDetector()
        
        test_data = pd.DataFrame([
            {'match_id': 'test', 'innings': 1, 'over': 1, 'ball': i+1,
             'runs_scored': 1, 'wicket_type': None}
            for i in range(6)
        ])
        
        # Multiple calls should not interfere
        results = []
        for _ in range(5):
            regime, confidence = detector.detect_regime(test_data)
            results.append((regime, confidence))
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result