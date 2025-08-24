# Purpose: Comprehensive unit tests for temporal decay system
# Author: WicketWise Team, Last Modified: 2025-08-23

import pytest
import math
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from crickformers.gnn.temporal_decay import (
    DecayType,
    DecayConfig,
    PerformanceEvent,
    ExponentialDecay,
    LinearDecay,
    SigmoidDecay,
    AdaptiveDecay,
    ContextAwareDecay,
    TemporalDecayEngine
)


class TestDecayConfig:
    """Test DecayConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = DecayConfig(decay_type=DecayType.EXPONENTIAL)
        
        assert config.decay_type == DecayType.EXPONENTIAL
        assert config.half_life_days == 365.0
        assert config.min_weight == 0.01
        assert config.max_days == 1095.0
        assert config.context_multipliers is None
        assert config.adaptive_params is None
    
    def test_custom_config(self):
        """Test custom configuration"""
        context_multipliers = {"format_t20": 1.5, "venue_home": 0.8}
        adaptive_params = {"form_window_days": 60, "form_threshold": 0.15}
        
        config = DecayConfig(
            decay_type=DecayType.ADAPTIVE,
            half_life_days=180.0,
            min_weight=0.05,
            max_days=730.0,
            context_multipliers=context_multipliers,
            adaptive_params=adaptive_params
        )
        
        assert config.decay_type == DecayType.ADAPTIVE
        assert config.half_life_days == 180.0
        assert config.min_weight == 0.05
        assert config.max_days == 730.0
        assert config.context_multipliers == context_multipliers
        assert config.adaptive_params == adaptive_params


class TestPerformanceEvent:
    """Test PerformanceEvent dataclass"""
    
    def test_basic_event(self):
        """Test basic performance event creation"""
        event = PerformanceEvent(
            event_id="match_123_ball_456",
            player_id="player_001",
            date=datetime(2024, 1, 15),
            performance_metrics={"runs": 45, "strike_rate": 125.0},
            context={"format": "t20", "venue": "home"}
        )
        
        assert event.event_id == "match_123_ball_456"
        assert event.player_id == "player_001"
        assert event.date == datetime(2024, 1, 15)
        assert event.performance_metrics["runs"] == 45
        assert event.context["format"] == "t20"
        assert event.importance_score == 1.0  # Default
        assert event.confidence_score == 1.0  # Default
    
    def test_event_with_scores(self):
        """Test event with custom importance and confidence scores"""
        event = PerformanceEvent(
            event_id="final_match",
            player_id="player_001",
            date=datetime.now(),
            performance_metrics={"wickets": 3},
            context={"tournament_stage": "final"},
            importance_score=2.5,
            confidence_score=0.95
        )
        
        assert event.importance_score == 2.5
        assert event.confidence_score == 0.95


class TestExponentialDecay:
    """Test ExponentialDecay function"""
    
    def test_initialization(self):
        """Test exponential decay initialization"""
        config = DecayConfig(DecayType.EXPONENTIAL, half_life_days=365.0)
        decay = ExponentialDecay(config)
        
        expected_lambda = math.log(2) / 365.0
        assert abs(decay.lambda_param - expected_lambda) < 1e-10
    
    def test_half_life_weight(self):
        """Test that weight is 0.5 at half-life"""
        config = DecayConfig(DecayType.EXPONENTIAL, half_life_days=365.0)
        decay = ExponentialDecay(config)
        
        weight = decay.calculate_weight(365.0)
        assert abs(weight - 0.5) < 1e-10
    
    def test_zero_days_weight(self):
        """Test that weight is 1.0 for zero days ago"""
        config = DecayConfig(DecayType.EXPONENTIAL, half_life_days=365.0)
        decay = ExponentialDecay(config)
        
        weight = decay.calculate_weight(0.0)
        assert weight == 1.0
    
    def test_future_date_weight(self):
        """Test that future dates get full weight"""
        config = DecayConfig(DecayType.EXPONENTIAL, half_life_days=365.0)
        decay = ExponentialDecay(config)
        
        weight = decay.calculate_weight(-30.0)  # 30 days in future
        assert weight == 1.0
    
    def test_max_days_limit(self):
        """Test that weights beyond max_days return min_weight"""
        config = DecayConfig(DecayType.EXPONENTIAL, half_life_days=365.0, max_days=1000.0, min_weight=0.01)
        decay = ExponentialDecay(config)
        
        weight = decay.calculate_weight(1500.0)  # Beyond max_days
        assert weight == 0.01
    
    def test_context_multipliers(self):
        """Test context-based weight adjustments"""
        context_multipliers = {"format_t20": 2.0}  # Faster decay for T20
        config = DecayConfig(
            DecayType.EXPONENTIAL, 
            half_life_days=365.0,
            context_multipliers=context_multipliers
        )
        decay = ExponentialDecay(config)
        
        # Without context
        weight_no_context = decay.calculate_weight(365.0)
        
        # With T20 context (should decay faster)
        context = {"format": "t20"}
        weight_with_context = decay.calculate_weight(365.0, context)
        
        assert weight_with_context < weight_no_context
        # With 2x multiplier, should be around 0.25 instead of 0.5
        assert abs(weight_with_context - 0.25) < 0.01
    
    def test_effective_half_life(self):
        """Test effective half-life calculation"""
        config = DecayConfig(DecayType.EXPONENTIAL, half_life_days=365.0)
        decay = ExponentialDecay(config)
        
        # Without context
        half_life = decay.get_effective_half_life()
        assert half_life == 365.0
        
        # With context multiplier
        context_multipliers = {"format_t20": 2.0}
        config.context_multipliers = context_multipliers
        decay = ExponentialDecay(config)
        
        context = {"format": "t20"}
        effective_half_life = decay.get_effective_half_life(context)
        assert effective_half_life == 182.5  # 365 / 2


class TestLinearDecay:
    """Test LinearDecay function"""
    
    def test_zero_days_weight(self):
        """Test that weight is 1.0 for zero days ago"""
        config = DecayConfig(DecayType.LINEAR, max_days=1000.0)
        decay = LinearDecay(config)
        
        weight = decay.calculate_weight(0.0)
        assert weight == 1.0
    
    def test_half_max_days_weight(self):
        """Test weight at half of max_days"""
        config = DecayConfig(DecayType.LINEAR, max_days=1000.0)
        decay = LinearDecay(config)
        
        weight = decay.calculate_weight(500.0)
        assert abs(weight - 0.5) < 1e-10
    
    def test_max_days_weight(self):
        """Test weight at max_days returns min_weight"""
        config = DecayConfig(DecayType.LINEAR, max_days=1000.0, min_weight=0.01)
        decay = LinearDecay(config)
        
        weight = decay.calculate_weight(1000.0)
        assert weight == 0.01
    
    def test_beyond_max_days(self):
        """Test weight beyond max_days"""
        config = DecayConfig(DecayType.LINEAR, max_days=1000.0, min_weight=0.01)
        decay = LinearDecay(config)
        
        weight = decay.calculate_weight(1500.0)
        assert weight == 0.01
    
    def test_context_adjustment(self):
        """Test context-based adjustments"""
        context_multipliers = {"format_t20": 2.0}
        config = DecayConfig(
            DecayType.LINEAR,
            max_days=1000.0,
            context_multipliers=context_multipliers
        )
        decay = LinearDecay(config)
        
        # With context, effective max_days becomes 500
        context = {"format": "t20"}
        weight = decay.calculate_weight(250.0, context)  # Half of effective max_days
        assert abs(weight - 0.5) < 1e-10


class TestSigmoidDecay:
    """Test SigmoidDecay function"""
    
    def test_initialization(self):
        """Test sigmoid decay initialization"""
        config = DecayConfig(DecayType.SIGMOID, half_life_days=365.0)
        decay = SigmoidDecay(config)
        
        assert decay.midpoint == 365.0
        assert abs(decay.steepness - (4.0 / 365.0)) < 1e-10
    
    def test_midpoint_weight(self):
        """Test that weight is 0.5 at midpoint"""
        config = DecayConfig(DecayType.SIGMOID, half_life_days=365.0)
        decay = SigmoidDecay(config)
        
        weight = decay.calculate_weight(365.0)
        assert abs(weight - 0.5) < 1e-6
    
    def test_zero_days_weight(self):
        """Test weight approaches 1.0 for zero days"""
        config = DecayConfig(DecayType.SIGMOID, half_life_days=365.0)
        decay = SigmoidDecay(config)
        
        weight = decay.calculate_weight(0.0)
        assert weight > 0.95  # Should be very close to 1.0
    
    def test_far_future_weight(self):
        """Test weight approaches min_weight for far future"""
        config = DecayConfig(DecayType.SIGMOID, half_life_days=365.0, min_weight=0.01)
        decay = SigmoidDecay(config)
        
        weight = decay.calculate_weight(2000.0)  # Far in the past
        assert weight < 0.05  # Should be very close to min_weight
    
    def test_custom_parameters(self):
        """Test sigmoid with custom parameters"""
        adaptive_params = {"midpoint": 180.0, "steepness": 0.02}
        config = DecayConfig(
            DecayType.SIGMOID,
            half_life_days=365.0,
            adaptive_params=adaptive_params
        )
        decay = SigmoidDecay(config)
        
        assert decay.midpoint == 180.0
        assert decay.steepness == 0.02


class TestAdaptiveDecay:
    """Test AdaptiveDecay function"""
    
    def test_initialization(self):
        """Test adaptive decay initialization"""
        config = DecayConfig(DecayType.ADAPTIVE, half_life_days=365.0)
        decay = AdaptiveDecay(config)
        
        assert decay.form_window_days == 90  # Default
        assert decay.form_threshold == 0.1   # Default
    
    def test_custom_parameters(self):
        """Test adaptive decay with custom parameters"""
        adaptive_params = {"form_window_days": 60, "form_threshold": 0.15}
        config = DecayConfig(
            DecayType.ADAPTIVE,
            half_life_days=365.0,
            adaptive_params=adaptive_params
        )
        decay = AdaptiveDecay(config)
        
        assert decay.form_window_days == 60
        assert decay.form_threshold == 0.15
    
    def test_good_form_adjustment(self):
        """Test weight adjustment for good recent form"""
        config = DecayConfig(DecayType.ADAPTIVE, half_life_days=365.0)
        decay = AdaptiveDecay(config)
        
        # Without form context
        weight_no_form = decay.calculate_weight(365.0)
        
        # With good form
        context = {"recent_form": "0.3"}  # Above threshold
        weight_good_form = decay.calculate_weight(365.0, context)
        
        assert weight_good_form > weight_no_form
    
    def test_tournament_importance_adjustment(self):
        """Test weight adjustment for tournament importance"""
        config = DecayConfig(DecayType.ADAPTIVE, half_life_days=365.0)
        decay = AdaptiveDecay(config)
        
        # Without importance context
        weight_normal = decay.calculate_weight(365.0)
        
        # With high importance
        context = {"tournament_importance": "2.0"}  # High importance
        weight_important = decay.calculate_weight(365.0, context)
        
        assert weight_important > weight_normal


class TestContextAwareDecay:
    """Test ContextAwareDecay function"""
    
    def test_initialization(self):
        """Test context-aware decay initialization"""
        config = DecayConfig(DecayType.CONTEXT_AWARE, half_life_days=365.0)
        decay = ContextAwareDecay(config)
        
        assert "t20" in decay.decay_functions
        assert "odi" in decay.decay_functions
        assert "test" in decay.decay_functions
        assert "playoff" in decay.decay_functions
        assert "final" in decay.decay_functions
    
    def test_format_specific_decay(self):
        """Test format-specific decay rates"""
        config = DecayConfig(DecayType.CONTEXT_AWARE, half_life_days=365.0)
        decay = ContextAwareDecay(config)
        
        # T20 should decay faster than ODI
        t20_weight = decay.calculate_weight(180.0, {"format": "t20"})
        odi_weight = decay.calculate_weight(180.0, {"format": "odi"})
        
        assert t20_weight < odi_weight  # T20 decays faster
    
    def test_tournament_stage_decay(self):
        """Test tournament stage-specific decay"""
        config = DecayConfig(DecayType.CONTEXT_AWARE, half_life_days=365.0)
        decay = ContextAwareDecay(config)
        
        playoff_weight = decay.calculate_weight(365.0, {"tournament_stage": "playoff"})
        final_weight = decay.calculate_weight(365.0, {"tournament_stage": "final"})
        
        # Finals should have longer relevance than playoffs
        assert final_weight > playoff_weight
    
    def test_no_context_fallback(self):
        """Test fallback to default decay when no context"""
        config = DecayConfig(DecayType.CONTEXT_AWARE, half_life_days=365.0)
        decay = ContextAwareDecay(config)
        
        weight = decay.calculate_weight(365.0)
        expected_weight = decay.default_decay.calculate_weight(365.0)
        
        assert abs(weight - expected_weight) < 1e-10


class TestTemporalDecayEngine:
    """Test TemporalDecayEngine"""
    
    @pytest.fixture
    def sample_events(self):
        """Create sample performance events for testing"""
        base_date = datetime(2024, 1, 1)
        events = []
        
        for i in range(5):
            event = PerformanceEvent(
                event_id=f"event_{i}",
                player_id="player_001",
                date=base_date + timedelta(days=i * 30),
                performance_metrics={"runs": 50 + i * 10, "strike_rate": 120.0 + i * 5},
                context={"format": "t20", "venue": "home"},
                importance_score=1.0 + i * 0.1
            )
            events.append(event)
        
        return events
    
    def test_initialization(self):
        """Test engine initialization"""
        engine = TemporalDecayEngine()
        
        assert engine.default_config.decay_type == DecayType.EXPONENTIAL
        assert len(engine.decay_functions) == 5  # All decay types
        assert len(engine.calculation_history) == 0
    
    def test_custom_config_initialization(self):
        """Test engine initialization with custom config"""
        config = DecayConfig(DecayType.LINEAR, half_life_days=180.0)
        engine = TemporalDecayEngine(config)
        
        assert engine.default_config.decay_type == DecayType.LINEAR
        assert engine.default_config.half_life_days == 180.0
    
    def test_calculate_temporal_weights(self, sample_events):
        """Test temporal weight calculation"""
        engine = TemporalDecayEngine()
        reference_date = datetime(2024, 6, 1)
        
        weighted_events = engine.calculate_temporal_weights(
            sample_events, 
            reference_date=reference_date,
            decay_type=DecayType.EXPONENTIAL
        )
        
        assert len(weighted_events) == 5
        
        # Check that events are sorted by weight (descending)
        weights = [w for _, w in weighted_events]
        assert weights == sorted(weights, reverse=True)
        
        # More recent events should have higher weights
        most_recent_event = max(sample_events, key=lambda e: e.date)
        most_recent_weight = next(w for e, w in weighted_events if e.event_id == most_recent_event.event_id)
        
        oldest_event = min(sample_events, key=lambda e: e.date)
        oldest_weight = next(w for e, w in weighted_events if e.event_id == oldest_event.event_id)
        
        assert most_recent_weight > oldest_weight
    
    def test_empty_events_list(self):
        """Test handling of empty events list"""
        engine = TemporalDecayEngine()
        weighted_events = engine.calculate_temporal_weights([])
        
        assert weighted_events == []
    
    def test_importance_score_effect(self):
        """Test that importance score affects final weight"""
        engine = TemporalDecayEngine()
        
        # Create two identical events with different importance scores
        base_event = PerformanceEvent(
            event_id="base_event",
            player_id="player_001",
            date=datetime(2024, 1, 1),
            performance_metrics={"runs": 50},
            context={},
            importance_score=1.0
        )
        
        important_event = PerformanceEvent(
            event_id="important_event",
            player_id="player_001",
            date=datetime(2024, 1, 1),
            performance_metrics={"runs": 50},
            context={},
            importance_score=2.0
        )
        
        weighted_events = engine.calculate_temporal_weights([base_event, important_event])
        
        base_weight = next(w for e, w in weighted_events if e.event_id == "base_event")
        important_weight = next(w for e, w in weighted_events if e.event_id == "important_event")
        
        assert important_weight > base_weight
    
    def test_get_weighted_aggregation_mean(self, sample_events):
        """Test weighted mean aggregation"""
        engine = TemporalDecayEngine()
        
        result = engine.get_weighted_aggregation(
            sample_events,
            "runs",
            decay_type=DecayType.EXPONENTIAL,
            aggregation_type="weighted_mean"
        )
        
        assert "value" in result
        assert "confidence" in result
        assert "sample_size" in result
        assert result["sample_size"] == 5
        assert 0 <= result["confidence"] <= 1
        assert result["value"] > 0  # Should be positive for runs
    
    def test_get_weighted_aggregation_sum(self, sample_events):
        """Test weighted sum aggregation"""
        engine = TemporalDecayEngine()
        
        result = engine.get_weighted_aggregation(
            sample_events,
            "runs",
            aggregation_type="weighted_sum"
        )
        
        assert result["value"] > 0
        assert result["sample_size"] == 5
    
    def test_get_weighted_aggregation_max(self, sample_events):
        """Test max aggregation"""
        engine = TemporalDecayEngine()
        
        result = engine.get_weighted_aggregation(
            sample_events,
            "runs",
            aggregation_type="max"
        )
        
        # Should return the value from the highest weighted event
        assert result["value"] > 0
        assert result["sample_size"] == 5
    
    def test_get_weighted_aggregation_trend(self, sample_events):
        """Test trend aggregation"""
        engine = TemporalDecayEngine()
        
        result = engine.get_weighted_aggregation(
            sample_events,
            "runs",
            aggregation_type="recent_trend"
        )
        
        # With increasing runs (50, 60, 70, 80, 90), trend should be positive
        assert result["value"] > 0  # Positive trend
        assert result["sample_size"] == 5
    
    def test_missing_metric_handling(self, sample_events):
        """Test handling of missing metrics"""
        engine = TemporalDecayEngine()
        
        result = engine.get_weighted_aggregation(
            sample_events,
            "nonexistent_metric",
            aggregation_type="weighted_mean"
        )
        
        assert result["value"] == 0.0
        assert result["confidence"] == 0.0
        assert result["sample_size"] == 0
    
    def test_analyze_decay_patterns(self, sample_events):
        """Test decay pattern analysis"""
        engine = TemporalDecayEngine()
        
        analysis = engine.analyze_decay_patterns(sample_events, "runs")
        
        # Should have results for all decay types
        assert len(analysis) == 5
        assert "exponential" in analysis
        assert "linear" in analysis
        assert "sigmoid" in analysis
        assert "adaptive" in analysis
        assert "context_aware" in analysis
        
        # Each result should have value, confidence, sample_size
        for decay_type, result in analysis.items():
            if "error" not in result:  # Skip any that errored
                assert "value" in result
                assert "confidence" in result
                assert "sample_size" in result
    
    def test_calculation_statistics_tracking(self, sample_events):
        """Test that calculation statistics are tracked"""
        engine = TemporalDecayEngine()
        
        # Perform some calculations
        engine.calculate_temporal_weights(sample_events, decay_type=DecayType.EXPONENTIAL)
        engine.calculate_temporal_weights(sample_events, decay_type=DecayType.LINEAR)
        
        stats = engine.get_calculation_statistics()
        
        assert stats["total_calculations"] == 10  # 5 events × 2 decay types
        assert stats["unique_players"] == 1
        assert "exponential" in stats["decay_types_used"]
        assert "linear" in stats["decay_types_used"]
        assert "average_temporal_weight" in stats
        assert "weight_distribution" in stats
    
    def test_calculation_statistics_empty(self):
        """Test statistics when no calculations performed"""
        engine = TemporalDecayEngine()
        
        stats = engine.get_calculation_statistics()
        assert "message" in stats
        assert stats["message"] == "No calculations performed yet"


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    def test_player_form_analysis(self):
        """Test analyzing player form over time with different contexts"""
        engine = TemporalDecayEngine()
        
        # Create realistic performance events
        events = []
        base_date = datetime(2024, 1, 1)
        
        # Good early form
        for i in range(3):
            events.append(PerformanceEvent(
                event_id=f"early_{i}",
                player_id="kohli",
                date=base_date + timedelta(days=i * 10),
                performance_metrics={"runs": 80 + i * 5, "strike_rate": 140.0},
                context={"format": "t20", "venue": "home"},
                importance_score=1.0
            ))
        
        # Poor middle form
        for i in range(3):
            events.append(PerformanceEvent(
                event_id=f"middle_{i}",
                player_id="kohli",
                date=base_date + timedelta(days=60 + i * 10),
                performance_metrics={"runs": 20 + i * 2, "strike_rate": 95.0},
                context={"format": "t20", "venue": "away"},
                importance_score=1.0
            ))
        
        # Recent good form
        for i in range(4):
            events.append(PerformanceEvent(
                event_id=f"recent_{i}",
                player_id="kohli",
                date=base_date + timedelta(days=120 + i * 5),
                performance_metrics={"runs": 70 + i * 8, "strike_rate": 150.0},
                context={"format": "t20", "venue": "home"},
                importance_score=1.2
            ))
        
        reference_date = base_date + timedelta(days=150)
        
        # Test different decay functions
        exponential_result = engine.get_weighted_aggregation(
            events, "runs", DecayType.EXPONENTIAL, "weighted_mean"
        )
        
        linear_result = engine.get_weighted_aggregation(
            events, "runs", DecayType.LINEAR, "weighted_mean"
        )
        
        trend_result = engine.get_weighted_aggregation(
            events, "runs", DecayType.EXPONENTIAL, "recent_trend"
        )
        
        # Recent form should dominate with temporal decay
        assert exponential_result["value"] > 50  # Should be influenced by recent good form
        assert trend_result["value"] > 0  # Positive trend due to recent improvement
        
        # Should have reasonable confidence
        assert exponential_result["confidence"] > 0.5
        assert linear_result["confidence"] > 0.5
    
    def test_tournament_importance_weighting(self):
        """Test that tournament importance affects weighting appropriately"""
        engine = TemporalDecayEngine()
        
        base_date = datetime(2024, 1, 1)
        
        # Regular match performance
        regular_event = PerformanceEvent(
            event_id="regular",
            player_id="player",
            date=base_date,
            performance_metrics={"runs": 50},
            context={"format": "t20"},
            importance_score=1.0
        )
        
        # Final match performance (same date, same performance, higher importance)
        final_event = PerformanceEvent(
            event_id="final",
            player_id="player",
            date=base_date,
            performance_metrics={"runs": 50},
            context={"format": "t20", "tournament_stage": "final"},
            importance_score=3.0
        )
        
        weighted_events = engine.calculate_temporal_weights([regular_event, final_event])
        
        regular_weight = next(w for e, w in weighted_events if e.event_id == "regular")
        final_weight = next(w for e, w in weighted_events if e.event_id == "final")
        
        # Final should have significantly higher weight
        assert final_weight > regular_weight * 2.5  # Should be close to 3x
    
    def test_context_aware_format_differences(self):
        """Test that different formats have appropriate decay rates"""
        engine = TemporalDecayEngine()
        
        base_date = datetime(2024, 1, 1)
        reference_date = base_date + timedelta(days=365)  # 1 year later
        
        # Same performance in different formats
        t20_event = PerformanceEvent(
            event_id="t20",
            player_id="player",
            date=base_date,
            performance_metrics={"runs": 50},
            context={"format": "t20"}
        )
        
        test_event = PerformanceEvent(
            event_id="test",
            player_id="player", 
            date=base_date,
            performance_metrics={"runs": 50},
            context={"format": "test"}
        )
        
        # Calculate weights with context-aware decay
        t20_weighted = engine.calculate_temporal_weights(
            [t20_event], reference_date, DecayType.CONTEXT_AWARE
        )
        test_weighted = engine.calculate_temporal_weights(
            [test_event], reference_date, DecayType.CONTEXT_AWARE
        )
        
        t20_weight = t20_weighted[0][1]
        test_weight = test_weighted[0][1]
        
        # Test performances should remain relevant longer than T20
        assert test_weight > t20_weight
