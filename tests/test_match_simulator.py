# Purpose: Unit tests for Match Simulator with counterfactual analysis
# Author: Shamus Rae, Last Modified: 2024-01-15

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime

from crickformers.match_simulator import (
    CounterfactualEventType,
    CounterfactualEvent,
    SimulationContext,
    PredictionResult,
    SimulationComparison,
    CrickFormerInferenceWrapper,
    MockCrickFormerModel,
    CounterfactualEventGenerator,
    MatchSimulator,
    create_simulation_context_from_match_state
)


class TestCounterfactualEvent:
    """Test counterfactual event creation and validation."""
    
    def test_event_creation(self):
        """Test creating a counterfactual event."""
        original_outcome = {'runs_scored': 1, 'wicket_type': None}
        modified_outcome = {'runs_scored': 0, 'wicket_type': 'caught'}
        
        event = CounterfactualEvent(
            event_type=CounterfactualEventType.CATCH_TAKEN,
            ball_id="test_ball_1",
            original_outcome=original_outcome,
            modified_outcome=modified_outcome,
            description="Test catch taken",
            confidence=0.8
        )
        
        assert event.event_type == CounterfactualEventType.CATCH_TAKEN
        assert event.ball_id == "test_ball_1"
        assert event.original_outcome == original_outcome
        assert event.modified_outcome == modified_outcome
        assert event.description == "Test catch taken"
        assert event.confidence == 0.8
    
    def test_event_validation(self):
        """Test event validation."""
        original_outcome = {'runs_scored': 1, 'wicket_type': None}
        modified_outcome = {'runs_scored': 0, 'wicket_type': 'caught'}
        
        # Invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            CounterfactualEvent(
                event_type=CounterfactualEventType.CATCH_TAKEN,
                ball_id="test_ball_1",
                original_outcome=original_outcome,
                modified_outcome=modified_outcome,
                description="Test",
                confidence=1.5
            )
        
        # Empty ball ID
        with pytest.raises(ValueError, match="Ball ID cannot be empty"):
            CounterfactualEvent(
                event_type=CounterfactualEventType.CATCH_TAKEN,
                ball_id="",
                original_outcome=original_outcome,
                modified_outcome=modified_outcome,
                description="Test"
            )


class TestSimulationContext:
    """Test simulation context creation and methods."""
    
    def test_context_creation(self):
        """Test creating simulation context."""
        context = SimulationContext(
            match_id="test_match",
            innings=1,
            over=10.0,
            ball=3,
            current_score=85,
            wickets_fallen=2,
            balls_remaining=78,
            target_score=None,
            current_batter="Kohli",
            current_bowler="Starc"
        )
        
        assert context.match_id == "test_match"
        assert context.innings == 1
        assert context.over == 10.0
        assert context.current_score == 85
        assert context.wickets_fallen == 2
        assert context.balls_remaining == 78
        assert context.current_batter == "Kohli"
    
    def test_context_to_dict(self):
        """Test converting context to dictionary."""
        context = SimulationContext(
            match_id="test_match",
            innings=2,
            over=15.0,
            ball=4,
            current_score=120,
            wickets_fallen=4,
            balls_remaining=32,
            target_score=180,
            required_run_rate=11.25
        )
        
        context_dict = context.to_dict()
        
        assert isinstance(context_dict, dict)
        assert context_dict['match_id'] == "test_match"
        assert context_dict['target_score'] == 180
        assert context_dict['required_run_rate'] == 11.25


class TestPredictionResult:
    """Test prediction result creation and validation."""
    
    def test_prediction_creation(self):
        """Test creating prediction result."""
        prediction = PredictionResult(
            win_probability=0.65,
            over_runs_prediction=8.5,
            next_ball_runs=1,
            wicket_probability=0.12,
            boundary_probability=0.25,
            runs_distribution={0: 0.3, 1: 0.4, 2: 0.15, 4: 0.1, 6: 0.05},
            dismissal_type_probs={'caught': 0.4, 'bowled': 0.3, 'lbw': 0.3},
            prediction_confidence=0.78,
            model_uncertainty=0.15
        )
        
        assert prediction.win_probability == 0.65
        assert prediction.over_runs_prediction == 8.5
        assert prediction.next_ball_runs == 1
        assert prediction.prediction_confidence == 0.78
        assert len(prediction.runs_distribution) == 5
        assert len(prediction.dismissal_type_probs) == 3
    
    def test_prediction_validation(self):
        """Test prediction validation."""
        # Invalid win probability
        with pytest.raises(ValueError, match="Win probability must be between 0 and 1"):
            PredictionResult(
                win_probability=1.5,
                over_runs_prediction=8.5,
                next_ball_runs=1,
                wicket_probability=0.12,
                boundary_probability=0.25,
                runs_distribution={},
                dismissal_type_probs={},
                prediction_confidence=0.78,
                model_uncertainty=0.15
            )
        
        # Invalid confidence
        with pytest.raises(ValueError, match="Prediction confidence must be between 0 and 1"):
            PredictionResult(
                win_probability=0.65,
                over_runs_prediction=8.5,
                next_ball_runs=1,
                wicket_probability=0.12,
                boundary_probability=0.25,
                runs_distribution={},
                dismissal_type_probs={},
                prediction_confidence=1.2,
                model_uncertainty=0.15
            )


class TestSimulationComparison:
    """Test simulation comparison calculations."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        original = PredictionResult(
            win_probability=0.60,
            over_runs_prediction=7.0,
            next_ball_runs=1,
            wicket_probability=0.10,
            boundary_probability=0.20,
            runs_distribution={0: 0.4, 1: 0.35, 2: 0.15, 4: 0.07, 6: 0.03},
            dismissal_type_probs={'caught': 0.5, 'bowled': 0.3, 'lbw': 0.2},
            prediction_confidence=0.75,
            model_uncertainty=0.12
        )
        
        counterfactual = PredictionResult(
            win_probability=0.45,
            over_runs_prediction=5.5,
            next_ball_runs=0,
            wicket_probability=0.25,
            boundary_probability=0.15,
            runs_distribution={0: 0.6, 1: 0.25, 2: 0.1, 4: 0.04, 6: 0.01},
            dismissal_type_probs={'caught': 0.6, 'bowled': 0.25, 'lbw': 0.15},
            prediction_confidence=0.80,
            model_uncertainty=0.10
        )
        
        return original, counterfactual
    
    @pytest.fixture
    def sample_event(self):
        """Create sample counterfactual event."""
        return CounterfactualEvent(
            event_type=CounterfactualEventType.CATCH_TAKEN,
            ball_id="test_ball",
            original_outcome={'runs_scored': 1, 'wicket_type': None},
            modified_outcome={'runs_scored': 0, 'wicket_type': 'caught'},
            description="Test catch taken"
        )
    
    @pytest.fixture
    def sample_context(self):
        """Create sample simulation context."""
        return SimulationContext(
            match_id="test_match",
            innings=1,
            over=10.0,
            ball=3,
            current_score=85,
            wickets_fallen=2,
            balls_remaining=78
        )
    
    def test_comparison_creation(self, sample_predictions, sample_event, sample_context):
        """Test creating simulation comparison."""
        original, counterfactual = sample_predictions
        
        comparison = SimulationComparison(
            original_prediction=original,
            counterfactual_prediction=counterfactual,
            event=sample_event,
            context=sample_context,
            win_probability_impact=0.0,  # Will be calculated
            score_impact=0.0,
            wicket_impact=0.0,
            impact_magnitude="",
            confidence_interval=(0.0, 0.0)
        )
        
        # Check calculated impacts (with floating point tolerance)
        assert abs(comparison.win_probability_impact - (-0.15)) < 1e-10  # 0.45 - 0.60
        assert comparison.score_impact == -1.5  # 5.5 - 7.0
        assert comparison.wicket_impact == 0.15  # 0.25 - 0.10
        assert comparison.impact_magnitude == "large"  # Max impact > 0.15
    
    def test_impact_magnitude_calculation(self, sample_context, sample_event):
        """Test impact magnitude classification."""
        # Negligible impact
        original = PredictionResult(0.50, 7.0, 1, 0.10, 0.20, {}, {}, 0.75, 0.12)
        counterfactual = PredictionResult(0.51, 7.1, 1, 0.11, 0.21, {}, {}, 0.75, 0.12)
        
        comparison = SimulationComparison(
            original, counterfactual, sample_event, sample_context,
            0.0, 0.0, 0.0, "", (0.0, 0.0)
        )
        assert comparison.impact_magnitude == "negligible"
        
        # Small impact
        original = PredictionResult(0.50, 7.0, 1, 0.10, 0.20, {}, {}, 0.75, 0.12)
        counterfactual = PredictionResult(0.53, 7.3, 1, 0.13, 0.23, {}, {}, 0.75, 0.12)
        
        comparison = SimulationComparison(
            original, counterfactual, sample_event, sample_context,
            0.0, 0.0, 0.0, "", (0.0, 0.0)
        )
        assert comparison.impact_magnitude == "small"
        
        # Medium impact
        original = PredictionResult(0.50, 7.0, 1, 0.10, 0.20, {}, {}, 0.75, 0.12)
        counterfactual = PredictionResult(0.60, 8.0, 1, 0.20, 0.30, {}, {}, 0.75, 0.12)
        
        comparison = SimulationComparison(
            original, counterfactual, sample_event, sample_context,
            0.0, 0.0, 0.0, "", (0.0, 0.0)
        )
        assert comparison.impact_magnitude == "medium"
        
        # Large impact
        original = PredictionResult(0.50, 7.0, 1, 0.10, 0.20, {}, {}, 0.75, 0.12)
        counterfactual = PredictionResult(0.75, 10.0, 1, 0.30, 0.40, {}, {}, 0.75, 0.12)
        
        comparison = SimulationComparison(
            original, counterfactual, sample_event, sample_context,
            0.0, 0.0, 0.0, "", (0.0, 0.0)
        )
        assert comparison.impact_magnitude == "large"


class TestMockCrickFormerModel:
    """Test mock CrickFormer model."""
    
    def test_model_prediction(self):
        """Test mock model prediction."""
        model = MockCrickFormerModel()
        
        features = {
            'current_score': 100,
            'wickets_fallen': 3,
            'balls_remaining': 60,
            'target_score': 180,
            'recent_run_rate': 8.0
        }
        
        prediction = model.predict(features)
        
        assert isinstance(prediction, dict)
        assert 'win_probability' in prediction
        assert 'over_runs' in prediction
        assert 'next_ball_runs' in prediction
        assert 'wicket_probability' in prediction
        assert 'boundary_probability' in prediction
        assert 'runs_distribution' in prediction
        assert 'dismissal_type_probs' in prediction
        
        # Check value ranges
        assert 0 <= prediction['win_probability'] <= 1
        assert prediction['over_runs'] >= 0
        assert prediction['next_ball_runs'] in [0, 1, 2, 3, 4, 6]
        assert 0 <= prediction['wicket_probability'] <= 1
        assert 0 <= prediction['boundary_probability'] <= 1
    
    def test_model_prediction_variability(self):
        """Test that model produces variable predictions."""
        model = MockCrickFormerModel()
        
        features = {
            'current_score': 100,
            'wickets_fallen': 3,
            'balls_remaining': 60,
            'recent_run_rate': 8.0
        }
        
        # Generate multiple predictions
        predictions = [model.predict(features) for _ in range(10)]
        
        # Check that predictions vary
        win_probs = [p['win_probability'] for p in predictions]
        assert len(set(win_probs)) > 1  # Should have some variation


class TestCrickFormerInferenceWrapper:
    """Test CrickFormer inference wrapper."""
    
    @pytest.fixture
    def wrapper(self):
        """Create inference wrapper."""
        return CrickFormerInferenceWrapper()
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context."""
        return SimulationContext(
            match_id="test_match",
            innings=1,
            over=10.0,
            ball=3,
            current_score=85,
            wickets_fallen=2,
            balls_remaining=78
        )
    
    @pytest.fixture
    def sample_ball_sequence(self):
        """Create sample ball sequence."""
        return pd.DataFrame([
            {'ball_id': 'ball_1', 'runs_scored': 1, 'wicket_type': None},
            {'ball_id': 'ball_2', 'runs_scored': 4, 'wicket_type': None},
            {'ball_id': 'ball_3', 'runs_scored': 0, 'wicket_type': 'bowled'},
            {'ball_id': 'ball_4', 'runs_scored': 2, 'wicket_type': None},
            {'ball_id': 'ball_5', 'runs_scored': 1, 'wicket_type': None}
        ])
    
    def test_wrapper_initialization(self, wrapper):
        """Test wrapper initialization."""
        assert wrapper.model is not None
        assert isinstance(wrapper.model, MockCrickFormerModel)
    
    def test_prediction_without_modifications(self, wrapper, sample_context, sample_ball_sequence):
        """Test prediction without modifications."""
        result = wrapper.predict(sample_context, sample_ball_sequence)
        
        assert isinstance(result, PredictionResult)
        assert 0 <= result.win_probability <= 1
        assert result.over_runs_prediction >= 0
        assert result.next_ball_runs in [0, 1, 2, 3, 4, 6]
    
    def test_prediction_with_modifications(self, wrapper, sample_context, sample_ball_sequence):
        """Test prediction with modifications."""
        modifications = {
            'ball_3': {'runs_scored': 0, 'wicket_type': None}  # Remove wicket
        }
        
        result = wrapper.predict(sample_context, sample_ball_sequence, modifications)
        
        assert isinstance(result, PredictionResult)
        assert 0 <= result.win_probability <= 1
    
    def test_apply_modifications(self, wrapper, sample_ball_sequence):
        """Test applying modifications to ball sequence."""
        modifications = {
            'ball_2': {'runs_scored': 6, 'wicket_type': None},
            'ball_3': {'runs_scored': 1, 'wicket_type': None}
        }
        
        modified_sequence = wrapper._apply_modifications(sample_ball_sequence, modifications)
        
        # Check modifications were applied
        ball_2_row = modified_sequence[modified_sequence['ball_id'] == 'ball_2'].iloc[0]
        assert ball_2_row['runs_scored'] == 6
        
        ball_3_row = modified_sequence[modified_sequence['ball_id'] == 'ball_3'].iloc[0]
        assert ball_3_row['runs_scored'] == 1
        assert ball_3_row['wicket_type'] is None
    
    def test_feature_extraction(self, wrapper, sample_context, sample_ball_sequence):
        """Test feature extraction."""
        features = wrapper._extract_features(sample_context, sample_ball_sequence)
        
        assert isinstance(features, dict)
        assert 'current_score' in features
        assert 'wickets_fallen' in features
        assert 'balls_remaining' in features
        assert 'recent_run_rate' in features
        assert 'recent_wicket_rate' in features
        assert 'boundary_percentage' in features
        
        # Check calculated features
        assert features['current_score'] == 85
        assert features['wickets_fallen'] == 2
        assert features['balls_remaining'] == 78
    
    def test_recent_run_rate_calculation(self, wrapper):
        """Test recent run rate calculation."""
        ball_sequence = pd.DataFrame([
            {'runs_scored': 1}, {'runs_scored': 4}, {'runs_scored': 0},
            {'runs_scored': 2}, {'runs_scored': 1}, {'runs_scored': 6}
        ])
        
        run_rate = wrapper._calculate_recent_run_rate(ball_sequence)
        
        # Total runs = 14, balls = 6, rate = (14/6) * 6 = 14
        assert run_rate == 14.0
    
    def test_recent_wicket_rate_calculation(self, wrapper):
        """Test recent wicket rate calculation."""
        ball_sequence = pd.DataFrame([
            {'wicket_type': None}, {'wicket_type': 'caught'}, {'wicket_type': None},
            {'wicket_type': None}, {'wicket_type': 'bowled'}, {'wicket_type': None}
        ])
        
        wicket_rate = wrapper._calculate_recent_wicket_rate(ball_sequence)
        
        # 2 wickets in 6 balls = 2/6 = 0.333...
        assert abs(wicket_rate - (2/6)) < 0.001
    
    def test_boundary_percentage_calculation(self, wrapper):
        """Test boundary percentage calculation."""
        ball_sequence = pd.DataFrame([
            {'runs_scored': 1}, {'runs_scored': 4}, {'runs_scored': 0},
            {'runs_scored': 2}, {'runs_scored': 6}, {'runs_scored': 1}
        ])
        
        boundary_pct = wrapper._calculate_boundary_percentage(ball_sequence)
        
        # 2 boundaries (4 and 6) in 6 balls = 2/6 = 0.333...
        assert abs(boundary_pct - (2/6)) < 0.001


class TestCounterfactualEventGenerator:
    """Test counterfactual event generator."""
    
    def test_catch_taken_scenario(self):
        """Test creating catch taken scenario."""
        original_outcome = {'runs_scored': 1, 'wicket_type': None}
        
        event = CounterfactualEventGenerator.create_catch_scenario(
            "test_ball", original_outcome, taken=True
        )
        
        assert event.event_type == CounterfactualEventType.CATCH_TAKEN
        assert event.ball_id == "test_ball"
        assert event.original_outcome == original_outcome
        assert event.modified_outcome['runs_scored'] == 0
        assert event.modified_outcome['wicket_type'] == 'caught'
        assert "Catch taken" in event.description
    
    def test_catch_dropped_scenario(self):
        """Test creating catch dropped scenario."""
        original_outcome = {'runs_scored': 0, 'wicket_type': 'caught'}
        
        event = CounterfactualEventGenerator.create_catch_scenario(
            "test_ball", original_outcome, taken=False
        )
        
        assert event.event_type == CounterfactualEventType.CATCH_DROPPED
        assert event.modified_outcome['wicket_type'] is None
        assert "Catch dropped" in event.description
    
    def test_boundary_hit_scenario(self):
        """Test creating boundary hit scenario."""
        original_outcome = {'runs_scored': 1, 'wicket_type': None}
        
        event = CounterfactualEventGenerator.create_boundary_scenario(
            "test_ball", original_outcome, boundary_hit=True
        )
        
        assert event.event_type == CounterfactualEventType.BOUNDARY_HIT
        assert event.modified_outcome['runs_scored'] == 4
        assert event.modified_outcome['wicket_type'] is None
        assert "Boundary hit" in event.description
    
    def test_boundary_stopped_scenario(self):
        """Test creating boundary stopped scenario."""
        original_outcome = {'runs_scored': 4, 'wicket_type': None}
        
        event = CounterfactualEventGenerator.create_boundary_scenario(
            "test_ball", original_outcome, boundary_hit=False
        )
        
        assert event.event_type == CounterfactualEventType.BOUNDARY_STOPPED
        assert event.modified_outcome['runs_scored'] <= 2
        assert "Boundary stopped" in event.description
    
    def test_run_out_successful_scenario(self):
        """Test creating successful run out scenario."""
        original_outcome = {'runs_scored': 2, 'wicket_type': None}
        
        event = CounterfactualEventGenerator.create_run_out_scenario(
            "test_ball", original_outcome, successful=True
        )
        
        assert event.event_type == CounterfactualEventType.RUN_OUT_SUCCESS
        assert event.modified_outcome['wicket_type'] == 'run_out'
        assert event.modified_outcome['runs_scored'] <= 1
        assert "Run out successful" in event.description
    
    def test_run_out_failed_scenario(self):
        """Test creating failed run out scenario."""
        original_outcome = {'runs_scored': 1, 'wicket_type': 'run_out'}
        
        event = CounterfactualEventGenerator.create_run_out_scenario(
            "test_ball", original_outcome, successful=False
        )
        
        assert event.event_type == CounterfactualEventType.RUN_OUT_FAILED
        assert event.modified_outcome['wicket_type'] is None
        assert "Run out failed" in event.description


class TestMatchSimulator:
    """Test main match simulator."""
    
    @pytest.fixture
    def simulator(self):
        """Create match simulator."""
        return MatchSimulator()
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context."""
        return SimulationContext(
            match_id="test_match",
            innings=1,
            over=10.0,
            ball=3,
            current_score=85,
            wickets_fallen=2,
            balls_remaining=78
        )
    
    @pytest.fixture
    def sample_ball_sequence(self):
        """Create sample ball sequence."""
        return pd.DataFrame([
            {'ball_id': 'ball_1', 'runs_scored': 1, 'wicket_type': None},
            {'ball_id': 'ball_2', 'runs_scored': 4, 'wicket_type': None},
            {'ball_id': 'ball_3', 'runs_scored': 0, 'wicket_type': 'bowled'},
            {'ball_id': 'ball_4', 'runs_scored': 2, 'wicket_type': None},
            {'ball_id': 'ball_5', 'runs_scored': 1, 'wicket_type': None}
        ])
    
    @pytest.fixture
    def sample_event(self):
        """Create sample counterfactual event."""
        return CounterfactualEvent(
            event_type=CounterfactualEventType.CATCH_TAKEN,
            ball_id="ball_2",
            original_outcome={'runs_scored': 4, 'wicket_type': None},
            modified_outcome={'runs_scored': 0, 'wicket_type': 'caught'},
            description="Boundary catch taken"
        )
    
    def test_simulator_initialization(self, simulator):
        """Test simulator initialization."""
        assert simulator.inference_wrapper is not None
        assert simulator.event_generator is not None
    
    def test_simulate_counterfactual(self, simulator, sample_context, sample_ball_sequence, sample_event):
        """Test simulating a counterfactual event."""
        comparison = simulator.simulate_counterfactual(
            sample_context, sample_ball_sequence, sample_event
        )
        
        assert isinstance(comparison, SimulationComparison)
        assert comparison.event == sample_event
        assert comparison.context == sample_context
        assert isinstance(comparison.original_prediction, PredictionResult)
        assert isinstance(comparison.counterfactual_prediction, PredictionResult)
        assert comparison.impact_magnitude in ['negligible', 'small', 'medium', 'large']
    
    def test_simulate_multiple_scenarios(self, simulator, sample_context, sample_ball_sequence):
        """Test simulating multiple scenarios."""
        events = [
            CounterfactualEvent(
                event_type=CounterfactualEventType.CATCH_TAKEN,
                ball_id="ball_2",
                original_outcome={'runs_scored': 4, 'wicket_type': None},
                modified_outcome={'runs_scored': 0, 'wicket_type': 'caught'},
                description="Catch taken"
            ),
            CounterfactualEvent(
                event_type=CounterfactualEventType.BOUNDARY_STOPPED,
                ball_id="ball_2",
                original_outcome={'runs_scored': 4, 'wicket_type': None},
                modified_outcome={'runs_scored': 2, 'wicket_type': None},
                description="Boundary stopped"
            )
        ]
        
        comparisons = simulator.simulate_multiple_scenarios(
            sample_context, sample_ball_sequence, events
        )
        
        assert len(comparisons) == 2
        assert all(isinstance(c, SimulationComparison) for c in comparisons)
    
    def test_generate_suggested_scenarios(self, simulator, sample_context, sample_ball_sequence):
        """Test generating suggested scenarios."""
        scenarios = simulator.generate_suggested_scenarios(
            sample_context, sample_ball_sequence, num_scenarios=3
        )
        
        assert len(scenarios) <= 3
        assert all(isinstance(s, CounterfactualEvent) for s in scenarios)
        
        # Check that scenarios are relevant to the ball sequence
        ball_ids = sample_ball_sequence['ball_id'].tolist()
        for scenario in scenarios:
            assert scenario.ball_id in ball_ids
    
    def test_analyze_scenario_impact(self, simulator):
        """Test analyzing scenario impact."""
        # Create mock comparisons
        comparisons = []
        
        for i in range(5):
            original = PredictionResult(0.5, 7.0, 1, 0.1, 0.2, {}, {}, 0.75, 0.12)
            counterfactual = PredictionResult(
                0.5 + (i-2)*0.05,  # Varying win probability
                7.0 + (i-2)*0.5,   # Varying score
                1, 0.1, 0.2, {}, {}, 0.75, 0.12
            )
            
            event = CounterfactualEvent(
                CounterfactualEventType.CATCH_TAKEN,
                f"ball_{i}",
                {'runs_scored': 1},
                {'runs_scored': 0, 'wicket_type': 'caught'},
                f"Event {i}"
            )
            
            context = SimulationContext("test", 1, 10.0, 1, 100, 2, 60)
            
            comparison = SimulationComparison(
                original, counterfactual, event, context,
                0.0, 0.0, 0.0, "", (0.0, 0.0)
            )
            comparisons.append(comparison)
        
        analysis = simulator.analyze_scenario_impact(comparisons)
        
        assert 'total_scenarios' in analysis
        assert 'avg_win_prob_impact' in analysis
        assert 'max_win_prob_impact' in analysis
        assert 'impact_distribution' in analysis
        assert analysis['total_scenarios'] == 5


class TestCreateSimulationContext:
    """Test simulation context creation from match data."""
    
    @pytest.fixture
    def sample_match_data(self):
        """Create sample match data."""
        return pd.DataFrame([
            {
                'ball_id': 'ball_1', 'match_id': 'match1', 'innings': 1,
                'over': 1, 'ball': 1, 'runs_scored': 1, 'wicket_type': None,
                'batter': 'Player1', 'bowler': 'Bowler1'
            },
            {
                'ball_id': 'ball_2', 'match_id': 'match1', 'innings': 1,
                'over': 1, 'ball': 2, 'runs_scored': 4, 'wicket_type': None,
                'batter': 'Player1', 'bowler': 'Bowler1'
            },
            {
                'ball_id': 'ball_3', 'match_id': 'match1', 'innings': 1,
                'over': 1, 'ball': 3, 'runs_scored': 0, 'wicket_type': 'bowled',
                'batter': 'Player1', 'bowler': 'Bowler1'
            },
            {
                'ball_id': 'ball_4', 'match_id': 'match1', 'innings': 1,
                'over': 1, 'ball': 4, 'runs_scored': 2, 'wicket_type': None,
                'batter': 'Player2', 'bowler': 'Bowler1'
            }
        ])
    
    def test_create_context_from_latest_ball(self, sample_match_data):
        """Test creating context from latest ball."""
        context = create_simulation_context_from_match_state(sample_match_data)
        
        assert context.match_id == 'match1'
        assert context.innings == 1
        assert context.over == 1
        assert context.ball == 4
        assert context.current_score == 7  # 1 + 4 + 0 + 2
        assert context.wickets_fallen == 1  # One wicket
        assert context.balls_remaining == 116  # 120 - 4 balls
        assert context.current_batter == 'Player2'
        assert context.current_bowler == 'Bowler1'
    
    def test_create_context_from_specific_ball(self, sample_match_data):
        """Test creating context from specific ball."""
        context = create_simulation_context_from_match_state(
            sample_match_data, target_ball='ball_2'
        )
        
        assert context.match_id == 'match1'
        assert context.innings == 1
        assert context.over == 1
        assert context.ball == 2
        assert context.current_score == 5  # 1 + 4 (up to ball_2)
        assert context.wickets_fallen == 0  # No wickets yet
        assert context.balls_remaining == 118  # 120 - 2 balls
    
    def test_create_context_empty_data(self):
        """Test creating context with empty data."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Match data cannot be empty"):
            create_simulation_context_from_match_state(empty_data)
    
    def test_create_context_invalid_ball_id(self, sample_match_data):
        """Test creating context with invalid ball ID."""
        with pytest.raises(ValueError, match="Ball ID nonexistent not found"):
            create_simulation_context_from_match_state(
                sample_match_data, target_ball='nonexistent'
            )
    
    def test_create_context_second_innings(self):
        """Test creating context for second innings."""
        match_data = pd.DataFrame([
            # First innings
            {'ball_id': 'ball_1', 'match_id': 'match1', 'innings': 1,
             'over': 20, 'ball': 6, 'runs_scored': 6, 'wicket_type': None},
            # Second innings
            {'ball_id': 'ball_2', 'match_id': 'match1', 'innings': 2,
             'over': 5, 'ball': 3, 'runs_scored': 4, 'wicket_type': None}
        ])
        
        context = create_simulation_context_from_match_state(match_data)
        
        assert context.innings == 2
        assert context.target_score == 7  # First innings score + 1
        assert context.required_run_rate is not None
        assert context.required_run_rate > 0


class TestIntegration:
    """Integration tests for the complete simulation system."""
    
    def test_end_to_end_simulation(self):
        """Test complete end-to-end simulation."""
        # Create match data
        match_data = pd.DataFrame([
            {'ball_id': f'ball_{i}', 'match_id': 'test_match', 'innings': 1,
             'over': (i // 6) + 1, 'ball': (i % 6) + 1,
             'runs_scored': np.random.choice([0, 1, 2, 4, 6], p=[0.4, 0.3, 0.15, 0.1, 0.05]),
             'wicket_type': 'caught' if i % 20 == 0 else None,
             'batter': 'Player1', 'bowler': 'Bowler1'}
            for i in range(30)  # 5 overs
        ])
        
        # Create simulation context
        context = create_simulation_context_from_match_state(match_data)
        
        # Create simulator
        simulator = MatchSimulator()
        
        # Generate suggested scenarios
        scenarios = simulator.generate_suggested_scenarios(context, match_data, 3)
        
        assert len(scenarios) <= 3
        
        # Simulate scenarios
        if scenarios:
            comparisons = simulator.simulate_multiple_scenarios(context, match_data, scenarios)
            
            assert len(comparisons) <= len(scenarios)
            assert all(isinstance(c, SimulationComparison) for c in comparisons)
            
            # Analyze impact
            analysis = simulator.analyze_scenario_impact(comparisons)
            
            assert 'total_scenarios' in analysis
            assert analysis['total_scenarios'] == len(comparisons)
    
    def test_consistency_across_runs(self):
        """Test that simulation results are consistent across multiple runs."""
        # Create deterministic match data
        np.random.seed(42)
        
        match_data = pd.DataFrame([
            {'ball_id': 'ball_1', 'match_id': 'test', 'innings': 1,
             'over': 1, 'ball': 1, 'runs_scored': 1, 'wicket_type': None},
            {'ball_id': 'ball_2', 'match_id': 'test', 'innings': 1,
             'over': 1, 'ball': 2, 'runs_scored': 4, 'wicket_type': None}
        ])
        
        context = create_simulation_context_from_match_state(match_data)
        
        event = CounterfactualEvent(
            CounterfactualEventType.CATCH_TAKEN,
            'ball_2',
            {'runs_scored': 4, 'wicket_type': None},
            {'runs_scored': 0, 'wicket_type': 'caught'},
            "Test catch"
        )
        
        simulator = MatchSimulator()
        
        # Run simulation multiple times with same seed
        results = []
        for _ in range(3):
            np.random.seed(42)  # Reset seed for consistency
            comparison = simulator.simulate_counterfactual(context, match_data, event)
            results.append(comparison)
        
        # Results should be similar (allowing for some model randomness)
        win_probs = [r.original_prediction.win_probability for r in results]
        assert max(win_probs) - min(win_probs) < 0.1  # Within 10% variation
    
    def test_api_compatibility(self):
        """Test API compatibility for UI integration."""
        # Test that all main classes can be imported and instantiated
        simulator = MatchSimulator()
        generator = CounterfactualEventGenerator()
        wrapper = CrickFormerInferenceWrapper()
        
        assert simulator is not None
        assert generator is not None
        assert wrapper is not None
        
        # Test that main methods exist and are callable
        assert hasattr(simulator, 'simulate_counterfactual')
        assert hasattr(simulator, 'simulate_multiple_scenarios')
        assert hasattr(simulator, 'generate_suggested_scenarios')
        assert hasattr(generator, 'create_catch_scenario')
        assert hasattr(wrapper, 'predict')
        
        # Test that key data classes can be created
        context = SimulationContext("test", 1, 10.0, 1, 100, 2, 60)
        assert context is not None
        
        prediction = PredictionResult(0.5, 7.0, 1, 0.1, 0.2, {}, {}, 0.75, 0.12)
        assert prediction is not None