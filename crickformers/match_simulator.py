# Purpose: Match Simulator with counterfactual analysis capabilities
# Author: Shamus Rae, Last Modified: 2024-01-15

"""
This module implements a comprehensive Match Simulator that allows users to
explore counterfactual scenarios by modifying specific events (e.g., catches
taken vs dropped) and comparing the resulting predictions side-by-side.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import copy
from datetime import datetime

logger = logging.getLogger(__name__)


class CounterfactualEventType(Enum):
    """Types of counterfactual events that can be simulated."""
    
    CATCH_TAKEN = "catch_taken"
    CATCH_DROPPED = "catch_dropped"
    WICKET_TAKEN = "wicket_taken"
    WICKET_MISSED = "wicket_missed"
    BOUNDARY_HIT = "boundary_hit"
    BOUNDARY_STOPPED = "boundary_stopped"
    RUN_OUT_SUCCESS = "run_out_success"
    RUN_OUT_FAILED = "run_out_failed"
    NO_BALL_CALLED = "no_ball_called"
    WIDE_CALLED = "wide_called"
    REVIEW_OVERTURNED = "review_overturned"
    REVIEW_UPHELD = "review_upheld"


@dataclass
class CounterfactualEvent:
    """Represents a counterfactual event to be simulated."""
    
    event_type: CounterfactualEventType
    ball_id: str                      # Unique identifier for the ball
    original_outcome: Dict[str, Any]  # Original ball outcome
    modified_outcome: Dict[str, Any]  # Modified ball outcome
    description: str                  # Human-readable description
    confidence: float = 1.0           # Confidence in the modification (0-1)
    
    def __post_init__(self):
        """Validate the counterfactual event."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        if not self.ball_id:
            raise ValueError("Ball ID cannot be empty")


@dataclass
class SimulationContext:
    """Context for match simulation including current state."""
    
    match_id: str
    innings: int
    over: float
    ball: int
    current_score: int
    wickets_fallen: int
    balls_remaining: int
    target_score: Optional[int] = None
    required_run_rate: Optional[float] = None
    
    # Player context
    current_batter: str = ""
    current_bowler: str = ""
    captain: str = ""
    
    # Match conditions
    venue: str = ""
    conditions: str = "normal"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model input."""
        return asdict(self)


@dataclass
class PredictionResult:
    """Results from CrickFormer model prediction."""
    
    win_probability: float
    over_runs_prediction: float
    next_ball_runs: int
    wicket_probability: float
    boundary_probability: float
    
    # Detailed predictions
    runs_distribution: Dict[int, float]  # Probability distribution for runs (0-6)
    dismissal_type_probs: Dict[str, float]  # Probability by dismissal type
    
    # Confidence metrics
    prediction_confidence: float
    model_uncertainty: float
    
    # Additional context
    regime_prediction: str = ""
    pressure_score: float = 0.0
    
    def __post_init__(self):
        """Validate prediction results."""
        if not 0 <= self.win_probability <= 1:
            raise ValueError("Win probability must be between 0 and 1")
        
        if not 0 <= self.prediction_confidence <= 1:
            raise ValueError("Prediction confidence must be between 0 and 1")


@dataclass
class SimulationComparison:
    """Side-by-side comparison of original vs counterfactual outcomes."""
    
    original_prediction: PredictionResult
    counterfactual_prediction: PredictionResult
    event: CounterfactualEvent
    context: SimulationContext
    
    # Impact analysis
    win_probability_impact: float
    score_impact: float
    wicket_impact: float
    
    # Statistical significance
    impact_magnitude: str  # "negligible", "small", "medium", "large"
    confidence_interval: Tuple[float, float]
    
    def __post_init__(self):
        """Calculate impact metrics."""
        self.win_probability_impact = (
            self.counterfactual_prediction.win_probability - 
            self.original_prediction.win_probability
        )
        
        self.score_impact = (
            self.counterfactual_prediction.over_runs_prediction - 
            self.original_prediction.over_runs_prediction
        )
        
        self.wicket_impact = (
            self.counterfactual_prediction.wicket_probability - 
            self.original_prediction.wicket_probability
        )
        
        # Determine impact magnitude
        max_impact = max(
            abs(self.win_probability_impact),
            abs(self.score_impact) / 10.0,  # Normalize score impact
            abs(self.wicket_impact)
        )
        
        if max_impact < 0.02:
            self.impact_magnitude = "negligible"
        elif max_impact < 0.05:
            self.impact_magnitude = "small"
        elif max_impact < 0.15:
            self.impact_magnitude = "medium"
        else:
            self.impact_magnitude = "large"


class CrickFormerInferenceWrapper:
    """Wrapper for CrickFormer model inference with counterfactual support."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the inference wrapper.
        
        Args:
            model_path: Path to trained CrickFormer model
        """
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the CrickFormer model."""
        # Mock model loading for now - in production this would load actual model
        logger.info(f"Loading CrickFormer model from {self.model_path or 'default path'}")
        self.model = MockCrickFormerModel()
    
    def predict(
        self,
        context: SimulationContext,
        ball_sequence: pd.DataFrame,
        modified_outcomes: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> PredictionResult:
        """
        Make predictions using CrickFormer model.
        
        Args:
            context: Current match context
            ball_sequence: Recent ball-by-ball data
            modified_outcomes: Dictionary of ball_id -> modified outcome
        
        Returns:
            Prediction results
        """
        # Apply modifications to ball sequence if provided
        if modified_outcomes:
            ball_sequence = self._apply_modifications(ball_sequence, modified_outcomes)
        
        # Extract features for model input
        features = self._extract_features(context, ball_sequence)
        
        # Run model inference
        predictions = self.model.predict(features)
        
        return self._parse_predictions(predictions, context)
    
    def _apply_modifications(
        self,
        ball_sequence: pd.DataFrame,
        modifications: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Apply counterfactual modifications to ball sequence."""
        modified_sequence = ball_sequence.copy()
        
        for ball_id, modifications_dict in modifications.items():
            # Find the ball to modify
            mask = modified_sequence['ball_id'] == ball_id
            
            if not mask.any():
                logger.warning(f"Ball ID {ball_id} not found in sequence")
                continue
            
            # Apply modifications
            for column, new_value in modifications_dict.items():
                if column in modified_sequence.columns:
                    modified_sequence.loc[mask, column] = new_value
                    logger.debug(f"Modified {column} to {new_value} for ball {ball_id}")
        
        return modified_sequence
    
    def _extract_features(
        self,
        context: SimulationContext,
        ball_sequence: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract features for model input."""
        features = {
            # Context features
            'current_score': context.current_score,
            'wickets_fallen': context.wickets_fallen,
            'balls_remaining': context.balls_remaining,
            'target_score': context.target_score or 0,
            'required_run_rate': context.required_run_rate or 0.0,
            'over': context.over,
            
            # Recent performance features
            'recent_run_rate': self._calculate_recent_run_rate(ball_sequence),
            'recent_wicket_rate': self._calculate_recent_wicket_rate(ball_sequence),
            'boundary_percentage': self._calculate_boundary_percentage(ball_sequence),
            
            # Player features (simplified)
            'batter_form': 0.7,  # Mock values
            'bowler_form': 0.6,
            
            # Sequence length
            'sequence_length': len(ball_sequence)
        }
        
        return features
    
    def _calculate_recent_run_rate(self, ball_sequence: pd.DataFrame) -> float:
        """Calculate recent run rate from ball sequence."""
        if len(ball_sequence) == 0:
            return 6.0  # Default run rate
        
        recent_balls = ball_sequence.tail(12)  # Last 2 overs
        total_runs = recent_balls['runs_scored'].sum()
        return (total_runs / len(recent_balls)) * 6.0
    
    def _calculate_recent_wicket_rate(self, ball_sequence: pd.DataFrame) -> float:
        """Calculate recent wicket rate from ball sequence."""
        if len(ball_sequence) == 0:
            return 0.1
        
        recent_balls = ball_sequence.tail(18)  # Last 3 overs
        wickets = recent_balls['wicket_type'].notna().sum()
        return wickets / len(recent_balls)
    
    def _calculate_boundary_percentage(self, ball_sequence: pd.DataFrame) -> float:
        """Calculate boundary percentage from ball sequence."""
        if len(ball_sequence) == 0:
            return 0.15
        
        recent_balls = ball_sequence.tail(24)  # Last 4 overs
        boundaries = (recent_balls['runs_scored'] >= 4).sum()
        return boundaries / len(recent_balls)
    
    def _parse_predictions(
        self,
        raw_predictions: Dict[str, Any],
        context: SimulationContext
    ) -> PredictionResult:
        """Parse raw model predictions into structured result."""
        return PredictionResult(
            win_probability=raw_predictions['win_probability'],
            over_runs_prediction=raw_predictions['over_runs'],
            next_ball_runs=raw_predictions['next_ball_runs'],
            wicket_probability=raw_predictions['wicket_probability'],
            boundary_probability=raw_predictions['boundary_probability'],
            runs_distribution=raw_predictions['runs_distribution'],
            dismissal_type_probs=raw_predictions['dismissal_type_probs'],
            prediction_confidence=raw_predictions['confidence'],
            model_uncertainty=raw_predictions['uncertainty'],
            regime_prediction=raw_predictions.get('regime', 'unknown'),
            pressure_score=raw_predictions.get('pressure_score', 0.0)
        )


class MockCrickFormerModel:
    """Mock CrickFormer model for demonstration purposes."""
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock predictions based on features."""
        # Base predictions influenced by context
        current_score = features.get('current_score', 100)
        wickets_fallen = features.get('wickets_fallen', 3)
        balls_remaining = features.get('balls_remaining', 60)
        target_score = features.get('target_score', 0)
        
        # Calculate base win probability
        if target_score > 0:  # Chasing
            runs_needed = target_score - current_score
            required_rr = (runs_needed / balls_remaining) * 6 if balls_remaining > 0 else 20
            win_prob = max(0.1, min(0.9, 1.0 - (required_rr - 6) / 10))
        else:  # Batting first
            projected_score = current_score + (balls_remaining / 6) * features.get('recent_run_rate', 6)
            win_prob = min(0.85, max(0.15, projected_score / 200))
        
        # Add some realistic noise
        win_prob += np.random.normal(0, 0.05)
        win_prob = max(0.01, min(0.99, win_prob))
        
        # Other predictions
        over_runs = max(0, features.get('recent_run_rate', 6) + np.random.normal(0, 2))
        next_ball_runs = np.random.choice([0, 1, 2, 3, 4, 6], p=[0.4, 0.25, 0.15, 0.05, 0.1, 0.05])
        
        wicket_prob = min(0.3, max(0.02, 0.08 + (wickets_fallen / 10) * 0.1))
        boundary_prob = min(0.4, max(0.05, features.get('boundary_percentage', 0.15)))
        
        return {
            'win_probability': win_prob,
            'over_runs': over_runs,
            'next_ball_runs': next_ball_runs,
            'wicket_probability': wicket_prob,
            'boundary_probability': boundary_prob,
            'runs_distribution': {
                0: 0.4, 1: 0.25, 2: 0.15, 3: 0.05, 4: 0.1, 6: 0.05
            },
            'dismissal_type_probs': {
                'caught': 0.4, 'bowled': 0.25, 'lbw': 0.2, 'run_out': 0.1, 'stumped': 0.05
            },
            'confidence': 0.75 + np.random.normal(0, 0.1),
            'uncertainty': 0.15 + np.random.normal(0, 0.05),
            'regime': np.random.choice(['attacking', 'consolidating', 'pressure']),
            'pressure_score': min(1.0, max(0.0, wickets_fallen / 5 + (20 - balls_remaining/6) / 20))
        }


class CounterfactualEventGenerator:
    """Generates counterfactual events based on match context."""
    
    @staticmethod
    def create_catch_scenario(
        ball_id: str,
        original_outcome: Dict[str, Any],
        taken: bool = True
    ) -> CounterfactualEvent:
        """Create a catch taken/dropped scenario."""
        if taken:
            # Catch taken - batter dismissed
            modified_outcome = original_outcome.copy()
            modified_outcome['wicket_type'] = 'caught'
            modified_outcome['runs_scored'] = 0
            description = f"Catch taken on ball {ball_id}"
            event_type = CounterfactualEventType.CATCH_TAKEN
        else:
            # Catch dropped - batter survives
            modified_outcome = original_outcome.copy()
            modified_outcome['wicket_type'] = None
            modified_outcome['runs_scored'] = original_outcome.get('runs_scored', 1)
            description = f"Catch dropped on ball {ball_id}"
            event_type = CounterfactualEventType.CATCH_DROPPED
        
        return CounterfactualEvent(
            event_type=event_type,
            ball_id=ball_id,
            original_outcome=original_outcome,
            modified_outcome=modified_outcome,
            description=description
        )
    
    @staticmethod
    def create_boundary_scenario(
        ball_id: str,
        original_outcome: Dict[str, Any],
        boundary_hit: bool = True
    ) -> CounterfactualEvent:
        """Create a boundary hit/stopped scenario."""
        if boundary_hit:
            # Boundary hit
            modified_outcome = original_outcome.copy()
            modified_outcome['runs_scored'] = 4  # Assume four
            modified_outcome['wicket_type'] = None
            description = f"Boundary hit on ball {ball_id}"
            event_type = CounterfactualEventType.BOUNDARY_HIT
        else:
            # Boundary stopped
            modified_outcome = original_outcome.copy()
            modified_outcome['runs_scored'] = min(2, original_outcome.get('runs_scored', 1))
            description = f"Boundary stopped on ball {ball_id}"
            event_type = CounterfactualEventType.BOUNDARY_STOPPED
        
        return CounterfactualEvent(
            event_type=event_type,
            ball_id=ball_id,
            original_outcome=original_outcome,
            modified_outcome=modified_outcome,
            description=description
        )
    
    @staticmethod
    def create_run_out_scenario(
        ball_id: str,
        original_outcome: Dict[str, Any],
        successful: bool = True
    ) -> CounterfactualEvent:
        """Create a run out successful/failed scenario."""
        if successful:
            # Run out successful
            modified_outcome = original_outcome.copy()
            modified_outcome['wicket_type'] = 'run_out'
            modified_outcome['runs_scored'] = min(1, original_outcome.get('runs_scored', 1))
            description = f"Run out successful on ball {ball_id}"
            event_type = CounterfactualEventType.RUN_OUT_SUCCESS
        else:
            # Run out failed
            modified_outcome = original_outcome.copy()
            modified_outcome['wicket_type'] = None
            modified_outcome['runs_scored'] = original_outcome.get('runs_scored', 1)
            description = f"Run out failed on ball {ball_id}"
            event_type = CounterfactualEventType.RUN_OUT_FAILED
        
        return CounterfactualEvent(
            event_type=event_type,
            ball_id=ball_id,
            original_outcome=original_outcome,
            modified_outcome=modified_outcome,
            description=description
        )


class MatchSimulator:
    """Main match simulator with counterfactual analysis capabilities."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the match simulator.
        
        Args:
            model_path: Path to trained CrickFormer model
        """
        self.inference_wrapper = CrickFormerInferenceWrapper(model_path)
        self.event_generator = CounterfactualEventGenerator()
    
    def simulate_counterfactual(
        self,
        context: SimulationContext,
        ball_sequence: pd.DataFrame,
        counterfactual_event: CounterfactualEvent
    ) -> SimulationComparison:
        """
        Simulate a counterfactual event and compare outcomes.
        
        Args:
            context: Current match context
            ball_sequence: Recent ball-by-ball data
            counterfactual_event: Event to simulate
        
        Returns:
            Side-by-side comparison of outcomes
        """
        logger.info(f"Simulating counterfactual: {counterfactual_event.description}")
        
        # Get original prediction
        original_prediction = self.inference_wrapper.predict(context, ball_sequence)
        
        # Get counterfactual prediction
        modifications = {
            counterfactual_event.ball_id: counterfactual_event.modified_outcome
        }
        counterfactual_prediction = self.inference_wrapper.predict(
            context, ball_sequence, modifications
        )
        
        # Create comparison
        comparison = SimulationComparison(
            original_prediction=original_prediction,
            counterfactual_prediction=counterfactual_prediction,
            event=counterfactual_event,
            context=context,
            win_probability_impact=0.0,  # Will be calculated in __post_init__
            score_impact=0.0,
            wicket_impact=0.0,
            impact_magnitude="",
            confidence_interval=(0.0, 0.0)
        )
        
        # Calculate confidence interval (simplified)
        uncertainty = max(
            original_prediction.model_uncertainty,
            counterfactual_prediction.model_uncertainty
        )
        
        comparison.confidence_interval = (
            comparison.win_probability_impact - 1.96 * uncertainty,
            comparison.win_probability_impact + 1.96 * uncertainty
        )
        
        return comparison
    
    def simulate_multiple_scenarios(
        self,
        context: SimulationContext,
        ball_sequence: pd.DataFrame,
        events: List[CounterfactualEvent]
    ) -> List[SimulationComparison]:
        """
        Simulate multiple counterfactual scenarios.
        
        Args:
            context: Current match context
            ball_sequence: Recent ball-by-ball data
            events: List of events to simulate
        
        Returns:
            List of simulation comparisons
        """
        comparisons = []
        
        for event in events:
            try:
                comparison = self.simulate_counterfactual(context, ball_sequence, event)
                comparisons.append(comparison)
            except Exception as e:
                logger.error(f"Failed to simulate event {event.description}: {e}")
        
        return comparisons
    
    def generate_suggested_scenarios(
        self,
        context: SimulationContext,
        ball_sequence: pd.DataFrame,
        num_scenarios: int = 5
    ) -> List[CounterfactualEvent]:
        """
        Generate suggested counterfactual scenarios based on match context.
        
        Args:
            context: Current match context
            ball_sequence: Recent ball-by-ball data
            num_scenarios: Number of scenarios to generate
        
        Returns:
            List of suggested counterfactual events
        """
        scenarios = []
        
        if len(ball_sequence) == 0:
            return scenarios
        
        # Get recent balls for scenario generation
        recent_balls = ball_sequence.tail(6)
        
        for _, ball_row in recent_balls.iterrows():
            ball_id = ball_row.get('ball_id', f"ball_{len(scenarios)}")
            original_outcome = ball_row.to_dict()
            
            # Generate catch scenarios
            if ball_row.get('runs_scored', 0) > 0 and not ball_row.get('wicket_type'):
                # Could have been caught
                scenarios.append(
                    self.event_generator.create_catch_scenario(
                        ball_id, original_outcome, taken=True
                    )
                )
            
            if ball_row.get('wicket_type') == 'caught':
                # Could have been dropped
                scenarios.append(
                    self.event_generator.create_catch_scenario(
                        ball_id, original_outcome, taken=False
                    )
                )
            
            # Generate boundary scenarios
            if ball_row.get('runs_scored', 0) in [1, 2, 3]:
                # Could have been a boundary
                scenarios.append(
                    self.event_generator.create_boundary_scenario(
                        ball_id, original_outcome, boundary_hit=True
                    )
                )
            
            if ball_row.get('runs_scored', 0) >= 4:
                # Could have been stopped
                scenarios.append(
                    self.event_generator.create_boundary_scenario(
                        ball_id, original_outcome, boundary_hit=False
                    )
                )
            
            # Generate run out scenarios
            if ball_row.get('runs_scored', 0) > 0 and not ball_row.get('wicket_type'):
                # Could have been run out
                scenarios.append(
                    self.event_generator.create_run_out_scenario(
                        ball_id, original_outcome, successful=True
                    )
                )
            
            if len(scenarios) >= num_scenarios:
                break
        
        return scenarios[:num_scenarios]
    
    def analyze_scenario_impact(
        self,
        comparisons: List[SimulationComparison]
    ) -> Dict[str, Any]:
        """
        Analyze the overall impact of multiple scenarios.
        
        Args:
            comparisons: List of simulation comparisons
        
        Returns:
            Impact analysis summary
        """
        if not comparisons:
            return {}
        
        win_prob_impacts = [c.win_probability_impact for c in comparisons]
        score_impacts = [c.score_impact for c in comparisons]
        
        return {
            'total_scenarios': len(comparisons),
            'avg_win_prob_impact': np.mean(win_prob_impacts),
            'max_win_prob_impact': max(win_prob_impacts),
            'min_win_prob_impact': min(win_prob_impacts),
            'avg_score_impact': np.mean(score_impacts),
            'max_score_impact': max(score_impacts),
            'min_score_impact': min(score_impacts),
            'high_impact_scenarios': len([c for c in comparisons if c.impact_magnitude in ['medium', 'large']]),
            'impact_distribution': {
                'negligible': len([c for c in comparisons if c.impact_magnitude == 'negligible']),
                'small': len([c for c in comparisons if c.impact_magnitude == 'small']),
                'medium': len([c for c in comparisons if c.impact_magnitude == 'medium']),
                'large': len([c for c in comparisons if c.impact_magnitude == 'large'])
            }
        }


def create_simulation_context_from_match_state(
    match_data: pd.DataFrame,
    target_ball: Optional[str] = None
) -> SimulationContext:
    """
    Create simulation context from match data.
    
    Args:
        match_data: Ball-by-ball match data
        target_ball: Specific ball to simulate from (None for latest)
    
    Returns:
        Simulation context
    """
    if len(match_data) == 0:
        raise ValueError("Match data cannot be empty")
    
    # Get the target ball or use the latest
    if target_ball:
        target_row = match_data[match_data['ball_id'] == target_ball]
        if len(target_row) == 0:
            raise ValueError(f"Ball ID {target_ball} not found in match data")
        current_ball = target_row.iloc[0]
    else:
        current_ball = match_data.iloc[-1]
    
    # Calculate current score and wickets
    match_id = current_ball['match_id']
    innings = current_ball['innings']
    
    innings_data = match_data[
        (match_data['match_id'] == match_id) & 
        (match_data['innings'] == innings)
    ]
    
    if target_ball:
        # For specific ball, calculate up to that ball only
        target_ball_index = match_data[match_data['ball_id'] == target_ball].index[0]
        innings_data_up_to_target = innings_data[innings_data.index <= target_ball_index]
        current_score = innings_data_up_to_target['runs_scored'].sum()
        wickets_fallen = innings_data_up_to_target['wicket_type'].notna().sum()
    else:
        # For latest ball, use all innings data
        current_score = innings_data['runs_scored'].sum()
        wickets_fallen = innings_data['wicket_type'].notna().sum()
    
    # Calculate balls remaining (assuming T20)
    current_over = current_ball['over']
    current_ball_in_over = current_ball['ball']
    balls_bowled = (current_over - 1) * 6 + current_ball_in_over
    balls_remaining = max(0, 120 - balls_bowled)  # T20 = 120 balls
    
    # Calculate target and required run rate if chasing
    target_score = None
    required_run_rate = None
    
    if innings == 2:  # Second innings
        first_innings_data = match_data[
            (match_data['match_id'] == match_id) & 
            (match_data['innings'] == 1)
        ]
        if len(first_innings_data) > 0:
            target_score = first_innings_data['runs_scored'].sum() + 1
            runs_needed = target_score - current_score
            required_run_rate = (runs_needed / balls_remaining) * 6 if balls_remaining > 0 else 0
    
    return SimulationContext(
        match_id=match_id,
        innings=innings,
        over=current_over,
        ball=current_ball_in_over,
        current_score=current_score,
        wickets_fallen=wickets_fallen,
        balls_remaining=balls_remaining,
        target_score=target_score,
        required_run_rate=required_run_rate,
        current_batter=current_ball.get('batter', ''),
        current_bowler=current_ball.get('bowler', ''),
        captain=current_ball.get('captain', ''),
        venue=current_ball.get('venue', ''),
        conditions=current_ball.get('conditions', 'normal')
    )