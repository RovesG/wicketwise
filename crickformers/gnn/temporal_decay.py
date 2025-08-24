# Purpose: Temporal decay functions for Knowledge Graph relationships
# Author: WicketWise Team, Last Modified: 2025-08-23

"""
Advanced temporal decay functions that weight cricket performance data
based on recency, context, and situational relevance.

The temporal decay system recognizes that:
- Recent performances are more predictive than historical ones
- Different contexts (formats, venues, conditions) have different decay rates
- Player form cycles and career phases affect decay patterns
- Match importance influences how long performances remain relevant
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DecayType(Enum):
    """Types of temporal decay functions"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear" 
    SIGMOID = "sigmoid"
    ADAPTIVE = "adaptive"
    CONTEXT_AWARE = "context_aware"


class ContextType(Enum):
    """Different contexts that affect decay rates"""
    FORMAT = "format"  # T20, ODI, Test
    VENUE = "venue"    # Home, away, neutral
    TOURNAMENT = "tournament"  # Regular, playoff, final
    CONDITIONS = "conditions"  # Weather, pitch type
    OPPOSITION = "opposition"  # Strength of opponent


@dataclass
class DecayConfig:
    """Configuration for temporal decay functions"""
    decay_type: DecayType
    half_life_days: float = 365.0  # Days for weight to reduce by half
    min_weight: float = 0.01       # Minimum weight (never fully zero)
    max_days: float = 1095.0       # Maximum days to consider (3 years)
    context_multipliers: Optional[Dict[str, float]] = None
    adaptive_params: Optional[Dict[str, float]] = None


@dataclass
class PerformanceEvent:
    """A single performance event with temporal context"""
    event_id: str
    player_id: str
    date: datetime
    performance_metrics: Dict[str, float]
    context: Dict[str, str]
    importance_score: float = 1.0
    confidence_score: float = 1.0


class TemporalDecayFunction(ABC):
    """Abstract base class for temporal decay functions"""
    
    def __init__(self, config: DecayConfig):
        self.config = config
        
    @abstractmethod
    def calculate_weight(self, days_ago: float, context: Optional[Dict[str, str]] = None) -> float:
        """Calculate decay weight for a given time difference"""
        pass
    
    @abstractmethod
    def get_effective_half_life(self, context: Optional[Dict[str, str]] = None) -> float:
        """Get effective half-life considering context"""
        pass
    
    def validate_weight(self, weight: float) -> float:
        """Ensure weight is within valid bounds"""
        return max(self.config.min_weight, min(1.0, weight))


class ExponentialDecay(TemporalDecayFunction):
    """Exponential decay: weight = e^(-λ * days_ago)"""
    
    def __init__(self, config: DecayConfig):
        super().__init__(config)
        # λ = ln(2) / half_life (for half-life decay)
        self.lambda_param = math.log(2) / config.half_life_days
        
    def calculate_weight(self, days_ago: float, context: Optional[Dict[str, str]] = None) -> float:
        """Calculate exponential decay weight"""
        if days_ago < 0:
            return 1.0  # Future events get full weight
            
        if days_ago > self.config.max_days:
            return self.config.min_weight
            
        effective_lambda = self.lambda_param
        
        # Apply context multipliers
        if context and self.config.context_multipliers:
            multiplier = 1.0
            for ctx_key, ctx_value in context.items():
                ctx_multiplier = self.config.context_multipliers.get(f"{ctx_key}_{ctx_value}", 1.0)
                multiplier *= ctx_multiplier
            effective_lambda *= multiplier
            
        weight = math.exp(-effective_lambda * days_ago)
        return self.validate_weight(weight)
    
    def get_effective_half_life(self, context: Optional[Dict[str, str]] = None) -> float:
        """Get effective half-life considering context"""
        base_half_life = self.config.half_life_days
        
        if context and self.config.context_multipliers:
            multiplier = 1.0
            for ctx_key, ctx_value in context.items():
                ctx_multiplier = self.config.context_multipliers.get(f"{ctx_key}_{ctx_value}", 1.0)
                multiplier *= ctx_multiplier
            # Inverse relationship: higher multiplier = shorter half-life
            return base_half_life / multiplier
            
        return base_half_life


class LinearDecay(TemporalDecayFunction):
    """Linear decay: weight = max(0, 1 - (days_ago / max_days))"""
    
    def calculate_weight(self, days_ago: float, context: Optional[Dict[str, str]] = None) -> float:
        """Calculate linear decay weight"""
        if days_ago < 0:
            return 1.0
            
        effective_max_days = self.config.max_days
        
        # Apply context adjustments
        if context and self.config.context_multipliers:
            multiplier = 1.0
            for ctx_key, ctx_value in context.items():
                ctx_multiplier = self.config.context_multipliers.get(f"{ctx_key}_{ctx_value}", 1.0)
                multiplier *= ctx_multiplier
            effective_max_days /= multiplier
            
        if days_ago >= effective_max_days:
            return self.config.min_weight
            
        weight = 1.0 - (days_ago / effective_max_days)
        return self.validate_weight(weight)
    
    def get_effective_half_life(self, context: Optional[Dict[str, str]] = None) -> float:
        """For linear decay, half-life is when weight = 0.5"""
        effective_max_days = self.config.max_days
        
        if context and self.config.context_multipliers:
            multiplier = 1.0
            for ctx_key, ctx_value in context.items():
                ctx_multiplier = self.config.context_multipliers.get(f"{ctx_key}_{ctx_value}", 1.0)
                multiplier *= ctx_multiplier
            effective_max_days /= multiplier
            
        return effective_max_days * 0.5


class SigmoidDecay(TemporalDecayFunction):
    """Sigmoid decay: weight = 1 / (1 + e^(k * (days_ago - midpoint)))"""
    
    def __init__(self, config: DecayConfig):
        super().__init__(config)
        # Default sigmoid parameters
        self.midpoint = config.half_life_days  # Inflection point
        self.steepness = 4.0 / config.half_life_days  # Controls steepness
        
        if config.adaptive_params:
            self.midpoint = config.adaptive_params.get("midpoint", self.midpoint)
            self.steepness = config.adaptive_params.get("steepness", self.steepness)
    
    def calculate_weight(self, days_ago: float, context: Optional[Dict[str, str]] = None) -> float:
        """Calculate sigmoid decay weight"""
        if days_ago < 0:
            return 1.0
            
        effective_midpoint = self.midpoint
        effective_steepness = self.steepness
        
        # Apply context adjustments
        if context and self.config.context_multipliers:
            multiplier = 1.0
            for ctx_key, ctx_value in context.items():
                ctx_multiplier = self.config.context_multipliers.get(f"{ctx_key}_{ctx_value}", 1.0)
                multiplier *= ctx_multiplier
            effective_midpoint /= multiplier
            effective_steepness *= multiplier
            
        # Sigmoid function
        try:
            weight = 1.0 / (1.0 + math.exp(effective_steepness * (days_ago - effective_midpoint)))
        except OverflowError:
            weight = 0.0 if days_ago > effective_midpoint else 1.0
            
        return self.validate_weight(weight)
    
    def get_effective_half_life(self, context: Optional[Dict[str, str]] = None) -> float:
        """For sigmoid, half-life is the midpoint"""
        effective_midpoint = self.midpoint
        
        if context and self.config.context_multipliers:
            multiplier = 1.0
            for ctx_key, ctx_value in context.items():
                ctx_multiplier = self.config.context_multipliers.get(f"{ctx_key}_{ctx_value}", 1.0)
                multiplier *= ctx_multiplier
            effective_midpoint /= multiplier
            
        return effective_midpoint


class AdaptiveDecay(TemporalDecayFunction):
    """Adaptive decay that adjusts based on player form and context"""
    
    def __init__(self, config: DecayConfig):
        super().__init__(config)
        self.base_decay = ExponentialDecay(config)
        
        # Adaptive parameters
        self.form_window_days = config.adaptive_params.get("form_window_days", 90) if config.adaptive_params else 90
        self.form_threshold = config.adaptive_params.get("form_threshold", 0.1) if config.adaptive_params else 0.1
        
    def calculate_weight(self, days_ago: float, context: Optional[Dict[str, str]] = None) -> float:
        """Calculate adaptive decay weight"""
        base_weight = self.base_decay.calculate_weight(days_ago, context)
        
        # Adaptive adjustments based on context
        if context:
            # If player was in good form recently, extend decay
            if "recent_form" in context:
                form_score = float(context.get("recent_form", "0"))
                if form_score > self.form_threshold:
                    # Good form extends relevance
                    form_multiplier = 1.0 + (form_score * 0.5)
                    base_weight = min(1.0, base_weight * form_multiplier)
                    
            # Tournament importance affects decay
            if "tournament_importance" in context:
                importance = float(context.get("tournament_importance", "1.0"))
                if importance > 1.5:  # High importance match
                    importance_multiplier = 1.0 + ((importance - 1.0) * 0.3)
                    base_weight = min(1.0, base_weight * importance_multiplier)
                    
        return self.validate_weight(base_weight)
    
    def get_effective_half_life(self, context: Optional[Dict[str, str]] = None) -> float:
        """Get adaptive half-life"""
        return self.base_decay.get_effective_half_life(context)


class ContextAwareDecay(TemporalDecayFunction):
    """Context-aware decay with different functions for different contexts"""
    
    def __init__(self, config: DecayConfig):
        super().__init__(config)
        
        # Different decay functions for different contexts
        self.decay_functions = {
            "t20": ExponentialDecay(DecayConfig(DecayType.EXPONENTIAL, half_life_days=180)),
            "odi": ExponentialDecay(DecayConfig(DecayType.EXPONENTIAL, half_life_days=365)),
            "test": ExponentialDecay(DecayConfig(DecayType.EXPONENTIAL, half_life_days=730)),
            "playoff": SigmoidDecay(DecayConfig(DecayType.SIGMOID, half_life_days=270)),
            "final": SigmoidDecay(DecayConfig(DecayType.SIGMOID, half_life_days=450)),
        }
        
        self.default_decay = ExponentialDecay(config)
        
    def calculate_weight(self, days_ago: float, context: Optional[Dict[str, str]] = None) -> float:
        """Calculate context-aware decay weight"""
        if not context:
            return self.default_decay.calculate_weight(days_ago, context)
            
        # Choose appropriate decay function based on context
        decay_function = self.default_decay
        
        if "format" in context:
            format_key = context["format"].lower()
            if format_key in self.decay_functions:
                decay_function = self.decay_functions[format_key]
                
        elif "tournament_stage" in context:
            stage_key = context["tournament_stage"].lower()
            if stage_key in self.decay_functions:
                decay_function = self.decay_functions[stage_key]
                
        return decay_function.calculate_weight(days_ago, context)
    
    def get_effective_half_life(self, context: Optional[Dict[str, str]] = None) -> float:
        """Get context-specific half-life"""
        if not context:
            return self.default_decay.get_effective_half_life(context)
            
        # Choose appropriate decay function
        decay_function = self.default_decay
        
        if "format" in context:
            format_key = context["format"].lower()
            if format_key in self.decay_functions:
                decay_function = self.decay_functions[format_key]
                
        return decay_function.get_effective_half_life(context)


class TemporalDecayEngine:
    """Main engine for applying temporal decay to cricket performance data"""
    
    def __init__(self, default_config: Optional[DecayConfig] = None):
        self.default_config = default_config or DecayConfig(
            decay_type=DecayType.EXPONENTIAL,
            half_life_days=365.0,
            min_weight=0.01,
            max_days=1095.0
        )
        
        self.decay_functions: Dict[DecayType, TemporalDecayFunction] = {}
        self._initialize_decay_functions()
        
        # Performance tracking
        self.calculation_history: List[Dict[str, any]] = []
        
    def _initialize_decay_functions(self):
        """Initialize all decay function types"""
        self.decay_functions = {
            DecayType.EXPONENTIAL: ExponentialDecay(self.default_config),
            DecayType.LINEAR: LinearDecay(self.default_config),
            DecayType.SIGMOID: SigmoidDecay(self.default_config),
            DecayType.ADAPTIVE: AdaptiveDecay(self.default_config),
            DecayType.CONTEXT_AWARE: ContextAwareDecay(self.default_config)
        }
        
    def calculate_temporal_weights(
        self, 
        events: List[PerformanceEvent],
        reference_date: Optional[datetime] = None,
        decay_type: DecayType = DecayType.EXPONENTIAL
    ) -> List[Tuple[PerformanceEvent, float]]:
        """
        Calculate temporal weights for a list of performance events
        
        Args:
            events: List of performance events
            reference_date: Reference date for calculating decay (default: now)
            decay_type: Type of decay function to use
            
        Returns:
            List of (event, weight) tuples
        """
        if not events:
            return []
            
        reference_date = reference_date or datetime.now()
        decay_function = self.decay_functions.get(decay_type, self.decay_functions[DecayType.EXPONENTIAL])
        
        weighted_events = []
        
        for event in events:
            # Calculate days between reference and event
            days_ago = (reference_date - event.date).total_seconds() / (24 * 3600)
            
            # Calculate temporal weight
            temporal_weight = decay_function.calculate_weight(days_ago, event.context)
            
            # Apply importance and confidence multipliers
            final_weight = temporal_weight * event.importance_score * event.confidence_score
            final_weight = max(self.default_config.min_weight, min(1.0, final_weight))
            
            weighted_events.append((event, final_weight))
            
            # Track calculation for analysis
            self.calculation_history.append({
                "event_id": event.event_id,
                "player_id": event.player_id,
                "days_ago": days_ago,
                "temporal_weight": temporal_weight,
                "importance_score": event.importance_score,
                "confidence_score": event.confidence_score,
                "final_weight": final_weight,
                "decay_type": decay_type.value,
                "reference_date": reference_date.isoformat()
            })
            
        # Sort by weight (descending)
        weighted_events.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(
            f"Calculated temporal weights for {len(events)} events using {decay_type.value} decay. "
            f"Average weight: {np.mean([w for _, w in weighted_events]):.3f}"
        )
        
        return weighted_events
    
    def get_weighted_aggregation(
        self,
        events: List[PerformanceEvent],
        metric_name: str,
        decay_type: DecayType = DecayType.EXPONENTIAL,
        aggregation_type: str = "weighted_mean"
    ) -> Dict[str, float]:
        """
        Get weighted aggregation of a specific metric across events
        
        Args:
            events: List of performance events
            metric_name: Name of metric to aggregate
            decay_type: Type of decay function
            aggregation_type: Type of aggregation (weighted_mean, weighted_sum, max, etc.)
            
        Returns:
            Dictionary with aggregated statistics
        """
        weighted_events = self.calculate_temporal_weights(events, decay_type=decay_type)
        
        if not weighted_events:
            return {"value": 0.0, "confidence": 0.0, "sample_size": 0}
            
        # Extract metric values and weights
        values = []
        weights = []
        
        for event, weight in weighted_events:
            if metric_name in event.performance_metrics:
                values.append(event.performance_metrics[metric_name])
                weights.append(weight)
                
        if not values:
            return {"value": 0.0, "confidence": 0.0, "sample_size": 0}
            
        values = np.array(values)
        weights = np.array(weights)
        
        # Calculate aggregation
        result = {"sample_size": len(values)}
        
        if aggregation_type == "weighted_mean":
            if weights.sum() > 0:
                result["value"] = np.average(values, weights=weights)
                # Confidence based on effective sample size
                effective_n = (weights.sum() ** 2) / (weights ** 2).sum()
                result["confidence"] = min(1.0, effective_n / 10.0)  # Confidence plateaus at 10 samples
            else:
                result["value"] = 0.0
                result["confidence"] = 0.0
                
        elif aggregation_type == "weighted_sum":
            result["value"] = (values * weights).sum()
            result["confidence"] = min(1.0, weights.sum())
            
        elif aggregation_type == "max":
            max_idx = np.argmax(weights)
            result["value"] = values[max_idx]
            result["confidence"] = weights[max_idx]
            
        elif aggregation_type == "recent_trend":
            # Calculate trend over recent events (chronologically ordered)
            # First, we need to sort events by date to get proper chronological order
            event_data = []
            for event, weight in weighted_events:
                if metric_name in event.performance_metrics:
                    event_data.append({
                        'date': event.date,
                        'value': event.performance_metrics[metric_name],
                        'weight': weight
                    })
            
            if len(event_data) >= 2:
                # Sort by date to get chronological order
                event_data.sort(key=lambda x: x['date'])
                
                # Take recent events (top 50% by weight, but maintain chronological order)
                n_recent = max(2, len(event_data) // 2)
                recent_events = event_data[-n_recent:]  # Most recent chronologically
                
                # Extract values in chronological order
                recent_values = [e['value'] for e in recent_events]
                recent_weights = [e['weight'] for e in recent_events]
                
                # Calculate weighted trend (more recent events have higher influence)
                x = np.arange(len(recent_values))
                trend = np.polyfit(x, recent_values, 1, w=recent_weights)[0]  # Weighted slope
                result["value"] = trend
                result["confidence"] = min(1.0, len(recent_values) / 5.0)
            else:
                result["value"] = 0.0
                result["confidence"] = 0.0
                
        return result
    
    def analyze_decay_patterns(
        self, 
        events: List[PerformanceEvent],
        metric_name: str
    ) -> Dict[str, any]:
        """
        Analyze how different decay functions affect metric aggregation
        
        Args:
            events: List of performance events
            metric_name: Metric to analyze
            
        Returns:
            Analysis results for different decay types
        """
        analysis = {}
        
        for decay_type in DecayType:
            try:
                result = self.get_weighted_aggregation(events, metric_name, decay_type)
                analysis[decay_type.value] = result
            except Exception as e:
                logger.warning(f"Failed to analyze {decay_type.value} decay: {e}")
                analysis[decay_type.value] = {"error": str(e)}
                
        return analysis
    
    def get_calculation_statistics(self) -> Dict[str, any]:
        """Get statistics about decay calculations"""
        if not self.calculation_history:
            return {"message": "No calculations performed yet"}
            
        df = pd.DataFrame(self.calculation_history)
        
        return {
            "total_calculations": len(df),
            "unique_players": df["player_id"].nunique(),
            "decay_types_used": df["decay_type"].value_counts().to_dict(),
            "average_temporal_weight": df["temporal_weight"].mean(),
            "average_final_weight": df["final_weight"].mean(),
            "weight_distribution": {
                "min": df["final_weight"].min(),
                "25%": df["final_weight"].quantile(0.25),
                "50%": df["final_weight"].quantile(0.50),
                "75%": df["final_weight"].quantile(0.75),
                "max": df["final_weight"].max()
            }
        }
