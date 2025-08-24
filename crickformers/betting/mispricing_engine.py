# Purpose: Mispricing detection engine for betting opportunities
# Author: WicketWise Team, Last Modified: 2025-08-23

"""
Advanced mispricing detection engine that identifies betting value by:
- Aggregating odds from multiple bookmakers
- Converting odds to implied probabilities
- Comparing with AI model predictions
- Detecting statistical arbitrage opportunities
- Calculating expected value and confidence intervals
- Tracking market inefficiencies over time

The engine uses sophisticated statistical methods to identify when bookmaker
odds significantly deviate from true probability estimates.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import statistics

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class BetType(Enum):
    """Types of cricket bets"""
    MATCH_WINNER = "match_winner"
    TOTAL_RUNS = "total_runs"
    PLAYER_RUNS = "player_runs"
    PLAYER_WICKETS = "player_wickets"
    NEXT_WICKET = "next_wicket"
    NEXT_BOUNDARY = "next_boundary"
    INNINGS_RUNS = "innings_runs"
    POWERPLAY_RUNS = "powerplay_runs"
    METHOD_OF_DISMISSAL = "method_of_dismissal"
    TOSS_WINNER = "toss_winner"


class BookmakerType(Enum):
    """Types of bookmakers"""
    SHARP = "sharp"          # Professional, low-margin bookmakers
    SOFT = "soft"            # Recreational, high-margin bookmakers  
    EXCHANGE = "exchange"    # Betting exchanges (Betfair, etc.)
    OFFSHORE = "offshore"    # Offshore bookmakers
    REGULATED = "regulated"  # Regulated domestic bookmakers


@dataclass
class OddsData:
    """Single odds data point from a bookmaker"""
    bookmaker_id: str
    bookmaker_type: BookmakerType
    bet_type: BetType
    market_description: str
    odds: float
    stake_limit: float
    timestamp: datetime
    match_id: str
    selection: str  # What is being bet on
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class ImpliedProbability:
    """Implied probability calculation from odds"""
    probability: float
    overround: float  # Bookmaker margin
    true_probability: float  # Adjusted for overround
    confidence_interval: Tuple[float, float]
    sample_size: int


@dataclass
class ValueOpportunity:
    """Identified value betting opportunity"""
    opportunity_id: str
    match_id: str
    bet_type: BetType
    selection: str
    bookmaker_odds: float
    model_probability: float
    implied_probability: float
    expected_value: float
    kelly_fraction: float
    confidence_score: float
    risk_level: str
    max_stake: float
    timestamp: datetime
    expires_at: datetime
    metadata: Dict[str, any] = field(default_factory=dict)


class OddsConverter:
    """Utility class for odds conversions and probability calculations"""
    
    @staticmethod
    def decimal_to_probability(decimal_odds: float) -> float:
        """Convert decimal odds to implied probability"""
        if decimal_odds <= 1.0:
            raise ValueError(f"Invalid decimal odds: {decimal_odds}")
        return 1.0 / decimal_odds
    
    @staticmethod
    def probability_to_decimal(probability: float) -> float:
        """Convert probability to decimal odds"""
        if probability <= 0 or probability >= 1:
            raise ValueError(f"Invalid probability: {probability}")
        return 1.0 / probability
    
    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal odds"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    @staticmethod
    def fractional_to_decimal(numerator: int, denominator: int) -> float:
        """Convert fractional odds to decimal odds"""
        return (numerator / denominator) + 1
    
    @staticmethod
    def calculate_overround(implied_probabilities: List[float]) -> float:
        """Calculate overround (bookmaker margin) from implied probabilities"""
        return sum(implied_probabilities) - 1.0
    
    @staticmethod
    def remove_overround(implied_probabilities: List[float]) -> List[float]:
        """Remove overround to get true probabilities"""
        total = sum(implied_probabilities)
        if total <= 1.0:
            return implied_probabilities
        
        # Proportional method
        return [prob / total for prob in implied_probabilities]


class MispricingEngine:
    """Core engine for detecting betting value and mispricing"""
    
    def __init__(self, config: Optional[Dict[str, any]] = None):
        self.config = config or {}
        
        # Configuration parameters
        self.min_edge_threshold = self.config.get("min_edge_threshold", 0.05)  # 5% minimum edge
        self.max_kelly_fraction = self.config.get("max_kelly_fraction", 0.25)  # 25% max Kelly
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.max_odds_age_minutes = self.config.get("max_odds_age_minutes", 10)
        
        # Bookmaker reliability weights
        self.bookmaker_weights = {
            BookmakerType.SHARP: 1.0,
            BookmakerType.EXCHANGE: 0.9,
            BookmakerType.REGULATED: 0.7,
            BookmakerType.SOFT: 0.5,
            BookmakerType.OFFSHORE: 0.3
        }
        
        # Historical tracking
        self.odds_history: List[OddsData] = []
        self.mispricing_history: List[ValueOpportunity] = []
        self.performance_tracking = {
            "total_opportunities": 0,
            "profitable_bets": 0,
            "total_profit": 0.0,
            "roi": 0.0,
            "sharpe_ratio": 0.0
        }
    
    def add_odds_data(self, odds_data: List[OddsData]) -> None:
        """Add new odds data to the engine"""
        current_time = datetime.now()
        
        # Filter out stale odds
        fresh_odds = [
            odds for odds in odds_data
            if (current_time - odds.timestamp).total_seconds() < (self.max_odds_age_minutes * 60)
        ]
        
        self.odds_history.extend(fresh_odds)
        
        # Keep only recent odds (last 24 hours)
        cutoff_time = current_time - timedelta(hours=24)
        self.odds_history = [
            odds for odds in self.odds_history
            if odds.timestamp > cutoff_time
        ]
        
        logger.info(f"Added {len(fresh_odds)} fresh odds, {len(odds_data) - len(fresh_odds)} stale odds filtered")
    
    def calculate_implied_probabilities(self, odds_list: List[OddsData]) -> Dict[str, ImpliedProbability]:
        """Calculate implied probabilities for a set of odds"""
        if not odds_list:
            return {}
        
        # Group by selection
        selection_odds = {}
        for odds in odds_list:
            selection = odds.selection
            if selection not in selection_odds:
                selection_odds[selection] = []
            selection_odds[selection].append(odds)
        
        implied_probs = {}
        
        for selection, odds_group in selection_odds.items():
            # Get all odds for this selection
            decimal_odds = [odds.odds for odds in odds_group]
            implied_probs_raw = [OddsConverter.decimal_to_probability(odds) for odds in decimal_odds]
            
            # Calculate weighted average based on bookmaker reliability
            weights = [self.bookmaker_weights.get(odds.bookmaker_type, 0.5) for odds in odds_group]
            
            if sum(weights) > 0:
                weighted_prob = np.average(implied_probs_raw, weights=weights)
            else:
                weighted_prob = np.mean(implied_probs_raw)
            
            # Calculate overround
            overround = OddsConverter.calculate_overround(implied_probs_raw)
            
            # Adjust for overround
            true_prob = weighted_prob / (1 + overround) if overround > 0 else weighted_prob
            
            # Calculate confidence interval
            if len(implied_probs_raw) > 1:
                std_error = statistics.stdev(implied_probs_raw) / math.sqrt(len(implied_probs_raw))
                margin_error = 1.96 * std_error  # 95% confidence
                ci_lower = max(0.001, weighted_prob - margin_error)
                ci_upper = min(0.999, weighted_prob + margin_error)
                confidence_interval = (ci_lower, ci_upper)
            else:
                confidence_interval = (weighted_prob * 0.9, weighted_prob * 1.1)
            
            implied_probs[selection] = ImpliedProbability(
                probability=weighted_prob,
                overround=overround,
                true_probability=true_prob,
                confidence_interval=confidence_interval,
                sample_size=len(implied_probs_raw)
            )
        
        return implied_probs
    
    def detect_mispricing(
        self, 
        model_predictions: Dict[str, float],
        current_odds: List[OddsData],
        match_id: str
    ) -> List[ValueOpportunity]:
        """
        Detect mispricing opportunities by comparing model predictions with market odds
        
        Args:
            model_predictions: Dictionary of {selection: probability} from AI model
            current_odds: List of current odds from bookmakers
            match_id: Match identifier
            
        Returns:
            List of value opportunities found
        """
        opportunities = []
        
        if not model_predictions or not current_odds:
            return opportunities
        
        # Calculate market implied probabilities
        market_probs = self.calculate_implied_probabilities(current_odds)
        
        # Compare model predictions with market
        for selection, model_prob in model_predictions.items():
            if selection in market_probs:
                market_data = market_probs[selection]
                market_prob = market_data.true_probability
                
                # Calculate expected value
                best_odds = self._find_best_odds(current_odds, selection)
                if best_odds:
                    expected_value = self._calculate_expected_value(model_prob, best_odds.odds)
                    
                    # Check if this is a value opportunity
                    if expected_value > self.min_edge_threshold:
                        kelly_fraction = self._calculate_kelly_fraction(model_prob, best_odds.odds)
                        kelly_fraction = min(kelly_fraction, self.max_kelly_fraction)
                        
                        # Calculate confidence score
                        confidence = self._calculate_confidence_score(
                            model_prob, market_prob, market_data.sample_size
                        )
                        
                        if confidence >= self.confidence_threshold:
                            # Determine risk level
                            risk_level = self._assess_risk_level(expected_value, kelly_fraction, confidence)
                            
                            opportunity = ValueOpportunity(
                                opportunity_id=f"opp_{match_id}_{selection}_{int(datetime.now().timestamp())}",
                                match_id=match_id,
                                bet_type=best_odds.bet_type,
                                selection=selection,
                                bookmaker_odds=best_odds.odds,
                                model_probability=model_prob,
                                implied_probability=market_prob,
                                expected_value=expected_value,
                                kelly_fraction=kelly_fraction,
                                confidence_score=confidence,
                                risk_level=risk_level,
                                max_stake=best_odds.stake_limit,
                                timestamp=datetime.now(),
                                expires_at=datetime.now() + timedelta(minutes=15),
                                metadata={
                                    "bookmaker_id": best_odds.bookmaker_id,
                                    "market_sample_size": market_data.sample_size,
                                    "market_overround": market_data.overround,
                                    "prob_difference": model_prob - market_prob
                                }
                            )
                            
                            opportunities.append(opportunity)
                            
                            logger.info(
                                f"🎯 Value opportunity found: {selection} @ {best_odds.odds:.2f} "
                                f"(EV: {expected_value:.1%}, Kelly: {kelly_fraction:.1%}, "
                                f"Confidence: {confidence:.1%})"
                            )
        
        # Store opportunities for tracking
        self.mispricing_history.extend(opportunities)
        self.performance_tracking["total_opportunities"] += len(opportunities)
        
        return opportunities
    
    def _find_best_odds(self, odds_list: List[OddsData], selection: str) -> Optional[OddsData]:
        """Find the best (highest) odds for a selection"""
        selection_odds = [odds for odds in odds_list if odds.selection == selection]
        
        if not selection_odds:
            return None
        
        # Return odds with highest value
        return max(selection_odds, key=lambda x: x.odds)
    
    def _calculate_expected_value(self, true_probability: float, decimal_odds: float) -> float:
        """Calculate expected value of a bet"""
        return (true_probability * (decimal_odds - 1)) - (1 - true_probability)
    
    def _calculate_kelly_fraction(self, true_probability: float, decimal_odds: float) -> float:
        """Calculate Kelly criterion fraction"""
        b = decimal_odds - 1  # Net odds
        p = true_probability
        q = 1 - p
        
        if b <= 0 or p <= 0:
            return 0.0
        
        kelly = (b * p - q) / b
        return max(0.0, kelly)  # Never bet negative Kelly
    
    def _calculate_confidence_score(
        self, 
        model_prob: float, 
        market_prob: float, 
        market_sample_size: int
    ) -> float:
        """Calculate confidence score for the opportunity"""
        # Base confidence on probability difference
        prob_diff = abs(model_prob - market_prob)
        diff_confidence = min(1.0, prob_diff / 0.3)  # Normalize to 30% max difference
        
        # Adjust for market sample size (more bookmakers = higher confidence in market)
        sample_confidence = min(1.0, market_sample_size / 5.0)  # Normalize to 5 bookmakers
        
        # Combine confidences
        combined_confidence = (diff_confidence + sample_confidence) / 2
        
        return combined_confidence
    
    def _assess_risk_level(self, expected_value: float, kelly_fraction: float, confidence: float) -> str:
        """Assess risk level of the opportunity"""
        # High risk: Low confidence or high Kelly fraction
        if confidence < 0.6 or kelly_fraction > 0.15:
            return "high"
        
        # Medium risk: Moderate confidence and Kelly
        elif confidence < 0.8 or kelly_fraction > 0.08:
            return "medium"
        
        # Low risk: High confidence and conservative Kelly
        else:
            return "low"
    
    def detect_arbitrage_opportunities(self, odds_list: List[OddsData]) -> List[Dict[str, any]]:
        """
        Detect arbitrage opportunities across bookmakers
        
        Returns:
            List of arbitrage opportunities with profit calculations
        """
        arbitrage_ops = []
        
        # Group odds by bet type and match
        grouped_odds = {}
        for odds in odds_list:
            key = (odds.match_id, odds.bet_type.value)
            if key not in grouped_odds:
                grouped_odds[key] = {}
            
            selection = odds.selection
            if selection not in grouped_odds[key]:
                grouped_odds[key][selection] = []
            grouped_odds[key][selection].append(odds)
        
        # Check each market for arbitrage
        for (match_id, bet_type), market_odds in grouped_odds.items():
            if len(market_odds) >= 2:  # Need at least 2 outcomes
                arb_opportunity = self._check_arbitrage_in_market(match_id, bet_type, market_odds)
                if arb_opportunity:
                    arbitrage_ops.append(arb_opportunity)
        
        return arbitrage_ops
    
    def _check_arbitrage_in_market(
        self, 
        match_id: str, 
        bet_type: str, 
        market_odds: Dict[str, List[OddsData]]
    ) -> Optional[Dict[str, any]]:
        """Check if arbitrage exists in a specific market"""
        
        # Get best odds for each selection
        best_odds = {}
        for selection, odds_list in market_odds.items():
            best = max(odds_list, key=lambda x: x.odds)
            best_odds[selection] = best
        
        # Calculate total implied probability
        total_implied_prob = sum(
            OddsConverter.decimal_to_probability(odds.odds)
            for odds in best_odds.values()
        )
        
        # Arbitrage exists if total implied probability < 1.0
        if total_implied_prob < 0.98:  # Allow 2% margin for fees
            profit_margin = 1.0 - total_implied_prob
            
            # Calculate optimal stakes
            stakes = self._calculate_arbitrage_stakes(best_odds, 1000.0)  # $1000 total stake
            
            return {
                "match_id": match_id,
                "bet_type": bet_type,
                "profit_margin": profit_margin,
                "total_implied_prob": total_implied_prob,
                "best_odds": {sel: odds.odds for sel, odds in best_odds.items()},
                "optimal_stakes": stakes,
                "bookmakers": {sel: odds.bookmaker_id for sel, odds in best_odds.items()},
                "expires_at": min(odds.timestamp for odds in best_odds.values()) + timedelta(minutes=5)
            }
        
        return None
    
    def _calculate_arbitrage_stakes(self, best_odds: Dict[str, OddsData], total_stake: float) -> Dict[str, float]:
        """Calculate optimal stakes for arbitrage opportunity"""
        stakes = {}
        
        # Calculate stakes proportional to inverse of odds
        total_inverse_odds = sum(1.0 / odds.odds for odds in best_odds.values())
        
        for selection, odds in best_odds.items():
            stake_proportion = (1.0 / odds.odds) / total_inverse_odds
            stakes[selection] = total_stake * stake_proportion
        
        return stakes
    
    def analyze_market_efficiency(self, match_id: str) -> Dict[str, any]:
        """Analyze market efficiency for a specific match"""
        match_odds = [odds for odds in self.odds_history if odds.match_id == match_id]
        
        if not match_odds:
            return {"error": "No odds data found for match"}
        
        # Group by bet type
        bet_type_analysis = {}
        
        for bet_type in set(odds.bet_type for odds in match_odds):
            type_odds = [odds for odds in match_odds if odds.bet_type == bet_type]
            
            # Calculate market metrics
            implied_probs = self.calculate_implied_probabilities(type_odds)
            
            # Calculate spread (difference between best and worst odds)
            odds_values = [odds.odds for odds in type_odds]
            odds_spread = max(odds_values) - min(odds_values) if len(odds_values) > 1 else 0.0
            
            # Calculate overround statistics
            overrounds = [prob.overround for prob in implied_probs.values()]
            avg_overround = np.mean(overrounds) if overrounds else 0.0
            
            bet_type_analysis[bet_type.value] = {
                "num_bookmakers": len(set(odds.bookmaker_id for odds in type_odds)),
                "odds_spread": odds_spread,
                "avg_overround": avg_overround,
                "implied_probabilities": {
                    sel: {"prob": prob.probability, "true_prob": prob.true_probability}
                    for sel, prob in implied_probs.items()
                },
                "market_efficiency_score": self._calculate_efficiency_score(odds_spread, avg_overround)
            }
        
        return {
            "match_id": match_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "bet_types": bet_type_analysis,
            "overall_efficiency": np.mean([
                analysis["market_efficiency_score"] 
                for analysis in bet_type_analysis.values()
            ])
        }
    
    def _calculate_efficiency_score(self, odds_spread: float, avg_overround: float) -> float:
        """Calculate market efficiency score (0-1, higher = more efficient)"""
        # Lower spread = more efficient
        spread_score = max(0.0, 1.0 - (odds_spread / 2.0))  # Normalize to 2.0 max spread
        
        # Lower overround = more efficient
        overround_score = max(0.0, 1.0 - (avg_overround / 0.2))  # Normalize to 20% max overround
        
        # Combine scores
        efficiency = (spread_score + overround_score) / 2
        return min(1.0, efficiency)
    
    def get_performance_summary(self) -> Dict[str, any]:
        """Get performance summary of mispricing detection"""
        total_ops = len(self.mispricing_history)
        
        if total_ops == 0:
            return {"message": "No mispricing opportunities tracked yet"}
        
        # Calculate statistics
        expected_values = [opp.expected_value for opp in self.mispricing_history]
        kelly_fractions = [opp.kelly_fraction for opp in self.mispricing_history]
        confidence_scores = [opp.confidence_score for opp in self.mispricing_history]
        
        # Risk distribution
        risk_distribution = {}
        for opp in self.mispricing_history:
            risk_level = opp.risk_level
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
        
        # Time distribution
        recent_ops = [
            opp for opp in self.mispricing_history
            if (datetime.now() - opp.timestamp).total_seconds() < (24 * 3600)
        ]
        
        return {
            "total_opportunities": total_ops,
            "recent_opportunities_24h": len(recent_ops),
            "expected_value_stats": {
                "mean": np.mean(expected_values),
                "median": np.median(expected_values),
                "std": np.std(expected_values),
                "min": np.min(expected_values),
                "max": np.max(expected_values)
            },
            "kelly_fraction_stats": {
                "mean": np.mean(kelly_fractions),
                "median": np.median(kelly_fractions),
                "max": np.max(kelly_fractions)
            },
            "confidence_stats": {
                "mean": np.mean(confidence_scores),
                "median": np.median(confidence_scores),
                "min": np.min(confidence_scores)
            },
            "risk_distribution": risk_distribution,
            "bet_type_distribution": {
                bet_type.value: sum(1 for opp in self.mispricing_history if opp.bet_type == bet_type)
                for bet_type in BetType
            }
        }
    
    def clear_expired_opportunities(self) -> int:
        """Remove expired opportunities and return count removed"""
        current_time = datetime.now()
        initial_count = len(self.mispricing_history)
        
        self.mispricing_history = [
            opp for opp in self.mispricing_history
            if opp.expires_at > current_time
        ]
        
        removed_count = initial_count - len(self.mispricing_history)
        
        if removed_count > 0:
            logger.info(f"🧹 Removed {removed_count} expired opportunities")
        
        return removed_count
