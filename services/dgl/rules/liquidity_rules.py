# Purpose: Liquidity and execution constraint rule implementations
# Author: WicketWise AI, Last Modified: 2024

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import time
import threading
from collections import defaultdict, deque
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import BetProposal, ExposureSnapshot, RuleId, LiquidityInfo
from config import LiquidityConfig


logger = logging.getLogger(__name__)


class RuleViolation:
    """Represents a rule violation with detailed context"""
    
    def __init__(self, rule_id: RuleId, message: str, 
                 current_value: Optional[float] = None,
                 threshold: Optional[float] = None,
                 severity: str = "ERROR",
                 suggested_amendment: Optional[dict] = None):
        self.rule_id = rule_id
        self.message = message
        self.current_value = current_value
        self.threshold = threshold
        self.severity = severity
        self.suggested_amendment = suggested_amendment


class OddsRangeRule:
    """
    Rule: Odds range validation
    
    Ensures that bet odds fall within acceptable minimum and maximum thresholds.
    Protects against extremely low odds (poor value) and extremely high odds (high risk).
    """
    
    def __init__(self, config: LiquidityConfig):
        self.config = config
        self.rule_id_min = RuleId.LIQ_MIN_ODDS
        self.rule_id_max = RuleId.LIQ_MAX_ODDS
        
    def evaluate(self, proposal: BetProposal, exposure: ExposureSnapshot) -> List[RuleViolation]:
        """Evaluate odds range constraints"""
        violations = []
        
        # Check minimum odds threshold
        if proposal.odds < self.config.min_odds_threshold:
            violations.append(RuleViolation(
                rule_id=self.rule_id_min,
                message=f"Odds {proposal.odds:.3f} below minimum threshold {self.config.min_odds_threshold:.3f}. "
                       f"Low odds provide poor value and high risk of small profits.",
                current_value=proposal.odds,
                threshold=self.config.min_odds_threshold,
                severity="ERROR"
            ))
        
        # Check maximum odds threshold
        if proposal.odds > self.config.max_odds_threshold:
            violations.append(RuleViolation(
                rule_id=self.rule_id_max,
                message=f"Odds {proposal.odds:.3f} above maximum threshold {self.config.max_odds_threshold:.3f}. "
                       f"High odds indicate low probability events with excessive risk.",
                current_value=proposal.odds,
                threshold=self.config.max_odds_threshold,
                severity="ERROR"
            ))
        
        return violations


class SlippageRule:
    """
    Rule: Slippage limit validation
    
    Ensures that the difference between requested odds and fair odds doesn't exceed
    acceptable slippage limits, protecting against poor execution prices.
    """
    
    def __init__(self, config: LiquidityConfig):
        self.config = config
        self.rule_id = RuleId.LIQ_SLIPPAGE_LIMIT
        
    def evaluate(self, proposal: BetProposal, exposure: ExposureSnapshot) -> Optional[RuleViolation]:
        """Evaluate slippage constraints"""
        
        # Skip if no fair odds provided
        if not proposal.fair_odds or proposal.fair_odds <= 0:
            return None
        
        # Calculate slippage in basis points
        slippage_bps = abs((proposal.odds - proposal.fair_odds) / proposal.fair_odds) * 10000
        
        # Check if slippage exceeds limit
        if slippage_bps > self.config.slippage_bps_limit:
            # Determine if odds are favorable or unfavorable
            if proposal.odds > proposal.fair_odds:
                direction = "favorable" if proposal.side.value == "BACK" else "unfavorable"
            else:
                direction = "unfavorable" if proposal.side.value == "BACK" else "favorable"
            
            # Calculate suggested odds within slippage limit
            max_slippage_ratio = self.config.slippage_bps_limit / 10000
            if proposal.odds > proposal.fair_odds:
                suggested_odds = proposal.fair_odds * (1 + max_slippage_ratio)
            else:
                suggested_odds = proposal.fair_odds * (1 - max_slippage_ratio)
            
            return RuleViolation(
                rule_id=self.rule_id,
                message=f"Slippage {slippage_bps:.0f}bps exceeds limit {self.config.slippage_bps_limit}bps. "
                       f"Requested odds {proposal.odds:.3f} vs fair odds {proposal.fair_odds:.3f} "
                       f"({direction} slippage). Consider odds closer to fair value.",
                current_value=slippage_bps,
                threshold=self.config.slippage_bps_limit,
                severity="WARNING" if slippage_bps < self.config.slippage_bps_limit * 1.5 else "ERROR",
                suggested_amendment={"odds": round(suggested_odds, 3)}
            )
        
        return None


class LiquidityFractionRule:
    """
    Rule: Market liquidity fraction limit
    
    Ensures that bets don't consume too large a fraction of available market liquidity,
    which could cause significant market impact and poor execution.
    """
    
    def __init__(self, config: LiquidityConfig):
        self.config = config
        self.rule_id = RuleId.LIQ_FRACTION_LIMIT
        
    def evaluate(self, proposal: BetProposal, exposure: ExposureSnapshot) -> Optional[RuleViolation]:
        """Evaluate liquidity fraction constraints"""
        
        # Skip if no liquidity information provided
        if not proposal.liquidity or proposal.liquidity.available <= 0:
            return None
        
        # Calculate fraction of available liquidity this bet would consume
        fraction_used = (proposal.stake / proposal.liquidity.available) * 100
        
        # Check if fraction exceeds limit
        if fraction_used > self.config.max_fraction_of_available_liquidity:
            # Calculate suggested stake within liquidity limit
            max_allowed_stake = proposal.liquidity.available * (self.config.max_fraction_of_available_liquidity / 100)
            
            # Analyze market depth if available
            depth_analysis = ""
            if proposal.liquidity.market_depth:
                total_depth_at_odds = sum(
                    depth.size for depth in proposal.liquidity.market_depth 
                    if abs(depth.odds - proposal.odds) < 0.01
                )
                if total_depth_at_odds > 0:
                    depth_analysis = f" Available at requested odds: {total_depth_at_odds:.0f}."
            
            return RuleViolation(
                rule_id=self.rule_id,
                message=f"Would consume {fraction_used:.1f}% of available liquidity "
                       f"(limit: {self.config.max_fraction_of_available_liquidity:.1f}%). "
                       f"Large orders can cause market impact and slippage.{depth_analysis}",
                current_value=fraction_used,
                threshold=self.config.max_fraction_of_available_liquidity,
                severity="WARNING" if fraction_used < self.config.max_fraction_of_available_liquidity * 1.2 else "ERROR",
                suggested_amendment={"stake": round(max_allowed_stake, 2)}
            )
        
        return None


class MarketDepthAnalysisRule:
    """
    Rule: Market depth analysis and warnings
    
    Analyzes market depth to provide warnings about potential execution issues,
    even if the bet doesn't violate hard limits.
    """
    
    def __init__(self, config: LiquidityConfig):
        self.config = config
        self.rule_id = RuleId.LIQ_FRACTION_LIMIT  # Reuse for depth analysis
        
    def evaluate(self, proposal: BetProposal, exposure: ExposureSnapshot) -> Optional[RuleViolation]:
        """Evaluate market depth and provide warnings"""
        
        # Skip if no market depth information
        if not proposal.liquidity or not proposal.liquidity.market_depth:
            return None
        
        market_depth = proposal.liquidity.market_depth
        
        # Find liquidity at or near requested odds
        liquidity_at_odds = 0
        liquidity_within_1_tick = 0
        
        for depth in market_depth:
            if abs(depth.odds - proposal.odds) < 0.005:  # Exact match (within 0.005)
                liquidity_at_odds += depth.size
            elif abs(depth.odds - proposal.odds) < 0.02:  # Within 1 tick (0.02)
                liquidity_within_1_tick += depth.size
        
        # Check if there's insufficient liquidity at requested odds
        if liquidity_at_odds < proposal.stake:
            shortage = proposal.stake - liquidity_at_odds
            
            if liquidity_within_1_tick >= proposal.stake:
                # Liquidity available within 1 tick
                return RuleViolation(
                    rule_id=self.rule_id,
                    message=f"Insufficient liquidity at exact odds {proposal.odds:.3f} "
                           f"(available: {liquidity_at_odds:.0f}, needed: {proposal.stake:.0f}). "
                           f"However, {liquidity_within_1_tick:.0f} available within 1 tick. "
                           f"May experience minor slippage.",
                    current_value=liquidity_at_odds,
                    threshold=proposal.stake,
                    severity="WARNING"
                )
            else:
                # Insufficient liquidity even within 1 tick
                return RuleViolation(
                    rule_id=self.rule_id,
                    message=f"Insufficient market depth for full execution. "
                           f"Available at odds: {liquidity_at_odds:.0f}, "
                           f"within 1 tick: {liquidity_within_1_tick:.0f}, "
                           f"required: {proposal.stake:.0f}. "
                           f"Order may experience significant slippage or partial fills.",
                    current_value=liquidity_within_1_tick,
                    threshold=proposal.stake,
                    severity="ERROR",
                    suggested_amendment={"stake": max(liquidity_at_odds, liquidity_within_1_tick)}
                )
        
        return None


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter implementation
    
    Provides smooth rate limiting with burst capacity while maintaining
    average rate limits over time.
    """
    
    def __init__(self, rate: float, capacity: int, window_seconds: int = 60):
        """
        Initialize token bucket rate limiter
        
        Args:
            rate: Tokens per second
            capacity: Maximum bucket capacity (burst size)
            window_seconds: Time window for rate calculation
        """
        self.rate = rate
        self.capacity = capacity
        self.window_seconds = window_seconds
        
        # Current state
        self.tokens = capacity
        self.last_update = time.time()
        
        # Thread safety
        self.lock = threading.RLock()
        
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if rate limited
        """
        with self.lock:
            now = time.time()
            
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + (elapsed * self.rate))
            self.last_update = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get estimated wait time for tokens to be available"""
        with self.lock:
            if self.tokens >= tokens:
                return 0.0
            
            needed_tokens = tokens - self.tokens
            return needed_tokens / self.rate


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter implementation
    
    Provides precise rate limiting over a sliding time window.
    """
    
    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize sliding window rate limiter
        
        Args:
            max_requests: Maximum requests in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
        # Request timestamps
        self.requests = deque()
        
        # Thread safety
        self.lock = threading.RLock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed under rate limit"""
        with self.lock:
            now = time.time()
            
            # Remove old requests outside the window
            cutoff_time = now - self.window_seconds
            while self.requests and self.requests[0] <= cutoff_time:
                self.requests.popleft()
            
            # Check if we're under the limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def get_reset_time(self) -> float:
        """Get time until rate limit resets"""
        with self.lock:
            if not self.requests:
                return 0.0
            
            oldest_request = self.requests[0]
            reset_time = oldest_request + self.window_seconds
            return max(0.0, reset_time - time.time())


class RateLimitRule:
    """
    Rule: Rate limiting for bet submissions
    
    Implements multiple rate limiting strategies to prevent abuse and ensure
    fair market access while protecting against DDoS attacks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize rate limiting rule
        
        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self.rule_id = RuleId.RATE_LIMIT_EXCEEDED
        
        # Rate limiters by key (market, user, global)
        self.market_limiters: Dict[str, SlidingWindowRateLimiter] = {}
        self.global_limiter = TokenBucketRateLimiter(
            rate=config.get('global_rate', 10.0),  # 10 requests per second
            capacity=config.get('global_burst', 50),  # Burst of 50
            window_seconds=60
        )
        
        # Configuration
        self.market_rate_limit = config.get('market_requests', 5)
        self.market_window = config.get('market_window_seconds', 120)
        
        # Thread safety
        self.lock = threading.RLock()
        
    def evaluate(self, proposal: BetProposal, exposure: ExposureSnapshot) -> Optional[RuleViolation]:
        """Evaluate rate limiting constraints"""
        
        # Check global rate limit
        if not self.global_limiter.consume():
            wait_time = self.global_limiter.get_wait_time()
            return RuleViolation(
                rule_id=self.rule_id,
                message=f"Global rate limit exceeded. Too many requests across all markets. "
                       f"Wait {wait_time:.1f} seconds before retrying.",
                current_value=0,  # No tokens available
                threshold=1,
                severity="ERROR"
            )
        
        # Check market-specific rate limit
        market_id = proposal.market_id
        
        with self.lock:
            if market_id not in self.market_limiters:
                self.market_limiters[market_id] = SlidingWindowRateLimiter(
                    max_requests=self.market_rate_limit,
                    window_seconds=self.market_window
                )
            
            market_limiter = self.market_limiters[market_id]
        
        if not market_limiter.is_allowed():
            reset_time = market_limiter.get_reset_time()
            return RuleViolation(
                rule_id=self.rule_id,
                message=f"Market rate limit exceeded for {market_id}. "
                       f"Maximum {self.market_rate_limit} requests per {self.market_window} seconds. "
                       f"Rate limit resets in {reset_time:.1f} seconds.",
                current_value=self.market_rate_limit,  # At limit
                threshold=self.market_rate_limit,
                severity="WARNING" if reset_time < 30 else "ERROR"
            )
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        with self.lock:
            return {
                "global_tokens_available": self.global_limiter.tokens,
                "global_capacity": self.global_limiter.capacity,
                "markets_tracked": len(self.market_limiters),
                "market_rate_limit": self.market_rate_limit,
                "market_window_seconds": self.market_window
            }


class LiquidityRuleEngine:
    """
    Liquidity rule engine that evaluates all liquidity and execution constraints
    """
    
    def __init__(self, config: LiquidityConfig, rate_limit_config: Optional[Dict[str, Any]] = None):
        self.config = config
        
        # Initialize all liquidity rules
        self.rules = [
            OddsRangeRule(config),
            SlippageRule(config),
            LiquidityFractionRule(config),
            MarketDepthAnalysisRule(config)
        ]
        
        # Initialize rate limiting if configured
        if rate_limit_config:
            self.rate_limit_rule = RateLimitRule(rate_limit_config)
            self.rules.append(self.rate_limit_rule)
        else:
            self.rate_limit_rule = None
        
        logger.info(f"Initialized LiquidityRuleEngine with {len(self.rules)} rules")
    
    def evaluate_all(self, proposal: BetProposal, exposure: ExposureSnapshot) -> List[RuleViolation]:
        """
        Evaluate all liquidity rules against a proposal
        
        Args:
            proposal: The bet proposal to evaluate
            exposure: Current exposure snapshot
            
        Returns:
            List of rule violations (empty if no violations)
        """
        violations = []
        
        for rule in self.rules:
            try:
                if hasattr(rule, 'evaluate'):
                    result = rule.evaluate(proposal, exposure)
                    
                    # Handle both single violations and lists
                    if result:
                        if isinstance(result, list):
                            violations.extend(result)
                        else:
                            violations.append(result)
                        
                        logger.debug(f"Liquidity rule violation: {rule.__class__.__name__}")
                        
            except Exception as e:
                logger.error(f"Error evaluating liquidity rule {rule.__class__.__name__}: {str(e)}")
                # Create a generic violation for rule evaluation errors
                violations.append(RuleViolation(
                    rule_id=getattr(rule, 'rule_id', RuleId.LIQ_FRACTION_LIMIT),
                    message=f"Liquidity rule evaluation error: {str(e)}",
                    severity="ERROR"
                ))
        
        return violations
    
    def get_liquidity_analysis(self, proposal: BetProposal) -> Dict[str, Any]:
        """Get detailed liquidity analysis for a proposal"""
        analysis = {
            "odds_analysis": {
                "requested_odds": proposal.odds,
                "fair_odds": proposal.fair_odds,
                "within_range": (
                    self.config.min_odds_threshold <= proposal.odds <= self.config.max_odds_threshold
                )
            },
            "liquidity_analysis": {},
            "rate_limit_status": {}
        }
        
        # Slippage analysis
        if proposal.fair_odds:
            slippage_bps = abs((proposal.odds - proposal.fair_odds) / proposal.fair_odds) * 10000
            analysis["slippage_analysis"] = {
                "slippage_bps": slippage_bps,
                "within_limit": slippage_bps <= self.config.slippage_bps_limit,
                "limit_bps": self.config.slippage_bps_limit
            }
        
        # Liquidity analysis
        if proposal.liquidity:
            fraction_used = (proposal.stake / proposal.liquidity.available) * 100
            analysis["liquidity_analysis"] = {
                "available_liquidity": proposal.liquidity.available,
                "requested_stake": proposal.stake,
                "fraction_used_pct": fraction_used,
                "within_limit": fraction_used <= self.config.max_fraction_of_available_liquidity,
                "limit_pct": self.config.max_fraction_of_available_liquidity
            }
        
        # Rate limit status
        if self.rate_limit_rule:
            analysis["rate_limit_status"] = self.rate_limit_rule.get_statistics()
        
        return analysis
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about liquidity rules"""
        stats = {
            "total_rules": len(self.rules),
            "rule_types": [rule.__class__.__name__ for rule in self.rules],
            "config": {
                "min_odds_threshold": self.config.min_odds_threshold,
                "max_odds_threshold": self.config.max_odds_threshold,
                "slippage_bps_limit": self.config.slippage_bps_limit,
                "max_fraction_of_available_liquidity": self.config.max_fraction_of_available_liquidity
            }
        }
        
        # Add rate limiting statistics if available
        if self.rate_limit_rule:
            stats["rate_limiting"] = self.rate_limit_rule.get_statistics()
        
        return stats
