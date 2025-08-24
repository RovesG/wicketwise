# Purpose: Scenario generator for comprehensive DGL testing
# Author: WicketWise AI, Last Modified: 2024

"""
Scenario Generator

Generates realistic testing scenarios for DGL validation:
- Edge case scenarios
- Stress test conditions
- Production-like workloads
- Regulatory compliance scenarios
"""

import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import math

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import BetProposal, BetSide, LiquidityInfo, MarketDepth
from client.orchestrator_mock import MarketType


logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of test scenarios"""
    NORMAL_OPERATIONS = "normal_operations"
    EDGE_CASES = "edge_cases"
    STRESS_CONDITIONS = "stress_conditions"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    MARKET_VOLATILITY = "market_volatility"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    HIGH_VOLUME = "high_volume"
    RISK_LIMITS = "risk_limits"


@dataclass
class ScenarioConfig:
    """Configuration for scenario generation"""
    scenario_type: ScenarioType
    num_proposals: int = 100
    time_span_hours: int = 24
    market_conditions: Dict[str, float] = None
    risk_parameters: Dict[str, float] = None
    custom_rules: Dict[str, Any] = None


class ScenarioGenerator:
    """
    Generates comprehensive test scenarios for DGL validation
    
    Creates realistic and edge-case scenarios to thoroughly test
    DGL behavior under various market and operational conditions.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize scenario generator
        
        Args:
            seed: Random seed for reproducible scenarios
        """
        if seed:
            random.seed(seed)
        
        self.generated_scenarios = []
        
        # Market data templates
        self.match_templates = self._create_match_templates()
        self.market_templates = self._create_market_templates()
        
        logger.info("Scenario generator initialized")
    
    def _create_match_templates(self) -> List[Dict[str, Any]]:
        """Create realistic cricket match templates"""
        return [
            {
                "format": "Test",
                "teams": ["England", "Australia"],
                "venue": "Lords",
                "expected_duration_hours": 40,
                "typical_total_runs": 350,
                "volatility": 0.3
            },
            {
                "format": "ODI", 
                "teams": ["India", "Pakistan"],
                "venue": "Mumbai",
                "expected_duration_hours": 8,
                "typical_total_runs": 280,
                "volatility": 0.4
            },
            {
                "format": "T20",
                "teams": ["England", "South Africa"],
                "venue": "Cape Town",
                "expected_duration_hours": 3,
                "typical_total_runs": 160,
                "volatility": 0.6
            },
            {
                "format": "T20",
                "teams": ["Australia", "New Zealand"],
                "venue": "Melbourne",
                "expected_duration_hours": 3,
                "typical_total_runs": 170,
                "volatility": 0.5
            }
        ]
    
    def _create_market_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create market type templates with typical characteristics"""
        return {
            "match_winner": {
                "typical_odds_range": (1.4, 4.0),
                "liquidity_factor": 1.0,
                "volatility": 0.2
            },
            "total_runs": {
                "typical_odds_range": (1.7, 2.3),
                "liquidity_factor": 0.8,
                "volatility": 0.3
            },
            "top_batsman": {
                "typical_odds_range": (3.0, 15.0),
                "liquidity_factor": 0.6,
                "volatility": 0.5
            },
            "method_of_dismissal": {
                "typical_odds_range": (2.0, 8.0),
                "liquidity_factor": 0.4,
                "volatility": 0.7
            },
            "innings_runs": {
                "typical_odds_range": (1.8, 2.2),
                "liquidity_factor": 0.7,
                "volatility": 0.4
            }
        }
    
    async def generate_scenario(self, config: ScenarioConfig) -> List[BetProposal]:
        """
        Generate test scenario based on configuration
        
        Args:
            config: Scenario configuration
            
        Returns:
            List of bet proposals for the scenario
        """
        logger.info(f"Generating {config.scenario_type.value} scenario with {config.num_proposals} proposals")
        
        if config.scenario_type == ScenarioType.NORMAL_OPERATIONS:
            return await self._generate_normal_operations(config)
        elif config.scenario_type == ScenarioType.EDGE_CASES:
            return await self._generate_edge_cases(config)
        elif config.scenario_type == ScenarioType.STRESS_CONDITIONS:
            return await self._generate_stress_conditions(config)
        elif config.scenario_type == ScenarioType.REGULATORY_COMPLIANCE:
            return await self._generate_regulatory_compliance(config)
        elif config.scenario_type == ScenarioType.MARKET_VOLATILITY:
            return await self._generate_market_volatility(config)
        elif config.scenario_type == ScenarioType.LIQUIDITY_CRISIS:
            return await self._generate_liquidity_crisis(config)
        elif config.scenario_type == ScenarioType.HIGH_VOLUME:
            return await self._generate_high_volume(config)
        elif config.scenario_type == ScenarioType.RISK_LIMITS:
            return await self._generate_risk_limits(config)
        else:
            raise ValueError(f"Unknown scenario type: {config.scenario_type}")
    
    async def _generate_normal_operations(self, config: ScenarioConfig) -> List[BetProposal]:
        """Generate normal operational scenario"""
        proposals = []
        
        for i in range(config.num_proposals):
            # Select random match and market
            match_template = random.choice(self.match_templates)
            market_type = random.choice(list(self.market_templates.keys()))
            market_template = self.market_templates[market_type]
            
            # Generate realistic parameters
            odds_range = market_template["typical_odds_range"]
            odds = random.uniform(*odds_range)
            
            # Stake distribution (most bets are small to medium)
            if random.random() < 0.7:  # 70% small bets
                stake = random.uniform(100, 500)
            elif random.random() < 0.9:  # 20% medium bets
                stake = random.uniform(500, 1500)
            else:  # 10% large bets
                stake = random.uniform(1500, 3000)
            
            # Generate liquidity
            base_liquidity = random.uniform(10000, 50000) * market_template["liquidity_factor"]
            liquidity = self._generate_liquidity(odds, base_liquidity)
            
            proposal = BetProposal(
                market_id=f"match_{i % 5}_{market_type}",
                match_id=f"match_{i % 5}",
                side=random.choice([BetSide.BACK, BetSide.LAY]),
                selection=self._generate_selection(market_type, match_template),
                odds=odds,
                stake=stake,
                model_confidence=random.uniform(0.7, 0.95),
                fair_odds=odds * random.uniform(0.95, 1.05),
                expected_edge_pct=random.uniform(1.0, 8.0),
                liquidity=liquidity,
                features={
                    "scenario_type": "normal_operations",
                    "match_format": match_template["format"],
                    "market_type": market_type,
                    "proposal_index": i
                }
            )
            
            proposals.append(proposal)
        
        return proposals
    
    async def _generate_edge_cases(self, config: ScenarioConfig) -> List[BetProposal]:
        """Generate edge case scenarios"""
        proposals = []
        
        edge_cases = [
            # Minimum values
            {"odds": 1.01, "stake": 1.0, "confidence": 0.01},
            {"odds": 1.02, "stake": 5.0, "confidence": 0.1},
            
            # Maximum values  
            {"odds": 999.0, "stake": 50000.0, "confidence": 0.99},
            {"odds": 100.0, "stake": 10000.0, "confidence": 1.0},
            
            # Boundary conditions
            {"odds": 1.25, "stake": 2000.0, "confidence": 0.5},  # Min odds threshold
            {"odds": 10.0, "stake": 2000.0, "confidence": 0.5},   # Max odds threshold
            
            # Unusual combinations
            {"odds": 50.0, "stake": 100.0, "confidence": 0.9},   # High odds, low stake, high confidence
            {"odds": 1.1, "stake": 5000.0, "confidence": 0.3},   # Low odds, high stake, low confidence
        ]
        
        for i, edge_case in enumerate(edge_cases * (config.num_proposals // len(edge_cases) + 1)):
            if len(proposals) >= config.num_proposals:
                break
            
            match_template = random.choice(self.match_templates)
            market_type = random.choice(list(self.market_templates.keys()))
            
            # Create minimal or no liquidity for some edge cases
            if random.random() < 0.3:
                liquidity = LiquidityInfo(
                    available=edge_case["stake"] * 0.5,  # Insufficient liquidity
                    market_depth=[MarketDepth(odds=edge_case["odds"], size=edge_case["stake"] * 0.3)]
                )
            else:
                liquidity = self._generate_liquidity(edge_case["odds"], 20000)
            
            proposal = BetProposal(
                market_id=f"edge_case_{i}_{market_type}",
                match_id=f"edge_match_{i % 3}",
                side=random.choice([BetSide.BACK, BetSide.LAY]),
                selection=self._generate_selection(market_type, match_template),
                odds=edge_case["odds"],
                stake=edge_case["stake"],
                model_confidence=edge_case["confidence"],
                fair_odds=edge_case["odds"] * random.uniform(0.8, 1.2),
                expected_edge_pct=random.uniform(-5.0, 15.0),  # Include negative edges
                liquidity=liquidity,
                features={
                    "scenario_type": "edge_cases",
                    "edge_case_type": f"case_{i % len(edge_cases)}",
                    "match_format": match_template["format"]
                }
            )
            
            proposals.append(proposal)
        
        return proposals[:config.num_proposals]
    
    async def _generate_stress_conditions(self, config: ScenarioConfig) -> List[BetProposal]:
        """Generate stress test conditions"""
        proposals = []
        
        # Stress patterns
        patterns = [
            "rapid_fire",      # Many bets in short time
            "large_stakes",    # Unusually large stakes
            "volatile_odds",   # Rapidly changing odds
            "low_liquidity",   # Poor liquidity conditions
            "high_correlation" # Many correlated bets
        ]
        
        for i in range(config.num_proposals):
            pattern = patterns[i % len(patterns)]
            match_template = random.choice(self.match_templates)
            market_type = random.choice(list(self.market_templates.keys()))
            
            if pattern == "rapid_fire":
                # Small to medium bets, high frequency
                odds = random.uniform(1.5, 4.0)
                stake = random.uniform(200, 800)
                confidence = random.uniform(0.6, 0.9)
                
            elif pattern == "large_stakes":
                # Very large stakes
                odds = random.uniform(1.8, 3.0)
                stake = random.uniform(5000, 20000)
                confidence = random.uniform(0.8, 0.95)
                
            elif pattern == "volatile_odds":
                # Extreme odds
                if random.random() < 0.5:
                    odds = random.uniform(1.01, 1.3)  # Very low odds
                else:
                    odds = random.uniform(15.0, 100.0)  # Very high odds
                stake = random.uniform(500, 2000)
                confidence = random.uniform(0.4, 0.8)
                
            elif pattern == "low_liquidity":
                # Normal parameters but poor liquidity
                odds = random.uniform(2.0, 5.0)
                stake = random.uniform(1000, 3000)
                confidence = random.uniform(0.7, 0.9)
                
            else:  # high_correlation
                # Many bets on same match/market
                odds = random.uniform(1.8, 2.5)
                stake = random.uniform(800, 1500)
                confidence = random.uniform(0.75, 0.95)
            
            # Generate appropriate liquidity
            if pattern == "low_liquidity":
                base_liquidity = stake * random.uniform(0.5, 1.2)  # Tight liquidity
            else:
                base_liquidity = random.uniform(15000, 40000)
            
            liquidity = self._generate_liquidity(odds, base_liquidity)
            
            # For high correlation, use same match/market IDs
            if pattern == "high_correlation":
                market_id = f"stress_corr_{market_type}"
                match_id = "stress_match_1"
            else:
                market_id = f"stress_{pattern}_{i}_{market_type}"
                match_id = f"stress_match_{i % 10}"
            
            proposal = BetProposal(
                market_id=market_id,
                match_id=match_id,
                side=random.choice([BetSide.BACK, BetSide.LAY]),
                selection=self._generate_selection(market_type, match_template),
                odds=odds,
                stake=stake,
                model_confidence=confidence,
                fair_odds=odds * random.uniform(0.9, 1.1),
                expected_edge_pct=random.uniform(0.5, 12.0),
                liquidity=liquidity,
                features={
                    "scenario_type": "stress_conditions",
                    "stress_pattern": pattern,
                    "match_format": match_template["format"]
                }
            )
            
            proposals.append(proposal)
        
        return proposals
    
    async def _generate_regulatory_compliance(self, config: ScenarioConfig) -> List[BetProposal]:
        """Generate regulatory compliance test scenarios"""
        proposals = []
        
        # Compliance test cases
        compliance_cases = [
            "dual_approval_threshold",  # Bets requiring dual approval
            "jurisdiction_limits",      # Jurisdiction-specific limits
            "currency_validation",      # Different currencies
            "blocked_markets",          # Blocked market types
            "audit_trail",             # Audit trail validation
            "mfa_requirements"         # MFA requirement triggers
        ]
        
        for i in range(config.num_proposals):
            case = compliance_cases[i % len(compliance_cases)]
            match_template = random.choice(self.match_templates)
            market_type = random.choice(list(self.market_templates.keys()))
            
            if case == "dual_approval_threshold":
                # Stakes above dual approval threshold (£2000)
                stake = random.uniform(2100, 5000)
                odds = random.uniform(1.5, 3.0)
                
            elif case == "jurisdiction_limits":
                # Test different jurisdiction rules
                stake = random.uniform(500, 2000)
                odds = random.uniform(1.8, 4.0)
                
            elif case == "currency_validation":
                # Different currencies
                stake = random.uniform(300, 1500)
                odds = random.uniform(1.6, 3.5)
                
            elif case == "blocked_markets":
                # Markets that might be blocked
                stake = random.uniform(400, 1200)
                odds = random.uniform(2.0, 6.0)
                
            else:  # audit_trail, mfa_requirements
                stake = random.uniform(800, 2500)
                odds = random.uniform(1.7, 4.5)
            
            liquidity = self._generate_liquidity(odds, random.uniform(20000, 60000))
            
            proposal = BetProposal(
                market_id=f"compliance_{case}_{i}_{market_type}",
                match_id=f"compliance_match_{i % 5}",
                side=random.choice([BetSide.BACK, BetSide.LAY]),
                selection=self._generate_selection(market_type, match_template),
                odds=odds,
                stake=stake,
                currency="GBP" if case != "currency_validation" else random.choice(["GBP", "USD", "EUR"]),
                model_confidence=random.uniform(0.7, 0.9),
                fair_odds=odds * random.uniform(0.95, 1.05),
                expected_edge_pct=random.uniform(2.0, 6.0),
                liquidity=liquidity,
                features={
                    "scenario_type": "regulatory_compliance",
                    "compliance_case": case,
                    "requires_dual_approval": stake > 2000,
                    "match_format": match_template["format"]
                }
            )
            
            proposals.append(proposal)
        
        return proposals
    
    async def _generate_market_volatility(self, config: ScenarioConfig) -> List[BetProposal]:
        """Generate market volatility scenarios"""
        proposals = []
        
        # Simulate market events that cause volatility
        volatility_events = [
            "wicket_fall", "boundary_hit", "weather_delay", 
            "injury_concern", "tactical_change", "momentum_shift"
        ]
        
        for i in range(config.num_proposals):
            event = volatility_events[i % len(volatility_events)]
            match_template = random.choice(self.match_templates)
            market_type = random.choice(list(self.market_templates.keys()))
            
            # Adjust parameters based on volatility event
            base_volatility = match_template["volatility"]
            
            if event in ["wicket_fall", "injury_concern"]:
                # High impact events - significant odds movement
                odds_volatility = base_volatility * 2.0
                liquidity_impact = 0.6  # Reduced liquidity
            elif event in ["boundary_hit", "tactical_change"]:
                # Medium impact events
                odds_volatility = base_volatility * 1.5
                liquidity_impact = 0.8
            else:
                # Lower impact events
                odds_volatility = base_volatility * 1.2
                liquidity_impact = 0.9
            
            # Generate odds with volatility
            base_odds = random.uniform(*self.market_templates[market_type]["typical_odds_range"])
            odds_movement = random.uniform(-odds_volatility, odds_volatility)
            odds = max(1.01, base_odds * (1 + odds_movement))
            
            stake = random.uniform(300, 2000)
            
            # Adjust liquidity based on market impact
            base_liquidity = random.uniform(15000, 45000) * liquidity_impact
            liquidity = self._generate_liquidity(odds, base_liquidity)
            
            proposal = BetProposal(
                market_id=f"volatile_{event}_{i}_{market_type}",
                match_id=f"volatile_match_{i % 3}",
                side=random.choice([BetSide.BACK, BetSide.LAY]),
                selection=self._generate_selection(market_type, match_template),
                odds=odds,
                stake=stake,
                model_confidence=random.uniform(0.5, 0.85),  # Lower confidence in volatile conditions
                fair_odds=odds * random.uniform(0.85, 1.15),  # Wider fair odds spread
                expected_edge_pct=random.uniform(-2.0, 10.0),
                liquidity=liquidity,
                features={
                    "scenario_type": "market_volatility",
                    "volatility_event": event,
                    "odds_volatility": odds_volatility,
                    "match_format": match_template["format"]
                }
            )
            
            proposals.append(proposal)
        
        return proposals
    
    async def _generate_liquidity_crisis(self, config: ScenarioConfig) -> List[BetProposal]:
        """Generate liquidity crisis scenarios"""
        proposals = []
        
        for i in range(config.num_proposals):
            match_template = random.choice(self.match_templates)
            market_type = random.choice(list(self.market_templates.keys()))
            
            # Crisis severity increases over time
            crisis_severity = min(1.0, i / (config.num_proposals * 0.7))
            
            odds = random.uniform(*self.market_templates[market_type]["typical_odds_range"])
            stake = random.uniform(500, 2500)
            
            # Severely reduced liquidity
            normal_liquidity = random.uniform(20000, 50000)
            crisis_liquidity = normal_liquidity * (1.0 - crisis_severity * 0.8)  # Up to 80% reduction
            
            # Create sparse market depth
            depth_levels = max(1, int(3 * (1 - crisis_severity)))
            market_depth = []
            
            for j in range(depth_levels):
                depth_odds = odds + (j * 0.05 * random.choice([-1, 1]))
                depth_size = crisis_liquidity * random.uniform(0.1, 0.4) / depth_levels
                market_depth.append(MarketDepth(odds=depth_odds, size=depth_size))
            
            liquidity = LiquidityInfo(
                available=crisis_liquidity,
                market_depth=market_depth
            )
            
            proposal = BetProposal(
                market_id=f"crisis_{i}_{market_type}",
                match_id=f"crisis_match_{i % 4}",
                side=random.choice([BetSide.BACK, BetSide.LAY]),
                selection=self._generate_selection(market_type, match_template),
                odds=odds,
                stake=stake,
                model_confidence=random.uniform(0.6, 0.8),
                fair_odds=odds * random.uniform(0.9, 1.1),
                expected_edge_pct=random.uniform(1.0, 8.0),
                liquidity=liquidity,
                features={
                    "scenario_type": "liquidity_crisis",
                    "crisis_severity": crisis_severity,
                    "liquidity_reduction_pct": crisis_severity * 80,
                    "match_format": match_template["format"]
                }
            )
            
            proposals.append(proposal)
        
        return proposals
    
    async def _generate_high_volume(self, config: ScenarioConfig) -> List[BetProposal]:
        """Generate high volume scenarios"""
        proposals = []
        
        # Simulate busy periods with high proposal volume
        for i in range(config.num_proposals):
            match_template = random.choice(self.match_templates)
            market_type = random.choice(list(self.market_templates.keys()))
            
            # Cluster proposals around popular matches/markets
            popular_match_id = f"popular_match_{i % 3}"  # Only 3 popular matches
            popular_market_types = ["match_winner", "total_runs"]  # Popular markets
            
            if random.random() < 0.7:  # 70% on popular markets
                market_type = random.choice(popular_market_types)
            
            odds = random.uniform(*self.market_templates[market_type]["typical_odds_range"])
            
            # Varied stake sizes but trending smaller for high volume
            if random.random() < 0.8:
                stake = random.uniform(100, 800)  # Smaller bets
            else:
                stake = random.uniform(800, 2000)  # Some larger bets
            
            liquidity = self._generate_liquidity(odds, random.uniform(25000, 75000))
            
            proposal = BetProposal(
                market_id=f"{popular_match_id}_{market_type}",
                match_id=popular_match_id,
                side=random.choice([BetSide.BACK, BetSide.LAY]),
                selection=self._generate_selection(market_type, match_template),
                odds=odds,
                stake=stake,
                model_confidence=random.uniform(0.65, 0.9),
                fair_odds=odds * random.uniform(0.95, 1.05),
                expected_edge_pct=random.uniform(1.5, 6.0),
                liquidity=liquidity,
                features={
                    "scenario_type": "high_volume",
                    "volume_cluster": i // 20,  # Group into clusters
                    "match_format": match_template["format"]
                }
            )
            
            proposals.append(proposal)
        
        return proposals
    
    async def _generate_risk_limits(self, config: ScenarioConfig) -> List[BetProposal]:
        """Generate risk limit testing scenarios"""
        proposals = []
        
        # Test different risk limit scenarios
        risk_scenarios = [
            "bankroll_limit_test",
            "match_exposure_test", 
            "market_concentration_test",
            "correlation_limit_test",
            "daily_loss_limit_test"
        ]
        
        for i in range(config.num_proposals):
            scenario = risk_scenarios[i % len(risk_scenarios)]
            match_template = random.choice(self.match_templates)
            market_type = random.choice(list(self.market_templates.keys()))
            
            if scenario == "bankroll_limit_test":
                # Test bankroll exposure limits (aim for 3-6% of assumed £100k bankroll)
                stake = random.uniform(3000, 6000)
                odds = random.uniform(1.5, 3.0)
                
            elif scenario == "match_exposure_test":
                # Test per-match limits (use same match ID)
                stake = random.uniform(1500, 3000)
                odds = random.uniform(1.8, 4.0)
                match_id = "risk_test_match_1"  # Concentrate on one match
                
            elif scenario == "market_concentration_test":
                # Test market concentration (same market type)
                stake = random.uniform(800, 2000)
                odds = random.uniform(2.0, 5.0)
                market_type = "match_winner"  # Concentrate on one market type
                
            elif scenario == "correlation_limit_test":
                # Test correlated positions
                stake = random.uniform(1000, 2500)
                odds = random.uniform(1.6, 3.5)
                
            else:  # daily_loss_limit_test
                # Test daily P&L limits
                stake = random.uniform(2000, 4000)
                odds = random.uniform(1.4, 2.5)
            
            # Set match_id for concentration tests
            if scenario == "match_exposure_test":
                match_id = "risk_test_match_1"
            elif scenario == "correlation_limit_test":
                match_id = f"corr_match_{i % 2}"  # Two correlated matches
            else:
                match_id = f"risk_match_{i % 8}"
            
            liquidity = self._generate_liquidity(odds, random.uniform(30000, 80000))
            
            proposal = BetProposal(
                market_id=f"{match_id}_{market_type}",
                match_id=match_id,
                side=random.choice([BetSide.BACK, BetSide.LAY]),
                selection=self._generate_selection(market_type, match_template),
                odds=odds,
                stake=stake,
                model_confidence=random.uniform(0.7, 0.95),
                fair_odds=odds * random.uniform(0.95, 1.05),
                expected_edge_pct=random.uniform(2.0, 7.0),
                liquidity=liquidity,
                features={
                    "scenario_type": "risk_limits",
                    "risk_scenario": scenario,
                    "match_format": match_template["format"]
                }
            )
            
            proposals.append(proposal)
        
        return proposals
    
    def _generate_liquidity(self, odds: float, base_amount: float) -> LiquidityInfo:
        """Generate realistic liquidity information"""
        # Create market depth around the odds
        depth_levels = random.randint(2, 5)
        market_depth = []
        
        for i in range(depth_levels):
            # Spread odds around the target
            odds_offset = (i - depth_levels // 2) * 0.02
            depth_odds = max(1.01, odds + odds_offset)
            
            # Size decreases as we move away from best odds
            size_factor = 1.0 / (1 + abs(i - depth_levels // 2) * 0.3)
            depth_size = base_amount * size_factor * random.uniform(0.2, 0.5)
            
            market_depth.append(MarketDepth(odds=depth_odds, size=depth_size))
        
        return LiquidityInfo(
            available=base_amount,
            market_depth=market_depth
        )
    
    def _generate_selection(self, market_type: str, match_template: Dict[str, Any]) -> str:
        """Generate realistic selection based on market type and match"""
        teams = match_template["teams"]
        
        if market_type == "match_winner":
            if match_template["format"] == "Test":
                return random.choice([teams[0], teams[1], "Draw"])
            else:
                return random.choice(teams)
        elif market_type == "total_runs":
            total = match_template["typical_total_runs"]
            line = total + random.randint(-30, 30)
            return random.choice([f"Over {line}.5", f"Under {line}.5"])
        elif market_type == "top_batsman":
            team = random.choice(teams)
            player_num = random.randint(1, 11)
            return f"{team}_Player_{player_num}"
        elif market_type == "method_of_dismissal":
            methods = ["Bowled", "Caught", "LBW", "Run Out", "Stumped"]
            return random.choice(methods)
        elif market_type == "innings_runs":
            innings = random.choice(["1st", "2nd"])
            return f"{innings} Innings Total"
        else:
            return f"Selection_{random.randint(1, 5)}"
    
    def get_scenario_summary(self) -> Dict[str, Any]:
        """Get summary of generated scenarios"""
        return {
            "total_scenarios_generated": len(self.generated_scenarios),
            "available_scenario_types": [t.value for t in ScenarioType],
            "match_templates": len(self.match_templates),
            "market_templates": len(self.market_templates)
        }


# Convenience functions for quick scenario generation

async def generate_comprehensive_test_suite(num_proposals_per_scenario: int = 50) -> Dict[ScenarioType, List[BetProposal]]:
    """
    Generate comprehensive test suite with all scenario types
    
    Args:
        num_proposals_per_scenario: Number of proposals per scenario type
        
    Returns:
        Dictionary mapping scenario types to proposal lists
    """
    generator = ScenarioGenerator(seed=42)  # Reproducible scenarios
    
    test_suite = {}
    
    for scenario_type in ScenarioType:
        config = ScenarioConfig(
            scenario_type=scenario_type,
            num_proposals=num_proposals_per_scenario
        )
        
        proposals = await generator.generate_scenario(config)
        test_suite[scenario_type] = proposals
        
        logger.info(f"Generated {len(proposals)} proposals for {scenario_type.value}")
    
    return test_suite


async def generate_quick_test_scenarios(total_proposals: int = 100) -> List[BetProposal]:
    """
    Generate quick mixed test scenarios
    
    Args:
        total_proposals: Total number of proposals to generate
        
    Returns:
        Mixed list of test proposals
    """
    generator = ScenarioGenerator()
    
    # Mix of scenario types
    scenarios = [
        (ScenarioType.NORMAL_OPERATIONS, 0.4),
        (ScenarioType.EDGE_CASES, 0.2),
        (ScenarioType.STRESS_CONDITIONS, 0.15),
        (ScenarioType.MARKET_VOLATILITY, 0.15),
        (ScenarioType.RISK_LIMITS, 0.1)
    ]
    
    all_proposals = []
    
    for scenario_type, proportion in scenarios:
        num_proposals = int(total_proposals * proportion)
        if num_proposals > 0:
            config = ScenarioConfig(
                scenario_type=scenario_type,
                num_proposals=num_proposals
            )
            
            proposals = await generator.generate_scenario(config)
            all_proposals.extend(proposals)
    
    # Shuffle to mix scenarios
    random.shuffle(all_proposals)
    
    return all_proposals[:total_proposals]
