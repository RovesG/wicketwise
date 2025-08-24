# Purpose: Mock orchestrator for DGL integration testing
# Author: WicketWise AI, Last Modified: 2024

"""
Mock Orchestrator

Simulates a betting orchestrator system that integrates with DGL:
- Generates realistic bet proposals
- Simulates market conditions
- Tests DGL integration patterns
- Provides performance benchmarking
"""

import asyncio
import random
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import (
    BetProposal, BetSide, DecisionType, GovernanceDecision,
    LiquidityInfo, MarketDepth
)
from client.dgl_client import DGLClient, DGLClientConfig


logger = logging.getLogger(__name__)


class MarketType(Enum):
    """Types of betting markets"""
    MATCH_WINNER = "match_winner"
    TOTAL_RUNS = "total_runs"
    TOP_BATSMAN = "top_batsman"
    METHOD_OF_DISMISSAL = "method_of_dismissal"
    INNINGS_RUNS = "innings_runs"


class OrchestratorMode(Enum):
    """Orchestrator operating modes"""
    CONSERVATIVE = "conservative"  # Low risk, high approval rate
    AGGRESSIVE = "aggressive"     # Higher risk, more rejections expected
    BALANCED = "balanced"         # Mix of risk levels
    STRESS_TEST = "stress_test"   # High volume, rapid requests


@dataclass
class MarketCondition:
    """Market condition simulation"""
    liquidity_multiplier: float = 1.0
    volatility_factor: float = 1.0
    slippage_factor: float = 1.0
    odds_drift: float = 0.0


@dataclass
class OrchestratorStats:
    """Orchestrator performance statistics"""
    total_proposals: int = 0
    approved_proposals: int = 0
    rejected_proposals: int = 0
    amended_proposals: int = 0
    errors: int = 0
    avg_response_time_ms: float = 0.0
    total_stake_requested: float = 0.0
    total_stake_approved: float = 0.0


class MockOrchestrator:
    """
    Mock betting orchestrator for DGL integration testing
    
    Simulates realistic betting scenarios and integrates with DGL
    for governance decisions.
    """
    
    def __init__(
        self,
        dgl_client: DGLClient,
        mode: OrchestratorMode = OrchestratorMode.BALANCED,
        market_condition: Optional[MarketCondition] = None
    ):
        """
        Initialize mock orchestrator
        
        Args:
            dgl_client: DGL client for governance integration
            mode: Operating mode for different test scenarios
            market_condition: Market condition simulation parameters
        """
        self.dgl_client = dgl_client
        self.mode = mode
        self.market_condition = market_condition or MarketCondition()
        
        # Statistics tracking
        self.stats = OrchestratorStats()
        self.response_times: List[float] = []
        
        # Market data simulation
        self.active_matches = self._generate_active_matches()
        self.market_prices = self._initialize_market_prices()
        
        # Callback hooks for custom behavior
        self.pre_proposal_hook: Optional[Callable] = None
        self.post_decision_hook: Optional[Callable] = None
        
        logger.info(f"Mock orchestrator initialized in {mode.value} mode")
    
    def _generate_active_matches(self) -> List[Dict[str, Any]]:
        """Generate mock active cricket matches"""
        matches = []
        
        match_templates = [
            {"home": "England", "away": "Australia", "format": "Test", "venue": "Lords"},
            {"home": "India", "away": "Pakistan", "format": "ODI", "venue": "Mumbai"},
            {"home": "South Africa", "away": "New Zealand", "format": "T20", "venue": "Cape Town"},
            {"home": "West Indies", "away": "Sri Lanka", "format": "ODI", "venue": "Bridgetown"},
            {"home": "Bangladesh", "away": "Afghanistan", "format": "T20", "venue": "Dhaka"}
        ]
        
        for i, template in enumerate(match_templates):
            match = {
                "match_id": f"match_{i+1:03d}",
                "home_team": template["home"],
                "away_team": template["away"],
                "format": template["format"],
                "venue": template["venue"],
                "start_time": datetime.now() + timedelta(hours=random.randint(1, 48)),
                "status": "scheduled",
                "current_score": None
            }
            matches.append(match)
        
        return matches
    
    def _initialize_market_prices(self) -> Dict[str, Dict[str, float]]:
        """Initialize market prices for active matches"""
        prices = {}
        
        for match in self.active_matches:
            match_id = match["match_id"]
            prices[match_id] = {
                # Match winner odds
                f"{match['home_team']}_win": random.uniform(1.8, 3.5),
                f"{match['away_team']}_win": random.uniform(1.8, 3.5),
                "draw": random.uniform(3.0, 8.0) if match["format"] == "Test" else None,
                
                # Total runs markets
                "total_runs_over_300": random.uniform(1.7, 2.3),
                "total_runs_under_300": random.uniform(1.7, 2.3),
                
                # Top batsman (simplified)
                "top_batsman_home": random.uniform(3.0, 8.0),
                "top_batsman_away": random.uniform(3.0, 8.0)
            }
        
        return prices
    
    async def generate_proposal(
        self,
        match_id: Optional[str] = None,
        market_type: Optional[MarketType] = None,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> BetProposal:
        """
        Generate a realistic bet proposal
        
        Args:
            match_id: Specific match to bet on (random if None)
            market_type: Type of market (random if None)
            custom_params: Custom proposal parameters
            
        Returns:
            Generated bet proposal
        """
        # Select match
        if match_id:
            match = next((m for m in self.active_matches if m["match_id"] == match_id), None)
            if not match:
                raise ValueError(f"Match {match_id} not found")
        else:
            match = random.choice(self.active_matches)
        
        # Select market type
        if not market_type:
            market_type = random.choice(list(MarketType))
        
        # Generate proposal based on mode
        if self.mode == OrchestratorMode.CONSERVATIVE:
            stake_range = (100, 500)
            odds_range = (1.5, 3.0)
        elif self.mode == OrchestratorMode.AGGRESSIVE:
            stake_range = (500, 2000)
            odds_range = (1.2, 8.0)
        elif self.mode == OrchestratorMode.STRESS_TEST:
            stake_range = (50, 1000)
            odds_range = (1.1, 15.0)
        else:  # BALANCED
            stake_range = (200, 1000)
            odds_range = (1.4, 5.0)
        
        # Generate proposal parameters
        stake = random.uniform(*stake_range)
        odds = random.uniform(*odds_range)
        side = random.choice([BetSide.BACK, BetSide.LAY])
        
        # Apply market conditions
        odds *= (1 + self.market_condition.odds_drift)
        
        # Generate market-specific selection
        if market_type == MarketType.MATCH_WINNER:
            selections = [match["home_team"], match["away_team"]]
            if match["format"] == "Test":
                selections.append("Draw")
            selection = random.choice(selections)
            market_id = f"{match['match_id']}_winner"
        elif market_type == MarketType.TOTAL_RUNS:
            selection = random.choice(["Over 300.5", "Under 300.5", "Over 250.5", "Under 250.5"])
            market_id = f"{match['match_id']}_total_runs"
        elif market_type == MarketType.TOP_BATSMAN:
            selection = f"Player_{random.randint(1, 11)}"
            market_id = f"{match['match_id']}_top_batsman"
        else:
            selection = f"Selection_{random.randint(1, 5)}"
            market_id = f"{match['match_id']}_{market_type.value}"
        
        # Generate liquidity information
        base_liquidity = random.uniform(5000, 50000) * self.market_condition.liquidity_multiplier
        
        market_depth = []
        for i in range(3):
            depth_odds = odds + (i * 0.02 * random.choice([-1, 1]))
            depth_size = base_liquidity * random.uniform(0.1, 0.4)
            market_depth.append(MarketDepth(odds=depth_odds, size=depth_size))
        
        liquidity = LiquidityInfo(
            available=base_liquidity,
            market_depth=market_depth
        )
        
        # Calculate fair odds (with some noise)
        fair_odds = odds * random.uniform(0.95, 1.05)
        
        # Apply custom parameters if provided
        if custom_params:
            stake = custom_params.get("stake", stake)
            odds = custom_params.get("odds", odds)
            fair_odds = custom_params.get("fair_odds", fair_odds)
            selection = custom_params.get("selection", selection)
        
        # Execute pre-proposal hook if set
        if self.pre_proposal_hook:
            await self.pre_proposal_hook(match, market_type, stake, odds)
        
        proposal = BetProposal(
            market_id=market_id,
            match_id=match["match_id"],
            side=side,
            selection=selection,
            odds=odds,
            stake=stake,
            model_confidence=random.uniform(0.6, 0.95),
            fair_odds=fair_odds,
            expected_edge_pct=abs(odds - fair_odds) / fair_odds * 100,
            liquidity=liquidity,
            features={
                "match_format": match["format"],
                "venue": match["venue"],
                "market_type": market_type.value,
                "generated_at": datetime.now().isoformat()
            }
        )
        
        return proposal
    
    async def submit_proposal(self, proposal: BetProposal) -> GovernanceDecision:
        """
        Submit proposal to DGL and handle response
        
        Args:
            proposal: Bet proposal to submit
            
        Returns:
            Governance decision from DGL
        """
        start_time = datetime.now()
        
        try:
            # Submit to DGL
            decision = await self.dgl_client.evaluate_proposal(proposal)
            
            # Update statistics
            self.stats.total_proposals += 1
            self.stats.total_stake_requested += proposal.stake
            
            if decision.decision == DecisionType.APPROVE:
                self.stats.approved_proposals += 1
                self.stats.total_stake_approved += proposal.stake
            elif decision.decision == DecisionType.REJECT:
                self.stats.rejected_proposals += 1
            elif decision.decision == DecisionType.AMEND:
                self.stats.amended_proposals += 1
                # If amended, use amended stake for approved amount
                if hasattr(decision, 'amended_proposal') and decision.amended_proposal:
                    amended_stake = decision.amended_proposal.get('stake', proposal.stake)
                    self.stats.total_stake_approved += amended_stake
            
            # Track response time
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self.response_times.append(response_time)
            
            # Update average response time
            self.stats.avg_response_time_ms = sum(self.response_times) / len(self.response_times)
            
            # Execute post-decision hook if set
            if self.post_decision_hook:
                await self.post_decision_hook(proposal, decision, response_time)
            
            logger.debug(f"Proposal {proposal.market_id} -> {decision.decision.value} ({response_time:.1f}ms)")
            
            return decision
            
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Error submitting proposal: {str(e)}")
            raise
    
    async def run_simulation(
        self,
        duration_minutes: int = 10,
        proposals_per_minute: int = 6,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run orchestrator simulation
        
        Args:
            duration_minutes: How long to run simulation
            proposals_per_minute: Rate of proposal generation
            progress_callback: Optional callback for progress updates
            
        Returns:
            Simulation results and statistics
        """
        logger.info(f"Starting {duration_minutes}min simulation at {proposals_per_minute} proposals/min")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        proposal_interval = 60.0 / proposals_per_minute  # seconds between proposals
        
        simulation_results = {
            "start_time": start_time.isoformat(),
            "duration_minutes": duration_minutes,
            "target_proposals_per_minute": proposals_per_minute,
            "proposals": [],
            "decisions": [],
            "errors": []
        }
        
        proposal_count = 0
        
        while datetime.now() < end_time:
            try:
                # Generate and submit proposal
                proposal = await self.generate_proposal()
                decision = await self.submit_proposal(proposal)
                
                proposal_count += 1
                
                # Store results
                simulation_results["proposals"].append({
                    "id": proposal_count,
                    "market_id": proposal.market_id,
                    "stake": proposal.stake,
                    "odds": proposal.odds,
                    "timestamp": datetime.now().isoformat()
                })
                
                simulation_results["decisions"].append({
                    "proposal_id": proposal_count,
                    "decision": decision.decision.value,
                    "processing_time_ms": decision.processing_time_ms,
                    "rules_triggered": len(decision.rule_ids_triggered),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Progress callback
                if progress_callback:
                    elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
                    progress = min(elapsed_minutes / duration_minutes, 1.0)
                    await progress_callback(progress, proposal_count, self.stats)
                
                # Wait for next proposal
                await asyncio.sleep(proposal_interval)
                
            except Exception as e:
                error_info = {
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "proposal_count": proposal_count
                }
                simulation_results["errors"].append(error_info)
                logger.error(f"Simulation error: {str(e)}")
                
                # Continue simulation despite errors
                await asyncio.sleep(1.0)
        
        # Calculate final statistics
        total_time = (datetime.now() - start_time).total_seconds()
        actual_rate = proposal_count / (total_time / 60) if total_time > 0 else 0
        
        simulation_results.update({
            "end_time": datetime.now().isoformat(),
            "total_proposals": proposal_count,
            "actual_proposals_per_minute": actual_rate,
            "total_errors": len(simulation_results["errors"]),
            "final_stats": self.get_statistics()
        })
        
        logger.info(f"Simulation completed: {proposal_count} proposals, {actual_rate:.1f} proposals/min")
        
        return simulation_results
    
    async def stress_test(
        self,
        concurrent_requests: int = 10,
        total_requests: int = 100
    ) -> Dict[str, Any]:
        """
        Run concurrent stress test
        
        Args:
            concurrent_requests: Number of concurrent requests
            total_requests: Total requests to make
            
        Returns:
            Stress test results
        """
        logger.info(f"Starting stress test: {concurrent_requests} concurrent, {total_requests} total")
        
        start_time = datetime.now()
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def submit_single_proposal(proposal_id: int):
            async with semaphore:
                try:
                    proposal = await self.generate_proposal()
                    decision = await self.submit_proposal(proposal)
                    return {
                        "id": proposal_id,
                        "success": True,
                        "decision": decision.decision.value,
                        "processing_time_ms": decision.processing_time_ms
                    }
                except Exception as e:
                    return {
                        "id": proposal_id,
                        "success": False,
                        "error": str(e)
                    }
        
        # Submit all requests concurrently
        tasks = [submit_single_proposal(i) for i in range(total_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
        failed = total_requests - successful
        
        total_time = (datetime.now() - start_time).total_seconds()
        throughput = total_requests / total_time if total_time > 0 else 0
        
        stress_results = {
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": total_time,
            "concurrent_requests": concurrent_requests,
            "total_requests": total_requests,
            "successful_requests": successful,
            "failed_requests": failed,
            "success_rate_pct": (successful / total_requests) * 100,
            "throughput_requests_per_second": throughput,
            "results": results
        }
        
        logger.info(f"Stress test completed: {successful}/{total_requests} successful ({throughput:.1f} req/s)")
        
        return stress_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics"""
        total_decisions = (
            self.stats.approved_proposals + 
            self.stats.rejected_proposals + 
            self.stats.amended_proposals
        )
        
        approval_rate = (
            (self.stats.approved_proposals / total_decisions * 100) 
            if total_decisions > 0 else 0
        )
        
        return {
            "mode": self.mode.value,
            "total_proposals": self.stats.total_proposals,
            "total_decisions": total_decisions,
            "approved": self.stats.approved_proposals,
            "rejected": self.stats.rejected_proposals,
            "amended": self.stats.amended_proposals,
            "errors": self.stats.errors,
            "approval_rate_pct": approval_rate,
            "avg_response_time_ms": self.stats.avg_response_time_ms,
            "total_stake_requested": self.stats.total_stake_requested,
            "total_stake_approved": self.stats.total_stake_approved,
            "stake_approval_rate_pct": (
                (self.stats.total_stake_approved / self.stats.total_stake_requested * 100)
                if self.stats.total_stake_requested > 0 else 0
            )
        }
    
    def reset_statistics(self):
        """Reset all statistics counters"""
        self.stats = OrchestratorStats()
        self.response_times = []
        logger.info("Orchestrator statistics reset")


# Convenience functions for quick testing

async def quick_simulation(
    dgl_base_url: str = "http://localhost:8001",
    duration_minutes: int = 5,
    mode: OrchestratorMode = OrchestratorMode.BALANCED
) -> Dict[str, Any]:
    """
    Quick simulation with default settings
    
    Args:
        dgl_base_url: DGL service URL
        duration_minutes: Simulation duration
        mode: Orchestrator mode
        
    Returns:
        Simulation results
    """
    config = DGLClientConfig(base_url=dgl_base_url)
    
    async with DGLClient(config) as dgl_client:
        orchestrator = MockOrchestrator(dgl_client, mode)
        return await orchestrator.run_simulation(duration_minutes=duration_minutes)


async def quick_stress_test(
    dgl_base_url: str = "http://localhost:8001",
    concurrent_requests: int = 5,
    total_requests: int = 50
) -> Dict[str, Any]:
    """
    Quick stress test with default settings
    
    Args:
        dgl_base_url: DGL service URL
        concurrent_requests: Concurrent request limit
        total_requests: Total requests to make
        
    Returns:
        Stress test results
    """
    config = DGLClientConfig(base_url=dgl_base_url)
    
    async with DGLClient(config) as dgl_client:
        orchestrator = MockOrchestrator(dgl_client, OrchestratorMode.STRESS_TEST)
        return await orchestrator.stress_test(concurrent_requests, total_requests)
