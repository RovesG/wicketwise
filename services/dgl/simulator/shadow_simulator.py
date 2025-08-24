# Purpose: Shadow mode simulator for DGL testing
# Author: WicketWise AI, Last Modified: 2024

"""
Shadow Simulator

Implements comprehensive shadow mode testing for DGL:
- Parallel decision making (shadow vs production)
- Decision comparison and analysis
- Shadow mode performance validation
- Production readiness assessment
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import BetProposal, GovernanceDecision, DecisionType, RuleId
from client.dgl_client import DGLClient, DGLClientConfig
from client.orchestrator_mock import MockOrchestrator, OrchestratorMode


logger = logging.getLogger(__name__)


class ShadowMode(Enum):
    """Shadow mode operation types"""
    OBSERVE_ONLY = "observe_only"      # Log decisions without comparison
    COMPARE_DECISIONS = "compare_decisions"  # Compare shadow vs production
    VALIDATE_RULES = "validate_rules"   # Validate rule consistency
    PERFORMANCE_TEST = "performance_test"  # Test performance characteristics


@dataclass
class ProductionDecision:
    """Simulated production decision for comparison"""
    decision: DecisionType
    reasoning: str
    processing_time_ms: float
    confidence_score: float
    timestamp: datetime
    source: str = "production_mock"


@dataclass
class ShadowComparison:
    """Comparison between shadow and production decisions"""
    proposal_id: str
    shadow_decision: GovernanceDecision
    production_decision: ProductionDecision
    decisions_match: bool
    confidence_delta: float
    processing_time_delta_ms: float
    analysis: Dict[str, Any]
    timestamp: datetime


@dataclass
class ShadowTestResult:
    """Result of shadow mode testing"""
    test_id: str
    mode: ShadowMode
    duration_seconds: float
    total_proposals: int
    shadow_decisions: List[GovernanceDecision]
    production_decisions: List[ProductionDecision]
    comparisons: List[ShadowComparison]
    metrics: Dict[str, Any]
    summary: Dict[str, Any]
    timestamp: datetime


class ProductionMockEngine:
    """
    Mock production decision engine for shadow testing
    
    Simulates existing production decision logic with configurable
    behavior patterns and biases.
    """
    
    def __init__(self, 
                 approval_bias: float = 0.7,
                 conservative_factor: float = 1.0,
                 processing_delay_ms: float = 15.0):
        """
        Initialize production mock engine
        
        Args:
            approval_bias: Tendency to approve (0.0-1.0)
            conservative_factor: Risk aversion multiplier
            processing_delay_ms: Simulated processing delay
        """
        self.approval_bias = approval_bias
        self.conservative_factor = conservative_factor
        self.processing_delay_ms = processing_delay_ms
        
        # Decision patterns
        self.decision_count = 0
        self.recent_decisions = []
        
    async def make_decision(self, proposal: BetProposal) -> ProductionDecision:
        """
        Make a production-style decision
        
        Args:
            proposal: Bet proposal to evaluate
            
        Returns:
            Production decision
        """
        start_time = datetime.now()
        
        # Simulate processing delay
        await asyncio.sleep(self.processing_delay_ms / 1000)
        
        # Simple decision logic based on stake and odds
        risk_score = self._calculate_risk_score(proposal)
        
        # Apply approval bias and conservative factor
        adjusted_risk = risk_score * self.conservative_factor
        approval_threshold = 1.0 - self.approval_bias
        
        if adjusted_risk < approval_threshold:
            decision = DecisionType.APPROVE
            reasoning = f"Low risk score {risk_score:.3f}, approved"
            confidence = 0.8 + (approval_threshold - adjusted_risk) * 0.2
        elif adjusted_risk < approval_threshold + 0.2:
            decision = DecisionType.AMEND
            reasoning = f"Moderate risk score {risk_score:.3f}, suggest amendment"
            confidence = 0.6 + (0.2 - (adjusted_risk - approval_threshold)) * 0.2
        else:
            decision = DecisionType.REJECT
            reasoning = f"High risk score {risk_score:.3f}, rejected"
            confidence = 0.7 + (adjusted_risk - approval_threshold - 0.2) * 0.3
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        production_decision = ProductionDecision(
            decision=decision,
            reasoning=reasoning,
            processing_time_ms=processing_time,
            confidence_score=min(confidence, 1.0),
            timestamp=datetime.now()
        )
        
        # Track decision history
        self.decision_count += 1
        self.recent_decisions.append(production_decision)
        if len(self.recent_decisions) > 100:
            self.recent_decisions.pop(0)
        
        return production_decision
    
    def _calculate_risk_score(self, proposal: BetProposal) -> float:
        """Calculate simple risk score for production mock"""
        # Normalize stake (assuming max reasonable stake of 5000)
        stake_factor = min(proposal.stake / 5000.0, 1.0)
        
        # Odds factor (higher odds = higher risk)
        odds_factor = min((proposal.odds - 1.0) / 9.0, 1.0)  # Normalize to 1-10 range
        
        # Model confidence factor (lower confidence = higher risk)
        confidence_factor = 1.0 - proposal.model_confidence
        
        # Combine factors
        risk_score = (stake_factor * 0.4 + odds_factor * 0.3 + confidence_factor * 0.3)
        
        return min(risk_score, 1.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get production engine statistics"""
        if not self.recent_decisions:
            return {"total_decisions": 0}
        
        approvals = sum(1 for d in self.recent_decisions if d.decision == DecisionType.APPROVE)
        rejections = sum(1 for d in self.recent_decisions if d.decision == DecisionType.REJECT)
        amendments = sum(1 for d in self.recent_decisions if d.decision == DecisionType.AMEND)
        
        avg_processing_time = sum(d.processing_time_ms for d in self.recent_decisions) / len(self.recent_decisions)
        avg_confidence = sum(d.confidence_score for d in self.recent_decisions) / len(self.recent_decisions)
        
        return {
            "total_decisions": self.decision_count,
            "recent_decisions": len(self.recent_decisions),
            "approval_rate_pct": (approvals / len(self.recent_decisions)) * 100,
            "rejection_rate_pct": (rejections / len(self.recent_decisions)) * 100,
            "amendment_rate_pct": (amendments / len(self.recent_decisions)) * 100,
            "avg_processing_time_ms": avg_processing_time,
            "avg_confidence_score": avg_confidence,
            "approval_bias": self.approval_bias,
            "conservative_factor": self.conservative_factor
        }


class ShadowSimulator:
    """
    Shadow mode simulator for comprehensive DGL testing
    
    Runs DGL in shadow mode alongside simulated production decisions
    to validate behavior, performance, and decision quality.
    """
    
    def __init__(self,
                 dgl_client: DGLClient,
                 production_mock: Optional[ProductionMockEngine] = None):
        """
        Initialize shadow simulator
        
        Args:
            dgl_client: DGL client for shadow decisions
            production_mock: Mock production engine (creates default if None)
        """
        self.dgl_client = dgl_client
        self.production_mock = production_mock or ProductionMockEngine()
        
        # Test results storage
        self.test_results: List[ShadowTestResult] = []
        
        # Callbacks for custom analysis
        self.decision_callback: Optional[Callable] = None
        self.comparison_callback: Optional[Callable] = None
        
        logger.info("Shadow simulator initialized")
    
    async def run_shadow_test(self,
                              mode: ShadowMode,
                              proposals: List[BetProposal],
                              test_id: Optional[str] = None) -> ShadowTestResult:
        """
        Run shadow mode test with given proposals
        
        Args:
            mode: Shadow testing mode
            proposals: Proposals to test
            test_id: Optional test identifier
            
        Returns:
            Shadow test results
        """
        test_id = test_id or f"shadow_test_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        
        logger.info(f"Starting shadow test {test_id} in {mode.value} mode with {len(proposals)} proposals")
        
        shadow_decisions = []
        production_decisions = []
        comparisons = []
        
        # Process each proposal
        for i, proposal in enumerate(proposals):
            try:
                # Run both shadow and production decisions concurrently
                shadow_task = asyncio.create_task(self.dgl_client.evaluate_proposal(proposal))
                production_task = asyncio.create_task(self.production_mock.make_decision(proposal))
                
                shadow_decision, production_decision = await asyncio.gather(
                    shadow_task, production_task, return_exceptions=True
                )
                
                # Handle exceptions
                if isinstance(shadow_decision, Exception):
                    logger.error(f"Shadow decision failed for proposal {i}: {str(shadow_decision)}")
                    continue
                
                if isinstance(production_decision, Exception):
                    logger.error(f"Production decision failed for proposal {i}: {str(production_decision)}")
                    continue
                
                shadow_decisions.append(shadow_decision)
                production_decisions.append(production_decision)
                
                # Create comparison if in comparison mode
                if mode == ShadowMode.COMPARE_DECISIONS:
                    comparison = self._create_comparison(
                        f"{test_id}_proposal_{i}",
                        shadow_decision,
                        production_decision
                    )
                    comparisons.append(comparison)
                    
                    # Execute comparison callback if set
                    if self.comparison_callback:
                        await self.comparison_callback(comparison)
                
                # Execute decision callback if set
                if self.decision_callback:
                    await self.decision_callback(proposal, shadow_decision, production_decision)
                
                logger.debug(f"Processed proposal {i+1}/{len(proposals)}")
                
            except Exception as e:
                logger.error(f"Error processing proposal {i}: {str(e)}")
        
        # Calculate test duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Generate metrics and summary
        metrics = self._calculate_metrics(shadow_decisions, production_decisions, comparisons)
        summary = self._generate_summary(mode, metrics, comparisons)
        
        # Create test result
        result = ShadowTestResult(
            test_id=test_id,
            mode=mode,
            duration_seconds=duration,
            total_proposals=len(proposals),
            shadow_decisions=shadow_decisions,
            production_decisions=production_decisions,
            comparisons=comparisons,
            metrics=metrics,
            summary=summary,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        
        logger.info(f"Shadow test {test_id} completed in {duration:.2f}s")
        logger.info(f"Agreement rate: {metrics.get('agreement_rate_pct', 0):.1f}%")
        
        return result
    
    async def run_continuous_shadow_test(self,
                                         mode: ShadowMode,
                                         duration_minutes: int = 10,
                                         proposals_per_minute: int = 6,
                                         orchestrator_mode: OrchestratorMode = OrchestratorMode.BALANCED) -> ShadowTestResult:
        """
        Run continuous shadow test with generated proposals
        
        Args:
            mode: Shadow testing mode
            duration_minutes: Test duration
            proposals_per_minute: Rate of proposal generation
            orchestrator_mode: Orchestrator mode for proposal generation
            
        Returns:
            Shadow test results
        """
        logger.info(f"Starting continuous shadow test for {duration_minutes} minutes")
        
        # Create orchestrator for proposal generation
        orchestrator = MockOrchestrator(self.dgl_client, orchestrator_mode)
        
        # Generate proposals
        total_proposals = duration_minutes * proposals_per_minute
        proposals = []
        
        for i in range(total_proposals):
            proposal = await orchestrator.generate_proposal()
            proposals.append(proposal)
            
            # Small delay to simulate realistic timing
            if i < total_proposals - 1:
                await asyncio.sleep(60.0 / proposals_per_minute)
        
        # Run shadow test with generated proposals
        return await self.run_shadow_test(mode, proposals, f"continuous_{duration_minutes}min")
    
    def _create_comparison(self,
                          proposal_id: str,
                          shadow_decision: GovernanceDecision,
                          production_decision: ProductionDecision) -> ShadowComparison:
        """Create detailed comparison between shadow and production decisions"""
        
        decisions_match = shadow_decision.decision == production_decision.decision
        confidence_delta = shadow_decision.confidence_score - production_decision.confidence_score
        processing_time_delta = shadow_decision.processing_time_ms - production_decision.processing_time_ms
        
        # Detailed analysis
        analysis = {
            "decision_match": decisions_match,
            "shadow_decision": shadow_decision.decision.value,
            "production_decision": production_decision.decision.value,
            "shadow_rules_triggered": len(shadow_decision.rule_ids_triggered),
            "shadow_reasoning_length": len(shadow_decision.reasoning),
            "production_reasoning_length": len(production_decision.reasoning),
            "confidence_comparison": {
                "shadow_higher": shadow_decision.confidence_score > production_decision.confidence_score,
                "delta_abs": abs(confidence_delta),
                "delta_pct": (confidence_delta / production_decision.confidence_score) * 100
            },
            "performance_comparison": {
                "shadow_faster": shadow_decision.processing_time_ms < production_decision.processing_time_ms,
                "delta_ms": processing_time_delta,
                "delta_pct": (processing_time_delta / production_decision.processing_time_ms) * 100
            }
        }
        
        # Risk assessment comparison
        if decisions_match:
            analysis["risk_assessment"] = "aligned"
        elif shadow_decision.decision == DecisionType.REJECT and production_decision.decision == DecisionType.APPROVE:
            analysis["risk_assessment"] = "shadow_more_conservative"
        elif shadow_decision.decision == DecisionType.APPROVE and production_decision.decision == DecisionType.REJECT:
            analysis["risk_assessment"] = "shadow_more_aggressive"
        else:
            analysis["risk_assessment"] = "mixed_signals"
        
        return ShadowComparison(
            proposal_id=proposal_id,
            shadow_decision=shadow_decision,
            production_decision=production_decision,
            decisions_match=decisions_match,
            confidence_delta=confidence_delta,
            processing_time_delta_ms=processing_time_delta,
            analysis=analysis,
            timestamp=datetime.now()
        )
    
    def _calculate_metrics(self,
                          shadow_decisions: List[GovernanceDecision],
                          production_decisions: List[ProductionDecision],
                          comparisons: List[ShadowComparison]) -> Dict[str, Any]:
        """Calculate comprehensive metrics for shadow test"""
        
        if not shadow_decisions or not production_decisions:
            return {"error": "No decisions to analyze"}
        
        # Basic decision metrics
        shadow_approvals = sum(1 for d in shadow_decisions if d.decision == DecisionType.APPROVE)
        shadow_rejections = sum(1 for d in shadow_decisions if d.decision == DecisionType.REJECT)
        shadow_amendments = sum(1 for d in shadow_decisions if d.decision == DecisionType.AMEND)
        
        production_approvals = sum(1 for d in production_decisions if d.decision == DecisionType.APPROVE)
        production_rejections = sum(1 for d in production_decisions if d.decision == DecisionType.REJECT)
        production_amendments = sum(1 for d in production_decisions if d.decision == DecisionType.AMEND)
        
        # Performance metrics
        shadow_avg_time = sum(d.processing_time_ms for d in shadow_decisions) / len(shadow_decisions)
        production_avg_time = sum(d.processing_time_ms for d in production_decisions) / len(production_decisions)
        
        shadow_avg_confidence = sum(d.confidence_score for d in shadow_decisions) / len(shadow_decisions)
        production_avg_confidence = sum(d.confidence_score for d in production_decisions) / len(production_decisions)
        
        metrics = {
            "total_decisions": len(shadow_decisions),
            "shadow_metrics": {
                "approval_rate_pct": (shadow_approvals / len(shadow_decisions)) * 100,
                "rejection_rate_pct": (shadow_rejections / len(shadow_decisions)) * 100,
                "amendment_rate_pct": (shadow_amendments / len(shadow_decisions)) * 100,
                "avg_processing_time_ms": shadow_avg_time,
                "avg_confidence_score": shadow_avg_confidence
            },
            "production_metrics": {
                "approval_rate_pct": (production_approvals / len(production_decisions)) * 100,
                "rejection_rate_pct": (production_rejections / len(production_decisions)) * 100,
                "amendment_rate_pct": (production_amendments / len(production_decisions)) * 100,
                "avg_processing_time_ms": production_avg_time,
                "avg_confidence_score": production_avg_confidence
            }
        }
        
        # Comparison metrics if available
        if comparisons:
            agreements = sum(1 for c in comparisons if c.decisions_match)
            
            metrics.update({
                "agreement_rate_pct": (agreements / len(comparisons)) * 100,
                "disagreement_count": len(comparisons) - agreements,
                "avg_confidence_delta": sum(c.confidence_delta for c in comparisons) / len(comparisons),
                "avg_processing_time_delta_ms": sum(c.processing_time_delta_ms for c in comparisons) / len(comparisons),
                "shadow_more_conservative_count": sum(
                    1 for c in comparisons 
                    if c.analysis.get("risk_assessment") == "shadow_more_conservative"
                ),
                "shadow_more_aggressive_count": sum(
                    1 for c in comparisons 
                    if c.analysis.get("risk_assessment") == "shadow_more_aggressive"
                )
            })
        
        return metrics
    
    def _generate_summary(self,
                         mode: ShadowMode,
                         metrics: Dict[str, Any],
                         comparisons: List[ShadowComparison]) -> Dict[str, Any]:
        """Generate human-readable summary of shadow test results"""
        
        summary = {
            "test_mode": mode.value,
            "overall_status": "unknown"
        }
        
        if mode == ShadowMode.COMPARE_DECISIONS and comparisons:
            agreement_rate = metrics.get("agreement_rate_pct", 0)
            
            if agreement_rate >= 90:
                summary["overall_status"] = "excellent_alignment"
                summary["recommendation"] = "Shadow system shows excellent alignment with production"
            elif agreement_rate >= 80:
                summary["overall_status"] = "good_alignment"
                summary["recommendation"] = "Shadow system shows good alignment, minor differences acceptable"
            elif agreement_rate >= 70:
                summary["overall_status"] = "moderate_alignment"
                summary["recommendation"] = "Shadow system shows moderate alignment, investigate differences"
            else:
                summary["overall_status"] = "poor_alignment"
                summary["recommendation"] = "Shadow system shows poor alignment, requires investigation"
            
            # Performance comparison
            time_delta = metrics.get("avg_processing_time_delta_ms", 0)
            if time_delta < 0:
                summary["performance_status"] = "shadow_faster"
            elif time_delta < 10:
                summary["performance_status"] = "comparable_performance"
            else:
                summary["performance_status"] = "shadow_slower"
        
        elif mode == ShadowMode.OBSERVE_ONLY:
            summary["overall_status"] = "observation_complete"
            summary["recommendation"] = "Shadow system observed successfully, ready for comparison testing"
        
        # Key insights
        insights = []
        
        if metrics.get("shadow_metrics", {}).get("approval_rate_pct", 0) > 80:
            insights.append("Shadow system has high approval rate")
        
        if metrics.get("shadow_metrics", {}).get("avg_processing_time_ms", 0) < 50:
            insights.append("Shadow system shows fast processing times")
        
        if metrics.get("agreement_rate_pct", 0) > 85:
            insights.append("High agreement with production decisions")
        
        summary["key_insights"] = insights
        
        return summary
    
    def get_test_history(self) -> List[Dict[str, Any]]:
        """Get summary of all test results"""
        return [
            {
                "test_id": result.test_id,
                "mode": result.mode.value,
                "duration_seconds": result.duration_seconds,
                "total_proposals": result.total_proposals,
                "agreement_rate_pct": result.metrics.get("agreement_rate_pct", 0),
                "overall_status": result.summary.get("overall_status"),
                "timestamp": result.timestamp.isoformat()
            }
            for result in self.test_results
        ]
    
    def export_test_results(self, file_path: str, test_id: Optional[str] = None):
        """Export test results to JSON file"""
        if test_id:
            results_to_export = [r for r in self.test_results if r.test_id == test_id]
        else:
            results_to_export = self.test_results
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_tests": len(results_to_export),
            "test_results": []
        }
        
        for result in results_to_export:
            export_data["test_results"].append({
                "test_id": result.test_id,
                "mode": result.mode.value,
                "duration_seconds": result.duration_seconds,
                "total_proposals": result.total_proposals,
                "metrics": result.metrics,
                "summary": result.summary,
                "timestamp": result.timestamp.isoformat(),
                "comparisons_count": len(result.comparisons)
            })
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(results_to_export)} test results to {file_path}")


# Convenience functions for quick shadow testing

async def quick_shadow_comparison(
    dgl_base_url: str = "http://localhost:8001",
    num_proposals: int = 20,
    production_approval_bias: float = 0.7
) -> ShadowTestResult:
    """
    Quick shadow comparison test
    
    Args:
        dgl_base_url: DGL service URL
        num_proposals: Number of proposals to test
        production_approval_bias: Production mock approval tendency
        
    Returns:
        Shadow test results
    """
    client_config = DGLClientConfig(base_url=dgl_base_url)
    production_mock = ProductionMockEngine(approval_bias=production_approval_bias)
    
    async with DGLClient(client_config) as dgl_client:
        simulator = ShadowSimulator(dgl_client, production_mock)
        orchestrator = MockOrchestrator(dgl_client, OrchestratorMode.BALANCED)
        
        # Generate proposals
        proposals = []
        for _ in range(num_proposals):
            proposal = await orchestrator.generate_proposal()
            proposals.append(proposal)
        
        return await simulator.run_shadow_test(ShadowMode.COMPARE_DECISIONS, proposals)


async def quick_shadow_performance_test(
    dgl_base_url: str = "http://localhost:8001",
    duration_minutes: int = 5
) -> ShadowTestResult:
    """
    Quick shadow performance test
    
    Args:
        dgl_base_url: DGL service URL
        duration_minutes: Test duration
        
    Returns:
        Shadow test results
    """
    client_config = DGLClientConfig(base_url=dgl_base_url)
    production_mock = ProductionMockEngine()
    
    async with DGLClient(client_config) as dgl_client:
        simulator = ShadowSimulator(dgl_client, production_mock)
        
        return await simulator.run_continuous_shadow_test(
            ShadowMode.PERFORMANCE_TEST,
            duration_minutes=duration_minutes,
            proposals_per_minute=12
        )
