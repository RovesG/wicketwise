# Purpose: Production mirroring system for DGL validation
# Author: WicketWise AI, Last Modified: 2024

"""
Production Mirror

Mirrors production traffic and decisions for DGL validation:
- Real-time traffic mirroring
- Decision comparison and analysis
- Production behavior modeling
- Gradual rollout support
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from collections import deque, defaultdict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import BetProposal, GovernanceDecision, DecisionType, RuleId
from client.dgl_client import DGLClient, DGLClientConfig
from simulator.shadow_simulator import ShadowSimulator, ProductionMockEngine, ShadowComparison


logger = logging.getLogger(__name__)


class MirrorMode(Enum):
    """Production mirroring modes"""
    PASSIVE_OBSERVATION = "passive_observation"  # Observe only, no comparison
    ACTIVE_COMPARISON = "active_comparison"      # Compare decisions actively
    GRADUAL_ROLLOUT = "gradual_rollout"         # Gradually increase DGL usage
    CANARY_TESTING = "canary_testing"           # Test with small percentage of traffic


class RolloutStrategy(Enum):
    """Rollout strategies for production deployment"""
    PERCENTAGE_BASED = "percentage_based"       # Roll out to X% of traffic
    USER_BASED = "user_based"                   # Roll out to specific users
    MARKET_BASED = "market_based"               # Roll out to specific markets
    TIME_BASED = "time_based"                   # Roll out during specific times


@dataclass
class MirrorConfig:
    """Configuration for production mirroring"""
    mode: MirrorMode = MirrorMode.PASSIVE_OBSERVATION
    rollout_strategy: RolloutStrategy = RolloutStrategy.PERCENTAGE_BASED
    rollout_percentage: float = 5.0  # Start with 5% of traffic
    target_percentage: float = 100.0  # Eventually 100%
    rollout_increment: float = 5.0   # Increase by 5% each step
    rollout_interval_hours: int = 24  # Increase every 24 hours
    enable_automatic_rollback: bool = True
    rollback_error_threshold: float = 5.0  # Rollback if >5% errors
    rollback_disagreement_threshold: float = 20.0  # Rollback if >20% disagreement


@dataclass
class TrafficSample:
    """Sample of production traffic"""
    sample_id: str
    timestamp: datetime
    proposal: BetProposal
    production_decision: Optional[GovernanceDecision] = None
    dgl_decision: Optional[GovernanceDecision] = None
    mirror_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MirrorMetrics:
    """Metrics for production mirroring"""
    total_traffic: int = 0
    mirrored_traffic: int = 0
    mirror_percentage: float = 0.0
    decisions_compared: int = 0
    agreement_count: int = 0
    disagreement_count: int = 0
    agreement_rate_pct: float = 0.0
    dgl_errors: int = 0
    dgl_error_rate_pct: float = 0.0
    avg_dgl_response_time_ms: float = 0.0
    avg_production_response_time_ms: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class RolloutStatus:
    """Status of gradual rollout"""
    current_percentage: float = 0.0
    target_percentage: float = 100.0
    rollout_start_time: datetime = field(default_factory=datetime.now)
    last_increment_time: Optional[datetime] = None
    next_increment_time: Optional[datetime] = None
    rollout_paused: bool = False
    rollback_triggered: bool = False
    rollback_reason: Optional[str] = None


class ProductionMirror:
    """
    Production mirroring system for DGL validation and rollout
    
    Provides comprehensive production traffic mirroring, decision comparison,
    and gradual rollout capabilities for safe DGL deployment.
    """
    
    def __init__(self,
                 dgl_client: DGLClient,
                 config: Optional[MirrorConfig] = None):
        """
        Initialize production mirror
        
        Args:
            dgl_client: DGL client for shadow decisions
            config: Mirror configuration
        """
        self.dgl_client = dgl_client
        self.config = config or MirrorConfig()
        
        # Traffic and decision tracking
        self.traffic_samples: deque = deque(maxlen=10000)  # Keep last 10k samples
        self.recent_comparisons: deque = deque(maxlen=1000)  # Keep last 1k comparisons
        
        # Metrics and status
        self.metrics = MirrorMetrics()
        self.rollout_status = RolloutStatus(
            current_percentage=self.config.rollout_percentage,
            target_percentage=self.config.target_percentage
        )
        
        # Production mock for testing
        self.production_mock = ProductionMockEngine()
        
        # Callbacks for monitoring
        self.traffic_callback: Optional[Callable] = None
        self.comparison_callback: Optional[Callable] = None
        self.rollout_callback: Optional[Callable] = None
        
        # Rollout control
        self._rollout_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        logger.info(f"Production mirror initialized in {self.config.mode.value} mode")
    
    async def start_mirroring(self):
        """Start production mirroring process"""
        logger.info("Starting production mirroring")
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start rollout task if in gradual rollout mode
        if self.config.mode == MirrorMode.GRADUAL_ROLLOUT:
            self._rollout_task = asyncio.create_task(self._rollout_loop())
        
        logger.info("Production mirroring started")
    
    async def stop_mirroring(self):
        """Stop production mirroring process"""
        logger.info("Stopping production mirroring")
        
        # Cancel tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._rollout_task:
            self._rollout_task.cancel()
            try:
                await self._rollout_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Production mirroring stopped")
    
    async def mirror_traffic_sample(self, proposal: BetProposal) -> TrafficSample:
        """
        Mirror a single traffic sample
        
        Args:
            proposal: Production bet proposal
            
        Returns:
            Traffic sample with mirroring results
        """
        sample_id = f"mirror_{uuid.uuid4().hex[:8]}"
        sample = TrafficSample(
            sample_id=sample_id,
            timestamp=datetime.now(),
            proposal=proposal
        )
        
        # Determine if this sample should be mirrored
        should_mirror = self._should_mirror_sample(proposal)
        
        if should_mirror:
            try:
                # Get production decision (mock for testing)
                production_start = datetime.now()
                production_decision = await self.production_mock.make_decision(proposal)
                production_time = (datetime.now() - production_start).total_seconds() * 1000
                
                # Convert production decision to GovernanceDecision format
                sample.production_decision = GovernanceDecision(
                    decision=production_decision.decision,
                    rule_ids_triggered=[],  # Production mock doesn't provide rule IDs
                    reasoning=production_decision.reasoning,
                    confidence_score=production_decision.confidence_score,
                    processing_time_ms=production_decision.processing_time_ms,
                    audit_ref=f"prod_{sample_id}"
                )
                
                # Get DGL decision
                if self.config.mode in [MirrorMode.ACTIVE_COMPARISON, MirrorMode.GRADUAL_ROLLOUT]:
                    dgl_start = datetime.now()
                    sample.dgl_decision = await self.dgl_client.evaluate_proposal(proposal)
                    dgl_time = (datetime.now() - dgl_start).total_seconds() * 1000
                    
                    # Update metrics
                    self.metrics.mirrored_traffic += 1
                    self.metrics.avg_dgl_response_time_ms = self._update_average(
                        self.metrics.avg_dgl_response_time_ms,
                        dgl_time,
                        self.metrics.mirrored_traffic
                    )
                    
                    # Create comparison if both decisions available
                    if sample.production_decision and sample.dgl_decision:
                        comparison = self._create_comparison(sample)
                        self.recent_comparisons.append(comparison)
                        
                        # Update agreement metrics
                        self.metrics.decisions_compared += 1
                        if comparison.decisions_match:
                            self.metrics.agreement_count += 1
                        else:
                            self.metrics.disagreement_count += 1
                        
                        self.metrics.agreement_rate_pct = (
                            self.metrics.agreement_count / self.metrics.decisions_compared
                        ) * 100
                        
                        # Execute comparison callback
                        if self.comparison_callback:
                            await self.comparison_callback(comparison)
                
                # Update production metrics
                self.metrics.avg_production_response_time_ms = self._update_average(
                    self.metrics.avg_production_response_time_ms,
                    production_time,
                    self.metrics.total_traffic
                )
                
            except Exception as e:
                logger.error(f"Error mirroring sample {sample_id}: {str(e)}")
                self.metrics.dgl_errors += 1
                sample.mirror_metadata["error"] = str(e)
        
        # Update traffic metrics
        self.metrics.total_traffic += 1
        self.metrics.mirror_percentage = (
            self.metrics.mirrored_traffic / self.metrics.total_traffic
        ) * 100 if self.metrics.total_traffic > 0 else 0
        
        self.metrics.dgl_error_rate_pct = (
            self.metrics.dgl_errors / max(self.metrics.mirrored_traffic, 1)
        ) * 100
        
        self.metrics.last_updated = datetime.now()
        
        # Store sample
        self.traffic_samples.append(sample)
        
        # Execute traffic callback
        if self.traffic_callback:
            await self.traffic_callback(sample)
        
        return sample
    
    def _should_mirror_sample(self, proposal: BetProposal) -> bool:
        """Determine if a traffic sample should be mirrored"""
        
        if self.config.mode == MirrorMode.PASSIVE_OBSERVATION:
            return False
        
        # Check rollout percentage
        import random
        if random.random() * 100 > self.rollout_status.current_percentage:
            return False
        
        # Apply rollout strategy filters
        if self.config.rollout_strategy == RolloutStrategy.MARKET_BASED:
            # Only mirror specific markets (example: match_winner markets)
            if "match_winner" not in proposal.market_id:
                return False
        
        elif self.config.rollout_strategy == RolloutStrategy.TIME_BASED:
            # Only mirror during specific hours (example: 9 AM - 5 PM UTC)
            current_hour = datetime.now().hour
            if not (9 <= current_hour <= 17):
                return False
        
        # Check if rollout is paused or rolled back
        if self.rollout_status.rollout_paused or self.rollout_status.rollback_triggered:
            return False
        
        return True
    
    def _create_comparison(self, sample: TrafficSample) -> ShadowComparison:
        """Create comparison between production and DGL decisions"""
        
        # Convert production decision for comparison
        from simulator.shadow_simulator import ProductionDecision
        prod_decision = ProductionDecision(
            decision=sample.production_decision.decision,
            reasoning=sample.production_decision.reasoning,
            processing_time_ms=sample.production_decision.processing_time_ms,
            confidence_score=sample.production_decision.confidence_score,
            timestamp=sample.timestamp
        )
        
        decisions_match = sample.dgl_decision.decision == sample.production_decision.decision
        confidence_delta = sample.dgl_decision.confidence_score - sample.production_decision.confidence_score
        processing_time_delta = sample.dgl_decision.processing_time_ms - sample.production_decision.processing_time_ms
        
        # Analysis
        analysis = {
            "decision_match": decisions_match,
            "dgl_decision": sample.dgl_decision.decision.value,
            "production_decision": sample.production_decision.decision.value,
            "confidence_delta": confidence_delta,
            "processing_time_delta_ms": processing_time_delta,
            "dgl_rules_triggered": len(sample.dgl_decision.rule_ids_triggered),
            "sample_id": sample.sample_id
        }
        
        from simulator.shadow_simulator import ShadowComparison
        return ShadowComparison(
            proposal_id=sample.sample_id,
            shadow_decision=sample.dgl_decision,
            production_decision=prod_decision,
            decisions_match=decisions_match,
            confidence_delta=confidence_delta,
            processing_time_delta_ms=processing_time_delta,
            analysis=analysis,
            timestamp=sample.timestamp
        )
    
    def _update_average(self, current_avg: float, new_value: float, count: int) -> float:
        """Update running average"""
        if count <= 1:
            return new_value
        return ((current_avg * (count - 1)) + new_value) / count
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Check for rollback conditions
                if self.config.enable_automatic_rollback:
                    await self._check_rollback_conditions()
                
                # Log metrics
                logger.debug(f"Mirror metrics: {self.metrics.agreement_rate_pct:.1f}% agreement, "
                           f"{self.metrics.dgl_error_rate_pct:.1f}% error rate")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
    
    async def _rollout_loop(self):
        """Background rollout management loop"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Check if it's time to increment rollout
                if self._should_increment_rollout():
                    await self._increment_rollout()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rollout loop: {str(e)}")
    
    def _should_increment_rollout(self) -> bool:
        """Check if rollout should be incremented"""
        
        # Don't increment if paused or rolled back
        if self.rollout_status.rollout_paused or self.rollout_status.rollback_triggered:
            return False
        
        # Don't increment if already at target
        if self.rollout_status.current_percentage >= self.rollout_status.target_percentage:
            return False
        
        # Check time interval
        if self.rollout_status.last_increment_time:
            time_since_last = datetime.now() - self.rollout_status.last_increment_time
            if time_since_last.total_seconds() < self.config.rollout_interval_hours * 3600:
                return False
        
        # Check stability metrics
        if self.metrics.decisions_compared < 100:  # Need minimum sample size
            return False
        
        if self.metrics.dgl_error_rate_pct > self.config.rollback_error_threshold:
            return False
        
        if self.metrics.agreement_rate_pct < (100 - self.config.rollback_disagreement_threshold):
            return False
        
        return True
    
    async def _increment_rollout(self):
        """Increment rollout percentage"""
        old_percentage = self.rollout_status.current_percentage
        new_percentage = min(
            self.rollout_status.current_percentage + self.config.rollout_increment,
            self.rollout_status.target_percentage
        )
        
        self.rollout_status.current_percentage = new_percentage
        self.rollout_status.last_increment_time = datetime.now()
        self.rollout_status.next_increment_time = (
            datetime.now() + timedelta(hours=self.config.rollout_interval_hours)
        )
        
        logger.info(f"Rollout incremented: {old_percentage:.1f}% -> {new_percentage:.1f}%")
        
        # Execute rollout callback
        if self.rollout_callback:
            await self.rollout_callback({
                "action": "increment",
                "old_percentage": old_percentage,
                "new_percentage": new_percentage,
                "metrics": self.get_metrics_summary()
            })
    
    async def _check_rollback_conditions(self):
        """Check if rollback should be triggered"""
        
        if self.rollout_status.rollback_triggered:
            return
        
        # Need minimum sample size for rollback decisions
        if self.metrics.decisions_compared < 50:
            return
        
        rollback_reason = None
        
        # Check error rate
        if self.metrics.dgl_error_rate_pct > self.config.rollback_error_threshold:
            rollback_reason = f"High error rate: {self.metrics.dgl_error_rate_pct:.1f}%"
        
        # Check disagreement rate
        elif self.metrics.agreement_rate_pct < (100 - self.config.rollback_disagreement_threshold):
            rollback_reason = f"High disagreement rate: {100 - self.metrics.agreement_rate_pct:.1f}%"
        
        if rollback_reason:
            await self._trigger_rollback(rollback_reason)
    
    async def _trigger_rollback(self, reason: str):
        """Trigger automatic rollback"""
        logger.warning(f"Triggering rollback: {reason}")
        
        old_percentage = self.rollout_status.current_percentage
        
        self.rollout_status.rollback_triggered = True
        self.rollout_status.rollback_reason = reason
        self.rollout_status.current_percentage = 0.0  # Stop all mirroring
        
        # Execute rollout callback
        if self.rollout_callback:
            await self.rollout_callback({
                "action": "rollback",
                "reason": reason,
                "old_percentage": old_percentage,
                "new_percentage": 0.0,
                "metrics": self.get_metrics_summary()
            })
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            "traffic_metrics": {
                "total_traffic": self.metrics.total_traffic,
                "mirrored_traffic": self.metrics.mirrored_traffic,
                "mirror_percentage": self.metrics.mirror_percentage
            },
            "decision_metrics": {
                "decisions_compared": self.metrics.decisions_compared,
                "agreement_rate_pct": self.metrics.agreement_rate_pct,
                "disagreement_count": self.metrics.disagreement_count
            },
            "performance_metrics": {
                "dgl_error_rate_pct": self.metrics.dgl_error_rate_pct,
                "avg_dgl_response_time_ms": self.metrics.avg_dgl_response_time_ms,
                "avg_production_response_time_ms": self.metrics.avg_production_response_time_ms
            },
            "rollout_status": {
                "current_percentage": self.rollout_status.current_percentage,
                "target_percentage": self.rollout_status.target_percentage,
                "rollout_paused": self.rollout_status.rollout_paused,
                "rollback_triggered": self.rollout_status.rollback_triggered,
                "rollback_reason": self.rollout_status.rollback_reason
            },
            "last_updated": self.metrics.last_updated.isoformat()
        }
    
    def get_recent_comparisons(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent decision comparisons"""
        recent = list(self.recent_comparisons)[-limit:]
        
        return [
            {
                "sample_id": comp.proposal_id,
                "timestamp": comp.timestamp.isoformat(),
                "decisions_match": comp.decisions_match,
                "dgl_decision": comp.shadow_decision.decision.value,
                "production_decision": comp.production_decision.decision.value,
                "confidence_delta": comp.confidence_delta,
                "processing_time_delta_ms": comp.processing_time_delta_ms,
                "analysis": comp.analysis
            }
            for comp in recent
        ]
    
    def pause_rollout(self):
        """Pause rollout progression"""
        self.rollout_status.rollout_paused = True
        logger.info("Rollout paused")
    
    def resume_rollout(self):
        """Resume rollout progression"""
        if not self.rollout_status.rollback_triggered:
            self.rollout_status.rollout_paused = False
            logger.info("Rollout resumed")
    
    def reset_rollback(self):
        """Reset rollback status (manual intervention)"""
        self.rollout_status.rollback_triggered = False
        self.rollout_status.rollback_reason = None
        self.rollout_status.rollout_paused = False
        logger.info("Rollback status reset")
    
    def export_mirror_data(self, file_path: str):
        """Export mirroring data to JSON file"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "config": {
                "mode": self.config.mode.value,
                "rollout_strategy": self.config.rollout_strategy.value,
                "rollout_percentage": self.config.rollout_percentage,
                "target_percentage": self.config.target_percentage
            },
            "metrics_summary": self.get_metrics_summary(),
            "recent_comparisons": self.get_recent_comparisons(100),
            "traffic_samples_count": len(self.traffic_samples)
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported mirror data to {file_path}")


# Convenience functions for production mirroring

async def setup_canary_testing(
    dgl_client: DGLClient,
    canary_percentage: float = 1.0
) -> ProductionMirror:
    """
    Set up canary testing with small percentage of traffic
    
    Args:
        dgl_client: DGL client
        canary_percentage: Percentage of traffic for canary testing
        
    Returns:
        Configured production mirror
    """
    config = MirrorConfig(
        mode=MirrorMode.CANARY_TESTING,
        rollout_strategy=RolloutStrategy.PERCENTAGE_BASED,
        rollout_percentage=canary_percentage,
        target_percentage=canary_percentage,  # Stay at canary level
        enable_automatic_rollback=True
    )
    
    mirror = ProductionMirror(dgl_client, config)
    await mirror.start_mirroring()
    
    return mirror


async def setup_gradual_rollout(
    dgl_client: DGLClient,
    start_percentage: float = 5.0,
    target_percentage: float = 100.0,
    increment: float = 5.0,
    interval_hours: int = 24
) -> ProductionMirror:
    """
    Set up gradual rollout with automatic progression
    
    Args:
        dgl_client: DGL client
        start_percentage: Starting percentage
        target_percentage: Target percentage
        increment: Increment per step
        interval_hours: Hours between increments
        
    Returns:
        Configured production mirror
    """
    config = MirrorConfig(
        mode=MirrorMode.GRADUAL_ROLLOUT,
        rollout_strategy=RolloutStrategy.PERCENTAGE_BASED,
        rollout_percentage=start_percentage,
        target_percentage=target_percentage,
        rollout_increment=increment,
        rollout_interval_hours=interval_hours,
        enable_automatic_rollback=True
    )
    
    mirror = ProductionMirror(dgl_client, config)
    await mirror.start_mirroring()
    
    return mirror
