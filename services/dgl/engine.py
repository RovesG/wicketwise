# Purpose: DGL Rule Engine - validates bet proposals against governance rules
# Author: WicketWise AI, Last Modified: 2024

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time
import logging
from decimal import Decimal

from schemas import (
    BetProposal, GovernanceDecision, DecisionType, RuleId, 
    GovernanceState, BetAmendment, ExposureSnapshot, AuditRecord
)
from config import DGLConfig
from store import ExposureStore, PnLStore, AuditStore

# Import rule engines
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'rules'))

from rules.bankroll_rules import BankrollRuleEngine, RuleViolation as BankrollViolation
from rules.pnl_rules import PnLRuleEngine, RuleViolation as PnLViolation
from rules.liquidity_rules import LiquidityRuleEngine, RuleViolation as LiquidityViolation


logger = logging.getLogger(__name__)


class RuleViolation:
    """Represents a rule violation with context"""
    
    def __init__(self, rule_id: RuleId, message: str, 
                 current_value: Optional[float] = None,
                 threshold: Optional[float] = None,
                 severity: str = "ERROR"):
        self.rule_id = rule_id
        self.message = message
        self.current_value = current_value
        self.threshold = threshold
        self.severity = severity
        self.timestamp = datetime.utcnow()


class RuleEngine:
    """
    DGL Rule Engine - validates bet proposals against governance rules
    
    The engine evaluates proposals against:
    - Bankroll and exposure limits
    - P&L protection guards  
    - Liquidity and execution constraints
    - Rate limiting rules
    - Concentration and correlation limits
    - Compliance and operational controls
    """
    
    def __init__(self, config: DGLConfig, 
                 exposure_store: ExposureStore,
                 pnl_store: PnLStore,
                 audit_store: AuditStore):
        self.config = config
        self.exposure_store = exposure_store
        self.pnl_store = pnl_store
        self.audit_store = audit_store
        
        # Current governance state
        self._current_state = GovernanceState(config.mode)
        self._kill_switch_active = config.ops.kill_switch_enabled
        
        # Performance tracking
        self._decision_count = 0
        self._processing_times: List[float] = []
        
        # Initialize rule engines
        self.bankroll_engine = BankrollRuleEngine(config.bankroll)
        self.pnl_engine = PnLRuleEngine(config.pnl_guards, pnl_store)
        
        # Initialize liquidity engine with rate limiting config
        rate_limit_config = {
            'global_rate': 10.0,
            'global_burst': 50,
            'market_requests': config.ops.rate_limit.count,
            'market_window_seconds': config.ops.rate_limit.per_seconds
        }
        self.liquidity_engine = LiquidityRuleEngine(config.liquidity, rate_limit_config)
        
        logger.info(f"RuleEngine initialized in {self._current_state} mode with enhanced rule engines")
    
    def evaluate_proposal(self, proposal: BetProposal) -> GovernanceDecision:
        """
        Evaluate a bet proposal against all governance rules
        
        Args:
            proposal: The bet proposal to evaluate
            
        Returns:
            GovernanceDecision with APPROVE/REJECT/AMEND and reasons
        """
        start_time = time.time()
        
        try:
            # Check if DGL is in KILLED state or kill switch is active
            if self._current_state == GovernanceState.KILLED or self._kill_switch_active:
                return self._create_rejection_decision(
                    proposal, 
                    [RuleId.OPS_KILL_SWITCH],
                    "All betting is currently disabled (kill switch active)"
                )
            
            # In SHADOW mode, evaluate but don't actually enforce
            if self._current_state == GovernanceState.SHADOW and self.config.ops.shadow_mode_log_only:
                logger.info(f"SHADOW MODE: Evaluating proposal {proposal.proposal_id}")
            
            # Collect all rule violations
            violations = self._evaluate_all_rules(proposal)
            
            # Determine decision based on violations
            decision = self._make_decision(proposal, violations)
            
            # Record processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self._processing_times.append(processing_time)
            self._decision_count += 1
            
            # Keep only last 1000 processing times for stats
            if len(self._processing_times) > 1000:
                self._processing_times = self._processing_times[-1000:]
            
            decision.processing_time_ms = processing_time
            
            # Create audit record
            audit_record = self._create_audit_record(proposal, decision)
            self.audit_store.append_record(audit_record)
            decision.audit_ref = f"audit:{audit_record.audit_id}"
            
            logger.info(f"Decision for {proposal.proposal_id}: {decision.decision} "
                       f"({processing_time:.2f}ms)")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error evaluating proposal {proposal.proposal_id}: {str(e)}")
            # Return safe rejection on any error
            return self._create_rejection_decision(
                proposal,
                [],
                f"Internal error during evaluation: {str(e)}"
            )
    
    def _evaluate_all_rules(self, proposal: BetProposal) -> List[RuleViolation]:
        """Evaluate proposal against all governance rules"""
        violations = []
        
        # Get current exposure snapshot
        exposure = self.exposure_store.get_current_exposure()
        
        # Bankroll and exposure rules
        violations.extend(self._check_bankroll_rules(proposal, exposure))
        
        # P&L guard rules
        violations.extend(self._check_pnl_rules(proposal))
        
        # Liquidity rules
        violations.extend(self._check_liquidity_rules(proposal))
        
        # Rate limiting rules
        violations.extend(self._check_rate_limit_rules(proposal))
        
        # Concentration rules
        violations.extend(self._check_concentration_rules(proposal, exposure))
        
        # Compliance rules
        violations.extend(self._check_compliance_rules(proposal))
        
        return violations
    
    def _check_bankroll_rules(self, proposal: BetProposal, 
                            exposure: ExposureSnapshot) -> List[RuleViolation]:
        """Check bankroll and exposure limit rules using enhanced rule engine"""
        try:
            # Use the enhanced bankroll rule engine
            bankroll_violations = self.bankroll_engine.evaluate_all(proposal, exposure)
            
            # Convert to the expected RuleViolation format
            violations = []
            for violation in bankroll_violations:
                violations.append(RuleViolation(
                    violation.rule_id,
                    violation.message,
                    current_value=violation.current_value,
                    threshold=violation.threshold
                ))
            
            return violations
            
        except Exception as e:
            logger.error(f"Error in bankroll rules evaluation: {str(e)}")
            # Fallback to basic violation
            return [RuleViolation(
                RuleId.BANKROLL_MAX_EXPOSURE,
                f"Bankroll rule evaluation error: {str(e)}"
            )]
    
    def _check_pnl_rules(self, proposal: BetProposal) -> List[RuleViolation]:
        """Check P&L protection guard rules using enhanced rule engine"""
        try:
            # Get current exposure for P&L rule evaluation
            exposure = self.exposure_store.get_current_exposure()
            
            # Use the enhanced P&L rule engine
            pnl_violations = self.pnl_engine.evaluate_all(proposal, exposure)
            
            # Convert to the expected RuleViolation format
            violations = []
            for violation in pnl_violations:
                violations.append(RuleViolation(
                    violation.rule_id,
                    violation.message,
                    current_value=violation.current_value,
                    threshold=violation.threshold
                ))
            
            return violations
            
        except Exception as e:
            logger.error(f"Error in P&L rules evaluation: {str(e)}")
            # Fallback to basic violation
            return [RuleViolation(
                RuleId.PNL_DAILY_LOSS_LIMIT,
                f"P&L rule evaluation error: {str(e)}"
            )]
    
    def _check_liquidity_rules(self, proposal: BetProposal) -> List[RuleViolation]:
        """Check liquidity and execution constraint rules using enhanced rule engine"""
        try:
            # Get current exposure for liquidity rule evaluation
            exposure = self.exposure_store.get_current_exposure()
            
            # Use the enhanced liquidity rule engine
            liquidity_violations = self.liquidity_engine.evaluate_all(proposal, exposure)
            
            # Convert to the expected RuleViolation format
            violations = []
            for violation in liquidity_violations:
                violations.append(RuleViolation(
                    violation.rule_id,
                    violation.message,
                    current_value=violation.current_value,
                    threshold=violation.threshold
                ))
            
            return violations
            
        except Exception as e:
            logger.error(f"Error in liquidity rules evaluation: {str(e)}")
            # Fallback to basic violation
            return [RuleViolation(
                RuleId.LIQ_FRACTION_LIMIT,
                f"Liquidity rule evaluation error: {str(e)}"
            )]
    
    def _check_rate_limit_rules(self, proposal: BetProposal) -> List[RuleViolation]:
        """Check rate limiting rules"""
        violations = []
        
        # Check market-level rate limiting
        # This is a simplified implementation - in production would use proper rate limiter
        recent_bets = self.audit_store.get_recent_decisions_for_market(
            proposal.market_id, 
            seconds=self.config.ops.rate_limit.per_seconds
        )
        
        if len(recent_bets) >= self.config.ops.rate_limit.count:
            violations.append(RuleViolation(
                RuleId.RATE_LIMIT_EXCEEDED,
                f"Rate limit exceeded: {len(recent_bets)} bets in last "
                f"{self.config.ops.rate_limit.per_seconds} seconds",
                current_value=len(recent_bets),
                threshold=self.config.ops.rate_limit.count
            ))
        
        return violations
    
    def _check_concentration_rules(self, proposal: BetProposal, 
                                 exposure: ExposureSnapshot) -> List[RuleViolation]:
        """Check concentration and correlation limit rules"""
        violations = []
        
        # Count concurrent markets for this match
        match_markets = sum(1 for market_id in exposure.per_market_exposure.keys() 
                          if market_id.startswith(proposal.match_id))
        
        if match_markets >= self.config.concentration.max_concurrent_markets_per_match:
            violations.append(RuleViolation(
                RuleId.CONC_MAX_MARKETS_PER_MATCH,
                f"Already have {match_markets} markets for match {proposal.match_id}, "
                f"exceeds limit {self.config.concentration.max_concurrent_markets_per_match}",
                current_value=match_markets,
                threshold=self.config.concentration.max_concurrent_markets_per_match
            ))
        
        # Check correlation group exposure
        if proposal.correlation_group:
            correlation_exposure = exposure.per_correlation_group.get(proposal.correlation_group, 0.0)
            potential_exposure = self._calculate_potential_exposure(proposal)
            new_correlation_exposure = correlation_exposure + potential_exposure
            max_correlation_exposure = exposure.bankroll * (
                self.config.concentration.max_correlation_group_exposure_pct / 100
            )
            
            if new_correlation_exposure > max_correlation_exposure:
                violations.append(RuleViolation(
                    RuleId.CONC_CORRELATION_GROUP,
                    f"Correlation group exposure would exceed "
                    f"{self.config.concentration.max_correlation_group_exposure_pct}% of bankroll",
                    current_value=new_correlation_exposure,
                    threshold=max_correlation_exposure
                ))
        
        return violations
    
    def _check_compliance_rules(self, proposal: BetProposal) -> List[RuleViolation]:
        """Check compliance and operational control rules"""
        violations = []
        
        # Currency check
        if proposal.currency not in self.config.compliance.allowed_currencies:
            violations.append(RuleViolation(
                RuleId.COMP_CURRENCY,
                f"Currency {proposal.currency} not in allowed list: "
                f"{', '.join(self.config.compliance.allowed_currencies)}"
            ))
        
        # Blocked markets check
        if proposal.market_id in self.config.compliance.blocked_markets:
            violations.append(RuleViolation(
                RuleId.COMP_BLOCKED_MARKET,
                f"Market {proposal.market_id} is blocked"
            ))
        
        # Dual approval threshold check
        if proposal.stake > self.config.ops.require_dual_approval_threshold_gbp:
            # Convert to GBP if needed (simplified - would need real FX rates)
            stake_gbp = proposal.stake
            if proposal.currency != "GBP":
                # Placeholder conversion - in production would use real FX service
                stake_gbp = proposal.stake  # Assume 1:1 for now
            
            if stake_gbp > self.config.ops.require_dual_approval_threshold_gbp:
                violations.append(RuleViolation(
                    RuleId.COMP_DUAL_APPROVAL,
                    f"Stake £{stake_gbp:.2f} exceeds dual approval threshold "
                    f"£{self.config.ops.require_dual_approval_threshold_gbp:.2f}",
                    severity="WARNING"  # Not a hard rejection, just requires approval
                ))
        
        return violations
    
    def _calculate_potential_exposure(self, proposal: BetProposal) -> float:
        """Calculate potential exposure for a bet proposal"""
        if proposal.side.value == "BACK":
            # For backing, exposure is the stake amount
            return proposal.stake
        else:
            # For laying, exposure is (odds - 1) * stake
            return (proposal.odds - 1) * proposal.stake
    
    def _make_decision(self, proposal: BetProposal, 
                      violations: List[RuleViolation]) -> GovernanceDecision:
        """Make final decision based on rule violations"""
        
        # Filter out warnings (dual approval requirements)
        hard_violations = [v for v in violations if v.severity != "WARNING"]
        warning_violations = [v for v in violations if v.severity == "WARNING"]
        
        # If we have hard violations, reject
        if hard_violations:
            return self._create_rejection_decision(
                proposal,
                [v.rule_id for v in hard_violations],
                self._format_violation_messages(hard_violations)
            )
        
        # If we only have warnings, we might be able to approve with amendments
        if warning_violations:
            # For now, just approve with warnings logged
            # In a full implementation, this would trigger dual approval workflow
            return self._create_approval_decision(
                proposal,
                [v.rule_id for v in warning_violations],
                f"Approved with warnings: {self._format_violation_messages(warning_violations)}"
            )
        
        # No violations - approve
        return self._create_approval_decision(
            proposal,
            [],
            "All governance rules satisfied"
        )
    
    def _create_approval_decision(self, proposal: BetProposal, 
                                rule_ids: List[RuleId],
                                message: str) -> GovernanceDecision:
        """Create an approval decision"""
        return GovernanceDecision(
            proposal_id=proposal.proposal_id,
            decision=DecisionType.APPROVE,
            rule_ids_triggered=rule_ids,
            human_message=message,
            state=self._current_state,
            ttl_seconds=5,
            audit_ref="pending"  # Will be updated after audit record creation
        )
    
    def _create_rejection_decision(self, proposal: BetProposal,
                                 rule_ids: List[RuleId],
                                 message: str) -> GovernanceDecision:
        """Create a rejection decision"""
        return GovernanceDecision(
            proposal_id=proposal.proposal_id,
            decision=DecisionType.REJECT,
            rule_ids_triggered=rule_ids,
            human_message=message,
            state=self._current_state,
            ttl_seconds=5,
            audit_ref="pending"  # Will be updated after audit record creation
        )
    
    def _create_amendment_decision(self, proposal: BetProposal,
                                 amendment: BetAmendment,
                                 rule_ids: List[RuleId],
                                 message: str) -> GovernanceDecision:
        """Create an amendment decision"""
        return GovernanceDecision(
            proposal_id=proposal.proposal_id,
            decision=DecisionType.AMEND,
            amendment=amendment,
            rule_ids_triggered=rule_ids,
            human_message=message,
            state=self._current_state,
            ttl_seconds=5,
            audit_ref="pending"  # Will be updated after audit record creation
        )
    
    def _format_violation_messages(self, violations: List[RuleViolation]) -> str:
        """Format violation messages for human consumption"""
        if not violations:
            return ""
        
        messages = [v.message for v in violations]
        if len(messages) == 1:
            return messages[0]
        
        return "; ".join(messages)
    
    def _create_audit_record(self, proposal: BetProposal, 
                           decision: GovernanceDecision) -> AuditRecord:
        """Create audit record for the decision"""
        exposure_snapshot = self.exposure_store.get_current_exposure()
        
        return AuditRecord(
            proposal_id=proposal.proposal_id,
            decision=decision.decision,
            rule_ids=decision.rule_ids_triggered,
            snapshot=exposure_snapshot
        )
    
    # State management methods
    
    def set_state(self, new_state: GovernanceState) -> bool:
        """Change governance state"""
        old_state = self._current_state
        self._current_state = new_state
        
        logger.info(f"DGL state changed from {old_state} to {new_state}")
        return True
    
    def get_state(self) -> GovernanceState:
        """Get current governance state"""
        return self._current_state
    
    def activate_kill_switch(self) -> bool:
        """Activate emergency kill switch"""
        self._kill_switch_active = True
        logger.warning("DGL kill switch activated - all betting disabled")
        return True
    
    def deactivate_kill_switch(self) -> bool:
        """Deactivate emergency kill switch"""
        self._kill_switch_active = False
        logger.info("DGL kill switch deactivated")
        return True
    
    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active"""
        return self._kill_switch_active
    
    # Statistics and monitoring
    
    def get_statistics(self) -> Dict:
        """Get engine performance statistics"""
        if not self._processing_times:
            return {
                "total_decisions": self._decision_count,
                "avg_processing_time_ms": 0.0,
                "p99_processing_time_ms": 0.0,
                "current_state": self._current_state.value,
                "kill_switch_active": self._kill_switch_active
            }
        
        sorted_times = sorted(self._processing_times)
        p99_index = int(len(sorted_times) * 0.99)
        
        return {
            "total_decisions": self._decision_count,
            "avg_processing_time_ms": sum(self._processing_times) / len(self._processing_times),
            "p99_processing_time_ms": sorted_times[p99_index] if sorted_times else 0.0,
            "current_state": self._current_state.value,
            "kill_switch_active": self._kill_switch_active,
            "recent_decisions_count": len(self._processing_times)
        }
