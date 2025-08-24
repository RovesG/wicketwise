# Purpose: P&L protection guard rule implementations
# Author: WicketWise AI, Last Modified: 2024

from typing import List, Optional
from datetime import datetime, timedelta
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import BetProposal, ExposureSnapshot, RuleId
from config import PnLGuardsConfig
from store import PnLStore


logger = logging.getLogger(__name__)


class RuleViolation:
    """Represents a rule violation with detailed context"""
    
    def __init__(self, rule_id: RuleId, message: str, 
                 current_value: Optional[float] = None,
                 threshold: Optional[float] = None,
                 severity: str = "ERROR"):
        self.rule_id = rule_id
        self.message = message
        self.current_value = current_value
        self.threshold = threshold
        self.severity = severity


class DailyLossLimitRule:
    """
    Rule: Daily loss limit protection
    
    Prevents new bets if daily losses exceed a specified percentage of bankroll.
    This is a critical risk management rule to prevent catastrophic daily losses.
    """
    
    def __init__(self, config: PnLGuardsConfig, pnl_store: PnLStore):
        self.config = config
        self.pnl_store = pnl_store
        self.rule_id = RuleId.PNL_DAILY_LOSS_LIMIT
        
    def evaluate(self, proposal: BetProposal, exposure: ExposureSnapshot) -> Optional[RuleViolation]:
        """
        Evaluate the daily loss limit rule
        
        Args:
            proposal: The bet proposal to evaluate
            exposure: Current exposure snapshot
            
        Returns:
            RuleViolation if daily losses exceed limit, None otherwise
        """
        try:
            # Get current daily P&L
            daily_pnl = self.pnl_store.get_daily_pnl()
            
            # Calculate daily loss limit (negative value)
            daily_loss_limit = -(exposure.bankroll * (self.config.daily_loss_limit_pct / 100))
            
            # Check if we're already at or below the loss limit
            if daily_pnl <= daily_loss_limit:
                loss_amount = abs(daily_pnl)
                limit_amount = abs(daily_loss_limit)
                
                return RuleViolation(
                    rule_id=self.rule_id,
                    message=f"Daily loss limit exceeded. Current loss: {loss_amount:.2f}, "
                           f"Limit: {limit_amount:.2f} ({self.config.daily_loss_limit_pct}% of bankroll). "
                           f"No new bets allowed until next trading day.",
                    current_value=loss_amount,
                    threshold=limit_amount,
                    severity="CRITICAL"
                )
            
            # Check if we're approaching the limit (within 80% of limit)
            warning_threshold = daily_loss_limit * 0.8
            if daily_pnl <= warning_threshold:
                loss_amount = abs(daily_pnl)
                limit_amount = abs(daily_loss_limit)
                remaining = limit_amount - loss_amount
                
                return RuleViolation(
                    rule_id=self.rule_id,
                    message=f"Approaching daily loss limit. Current loss: {loss_amount:.2f}, "
                           f"Remaining before limit: {remaining:.2f}. Exercise caution.",
                    current_value=loss_amount,
                    threshold=limit_amount,
                    severity="WARNING"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating daily loss limit rule: {str(e)}")
            return RuleViolation(
                rule_id=self.rule_id,
                message=f"Unable to evaluate daily P&L: {str(e)}",
                severity="ERROR"
            )


class SessionLossLimitRule:
    """
    Rule: Session loss limit protection
    
    Prevents new bets if session losses exceed a specified percentage of bankroll.
    Provides intra-session risk management to prevent large losses within a single session.
    """
    
    def __init__(self, config: PnLGuardsConfig, pnl_store: PnLStore):
        self.config = config
        self.pnl_store = pnl_store
        self.rule_id = RuleId.PNL_SESSION_LOSS_LIMIT
        
    def evaluate(self, proposal: BetProposal, exposure: ExposureSnapshot) -> Optional[RuleViolation]:
        """
        Evaluate the session loss limit rule
        
        Args:
            proposal: The bet proposal to evaluate
            exposure: Current exposure snapshot
            
        Returns:
            RuleViolation if session losses exceed limit, None otherwise
        """
        try:
            # Get current session P&L
            session_pnl = self.pnl_store.get_session_pnl()
            
            # Calculate session loss limit (negative value)
            session_loss_limit = -(exposure.bankroll * (self.config.session_loss_limit_pct / 100))
            
            # Check if we're already at or below the loss limit
            if session_pnl <= session_loss_limit:
                loss_amount = abs(session_pnl)
                limit_amount = abs(session_loss_limit)
                
                return RuleViolation(
                    rule_id=self.rule_id,
                    message=f"Session loss limit exceeded. Current session loss: {loss_amount:.2f}, "
                           f"Limit: {limit_amount:.2f} ({self.config.session_loss_limit_pct}% of bankroll). "
                           f"No new bets allowed until session reset.",
                    current_value=loss_amount,
                    threshold=limit_amount,
                    severity="CRITICAL"
                )
            
            # Check if we're approaching the limit (within 80% of limit)
            warning_threshold = session_loss_limit * 0.8
            if session_pnl <= warning_threshold:
                loss_amount = abs(session_pnl)
                limit_amount = abs(session_loss_limit)
                remaining = limit_amount - loss_amount
                
                return RuleViolation(
                    rule_id=self.rule_id,
                    message=f"Approaching session loss limit. Current session loss: {loss_amount:.2f}, "
                           f"Remaining before limit: {remaining:.2f}. Consider session break.",
                    current_value=loss_amount,
                    threshold=limit_amount,
                    severity="WARNING"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating session loss limit rule: {str(e)}")
            return RuleViolation(
                rule_id=self.rule_id,
                message=f"Unable to evaluate session P&L: {str(e)}",
                severity="ERROR"
            )


class PnLTrendAnalysisRule:
    """
    Rule: P&L trend analysis and early warning
    
    Analyzes recent P&L trends to provide early warnings about deteriorating performance.
    This is a proactive risk management rule that doesn't block bets but provides warnings.
    """
    
    def __init__(self, config: PnLGuardsConfig, pnl_store: PnLStore):
        self.config = config
        self.pnl_store = pnl_store
        self.rule_id = RuleId.PNL_DAILY_LOSS_LIMIT  # Reuse for trend analysis
        
    def evaluate(self, proposal: BetProposal, exposure: ExposureSnapshot) -> Optional[RuleViolation]:
        """
        Evaluate P&L trends for early warning signals
        
        Args:
            proposal: The bet proposal to evaluate
            exposure: Current exposure snapshot
            
        Returns:
            RuleViolation with warning if negative trends detected, None otherwise
        """
        try:
            # Get P&L history for trend analysis
            pnl_history = self.pnl_store.get_pnl_history(days=7)  # Last 7 days
            
            if len(pnl_history) < 3:
                # Not enough data for trend analysis
                return None
            
            # Calculate recent performance metrics
            recent_days = list(pnl_history.values())[-3:]  # Last 3 days
            negative_days = sum(1 for pnl in recent_days if pnl < 0)
            total_recent_pnl = sum(recent_days)
            
            # Check for concerning trends
            if negative_days >= 2 and total_recent_pnl < 0:
                avg_daily_loss = abs(total_recent_pnl / len(recent_days))
                
                return RuleViolation(
                    rule_id=self.rule_id,
                    message=f"Negative P&L trend detected: {negative_days} losing days in last 3 days. "
                           f"Average daily loss: {avg_daily_loss:.2f}. Consider strategy review.",
                    current_value=total_recent_pnl,
                    threshold=0.0,
                    severity="WARNING"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating P&L trend: {str(e)}")
            return None  # Don't block bets for trend analysis errors


class PnLRuleEngine:
    """
    P&L rule engine that evaluates all P&L protection rules
    """
    
    def __init__(self, config: PnLGuardsConfig, pnl_store: PnLStore):
        self.config = config
        self.pnl_store = pnl_store
        
        # Initialize all P&L rules
        self.rules = [
            DailyLossLimitRule(config, pnl_store),
            SessionLossLimitRule(config, pnl_store),
            PnLTrendAnalysisRule(config, pnl_store)
        ]
        
        logger.info(f"Initialized PnLRuleEngine with {len(self.rules)} rules")
    
    def evaluate_all(self, proposal: BetProposal, exposure: ExposureSnapshot) -> List[RuleViolation]:
        """
        Evaluate all P&L rules against a proposal
        
        Args:
            proposal: The bet proposal to evaluate
            exposure: Current exposure snapshot
            
        Returns:
            List of rule violations (empty if no violations)
        """
        violations = []
        
        for rule in self.rules:
            try:
                violation = rule.evaluate(proposal, exposure)
                if violation:
                    violations.append(violation)
                    logger.debug(f"P&L rule violation: {violation.rule_id.value} - {violation.message}")
            except Exception as e:
                logger.error(f"Error evaluating P&L rule {rule.rule_id.value}: {str(e)}")
                # Create a generic violation for rule evaluation errors
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    message=f"P&L rule evaluation error: {str(e)}",
                    severity="ERROR"
                ))
        
        return violations
    
    def get_pnl_summary(self) -> dict:
        """Get current P&L summary"""
        try:
            return {
                "daily_pnl": self.pnl_store.get_daily_pnl(),
                "session_pnl": self.pnl_store.get_session_pnl(),
                "pnl_history_7d": self.pnl_store.get_pnl_history(days=7),
                "daily_loss_limit_pct": self.config.daily_loss_limit_pct,
                "session_loss_limit_pct": self.config.session_loss_limit_pct
            }
        except Exception as e:
            logger.error(f"Error getting P&L summary: {str(e)}")
            return {"error": str(e)}
    
    def reset_session(self) -> bool:
        """Reset the current trading session"""
        try:
            self.pnl_store.start_new_session()
            logger.info("Trading session reset")
            return True
        except Exception as e:
            logger.error(f"Error resetting session: {str(e)}")
            return False
    
    def get_statistics(self) -> dict:
        """Get statistics about P&L rules"""
        pnl_summary = self.get_pnl_summary()
        
        return {
            "total_rules": len(self.rules),
            "rule_ids": [rule.rule_id.value for rule in self.rules],
            "config": {
                "daily_loss_limit_pct": self.config.daily_loss_limit_pct,
                "session_loss_limit_pct": self.config.session_loss_limit_pct
            },
            "current_pnl": pnl_summary
        }
