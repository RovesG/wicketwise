# Purpose: Bankroll and exposure limit rule implementations
# Author: WicketWise AI, Last Modified: 2024

from typing import List, Optional
from decimal import Decimal
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import BetProposal, ExposureSnapshot, RuleId, BetSide
from config import BankrollConfig


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


class BankrollExposureRule:
    """
    Rule: Maximum total bankroll exposure percentage
    
    Ensures that the total open exposure across all positions does not exceed
    a specified percentage of the total bankroll.
    """
    
    def __init__(self, config: BankrollConfig):
        self.config = config
        self.rule_id = RuleId.BANKROLL_MAX_EXPOSURE
        
    def evaluate(self, proposal: BetProposal, exposure: ExposureSnapshot) -> Optional[RuleViolation]:
        """
        Evaluate the bankroll exposure rule
        
        Args:
            proposal: The bet proposal to evaluate
            exposure: Current exposure snapshot
            
        Returns:
            RuleViolation if rule is violated, None otherwise
        """
        # Calculate potential new exposure from this bet
        potential_exposure = self._calculate_bet_exposure(proposal)
        
        # Calculate total exposure after this bet
        new_total_exposure = exposure.open_exposure + potential_exposure
        
        # Calculate maximum allowed exposure
        max_allowed_exposure = exposure.bankroll * (self.config.max_bankroll_exposure_pct / 100)
        
        # Check if rule would be violated
        if new_total_exposure > max_allowed_exposure:
            # Calculate suggested stake reduction
            available_exposure = max_allowed_exposure - exposure.open_exposure
            
            if available_exposure <= 0:
                # No room for any new bets
                return RuleViolation(
                    rule_id=self.rule_id,
                    message=f"Total exposure ({exposure.open_exposure:.2f}) already at maximum "
                           f"({self.config.max_bankroll_exposure_pct}% of bankroll). No new bets allowed.",
                    current_value=exposure.open_exposure,
                    threshold=max_allowed_exposure,
                    severity="ERROR"
                )
            else:
                # Suggest reduced stake
                if proposal.side == BetSide.BACK:
                    suggested_stake = available_exposure
                else:  # LAY
                    suggested_stake = available_exposure / (proposal.odds - 1)
                
                return RuleViolation(
                    rule_id=self.rule_id,
                    message=f"Bet would exceed maximum bankroll exposure "
                           f"({self.config.max_bankroll_exposure_pct}%). "
                           f"Suggested stake reduction to {suggested_stake:.2f}",
                    current_value=new_total_exposure,
                    threshold=max_allowed_exposure,
                    severity="ERROR",
                    suggested_amendment={"stake": suggested_stake}
                )
        
        return None
    
    def _calculate_bet_exposure(self, proposal: BetProposal) -> float:
        """Calculate the exposure for a bet proposal"""
        if proposal.side == BetSide.BACK:
            # For backing, exposure is the stake amount
            return proposal.stake
        else:
            # For laying, exposure is (odds - 1) * stake
            return (proposal.odds - 1) * proposal.stake


class PerMatchExposureRule:
    """
    Rule: Maximum exposure per match
    
    Limits the total exposure for any single match to a percentage of bankroll.
    """
    
    def __init__(self, config: BankrollConfig):
        self.config = config
        self.rule_id = RuleId.EXPO_PER_MATCH_MAX
        
    def evaluate(self, proposal: BetProposal, exposure: ExposureSnapshot) -> Optional[RuleViolation]:
        """Evaluate the per-match exposure rule"""
        # Calculate potential new exposure from this bet
        potential_exposure = self._calculate_bet_exposure(proposal)
        
        # Get current match exposure
        current_match_exposure = exposure.per_match_exposure.get(proposal.match_id, 0.0)
        
        # Calculate new match exposure
        new_match_exposure = current_match_exposure + potential_exposure
        
        # Calculate maximum allowed match exposure
        max_match_exposure = exposure.bankroll * (self.config.per_match_max_pct / 100)
        
        # Check if rule would be violated
        if new_match_exposure > max_match_exposure:
            # Calculate available exposure for this match
            available_exposure = max_match_exposure - current_match_exposure
            
            if available_exposure <= 0:
                return RuleViolation(
                    rule_id=self.rule_id,
                    message=f"Match {proposal.match_id} already at maximum exposure "
                           f"({self.config.per_match_max_pct}% of bankroll). No new bets allowed.",
                    current_value=current_match_exposure,
                    threshold=max_match_exposure,
                    severity="ERROR"
                )
            else:
                # Suggest reduced stake
                if proposal.side == BetSide.BACK:
                    suggested_stake = available_exposure
                else:  # LAY
                    suggested_stake = available_exposure / (proposal.odds - 1)
                
                return RuleViolation(
                    rule_id=self.rule_id,
                    message=f"Bet would exceed maximum match exposure "
                           f"({self.config.per_match_max_pct}%). "
                           f"Suggested stake reduction to {suggested_stake:.2f}",
                    current_value=new_match_exposure,
                    threshold=max_match_exposure,
                    severity="ERROR",
                    suggested_amendment={"stake": suggested_stake}
                )
        
        return None
    
    def _calculate_bet_exposure(self, proposal: BetProposal) -> float:
        """Calculate the exposure for a bet proposal"""
        if proposal.side == BetSide.BACK:
            return proposal.stake
        else:
            return (proposal.odds - 1) * proposal.stake


class PerMarketExposureRule:
    """
    Rule: Maximum exposure per market
    
    Limits the total exposure for any single market to a percentage of bankroll.
    """
    
    def __init__(self, config: BankrollConfig):
        self.config = config
        self.rule_id = RuleId.EXPO_PER_MARKET_MAX
        
    def evaluate(self, proposal: BetProposal, exposure: ExposureSnapshot) -> Optional[RuleViolation]:
        """Evaluate the per-market exposure rule"""
        # Calculate potential new exposure from this bet
        potential_exposure = self._calculate_bet_exposure(proposal)
        
        # Get current market exposure
        current_market_exposure = exposure.per_market_exposure.get(proposal.market_id, 0.0)
        
        # Calculate new market exposure
        new_market_exposure = current_market_exposure + potential_exposure
        
        # Calculate maximum allowed market exposure
        max_market_exposure = exposure.bankroll * (self.config.per_market_max_pct / 100)
        
        # Check if rule would be violated
        if new_market_exposure > max_market_exposure:
            # Calculate available exposure for this market
            available_exposure = max_market_exposure - current_market_exposure
            
            if available_exposure <= 0:
                return RuleViolation(
                    rule_id=self.rule_id,
                    message=f"Market {proposal.market_id} already at maximum exposure "
                           f"({self.config.per_market_max_pct}% of bankroll). No new bets allowed.",
                    current_value=current_market_exposure,
                    threshold=max_market_exposure,
                    severity="ERROR"
                )
            else:
                # Suggest reduced stake
                if proposal.side == BetSide.BACK:
                    suggested_stake = available_exposure
                else:  # LAY
                    suggested_stake = available_exposure / (proposal.odds - 1)
                
                return RuleViolation(
                    rule_id=self.rule_id,
                    message=f"Bet would exceed maximum market exposure "
                           f"({self.config.per_market_max_pct}%). "
                           f"Suggested stake reduction to {suggested_stake:.2f}",
                    current_value=new_market_exposure,
                    threshold=max_market_exposure,
                    severity="ERROR",
                    suggested_amendment={"stake": suggested_stake}
                )
        
        return None
    
    def _calculate_bet_exposure(self, proposal: BetProposal) -> float:
        """Calculate the exposure for a bet proposal"""
        if proposal.side == BetSide.BACK:
            return proposal.stake
        else:
            return (proposal.odds - 1) * proposal.stake


class PerBetExposureRule:
    """
    Rule: Maximum exposure per single bet
    
    Limits the exposure for any individual bet to a percentage of bankroll.
    """
    
    def __init__(self, config: BankrollConfig):
        self.config = config
        self.rule_id = RuleId.EXPO_PER_BET_MAX
        
    def evaluate(self, proposal: BetProposal, exposure: ExposureSnapshot) -> Optional[RuleViolation]:
        """Evaluate the per-bet exposure rule"""
        # Calculate exposure for this bet
        bet_exposure = self._calculate_bet_exposure(proposal)
        
        # Calculate maximum allowed bet exposure
        max_bet_exposure = exposure.bankroll * (self.config.per_bet_max_pct / 100)
        
        # Check if rule would be violated
        if bet_exposure > max_bet_exposure:
            # Suggest reduced stake
            if proposal.side == BetSide.BACK:
                suggested_stake = max_bet_exposure
            else:  # LAY
                suggested_stake = max_bet_exposure / (proposal.odds - 1)
            
            return RuleViolation(
                rule_id=self.rule_id,
                message=f"Bet exposure ({bet_exposure:.2f}) exceeds maximum per-bet limit "
                       f"({self.config.per_bet_max_pct}% of bankroll). "
                       f"Suggested stake reduction to {suggested_stake:.2f}",
                current_value=bet_exposure,
                threshold=max_bet_exposure,
                severity="ERROR",
                suggested_amendment={"stake": suggested_stake}
            )
        
        return None
    
    def _calculate_bet_exposure(self, proposal: BetProposal) -> float:
        """Calculate the exposure for a bet proposal"""
        if proposal.side == BetSide.BACK:
            return proposal.stake
        else:
            return (proposal.odds - 1) * proposal.stake


class BankrollRuleEngine:
    """
    Bankroll rule engine that evaluates all bankroll-related rules
    """
    
    def __init__(self, config: BankrollConfig):
        self.config = config
        
        # Initialize all bankroll rules
        self.rules = [
            BankrollExposureRule(config),
            PerMatchExposureRule(config),
            PerMarketExposureRule(config),
            PerBetExposureRule(config)
        ]
        
        logger.info(f"Initialized BankrollRuleEngine with {len(self.rules)} rules")
    
    def evaluate_all(self, proposal: BetProposal, exposure: ExposureSnapshot) -> List[RuleViolation]:
        """
        Evaluate all bankroll rules against a proposal
        
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
                    logger.debug(f"Rule violation: {violation.rule_id.value} - {violation.message}")
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.rule_id.value}: {str(e)}")
                # Create a generic violation for rule evaluation errors
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    message=f"Rule evaluation error: {str(e)}",
                    severity="ERROR"
                ))
        
        return violations
    
    def get_rule_by_id(self, rule_id: RuleId) -> Optional[object]:
        """Get a specific rule by its ID"""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None
    
    def get_statistics(self) -> dict:
        """Get statistics about bankroll rules"""
        return {
            "total_rules": len(self.rules),
            "rule_ids": [rule.rule_id.value for rule in self.rules],
            "config": {
                "max_bankroll_exposure_pct": self.config.max_bankroll_exposure_pct,
                "per_match_max_pct": self.config.per_match_max_pct,
                "per_market_max_pct": self.config.per_market_max_pct,
                "per_bet_max_pct": self.config.per_bet_max_pct
            }
        }
