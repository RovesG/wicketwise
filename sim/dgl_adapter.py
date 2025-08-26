# Purpose: SIM DGL adapter for risk enforcement in simulation mode
# Author: WicketWise AI, Last Modified: 2024

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json

try:
    from .strategy import StrategyAction, AccountState, OrderSide
    from .config import RiskProfile
except ImportError:
    from strategy import StrategyAction, AccountState, OrderSide
    from config import RiskProfile


class DGLDecision(Enum):
    """DGL decision types"""
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    AMEND = "AMEND"


@dataclass
class DGLResponse:
    """DGL decision response"""
    decision: DGLDecision
    reason: str
    amended_size: Optional[float] = None
    rule_ids_triggered: List[str] = None
    audit_ref: str = ""
    
    def __post_init__(self):
        if self.rule_ids_triggered is None:
            self.rule_ids_triggered = []
        if not self.audit_ref:
            self.audit_ref = f"sim_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "amended_size": self.amended_size,
            "rule_ids_triggered": self.rule_ids_triggered,
            "audit_ref": self.audit_ref
        }


class SimDGLAdapter:
    """
    Simulation DGL adapter that enforces risk rules in simulation mode
    
    Implements the same rule logic as production DGL but routes to simulation
    audit namespace and uses simplified rule checking.
    """
    
    def __init__(self, risk_profile: RiskProfile):
        self.risk_profile = risk_profile
        self.audit_log: List[Dict[str, Any]] = []
        
        # Track exposures per market
        self.market_exposures: Dict[str, float] = {}
        self.total_exposure = 0.0
        
        # Track P&L for loss limits
        self.daily_pnl = 0.0
        self.session_pnl = 0.0
        self.consecutive_losses = 0
        
        # Rule violation counts
        self.violations: Dict[str, int] = {}
    
    def evaluate_action(self, action: StrategyAction, account_state: AccountState) -> DGLResponse:
        """
        Evaluate strategy action against DGL rules
        
        Args:
            action: Strategy action to evaluate
            account_state: Current account state
            
        Returns:
            DGL response with decision and reasoning
        """
        
        # Calculate potential exposure from this action
        potential_exposure = self._calculate_potential_exposure(action)
        
        # Check bankroll rules
        bankroll_check = self._check_bankroll_rules(action, account_state, potential_exposure)
        if bankroll_check.decision != DGLDecision.APPROVE:
            self._log_decision(action, bankroll_check)
            return bankroll_check
        
        # Check P&L protection rules
        pnl_check = self._check_pnl_rules(action, account_state)
        if pnl_check.decision != DGLDecision.APPROVE:
            self._log_decision(action, pnl_check)
            return pnl_check
        
        # Check market concentration limits
        concentration_check = self._check_concentration_rules(action, account_state, potential_exposure)
        if concentration_check.decision != DGLDecision.APPROVE:
            self._log_decision(action, concentration_check)
            return concentration_check
        
        # Check position size limits
        size_check = self._check_position_size_rules(action, account_state)
        if size_check.decision != DGLDecision.APPROVE:
            self._log_decision(action, size_check)
            return size_check
        
        # All checks passed
        approved_response = DGLResponse(
            decision=DGLDecision.APPROVE,
            reason="All governance rules satisfied",
            rule_ids_triggered=[]
        )
        
        self._log_decision(action, approved_response)
        return approved_response
    
    def _check_bankroll_rules(self, action: StrategyAction, account_state: AccountState, 
                             potential_exposure: float) -> DGLResponse:
        """Check bankroll and exposure limits"""
        
        total_balance = account_state.total_balance()
        
        # Check total bankroll exposure
        new_total_exposure = account_state.exposure + potential_exposure
        max_total_exposure = total_balance * (self.risk_profile.max_exposure_pct / 100.0)
        
        if new_total_exposure > max_total_exposure:
            return DGLResponse(
                decision=DGLDecision.REJECT,
                reason=f"Total exposure limit exceeded: {new_total_exposure:.2f} > {max_total_exposure:.2f}",
                rule_ids_triggered=["BANKROLL_TOTAL_EXPOSURE"]
            )
        
        # Check per-market exposure
        market_exposure = self.market_exposures.get(action.market_id, 0.0) + potential_exposure
        max_market_exposure = total_balance * (self.risk_profile.per_market_cap_pct / 100.0)
        
        if market_exposure > max_market_exposure:
            return DGLResponse(
                decision=DGLDecision.REJECT,
                reason=f"Market exposure limit exceeded: {market_exposure:.2f} > {max_market_exposure:.2f}",
                rule_ids_triggered=["BANKROLL_MARKET_EXPOSURE"]
            )
        
        # Check per-bet size
        bet_size = action.size * action.price if action.side == OrderSide.BACK else action.size * (action.price - 1.0)
        max_bet_size = total_balance * (self.risk_profile.per_bet_cap_pct / 100.0)
        
        if bet_size > max_bet_size:
            # Try to amend size
            amended_size = max_bet_size / action.price if action.side == OrderSide.BACK else max_bet_size / (action.price - 1.0)
            
            if amended_size >= action.size * 0.1:  # At least 10% of original size
                return DGLResponse(
                    decision=DGLDecision.AMEND,
                    reason=f"Bet size reduced to comply with limits: {bet_size:.2f} > {max_bet_size:.2f}",
                    amended_size=amended_size,
                    rule_ids_triggered=["BANKROLL_BET_SIZE"]
                )
            else:
                return DGLResponse(
                    decision=DGLDecision.REJECT,
                    reason=f"Bet size too large even after amendment: {bet_size:.2f} > {max_bet_size:.2f}",
                    rule_ids_triggered=["BANKROLL_BET_SIZE"]
                )
        
        return DGLResponse(decision=DGLDecision.APPROVE, reason="Bankroll checks passed")
    
    def _check_pnl_rules(self, action: StrategyAction, account_state: AccountState) -> DGLResponse:
        """Check P&L protection rules"""
        
        # Check daily loss limit
        daily_loss_limit = account_state.total_balance() * (self.risk_profile.max_exposure_pct / 100.0) * 0.4  # 40% of max exposure as daily limit
        
        if self.daily_pnl < -daily_loss_limit:
            return DGLResponse(
                decision=DGLDecision.REJECT,
                reason=f"Daily loss limit exceeded: {self.daily_pnl:.2f} < -{daily_loss_limit:.2f}",
                rule_ids_triggered=["PNL_DAILY_LOSS"]
            )
        
        # Check consecutive losses (simplified)
        if self.consecutive_losses >= 5:
            return DGLResponse(
                decision=DGLDecision.REJECT,
                reason=f"Consecutive loss limit exceeded: {self.consecutive_losses} >= 5",
                rule_ids_triggered=["PNL_CONSECUTIVE_LOSS"]
            )
        
        return DGLResponse(decision=DGLDecision.APPROVE, reason="P&L checks passed")
    
    def _check_concentration_rules(self, action: StrategyAction, account_state: AccountState,
                                  potential_exposure: float) -> DGLResponse:
        """Check market concentration limits"""
        
        # Check correlation limits (simplified - would use actual correlation matrix)
        total_markets = len(self.market_exposures) + (1 if action.market_id not in self.market_exposures else 0)
        
        if total_markets > 10:  # Max 10 concurrent markets
            return DGLResponse(
                decision=DGLDecision.REJECT,
                reason=f"Too many concurrent markets: {total_markets} > 10",
                rule_ids_triggered=["CONCENTRATION_MARKET_COUNT"]
            )
        
        return DGLResponse(decision=DGLDecision.APPROVE, reason="Concentration checks passed")
    
    def _check_position_size_rules(self, action: StrategyAction, account_state: AccountState) -> DGLResponse:
        """Check position size and liquidity rules"""
        
        # Check minimum bet size
        min_bet_size = 10.0  # Minimum £10 bet
        bet_size = action.size * action.price if action.side == OrderSide.BACK else action.size * (action.price - 1.0)
        
        if bet_size < min_bet_size:
            return DGLResponse(
                decision=DGLDecision.REJECT,
                reason=f"Bet size too small: {bet_size:.2f} < {min_bet_size:.2f}",
                rule_ids_triggered=["POSITION_MIN_SIZE"]
            )
        
        # Check odds limits
        if action.price < 1.01 or action.price > 1000.0:
            return DGLResponse(
                decision=DGLDecision.REJECT,
                reason=f"Odds outside acceptable range: {action.price}",
                rule_ids_triggered=["POSITION_ODDS_RANGE"]
            )
        
        return DGLResponse(decision=DGLDecision.APPROVE, reason="Position size checks passed")
    
    def _calculate_potential_exposure(self, action: StrategyAction) -> float:
        """Calculate potential exposure from action"""
        if action.side == OrderSide.BACK:
            # Back bet - exposure is the stake
            return action.size * action.price
        else:
            # Lay bet - exposure is the liability
            return action.size * (action.price - 1.0)
    
    def update_exposures(self, action: StrategyAction, fill_size: float):
        """Update exposure tracking after fill"""
        exposure_change = self._calculate_potential_exposure(action) * (fill_size / action.size)
        
        if action.market_id not in self.market_exposures:
            self.market_exposures[action.market_id] = 0.0
        
        self.market_exposures[action.market_id] += exposure_change
        self.total_exposure += exposure_change
    
    def update_pnl(self, pnl_change: float):
        """Update P&L tracking"""
        self.daily_pnl += pnl_change
        self.session_pnl += pnl_change
        
        # Track consecutive losses
        if pnl_change < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
    
    def _log_decision(self, action: StrategyAction, response: DGLResponse):
        """Log DGL decision to audit trail"""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action.to_dict(),
            "decision": response.to_dict(),
            "account_exposure": self.total_exposure,
            "market_exposures": self.market_exposures.copy(),
            "daily_pnl": self.daily_pnl,
            "consecutive_losses": self.consecutive_losses
        }
        
        self.audit_log.append(audit_entry)
        
        # Track violations
        if response.decision != DGLDecision.APPROVE:
            for rule_id in response.rule_ids_triggered:
                self.violations[rule_id] = self.violations.get(rule_id, 0) + 1
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get complete audit log"""
        return self.audit_log.copy()
    
    def get_violations_summary(self) -> Dict[str, int]:
        """Get summary of rule violations"""
        return self.violations.copy()
    
    def get_current_exposures(self) -> Dict[str, Any]:
        """Get current exposure summary"""
        return {
            "total_exposure": self.total_exposure,
            "market_exposures": self.market_exposures.copy(),
            "daily_pnl": self.daily_pnl,
            "session_pnl": self.session_pnl,
            "consecutive_losses": self.consecutive_losses,
            "violation_count": sum(self.violations.values())
        }
    
    def reset(self):
        """Reset DGL state for new simulation"""
        self.market_exposures.clear()
        self.total_exposure = 0.0
        self.daily_pnl = 0.0
        self.session_pnl = 0.0
        self.consecutive_losses = 0
        self.violations.clear()
        self.audit_log.clear()
    
    def export_audit_log(self, file_path: str):
        """Export audit log to file"""
        with open(file_path, 'w') as f:
            for entry in self.audit_log:
                f.write(json.dumps(entry) + '\n')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DGL statistics"""
        total_decisions = len(self.audit_log)
        approvals = sum(1 for entry in self.audit_log if entry["decision"]["decision"] == "APPROVE")
        rejections = sum(1 for entry in self.audit_log if entry["decision"]["decision"] == "REJECT")
        amendments = sum(1 for entry in self.audit_log if entry["decision"]["decision"] == "AMEND")
        
        return {
            "total_decisions": total_decisions,
            "approvals": approvals,
            "rejections": rejections,
            "amendments": amendments,
            "approval_rate": approvals / total_decisions if total_decisions > 0 else 0.0,
            "rejection_rate": rejections / total_decisions if total_decisions > 0 else 0.0,
            "amendment_rate": amendments / total_decisions if total_decisions > 0 else 0.0,
            "violations_by_rule": self.violations.copy(),
            "current_exposures": self.get_current_exposures()
        }
