# Purpose: Test SIM DGL enforcement functionality
# Author: WicketWise AI, Last Modified: 2024

import pytest
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.dgl_adapter import SimDGLAdapter, DGLDecision, DGLResponse
from sim.strategy import StrategyAction, OrderSide, OrderType, AccountState
from sim.config import RiskProfile


class TestSimDGLAdapter:
    """Test SIM DGL adapter functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.risk_profile = RiskProfile(
            bankroll=100000.0,
            max_exposure_pct=5.0,
            per_market_cap_pct=2.0,
            per_bet_cap_pct=0.5,
            correlation_cap=0.8
        )
        
        self.dgl = SimDGLAdapter(self.risk_profile)
        self.account_state = AccountState(cash=100000.0)
    
    def _create_test_action(self, side: OrderSide, price: float, size: float) -> StrategyAction:
        """Create test strategy action"""
        return StrategyAction(
            ts=datetime.now().isoformat(),
            type=OrderType.LIMIT,
            side=side,
            market_id="match_odds",
            selection_id="home",
            price=price,
            size=size,
            client_order_id=f"test_order_{datetime.now().timestamp()}"
        )
    
    def test_dgl_initialization(self):
        """Test DGL adapter initialization"""
        assert self.dgl.risk_profile.bankroll == 100000.0
        assert self.dgl.risk_profile.max_exposure_pct == 5.0
        assert len(self.dgl.audit_log) == 0
        assert self.dgl.total_exposure == 0.0
    
    def test_approve_normal_action(self):
        """Test approval of normal action within limits"""
        action = self._create_test_action(OrderSide.BACK, 2.0, 1000.0)
        
        response = self.dgl.evaluate_action(action, self.account_state)
        
        assert response.decision == DGLDecision.APPROVE
        assert response.reason == "All governance rules satisfied"
        assert len(response.rule_ids_triggered) == 0
    
    def test_reject_excessive_exposure(self):
        """Test rejection of action exceeding total exposure limit"""
        # Create action that would exceed 5% of bankroll (£5,000)
        action = self._create_test_action(OrderSide.BACK, 2.0, 3000.0)  # £6,000 exposure
        
        response = self.dgl.evaluate_action(action, self.account_state)
        
        assert response.decision == DGLDecision.REJECT
        assert "Total exposure limit exceeded" in response.reason
        assert "BANKROLL_TOTAL_EXPOSURE" in response.rule_ids_triggered
    
    def test_reject_excessive_market_exposure(self):
        """Test rejection of action exceeding per-market limit"""
        # Set existing market exposure close to limit
        self.dgl.market_exposures["match_odds"] = 1800.0  # Close to 2% limit (£2,000)
        
        # Create action that would exceed market limit
        action = self._create_test_action(OrderSide.BACK, 2.0, 200.0)  # £400 more
        
        response = self.dgl.evaluate_action(action, self.account_state)
        
        assert response.decision == DGLDecision.REJECT
        assert "Market exposure limit exceeded" in response.reason
        assert "BANKROLL_MARKET_EXPOSURE" in response.rule_ids_triggered
    
    def test_amend_excessive_bet_size(self):
        """Test amendment of bet size exceeding per-bet limit"""
        # Create action exceeding 0.5% per-bet limit (£500)
        action = self._create_test_action(OrderSide.BACK, 2.0, 400.0)  # £800 stake
        
        response = self.dgl.evaluate_action(action, self.account_state)
        
        assert response.decision == DGLDecision.AMEND
        assert response.amended_size is not None
        assert response.amended_size < action.size
        assert "Bet size reduced" in response.reason
        assert "BANKROLL_BET_SIZE" in response.rule_ids_triggered
    
    def test_reject_tiny_amended_size(self):
        """Test rejection when amended size would be too small"""
        # Create action with very high stake that can't be reasonably amended
        action = self._create_test_action(OrderSide.BACK, 2.0, 5000.0)  # £10,000 stake
        
        response = self.dgl.evaluate_action(action, self.account_state)
        
        assert response.decision == DGLDecision.REJECT
        assert "Bet size too large even after amendment" in response.reason
        assert "BANKROLL_BET_SIZE" in response.rule_ids_triggered
    
    def test_reject_daily_loss_limit(self):
        """Test rejection due to daily loss limit"""
        # Set daily P&L to significant loss
        daily_loss_limit = self.account_state.total_balance() * 0.05 * 0.4  # 40% of max exposure
        self.dgl.daily_pnl = -daily_loss_limit - 100  # Exceed limit
        
        action = self._create_test_action(OrderSide.BACK, 2.0, 100.0)
        
        response = self.dgl.evaluate_action(action, self.account_state)
        
        assert response.decision == DGLDecision.REJECT
        assert "Daily loss limit exceeded" in response.reason
        assert "PNL_DAILY_LOSS" in response.rule_ids_triggered
    
    def test_reject_consecutive_losses(self):
        """Test rejection due to consecutive losses"""
        # Set consecutive losses to limit
        self.dgl.consecutive_losses = 5
        
        action = self._create_test_action(OrderSide.BACK, 2.0, 100.0)
        
        response = self.dgl.evaluate_action(action, self.account_state)
        
        assert response.decision == DGLDecision.REJECT
        assert "Consecutive loss limit exceeded" in response.reason
        assert "PNL_CONSECUTIVE_LOSS" in response.rule_ids_triggered
    
    def test_reject_too_many_markets(self):
        """Test rejection due to too many concurrent markets"""
        # Set up many market exposures
        for i in range(11):
            self.dgl.market_exposures[f"market_{i}"] = 100.0
        
        action = self._create_test_action(OrderSide.BACK, 2.0, 100.0)
        action.market_id = "new_market"  # Would be 12th market
        
        response = self.dgl.evaluate_action(action, self.account_state)
        
        assert response.decision == DGLDecision.REJECT
        assert "Too many concurrent markets" in response.reason
        assert "CONCENTRATION_MARKET_COUNT" in response.rule_ids_triggered
    
    def test_reject_minimum_bet_size(self):
        """Test rejection of bet below minimum size"""
        action = self._create_test_action(OrderSide.BACK, 2.0, 4.0)  # £8 bet (below £10 minimum)
        
        response = self.dgl.evaluate_action(action, self.account_state)
        
        assert response.decision == DGLDecision.REJECT
        assert "Bet size too small" in response.reason
        assert "POSITION_MIN_SIZE" in response.rule_ids_triggered
    
    def test_reject_invalid_odds(self):
        """Test rejection of invalid odds"""
        # Test odds too low
        action_low = self._create_test_action(OrderSide.BACK, 1.005, 100.0)
        response_low = self.dgl.evaluate_action(action_low, self.account_state)
        
        assert response_low.decision == DGLDecision.REJECT
        assert "Odds outside acceptable range" in response_low.reason
        assert "POSITION_ODDS_RANGE" in response_low.rule_ids_triggered
        
        # Test odds too high
        action_high = self._create_test_action(OrderSide.BACK, 1500.0, 100.0)
        response_high = self.dgl.evaluate_action(action_high, self.account_state)
        
        assert response_high.decision == DGLDecision.REJECT
        assert "Odds outside acceptable range" in response_high.reason
        assert "POSITION_ODDS_RANGE" in response_high.rule_ids_triggered
    
    def test_lay_bet_exposure_calculation(self):
        """Test exposure calculation for lay bets"""
        # Lay bet exposure is liability (size * (odds - 1))
        action = self._create_test_action(OrderSide.LAY, 3.0, 1000.0)  # £2,000 liability
        
        response = self.dgl.evaluate_action(action, self.account_state)
        
        # Should be approved as liability is within limits
        assert response.decision == DGLDecision.APPROVE
    
    def test_exposure_tracking_update(self):
        """Test exposure tracking after fills"""
        action = self._create_test_action(OrderSide.BACK, 2.0, 1000.0)
        
        # Simulate partial fill
        self.dgl.update_exposures(action, 500.0)  # 50% fill
        
        assert self.dgl.market_exposures["match_odds"] == 1000.0  # 500 * 2.0
        assert self.dgl.total_exposure == 1000.0
    
    def test_pnl_tracking_update(self):
        """Test P&L tracking updates"""
        # Winning trade
        self.dgl.update_pnl(100.0)
        assert self.dgl.daily_pnl == 100.0
        assert self.dgl.consecutive_losses == 0
        
        # Losing trade
        self.dgl.update_pnl(-50.0)
        assert self.dgl.daily_pnl == 50.0
        assert self.dgl.consecutive_losses == 1
        
        # Another losing trade
        self.dgl.update_pnl(-25.0)
        assert self.dgl.daily_pnl == 25.0
        assert self.dgl.consecutive_losses == 2
    
    def test_audit_logging(self):
        """Test audit log functionality"""
        action = self._create_test_action(OrderSide.BACK, 2.0, 1000.0)
        
        response = self.dgl.evaluate_action(action, self.account_state)
        
        # Should have logged the decision
        assert len(self.dgl.audit_log) == 1
        
        audit_entry = self.dgl.audit_log[0]
        assert "timestamp" in audit_entry
        assert "action" in audit_entry
        assert "decision" in audit_entry
        assert audit_entry["decision"]["decision"] == response.decision.value
    
    def test_violations_tracking(self):
        """Test violations tracking"""
        # Create action that violates multiple rules
        self.dgl.consecutive_losses = 5
        action = self._create_test_action(OrderSide.BACK, 2.0, 3000.0)  # Large bet + consecutive losses
        
        response = self.dgl.evaluate_action(action, self.account_state)
        
        violations = self.dgl.get_violations_summary()
        assert len(violations) > 0
        
        # Should track violation counts
        total_violations = sum(violations.values())
        assert total_violations > 0
    
    def test_current_exposures_summary(self):
        """Test current exposures summary"""
        self.dgl.market_exposures["match_odds"] = 1000.0
        self.dgl.total_exposure = 1000.0
        self.dgl.daily_pnl = 50.0
        
        exposures = self.dgl.get_current_exposures()
        
        assert exposures["total_exposure"] == 1000.0
        assert exposures["market_exposures"]["match_odds"] == 1000.0
        assert exposures["daily_pnl"] == 50.0
        assert "violation_count" in exposures
    
    def test_reset_functionality(self):
        """Test DGL reset functionality"""
        # Set up some state
        self.dgl.market_exposures["match_odds"] = 1000.0
        self.dgl.total_exposure = 1000.0
        self.dgl.daily_pnl = 100.0
        self.dgl.consecutive_losses = 2
        
        action = self._create_test_action(OrderSide.BACK, 2.0, 1000.0)
        self.dgl.evaluate_action(action, self.account_state)
        
        # Reset
        self.dgl.reset()
        
        # Should be clean state
        assert len(self.dgl.market_exposures) == 0
        assert self.dgl.total_exposure == 0.0
        assert self.dgl.daily_pnl == 0.0
        assert self.dgl.consecutive_losses == 0
        assert len(self.dgl.audit_log) == 0
        assert len(self.dgl.violations) == 0
    
    def test_statistics_generation(self):
        """Test DGL statistics generation"""
        # Generate some decisions
        actions = [
            self._create_test_action(OrderSide.BACK, 2.0, 100.0),  # Should approve
            self._create_test_action(OrderSide.BACK, 2.0, 3000.0),  # Should reject
            self._create_test_action(OrderSide.BACK, 2.0, 400.0),  # Should amend
        ]
        
        for action in actions:
            self.dgl.evaluate_action(action, self.account_state)
        
        stats = self.dgl.get_stats()
        
        assert stats["total_decisions"] == 3
        assert stats["approvals"] >= 0
        assert stats["rejections"] >= 0
        assert stats["amendments"] >= 0
        assert 0.0 <= stats["approval_rate"] <= 1.0
        assert "violations_by_rule" in stats
        assert "current_exposures" in stats


class TestDGLResponse:
    """Test DGL response functionality"""
    
    def test_response_creation(self):
        """Test DGL response creation"""
        response = DGLResponse(
            decision=DGLDecision.APPROVE,
            reason="Test approval",
            rule_ids_triggered=["TEST_RULE"]
        )
        
        assert response.decision == DGLDecision.APPROVE
        assert response.reason == "Test approval"
        assert response.rule_ids_triggered == ["TEST_RULE"]
        assert response.audit_ref.startswith("sim_audit_")
    
    def test_response_serialization(self):
        """Test DGL response serialization"""
        response = DGLResponse(
            decision=DGLDecision.AMEND,
            reason="Size amended",
            amended_size=500.0,
            rule_ids_triggered=["SIZE_RULE"]
        )
        
        response_dict = response.to_dict()
        
        assert response_dict["decision"] == "AMEND"
        assert response_dict["reason"] == "Size amended"
        assert response_dict["amended_size"] == 500.0
        assert response_dict["rule_ids_triggered"] == ["SIZE_RULE"]


if __name__ == "__main__":
    pytest.main([__file__])
