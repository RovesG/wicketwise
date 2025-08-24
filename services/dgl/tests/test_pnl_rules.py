# Purpose: Unit tests for P&L protection rule implementations
# Author: WicketWise AI, Last Modified: 2024

import pytest
from hypothesis import given, strategies as st, assume, settings
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from schemas import BetProposal, BetSide, ExposureSnapshot, RuleId
from config import PnLGuardsConfig
from repo.memory_repo import MemoryPnLStore
from rules.pnl_rules import (
    DailyLossLimitRule, SessionLossLimitRule, PnLTrendAnalysisRule, PnLRuleEngine
)


class TestDailyLossLimitRule:
    """Test suite for DailyLossLimitRule"""
    
    @pytest.fixture
    def config(self):
        """Create test P&L configuration"""
        return PnLGuardsConfig(
            daily_loss_limit_pct=3.0,
            session_loss_limit_pct=2.0
        )
    
    @pytest.fixture
    def pnl_store(self):
        """Create test P&L store"""
        return MemoryPnLStore()
    
    @pytest.fixture
    def rule(self, config, pnl_store):
        """Create daily loss limit rule instance"""
        return DailyLossLimitRule(config, pnl_store)
    
    def test_rule_initialization(self, rule, config):
        """Test rule initialization"""
        assert rule.config == config
        assert rule.rule_id == RuleId.PNL_DAILY_LOSS_LIMIT
    
    def test_no_violation_positive_pnl(self, rule, pnl_store):
        """Test no violation when daily P&L is positive"""
        # Set positive daily P&L
        pnl_store.update_pnl(500.0)
        
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=0.0,
            daily_pnl=500.0,
            session_pnl=500.0
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is None
    
    def test_no_violation_small_loss(self, rule, pnl_store):
        """Test no violation when daily loss is within limits"""
        # Set small daily loss (1% of 100k bankroll = 1000)
        pnl_store.update_pnl(-1000.0)
        
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=0.0,
            daily_pnl=-1000.0,
            session_pnl=-1000.0
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is None
    
    def test_warning_approaching_limit(self, rule, pnl_store):
        """Test warning when approaching daily loss limit"""
        # Set loss at 80% of limit (2.4% of 100k bankroll = 2400)
        pnl_store.update_pnl(-2400.0)
        
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=0.0,
            daily_pnl=-2400.0,
            session_pnl=-2400.0
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is not None
        assert violation.rule_id == RuleId.PNL_DAILY_LOSS_LIMIT
        assert violation.severity == "WARNING"
        assert "Approaching daily loss limit" in violation.message
    
    def test_critical_violation_exceeds_limit(self, rule, pnl_store):
        """Test critical violation when daily loss exceeds limit"""
        # Set loss exceeding limit (4% of 100k bankroll = 4000, limit is 3%)
        pnl_store.update_pnl(-4000.0)
        
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=0.0,
            daily_pnl=-4000.0,
            session_pnl=-4000.0
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is not None
        assert violation.rule_id == RuleId.PNL_DAILY_LOSS_LIMIT
        assert violation.severity == "CRITICAL"
        assert "Daily loss limit exceeded" in violation.message
        assert violation.current_value == 4000.0
        assert violation.threshold == 3000.0
    
    @given(
        bankroll=st.floats(min_value=10000, max_value=1000000),
        loss_pct=st.floats(min_value=0.1, max_value=10.0)
    )
    @settings(max_examples=20, deadline=1000)
    def test_property_loss_limit_calculation(self, rule, pnl_store, bankroll, loss_pct):
        """Property-based test for loss limit calculations"""
        assume(bankroll > 0)
        assume(0.1 <= loss_pct <= 10.0)
        
        # Set loss amount
        loss_amount = bankroll * (loss_pct / 100)
        pnl_store._daily_pnl.clear()  # Clear existing P&L
        pnl_store.update_pnl(-loss_amount)
        
        exposure = ExposureSnapshot(
            bankroll=bankroll,
            open_exposure=0.0,
            daily_pnl=-loss_amount,
            session_pnl=-loss_amount
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=100.0,
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        
        # Check violation based on loss percentage
        if loss_pct > 3.0:  # Exceeds 3% limit
            assert violation is not None
            assert violation.severity == "CRITICAL"
        elif loss_pct > 2.4:  # Approaching limit (80% of 3%)
            assert violation is not None
            assert violation.severity == "WARNING"
        else:
            assert violation is None


class TestSessionLossLimitRule:
    """Test suite for SessionLossLimitRule"""
    
    @pytest.fixture
    def config(self):
        return PnLGuardsConfig(
            daily_loss_limit_pct=3.0,
            session_loss_limit_pct=2.0
        )
    
    @pytest.fixture
    def pnl_store(self):
        return MemoryPnLStore()
    
    @pytest.fixture
    def rule(self, config, pnl_store):
        return SessionLossLimitRule(config, pnl_store)
    
    def test_no_violation_within_session_limit(self, rule, pnl_store):
        """Test no violation when session loss is within limits"""
        # Set session loss within limit (1% of 100k bankroll = 1000)
        pnl_store.update_pnl(-1000.0)
        
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=0.0,
            daily_pnl=-1000.0,
            session_pnl=-1000.0
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is None
    
    def test_violation_exceeds_session_limit(self, rule, pnl_store):
        """Test violation when session loss exceeds limit"""
        # Set session loss exceeding limit (3% of 100k bankroll = 3000, limit is 2%)
        pnl_store.update_pnl(-3000.0)
        
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=0.0,
            daily_pnl=-3000.0,
            session_pnl=-3000.0
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is not None
        assert violation.rule_id == RuleId.PNL_SESSION_LOSS_LIMIT
        assert violation.severity == "CRITICAL"
        assert "Session loss limit exceeded" in violation.message


class TestPnLTrendAnalysisRule:
    """Test suite for PnLTrendAnalysisRule"""
    
    @pytest.fixture
    def config(self):
        return PnLGuardsConfig(
            daily_loss_limit_pct=3.0,
            session_loss_limit_pct=2.0
        )
    
    @pytest.fixture
    def pnl_store(self):
        return MemoryPnLStore()
    
    @pytest.fixture
    def rule(self, config, pnl_store):
        return PnLTrendAnalysisRule(config, pnl_store)
    
    def test_no_violation_insufficient_data(self, rule, pnl_store):
        """Test no violation when insufficient data for trend analysis"""
        # Only add 1 day of data (need at least 3)
        pnl_store._daily_pnl["2025-01-01"] = -100.0
        
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=0.0,
            daily_pnl=-100.0,
            session_pnl=-100.0
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is None
    
    def test_warning_negative_trend(self, rule, pnl_store):
        """Test warning when negative trend is detected"""
        # Add 7 days of data with negative trend in last 3 days
        pnl_store._daily_pnl["2025-01-01"] = 100.0   # Day 1: positive
        pnl_store._daily_pnl["2025-01-02"] = 200.0   # Day 2: positive
        pnl_store._daily_pnl["2025-01-03"] = 50.0    # Day 3: positive
        pnl_store._daily_pnl["2025-01-04"] = -50.0   # Day 4: positive
        pnl_store._daily_pnl["2025-01-05"] = -200.0  # Day 5: negative (last 3 days start)
        pnl_store._daily_pnl["2025-01-06"] = -150.0  # Day 6: negative
        pnl_store._daily_pnl["2025-01-07"] = -100.0  # Day 7: negative
        
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=0.0,
            daily_pnl=-100.0,
            session_pnl=-100.0
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is not None
        assert violation.severity == "WARNING"
        assert "Negative P&L trend detected" in violation.message


class TestPnLRuleEngine:
    """Test suite for PnLRuleEngine"""
    
    @pytest.fixture
    def config(self):
        return PnLGuardsConfig(
            daily_loss_limit_pct=3.0,
            session_loss_limit_pct=2.0
        )
    
    @pytest.fixture
    def pnl_store(self):
        return MemoryPnLStore()
    
    @pytest.fixture
    def engine(self, config, pnl_store):
        return PnLRuleEngine(config, pnl_store)
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert len(engine.rules) == 3  # Daily, Session, Trend rules
        assert engine.config is not None
        assert engine.pnl_store is not None
    
    def test_evaluate_all_no_violations(self, engine, pnl_store):
        """Test evaluating all rules with no violations"""
        # Set positive P&L
        pnl_store.update_pnl(500.0)
        
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=0.0,
            daily_pnl=500.0,
            session_pnl=500.0
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violations = engine.evaluate_all(proposal, exposure)
        assert len(violations) == 0
    
    def test_evaluate_all_multiple_violations(self, engine, pnl_store):
        """Test evaluating all rules with violations"""
        # Set losses exceeding both daily and session limits
        pnl_store.update_pnl(-4000.0)  # 4% loss (exceeds both 3% daily and 2% session)
        
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=0.0,
            daily_pnl=-4000.0,
            session_pnl=-4000.0
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violations = engine.evaluate_all(proposal, exposure)
        assert len(violations) >= 2  # Should have both daily and session violations
        
        # Check that we have the expected violations
        rule_ids = [v.rule_id for v in violations]
        assert RuleId.PNL_DAILY_LOSS_LIMIT in rule_ids
        assert RuleId.PNL_SESSION_LOSS_LIMIT in rule_ids
    
    def test_get_pnl_summary(self, engine, pnl_store):
        """Test getting P&L summary"""
        # Add some P&L data
        pnl_store.update_pnl(-1000.0)
        
        summary = engine.get_pnl_summary()
        
        assert "daily_pnl" in summary
        assert "session_pnl" in summary
        assert "pnl_history_7d" in summary
        assert "daily_loss_limit_pct" in summary
        assert "session_loss_limit_pct" in summary
        assert summary["daily_pnl"] == -1000.0
        assert summary["session_pnl"] == -1000.0
    
    def test_reset_session(self, engine, pnl_store):
        """Test session reset functionality"""
        # Add some session P&L
        pnl_store.update_pnl(-1000.0)
        assert pnl_store.get_session_pnl() == -1000.0
        
        # Reset session
        success = engine.reset_session()
        assert success is True
        assert pnl_store.get_session_pnl() == 0.0
    
    def test_get_statistics(self, engine):
        """Test getting engine statistics"""
        stats = engine.get_statistics()
        
        assert "total_rules" in stats
        assert "rule_ids" in stats
        assert "config" in stats
        assert "current_pnl" in stats
        assert stats["total_rules"] == 3


def run_pnl_rules_tests():
    """Run all P&L rules tests"""
    print("📊 Running P&L Rules Tests")
    print("=" * 50)
    
    test_classes = [
        ("Daily Loss Limit Rule", TestDailyLossLimitRule),
        ("Session Loss Limit Rule", TestSessionLossLimitRule),
        ("P&L Trend Analysis Rule", TestPnLTrendAnalysisRule),
        ("P&L Rule Engine", TestPnLRuleEngine)
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for class_name, test_class in test_classes:
        print(f"\n📈 {class_name}")
        print("-" * 40)
        
        # Get test methods
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_')]
        
        class_passed = 0
        for test_method in test_methods:
            total_tests += 1
            try:
                # Create test instance
                test_instance = test_class()
                
                # Handle fixtures and run method
                method = getattr(test_instance, test_method)
                
                # Create fixtures
                config = PnLGuardsConfig(
                    daily_loss_limit_pct=3.0,
                    session_loss_limit_pct=2.0
                )
                pnl_store = MemoryPnLStore()
                
                if 'Daily' in class_name:
                    from rules.pnl_rules import DailyLossLimitRule
                    rule = DailyLossLimitRule(config, pnl_store)
                elif 'Session' in class_name:
                    from rules.pnl_rules import SessionLossLimitRule
                    rule = SessionLossLimitRule(config, pnl_store)
                elif 'Trend' in class_name:
                    from rules.pnl_rules import PnLTrendAnalysisRule
                    rule = PnLTrendAnalysisRule(config, pnl_store)
                elif 'Engine' in class_name:
                    from rules.pnl_rules import PnLRuleEngine
                    rule = PnLRuleEngine(config, pnl_store)
                
                # Run method with appropriate parameters
                import inspect
                sig = inspect.signature(method)
                params = list(sig.parameters.keys())
                
                if 'engine' in params and 'pnl_store' in params:
                    method(rule, pnl_store)
                elif 'rule' in params and 'pnl_store' in params:
                    method(rule, pnl_store)
                elif 'rule' in params and 'config' in params:
                    method(rule, config)
                elif 'rule' in params:
                    method(rule)
                elif 'engine' in params:
                    method(rule)  # rule is actually engine for engine tests
                else:
                    method()
                
                print(f"  ✅ {test_method}")
                passed_tests += 1
                class_passed += 1
                
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
        
        print(f"  📈 Class Results: {class_passed}/{len(test_methods)} passed")
    
    print(f"\n🏆 Overall P&L Rules Test Results: {passed_tests}/{total_tests} passed")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_pnl_rules_tests()
    exit(0 if success else 1)
