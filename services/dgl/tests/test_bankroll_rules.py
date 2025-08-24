# Purpose: Unit tests for bankroll rule implementations with property-based testing
# Author: WicketWise AI, Last Modified: 2024

import pytest
from hypothesis import given, strategies as st, assume, settings
from decimal import Decimal
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from schemas import BetProposal, BetSide, ExposureSnapshot, RuleId
from config import BankrollConfig
from rules.bankroll_rules import (
    BankrollExposureRule, PerMatchExposureRule, PerMarketExposureRule,
    PerBetExposureRule, BankrollRuleEngine
)


class TestBankrollExposureRule:
    """Test suite for BankrollExposureRule"""
    
    @pytest.fixture
    def config(self):
        """Create test bankroll configuration"""
        return BankrollConfig(
            max_bankroll_exposure_pct=5.0,
            per_match_max_pct=2.0,
            per_market_max_pct=1.0,
            per_bet_max_pct=0.5
        )
    
    @pytest.fixture
    def rule(self, config):
        """Create bankroll exposure rule instance"""
        return BankrollExposureRule(config)
    
    def test_rule_initialization(self, rule, config):
        """Test rule initialization"""
        assert rule.config == config
        assert rule.rule_id == RuleId.BANKROLL_MAX_EXPOSURE
    
    def test_no_violation_within_limits(self, rule):
        """Test no violation when bet is within limits"""
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=2000.0,  # 2% of bankroll
            daily_pnl=0.0,
            session_pnl=0.0
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=1000.0,  # Would result in 3% total exposure (within 5% limit)
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is None
    
    def test_violation_exceeds_limits(self, rule):
        """Test violation when bet exceeds bankroll limits"""
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=4000.0,  # 4% of bankroll
            daily_pnl=0.0,
            session_pnl=0.0
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=2000.0,  # Would result in 6% total exposure (exceeds 5% limit)
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is not None
        assert violation.rule_id == RuleId.BANKROLL_MAX_EXPOSURE
        assert "exceed maximum bankroll exposure" in violation.message
        assert violation.suggested_amendment is not None
        assert violation.suggested_amendment["stake"] == 1000.0  # Available exposure
    
    def test_lay_bet_exposure_calculation(self, rule):
        """Test exposure calculation for lay bets"""
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=0.0,
            daily_pnl=0.0,
            session_pnl=0.0
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.LAY,
            selection="Team A",
            odds=3.0,
            stake=1000.0,  # Exposure = (3.0 - 1) * 1000 = 2000
            model_confidence=0.8,
            fair_odds=2.9,
            expected_edge_pct=3.0
        )
        
        # Should be within limits (2% of 100k bankroll)
        violation = rule.evaluate(proposal, exposure)
        assert violation is None
    
    @given(
        bankroll=st.floats(min_value=10000, max_value=1000000),
        current_exposure_pct=st.floats(min_value=0, max_value=4.9),
        stake=st.floats(min_value=100, max_value=10000),
        odds=st.floats(min_value=1.1, max_value=10.0)
    )
    @settings(max_examples=50, deadline=1000)
    def test_property_exposure_calculation(self, rule, bankroll, current_exposure_pct, stake, odds):
        """Property-based test for exposure calculations"""
        assume(bankroll > 0)
        assume(0 <= current_exposure_pct < 5.0)
        assume(stake > 0)
        assume(odds > 1.0)
        
        current_exposure = bankroll * (current_exposure_pct / 100)
        
        exposure = ExposureSnapshot(
            bankroll=bankroll,
            open_exposure=current_exposure,
            daily_pnl=0.0,
            session_pnl=0.0
        )
        
        # Test BACK bet
        proposal_back = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=odds,
            stake=stake,
            model_confidence=0.8,
            fair_odds=odds * 0.95,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal_back, exposure)
        
        # Calculate expected total exposure
        bet_exposure = stake  # For BACK bets
        total_exposure = current_exposure + bet_exposure
        max_allowed = bankroll * 0.05  # 5% limit
        
        if total_exposure > max_allowed:
            assert violation is not None
            assert violation.rule_id == RuleId.BANKROLL_MAX_EXPOSURE
        else:
            assert violation is None


class TestPerMatchExposureRule:
    """Test suite for PerMatchExposureRule"""
    
    @pytest.fixture
    def config(self):
        return BankrollConfig(per_match_max_pct=2.0)
    
    @pytest.fixture
    def rule(self, config):
        return PerMatchExposureRule(config)
    
    def test_no_violation_new_match(self, rule):
        """Test no violation for new match"""
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=0.0,
            daily_pnl=0.0,
            session_pnl=0.0,
            per_match_exposure={}  # No existing match exposure
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="new:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=1500.0,  # 1.5% of bankroll (within 2% limit)
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is None
    
    def test_violation_existing_match(self, rule):
        """Test violation when adding to existing match exposure"""
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=1500.0,
            daily_pnl=0.0,
            session_pnl=0.0,
            per_match_exposure={"existing:match:1": 1500.0}  # 1.5% already exposed
        )
        
        proposal = BetProposal(
            market_id="test:market:2",
            match_id="existing:match:1",
            side=BetSide.BACK,
            selection="Team B",
            odds=2.0,
            stake=800.0,  # Would make total 2.3% (exceeds 2% limit)
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is not None
        assert violation.rule_id == RuleId.EXPO_PER_MATCH_MAX
        assert "exceed maximum match exposure" in violation.message
        assert violation.suggested_amendment["stake"] == 500.0  # Available exposure


class TestPerMarketExposureRule:
    """Test suite for PerMarketExposureRule"""
    
    @pytest.fixture
    def config(self):
        return BankrollConfig(per_market_max_pct=1.0)
    
    @pytest.fixture
    def rule(self, config):
        return PerMarketExposureRule(config)
    
    def test_no_violation_within_market_limit(self, rule):
        """Test no violation when within market limit"""
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=500.0,
            daily_pnl=0.0,
            session_pnl=0.0,
            per_market_exposure={"test:market:1": 500.0}
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=400.0,  # Would make total 0.9% (within 1% limit)
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is None
    
    def test_violation_exceeds_market_limit(self, rule):
        """Test violation when exceeding market limit"""
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=800.0,
            daily_pnl=0.0,
            session_pnl=0.0,
            per_market_exposure={"test:market:1": 800.0}
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=400.0,  # Would make total 1.2% (exceeds 1% limit)
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is not None
        assert violation.rule_id == RuleId.EXPO_PER_MARKET_MAX
        assert "exceed maximum market exposure" in violation.message


class TestPerBetExposureRule:
    """Test suite for PerBetExposureRule"""
    
    @pytest.fixture
    def config(self):
        return BankrollConfig(per_bet_max_pct=0.5)
    
    @pytest.fixture
    def rule(self, config):
        return PerBetExposureRule(config)
    
    def test_no_violation_within_bet_limit(self, rule):
        """Test no violation when bet is within limit"""
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=0.0,
            daily_pnl=0.0,
            session_pnl=0.0
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=400.0,  # 0.4% of bankroll (within 0.5% limit)
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is None
    
    def test_violation_exceeds_bet_limit(self, rule):
        """Test violation when bet exceeds limit"""
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=0.0,
            daily_pnl=0.0,
            session_pnl=0.0
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=800.0,  # 0.8% of bankroll (exceeds 0.5% limit)
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is not None
        assert violation.rule_id == RuleId.EXPO_PER_BET_MAX
        assert "exceeds maximum per-bet limit" in violation.message
        assert violation.suggested_amendment["stake"] == 500.0  # Max allowed


class TestBankrollRuleEngine:
    """Test suite for BankrollRuleEngine"""
    
    @pytest.fixture
    def config(self):
        return BankrollConfig(
            max_bankroll_exposure_pct=5.0,
            per_match_max_pct=2.0,
            per_market_max_pct=1.0,
            per_bet_max_pct=0.5
        )
    
    @pytest.fixture
    def engine(self, config):
        return BankrollRuleEngine(config)
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert len(engine.rules) == 4
        assert engine.config is not None
    
    def test_evaluate_all_no_violations(self, engine):
        """Test evaluating all rules with no violations"""
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=1000.0,
            daily_pnl=0.0,
            session_pnl=0.0,
            per_match_exposure={"match:1": 500.0},
            per_market_exposure={"market:1": 300.0}
        )
        
        proposal = BetProposal(
            market_id="market:2",
            match_id="match:2",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=400.0,  # All within limits
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violations = engine.evaluate_all(proposal, exposure)
        assert len(violations) == 0
    
    def test_evaluate_all_multiple_violations(self, engine):
        """Test evaluating all rules with multiple violations"""
        exposure = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=4500.0,  # 4.5% already exposed
            daily_pnl=0.0,
            session_pnl=0.0,
            per_match_exposure={"match:1": 1800.0},  # 1.8% for this match
            per_market_exposure={"market:1": 900.0}  # 0.9% for this market
        )
        
        proposal = BetProposal(
            market_id="market:1",  # Same market with existing exposure
            match_id="match:1",    # Same match with existing exposure
            side=BetSide.BACK,
            selection="Team A",
            odds=2.0,
            stake=800.0,  # Would violate multiple rules
            model_confidence=0.8,
            fair_odds=1.9,
            expected_edge_pct=5.0
        )
        
        violations = engine.evaluate_all(proposal, exposure)
        assert len(violations) >= 2  # Should have multiple violations
        
        # Check that we have the expected rule violations
        rule_ids = [v.rule_id for v in violations]
        assert RuleId.BANKROLL_MAX_EXPOSURE in rule_ids  # Total exposure would be 5.3%
        assert RuleId.EXPO_PER_BET_MAX in rule_ids       # Single bet is 0.8%
    
    def test_get_statistics(self, engine):
        """Test getting engine statistics"""
        stats = engine.get_statistics()
        
        assert "total_rules" in stats
        assert "rule_ids" in stats
        assert "config" in stats
        assert stats["total_rules"] == 4
        assert len(stats["rule_ids"]) == 4


def run_bankroll_rules_tests():
    """Run all bankroll rules tests"""
    print("💰 Running Bankroll Rules Tests")
    print("=" * 50)
    
    test_classes = [
        ("Bankroll Exposure Rule", TestBankrollExposureRule),
        ("Per-Match Exposure Rule", TestPerMatchExposureRule),
        ("Per-Market Exposure Rule", TestPerMarketExposureRule),
        ("Per-Bet Exposure Rule", TestPerBetExposureRule),
        ("Bankroll Rule Engine", TestBankrollRuleEngine)
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for class_name, test_class in test_classes:
        print(f"\n📊 {class_name}")
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
                
                # Create fixtures if needed
                if hasattr(test_instance, 'config'):
                    config = BankrollConfig(
                        max_bankroll_exposure_pct=5.0,
                        per_match_max_pct=2.0,
                        per_market_max_pct=1.0,
                        per_bet_max_pct=0.5
                    )
                    
                    if 'rule' in test_method:
                        if 'BankrollExposure' in class_name:
                            from rules.bankroll_rules import BankrollExposureRule
                            rule = BankrollExposureRule(config)
                        elif 'PerMatch' in class_name:
                            from rules.bankroll_rules import PerMatchExposureRule
                            rule = PerMatchExposureRule(config)
                        elif 'PerMarket' in class_name:
                            from rules.bankroll_rules import PerMarketExposureRule
                            rule = PerMarketExposureRule(config)
                        elif 'PerBet' in class_name:
                            from rules.bankroll_rules import PerBetExposureRule
                            rule = PerBetExposureRule(config)
                        elif 'Engine' in class_name:
                            from rules.bankroll_rules import BankrollRuleEngine
                            rule = BankrollRuleEngine(config)
                        
                        # Run method with fixtures
                        import inspect
                        sig = inspect.signature(method)
                        if 'rule' in sig.parameters and 'config' in sig.parameters:
                            method(rule, config)
                        elif 'rule' in sig.parameters:
                            method(rule)
                        elif 'engine' in sig.parameters:
                            method(rule)  # rule is actually engine for engine tests
                        else:
                            method()
                    else:
                        method()
                else:
                    method()
                
                print(f"  ✅ {test_method}")
                passed_tests += 1
                class_passed += 1
                
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
        
        print(f"  📈 Class Results: {class_passed}/{len(test_methods)} passed")
    
    print(f"\n🏆 Overall Bankroll Rules Test Results: {passed_tests}/{total_tests} passed")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_bankroll_rules_tests()
    exit(0 if success else 1)
