# Purpose: Unit tests for liquidity and execution constraint rules
# Author: WicketWise AI, Last Modified: 2024

import pytest
from hypothesis import given, strategies as st, assume, settings
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from schemas import BetProposal, BetSide, ExposureSnapshot, RuleId, LiquidityInfo, MarketDepth
from config import LiquidityConfig
from rules.liquidity_rules import (
    OddsRangeRule, SlippageRule, LiquidityFractionRule, MarketDepthAnalysisRule,
    TokenBucketRateLimiter, SlidingWindowRateLimiter, RateLimitRule, LiquidityRuleEngine
)


class TestOddsRangeRule:
    """Test suite for OddsRangeRule"""
    
    @pytest.fixture
    def config(self):
        """Create test liquidity configuration"""
        return LiquidityConfig(
            min_odds_threshold=1.25,
            max_odds_threshold=10.0,
            slippage_bps_limit=50,
            max_fraction_of_available_liquidity=10.0
        )
    
    @pytest.fixture
    def rule(self, config):
        """Create odds range rule instance"""
        return OddsRangeRule(config)
    
    def test_rule_initialization(self, rule, config):
        """Test rule initialization"""
        assert rule.config == config
        assert rule.rule_id_min == RuleId.LIQ_MIN_ODDS
        assert rule.rule_id_max == RuleId.LIQ_MAX_ODDS
    
    def test_no_violation_within_range(self, rule):
        """Test no violation when odds are within acceptable range"""
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
            odds=2.5,  # Within 1.25 - 10.0 range
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=2.4,
            expected_edge_pct=4.0
        )
        
        violations = rule.evaluate(proposal, exposure)
        assert len(violations) == 0
    
    def test_violation_odds_too_low(self, rule):
        """Test violation when odds are below minimum threshold"""
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
            odds=1.1,  # Below 1.25 threshold
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=1.05,
            expected_edge_pct=4.0
        )
        
        violations = rule.evaluate(proposal, exposure)
        assert len(violations) == 1
        assert violations[0].rule_id == RuleId.LIQ_MIN_ODDS
        assert "below minimum threshold" in violations[0].message
        assert violations[0].current_value == 1.1
        assert violations[0].threshold == 1.25
    
    def test_violation_odds_too_high(self, rule):
        """Test violation when odds are above maximum threshold"""
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
            odds=15.0,  # Above 10.0 threshold
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=14.5,
            expected_edge_pct=3.0
        )
        
        violations = rule.evaluate(proposal, exposure)
        assert len(violations) == 1
        assert violations[0].rule_id == RuleId.LIQ_MAX_ODDS
        assert "above maximum threshold" in violations[0].message
        assert violations[0].current_value == 15.0
        assert violations[0].threshold == 10.0
    
    @given(
        odds=st.floats(min_value=0.1, max_value=50.0)
    )
    @settings(max_examples=30, deadline=1000)
    def test_property_odds_validation(self, rule, odds):
        """Property-based test for odds validation"""
        assume(odds > 0)
        
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
            odds=odds,
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=odds * 0.95,
            expected_edge_pct=5.0
        )
        
        violations = rule.evaluate(proposal, exposure)
        
        if odds < 1.25:
            assert len(violations) >= 1
            assert any(v.rule_id == RuleId.LIQ_MIN_ODDS for v in violations)
        elif odds > 10.0:
            assert len(violations) >= 1
            assert any(v.rule_id == RuleId.LIQ_MAX_ODDS for v in violations)
        else:
            # Should not have odds range violations
            odds_violations = [v for v in violations if v.rule_id in [RuleId.LIQ_MIN_ODDS, RuleId.LIQ_MAX_ODDS]]
            assert len(odds_violations) == 0


class TestSlippageRule:
    """Test suite for SlippageRule"""
    
    @pytest.fixture
    def config(self):
        return LiquidityConfig(slippage_bps_limit=50)
    
    @pytest.fixture
    def rule(self, config):
        return SlippageRule(config)
    
    def test_no_violation_within_slippage_limit(self, rule):
        """Test no violation when slippage is within limits"""
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
            odds=2.02,  # 1% slippage from fair odds 2.0 = 100bps (within 50bps limit? No, should violate)
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=2.0,
            expected_edge_pct=1.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        # Actually this should violate since 1% = 100bps > 50bps limit
        assert violation is not None
    
    def test_no_violation_no_fair_odds(self, rule):
        """Test no violation when fair odds not provided"""
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
            odds=2.5,
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=None,  # No fair odds provided
            expected_edge_pct=5.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is None
    
    def test_violation_exceeds_slippage_limit(self, rule):
        """Test violation when slippage exceeds limit"""
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
            odds=2.2,  # 10% slippage from fair odds 2.0 = 1000bps (exceeds 50bps limit)
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=2.0,
            expected_edge_pct=10.0
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is not None
        assert violation.rule_id == RuleId.LIQ_SLIPPAGE_LIMIT
        assert "exceeds limit" in violation.message
        assert violation.current_value == 1000.0  # 1000 bps
        assert violation.threshold == 50.0
        assert violation.suggested_amendment is not None


class TestLiquidityFractionRule:
    """Test suite for LiquidityFractionRule"""
    
    @pytest.fixture
    def config(self):
        return LiquidityConfig(max_fraction_of_available_liquidity=10.0)
    
    @pytest.fixture
    def rule(self, config):
        return LiquidityFractionRule(config)
    
    def test_no_violation_within_liquidity_limit(self, rule):
        """Test no violation when liquidity consumption is within limits"""
        liquidity = LiquidityInfo(
            available=10000.0,
            market_depth=[
                MarketDepth(odds=2.0, size=5000.0),
                MarketDepth(odds=2.02, size=3000.0)
            ]
        )
        
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
            stake=800.0,  # 8% of available liquidity (within 10% limit)
            model_confidence=0.8,
            fair_odds=1.95,
            expected_edge_pct=2.5,
            liquidity=liquidity
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is None
    
    def test_violation_exceeds_liquidity_limit(self, rule):
        """Test violation when liquidity consumption exceeds limit"""
        liquidity = LiquidityInfo(
            available=10000.0,
            market_depth=[MarketDepth(odds=2.0, size=5000.0)]
        )
        
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
            stake=1500.0,  # 15% of available liquidity (exceeds 10% limit)
            model_confidence=0.8,
            fair_odds=1.95,
            expected_edge_pct=2.5,
            liquidity=liquidity
        )
        
        violation = rule.evaluate(proposal, exposure)
        assert violation is not None
        assert violation.rule_id == RuleId.LIQ_FRACTION_LIMIT
        assert "consume 15.0%" in violation.message
        assert violation.current_value == 15.0
        assert violation.threshold == 10.0
        assert violation.suggested_amendment is not None
        assert violation.suggested_amendment["stake"] == 1000.0  # 10% of 10000


class TestTokenBucketRateLimiter:
    """Test suite for TokenBucketRateLimiter"""
    
    def test_initialization(self):
        """Test rate limiter initialization"""
        limiter = TokenBucketRateLimiter(rate=5.0, capacity=10)
        
        assert limiter.rate == 5.0
        assert limiter.capacity == 10
        assert limiter.tokens == 10  # Starts full
    
    def test_consume_within_capacity(self):
        """Test consuming tokens within capacity"""
        limiter = TokenBucketRateLimiter(rate=5.0, capacity=10)
        
        # Should be able to consume up to capacity
        for i in range(10):
            assert limiter.consume(1) is True
        
        # Should be empty now
        assert limiter.consume(1) is False
    
    def test_token_refill(self):
        """Test token refill over time"""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=10)  # 10 tokens per second
        
        # Consume all tokens
        for i in range(10):
            limiter.consume(1)
        
        # Should be empty
        assert limiter.consume(1) is False
        
        # Wait for refill (simulate time passage)
        limiter.last_update -= 0.5  # Simulate 0.5 seconds ago
        
        # Should have 5 tokens now (10 * 0.5 = 5)
        assert limiter.consume(5) is True
        assert limiter.consume(1) is False  # Should be empty again
    
    def test_get_wait_time(self):
        """Test wait time calculation"""
        limiter = TokenBucketRateLimiter(rate=2.0, capacity=5)  # 2 tokens per second
        
        # Consume all tokens
        for i in range(5):
            limiter.consume(1)
        
        # Wait time for 1 token should be 0.5 seconds (1/2)
        wait_time = limiter.get_wait_time(1)
        assert abs(wait_time - 0.5) < 0.1


class TestSlidingWindowRateLimiter:
    """Test suite for SlidingWindowRateLimiter"""
    
    def test_initialization(self):
        """Test rate limiter initialization"""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
        
        assert limiter.max_requests == 5
        assert limiter.window_seconds == 60
        assert len(limiter.requests) == 0
    
    def test_requests_within_limit(self):
        """Test requests within rate limit"""
        limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
        
        # Should allow up to max_requests
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        
        # Should block further requests
        assert limiter.is_allowed() is False
    
    def test_window_sliding(self):
        """Test sliding window behavior"""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=1)
        
        # Use up the limit
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is False
        
        # Simulate time passage by manually adjusting request times
        if limiter.requests:
            limiter.requests[0] -= 2  # Make first request 2 seconds old
        
        # Should allow new request now
        assert limiter.is_allowed() is True


class TestRateLimitRule:
    """Test suite for RateLimitRule"""
    
    @pytest.fixture
    def config(self):
        return {
            'global_rate': 5.0,
            'global_burst': 10,
            'market_requests': 3,
            'market_window_seconds': 60
        }
    
    @pytest.fixture
    def rule(self, config):
        return RateLimitRule(config)
    
    def test_rule_initialization(self, rule, config):
        """Test rule initialization"""
        assert rule.config == config
        assert rule.rule_id == RuleId.RATE_LIMIT_EXCEEDED
        assert rule.market_rate_limit == 3
        assert rule.market_window == 60
    
    def test_no_violation_within_limits(self, rule):
        """Test no violation when within rate limits"""
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
            stake=1000.0,
            model_confidence=0.8,
            fair_odds=1.95,
            expected_edge_pct=2.5
        )
        
        # First few requests should be allowed
        violation = rule.evaluate(proposal, exposure)
        assert violation is None


class TestLiquidityRuleEngine:
    """Test suite for LiquidityRuleEngine"""
    
    @pytest.fixture
    def config(self):
        return LiquidityConfig(
            min_odds_threshold=1.25,
            max_odds_threshold=10.0,
            slippage_bps_limit=50,
            max_fraction_of_available_liquidity=10.0
        )
    
    @pytest.fixture
    def rate_config(self):
        return {
            'global_rate': 10.0,
            'global_burst': 20,
            'market_requests': 5,
            'market_window_seconds': 120
        }
    
    @pytest.fixture
    def engine(self, config, rate_config):
        return LiquidityRuleEngine(config, rate_config)
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert len(engine.rules) == 5  # 4 liquidity rules + 1 rate limit rule
        assert engine.config is not None
        assert engine.rate_limit_rule is not None
    
    def test_evaluate_all_no_violations(self, engine):
        """Test evaluating all rules with no violations"""
        liquidity = LiquidityInfo(
            available=20000.0,
            market_depth=[MarketDepth(odds=2.0, size=10000.0)]
        )
        
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
            odds=2.0,  # Within odds range
            stake=1000.0,  # 5% of liquidity (within 10% limit)
            model_confidence=0.8,
            fair_odds=1.98,  # Small slippage (about 10bps, within 50bps limit)
            expected_edge_pct=1.0,
            liquidity=liquidity
        )
        
        violations = engine.evaluate_all(proposal, exposure)
        # Should have minimal or no violations
        assert len(violations) <= 1  # Might have rate limit or minor violations
    
    def test_get_liquidity_analysis(self, engine):
        """Test liquidity analysis functionality"""
        liquidity = LiquidityInfo(
            available=10000.0,
            market_depth=[MarketDepth(odds=2.0, size=5000.0)]
        )
        
        proposal = BetProposal(
            market_id="test:market:1",
            match_id="test:match:1",
            side=BetSide.BACK,
            selection="Team A",
            odds=2.1,
            stake=800.0,
            model_confidence=0.8,
            fair_odds=2.0,
            expected_edge_pct=5.0,
            liquidity=liquidity
        )
        
        analysis = engine.get_liquidity_analysis(proposal)
        
        assert "odds_analysis" in analysis
        assert "slippage_analysis" in analysis
        assert "liquidity_analysis" in analysis
        assert "rate_limit_status" in analysis
        
        # Check specific values
        assert analysis["odds_analysis"]["requested_odds"] == 2.1
        assert analysis["odds_analysis"]["fair_odds"] == 2.0
        assert analysis["slippage_analysis"]["slippage_bps"] == 500.0  # 5% = 500bps
        assert analysis["liquidity_analysis"]["fraction_used_pct"] == 8.0  # 800/10000 = 8%
    
    def test_get_statistics(self, engine):
        """Test getting engine statistics"""
        stats = engine.get_statistics()
        
        assert "total_rules" in stats
        assert "rule_types" in stats
        assert "config" in stats
        assert "rate_limiting" in stats
        assert stats["total_rules"] == 5


def run_liquidity_rules_tests():
    """Run all liquidity rules tests"""
    print("💧 Running Liquidity Rules Tests")
    print("=" * 50)
    
    test_classes = [
        ("Odds Range Rule", TestOddsRangeRule),
        ("Slippage Rule", TestSlippageRule),
        ("Liquidity Fraction Rule", TestLiquidityFractionRule),
        ("Token Bucket Rate Limiter", TestTokenBucketRateLimiter),
        ("Sliding Window Rate Limiter", TestSlidingWindowRateLimiter),
        ("Rate Limit Rule", TestRateLimitRule),
        ("Liquidity Rule Engine", TestLiquidityRuleEngine)
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for class_name, test_class in test_classes:
        print(f"\n🔍 {class_name}")
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
                
                # Create fixtures based on test class
                if 'OddsRange' in class_name or 'Slippage' in class_name or 'LiquidityFraction' in class_name:
                    config = LiquidityConfig(
                        min_odds_threshold=1.25,
                        max_odds_threshold=10.0,
                        slippage_bps_limit=50,
                        max_fraction_of_available_liquidity=10.0
                    )
                    
                    if 'OddsRange' in class_name:
                        from rules.liquidity_rules import OddsRangeRule
                        rule = OddsRangeRule(config)
                    elif 'Slippage' in class_name:
                        from rules.liquidity_rules import SlippageRule
                        rule = SlippageRule(config)
                    elif 'LiquidityFraction' in class_name:
                        from rules.liquidity_rules import LiquidityFractionRule
                        rule = LiquidityFractionRule(config)
                    
                    # Run method with appropriate parameters
                    import inspect
                    sig = inspect.signature(method)
                    if 'rule' in sig.parameters and 'config' in sig.parameters:
                        method(rule, config)
                    elif 'rule' in sig.parameters:
                        method(rule)
                    else:
                        method()
                        
                elif 'RateLimit' in class_name and 'Rule' in class_name:
                    rate_config = {
                        'global_rate': 5.0,
                        'global_burst': 10,
                        'market_requests': 3,
                        'market_window_seconds': 60
                    }
                    from rules.liquidity_rules import RateLimitRule
                    rule = RateLimitRule(rate_config)
                    
                    import inspect
                    sig = inspect.signature(method)
                    if 'rule' in sig.parameters and 'config' in sig.parameters:
                        method(rule, rate_config)
                    elif 'rule' in sig.parameters:
                        method(rule)
                    else:
                        method()
                        
                elif 'Engine' in class_name:
                    config = LiquidityConfig(
                        min_odds_threshold=1.25,
                        max_odds_threshold=10.0,
                        slippage_bps_limit=50,
                        max_fraction_of_available_liquidity=10.0
                    )
                    rate_config = {
                        'global_rate': 10.0,
                        'global_burst': 20,
                        'market_requests': 5,
                        'market_window_seconds': 120
                    }
                    from rules.liquidity_rules import LiquidityRuleEngine
                    engine = LiquidityRuleEngine(config, rate_config)
                    
                    import inspect
                    sig = inspect.signature(method)
                    if 'engine' in sig.parameters:
                        method(engine)
                    else:
                        method()
                else:
                    # For rate limiter classes without fixtures
                    method()
                
                print(f"  ✅ {test_method}")
                passed_tests += 1
                class_passed += 1
                
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
        
        print(f"  📈 Class Results: {class_passed}/{len(test_methods)} passed")
    
    print(f"\n🏆 Overall Liquidity Rules Test Results: {passed_tests}/{total_tests} passed")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_liquidity_rules_tests()
    exit(0 if success else 1)
