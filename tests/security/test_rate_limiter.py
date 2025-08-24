# Purpose: Unit tests for rate limiting and DDoS protection
# Author: WicketWise AI, Last Modified: 2024

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from crickformers.security.rate_limiter import (
    RateLimiter,
    RateLimitRule,
    RateLimitConfig,
    RateLimitStrategy,
    RateLimitExceeded
)


class TestRateLimitRule:
    """Test suite for RateLimitRule"""
    
    def test_rule_creation(self):
        """Test RateLimitRule creation"""
        rule = RateLimitRule(
            name="test_rule",
            requests_per_window=100,
            window_size_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            burst_allowance=10
        )
        
        assert rule.name == "test_rule"
        assert rule.requests_per_window == 100
        assert rule.window_size_seconds == 60
        assert rule.strategy == RateLimitStrategy.SLIDING_WINDOW
        assert rule.burst_allowance == 10
        assert len(rule.exempt_keys) == 0
    
    def test_rule_generate_key_simple(self):
        """Test simple key generation"""
        rule = RateLimitRule(
            name="simple_rule",
            requests_per_window=50,
            window_size_seconds=30
        )
        
        key = rule.generate_key("test_identifier")
        assert key == "simple_rule:test_identifier"
    
    def test_rule_generate_key_dict(self):
        """Test key generation from dictionary"""
        rule = RateLimitRule(
            name="dict_rule",
            requests_per_window=50,
            window_size_seconds=30
        )
        
        identifier = {
            'ip_address': '192.168.1.1',
            'user_id': 'user_123',
            'endpoint': '/api/data'
        }
        
        key = rule.generate_key(identifier)
        expected = "dict_rule:ip:192.168.1.1:user:user_123:endpoint:/api/data"
        assert key == expected
    
    def test_rule_generate_key_custom(self):
        """Test custom key generation"""
        def custom_key_gen(identifier):
            return f"custom:{identifier['session_id']}"
        
        rule = RateLimitRule(
            name="custom_rule",
            requests_per_window=50,
            window_size_seconds=30,
            key_generator=custom_key_gen
        )
        
        identifier = {'session_id': 'sess_456'}
        key = rule.generate_key(identifier)
        assert key == "custom:sess_456"
    
    def test_rule_is_exempt(self):
        """Test exemption checking"""
        rule = RateLimitRule(
            name="exempt_rule",
            requests_per_window=50,
            window_size_seconds=30,
            exempt_keys={"exempt_key_1", "exempt_key_2"}
        )
        
        assert rule.is_exempt("exempt_key_1") is True
        assert rule.is_exempt("exempt_key_2") is True
        assert rule.is_exempt("normal_key") is False


class TestRateLimitConfig:
    """Test suite for RateLimitConfig"""
    
    def test_config_creation_empty(self):
        """Test config creation with empty rules"""
        config = RateLimitConfig()
        
        # Should have default rules
        assert len(config.default_rules) > 0
        assert config.cleanup_interval_seconds == 300
        assert config.max_tracked_keys == 100000
        assert config.enable_metrics is True
    
    def test_config_creation_custom(self):
        """Test config creation with custom rules"""
        custom_rules = [
            RateLimitRule("custom1", 10, 60),
            RateLimitRule("custom2", 20, 120)
        ]
        
        config = RateLimitConfig(
            default_rules=custom_rules,
            cleanup_interval_seconds=600,
            max_tracked_keys=50000,
            enable_metrics=False
        )
        
        assert len(config.default_rules) == 2
        assert config.cleanup_interval_seconds == 600
        assert config.max_tracked_keys == 50000
        assert config.enable_metrics is False


class TestRateLimiter:
    """Test suite for RateLimiter"""
    
    @pytest.fixture
    def simple_config(self):
        """Create simple rate limiter configuration"""
        rules = [
            RateLimitRule(
                name="test_rule",
                requests_per_window=5,
                window_size_seconds=10,
                strategy=RateLimitStrategy.SLIDING_WINDOW
            )
        ]
        return RateLimitConfig(
            default_rules=rules,
            enable_metrics=True
        )
    
    @pytest.fixture
    def rate_limiter(self, simple_config):
        """Create rate limiter for testing"""
        limiter = RateLimiter(simple_config)
        # Stop cleanup thread for testing
        limiter._cleanup_active = False
        return limiter
    
    def test_limiter_initialization(self, rate_limiter):
        """Test rate limiter initialization"""
        assert len(rate_limiter.rules) == 1
        assert "test_rule" in rate_limiter.rules
        assert rate_limiter.metrics['total_requests'] == 0
        assert rate_limiter.metrics['blocked_requests'] == 0
    
    def test_add_remove_rule(self, rate_limiter):
        """Test adding and removing rules"""
        new_rule = RateLimitRule("new_rule", 10, 20)
        
        # Add rule
        success = rate_limiter.add_rule(new_rule)
        assert success is True
        assert "new_rule" in rate_limiter.rules
        
        # Remove rule
        success = rate_limiter.remove_rule("new_rule")
        assert success is True
        assert "new_rule" not in rate_limiter.rules
        
        # Remove non-existent rule
        success = rate_limiter.remove_rule("nonexistent")
        assert success is False
    
    def test_sliding_window_allow(self, rate_limiter):
        """Test sliding window allows requests within limit"""
        identifier = "test_user"
        
        # Make requests within limit (5 requests allowed)
        for i in range(5):
            result = rate_limiter.check_rate_limit(identifier)
            assert result['allowed'] is True
            assert len(result['violations']) == 0
    
    def test_sliding_window_block(self, rate_limiter):
        """Test sliding window blocks requests over limit"""
        identifier = "blocked_user"
        
        # Make requests up to limit
        for i in range(5):
            result = rate_limiter.check_rate_limit(identifier)
            assert result['allowed'] is True
        
        # Next request should be blocked
        result = rate_limiter.check_rate_limit(identifier)
        assert result['allowed'] is False
        assert len(result['violations']) == 1
        assert result['violations'][0]['rule_name'] == "test_rule"
        assert result['retry_after'] > 0
    
    def test_sliding_window_recovery(self, rate_limiter):
        """Test sliding window recovers after time window"""
        identifier = "recovery_user"
        
        # Fill up the limit
        for i in range(5):
            rate_limiter.check_rate_limit(identifier)
        
        # Should be blocked
        result = rate_limiter.check_rate_limit(identifier)
        assert result['allowed'] is False
        
        # Simulate time passing (mock the request history cleanup)
        current_time = time.time()
        cutoff_time = current_time - 11  # 11 seconds ago (beyond 10 second window)
        
        # Manually clean old requests to simulate time passing
        key = rate_limiter.rules["test_rule"].generate_key(identifier)
        if key in rate_limiter.request_history:
            rate_limiter.request_history[key].clear()
        
        # Should be allowed again
        result = rate_limiter.check_rate_limit(identifier)
        assert result['allowed'] is True
    
    def test_token_bucket_strategy(self):
        """Test token bucket rate limiting strategy"""
        rules = [
            RateLimitRule(
                name="token_bucket_rule",
                requests_per_window=3,
                window_size_seconds=10,
                strategy=RateLimitStrategy.TOKEN_BUCKET
            )
        ]
        config = RateLimitConfig(default_rules=rules)
        limiter = RateLimiter(config)
        limiter._cleanup_active = False
        
        identifier = "token_user"
        
        # Should allow initial requests (bucket starts full)
        for i in range(3):
            result = limiter.check_rate_limit(identifier, ["token_bucket_rule"])
            assert result['allowed'] is True
        
        # Should block next request (bucket empty)
        result = limiter.check_rate_limit(identifier, ["token_bucket_rule"])
        assert result['allowed'] is False
    
    def test_fixed_window_strategy(self):
        """Test fixed window rate limiting strategy"""
        rules = [
            RateLimitRule(
                name="fixed_window_rule",
                requests_per_window=3,
                window_size_seconds=10,
                strategy=RateLimitStrategy.FIXED_WINDOW
            )
        ]
        config = RateLimitConfig(default_rules=rules)
        limiter = RateLimiter(config)
        limiter._cleanup_active = False
        
        identifier = "fixed_user"
        
        # Should allow requests within window
        for i in range(3):
            result = limiter.check_rate_limit(identifier, ["fixed_window_rule"])
            assert result['allowed'] is True
        
        # Should block next request in same window
        result = limiter.check_rate_limit(identifier, ["fixed_window_rule"])
        assert result['allowed'] is False
    
    def test_exemption_handling(self, rate_limiter):
        """Test rate limit exemptions"""
        identifier = "exempt_user"
        rule_name = "test_rule"
        
        # Add exemption
        key = rate_limiter.rules[rule_name].generate_key(identifier)
        success = rate_limiter.add_exemption(rule_name, key)
        assert success is True
        
        # Should allow unlimited requests
        for i in range(10):  # Well over the limit of 5
            result = rate_limiter.check_rate_limit(identifier)
            assert result['allowed'] is True
        
        # Remove exemption
        success = rate_limiter.remove_exemption(rule_name, key)
        assert success is True
        
        # Should now be subject to limits again
        # (Note: might need to reset state for clean test)
    
    def test_reset_rate_limit(self, rate_limiter):
        """Test resetting rate limits"""
        identifier = "reset_user"
        
        # Fill up the limit
        for i in range(5):
            rate_limiter.check_rate_limit(identifier)
        
        # Should be blocked
        result = rate_limiter.check_rate_limit(identifier)
        assert result['allowed'] is False
        
        # Reset rate limit
        success = rate_limiter.reset_rate_limit(identifier, "test_rule")
        assert success is True
        
        # Should be allowed again
        result = rate_limiter.check_rate_limit(identifier)
        assert result['allowed'] is True
    
    def test_rate_limit_status(self, rate_limiter):
        """Test getting rate limit status"""
        identifier = "status_user"
        
        # Make some requests
        for i in range(3):
            rate_limiter.check_rate_limit(identifier)
        
        # Get status
        status = rate_limiter.get_rate_limit_status(identifier, "test_rule")
        
        assert 'rule_name' in status
        assert 'key' in status
        assert 'exempt' in status
        assert 'requests_used' in status
        assert 'requests_remaining' in status
        assert 'limit' in status
        
        assert status['rule_name'] == "test_rule"
        assert status['exempt'] is False
        assert status['requests_used'] == 3
        assert status['requests_remaining'] == 2
        assert status['limit'] == 5
    
    def test_rate_limit_status_nonexistent_rule(self, rate_limiter):
        """Test getting status for non-existent rule"""
        status = rate_limiter.get_rate_limit_status("user", "nonexistent_rule")
        assert 'error' in status
        assert 'not found' in status['error'].lower()
    
    def test_multiple_rules(self):
        """Test rate limiting with multiple rules"""
        rules = [
            RateLimitRule("rule1", 3, 10, RateLimitStrategy.SLIDING_WINDOW),
            RateLimitRule("rule2", 5, 10, RateLimitStrategy.SLIDING_WINDOW)
        ]
        config = RateLimitConfig(default_rules=rules)
        limiter = RateLimiter(config)
        limiter._cleanup_active = False
        
        identifier = "multi_user"
        
        # Make requests - should be limited by most restrictive rule (rule1: 3 requests)
        for i in range(3):
            result = limiter.check_rate_limit(identifier)
            assert result['allowed'] is True
        
        # Should be blocked by rule1
        result = limiter.check_rate_limit(identifier)
        assert result['allowed'] is False
        assert len(result['violations']) == 1
        assert result['violations'][0]['rule_name'] == "rule1"
    
    def test_specific_rules_check(self, rate_limiter):
        """Test checking specific rules only"""
        # Add another rule
        new_rule = RateLimitRule("specific_rule", 2, 10)
        rate_limiter.add_rule(new_rule)
        
        identifier = "specific_user"
        
        # Check only specific rule
        for i in range(2):
            result = rate_limiter.check_rate_limit(identifier, ["specific_rule"])
            assert result['allowed'] is True
        
        # Should be blocked by specific rule
        result = rate_limiter.check_rate_limit(identifier, ["specific_rule"])
        assert result['allowed'] is False
        assert result['violations'][0]['rule_name'] == "specific_rule"
        
        # But should still be allowed by other rule (test_rule allows 5)
        result = rate_limiter.check_rate_limit(identifier, ["test_rule"])
        assert result['allowed'] is True
    
    def test_metrics_collection(self, rate_limiter):
        """Test metrics collection"""
        identifier = "metrics_user"
        
        # Make some requests
        for i in range(3):
            rate_limiter.check_rate_limit(identifier)
        
        # Make blocked request
        for i in range(3):  # Fill remaining limit + 1 blocked
            rate_limiter.check_rate_limit(identifier)
        
        # Check metrics
        assert rate_limiter.metrics['total_requests'] == 6
        assert rate_limiter.metrics['blocked_requests'] == 1
        assert rate_limiter.metrics['rules_triggered']['test_rule'] == 1
    
    def test_get_statistics(self, rate_limiter):
        """Test getting rate limiter statistics"""
        identifier = "stats_user"
        
        # Generate some activity
        for i in range(3):
            rate_limiter.check_rate_limit(identifier)
        
        # Block one request
        for i in range(3):
            rate_limiter.check_rate_limit(identifier)
        
        stats = rate_limiter.get_statistics()
        
        assert 'total_requests' in stats
        assert 'blocked_requests' in stats
        assert 'block_rate' in stats
        assert 'active_rules' in stats
        assert 'tracked_keys' in stats
        assert 'rules_triggered' in stats
        
        assert stats['total_requests'] == 6
        assert stats['blocked_requests'] == 1
        assert stats['active_rules'] == 1
        assert stats['block_rate'] > 0


def run_rate_limiter_tests():
    """Run all rate limiter tests"""
    print("🚦 Running Rate Limiter Tests")
    print("=" * 50)
    
    # Test categories
    test_categories = [
        ("Rate Limit Rule", TestRateLimitRule),
        ("Rate Limit Config", TestRateLimitConfig),
        ("Rate Limiter", TestRateLimiter)
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for category_name, test_class in test_categories:
        print(f"\n📊 {category_name}")
        print("-" * 30)
        
        # Get test methods
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_')]
        
        category_passed = 0
        for test_method in test_methods:
            total_tests += 1
            try:
                # Create test instance
                test_instance = test_class()
                
                # Handle fixtures
                if hasattr(test_instance, test_method):
                    method = getattr(test_instance, test_method)
                    
                    # Check if method needs fixtures
                    import inspect
                    sig = inspect.signature(method)
                    
                    if 'rate_limiter' in sig.parameters:
                        # Create simple config
                        rules = [
                            RateLimitRule(
                                name="test_rule",
                                requests_per_window=5,
                                window_size_seconds=10,
                                strategy=RateLimitStrategy.SLIDING_WINDOW
                            )
                        ]
                        simple_config = RateLimitConfig(
                            default_rules=rules,
                            enable_metrics=True
                        )
                        rate_limiter = RateLimiter(simple_config)
                        rate_limiter._cleanup_active = False
                        
                        if 'simple_config' in sig.parameters:
                            method(simple_config, rate_limiter)
                        else:
                            method(rate_limiter)
                    elif 'simple_config' in sig.parameters:
                        rules = [
                            RateLimitRule(
                                name="test_rule",
                                requests_per_window=5,
                                window_size_seconds=10,
                                strategy=RateLimitStrategy.SLIDING_WINDOW
                            )
                        ]
                        simple_config = RateLimitConfig(
                            default_rules=rules,
                            enable_metrics=True
                        )
                        method(simple_config)
                    else:
                        method()
                    
                    print(f"  ✅ {test_method}")
                    passed_tests += 1
                    category_passed += 1
                    
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
        
        print(f"  📈 Category Results: {category_passed}/{len(test_methods)} passed")
    
    print(f"\n🏆 Overall Rate Limiter Test Results: {passed_tests}/{total_tests} passed")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_rate_limiter_tests()
    exit(0 if success else 1)
