# Purpose: Rate limiting and DDoS protection system
# Author: WicketWise AI, Last Modified: 2024

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import hashlib


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitExceeded(Exception):
    """Rate limit exceeded exception"""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    name: str
    requests_per_window: int
    window_size_seconds: int
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_allowance: int = 0  # Additional requests allowed in burst
    key_generator: Optional[Callable[[Any], str]] = None
    exempt_keys: set = field(default_factory=set)
    
    def generate_key(self, identifier: Any) -> str:
        """Generate rate limit key for identifier"""
        if self.key_generator:
            return self.key_generator(identifier)
        
        # Default key generation
        if isinstance(identifier, dict):
            # For request context (IP, user_id, etc.)
            parts = []
            if 'ip_address' in identifier:
                parts.append(f"ip:{identifier['ip_address']}")
            if 'user_id' in identifier:
                parts.append(f"user:{identifier['user_id']}")
            if 'endpoint' in identifier:
                parts.append(f"endpoint:{identifier['endpoint']}")
            
            return f"{self.name}:" + ":".join(parts)
        else:
            return f"{self.name}:{str(identifier)}"
    
    def is_exempt(self, key: str) -> bool:
        """Check if key is exempt from rate limiting"""
        return key in self.exempt_keys


@dataclass
class RateLimitConfig:
    """Rate limiter configuration"""
    default_rules: List[RateLimitRule] = field(default_factory=list)
    cleanup_interval_seconds: int = 300  # 5 minutes
    max_tracked_keys: int = 100000
    enable_metrics: bool = True
    
    def __post_init__(self):
        """Initialize default rules if none provided"""
        if not self.default_rules:
            self.default_rules = [
                # Global rate limits
                RateLimitRule(
                    name="global_api",
                    requests_per_window=1000,
                    window_size_seconds=60,
                    strategy=RateLimitStrategy.SLIDING_WINDOW
                ),
                # Per-IP rate limits
                RateLimitRule(
                    name="per_ip",
                    requests_per_window=100,
                    window_size_seconds=60,
                    strategy=RateLimitStrategy.SLIDING_WINDOW
                ),
                # Per-user rate limits
                RateLimitRule(
                    name="per_user",
                    requests_per_window=200,
                    window_size_seconds=60,
                    strategy=RateLimitStrategy.SLIDING_WINDOW
                ),
                # Authentication endpoint limits
                RateLimitRule(
                    name="auth_endpoint",
                    requests_per_window=10,
                    window_size_seconds=300,  # 5 minutes
                    strategy=RateLimitStrategy.FIXED_WINDOW
                )
            ]


@dataclass
class RequestRecord:
    """Individual request record for rate limiting"""
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BucketState:
    """Token/Leaky bucket state"""
    tokens: float
    last_refill: float
    
    def __post_init__(self):
        if self.last_refill == 0:
            self.last_refill = time.time()


class RateLimiter:
    """Comprehensive rate limiting system"""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting data structures
        self.rules: Dict[str, RateLimitRule] = {}
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque())
        self.bucket_states: Dict[str, BucketState] = {}
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'blocked_requests': 0,
            'rules_triggered': defaultdict(int),
            'top_blocked_keys': defaultdict(int)
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize rules
        for rule in self.config.default_rules:
            self.add_rule(rule)
        
        # Start cleanup thread
        self._cleanup_active = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def add_rule(self, rule: RateLimitRule) -> bool:
        """Add rate limiting rule"""
        with self.lock:
            self.rules[rule.name] = rule
            self.logger.info(f"Added rate limit rule: {rule.name}")
            return True
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove rate limiting rule"""
        with self.lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                self.logger.info(f"Removed rate limit rule: {rule_name}")
                return True
            return False
    
    def check_rate_limit(self, identifier: Any, rule_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Check rate limits for identifier against specified rules"""
        with self.lock:
            current_time = time.time()
            
            # Update metrics
            if self.config.enable_metrics:
                self.metrics['total_requests'] += 1
            
            # Determine which rules to check
            rules_to_check = []
            if rule_names:
                rules_to_check = [self.rules[name] for name in rule_names if name in self.rules]
            else:
                rules_to_check = list(self.rules.values())
            
            # Check each rule
            violations = []
            for rule in rules_to_check:
                key = rule.generate_key(identifier)
                
                # Skip if exempt
                if rule.is_exempt(key):
                    continue
                
                # Check rate limit based on strategy
                is_allowed, retry_after = self._check_rule(rule, key, current_time)
                
                if not is_allowed:
                    violations.append({
                        'rule_name': rule.name,
                        'key': key,
                        'retry_after': retry_after,
                        'limit': rule.requests_per_window,
                        'window': rule.window_size_seconds
                    })
                    
                    # Update metrics
                    if self.config.enable_metrics:
                        self.metrics['blocked_requests'] += 1
                        self.metrics['rules_triggered'][rule.name] += 1
                        self.metrics['top_blocked_keys'][key] += 1
            
            # Return result
            if violations:
                # Find the longest retry_after time
                max_retry_after = max(v['retry_after'] for v in violations)
                
                return {
                    'allowed': False,
                    'violations': violations,
                    'retry_after': max_retry_after,
                    'message': f"Rate limit exceeded. Try again in {max_retry_after} seconds."
                }
            else:
                # Record successful request for all applicable rules
                for rule in rules_to_check:
                    key = rule.generate_key(identifier)
                    if not rule.is_exempt(key):
                        self._record_request(rule, key, current_time)
                
                return {
                    'allowed': True,
                    'violations': [],
                    'retry_after': 0,
                    'message': 'Request allowed'
                }
    
    def _check_rule(self, rule: RateLimitRule, key: str, current_time: float) -> Tuple[bool, int]:
        """Check specific rule for key"""
        if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._check_fixed_window(rule, key, current_time)
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._check_sliding_window(rule, key, current_time)
        elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._check_token_bucket(rule, key, current_time)
        elif rule.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return self._check_leaky_bucket(rule, key, current_time)
        else:
            # Default to sliding window
            return self._check_sliding_window(rule, key, current_time)
    
    def _check_fixed_window(self, rule: RateLimitRule, key: str, current_time: float) -> Tuple[bool, int]:
        """Check fixed window rate limit"""
        window_start = int(current_time // rule.window_size_seconds) * rule.window_size_seconds
        window_key = f"{key}:window:{window_start}"
        
        # Count requests in current window
        request_count = 0
        if window_key in self.request_history:
            request_count = len(self.request_history[window_key])
        
        # Check limit
        if request_count >= rule.requests_per_window:
            retry_after = int(window_start + rule.window_size_seconds - current_time)
            return False, max(1, retry_after)
        
        return True, 0
    
    def _check_sliding_window(self, rule: RateLimitRule, key: str, current_time: float) -> Tuple[bool, int]:
        """Check sliding window rate limit"""
        # Clean old requests
        cutoff_time = current_time - rule.window_size_seconds
        
        if key in self.request_history:
            # Remove old requests
            while (self.request_history[key] and 
                   self.request_history[key][0].timestamp < cutoff_time):
                self.request_history[key].popleft()
        
        # Count current requests
        request_count = len(self.request_history[key]) if key in self.request_history else 0
        
        # Check limit (including burst allowance)
        total_allowed = rule.requests_per_window + rule.burst_allowance
        if request_count >= total_allowed:
            # Calculate retry after based on oldest request
            if self.request_history[key]:
                oldest_request = self.request_history[key][0].timestamp
                retry_after = int(oldest_request + rule.window_size_seconds - current_time)
                return False, max(1, retry_after)
            else:
                return False, rule.window_size_seconds
        
        return True, 0
    
    def _check_token_bucket(self, rule: RateLimitRule, key: str, current_time: float) -> Tuple[bool, int]:
        """Check token bucket rate limit"""
        # Initialize bucket if not exists
        if key not in self.bucket_states:
            self.bucket_states[key] = BucketState(
                tokens=float(rule.requests_per_window),
                last_refill=current_time
            )
        
        bucket = self.bucket_states[key]
        
        # Refill tokens based on time elapsed
        time_elapsed = current_time - bucket.last_refill
        refill_rate = rule.requests_per_window / rule.window_size_seconds
        tokens_to_add = time_elapsed * refill_rate
        
        bucket.tokens = min(rule.requests_per_window, bucket.tokens + tokens_to_add)
        bucket.last_refill = current_time
        
        # Check if token available
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True, 0
        else:
            # Calculate retry after
            tokens_needed = 1.0 - bucket.tokens
            retry_after = int(tokens_needed / refill_rate)
            return False, max(1, retry_after)
    
    def _check_leaky_bucket(self, rule: RateLimitRule, key: str, current_time: float) -> Tuple[bool, int]:
        """Check leaky bucket rate limit"""
        # Similar to token bucket but with different semantics
        # Leaky bucket focuses on smoothing output rate
        
        if key not in self.bucket_states:
            self.bucket_states[key] = BucketState(
                tokens=0.0,  # Start empty
                last_refill=current_time
            )
        
        bucket = self.bucket_states[key]
        
        # Leak tokens (process requests) at steady rate
        time_elapsed = current_time - bucket.last_refill
        leak_rate = rule.requests_per_window / rule.window_size_seconds
        tokens_leaked = time_elapsed * leak_rate
        
        bucket.tokens = max(0, bucket.tokens - tokens_leaked)
        bucket.last_refill = current_time
        
        # Check if bucket has capacity
        if bucket.tokens < rule.requests_per_window:
            bucket.tokens += 1.0
            return True, 0
        else:
            # Bucket is full, calculate retry after
            retry_after = int((bucket.tokens - rule.requests_per_window + 1) / leak_rate)
            return False, max(1, retry_after)
    
    def _record_request(self, rule: RateLimitRule, key: str, current_time: float):
        """Record successful request"""
        if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            window_start = int(current_time // rule.window_size_seconds) * rule.window_size_seconds
            window_key = f"{key}:window:{window_start}"
            self.request_history[window_key].append(RequestRecord(current_time))
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            self.request_history[key].append(RequestRecord(current_time))
        # Token and leaky bucket strategies handle recording in check methods
    
    def reset_rate_limit(self, identifier: Any, rule_name: Optional[str] = None) -> bool:
        """Reset rate limit for identifier"""
        with self.lock:
            if rule_name and rule_name in self.rules:
                rule = self.rules[rule_name]
                key = rule.generate_key(identifier)
                
                # Clear history and bucket state
                if key in self.request_history:
                    del self.request_history[key]
                if key in self.bucket_states:
                    del self.bucket_states[key]
                
                self.logger.info(f"Reset rate limit for key: {key}, rule: {rule_name}")
                return True
            else:
                # Reset all rules for identifier
                reset_count = 0
                for rule in self.rules.values():
                    key = rule.generate_key(identifier)
                    if key in self.request_history:
                        del self.request_history[key]
                        reset_count += 1
                    if key in self.bucket_states:
                        del self.bucket_states[key]
                        reset_count += 1
                
                self.logger.info(f"Reset {reset_count} rate limits for identifier: {identifier}")
                return reset_count > 0
    
    def add_exemption(self, rule_name: str, key: str) -> bool:
        """Add exemption for specific key in rule"""
        with self.lock:
            if rule_name in self.rules:
                self.rules[rule_name].exempt_keys.add(key)
                self.logger.info(f"Added exemption for key: {key}, rule: {rule_name}")
                return True
            return False
    
    def remove_exemption(self, rule_name: str, key: str) -> bool:
        """Remove exemption for specific key in rule"""
        with self.lock:
            if rule_name in self.rules and key in self.rules[rule_name].exempt_keys:
                self.rules[rule_name].exempt_keys.remove(key)
                self.logger.info(f"Removed exemption for key: {key}, rule: {rule_name}")
                return True
            return False
    
    def get_rate_limit_status(self, identifier: Any, rule_name: str) -> Dict[str, Any]:
        """Get current rate limit status for identifier and rule"""
        with self.lock:
            if rule_name not in self.rules:
                return {'error': 'Rule not found'}
            
            rule = self.rules[rule_name]
            key = rule.generate_key(identifier)
            current_time = time.time()
            
            if rule.is_exempt(key):
                return {
                    'rule_name': rule_name,
                    'key': key,
                    'exempt': True,
                    'requests_remaining': float('inf'),
                    'reset_time': None
                }
            
            # Calculate remaining requests based on strategy
            if rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
                cutoff_time = current_time - rule.window_size_seconds
                if key in self.request_history:
                    # Clean old requests
                    while (self.request_history[key] and 
                           self.request_history[key][0].timestamp < cutoff_time):
                        self.request_history[key].popleft()
                    
                    current_requests = len(self.request_history[key])
                else:
                    current_requests = 0
                
                remaining = max(0, rule.requests_per_window - current_requests)
                reset_time = None
                if self.request_history[key]:
                    oldest_request = self.request_history[key][0].timestamp
                    reset_time = oldest_request + rule.window_size_seconds
                
                return {
                    'rule_name': rule_name,
                    'key': key,
                    'exempt': False,
                    'requests_used': current_requests,
                    'requests_remaining': remaining,
                    'limit': rule.requests_per_window,
                    'window_size': rule.window_size_seconds,
                    'reset_time': reset_time
                }
            
            elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
                if key in self.bucket_states:
                    bucket = self.bucket_states[key]
                    
                    # Update tokens
                    time_elapsed = current_time - bucket.last_refill
                    refill_rate = rule.requests_per_window / rule.window_size_seconds
                    tokens_to_add = time_elapsed * refill_rate
                    current_tokens = min(rule.requests_per_window, bucket.tokens + tokens_to_add)
                else:
                    current_tokens = rule.requests_per_window
                
                return {
                    'rule_name': rule_name,
                    'key': key,
                    'exempt': False,
                    'tokens_available': current_tokens,
                    'bucket_capacity': rule.requests_per_window,
                    'refill_rate': rule.requests_per_window / rule.window_size_seconds
                }
            
            # Default response for other strategies
            return {
                'rule_name': rule_name,
                'key': key,
                'exempt': False,
                'strategy': rule.strategy.value
            }
    
    def _cleanup_loop(self):
        """Background cleanup of old data"""
        while self._cleanup_active:
            try:
                time.sleep(self.config.cleanup_interval_seconds)
                self._cleanup_old_data()
            except Exception as e:
                self.logger.error(f"Cleanup error: {str(e)}")
    
    def _cleanup_old_data(self):
        """Clean up old request history and bucket states"""
        with self.lock:
            current_time = time.time()
            
            # Clean request history
            keys_to_remove = []
            for key, history in self.request_history.items():
                # Determine max age based on rules
                max_age = 3600  # Default 1 hour
                for rule in self.rules.values():
                    if key.startswith(f"{rule.name}:"):
                        max_age = max(max_age, rule.window_size_seconds * 2)
                
                cutoff_time = current_time - max_age
                
                # Remove old requests
                while history and history[0].timestamp < cutoff_time:
                    history.popleft()
                
                # Remove empty histories
                if not history:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.request_history[key]
            
            # Clean bucket states (remove inactive ones)
            bucket_keys_to_remove = []
            for key, bucket in self.bucket_states.items():
                if current_time - bucket.last_refill > 3600:  # 1 hour inactive
                    bucket_keys_to_remove.append(key)
            
            for key in bucket_keys_to_remove:
                del self.bucket_states[key]
            
            # Limit total tracked keys
            if len(self.request_history) > self.config.max_tracked_keys:
                # Remove oldest entries
                sorted_keys = sorted(
                    self.request_history.keys(),
                    key=lambda k: self.request_history[k][-1].timestamp if self.request_history[k] else 0
                )
                
                keys_to_remove = sorted_keys[:len(sorted_keys) - self.config.max_tracked_keys]
                for key in keys_to_remove:
                    del self.request_history[key]
            
            self.logger.debug(f"Cleanup completed. Tracking {len(self.request_history)} keys")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        with self.lock:
            return {
                'total_requests': self.metrics['total_requests'],
                'blocked_requests': self.metrics['blocked_requests'],
                'block_rate': (self.metrics['blocked_requests'] / max(1, self.metrics['total_requests'])) * 100,
                'active_rules': len(self.rules),
                'tracked_keys': len(self.request_history),
                'bucket_states': len(self.bucket_states),
                'rules_triggered': dict(self.metrics['rules_triggered']),
                'top_blocked_keys': dict(sorted(self.metrics['top_blocked_keys'].items(), key=lambda x: x[1], reverse=True)[:10])
            }
    
    def shutdown(self):
        """Shutdown rate limiter"""
        self._cleanup_active = False
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        self.logger.info("Rate limiter shutdown completed")
