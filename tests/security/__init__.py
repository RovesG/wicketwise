# Purpose: Security and authentication tests module initialization
# Author: WicketWise AI, Last Modified: 2024

"""
Security & Authentication Tests Module

This module contains comprehensive tests for the WicketWise security
and authentication system, including JWT handling, rate limiting,
and security monitoring.

Test Categories:
- Authentication and authorization management
- JWT token creation and validation
- Rate limiting and DDoS protection
- Security monitoring and threat detection
- Multi-factor authentication (MFA)
- Password policies and security
"""

__version__ = "1.0.0"
__author__ = "WicketWise AI"

# Test utilities and fixtures
from .test_auth_manager import (
    TestAuthManager,
    TestAuthUser,
    TestAuthToken
)

from .test_jwt_handler import (
    TestJWTHandler,
    TestTokenClaims,
    TestJWTConfig
)

from .test_rate_limiter import (
    TestRateLimiter,
    TestRateLimitRule,
    TestRateLimitConfig
)

# Security monitor tests (to be implemented)
# from .test_security_monitor import (
#     TestSecurityMonitor,
#     TestSecurityEvent,
#     TestSecurityThreat
# )

__all__ = [
    'TestAuthManager',
    'TestAuthUser',
    'TestAuthToken',
    'TestJWTHandler',
    'TestTokenClaims',
    'TestJWTConfig',
    'TestRateLimiter',
    'TestRateLimitRule',
    'TestRateLimitConfig'
    # Security monitor tests (to be implemented)
    # 'TestSecurityMonitor',
    # 'TestSecurityEvent',
    # 'TestSecurityThreat'
]
