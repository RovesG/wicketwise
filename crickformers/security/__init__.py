# Purpose: Security and authentication module for WicketWise platform
# Author: WicketWise AI, Last Modified: 2024

"""
Security & Authentication Module

This module provides comprehensive security and authentication features
for the WicketWise cricket intelligence platform, including JWT authentication,
OAuth integration, rate limiting, and security monitoring.

Key Components:
- JWT token management and validation
- OAuth 2.0 integration (Google, GitHub, Microsoft)
- Multi-factor authentication (MFA) support
- Rate limiting and DDoS protection
- API key management and validation
- Security event monitoring and alerting
- Role-based access control (RBAC)
- Session management and security
- Password hashing and validation
- Security headers and CSRF protection
"""

__version__ = "1.0.0"
__author__ = "WicketWise AI"

# Core security components
from .auth_manager import (
    AuthManager,
    AuthToken,
    AuthUser,
    AuthRole,
    AuthPermission,
    TokenType,
    AuthenticationError,
    AuthorizationError
)

from .jwt_handler import (
    JWTHandler,
    JWTConfig,
    TokenClaims,
    TokenValidationError,
    TokenExpiredError
)

from .rate_limiter import (
    RateLimiter,
    RateLimitRule,
    RateLimitExceeded,
    RateLimitConfig,
    RateLimitStrategy
)

from .security_monitor import (
    SecurityMonitor,
    SecurityEvent,
    SecurityThreat,
    SecurityAlert,
    ThreatLevel,
    SecurityEventType
)

__all__ = [
    # Authentication management
    'AuthManager',
    'AuthToken',
    'AuthUser',
    'AuthRole',
    'AuthPermission',
    'TokenType',
    'AuthenticationError',
    'AuthorizationError',
    
    # JWT handling
    'JWTHandler',
    'JWTConfig',
    'TokenClaims',
    'TokenValidationError',
    'TokenExpiredError',
    
    # Rate limiting
    'RateLimiter',
    'RateLimitRule',
    'RateLimitExceeded',
    'RateLimitConfig',
    'RateLimitStrategy',
    
    # Security monitoring
    'SecurityMonitor',
    'SecurityEvent',
    'SecurityThreat',
    'SecurityAlert',
    'ThreatLevel',
    'SecurityEventType'
]
