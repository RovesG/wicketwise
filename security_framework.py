#!/usr/bin/env python3
"""
WicketWise Security Framework
Comprehensive security layer with authentication, authorization, input validation, and audit logging

Author: WicketWise Team, Last Modified: 2025-01-21
"""

import jwt
import hashlib
import secrets
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from functools import wraps
from dataclasses import dataclass
from enum import Enum
import re
import json
import asyncio
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles with hierarchical permissions"""
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

@dataclass
class User:
    """Authenticated user with roles and permissions"""
    id: str
    username: str
    email: str
    roles: Set[UserRole]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True

class SecurityException(Exception):
    """Base security exception"""
    pass

class UnauthorizedException(SecurityException):
    """Authentication failed"""
    pass

class ForbiddenException(SecurityException):
    """Authorization failed"""
    pass

class RateLimitExceededException(SecurityException):
    """Rate limit exceeded"""
    pass

class ValidationException(SecurityException):
    """Input validation failed"""
    pass

class JWTAuthManager:
    """JWT-based authentication manager"""
    
    def __init__(self, secret_key: str, token_expiry_hours: int = 24):
        self.secret_key = secret_key
        self.token_expiry_hours = token_expiry_hours
        self.algorithm = "HS256"
        self.users: Dict[str, User] = {}  # In-memory user store (use database in production)
    
    def create_user(self, username: str, email: str, password: str, roles: Set[UserRole]) -> User:
        """Create a new user with hashed password"""
        user_id = secrets.token_urlsafe(16)
        password_hash = self._hash_password(password)
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            roles=roles,
            created_at=datetime.utcnow()
        )
        
        self.users[user_id] = user
        # Store password hash separately (never in User object)
        self._password_store[user_id] = password_hash
        
        logger.info(f"Created user: {username} with roles: {[r.value for r in roles]}")
        return user
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hash_hex = password_hash.split(':')
            password_check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return password_check.hex() == hash_hex
        except Exception:
            return False
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return JWT token"""
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username and u.is_active:
                user = u
                break
        
        if not user:
            logger.warning(f"Authentication failed: User not found - {username}")
            return None
        
        # Verify password
        password_hash = getattr(self, '_password_store', {}).get(user.id)
        if not password_hash or not self._verify_password(password, password_hash):
            logger.warning(f"Authentication failed: Invalid password - {username}")
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        
        # Create JWT token
        payload = {
            'user_id': user.id,
            'username': user.username,
            'roles': [role.value for role in user.roles],
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"User authenticated: {username}")
        return token
    
    def validate_token(self, token: str) -> Optional[User]:
        """Validate JWT token and return user"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload['user_id']
            
            user = self.users.get(user_id)
            if not user or not user.is_active:
                return None
            
            return user
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

class AdvancedRateLimiter:
    """Advanced rate limiter with multiple strategies"""
    
    def __init__(self):
        self.buckets: Dict[str, deque] = defaultdict(deque)
        self.limits: Dict[str, Dict[str, int]] = {
            # endpoint -> {time_window: max_requests}
            'enrich_matches': {'hour': 10, 'day': 50},
            'build_kg': {'hour': 3, 'day': 10},
            'train_model': {'hour': 1, 'day': 5},
            'default': {'minute': 60, 'hour': 1000}
        }
    
    def check_rate_limit(self, user_id: str, endpoint: str) -> bool:
        """Check if request is within rate limits"""
        current_time = time.time()
        key = f"{user_id}:{endpoint}"
        
        # Get limits for endpoint
        endpoint_limits = self.limits.get(endpoint, self.limits['default'])
        
        # Check each time window
        for window, max_requests in endpoint_limits.items():
            window_seconds = self._get_window_seconds(window)
            cutoff_time = current_time - window_seconds
            
            # Clean old requests
            bucket = self.buckets[f"{key}:{window}"]
            while bucket and bucket[0] < cutoff_time:
                bucket.popleft()
            
            # Check if over limit
            if len(bucket) >= max_requests:
                logger.warning(f"Rate limit exceeded: {user_id} on {endpoint} ({len(bucket)}/{max_requests} per {window})")
                return False
            
            # Add current request
            bucket.append(current_time)
        
        return True
    
    def _get_window_seconds(self, window: str) -> int:
        """Convert window string to seconds"""
        windows = {
            'minute': 60,
            'hour': 3600,
            'day': 86400
        }
        return windows.get(window, 3600)

class CricketInputValidator:
    """Cricket-specific input validator"""
    
    def __init__(self):
        self.schemas = {
            'enrich_matches': {
                'max_matches': {'type': int, 'min': 1, 'max': 1000},
                'priority_competitions': {'type': list, 'max_length': 20},
                'force_refresh': {'type': bool}
            },
            'player_query': {
                'player_name': {'type': str, 'min_length': 2, 'max_length': 100, 'pattern': r'^[a-zA-Z\s\-\.\']+$'},
                'format': {'type': str, 'choices': ['T20', 'ODI', 'Test', 'All']},
                'venue': {'type': str, 'max_length': 200, 'optional': True}
            },
            'match_data': {
                'home_team': {'type': str, 'min_length': 2, 'max_length': 50},
                'away_team': {'type': str, 'min_length': 2, 'max_length': 50},
                'venue': {'type': str, 'min_length': 2, 'max_length': 100},
                'date': {'type': str, 'pattern': r'^\d{4}-\d{2}-\d{2}$'},
                'competition': {'type': str, 'max_length': 100}
            }
        }
    
    def validate(self, data: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """Validate input data against schema"""
        schema = self.schemas.get(schema_name)
        if not schema:
            raise ValidationException(f"Unknown validation schema: {schema_name}")
        
        validated_data = {}
        
        for field, rules in schema.items():
            value = data.get(field)
            
            # Check if field is optional
            if value is None:
                if rules.get('optional', False):
                    continue
                else:
                    raise ValidationException(f"Required field missing: {field}")
            
            # Type validation
            expected_type = rules['type']
            if not isinstance(value, expected_type):
                raise ValidationException(f"Invalid type for {field}: expected {expected_type.__name__}, got {type(value).__name__}")
            
            # String validations
            if expected_type == str:
                if 'min_length' in rules and len(value) < rules['min_length']:
                    raise ValidationException(f"{field} too short: minimum {rules['min_length']} characters")
                
                if 'max_length' in rules and len(value) > rules['max_length']:
                    raise ValidationException(f"{field} too long: maximum {rules['max_length']} characters")
                
                if 'pattern' in rules and not re.match(rules['pattern'], value):
                    raise ValidationException(f"{field} format invalid")
                
                if 'choices' in rules and value not in rules['choices']:
                    raise ValidationException(f"{field} must be one of: {rules['choices']}")
            
            # Numeric validations
            elif expected_type in [int, float]:
                if 'min' in rules and value < rules['min']:
                    raise ValidationException(f"{field} too small: minimum {rules['min']}")
                
                if 'max' in rules and value > rules['max']:
                    raise ValidationException(f"{field} too large: maximum {rules['max']}")
            
            # List validations
            elif expected_type == list:
                if 'max_length' in rules and len(value) > rules['max_length']:
                    raise ValidationException(f"{field} too many items: maximum {rules['max_length']}")
            
            validated_data[field] = value
        
        return validated_data

class SecurityAuditLogger:
    """Security audit logging system"""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.audit_logger = logging.getLogger("security_audit")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.audit_logger.addHandler(handler)
        self.audit_logger.setLevel(logging.INFO)
    
    def log_authentication(self, username: str, success: bool, ip_address: str = None):
        """Log authentication attempts"""
        status = "SUCCESS" if success else "FAILED"
        self.audit_logger.info(f"AUTH_{status}: user={username}, ip={ip_address}")
    
    def log_authorization(self, user: User, endpoint: str, success: bool):
        """Log authorization attempts"""
        status = "SUCCESS" if success else "FAILED"
        self.audit_logger.info(f"AUTHZ_{status}: user={user.username}, endpoint={endpoint}, roles={[r.value for r in user.roles]}")
    
    def log_rate_limit(self, user_id: str, endpoint: str):
        """Log rate limit violations"""
        self.audit_logger.warning(f"RATE_LIMIT: user={user_id}, endpoint={endpoint}")
    
    def log_validation_error(self, user: User, endpoint: str, error: str):
        """Log validation errors"""
        self.audit_logger.warning(f"VALIDATION_ERROR: user={user.username}, endpoint={endpoint}, error={error}")
    
    def log_security_event(self, event_type: str, user: Optional[User], details: Dict[str, Any]):
        """Log general security events"""
        user_info = user.username if user else "anonymous"
        self.audit_logger.info(f"SECURITY_EVENT: type={event_type}, user={user_info}, details={json.dumps(details)}")

class WicketWiseSecurityFramework:
    """Main security framework orchestrator"""
    
    def __init__(self, secret_key: str):
        self.auth_manager = JWTAuthManager(secret_key)
        self.rate_limiter = AdvancedRateLimiter()
        self.input_validator = CricketInputValidator()
        self.audit_logger = SecurityAuditLogger()
        
        # Initialize password store
        self.auth_manager._password_store = {}
        
        # Create default admin user
        self._create_default_users()
    
    def _create_default_users(self):
        """Create default users for initial setup"""
        # Create admin user
        admin_user = self.auth_manager.create_user(
            username="admin",
            email="admin@wicketwise.com",
            password="ChangeMe123!",  # Should be changed on first login
            roles={UserRole.SUPER_ADMIN, UserRole.ADMIN, UserRole.ANALYST, UserRole.VIEWER}
        )
        
        # Create analyst user
        analyst_user = self.auth_manager.create_user(
            username="analyst",
            email="analyst@wicketwise.com", 
            password="Analyst123!",
            roles={UserRole.ANALYST, UserRole.VIEWER}
        )
        
        logger.info("Default users created - CHANGE DEFAULT PASSWORDS!")
    
    def secure_endpoint(self, required_roles: List[UserRole] = None, rate_limit_endpoint: str = None):
        """Decorator for securing API endpoints"""
        def decorator(func):
            @wraps(func)
            async def wrapper(request, *args, **kwargs):
                try:
                    # 1. Extract and validate token
                    auth_header = request.headers.get('Authorization', '')
                    if not auth_header.startswith('Bearer '):
                        raise UnauthorizedException("Missing or invalid authorization header")
                    
                    token = auth_header[7:]  # Remove 'Bearer '
                    user = self.auth_manager.validate_token(token)
                    if not user:
                        raise UnauthorizedException("Invalid or expired token")
                    
                    # 2. Check authorization
                    if required_roles:
                        if not any(role in user.roles for role in required_roles):
                            self.audit_logger.log_authorization(user, func.__name__, False)
                            raise ForbiddenException("Insufficient permissions")
                        self.audit_logger.log_authorization(user, func.__name__, True)
                    
                    # 3. Check rate limits
                    endpoint_name = rate_limit_endpoint or func.__name__
                    if not self.rate_limiter.check_rate_limit(user.id, endpoint_name):
                        self.audit_logger.log_rate_limit(user.id, endpoint_name)
                        raise RateLimitExceededException("Rate limit exceeded")
                    
                    # 4. Validate input data
                    if hasattr(request, 'json') and request.method in ['POST', 'PUT', 'PATCH']:
                        try:
                            validated_data = self.input_validator.validate(
                                await request.json(), endpoint_name
                            )
                            request.validated_data = validated_data
                        except ValidationException as e:
                            self.audit_logger.log_validation_error(user, endpoint_name, str(e))
                            raise
                    
                    # 5. Add user to request context
                    request.user = user
                    
                    # 6. Execute function
                    result = await func(request, *args, **kwargs)
                    
                    # 7. Log successful operation
                    self.audit_logger.log_security_event(
                        "API_SUCCESS",
                        user,
                        {"endpoint": func.__name__, "method": request.method}
                    )
                    
                    return result
                    
                except SecurityException as e:
                    # Log security violations
                    self.audit_logger.log_security_event(
                        "SECURITY_VIOLATION",
                        getattr(request, 'user', None),
                        {"endpoint": func.__name__, "error": str(e), "type": type(e).__name__}
                    )
                    raise
                    
            return wrapper
        return decorator

# Usage example
if __name__ == "__main__":
    # Initialize security framework
    security = WicketWiseSecurityFramework(secret_key="your-secret-key-change-in-production")
    
    # Example secured endpoint
    @security.secure_endpoint(
        required_roles=[UserRole.ADMIN], 
        rate_limit_endpoint='enrich_matches'
    )
    async def enrich_matches_endpoint(request):
        """Secured match enrichment endpoint"""
        user = request.user
        validated_data = request.validated_data
        
        # Your enrichment logic here
        return {"status": "success", "message": f"Enrichment started by {user.username}"}
    
    # Test authentication
    token = security.auth_manager.authenticate("admin", "ChangeMe123!")
    if token:
        print(f"Authentication successful: {token[:50]}...")
    else:
        print("Authentication failed")
