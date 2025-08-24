# Purpose: JWT token handling and validation system
# Author: WicketWise AI, Last Modified: 2024

import jwt
import time
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging


class TokenValidationError(Exception):
    """JWT token validation errors"""
    pass


class TokenExpiredError(TokenValidationError):
    """JWT token expired error"""
    pass


@dataclass
class JWTConfig:
    """JWT configuration settings"""
    secret_key: str
    algorithm: str = "HS256"
    access_token_ttl_minutes: int = 60
    refresh_token_ttl_days: int = 30
    issuer: str = "wicketwise"
    audience: str = "wicketwise-api"
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.secret_key:
            raise ValueError("JWT secret key is required")
        
        if len(self.secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters")


@dataclass
class TokenClaims:
    """JWT token claims structure"""
    user_id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    token_type: str = "access"
    issued_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    issuer: Optional[str] = None
    audience: Optional[str] = None
    jti: Optional[str] = None  # JWT ID
    custom_claims: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert claims to dictionary for JWT payload"""
        claims = {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'roles': self.roles,
            'permissions': self.permissions,
            'token_type': self.token_type
        }
        
        # Add standard JWT claims
        if self.issued_at:
            claims['iat'] = int(self.issued_at.timestamp())
        
        if self.expires_at:
            claims['exp'] = int(self.expires_at.timestamp())
        
        if self.issuer:
            claims['iss'] = self.issuer
        
        if self.audience:
            claims['aud'] = self.audience
        
        if self.jti:
            claims['jti'] = self.jti
        
        # Add custom claims
        claims.update(self.custom_claims)
        
        return claims
    
    @classmethod
    def from_dict(cls, claims_dict: Dict[str, Any]) -> 'TokenClaims':
        """Create TokenClaims from dictionary"""
        # Extract standard claims
        issued_at = None
        if 'iat' in claims_dict:
            issued_at = datetime.fromtimestamp(claims_dict['iat'])
        
        expires_at = None
        if 'exp' in claims_dict:
            expires_at = datetime.fromtimestamp(claims_dict['exp'])
        
        # Extract custom claims (everything not in standard fields)
        standard_fields = {
            'user_id', 'username', 'email', 'roles', 'permissions', 'token_type',
            'iat', 'exp', 'iss', 'aud', 'jti'
        }
        custom_claims = {k: v for k, v in claims_dict.items() if k not in standard_fields}
        
        return cls(
            user_id=claims_dict['user_id'],
            username=claims_dict['username'],
            email=claims_dict['email'],
            roles=claims_dict.get('roles', []),
            permissions=claims_dict.get('permissions', []),
            token_type=claims_dict.get('token_type', 'access'),
            issued_at=issued_at,
            expires_at=expires_at,
            issuer=claims_dict.get('iss'),
            audience=claims_dict.get('aud'),
            jti=claims_dict.get('jti'),
            custom_claims=custom_claims
        )


class JWTHandler:
    """JWT token creation and validation handler"""
    
    def __init__(self, config: JWTConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Token blacklist for revoked tokens
        self.blacklisted_tokens: set = set()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate JWT configuration"""
        try:
            # Test encoding/decoding with config
            test_payload = {'test': 'data'}
            token = jwt.encode(test_payload, self.config.secret_key, algorithm=self.config.algorithm)
            jwt.decode(token, self.config.secret_key, algorithms=[self.config.algorithm])
        except Exception as e:
            raise ValueError(f"Invalid JWT configuration: {str(e)}")
    
    def create_access_token(self, user_id: str, username: str, email: str,
                           roles: List[str], permissions: List[str],
                           custom_claims: Optional[Dict[str, Any]] = None) -> str:
        """Create JWT access token"""
        now = datetime.now()
        expires_at = now + timedelta(minutes=self.config.access_token_ttl_minutes)
        
        claims = TokenClaims(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles,
            permissions=permissions,
            token_type="access",
            issued_at=now,
            expires_at=expires_at,
            issuer=self.config.issuer,
            audience=self.config.audience,
            jti=self._generate_jti(),
            custom_claims=custom_claims or {}
        )
        
        return self._encode_token(claims)
    
    def create_refresh_token(self, user_id: str, username: str, email: str,
                            custom_claims: Optional[Dict[str, Any]] = None) -> str:
        """Create JWT refresh token"""
        now = datetime.now()
        expires_at = now + timedelta(days=self.config.refresh_token_ttl_days)
        
        claims = TokenClaims(
            user_id=user_id,
            username=username,
            email=email,
            roles=[],  # Refresh tokens don't include roles/permissions
            permissions=[],
            token_type="refresh",
            issued_at=now,
            expires_at=expires_at,
            issuer=self.config.issuer,
            audience=self.config.audience,
            jti=self._generate_jti(),
            custom_claims=custom_claims or {}
        )
        
        return self._encode_token(claims)
    
    def create_custom_token(self, claims: TokenClaims, ttl_minutes: Optional[int] = None) -> str:
        """Create custom JWT token with specific claims"""
        if not claims.issued_at:
            claims.issued_at = datetime.now()
        
        if not claims.expires_at and ttl_minutes:
            claims.expires_at = claims.issued_at + timedelta(minutes=ttl_minutes)
        
        if not claims.issuer:
            claims.issuer = self.config.issuer
        
        if not claims.audience:
            claims.audience = self.config.audience
        
        if not claims.jti:
            claims.jti = self._generate_jti()
        
        return self._encode_token(claims)
    
    def _encode_token(self, claims: TokenClaims) -> str:
        """Encode token claims to JWT"""
        try:
            payload = claims.to_dict()
            token = jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)
            
            self.logger.debug(f"Created JWT token for user: {claims.user_id}")
            return token
            
        except Exception as e:
            self.logger.error(f"Failed to encode JWT token: {str(e)}")
            raise TokenValidationError(f"Token encoding failed: {str(e)}")
    
    def validate_token(self, token: str, token_type: Optional[str] = None,
                      verify_expiration: bool = True) -> TokenClaims:
        """Validate and decode JWT token"""
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                raise TokenValidationError("Token has been revoked")
            
            # Decode token
            options = {
                'verify_exp': verify_expiration,
                'verify_aud': True,
                'verify_iss': True
            }
            
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                audience=self.config.audience,
                issuer=self.config.issuer,
                options=options
            )
            
            # Convert to TokenClaims
            claims = TokenClaims.from_dict(payload)
            
            # Verify token type if specified
            if token_type and claims.token_type != token_type:
                raise TokenValidationError(f"Expected {token_type} token, got {claims.token_type}")
            
            # Check if token is expired (manual check for custom handling)
            if verify_expiration and claims.expires_at and datetime.now() > claims.expires_at:
                raise TokenExpiredError("Token has expired")
            
            self.logger.debug(f"Validated JWT token for user: {claims.user_id}")
            return claims
            
        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise TokenValidationError(f"Invalid token: {str(e)}")
        except Exception as e:
            self.logger.error(f"Token validation error: {str(e)}")
            raise TokenValidationError(f"Token validation failed: {str(e)}")
    
    def refresh_access_token(self, refresh_token: str, roles: List[str],
                            permissions: List[str]) -> Dict[str, str]:
        """Create new access token from refresh token"""
        try:
            # Validate refresh token
            refresh_claims = self.validate_token(refresh_token, token_type="refresh")
            
            # Create new access token
            access_token = self.create_access_token(
                user_id=refresh_claims.user_id,
                username=refresh_claims.username,
                email=refresh_claims.email,
                roles=roles,
                permissions=permissions,
                custom_claims=refresh_claims.custom_claims
            )
            
            return {
                'access_token': access_token,
                'token_type': 'Bearer',
                'expires_in': self.config.access_token_ttl_minutes * 60
            }
            
        except Exception as e:
            self.logger.error(f"Token refresh failed: {str(e)}")
            raise TokenValidationError(f"Token refresh failed: {str(e)}")
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a JWT token by adding to blacklist"""
        try:
            # Validate token first to ensure it's legitimate
            claims = self.validate_token(token, verify_expiration=False)
            
            # Add to blacklist
            self.blacklisted_tokens.add(token)
            
            self.logger.info(f"Revoked JWT token for user: {claims.user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Token revocation failed: {str(e)}")
            return False
    
    def revoke_tokens_by_jti(self, jti_list: List[str]) -> int:
        """Revoke multiple tokens by their JTI (JWT ID)"""
        revoked_count = 0
        
        # Note: This is a simplified implementation
        # In production, you'd need to maintain a mapping of JTI to tokens
        # or use a more sophisticated blacklist system
        
        for jti in jti_list:
            # Add JTI to blacklist (would need token lookup in real implementation)
            self.blacklisted_tokens.add(jti)
            revoked_count += 1
        
        self.logger.info(f"Revoked {revoked_count} tokens by JTI")
        return revoked_count
    
    def extract_claims_without_validation(self, token: str) -> Optional[TokenClaims]:
        """Extract claims from token without validation (for debugging/logging)"""
        try:
            # Decode without verification
            payload = jwt.decode(token, options={"verify_signature": False})
            return TokenClaims.from_dict(payload)
        except Exception as e:
            self.logger.error(f"Failed to extract claims: {str(e)}")
            return None
    
    def _generate_jti(self) -> str:
        """Generate unique JWT ID"""
        timestamp = str(int(time.time() * 1000))
        return f"jwt_{timestamp}_{secrets.token_hex(8)}"
    
    def get_token_info(self, token: str) -> Dict[str, Any]:
        """Get information about a token without full validation"""
        try:
            claims = self.extract_claims_without_validation(token)
            if not claims:
                return {'error': 'Invalid token format'}
            
            is_blacklisted = token in self.blacklisted_tokens
            is_expired = claims.expires_at and datetime.now() > claims.expires_at
            
            return {
                'user_id': claims.user_id,
                'username': claims.username,
                'token_type': claims.token_type,
                'issued_at': claims.issued_at.isoformat() if claims.issued_at else None,
                'expires_at': claims.expires_at.isoformat() if claims.expires_at else None,
                'is_expired': is_expired,
                'is_blacklisted': is_blacklisted,
                'jti': claims.jti
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def cleanup_expired_blacklist(self) -> int:
        """Remove expired tokens from blacklist"""
        # Note: This is a simplified implementation
        # In production, you'd need to track token expiration times
        # and remove expired tokens from the blacklist
        
        initial_count = len(self.blacklisted_tokens)
        
        # For now, just clear all (in production, check expiration times)
        # self.blacklisted_tokens.clear()
        
        # Return 0 for now (would return actual cleanup count in production)
        return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get JWT handler statistics"""
        return {
            'blacklisted_tokens': len(self.blacklisted_tokens),
            'access_token_ttl_minutes': self.config.access_token_ttl_minutes,
            'refresh_token_ttl_days': self.config.refresh_token_ttl_days,
            'algorithm': self.config.algorithm,
            'issuer': self.config.issuer,
            'audience': self.config.audience
        }
