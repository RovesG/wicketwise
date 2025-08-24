# Purpose: Unit tests for JWT token handling and validation
# Author: WicketWise AI, Last Modified: 2024

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from crickformers.security.jwt_handler import (
    JWTHandler,
    JWTConfig,
    TokenClaims,
    TokenValidationError,
    TokenExpiredError
)


class TestJWTConfig:
    """Test suite for JWTConfig"""
    
    def test_config_creation_valid(self):
        """Test valid JWT configuration creation"""
        config = JWTConfig(
            secret_key="this_is_a_very_long_secret_key_for_testing_purposes_123456789",
            algorithm="HS256",
            access_token_ttl_minutes=60,
            refresh_token_ttl_days=30,
            issuer="test_issuer",
            audience="test_audience"
        )
        
        assert config.secret_key == "this_is_a_very_long_secret_key_for_testing_purposes_123456789"
        assert config.algorithm == "HS256"
        assert config.access_token_ttl_minutes == 60
        assert config.refresh_token_ttl_days == 30
        assert config.issuer == "test_issuer"
        assert config.audience == "test_audience"
    
    def test_config_creation_invalid_secret(self):
        """Test JWT configuration with invalid secret key"""
        # Empty secret key
        with pytest.raises(ValueError, match="JWT secret key is required"):
            JWTConfig(secret_key="")
        
        # Short secret key
        with pytest.raises(ValueError, match="at least 32 characters"):
            JWTConfig(secret_key="short_key")
    
    def test_config_defaults(self):
        """Test JWT configuration with default values"""
        config = JWTConfig(secret_key="this_is_a_very_long_secret_key_for_testing_purposes_123456789")
        
        assert config.algorithm == "HS256"
        assert config.access_token_ttl_minutes == 60
        assert config.refresh_token_ttl_days == 30
        assert config.issuer == "wicketwise"
        assert config.audience == "wicketwise-api"


class TestTokenClaims:
    """Test suite for TokenClaims"""
    
    def test_claims_creation(self):
        """Test TokenClaims creation"""
        now = datetime.now()
        expires = now + timedelta(hours=1)
        
        claims = TokenClaims(
            user_id="user_123",
            username="testuser",
            email="test@example.com",
            roles=["admin", "user"],
            permissions=["read", "write"],
            token_type="access",
            issued_at=now,
            expires_at=expires,
            issuer="test_issuer",
            audience="test_audience",
            jti="jwt_123",
            custom_claims={"client_id": "web_app"}
        )
        
        assert claims.user_id == "user_123"
        assert claims.username == "testuser"
        assert claims.email == "test@example.com"
        assert claims.roles == ["admin", "user"]
        assert claims.permissions == ["read", "write"]
        assert claims.token_type == "access"
        assert claims.issued_at == now
        assert claims.expires_at == expires
        assert claims.issuer == "test_issuer"
        assert claims.audience == "test_audience"
        assert claims.jti == "jwt_123"
        assert claims.custom_claims == {"client_id": "web_app"}
    
    def test_claims_to_dict(self):
        """Test TokenClaims to dictionary conversion"""
        now = datetime.now()
        expires = now + timedelta(hours=1)
        
        claims = TokenClaims(
            user_id="user_456",
            username="dictuser",
            email="dict@example.com",
            roles=["viewer"],
            permissions=["read"],
            token_type="refresh",
            issued_at=now,
            expires_at=expires,
            issuer="dict_issuer",
            audience="dict_audience",
            jti="jwt_456",
            custom_claims={"session_id": "sess_123"}
        )
        
        claims_dict = claims.to_dict()
        
        assert claims_dict['user_id'] == "user_456"
        assert claims_dict['username'] == "dictuser"
        assert claims_dict['email'] == "dict@example.com"
        assert claims_dict['roles'] == ["viewer"]
        assert claims_dict['permissions'] == ["read"]
        assert claims_dict['token_type'] == "refresh"
        assert claims_dict['iat'] == int(now.timestamp())
        assert claims_dict['exp'] == int(expires.timestamp())
        assert claims_dict['iss'] == "dict_issuer"
        assert claims_dict['aud'] == "dict_audience"
        assert claims_dict['jti'] == "jwt_456"
        assert claims_dict['session_id'] == "sess_123"
    
    def test_claims_from_dict(self):
        """Test TokenClaims from dictionary creation"""
        now = datetime.now()
        expires = now + timedelta(hours=1)
        
        claims_dict = {
            'user_id': "user_789",
            'username': "fromdict",
            'email': "fromdict@example.com",
            'roles': ["analyst"],
            'permissions': ["read", "execute"],
            'token_type': "access",
            'iat': int(now.timestamp()),
            'exp': int(expires.timestamp()),
            'iss': "fromdict_issuer",
            'aud': "fromdict_audience",
            'jti': "jwt_789",
            'custom_field': "custom_value"
        }
        
        claims = TokenClaims.from_dict(claims_dict)
        
        assert claims.user_id == "user_789"
        assert claims.username == "fromdict"
        assert claims.email == "fromdict@example.com"
        assert claims.roles == ["analyst"]
        assert claims.permissions == ["read", "execute"]
        assert claims.token_type == "access"
        assert abs((claims.issued_at - now).total_seconds()) < 1
        assert abs((claims.expires_at - expires).total_seconds()) < 1
        assert claims.issuer == "fromdict_issuer"
        assert claims.audience == "fromdict_audience"
        assert claims.jti == "jwt_789"
        assert claims.custom_claims == {"custom_field": "custom_value"}


class TestJWTHandler:
    """Test suite for JWTHandler"""
    
    @pytest.fixture
    def jwt_config(self):
        """Create JWT configuration for testing"""
        return JWTConfig(
            secret_key="test_secret_key_that_is_long_enough_for_security_requirements_123456789",
            algorithm="HS256",
            access_token_ttl_minutes=60,
            refresh_token_ttl_days=7,
            issuer="test_wicketwise",
            audience="test_api"
        )
    
    @pytest.fixture
    def jwt_handler(self, jwt_config):
        """Create JWT handler for testing"""
        return JWTHandler(jwt_config)
    
    def test_handler_initialization(self, jwt_handler):
        """Test JWT handler initialization"""
        assert jwt_handler.config is not None
        assert len(jwt_handler.blacklisted_tokens) == 0
    
    def test_handler_invalid_config(self):
        """Test JWT handler with invalid configuration"""
        invalid_config = JWTConfig(
            secret_key="this_is_a_very_long_secret_key_for_testing_purposes_but_invalid_algorithm",
            algorithm="INVALID_ALG"
        )
        
        with pytest.raises(ValueError, match="Invalid JWT configuration"):
            JWTHandler(invalid_config)
    
    def test_create_access_token(self, jwt_handler):
        """Test access token creation"""
        token = jwt_handler.create_access_token(
            user_id="user_123",
            username="testuser",
            email="test@example.com",
            roles=["admin", "user"],
            permissions=["read", "write", "delete"],
            custom_claims={"client_id": "web_client"}
        )
        
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are reasonably long
        
        # Verify token can be decoded
        claims = jwt_handler.validate_token(token, token_type="access")
        assert claims.user_id == "user_123"
        assert claims.username == "testuser"
        assert claims.email == "test@example.com"
        assert claims.roles == ["admin", "user"]
        assert claims.permissions == ["read", "write", "delete"]
        assert claims.token_type == "access"
        assert claims.custom_claims == {"client_id": "web_client"}
    
    def test_create_refresh_token(self, jwt_handler):
        """Test refresh token creation"""
        token = jwt_handler.create_refresh_token(
            user_id="user_456",
            username="refreshuser",
            email="refresh@example.com",
            custom_claims={"device_id": "mobile_123"}
        )
        
        assert isinstance(token, str)
        
        # Verify token can be decoded
        claims = jwt_handler.validate_token(token, token_type="refresh")
        assert claims.user_id == "user_456"
        assert claims.username == "refreshuser"
        assert claims.email == "refresh@example.com"
        assert claims.roles == []  # Refresh tokens don't include roles
        assert claims.permissions == []  # Refresh tokens don't include permissions
        assert claims.token_type == "refresh"
        assert claims.custom_claims == {"device_id": "mobile_123"}
    
    def test_create_custom_token(self, jwt_handler):
        """Test custom token creation"""
        claims = TokenClaims(
            user_id="user_789",
            username="customuser",
            email="custom@example.com",
            roles=["custom_role"],
            permissions=["custom_permission"],
            token_type="custom",
            custom_claims={"purpose": "password_reset"}
        )
        
        token = jwt_handler.create_custom_token(claims, ttl_minutes=15)
        
        assert isinstance(token, str)
        
        # Verify token
        validated_claims = jwt_handler.validate_token(token, token_type="custom")
        assert validated_claims.user_id == "user_789"
        assert validated_claims.token_type == "custom"
        assert validated_claims.custom_claims == {"purpose": "password_reset"}
        assert validated_claims.expires_at is not None
    
    def test_validate_token_success(self, jwt_handler):
        """Test successful token validation"""
        # Create token
        token = jwt_handler.create_access_token(
            user_id="validate_user",
            username="validateuser",
            email="validate@example.com",
            roles=["user"],
            permissions=["read"]
        )
        
        # Validate token
        claims = jwt_handler.validate_token(token)
        assert claims.user_id == "validate_user"
        assert claims.username == "validateuser"
        assert claims.token_type == "access"
    
    def test_validate_token_wrong_type(self, jwt_handler):
        """Test token validation with wrong type"""
        # Create access token
        token = jwt_handler.create_access_token(
            user_id="type_user",
            username="typeuser",
            email="type@example.com",
            roles=["user"],
            permissions=["read"]
        )
        
        # Try to validate as refresh token
        with pytest.raises(TokenValidationError, match="Expected refresh token"):
            jwt_handler.validate_token(token, token_type="refresh")
    
    def test_validate_token_invalid(self, jwt_handler):
        """Test validation of invalid token"""
        with pytest.raises(TokenValidationError, match="Invalid token"):
            jwt_handler.validate_token("invalid.token.here")
    
    def test_validate_token_expired(self, jwt_handler):
        """Test validation of expired token"""
        # Create token with very short TTL
        claims = TokenClaims(
            user_id="expired_user",
            username="expireduser",
            email="expired@example.com",
            roles=["user"],
            permissions=["read"],
            token_type="access",
            issued_at=datetime.now(),
            expires_at=datetime.now() - timedelta(seconds=1)  # Already expired
        )
        
        token = jwt_handler.create_custom_token(claims)
        
        with pytest.raises(TokenExpiredError, match="Token has expired"):
            jwt_handler.validate_token(token)
    
    def test_validate_token_blacklisted(self, jwt_handler):
        """Test validation of blacklisted token"""
        # Create token
        token = jwt_handler.create_access_token(
            user_id="blacklist_user",
            username="blacklistuser",
            email="blacklist@example.com",
            roles=["user"],
            permissions=["read"]
        )
        
        # Blacklist token
        jwt_handler.blacklisted_tokens.add(token)
        
        # Validation should fail
        with pytest.raises(TokenValidationError, match="Token has been revoked"):
            jwt_handler.validate_token(token)
    
    def test_refresh_access_token(self, jwt_handler):
        """Test refreshing access token from refresh token"""
        # Create refresh token
        refresh_token = jwt_handler.create_refresh_token(
            user_id="refresh_user",
            username="refreshuser",
            email="refresh@example.com"
        )
        
        # Refresh access token
        result = jwt_handler.refresh_access_token(
            refresh_token=refresh_token,
            roles=["admin"],
            permissions=["read", "write"]
        )
        
        assert 'access_token' in result
        assert 'token_type' in result
        assert 'expires_in' in result
        assert result['token_type'] == 'Bearer'
        
        # Validate new access token
        new_claims = jwt_handler.validate_token(result['access_token'], token_type="access")
        assert new_claims.user_id == "refresh_user"
        assert new_claims.roles == ["admin"]
        assert new_claims.permissions == ["read", "write"]
    
    def test_refresh_with_invalid_token(self, jwt_handler):
        """Test refresh with invalid refresh token"""
        with pytest.raises(TokenValidationError):
            jwt_handler.refresh_access_token(
                refresh_token="invalid_refresh_token",
                roles=["user"],
                permissions=["read"]
            )
    
    def test_revoke_token(self, jwt_handler):
        """Test token revocation"""
        # Create token
        token = jwt_handler.create_access_token(
            user_id="revoke_user",
            username="revokeuser",
            email="revoke@example.com",
            roles=["user"],
            permissions=["read"]
        )
        
        # Revoke token
        success = jwt_handler.revoke_token(token)
        assert success is True
        assert token in jwt_handler.blacklisted_tokens
        
        # Token should not validate
        with pytest.raises(TokenValidationError, match="Token has been revoked"):
            jwt_handler.validate_token(token)
    
    def test_revoke_invalid_token(self, jwt_handler):
        """Test revoking invalid token"""
        success = jwt_handler.revoke_token("invalid_token")
        assert success is False
    
    def test_extract_claims_without_validation(self, jwt_handler):
        """Test extracting claims without validation"""
        # Create token
        token = jwt_handler.create_access_token(
            user_id="extract_user",
            username="extractuser",
            email="extract@example.com",
            roles=["user"],
            permissions=["read"]
        )
        
        # Extract claims without validation
        claims = jwt_handler.extract_claims_without_validation(token)
        assert claims is not None
        assert claims.user_id == "extract_user"
        assert claims.username == "extractuser"
    
    def test_get_token_info(self, jwt_handler):
        """Test getting token information"""
        # Create token
        token = jwt_handler.create_access_token(
            user_id="info_user",
            username="infouser",
            email="info@example.com",
            roles=["user"],
            permissions=["read"]
        )
        
        # Get token info
        info = jwt_handler.get_token_info(token)
        
        assert 'user_id' in info
        assert 'username' in info
        assert 'token_type' in info
        assert 'issued_at' in info
        assert 'expires_at' in info
        assert 'is_expired' in info
        assert 'is_blacklisted' in info
        assert 'jti' in info
        
        assert info['user_id'] == "info_user"
        assert info['username'] == "infouser"
        assert info['token_type'] == "access"
        assert info['is_expired'] is False
        assert info['is_blacklisted'] is False
    
    def test_get_statistics(self, jwt_handler):
        """Test getting JWT handler statistics"""
        # Create some tokens and blacklist one
        token1 = jwt_handler.create_access_token("user1", "user1", "user1@example.com", [], [])
        token2 = jwt_handler.create_refresh_token("user2", "user2", "user2@example.com")
        
        jwt_handler.revoke_token(token1)
        
        stats = jwt_handler.get_statistics()
        
        assert 'blacklisted_tokens' in stats
        assert 'access_token_ttl_minutes' in stats
        assert 'refresh_token_ttl_days' in stats
        assert 'algorithm' in stats
        assert 'issuer' in stats
        assert 'audience' in stats
        
        assert stats['blacklisted_tokens'] == 1
        assert stats['access_token_ttl_minutes'] == 60
        assert stats['algorithm'] == "HS256"


def run_jwt_handler_tests():
    """Run all JWT handler tests"""
    print("🔑 Running JWT Handler Tests")
    print("=" * 50)
    
    # Test categories
    test_categories = [
        ("JWT Config", TestJWTConfig),
        ("Token Claims", TestTokenClaims),
        ("JWT Handler", TestJWTHandler)
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
                    
                    if 'jwt_handler' in sig.parameters:
                        jwt_config = JWTConfig(
                            secret_key="test_secret_key_that_is_long_enough_for_security_requirements_123456789",
                            algorithm="HS256",
                            access_token_ttl_minutes=60,
                            refresh_token_ttl_days=7,
                            issuer="test_wicketwise",
                            audience="test_api"
                        )
                        jwt_handler = JWTHandler(jwt_config)
                        
                        if 'jwt_config' in sig.parameters:
                            method(jwt_config, jwt_handler)
                        else:
                            method(jwt_handler)
                    elif 'jwt_config' in sig.parameters:
                        jwt_config = JWTConfig(
                            secret_key="test_secret_key_that_is_long_enough_for_security_requirements_123456789"
                        )
                        method(jwt_config)
                    else:
                        method()
                    
                    print(f"  ✅ {test_method}")
                    passed_tests += 1
                    category_passed += 1
                    
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
        
        print(f"  📈 Category Results: {category_passed}/{len(test_methods)} passed")
    
    print(f"\n🏆 Overall JWT Handler Test Results: {passed_tests}/{total_tests} passed")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_jwt_handler_tests()
    exit(0 if success else 1)
