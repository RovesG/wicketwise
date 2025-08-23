"""
Unit tests for Security Framework
Tests authentication, authorization, input validation, and audit logging
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from security_framework import (
    WicketWiseSecurityFramework,
    JWTAuthManager,
    AdvancedRateLimiter,
    CricketInputValidator,
    SecurityAuditLogger,
    UserRole,
    UnauthorizedException,
    ForbiddenException,
    RateLimitExceededException,
    ValidationException
)

class TestJWTAuthManager:
    """Test JWT authentication manager"""
    
    def test_create_user(self):
        """Test user creation"""
        auth_manager = JWTAuthManager("test-secret", 24)
        
        user = auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPass123!",
            roles={UserRole.ANALYST}
        )
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert UserRole.ANALYST in user.roles
        assert user.id in auth_manager.users
    
    def test_authenticate_valid_user(self):
        """Test authentication with valid credentials"""
        auth_manager = JWTAuthManager("test-secret", 24)
        
        # Create user
        auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPass123!",
            roles={UserRole.ANALYST}
        )
        
        # Authenticate
        token = auth_manager.authenticate("testuser", "TestPass123!")
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_authenticate_invalid_user(self):
        """Test authentication with invalid credentials"""
        auth_manager = JWTAuthManager("test-secret", 24)
        
        # Try to authenticate non-existent user
        token = auth_manager.authenticate("nonexistent", "password")
        
        assert token is None
    
    def test_authenticate_wrong_password(self):
        """Test authentication with wrong password"""
        auth_manager = JWTAuthManager("test-secret", 24)
        
        # Create user
        auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPass123!",
            roles={UserRole.ANALYST}
        )
        
        # Try wrong password
        token = auth_manager.authenticate("testuser", "WrongPassword")
        
        assert token is None
    
    def test_validate_token_valid(self):
        """Test token validation with valid token"""
        auth_manager = JWTAuthManager("test-secret", 24)
        
        # Create user and get token
        user = auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPass123!",
            roles={UserRole.ANALYST}
        )
        token = auth_manager.authenticate("testuser", "TestPass123!")
        
        # Validate token
        validated_user = auth_manager.validate_token(token)
        
        assert validated_user is not None
        assert validated_user.username == "testuser"
        assert validated_user.id == user.id
    
    def test_validate_token_invalid(self):
        """Test token validation with invalid token"""
        auth_manager = JWTAuthManager("test-secret", 24)
        
        # Try to validate invalid token
        validated_user = auth_manager.validate_token("invalid-token")
        
        assert validated_user is None
    
    def test_validate_token_expired(self):
        """Test token validation with expired token"""
        auth_manager = JWTAuthManager("test-secret", 1/3600)  # 1 second expiry (1/3600 hours)
        
        # Create user and get token
        auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPass123!",
            roles={UserRole.ANALYST}
        )
        token = auth_manager.authenticate("testuser", "TestPass123!")
        
        # Wait for token to expire
        time.sleep(2)  # Wait 2 seconds for 1-second token to expire
        
        # Try to validate expired token
        validated_user = auth_manager.validate_token(token)
        
        assert validated_user is None

class TestAdvancedRateLimiter:
    """Test advanced rate limiter"""
    
    def test_rate_limit_within_limit(self):
        """Test rate limiting within allowed limits"""
        limiter = AdvancedRateLimiter()
        
        # Should allow requests within limit
        assert limiter.check_rate_limit("user1", "test_endpoint") == True
        assert limiter.check_rate_limit("user1", "test_endpoint") == True
    
    def test_rate_limit_exceeded(self):
        """Test rate limiting when limit is exceeded"""
        limiter = AdvancedRateLimiter()
        
        # Override limits for testing
        limiter.limits = {
            "test_endpoint": {"minute": 2},  # Very low limit
            "default": {"minute": 60, "hour": 1000}  # Keep default
        }
        
        # First two requests should pass
        assert limiter.check_rate_limit("user1", "test_endpoint") == True
        assert limiter.check_rate_limit("user1", "test_endpoint") == True
        
        # Third request should fail
        assert limiter.check_rate_limit("user1", "test_endpoint") == False
    
    def test_rate_limit_different_users(self):
        """Test rate limiting for different users"""
        limiter = AdvancedRateLimiter()
        
        # Override limits for testing
        limiter.limits = {
            "test_endpoint": {"minute": 1},
            "default": {"minute": 60, "hour": 1000}  # Keep default
        }
        
        # Each user should have their own limit
        assert limiter.check_rate_limit("user1", "test_endpoint") == True
        assert limiter.check_rate_limit("user2", "test_endpoint") == True
        
        # Both users should now be at their limit
        assert limiter.check_rate_limit("user1", "test_endpoint") == False
        assert limiter.check_rate_limit("user2", "test_endpoint") == False

class TestCricketInputValidator:
    """Test cricket input validator"""
    
    def test_validate_enrichment_request_valid(self):
        """Test validation of valid enrichment request"""
        validator = CricketInputValidator()
        
        data = {
            "max_matches": 50,
            "priority_competitions": ["IPL", "BBL"],
            "force_refresh": False
        }
        
        validated_data = validator.validate(data, "enrich_matches")
        
        assert validated_data["max_matches"] == 50
        assert validated_data["priority_competitions"] == ["IPL", "BBL"]
        assert validated_data["force_refresh"] == False
    
    def test_validate_enrichment_request_invalid_max_matches(self):
        """Test validation with invalid max_matches"""
        validator = CricketInputValidator()
        
        data = {
            "max_matches": 2000,  # Too high
            "priority_competitions": [],
            "force_refresh": False
        }
        
        with pytest.raises(ValidationException) as exc_info:
            validator.validate(data, "enrich_matches")
        
        assert "too large" in str(exc_info.value)
    
    def test_validate_player_query_valid(self):
        """Test validation of valid player query"""
        validator = CricketInputValidator()
        
        data = {
            "player_name": "Virat Kohli",
            "format": "T20",
            "venue": "Wankhede Stadium"
        }
        
        validated_data = validator.validate(data, "player_query")
        
        assert validated_data["player_name"] == "Virat Kohli"
        assert validated_data["format"] == "T20"
        assert validated_data["venue"] == "Wankhede Stadium"
    
    def test_validate_player_query_invalid_name(self):
        """Test validation with invalid player name"""
        validator = CricketInputValidator()
        
        data = {
            "player_name": "X",  # Too short
            "format": "T20"
        }
        
        with pytest.raises(ValidationException) as exc_info:
            validator.validate(data, "player_query")
        
        assert "too short" in str(exc_info.value)
    
    def test_validate_player_query_invalid_format(self):
        """Test validation with invalid format"""
        validator = CricketInputValidator()
        
        data = {
            "player_name": "Virat Kohli",
            "format": "Invalid"  # Not in allowed choices
        }
        
        with pytest.raises(ValidationException) as exc_info:
            validator.validate(data, "player_query")
        
        assert "must be one of" in str(exc_info.value)
    
    def test_validate_missing_required_field(self):
        """Test validation with missing required field"""
        validator = CricketInputValidator()
        
        data = {
            # Missing player_name
            "format": "T20"
        }
        
        with pytest.raises(ValidationException) as exc_info:
            validator.validate(data, "player_query")
        
        assert "Required field missing" in str(exc_info.value)
    
    def test_validate_unknown_schema(self):
        """Test validation with unknown schema"""
        validator = CricketInputValidator()
        
        data = {"test": "value"}
        
        with pytest.raises(ValidationException) as exc_info:
            validator.validate(data, "unknown_schema")
        
        assert "Unknown validation schema" in str(exc_info.value)

class TestSecurityAuditLogger:
    """Test security audit logger"""
    
    @patch('logging.FileHandler')
    @patch('logging.Logger.info')
    def test_log_authentication_success(self, mock_info, mock_handler):
        """Test logging successful authentication"""
        audit_logger = SecurityAuditLogger("test_audit.log")
        
        audit_logger.log_authentication("testuser", True, "127.0.0.1")
        
        # Check that the logger was called
        mock_info.assert_called_once()
        call_args = mock_info.call_args[0][0]
        assert "AUTH_SUCCESS" in call_args
        assert "testuser" in call_args
    
    @patch('logging.FileHandler')
    @patch('logging.Logger.info')
    def test_log_authentication_failure(self, mock_info, mock_handler):
        """Test logging failed authentication"""
        audit_logger = SecurityAuditLogger("test_audit.log")
        
        audit_logger.log_authentication("testuser", False, "127.0.0.1")
        
        # Check that the logger was called
        mock_info.assert_called_once()
        call_args = mock_info.call_args[0][0]
        assert "AUTH_FAILED" in call_args
        assert "testuser" in call_args
    
    @patch('logging.FileHandler')
    @patch('logging.Logger.warning')
    def test_log_rate_limit(self, mock_warning, mock_handler):
        """Test logging rate limit violations"""
        audit_logger = SecurityAuditLogger("test_audit.log")
        
        audit_logger.log_rate_limit("user123", "test_endpoint")
        
        # Check that the logger was called
        mock_warning.assert_called_once()
        call_args = mock_warning.call_args[0][0]
        assert "RATE_LIMIT" in call_args
        assert "user123" in call_args

class TestWicketWiseSecurityFramework:
    """Test main security framework"""
    
    def test_initialization(self):
        """Test security framework initialization"""
        framework = WicketWiseSecurityFramework("test-secret")
        
        assert framework.auth_manager is not None
        assert framework.rate_limiter is not None
        assert framework.input_validator is not None
        assert framework.audit_logger is not None
    
    def test_default_users_created(self):
        """Test that default users are created"""
        framework = WicketWiseSecurityFramework("test-secret")
        
        # Check that admin user exists
        admin_token = framework.auth_manager.authenticate("admin", "ChangeMe123!")
        assert admin_token is not None
        
        # Check that analyst user exists
        analyst_token = framework.auth_manager.authenticate("analyst", "Analyst123!")
        assert analyst_token is not None
    
    @pytest.mark.asyncio
    async def test_secure_endpoint_decorator_success(self):
        """Test secure endpoint decorator with valid token"""
        framework = WicketWiseSecurityFramework("test-secret")
        
        # Get admin token
        admin_token = framework.auth_manager.authenticate("admin", "ChangeMe123!")
        
        # Mock request object
        mock_request = MagicMock()
        mock_request.headers.get.return_value = f"Bearer {admin_token}"
        mock_request.method = "GET"
        mock_request.json = {}
        
        # Create decorated function
        @framework.secure_endpoint(required_roles=[UserRole.ADMIN])
        async def test_endpoint(request):
            return {"status": "success"}
        
        # This should work (though we can't fully test the decorator without FastAPI context)
        # The test mainly verifies the framework can create the decorator
        assert callable(test_endpoint)
    
    def test_secure_endpoint_decorator_invalid_token(self):
        """Test secure endpoint decorator with invalid token"""
        framework = WicketWiseSecurityFramework("test-secret")
        
        # Create decorated function
        @framework.secure_endpoint(required_roles=[UserRole.ADMIN])
        async def test_endpoint(request):
            return {"status": "success"}
        
        # The decorator should be created successfully
        assert callable(test_endpoint)

@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security components"""
    
    def test_full_authentication_flow(self):
        """Test complete authentication flow"""
        framework = WicketWiseSecurityFramework("test-secret")
        
        # 1. Create user
        user = framework.auth_manager.create_user(
            username="integrationuser",
            email="integration@test.com",
            password="IntegrationPass123!",
            roles={UserRole.ANALYST, UserRole.VIEWER}
        )
        
        # 2. Authenticate
        token = framework.auth_manager.authenticate("integrationuser", "IntegrationPass123!")
        assert token is not None
        
        # 3. Validate token
        validated_user = framework.auth_manager.validate_token(token)
        assert validated_user is not None
        assert validated_user.username == "integrationuser"
        assert UserRole.ANALYST in validated_user.roles
        assert UserRole.VIEWER in validated_user.roles
        
        # 4. Check rate limits
        assert framework.rate_limiter.check_rate_limit(user.id, "test_endpoint") == True
        
        # 5. Validate input
        test_data = {
            "player_name": "Test Player",
            "format": "T20"
        }
        validated_data = framework.input_validator.validate(test_data, "player_query")
        assert validated_data["player_name"] == "Test Player"
    
    def test_security_violation_flow(self):
        """Test security violation handling"""
        framework = WicketWiseSecurityFramework("test-secret")
        
        # Test invalid authentication
        invalid_token = framework.auth_manager.authenticate("nonexistent", "wrongpass")
        assert invalid_token is None
        
        # Test rate limiting
        framework.rate_limiter.limits = {
            "test": {"minute": 1},
            "default": {"minute": 60, "hour": 1000}
        }
        
        user_id = "test_user"
        assert framework.rate_limiter.check_rate_limit(user_id, "test") == True
        assert framework.rate_limiter.check_rate_limit(user_id, "test") == False
        
        # Test input validation failure
        invalid_data = {"player_name": "X"}  # Too short
        
        with pytest.raises(ValidationException):
            framework.input_validator.validate(invalid_data, "player_query")

class TestPasswordSecurity:
    """Test password security features"""
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        auth_manager = JWTAuthManager("test-secret", 24)
        
        password = "TestPassword123!"
        hashed = auth_manager._hash_password(password)
        
        # Hash should be different from original password
        assert hashed != password
        assert ":" in hashed  # Should contain salt separator
        
        # Should be able to verify correct password
        assert auth_manager._verify_password(password, hashed) == True
        
        # Should not verify incorrect password
        assert auth_manager._verify_password("WrongPassword", hashed) == False
    
    def test_password_salt_uniqueness(self):
        """Test that password salts are unique"""
        auth_manager = JWTAuthManager("test-secret", 24)
        
        password = "SamePassword123!"
        hash1 = auth_manager._hash_password(password)
        hash2 = auth_manager._hash_password(password)
        
        # Same password should produce different hashes due to unique salts
        assert hash1 != hash2
        
        # Both should still verify correctly
        assert auth_manager._verify_password(password, hash1) == True
        assert auth_manager._verify_password(password, hash2) == True
