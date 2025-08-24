# Purpose: Unit tests for authentication and authorization management
# Author: WicketWise AI, Last Modified: 2024

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from crickformers.security.auth_manager import (
    AuthManager,
    AuthUser,
    AuthToken,
    AuthRole,
    AuthPermission,
    TokenType,
    AuthenticationError,
    AuthorizationError
)


class TestAuthUser:
    """Test suite for AuthUser data structure"""
    
    def test_user_creation(self):
        """Test AuthUser creation and basic properties"""
        user = AuthUser(
            user_id="user_001",
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
            roles={AuthRole.ANALYST},
            permissions={AuthPermission.READ_DATA},
            is_active=True,
            is_verified=True,
            mfa_enabled=False
        )
        
        assert user.user_id == "user_001"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.password_hash == "hashed_password"
        assert AuthRole.ANALYST in user.roles
        assert AuthPermission.READ_DATA in user.permissions
        assert user.is_active is True
        assert user.is_verified is True
        assert user.mfa_enabled is False
        assert user.failed_login_attempts == 0
        assert user.locked_until is None
    
    def test_user_has_role(self):
        """Test user role checking"""
        user = AuthUser(
            user_id="user_002",
            username="analyst",
            email="analyst@example.com",
            password_hash="hash",
            roles={AuthRole.ANALYST, AuthRole.VIEWER}
        )
        
        assert user.has_role(AuthRole.ANALYST) is True
        assert user.has_role(AuthRole.VIEWER) is True
        assert user.has_role(AuthRole.ADMIN) is False
        assert user.has_role(AuthRole.SUPER_ADMIN) is False
    
    def test_user_has_permission_direct(self):
        """Test user permission checking (direct permissions)"""
        user = AuthUser(
            user_id="user_003",
            username="testuser",
            email="test@example.com",
            password_hash="hash",
            permissions={AuthPermission.READ_DATA, AuthPermission.WRITE_DATA}
        )
        
        assert user.has_permission(AuthPermission.READ_DATA) is True
        assert user.has_permission(AuthPermission.WRITE_DATA) is True
        assert user.has_permission(AuthPermission.DELETE_DATA) is False
    
    def test_user_has_permission_role_based(self):
        """Test user permission checking (role-based permissions)"""
        user = AuthUser(
            user_id="user_004",
            username="admin",
            email="admin@example.com",
            password_hash="hash",
            roles={AuthRole.ADMIN}
        )
        
        # Admin should have many permissions
        assert user.has_permission(AuthPermission.READ_DATA) is True
        assert user.has_permission(AuthPermission.WRITE_DATA) is True
        assert user.has_permission(AuthPermission.VIEW_MODELS) is True
        assert user.has_permission(AuthPermission.MANAGE_USERS) is True
    
    def test_user_super_admin_permissions(self):
        """Test super admin has all permissions"""
        user = AuthUser(
            user_id="user_005",
            username="superadmin",
            email="superadmin@example.com",
            password_hash="hash",
            roles={AuthRole.SUPER_ADMIN}
        )
        
        # Super admin should have all permissions
        for permission in AuthPermission:
            assert user.has_permission(permission) is True
    
    def test_user_is_locked(self):
        """Test user account locking"""
        user = AuthUser(
            user_id="user_006",
            username="lockeduser",
            email="locked@example.com",
            password_hash="hash"
        )
        
        # User not locked initially
        assert user.is_locked() is False
        
        # Lock user for 30 minutes
        user.locked_until = datetime.now() + timedelta(minutes=30)
        assert user.is_locked() is True
        
        # User unlocked after time passes
        user.locked_until = datetime.now() - timedelta(minutes=1)
        assert user.is_locked() is False
    
    def test_user_to_dict(self):
        """Test user dictionary conversion"""
        user = AuthUser(
            user_id="user_007",
            username="dictuser",
            email="dict@example.com",
            password_hash="secret_hash",
            roles={AuthRole.VIEWER},
            permissions={AuthPermission.READ_DATA},
            mfa_enabled=True
        )
        
        # Test without sensitive data
        user_dict = user.to_dict(include_sensitive=False)
        assert user_dict['user_id'] == "user_007"
        assert user_dict['username'] == "dictuser"
        assert user_dict['email'] == "dict@example.com"
        assert 'password_hash' not in user_dict
        assert 'mfa_secret' not in user_dict
        assert user_dict['mfa_enabled'] is True
        
        # Test with sensitive data
        user_dict_sensitive = user.to_dict(include_sensitive=True)
        assert 'password_hash' in user_dict_sensitive
        assert 'mfa_secret' in user_dict_sensitive


class TestAuthToken:
    """Test suite for AuthToken data structure"""
    
    def test_token_creation(self):
        """Test AuthToken creation and basic properties"""
        created_at = datetime.now()
        expires_at = created_at + timedelta(hours=1)
        
        token = AuthToken(
            token_id="token_001",
            user_id="user_001",
            token_type=TokenType.ACCESS,
            token_hash="hashed_token",
            created_at=created_at,
            expires_at=expires_at
        )
        
        assert token.token_id == "token_001"
        assert token.user_id == "user_001"
        assert token.token_type == TokenType.ACCESS
        assert token.token_hash == "hashed_token"
        assert token.created_at == created_at
        assert token.expires_at == expires_at
        assert token.is_revoked is False
        assert token.last_used is None
    
    def test_token_is_expired(self):
        """Test token expiration checking"""
        # Non-expiring token
        token1 = AuthToken(
            token_id="token_002",
            user_id="user_001",
            token_type=TokenType.API_KEY,
            token_hash="hash",
            created_at=datetime.now()
        )
        assert token1.is_expired() is False
        
        # Expired token
        token2 = AuthToken(
            token_id="token_003",
            user_id="user_001",
            token_type=TokenType.ACCESS,
            token_hash="hash",
            created_at=datetime.now(),
            expires_at=datetime.now() - timedelta(minutes=1)
        )
        assert token2.is_expired() is True
        
        # Valid token
        token3 = AuthToken(
            token_id="token_004",
            user_id="user_001",
            token_type=TokenType.ACCESS,
            token_hash="hash",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1)
        )
        assert token3.is_expired() is False
    
    def test_token_is_valid(self):
        """Test token validity checking"""
        # Valid token
        token1 = AuthToken(
            token_id="token_005",
            user_id="user_001",
            token_type=TokenType.ACCESS,
            token_hash="hash",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1)
        )
        assert token1.is_valid() is True
        
        # Expired token
        token2 = AuthToken(
            token_id="token_006",
            user_id="user_001",
            token_type=TokenType.ACCESS,
            token_hash="hash",
            created_at=datetime.now(),
            expires_at=datetime.now() - timedelta(minutes=1)
        )
        assert token2.is_valid() is False
        
        # Revoked token
        token3 = AuthToken(
            token_id="token_007",
            user_id="user_001",
            token_type=TokenType.ACCESS,
            token_hash="hash",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
            is_revoked=True
        )
        assert token3.is_valid() is False
    
    def test_token_to_dict(self):
        """Test token dictionary conversion"""
        created_at = datetime.now()
        expires_at = created_at + timedelta(hours=1)
        
        token = AuthToken(
            token_id="token_008",
            user_id="user_001",
            token_type=TokenType.REFRESH,
            token_hash="hash",
            created_at=created_at,
            expires_at=expires_at,
            metadata={"client_id": "web_app"}
        )
        
        token_dict = token.to_dict()
        assert token_dict['token_id'] == "token_008"
        assert token_dict['user_id'] == "user_001"
        assert token_dict['token_type'] == "refresh"
        assert token_dict['created_at'] == created_at.isoformat()
        assert token_dict['expires_at'] == expires_at.isoformat()
        assert token_dict['is_revoked'] is False
        assert token_dict['metadata'] == {"client_id": "web_app"}


class TestAuthManager:
    """Test suite for AuthManager"""
    
    @pytest.fixture
    def auth_manager(self):
        """Create AuthManager instance"""
        config = {
            'password_policy': {
                'min_length': 8,
                'require_uppercase': True,
                'require_lowercase': True,
                'require_digits': True,
                'require_special': True
            },
            'lockout_policy': {
                'max_failed_attempts': 3,
                'lockout_duration_minutes': 15
            },
            'create_default_admin': False
        }
        return AuthManager(config)
    
    def test_auth_manager_initialization(self, auth_manager):
        """Test AuthManager initialization"""
        assert len(auth_manager.users) == 0
        assert len(auth_manager.tokens) == 0
        assert auth_manager.password_policy['min_length'] == 8
        assert auth_manager.lockout_policy['max_failed_attempts'] == 3
    
    def test_create_user_success(self, auth_manager):
        """Test successful user creation"""
        user = auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPass123!",
            roles={AuthRole.ANALYST},
            is_verified=True
        )
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_verified is True
        assert AuthRole.ANALYST in user.roles
        assert user.user_id in auth_manager.users
        
        # Password should be hashed
        assert user.password_hash != "TestPass123!"
        assert len(user.password_hash) > 20  # bcrypt hashes are long
    
    def test_create_user_duplicate(self, auth_manager):
        """Test creating duplicate user fails"""
        auth_manager.create_user(
            username="duplicate",
            email="dup@example.com",
            password="TestPass123!"
        )
        
        # Try to create user with same username
        with pytest.raises(AuthenticationError, match="already exists"):
            auth_manager.create_user(
                username="duplicate",
                email="different@example.com",
                password="TestPass123!"
            )
        
        # Try to create user with same email
        with pytest.raises(AuthenticationError, match="already exists"):
            auth_manager.create_user(
                username="different",
                email="dup@example.com",
                password="TestPass123!"
            )
    
    def test_create_user_weak_password(self, auth_manager):
        """Test creating user with weak password fails"""
        with pytest.raises(AuthenticationError, match="does not meet security requirements"):
            auth_manager.create_user(
                username="weakpass",
                email="weak@example.com",
                password="weak"
            )
    
    def test_authenticate_user_success(self, auth_manager):
        """Test successful user authentication"""
        # Create user
        auth_manager.create_user(
            username="authtest",
            email="auth@example.com",
            password="TestPass123!",
            is_verified=True
        )
        
        # Authenticate with username
        user = auth_manager.authenticate_user("authtest", "TestPass123!")
        assert user.username == "authtest"
        assert user.last_login is not None
        
        # Authenticate with email
        user = auth_manager.authenticate_user("auth@example.com", "TestPass123!")
        assert user.email == "auth@example.com"
    
    def test_authenticate_user_invalid_credentials(self, auth_manager):
        """Test authentication with invalid credentials"""
        auth_manager.create_user(
            username="authtest",
            email="auth@example.com",
            password="TestPass123!"
        )
        
        # Wrong password
        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            auth_manager.authenticate_user("authtest", "WrongPass123!")
        
        # Non-existent user
        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            auth_manager.authenticate_user("nonexistent", "TestPass123!")
    
    def test_authenticate_user_account_lockout(self, auth_manager):
        """Test account lockout after failed attempts"""
        user = auth_manager.create_user(
            username="locktest",
            email="lock@example.com",
            password="TestPass123!"
        )
        
        # Make failed attempts
        for i in range(3):  # lockout_policy max_failed_attempts = 3
            with pytest.raises(AuthenticationError):
                auth_manager.authenticate_user("locktest", "WrongPass!")
        
        # Account should be locked
        user = auth_manager.get_user(user.user_id)
        assert user.is_locked() is True
        
        # Even correct password should fail when locked
        with pytest.raises(AuthenticationError, match="temporarily locked"):
            auth_manager.authenticate_user("locktest", "TestPass123!")
    
    def test_create_token(self, auth_manager):
        """Test token creation"""
        user = auth_manager.create_user(
            username="tokentest",
            email="token@example.com",
            password="TestPass123!"
        )
        
        # Create access token
        token = auth_manager.create_token(user, TokenType.ACCESS)
        
        assert token.user_id == user.user_id
        assert token.token_type == TokenType.ACCESS
        assert token.expires_at is not None
        assert 'token_value' in token.metadata  # Should include actual token value
        assert token.token_id in auth_manager.tokens
        assert token.token_id in auth_manager.user_sessions[user.user_id]
    
    def test_validate_token(self, auth_manager):
        """Test token validation"""
        user = auth_manager.create_user(
            username="validatetest",
            email="validate@example.com",
            password="TestPass123!"
        )
        
        # Create and validate token
        token = auth_manager.create_token(user, TokenType.ACCESS)
        token_value = token.metadata['token_value']
        
        validated_token = auth_manager.validate_token(token_value)
        assert validated_token.token_id == token.token_id
        assert validated_token.user_id == user.user_id
        assert validated_token.last_used is not None
    
    def test_validate_invalid_token(self, auth_manager):
        """Test validation of invalid token"""
        with pytest.raises(AuthenticationError, match="Invalid token"):
            auth_manager.validate_token("invalid_token_value")
    
    def test_revoke_token(self, auth_manager):
        """Test token revocation"""
        user = auth_manager.create_user(
            username="revoketest",
            email="revoke@example.com",
            password="TestPass123!"
        )
        
        token = auth_manager.create_token(user, TokenType.ACCESS)
        
        # Revoke token
        success = auth_manager.revoke_token(token.token_id)
        assert success is True
        
        # Token should be revoked
        revoked_token = auth_manager.tokens[token.token_id]
        assert revoked_token.is_revoked is True
        
        # Token should not validate
        token_value = token.metadata['token_value']
        with pytest.raises(AuthenticationError, match="expired or revoked"):
            auth_manager.validate_token(token_value)
    
    def test_revoke_user_tokens(self, auth_manager):
        """Test revoking all user tokens"""
        user = auth_manager.create_user(
            username="revokealltest",
            email="revokeall@example.com",
            password="TestPass123!"
        )
        
        # Create multiple tokens
        token1 = auth_manager.create_token(user, TokenType.ACCESS)
        token2 = auth_manager.create_token(user, TokenType.REFRESH)
        token3 = auth_manager.create_token(user, TokenType.API_KEY)
        
        # Revoke all tokens
        revoked_count = auth_manager.revoke_user_tokens(user.user_id)
        assert revoked_count == 3
        
        # All tokens should be revoked
        for token_id in [token1.token_id, token2.token_id, token3.token_id]:
            token = auth_manager.tokens[token_id]
            assert token.is_revoked is True
    
    def test_setup_mfa(self, auth_manager):
        """Test MFA setup"""
        user = auth_manager.create_user(
            username="mfatest",
            email="mfa@example.com",
            password="TestPass123!"
        )
        
        # Setup MFA
        mfa_data = auth_manager.setup_mfa(user.user_id)
        
        assert 'secret' in mfa_data
        assert 'qr_code' in mfa_data
        assert 'totp_uri' in mfa_data
        assert mfa_data['qr_code'].startswith('data:image/png;base64,')
        
        # User should have MFA secret but not enabled yet
        updated_user = auth_manager.get_user(user.user_id)
        assert updated_user.mfa_secret is not None
        assert updated_user.mfa_enabled is False
    
    def test_enable_mfa(self, auth_manager):
        """Test MFA enabling with verification"""
        user = auth_manager.create_user(
            username="enablemfa",
            email="enablemfa@example.com",
            password="TestPass123!"
        )
        
        # Setup MFA
        mfa_data = auth_manager.setup_mfa(user.user_id)
        
        # Generate TOTP code for verification
        import pyotp
        totp = pyotp.TOTP(mfa_data['secret'])
        verification_code = totp.now()
        
        # Enable MFA
        success = auth_manager.enable_mfa(user.user_id, verification_code)
        assert success is True
        
        # User should have MFA enabled
        updated_user = auth_manager.get_user(user.user_id)
        assert updated_user.mfa_enabled is True
    
    def test_change_password(self, auth_manager):
        """Test password change"""
        user = auth_manager.create_user(
            username="changepass",
            email="changepass@example.com",
            password="OldPass123!"
        )
        
        # Change password
        success = auth_manager.change_password(
            user.user_id, 
            "OldPass123!", 
            "NewPass456!"
        )
        assert success is True
        
        # Old password should not work
        with pytest.raises(AuthenticationError):
            auth_manager.authenticate_user("changepass", "OldPass123!")
        
        # New password should work
        authenticated_user = auth_manager.authenticate_user("changepass", "NewPass456!")
        assert authenticated_user.user_id == user.user_id
    
    def test_get_user_methods(self, auth_manager):
        """Test user retrieval methods"""
        user = auth_manager.create_user(
            username="gettest",
            email="get@example.com",
            password="TestPass123!"
        )
        
        # Get by ID
        retrieved_user = auth_manager.get_user(user.user_id)
        assert retrieved_user.user_id == user.user_id
        
        # Get by username
        retrieved_user = auth_manager.get_user_by_username("gettest")
        assert retrieved_user.username == "gettest"
        
        # Get by email
        retrieved_user = auth_manager.get_user_by_email("get@example.com")
        assert retrieved_user.email == "get@example.com"
        
        # Non-existent user
        assert auth_manager.get_user("nonexistent") is None
        assert auth_manager.get_user_by_username("nonexistent") is None
        assert auth_manager.get_user_by_email("nonexistent@example.com") is None
    
    def test_get_statistics(self, auth_manager):
        """Test authentication statistics"""
        # Create users and tokens
        user1 = auth_manager.create_user("user1", "user1@example.com", "TestPass123!")
        user2 = auth_manager.create_user("user2", "user2@example.com", "TestPass123!", is_verified=True)
        
        auth_manager.create_token(user1, TokenType.ACCESS)
        auth_manager.create_token(user2, TokenType.REFRESH)
        
        stats = auth_manager.get_statistics()
        
        assert stats['total_users'] == 2
        assert stats['active_users'] == 2
        assert stats['verified_users'] == 1
        assert stats['total_tokens'] == 2
        assert stats['active_tokens'] == 2
        assert stats['locked_users'] == 0
        assert stats['mfa_enabled_users'] == 0


def run_auth_manager_tests():
    """Run all authentication manager tests"""
    print("🔐 Running Authentication Manager Tests")
    print("=" * 50)
    
    # Test categories
    test_categories = [
        ("Auth User", TestAuthUser),
        ("Auth Token", TestAuthToken),
        ("Auth Manager", TestAuthManager)
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
                    
                    if 'auth_manager' in sig.parameters:
                        config = {
                            'password_policy': {
                                'min_length': 8,
                                'require_uppercase': True,
                                'require_lowercase': True,
                                'require_digits': True,
                                'require_special': True
                            },
                            'lockout_policy': {
                                'max_failed_attempts': 3,
                                'lockout_duration_minutes': 15
                            },
                            'create_default_admin': False
                        }
                        auth_manager = AuthManager(config)
                        method(auth_manager)
                    else:
                        method()
                    
                    print(f"  ✅ {test_method}")
                    passed_tests += 1
                    category_passed += 1
                    
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
        
        print(f"  📈 Category Results: {category_passed}/{len(test_methods)} passed")
    
    print(f"\n🏆 Overall Auth Manager Test Results: {passed_tests}/{total_tests} passed")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_auth_manager_tests()
    exit(0 if success else 1)
