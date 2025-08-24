# Purpose: Authentication and authorization management system
# Author: WicketWise AI, Last Modified: 2024

import hashlib
import secrets
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
import bcrypt
import pyotp
import qrcode
from io import BytesIO
import base64


class TokenType(Enum):
    """Token types for authentication"""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    RESET_PASSWORD = "reset_password"
    EMAIL_VERIFICATION = "email_verification"
    MFA_CHALLENGE = "mfa_challenge"


class AuthRole(Enum):
    """User roles with hierarchical permissions"""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"
    GUEST = "guest"


class AuthPermission(Enum):
    """Granular permissions for resources"""
    # Data permissions
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    EXPORT_DATA = "export_data"
    
    # Model permissions
    VIEW_MODELS = "view_models"
    TRAIN_MODELS = "train_models"
    DEPLOY_MODELS = "deploy_models"
    DELETE_MODELS = "delete_models"
    
    # Agent permissions
    VIEW_AGENTS = "view_agents"
    CONFIGURE_AGENTS = "configure_agents"
    EXECUTE_AGENTS = "execute_agents"
    
    # System permissions
    VIEW_SYSTEM = "view_system"
    CONFIGURE_SYSTEM = "configure_system"
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT = "view_audit"
    
    # API permissions
    API_ACCESS = "api_access"
    RATE_LIMIT_EXEMPT = "rate_limit_exempt"


class AuthenticationError(Exception):
    """Authentication related errors"""
    pass


class AuthorizationError(Exception):
    """Authorization related errors"""
    pass


@dataclass
class AuthToken:
    """Authentication token data structure"""
    token_id: str
    user_id: str
    token_type: TokenType
    token_hash: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_revoked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if token is expired"""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if token is valid (not expired or revoked)"""
        return not self.is_expired() and not self.is_revoked
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert token to dictionary"""
        return {
            'token_id': self.token_id,
            'user_id': self.user_id,
            'token_type': self.token_type.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'is_revoked': self.is_revoked,
            'metadata': self.metadata
        }


@dataclass
class AuthUser:
    """User data structure with authentication info"""
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: Set[AuthRole] = field(default_factory=set)
    permissions: Set[AuthPermission] = field(default_factory=set)
    is_active: bool = True
    is_verified: bool = False
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_role(self, role: AuthRole) -> bool:
        """Check if user has specific role"""
        return role in self.roles
    
    def has_permission(self, permission: AuthPermission) -> bool:
        """Check if user has specific permission"""
        # Check direct permissions
        if permission in self.permissions:
            return True
        
        # Check role-based permissions
        role_permissions = self._get_role_permissions()
        return permission in role_permissions
    
    def _get_role_permissions(self) -> Set[AuthPermission]:
        """Get permissions based on user roles"""
        permissions = set()
        
        for role in self.roles:
            if role == AuthRole.SUPER_ADMIN:
                # Super admin has all permissions
                permissions.update(AuthPermission)
            elif role == AuthRole.ADMIN:
                permissions.update([
                    AuthPermission.READ_DATA, AuthPermission.WRITE_DATA, AuthPermission.EXPORT_DATA,
                    AuthPermission.VIEW_MODELS, AuthPermission.TRAIN_MODELS, AuthPermission.DEPLOY_MODELS,
                    AuthPermission.VIEW_AGENTS, AuthPermission.CONFIGURE_AGENTS, AuthPermission.EXECUTE_AGENTS,
                    AuthPermission.VIEW_SYSTEM, AuthPermission.CONFIGURE_SYSTEM, AuthPermission.MANAGE_USERS,
                    AuthPermission.VIEW_AUDIT, AuthPermission.API_ACCESS
                ])
            elif role == AuthRole.ANALYST:
                permissions.update([
                    AuthPermission.READ_DATA, AuthPermission.WRITE_DATA, AuthPermission.EXPORT_DATA,
                    AuthPermission.VIEW_MODELS, AuthPermission.TRAIN_MODELS,
                    AuthPermission.VIEW_AGENTS, AuthPermission.EXECUTE_AGENTS,
                    AuthPermission.VIEW_SYSTEM, AuthPermission.API_ACCESS
                ])
            elif role == AuthRole.VIEWER:
                permissions.update([
                    AuthPermission.READ_DATA, AuthPermission.VIEW_MODELS,
                    AuthPermission.VIEW_AGENTS, AuthPermission.VIEW_SYSTEM
                ])
            elif role == AuthRole.API_USER:
                permissions.update([
                    AuthPermission.API_ACCESS, AuthPermission.READ_DATA,
                    AuthPermission.VIEW_MODELS, AuthPermission.EXECUTE_AGENTS
                ])
            elif role == AuthRole.GUEST:
                permissions.update([
                    AuthPermission.READ_DATA, AuthPermission.VIEW_MODELS
                ])
        
        return permissions
    
    def is_locked(self) -> bool:
        """Check if user account is locked"""
        if not self.locked_until:
            return False
        return datetime.now() < self.locked_until
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert user to dictionary"""
        data = {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'roles': [role.value for role in self.roles],
            'permissions': [perm.value for perm in self.permissions],
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'mfa_enabled': self.mfa_enabled,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'failed_login_attempts': self.failed_login_attempts,
            'locked_until': self.locked_until.isoformat() if self.locked_until else None,
            'metadata': self.metadata
        }
        
        if include_sensitive:
            data['password_hash'] = self.password_hash
            data['mfa_secret'] = self.mfa_secret
        
        return data


class AuthManager:
    """Comprehensive authentication and authorization manager"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Storage
        self.users: Dict[str, AuthUser] = {}
        self.tokens: Dict[str, AuthToken] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)
        
        # Security configuration
        self.password_policy = self.config.get('password_policy', {
            'min_length': 8,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_digits': True,
            'require_special': True,
            'max_age_days': 90
        })
        
        self.lockout_policy = self.config.get('lockout_policy', {
            'max_failed_attempts': 5,
            'lockout_duration_minutes': 30,
            'reset_attempts_after_minutes': 60
        })
        
        self.token_config = self.config.get('token_config', {
            'access_token_ttl_minutes': 60,
            'refresh_token_ttl_days': 30,
            'api_key_ttl_days': 365,
            'reset_token_ttl_minutes': 15,
            'mfa_challenge_ttl_minutes': 5
        })
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize default admin user if configured
        if self.config.get('create_default_admin', False):
            self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin_config = self.config.get('default_admin', {
            'username': 'admin',
            'email': 'admin@wicketwise.ai',
            'password': 'WicketWise2024!'
        })
        
        try:
            self.create_user(
                username=admin_config['username'],
                email=admin_config['email'],
                password=admin_config['password'],
                roles={AuthRole.SUPER_ADMIN},
                is_verified=True
            )
            self.logger.info("Default admin user created successfully")
        except Exception as e:
            self.logger.warning(f"Failed to create default admin user: {str(e)}")
    
    def create_user(self, username: str, email: str, password: str,
                   roles: Optional[Set[AuthRole]] = None,
                   permissions: Optional[Set[AuthPermission]] = None,
                   is_verified: bool = False) -> AuthUser:
        """Create a new user"""
        with self.lock:
            # Check if user already exists
            if any(u.username == username or u.email == email for u in self.users.values()):
                raise AuthenticationError("User with this username or email already exists")
            
            # Validate password
            if not self._validate_password(password):
                raise AuthenticationError("Password does not meet security requirements")
            
            # Generate user ID and hash password
            user_id = self._generate_user_id()
            password_hash = self._hash_password(password)
            
            # Create user
            user = AuthUser(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                roles=roles or {AuthRole.VIEWER},
                permissions=permissions or set(),
                is_verified=is_verified
            )
            
            self.users[user_id] = user
            self.logger.info(f"User created: {username} ({user_id})")
            
            return user
    
    def authenticate_user(self, username: str, password: str,
                         mfa_code: Optional[str] = None) -> AuthUser:
        """Authenticate user with username/password and optional MFA"""
        with self.lock:
            # Find user by username or email
            user = None
            for u in self.users.values():
                if u.username == username or u.email == username:
                    user = u
                    break
            
            if not user:
                raise AuthenticationError("Invalid credentials")
            
            # Check if account is locked
            if user.is_locked():
                raise AuthenticationError("Account is temporarily locked")
            
            # Check if account is active
            if not user.is_active:
                raise AuthenticationError("Account is disabled")
            
            # Verify password
            if not self._verify_password(password, user.password_hash):
                user.failed_login_attempts += 1
                
                # Lock account if too many failed attempts
                if user.failed_login_attempts >= self.lockout_policy['max_failed_attempts']:
                    lockout_duration = timedelta(minutes=self.lockout_policy['lockout_duration_minutes'])
                    user.locked_until = datetime.now() + lockout_duration
                    self.logger.warning(f"Account locked due to failed attempts: {user.username}")
                
                raise AuthenticationError("Invalid credentials")
            
            # Check MFA if enabled
            if user.mfa_enabled:
                if not mfa_code:
                    raise AuthenticationError("MFA code required")
                
                if not self._verify_mfa_code(user, mfa_code):
                    user.failed_login_attempts += 1
                    raise AuthenticationError("Invalid MFA code")
            
            # Successful authentication
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.now()
            
            self.logger.info(f"User authenticated successfully: {user.username}")
            return user
    
    def create_token(self, user: AuthUser, token_type: TokenType,
                    expires_in_minutes: Optional[int] = None) -> AuthToken:
        """Create authentication token for user"""
        with self.lock:
            # Generate token
            token_value = secrets.token_urlsafe(32)
            token_hash = self._hash_token(token_value)
            token_id = self._generate_token_id()
            
            # Calculate expiration
            expires_at = None
            if expires_in_minutes:
                expires_at = datetime.now() + timedelta(minutes=expires_in_minutes)
            elif token_type == TokenType.ACCESS:
                expires_at = datetime.now() + timedelta(minutes=self.token_config['access_token_ttl_minutes'])
            elif token_type == TokenType.REFRESH:
                expires_at = datetime.now() + timedelta(days=self.token_config['refresh_token_ttl_days'])
            elif token_type == TokenType.API_KEY:
                expires_at = datetime.now() + timedelta(days=self.token_config['api_key_ttl_days'])
            elif token_type == TokenType.RESET_PASSWORD:
                expires_at = datetime.now() + timedelta(minutes=self.token_config['reset_token_ttl_minutes'])
            elif token_type == TokenType.MFA_CHALLENGE:
                expires_at = datetime.now() + timedelta(minutes=self.token_config['mfa_challenge_ttl_minutes'])
            
            # Create token
            token = AuthToken(
                token_id=token_id,
                user_id=user.user_id,
                token_type=token_type,
                token_hash=token_hash,
                created_at=datetime.now(),
                expires_at=expires_at
            )
            
            self.tokens[token_id] = token
            self.user_sessions[user.user_id].add(token_id)
            
            # Return token with actual value (only time it's available)
            token.metadata['token_value'] = token_value
            
            return token
    
    def validate_token(self, token_value: str) -> AuthToken:
        """Validate and return token if valid"""
        with self.lock:
            token_hash = self._hash_token(token_value)
            
            # Find token by hash
            token = None
            for t in self.tokens.values():
                if t.token_hash == token_hash:
                    token = t
                    break
            
            if not token:
                raise AuthenticationError("Invalid token")
            
            if not token.is_valid():
                raise AuthenticationError("Token expired or revoked")
            
            # Update last used
            token.last_used = datetime.now()
            
            return token
    
    def revoke_token(self, token_id: str) -> bool:
        """Revoke a specific token"""
        with self.lock:
            if token_id not in self.tokens:
                return False
            
            token = self.tokens[token_id]
            token.is_revoked = True
            
            # Remove from user sessions
            if token.user_id in self.user_sessions:
                self.user_sessions[token.user_id].discard(token_id)
            
            self.logger.info(f"Token revoked: {token_id}")
            return True
    
    def revoke_user_tokens(self, user_id: str, token_type: Optional[TokenType] = None) -> int:
        """Revoke all tokens for a user (optionally filtered by type)"""
        with self.lock:
            revoked_count = 0
            
            if user_id not in self.user_sessions:
                return 0
            
            token_ids = list(self.user_sessions[user_id])
            for token_id in token_ids:
                if token_id in self.tokens:
                    token = self.tokens[token_id]
                    if not token_type or token.token_type == token_type:
                        token.is_revoked = True
                        self.user_sessions[user_id].discard(token_id)
                        revoked_count += 1
            
            self.logger.info(f"Revoked {revoked_count} tokens for user {user_id}")
            return revoked_count
    
    def setup_mfa(self, user_id: str) -> Dict[str, str]:
        """Setup MFA for user and return QR code data"""
        with self.lock:
            if user_id not in self.users:
                raise AuthenticationError("User not found")
            
            user = self.users[user_id]
            
            # Generate MFA secret
            secret = pyotp.random_base32()
            user.mfa_secret = secret
            
            # Generate QR code
            totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
                name=user.email,
                issuer_name="WicketWise"
            )
            
            # Create QR code image
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(totp_uri)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            qr_code_data = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                'secret': secret,
                'qr_code': f"data:image/png;base64,{qr_code_data}",
                'totp_uri': totp_uri
            }
    
    def enable_mfa(self, user_id: str, verification_code: str) -> bool:
        """Enable MFA after verifying setup code"""
        with self.lock:
            if user_id not in self.users:
                raise AuthenticationError("User not found")
            
            user = self.users[user_id]
            
            if not user.mfa_secret:
                raise AuthenticationError("MFA not set up")
            
            if self._verify_mfa_code(user, verification_code):
                user.mfa_enabled = True
                self.logger.info(f"MFA enabled for user: {user.username}")
                return True
            
            return False
    
    def disable_mfa(self, user_id: str) -> bool:
        """Disable MFA for user"""
        with self.lock:
            if user_id not in self.users:
                return False
            
            user = self.users[user_id]
            user.mfa_enabled = False
            user.mfa_secret = None
            
            self.logger.info(f"MFA disabled for user: {user.username}")
            return True
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """Change user password"""
        with self.lock:
            if user_id not in self.users:
                raise AuthenticationError("User not found")
            
            user = self.users[user_id]
            
            # Verify old password
            if not self._verify_password(old_password, user.password_hash):
                raise AuthenticationError("Invalid current password")
            
            # Validate new password
            if not self._validate_password(new_password):
                raise AuthenticationError("New password does not meet security requirements")
            
            # Update password
            user.password_hash = self._hash_password(new_password)
            
            # Revoke all existing tokens to force re-authentication
            self.revoke_user_tokens(user_id)
            
            self.logger.info(f"Password changed for user: {user.username}")
            return True
    
    def get_user(self, user_id: str) -> Optional[AuthUser]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[AuthUser]:
        """Get user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def get_user_by_email(self, email: str) -> Optional[AuthUser]:
        """Get user by email"""
        for user in self.users.values():
            if user.email == email:
                return user
        return None
    
    def _validate_password(self, password: str) -> bool:
        """Validate password against policy"""
        policy = self.password_policy
        
        if len(password) < policy['min_length']:
            return False
        
        if policy['require_uppercase'] and not any(c.isupper() for c in password):
            return False
        
        if policy['require_lowercase'] and not any(c.islower() for c in password):
            return False
        
        if policy['require_digits'] and not any(c.isdigit() for c in password):
            return False
        
        if policy['require_special'] and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False
        
        return True
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _hash_token(self, token: str) -> str:
        """Hash token for storage"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def _verify_mfa_code(self, user: AuthUser, code: str) -> bool:
        """Verify MFA TOTP code"""
        if not user.mfa_secret:
            return False
        
        totp = pyotp.TOTP(user.mfa_secret)
        return totp.verify(code, valid_window=1)  # Allow 1 window tolerance
    
    def _generate_user_id(self) -> str:
        """Generate unique user ID"""
        timestamp = str(int(time.time() * 1000))
        return f"user_{timestamp}_{secrets.token_hex(4)}"
    
    def _generate_token_id(self) -> str:
        """Generate unique token ID"""
        timestamp = str(int(time.time() * 1000))
        return f"token_{timestamp}_{secrets.token_hex(4)}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get authentication statistics"""
        with self.lock:
            active_tokens = sum(1 for t in self.tokens.values() if t.is_valid())
            locked_users = sum(1 for u in self.users.values() if u.is_locked())
            mfa_enabled_users = sum(1 for u in self.users.values() if u.mfa_enabled)
            
            return {
                'total_users': len(self.users),
                'active_users': sum(1 for u in self.users.values() if u.is_active),
                'verified_users': sum(1 for u in self.users.values() if u.is_verified),
                'locked_users': locked_users,
                'mfa_enabled_users': mfa_enabled_users,
                'total_tokens': len(self.tokens),
                'active_tokens': active_tokens,
                'revoked_tokens': sum(1 for t in self.tokens.values() if t.is_revoked),
                'expired_tokens': sum(1 for t in self.tokens.values() if t.is_expired())
            }
