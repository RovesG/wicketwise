# Purpose: Role-Based Access Control (RBAC) for DGL governance
# Author: WicketWise AI, Last Modified: 2024

"""
RBAC Manager

Implements comprehensive role-based access control for DGL:
- Hierarchical role definitions
- Permission management
- User role assignments
- Access control validation
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .audit import GovernanceAuditStore


logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions for DGL operations"""
    
    # Governance permissions
    GOVERNANCE_VIEW = "governance.view"
    GOVERNANCE_ACTIVATE_SHADOW = "governance.activate_shadow"
    GOVERNANCE_ACTIVATE_LIVE = "governance.activate_live"
    GOVERNANCE_DEACTIVATE = "governance.deactivate"
    GOVERNANCE_DOWNGRADE_TO_SHADOW = "governance.downgrade_to_shadow"
    GOVERNANCE_EMERGENCY_KILL = "governance.emergency_kill"
    GOVERNANCE_RECOVER_FROM_KILL = "governance.recover_from_kill"
    
    # Rule management permissions
    RULES_VIEW = "rules.view"
    RULES_MODIFY_BANKROLL = "rules.modify_bankroll"
    RULES_MODIFY_PNL = "rules.modify_pnl"
    RULES_MODIFY_LIQUIDITY = "rules.modify_liquidity"
    RULES_MODIFY_RATE_LIMITS = "rules.modify_rate_limits"
    RULES_APPROVE_CHANGES = "rules.approve_changes"
    
    # Audit permissions
    AUDIT_VIEW = "audit.view"
    AUDIT_EXPORT = "audit.export"
    AUDIT_VERIFY_INTEGRITY = "audit.verify_integrity"
    AUDIT_MANAGE_RETENTION = "audit.manage_retention"
    
    # User management permissions
    USER_VIEW = "user.view"
    USER_CREATE = "user.create"
    USER_MODIFY = "user.modify"
    USER_DELETE = "user.delete"
    USER_ASSIGN_ROLES = "user.assign_roles"
    
    # System administration permissions
    SYSTEM_VIEW_METRICS = "system.view_metrics"
    SYSTEM_MODIFY_CONFIG = "system.modify_config"
    SYSTEM_MANAGE_ALERTS = "system.manage_alerts"
    SYSTEM_EMERGENCY_ACCESS = "system.emergency_access"
    
    # Approval permissions
    APPROVAL_CREATE = "approval.create"
    APPROVAL_APPROVE = "approval.approve"
    APPROVAL_ESCALATE = "approval.escalate"
    APPROVAL_CANCEL = "approval.cancel"


@dataclass
class Role:
    """Role definition with permissions and metadata"""
    name: str
    display_name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)
    is_system_role: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has specific permission"""
        return permission in self.permissions
    
    def add_permission(self, permission: Permission):
        """Add permission to role"""
        self.permissions.add(permission)
    
    def remove_permission(self, permission: Permission):
        """Remove permission from role"""
        self.permissions.discard(permission)


@dataclass
class User:
    """User with role assignments and metadata"""
    username: str
    display_name: str
    email: str
    roles: Set[str] = field(default_factory=set)
    is_active: bool = True
    is_system_user: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    mfa_enabled: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has specific role"""
        return role_name in self.roles
    
    def add_role(self, role_name: str):
        """Add role to user"""
        self.roles.add(role_name)
    
    def remove_role(self, role_name: str):
        """Remove role from user"""
        self.roles.discard(role_name)


class RBACManager:
    """
    Comprehensive Role-Based Access Control manager
    
    Manages roles, permissions, users, and access control
    validation with hierarchical role inheritance and
    comprehensive audit logging.
    """
    
    def __init__(self, audit_store: GovernanceAuditStore):
        """
        Initialize RBAC manager
        
        Args:
            audit_store: Audit store for logging access control events
        """
        self.audit_store = audit_store
        
        # Role and user storage
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        
        # Permission cache for performance
        self._permission_cache: Dict[str, Set[Permission]] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        
        # Initialize default roles
        self._initialize_default_roles()
        
        logger.info("RBAC manager initialized")
    
    def _initialize_default_roles(self):
        """Initialize default system roles"""
        
        # Viewer role - read-only access
        viewer_role = Role(
            name="viewer",
            display_name="Viewer",
            description="Read-only access to DGL system",
            permissions={
                Permission.GOVERNANCE_VIEW,
                Permission.RULES_VIEW,
                Permission.AUDIT_VIEW,
                Permission.SYSTEM_VIEW_METRICS
            },
            is_system_role=True
        )
        
        # Operator role - basic operational access
        operator_role = Role(
            name="operator",
            display_name="Operator",
            description="Basic operational access to DGL system",
            permissions={
                Permission.GOVERNANCE_VIEW,
                Permission.GOVERNANCE_ACTIVATE_SHADOW,
                Permission.GOVERNANCE_DEACTIVATE,
                Permission.RULES_VIEW,
                Permission.AUDIT_VIEW,
                Permission.SYSTEM_VIEW_METRICS,
                Permission.APPROVAL_CREATE
            },
            parent_roles={"viewer"},
            is_system_role=True
        )
        
        # Risk Manager role - risk and rule management
        risk_manager_role = Role(
            name="risk_manager",
            display_name="Risk Manager",
            description="Risk management and rule configuration",
            permissions={
                Permission.RULES_MODIFY_BANKROLL,
                Permission.RULES_MODIFY_PNL,
                Permission.RULES_MODIFY_LIQUIDITY,
                Permission.RULES_APPROVE_CHANGES,
                Permission.APPROVAL_APPROVE,
                Permission.APPROVAL_ESCALATE
            },
            parent_roles={"operator"},
            is_system_role=True
        )
        
        # Operations Manager role - operational control
        operations_manager_role = Role(
            name="operations_manager",
            display_name="Operations Manager",
            description="Operational control and system management",
            permissions={
                Permission.GOVERNANCE_ACTIVATE_LIVE,
                Permission.GOVERNANCE_DOWNGRADE_TO_SHADOW,
                Permission.RULES_MODIFY_RATE_LIMITS,
                Permission.SYSTEM_MODIFY_CONFIG,
                Permission.SYSTEM_MANAGE_ALERTS,
                Permission.APPROVAL_APPROVE,
                Permission.APPROVAL_ESCALATE
            },
            parent_roles={"risk_manager"},
            is_system_role=True
        )
        
        # Security Officer role - security and audit
        security_officer_role = Role(
            name="security_officer",
            display_name="Security Officer",
            description="Security management and audit oversight",
            permissions={
                Permission.AUDIT_EXPORT,
                Permission.AUDIT_VERIFY_INTEGRITY,
                Permission.USER_VIEW,
                Permission.USER_ASSIGN_ROLES,
                Permission.APPROVAL_APPROVE
            },
            parent_roles={"operator"},
            is_system_role=True
        )
        
        # Senior Manager role - high-level approvals
        senior_manager_role = Role(
            name="senior_manager",
            display_name="Senior Manager",
            description="Senior management with high-level approval authority",
            permissions={
                Permission.GOVERNANCE_EMERGENCY_KILL,
                Permission.APPROVAL_APPROVE,
                Permission.APPROVAL_ESCALATE,
                Permission.APPROVAL_CANCEL
            },
            parent_roles={"operations_manager", "security_officer"},
            is_system_role=True
        )
        
        # Administrator role - full system access
        administrator_role = Role(
            name="administrator",
            display_name="Administrator",
            description="Full system administration access",
            permissions=set(Permission),  # All permissions
            parent_roles={"senior_manager"},
            is_system_role=True
        )
        
        # Emergency Responder role - emergency actions only
        emergency_responder_role = Role(
            name="emergency_responder",
            display_name="Emergency Responder",
            description="Emergency response and kill switch access",
            permissions={
                Permission.GOVERNANCE_VIEW,
                Permission.GOVERNANCE_EMERGENCY_KILL,
                Permission.SYSTEM_EMERGENCY_ACCESS,
                Permission.APPROVAL_APPROVE
            },
            is_system_role=True
        )
        
        # Store default roles
        default_roles = [
            viewer_role,
            operator_role,
            risk_manager_role,
            operations_manager_role,
            security_officer_role,
            senior_manager_role,
            administrator_role,
            emergency_responder_role
        ]
        
        for role in default_roles:
            self.roles[role.name] = role
        
        logger.info(f"Initialized {len(default_roles)} default roles")
    
    def create_role(self,
                   name: str,
                   display_name: str,
                   description: str,
                   permissions: Set[Permission] = None,
                   parent_roles: Set[str] = None,
                   created_by: str = "system") -> Role:
        """
        Create new role
        
        Args:
            name: Unique role name
            display_name: Human-readable display name
            description: Role description
            permissions: Set of permissions for the role
            parent_roles: Set of parent role names for inheritance
            created_by: User creating the role
            
        Returns:
            Created role
            
        Raises:
            ValueError: If role already exists or parent roles invalid
        """
        if name in self.roles:
            raise ValueError(f"Role {name} already exists")
        
        # Validate parent roles
        if parent_roles:
            for parent_role in parent_roles:
                if parent_role not in self.roles:
                    raise ValueError(f"Parent role {parent_role} does not exist")
        
        role = Role(
            name=name,
            display_name=display_name,
            description=description,
            permissions=permissions or set(),
            parent_roles=parent_roles or set(),
            created_by=created_by
        )
        
        self.roles[name] = role
        
        # Clear permission cache
        self._clear_permission_cache()
        
        logger.info(f"Role created: {name} by {created_by}")
        
        return role
    
    def get_role(self, name: str) -> Optional[Role]:
        """Get role by name"""
        return self.roles.get(name)
    
    def list_roles(self) -> List[Role]:
        """List all roles"""
        return list(self.roles.values())
    
    def delete_role(self, name: str, deleted_by: str) -> bool:
        """
        Delete role
        
        Args:
            name: Role name to delete
            deleted_by: User deleting the role
            
        Returns:
            True if role was deleted
            
        Raises:
            ValueError: If role is system role or has dependencies
        """
        if name not in self.roles:
            return False
        
        role = self.roles[name]
        
        # Prevent deletion of system roles
        if role.is_system_role:
            raise ValueError("Cannot delete system roles")
        
        # Check if role is used as parent by other roles
        dependent_roles = [r.name for r in self.roles.values() if name in r.parent_roles]
        if dependent_roles:
            raise ValueError(f"Cannot delete role {name}, used by: {dependent_roles}")
        
        # Check if role is assigned to users
        users_with_role = [u.username for u in self.users.values() if name in u.roles]
        if users_with_role:
            raise ValueError(f"Cannot delete role {name}, assigned to users: {users_with_role}")
        
        del self.roles[name]
        
        # Clear permission cache
        self._clear_permission_cache()
        
        logger.info(f"Role deleted: {name} by {deleted_by}")
        
        return True
    
    def create_user(self,
                   username: str,
                   display_name: str,
                   email: str,
                   roles: Set[str] = None,
                   mfa_enabled: bool = False,
                   created_by: str = "system") -> User:
        """
        Create new user
        
        Args:
            username: Unique username
            display_name: Human-readable display name
            email: User email address
            roles: Set of role names to assign
            mfa_enabled: Whether MFA is enabled for user
            created_by: User creating this user
            
        Returns:
            Created user
            
        Raises:
            ValueError: If user already exists or roles invalid
        """
        if username in self.users:
            raise ValueError(f"User {username} already exists")
        
        # Validate roles
        if roles:
            for role_name in roles:
                if role_name not in self.roles:
                    raise ValueError(f"Role {role_name} does not exist")
        
        user = User(
            username=username,
            display_name=display_name,
            email=email,
            roles=roles or set(),
            mfa_enabled=mfa_enabled
        )
        
        self.users[username] = user
        
        # Clear permission cache for this user
        self._clear_user_permission_cache(username)
        
        logger.info(f"User created: {username} by {created_by}")
        
        return user
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.users.get(username)
    
    def list_users(self) -> List[User]:
        """List all users"""
        return list(self.users.values())
    
    def assign_role_to_user(self, username: str, role_name: str, assigned_by: str) -> bool:
        """
        Assign role to user
        
        Args:
            username: Username to assign role to
            role_name: Role name to assign
            assigned_by: User making the assignment
            
        Returns:
            True if role was assigned
            
        Raises:
            ValueError: If user or role doesn't exist
        """
        if username not in self.users:
            raise ValueError(f"User {username} does not exist")
        
        if role_name not in self.roles:
            raise ValueError(f"Role {role_name} does not exist")
        
        user = self.users[username]
        
        if role_name in user.roles:
            return False  # Already has role
        
        user.add_role(role_name)
        
        # Clear permission cache for this user
        self._clear_user_permission_cache(username)
        
        # Log role assignment
        self._log_rbac_event(
            event_type="role_assigned",
            user=assigned_by,
            details={
                "target_user": username,
                "role": role_name
            }
        )
        
        logger.info(f"Role {role_name} assigned to user {username} by {assigned_by}")
        
        return True
    
    def revoke_role_from_user(self, username: str, role_name: str, revoked_by: str) -> bool:
        """
        Revoke role from user
        
        Args:
            username: Username to revoke role from
            role_name: Role name to revoke
            revoked_by: User making the revocation
            
        Returns:
            True if role was revoked
        """
        if username not in self.users:
            return False
        
        user = self.users[username]
        
        if role_name not in user.roles:
            return False  # Doesn't have role
        
        user.remove_role(role_name)
        
        # Clear permission cache for this user
        self._clear_user_permission_cache(username)
        
        # Log role revocation
        self._log_rbac_event(
            event_type="role_revoked",
            user=revoked_by,
            details={
                "target_user": username,
                "role": role_name
            }
        )
        
        logger.info(f"Role {role_name} revoked from user {username} by {revoked_by}")
        
        return True
    
    def get_user_permissions(self, username: str) -> Set[Permission]:
        """
        Get all permissions for user (including inherited)
        
        Args:
            username: Username to get permissions for
            
        Returns:
            Set of all permissions for the user
        """
        # Check cache first
        if username in self._permission_cache:
            cache_time = self._cache_ttl.get(username, datetime.min)
            if datetime.now() - cache_time < timedelta(minutes=5):  # 5-minute cache
                return self._permission_cache[username]
        
        user = self.users.get(username)
        if not user:
            return set()
        
        all_permissions = set()
        
        # Get permissions from all user roles (including inherited)
        for role_name in user.roles:
            role_permissions = self._get_role_permissions_recursive(role_name)
            all_permissions.update(role_permissions)
        
        # Cache the result
        self._permission_cache[username] = all_permissions
        self._cache_ttl[username] = datetime.now()
        
        return all_permissions
    
    def has_permission(self, username: str, permission: Permission) -> bool:
        """
        Check if user has specific permission
        
        Args:
            username: Username to check
            permission: Permission to check for
            
        Returns:
            True if user has the permission
        """
        user_permissions = self.get_user_permissions(username)
        return permission in user_permissions
    
    def check_access(self, username: str, required_permissions: Set[Permission]) -> bool:
        """
        Check if user has all required permissions
        
        Args:
            username: Username to check
            required_permissions: Set of required permissions
            
        Returns:
            True if user has all required permissions
        """
        user_permissions = self.get_user_permissions(username)
        return required_permissions.issubset(user_permissions)
    
    def get_users_with_permission(self, permission: Permission) -> List[str]:
        """
        Get list of users who have specific permission
        
        Args:
            permission: Permission to search for
            
        Returns:
            List of usernames with the permission
        """
        users_with_permission = []
        
        for username in self.users:
            if self.has_permission(username, permission):
                users_with_permission.append(username)
        
        return users_with_permission
    
    def get_users_with_role(self, role_name: str) -> List[str]:
        """
        Get list of users who have specific role
        
        Args:
            role_name: Role name to search for
            
        Returns:
            List of usernames with the role
        """
        return [username for username, user in self.users.items() if role_name in user.roles]
    
    def _get_role_permissions_recursive(self, role_name: str, visited: Set[str] = None) -> Set[Permission]:
        """
        Get all permissions for role including inherited permissions
        
        Args:
            role_name: Role name to get permissions for
            visited: Set of visited roles to prevent cycles
            
        Returns:
            Set of all permissions for the role
        """
        if visited is None:
            visited = set()
        
        if role_name in visited:
            return set()  # Prevent infinite recursion
        
        visited.add(role_name)
        
        role = self.roles.get(role_name)
        if not role:
            return set()
        
        # Start with direct permissions
        all_permissions = role.permissions.copy()
        
        # Add permissions from parent roles
        for parent_role_name in role.parent_roles:
            parent_permissions = self._get_role_permissions_recursive(parent_role_name, visited.copy())
            all_permissions.update(parent_permissions)
        
        return all_permissions
    
    def _clear_permission_cache(self):
        """Clear all permission cache"""
        self._permission_cache.clear()
        self._cache_ttl.clear()
    
    def _clear_user_permission_cache(self, username: str):
        """Clear permission cache for specific user"""
        self._permission_cache.pop(username, None)
        self._cache_ttl.pop(username, None)
    
    def _log_rbac_event(self, event_type: str, user: str, details: Dict[str, Any]):
        """Log RBAC event to audit store"""
        
        audit_record = {
            "event_type": event_type,
            "user": user,
            "resource": "rbac_manager",
            "action": event_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        
        self.audit_store.append_record(audit_record)
    
    def export_rbac_config(self) -> Dict[str, Any]:
        """Export RBAC configuration"""
        return {
            "roles": {
                name: {
                    "display_name": role.display_name,
                    "description": role.description,
                    "permissions": [p.value for p in role.permissions],
                    "parent_roles": list(role.parent_roles),
                    "is_system_role": role.is_system_role,
                    "created_at": role.created_at.isoformat(),
                    "created_by": role.created_by
                }
                for name, role in self.roles.items()
            },
            "users": {
                username: {
                    "display_name": user.display_name,
                    "email": user.email,
                    "roles": list(user.roles),
                    "is_active": user.is_active,
                    "mfa_enabled": user.mfa_enabled,
                    "created_at": user.created_at.isoformat(),
                    "last_login": user.last_login.isoformat() if user.last_login else None
                }
                for username, user in self.users.items()
            }
        }
    
    def get_rbac_statistics(self) -> Dict[str, Any]:
        """Get RBAC system statistics"""
        return {
            "total_roles": len(self.roles),
            "system_roles": len([r for r in self.roles.values() if r.is_system_role]),
            "custom_roles": len([r for r in self.roles.values() if not r.is_system_role]),
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.is_active]),
            "mfa_enabled_users": len([u for u in self.users.values() if u.mfa_enabled]),
            "permission_cache_size": len(self._permission_cache),
            "total_permissions": len(Permission)
        }
