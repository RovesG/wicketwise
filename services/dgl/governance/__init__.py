# Purpose: DGL governance module for state management and approvals
# Author: WicketWise AI, Last Modified: 2024

"""
DGL Governance Module

Provides governance state management and approval workflows:
- State machine for governance transitions
- Dual approval mechanisms
- Role-based access control (RBAC)
- Multi-factor authentication (MFA) stubs
- Approval workflow orchestration
"""

from .state_machine import GovernanceStateMachine, GovernanceState, StateTransition
from .approval_engine import ApprovalEngine, ApprovalRequest, ApprovalDecision
from .rbac import RBACManager, Role, Permission, User
from .mfa import MFAManager, MFAChallenge, MFAResponse

__all__ = [
    "GovernanceStateMachine",
    "GovernanceState", 
    "StateTransition",
    "ApprovalEngine",
    "ApprovalRequest",
    "ApprovalDecision",
    "RBACManager",
    "Role",
    "Permission", 
    "User",
    "MFAManager",
    "MFAChallenge",
    "MFAResponse"
]
