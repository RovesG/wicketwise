# Purpose: Governance state machine for DGL system states
# Author: WicketWise AI, Last Modified: 2024

"""
Governance State Machine

Implements deterministic state transitions for DGL governance:
- READY → SHADOW → LIVE → KILLED state flow
- Secure state transitions with validation
- Audit logging for all state changes
- Emergency controls and kill switches
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
import asyncio
from contextlib import asynccontextmanager

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import GovernanceState
from .audit import GovernanceAuditStore


logger = logging.getLogger(__name__)


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted"""
    pass


class StateTransitionReason(Enum):
    """Reasons for state transitions"""
    MANUAL_ACTIVATION = "manual_activation"
    AUTOMATIC_PROGRESSION = "automatic_progression"
    EMERGENCY_STOP = "emergency_stop"
    SYSTEM_ERROR = "system_error"
    COMPLIANCE_VIOLATION = "compliance_violation"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SCHEDULED_MAINTENANCE = "scheduled_maintenance"
    DUAL_APPROVAL_COMPLETE = "dual_approval_complete"
    MFA_VERIFICATION_COMPLETE = "mfa_verification_complete"


@dataclass
class StateTransition:
    """Represents a state transition"""
    from_state: GovernanceState
    to_state: GovernanceState
    reason: StateTransitionReason
    initiated_by: str
    timestamp: datetime = field(default_factory=datetime.now)
    requires_dual_approval: bool = False
    requires_mfa: bool = False
    approval_count: int = 0
    mfa_verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_approved(self) -> bool:
        """Check if transition is fully approved"""
        dual_approval_ok = not self.requires_dual_approval or self.approval_count >= 2
        mfa_ok = not self.requires_mfa or self.mfa_verified
        return dual_approval_ok and mfa_ok


@dataclass
class StateTransitionRule:
    """Rules governing state transitions"""
    from_state: GovernanceState
    to_state: GovernanceState
    allowed: bool = True
    requires_dual_approval: bool = False
    requires_mfa: bool = False
    required_permissions: Set[str] = field(default_factory=set)
    cooldown_minutes: int = 0
    conditions: List[str] = field(default_factory=list)


class GovernanceStateMachine:
    """
    Deterministic state machine for DGL governance
    
    Manages secure transitions between governance states with
    comprehensive validation, approval workflows, and audit logging.
    """
    
    # Define valid state transitions with security requirements
    TRANSITION_RULES = {
        # From READY state
        (GovernanceState.READY, GovernanceState.SHADOW): StateTransitionRule(
            GovernanceState.READY, GovernanceState.SHADOW,
            requires_dual_approval=False,
            requires_mfa=True,
            required_permissions={"governance.activate_shadow"},
            cooldown_minutes=5
        ),
        
        # From SHADOW state
        (GovernanceState.SHADOW, GovernanceState.LIVE): StateTransitionRule(
            GovernanceState.SHADOW, GovernanceState.LIVE,
            requires_dual_approval=True,
            requires_mfa=True,
            required_permissions={"governance.activate_live"},
            cooldown_minutes=30,
            conditions=["shadow_validation_passed", "performance_metrics_acceptable"]
        ),
        
        (GovernanceState.SHADOW, GovernanceState.READY): StateTransitionRule(
            GovernanceState.SHADOW, GovernanceState.READY,
            requires_dual_approval=False,
            requires_mfa=False,
            required_permissions={"governance.deactivate"},
            cooldown_minutes=0
        ),
        
        # From LIVE state
        (GovernanceState.LIVE, GovernanceState.SHADOW): StateTransitionRule(
            GovernanceState.LIVE, GovernanceState.SHADOW,
            requires_dual_approval=True,
            requires_mfa=True,
            required_permissions={"governance.downgrade_to_shadow"},
            cooldown_minutes=10
        ),
        
        (GovernanceState.LIVE, GovernanceState.KILLED): StateTransitionRule(
            GovernanceState.LIVE, GovernanceState.KILLED,
            requires_dual_approval=False,  # Emergency kill switch
            requires_mfa=True,
            required_permissions={"governance.emergency_kill"},
            cooldown_minutes=0
        ),
        
        # From any state to KILLED (emergency)
        (GovernanceState.READY, GovernanceState.KILLED): StateTransitionRule(
            GovernanceState.READY, GovernanceState.KILLED,
            requires_dual_approval=False,
            requires_mfa=True,
            required_permissions={"governance.emergency_kill"},
            cooldown_minutes=0
        ),
        
        (GovernanceState.SHADOW, GovernanceState.KILLED): StateTransitionRule(
            GovernanceState.SHADOW, GovernanceState.KILLED,
            requires_dual_approval=False,
            requires_mfa=True,
            required_permissions={"governance.emergency_kill"},
            cooldown_minutes=0
        ),
        
        # Recovery from KILLED state (requires highest privileges)
        (GovernanceState.KILLED, GovernanceState.READY): StateTransitionRule(
            GovernanceState.KILLED, GovernanceState.READY,
            requires_dual_approval=True,
            requires_mfa=True,
            required_permissions={"governance.recover_from_kill"},
            cooldown_minutes=60,
            conditions=["incident_resolved", "security_clearance_obtained"]
        )
    }
    
    def __init__(self, audit_store: GovernanceAuditStore, initial_state: GovernanceState = GovernanceState.READY):
        """
        Initialize governance state machine
        
        Args:
            audit_store: Audit store for logging state changes
            initial_state: Initial governance state
        """
        self.current_state = initial_state
        self.audit_store = audit_store
        
        # Track pending transitions
        self.pending_transitions: Dict[str, StateTransition] = {}
        
        # Track state history
        self.state_history: List[Dict[str, Any]] = []
        
        # Track last transition times for cooldown enforcement
        self.last_transition_times: Dict[tuple, datetime] = {}
        
        # Emergency controls
        self.emergency_mode = False
        self.kill_switch_active = False
        
        # State change callbacks
        self.state_change_callbacks: List[callable] = []
        
        logger.info(f"Governance state machine initialized in {initial_state.value} state")
        
        # Log initial state
        self._log_state_change(
            from_state=None,
            to_state=initial_state,
            reason=StateTransitionReason.MANUAL_ACTIVATION,
            initiated_by="system",
            metadata={"initialization": True}
        )
    
    @property
    def state(self) -> GovernanceState:
        """Get current governance state"""
        return self.current_state
    
    def get_valid_transitions(self, user_permissions: Set[str] = None) -> List[GovernanceState]:
        """
        Get list of valid transitions from current state
        
        Args:
            user_permissions: Set of user permissions to check
            
        Returns:
            List of valid target states
        """
        valid_states = []
        
        for (from_state, to_state), rule in self.TRANSITION_RULES.items():
            if from_state == self.current_state and rule.allowed:
                # Check permissions if provided
                if user_permissions is not None:
                    if not rule.required_permissions.issubset(user_permissions):
                        continue
                
                # Check cooldown
                if self._is_in_cooldown(from_state, to_state):
                    continue
                
                valid_states.append(to_state)
        
        return valid_states
    
    async def initiate_transition(self,
                                 target_state: GovernanceState,
                                 reason: StateTransitionReason,
                                 initiated_by: str,
                                 user_permissions: Set[str] = None,
                                 metadata: Dict[str, Any] = None) -> str:
        """
        Initiate a state transition
        
        Args:
            target_state: Target governance state
            reason: Reason for transition
            initiated_by: User/system initiating transition
            user_permissions: Set of user permissions
            metadata: Additional metadata
            
        Returns:
            Transition ID for tracking
            
        Raises:
            StateTransitionError: If transition is not allowed
        """
        # Validate transition
        await self._validate_transition(target_state, user_permissions or set())
        
        # Get transition rule
        rule = self.TRANSITION_RULES.get((self.current_state, target_state))
        if not rule:
            raise StateTransitionError(f"No rule defined for transition {self.current_state.value} -> {target_state.value}")
        
        # Create transition request
        transition = StateTransition(
            from_state=self.current_state,
            to_state=target_state,
            reason=reason,
            initiated_by=initiated_by,
            requires_dual_approval=rule.requires_dual_approval,
            requires_mfa=rule.requires_mfa,
            metadata=metadata or {}
        )
        
        transition_id = f"transition_{int(datetime.now().timestamp())}_{len(self.pending_transitions)}"
        
        # Check if immediate execution is possible
        if not rule.requires_dual_approval and not rule.requires_mfa:
            # Execute immediately
            await self._execute_transition(transition)
            logger.info(f"Immediate transition executed: {self.current_state.value} -> {target_state.value}")
            return transition_id
        
        # Store pending transition
        self.pending_transitions[transition_id] = transition
        
        logger.info(f"Transition initiated: {transition_id} ({self.current_state.value} -> {target_state.value})")
        logger.info(f"Requires dual approval: {rule.requires_dual_approval}, Requires MFA: {rule.requires_mfa}")
        
        return transition_id
    
    async def approve_transition(self, transition_id: str, approver: str) -> bool:
        """
        Approve a pending transition
        
        Args:
            transition_id: ID of transition to approve
            approver: User providing approval
            
        Returns:
            True if transition is now fully approved and executed
            
        Raises:
            StateTransitionError: If transition not found or invalid
        """
        if transition_id not in self.pending_transitions:
            raise StateTransitionError(f"Transition {transition_id} not found")
        
        transition = self.pending_transitions[transition_id]
        
        # Prevent self-approval
        if approver == transition.initiated_by:
            raise StateTransitionError("Self-approval not allowed for dual approval transitions")
        
        # Add approval
        transition.approval_count += 1
        transition.metadata.setdefault("approvers", []).append({
            "user": approver,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Approval added to {transition_id} by {approver} (count: {transition.approval_count})")
        
        # Check if fully approved
        if transition.is_approved:
            await self._execute_transition(transition)
            del self.pending_transitions[transition_id]
            logger.info(f"Transition {transition_id} fully approved and executed")
            return True
        
        return False
    
    async def verify_mfa(self, transition_id: str, mfa_token: str) -> bool:
        """
        Verify MFA for a pending transition
        
        Args:
            transition_id: ID of transition
            mfa_token: MFA verification token
            
        Returns:
            True if transition is now fully approved and executed
            
        Raises:
            StateTransitionError: If transition not found or MFA invalid
        """
        if transition_id not in self.pending_transitions:
            raise StateTransitionError(f"Transition {transition_id} not found")
        
        transition = self.pending_transitions[transition_id]
        
        # Simulate MFA verification (in production, would verify against MFA service)
        if not self._verify_mfa_token(mfa_token):
            raise StateTransitionError("Invalid MFA token")
        
        transition.mfa_verified = True
        transition.metadata["mfa_verified_at"] = datetime.now().isoformat()
        
        logger.info(f"MFA verified for transition {transition_id}")
        
        # Check if fully approved
        if transition.is_approved:
            await self._execute_transition(transition)
            del self.pending_transitions[transition_id]
            logger.info(f"Transition {transition_id} fully approved and executed")
            return True
        
        return False
    
    async def emergency_kill(self, initiated_by: str, reason: str, mfa_token: str) -> str:
        """
        Execute emergency kill switch
        
        Args:
            initiated_by: User initiating kill switch
            reason: Reason for emergency kill
            mfa_token: MFA token for verification
            
        Returns:
            Transition ID
            
        Raises:
            StateTransitionError: If MFA verification fails
        """
        # Verify MFA for emergency kill
        if not self._verify_mfa_token(mfa_token):
            raise StateTransitionError("Invalid MFA token for emergency kill")
        
        # Create emergency transition
        transition = StateTransition(
            from_state=self.current_state,
            to_state=GovernanceState.KILLED,
            reason=StateTransitionReason.EMERGENCY_STOP,
            initiated_by=initiated_by,
            requires_dual_approval=False,
            requires_mfa=True,
            mfa_verified=True,
            metadata={
                "emergency_reason": reason,
                "mfa_verified_at": datetime.now().isoformat()
            }
        )
        
        # Execute immediately
        await self._execute_transition(transition)
        
        self.kill_switch_active = True
        self.emergency_mode = True
        
        logger.critical(f"EMERGENCY KILL SWITCH ACTIVATED by {initiated_by}: {reason}")
        
        return f"emergency_kill_{int(datetime.now().timestamp())}"
    
    def get_pending_transitions(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of pending transitions"""
        pending_summary = {}
        
        for transition_id, transition in self.pending_transitions.items():
            pending_summary[transition_id] = {
                "from_state": transition.from_state.value,
                "to_state": transition.to_state.value,
                "reason": transition.reason.value,
                "initiated_by": transition.initiated_by,
                "timestamp": transition.timestamp.isoformat(),
                "requires_dual_approval": transition.requires_dual_approval,
                "requires_mfa": transition.requires_mfa,
                "approval_count": transition.approval_count,
                "mfa_verified": transition.mfa_verified,
                "is_approved": transition.is_approved
            }
        
        return pending_summary
    
    def get_state_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent state change history"""
        return self.state_history[-limit:]
    
    def add_state_change_callback(self, callback: callable):
        """Add callback for state changes"""
        self.state_change_callbacks.append(callback)
    
    async def _validate_transition(self, target_state: GovernanceState, user_permissions: Set[str]):
        """Validate if transition is allowed"""
        
        # Check if transition rule exists
        rule = self.TRANSITION_RULES.get((self.current_state, target_state))
        if not rule or not rule.allowed:
            raise StateTransitionError(f"Transition {self.current_state.value} -> {target_state.value} not allowed")
        
        # Check permissions
        if not rule.required_permissions.issubset(user_permissions):
            missing = rule.required_permissions - user_permissions
            raise StateTransitionError(f"Missing required permissions: {missing}")
        
        # Check cooldown
        if self._is_in_cooldown(self.current_state, target_state):
            cooldown_end = self.last_transition_times.get((self.current_state, target_state), datetime.now())
            raise StateTransitionError(f"Transition in cooldown until {cooldown_end}")
        
        # Check conditions
        for condition in rule.conditions:
            if not await self._check_condition(condition):
                raise StateTransitionError(f"Condition not met: {condition}")
    
    def _is_in_cooldown(self, from_state: GovernanceState, to_state: GovernanceState) -> bool:
        """Check if transition is in cooldown period"""
        rule = self.TRANSITION_RULES.get((from_state, to_state))
        if not rule or rule.cooldown_minutes == 0:
            return False
        
        last_transition = self.last_transition_times.get((from_state, to_state))
        if not last_transition:
            return False
        
        cooldown_end = last_transition + timedelta(minutes=rule.cooldown_minutes)
        return datetime.now() < cooldown_end
    
    async def _check_condition(self, condition: str) -> bool:
        """Check if a transition condition is met"""
        # Mock condition checking - in production would check actual system state
        condition_checks = {
            "shadow_validation_passed": True,  # Would check shadow test results
            "performance_metrics_acceptable": True,  # Would check performance metrics
            "incident_resolved": True,  # Would check incident status
            "security_clearance_obtained": True  # Would check security clearance
        }
        
        return condition_checks.get(condition, False)
    
    def _verify_mfa_token(self, mfa_token: str) -> bool:
        """Verify MFA token (mock implementation)"""
        # Mock MFA verification - in production would verify against MFA service
        return len(mfa_token) >= 6 and mfa_token.isdigit()
    
    async def _execute_transition(self, transition: StateTransition):
        """Execute a validated and approved transition"""
        old_state = self.current_state
        new_state = transition.to_state
        
        # Update state
        self.current_state = new_state
        
        # Record transition time for cooldown
        self.last_transition_times[(old_state, new_state)] = datetime.now()
        
        # Log state change
        await self._log_state_change(
            from_state=old_state,
            to_state=new_state,
            reason=transition.reason,
            initiated_by=transition.initiated_by,
            metadata=transition.metadata
        )
        
        # Execute state change callbacks
        for callback in self.state_change_callbacks:
            try:
                await callback(old_state, new_state, transition)
            except Exception as e:
                logger.error(f"State change callback failed: {str(e)}")
        
        logger.info(f"State transition executed: {old_state.value} -> {new_state.value}")
    
    async def _log_state_change(self,
                               from_state: Optional[GovernanceState],
                               to_state: GovernanceState,
                               reason: StateTransitionReason,
                               initiated_by: str,
                               metadata: Dict[str, Any]):
        """Log state change to audit store"""
        
        # Create a simple audit record for governance events
        audit_record = {
            "event_type": "state_transition",
            "user": initiated_by,
            "resource": "governance_state_machine",
            "action": f"transition_{from_state.value if from_state else 'init'}_to_{to_state.value}",
            "details": {
                "from_state": from_state.value if from_state else None,
                "to_state": to_state.value,
                "reason": reason.value,
                "metadata": metadata
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in audit log
        self.audit_store.append_record(audit_record)
        
        # Add to state history
        self.state_history.append({
            "from_state": from_state.value if from_state else None,
            "to_state": to_state.value,
            "reason": reason.value,
            "initiated_by": initiated_by,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        })
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "current_state": self.current_state.value,
            "emergency_mode": self.emergency_mode,
            "kill_switch_active": self.kill_switch_active,
            "pending_transitions": len(self.pending_transitions),
            "valid_transitions": [state.value for state in self.get_valid_transitions()],
            "last_state_change": self.state_history[-1]["timestamp"] if self.state_history else None,
            "total_state_changes": len(self.state_history)
        }


# Utility functions for state machine management

async def create_governance_state_machine(audit_store: GovernanceAuditStore) -> GovernanceStateMachine:
    """Create and initialize governance state machine"""
    return GovernanceStateMachine(audit_store)


@asynccontextmanager
async def governance_state_context(state_machine: GovernanceStateMachine, target_state: GovernanceState):
    """Context manager for temporary state transitions"""
    original_state = state_machine.current_state
    
    try:
        if original_state != target_state:
            transition_id = await state_machine.initiate_transition(
                target_state=target_state,
                reason=StateTransitionReason.MANUAL_ACTIVATION,
                initiated_by="system_context"
            )
        
        yield state_machine
        
    finally:
        # Restore original state if changed
        if state_machine.current_state != original_state:
            await state_machine.initiate_transition(
                target_state=original_state,
                reason=StateTransitionReason.MANUAL_ACTIVATION,
                initiated_by="system_context"
            )
