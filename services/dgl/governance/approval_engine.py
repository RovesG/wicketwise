# Purpose: Approval engine for DGL dual approval workflows
# Author: WicketWise AI, Last Modified: 2024

"""
Approval Engine

Implements comprehensive dual approval workflows for DGL:
- Multi-level approval requirements
- Approval delegation and escalation
- Time-based approval expiration
- Approval audit trails
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
import uuid
import asyncio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .audit import GovernanceAuditStore


logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of approval requests"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalType(Enum):
    """Types of approval requests"""
    RULE_CHANGE = "rule_change"
    STATE_TRANSITION = "state_transition"
    EMERGENCY_ACTION = "emergency_action"
    CONFIGURATION_CHANGE = "configuration_change"
    USER_ACCESS = "user_access"
    SYSTEM_MAINTENANCE = "system_maintenance"


class ApprovalPriority(Enum):
    """Priority levels for approval requests"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ApprovalRequirement:
    """Defines approval requirements for different request types"""
    approval_type: ApprovalType
    required_approvals: int = 2
    required_roles: Set[str] = field(default_factory=set)
    expiry_hours: int = 24
    allow_self_approval: bool = False
    escalation_hours: int = 12
    escalation_roles: Set[str] = field(default_factory=set)


@dataclass
class ApprovalDecision:
    """Individual approval decision"""
    approver: str
    decision: str  # "approve" or "reject"
    timestamp: datetime = field(default_factory=datetime.now)
    comments: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalRequest:
    """Approval request with full context"""
    request_id: str
    approval_type: ApprovalType
    priority: ApprovalPriority
    title: str
    description: str
    requested_by: str
    requested_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    # Request details
    resource: str = ""
    action: str = ""
    current_value: Any = None
    proposed_value: Any = None
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Approval tracking
    status: ApprovalStatus = ApprovalStatus.PENDING
    decisions: List[ApprovalDecision] = field(default_factory=list)
    required_approvals: int = 2
    required_roles: Set[str] = field(default_factory=set)
    
    # Workflow control
    allow_self_approval: bool = False
    escalated: bool = False
    escalated_at: Optional[datetime] = None
    escalation_reason: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def approval_count(self) -> int:
        """Count of approve decisions"""
        return len([d for d in self.decisions if d.decision == "approve"])
    
    @property
    def rejection_count(self) -> int:
        """Count of reject decisions"""
        return len([d for d in self.decisions if d.decision == "reject"])
    
    @property
    def is_expired(self) -> bool:
        """Check if request has expired"""
        return self.expires_at is not None and datetime.now() > self.expires_at
    
    @property
    def is_fully_approved(self) -> bool:
        """Check if request has sufficient approvals"""
        return self.approval_count >= self.required_approvals and self.rejection_count == 0
    
    @property
    def is_rejected(self) -> bool:
        """Check if request has been rejected"""
        return self.rejection_count > 0
    
    @property
    def approvers(self) -> List[str]:
        """Get list of users who have approved"""
        return [d.approver for d in self.decisions if d.decision == "approve"]
    
    @property
    def rejectors(self) -> List[str]:
        """Get list of users who have rejected"""
        return [d.approver for d in self.decisions if d.decision == "reject"]


class ApprovalEngine:
    """
    Comprehensive approval engine for DGL workflows
    
    Manages dual approval processes with role-based requirements,
    escalation, expiration, and comprehensive audit trails.
    """
    
    # Default approval requirements by type
    DEFAULT_REQUIREMENTS = {
        ApprovalType.RULE_CHANGE: ApprovalRequirement(
            approval_type=ApprovalType.RULE_CHANGE,
            required_approvals=2,
            required_roles={"risk_manager", "compliance_officer"},
            expiry_hours=48,
            escalation_hours=24,
            escalation_roles={"senior_manager"}
        ),
        
        ApprovalType.STATE_TRANSITION: ApprovalRequirement(
            approval_type=ApprovalType.STATE_TRANSITION,
            required_approvals=2,
            required_roles={"operations_manager", "risk_manager"},
            expiry_hours=24,
            escalation_hours=12,
            escalation_roles={"senior_manager"}
        ),
        
        ApprovalType.EMERGENCY_ACTION: ApprovalRequirement(
            approval_type=ApprovalType.EMERGENCY_ACTION,
            required_approvals=1,
            required_roles={"senior_manager", "emergency_responder"},
            expiry_hours=2,
            escalation_hours=1,
            escalation_roles={"c_level"}
        ),
        
        ApprovalType.CONFIGURATION_CHANGE: ApprovalRequirement(
            approval_type=ApprovalType.CONFIGURATION_CHANGE,
            required_approvals=2,
            required_roles={"technical_lead", "operations_manager"},
            expiry_hours=72,
            escalation_hours=48,
            escalation_roles={"engineering_manager"}
        ),
        
        ApprovalType.USER_ACCESS: ApprovalRequirement(
            approval_type=ApprovalType.USER_ACCESS,
            required_approvals=2,
            required_roles={"security_officer", "manager"},
            expiry_hours=168,  # 1 week
            escalation_hours=72,
            escalation_roles={"security_manager"}
        ),
        
        ApprovalType.SYSTEM_MAINTENANCE: ApprovalRequirement(
            approval_type=ApprovalType.SYSTEM_MAINTENANCE,
            required_approvals=2,
            required_roles={"operations_manager", "technical_lead"},
            expiry_hours=168,  # 1 week
            escalation_hours=72,
            escalation_roles={"engineering_manager"}
        )
    }
    
    def __init__(self, audit_store: GovernanceAuditStore):
        """
        Initialize approval engine
        
        Args:
            audit_store: Audit store for logging approval activities
        """
        self.audit_store = audit_store
        
        # Active approval requests
        self.pending_requests: Dict[str, ApprovalRequest] = {}
        self.completed_requests: Dict[str, ApprovalRequest] = {}
        
        # Approval requirements (can be customized)
        self.approval_requirements = self.DEFAULT_REQUIREMENTS.copy()
        
        # Approval callbacks
        self.approval_callbacks: List[callable] = []
        self.rejection_callbacks: List[callable] = []
        self.escalation_callbacks: List[callable] = []
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._escalation_task: Optional[asyncio.Task] = None
        
        logger.info("Approval engine initialized")
    
    async def start(self):
        """Start background tasks"""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_requests())
        self._escalation_task = asyncio.create_task(self._check_escalations())
        logger.info("Approval engine background tasks started")
    
    async def stop(self):
        """Stop background tasks"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._escalation_task:
            self._escalation_task.cancel()
        logger.info("Approval engine background tasks stopped")
    
    async def create_approval_request(self,
                                    approval_type: ApprovalType,
                                    title: str,
                                    description: str,
                                    requested_by: str,
                                    priority: ApprovalPriority = ApprovalPriority.MEDIUM,
                                    resource: str = "",
                                    action: str = "",
                                    current_value: Any = None,
                                    proposed_value: Any = None,
                                    impact_assessment: Dict[str, Any] = None,
                                    metadata: Dict[str, Any] = None) -> str:
        """
        Create new approval request
        
        Args:
            approval_type: Type of approval request
            title: Brief title for the request
            description: Detailed description
            requested_by: User requesting approval
            priority: Priority level
            resource: Resource being modified
            action: Action being performed
            current_value: Current value/state
            proposed_value: Proposed new value/state
            impact_assessment: Assessment of change impact
            metadata: Additional metadata
            
        Returns:
            Request ID for tracking
        """
        request_id = f"approval_{approval_type.value}_{uuid.uuid4().hex[:8]}"
        
        # Get approval requirements
        requirements = self.approval_requirements.get(approval_type, self.DEFAULT_REQUIREMENTS[approval_type])
        
        # Calculate expiry time
        expires_at = datetime.now() + timedelta(hours=requirements.expiry_hours)
        
        # Create approval request
        request = ApprovalRequest(
            request_id=request_id,
            approval_type=approval_type,
            priority=priority,
            title=title,
            description=description,
            requested_by=requested_by,
            expires_at=expires_at,
            resource=resource,
            action=action,
            current_value=current_value,
            proposed_value=proposed_value,
            impact_assessment=impact_assessment or {},
            required_approvals=requirements.required_approvals,
            required_roles=requirements.required_roles,
            allow_self_approval=requirements.allow_self_approval,
            metadata=metadata or {}
        )
        
        # Store request
        self.pending_requests[request_id] = request
        
        # Log creation
        self._log_approval_event(
            event_type="approval_request_created",
            request_id=request_id,
            user=requested_by,
            details={
                "approval_type": approval_type.value,
                "priority": priority.value,
                "title": title,
                "expires_at": expires_at.isoformat()
            }
        )
        
        logger.info(f"Approval request created: {request_id} ({approval_type.value}) by {requested_by}")
        
        return request_id
    
    async def submit_approval_decision(self,
                                     request_id: str,
                                     approver: str,
                                     decision: str,
                                     comments: str = "",
                                     approver_roles: Set[str] = None,
                                     metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Submit approval decision
        
        Args:
            request_id: ID of approval request
            approver: User making the decision
            decision: "approve" or "reject"
            comments: Optional comments
            approver_roles: Roles of the approver
            metadata: Additional metadata
            
        Returns:
            Decision result with status
            
        Raises:
            ValueError: If request not found or invalid decision
        """
        if request_id not in self.pending_requests:
            raise ValueError(f"Approval request {request_id} not found")
        
        request = self.pending_requests[request_id]
        
        # Validate decision
        if decision not in ["approve", "reject"]:
            raise ValueError("Decision must be 'approve' or 'reject'")
        
        # Check if request has expired
        if request.is_expired:
            request.status = ApprovalStatus.EXPIRED
            await self._move_to_completed(request_id)
            raise ValueError("Approval request has expired")
        
        # Check self-approval rules
        if not request.allow_self_approval and approver == request.requested_by:
            raise ValueError("Self-approval not allowed for this request type")
        
        # Check if user has already decided
        existing_decision = next((d for d in request.decisions if d.approver == approver), None)
        if existing_decision:
            raise ValueError(f"User {approver} has already provided a decision")
        
        # Check role requirements
        if request.required_roles and approver_roles:
            if not request.required_roles.intersection(approver_roles):
                raise ValueError(f"Approver must have one of these roles: {request.required_roles}")
        
        # Create decision
        approval_decision = ApprovalDecision(
            approver=approver,
            decision=decision,
            comments=comments,
            metadata=metadata or {}
        )
        
        # Add decision to request
        request.decisions.append(approval_decision)
        
        # Update request status
        result = await self._evaluate_request_status(request)
        
        # Log decision
        self._log_approval_event(
            event_type=f"approval_{decision}",
            request_id=request_id,
            user=approver,
            details={
                "decision": decision,
                "comments": comments,
                "approval_count": request.approval_count,
                "rejection_count": request.rejection_count,
                "status": request.status.value
            }
        )
        
        logger.info(f"Approval decision submitted: {request_id} - {decision} by {approver}")
        
        return result
    
    async def get_approval_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get approval request by ID"""
        return (self.pending_requests.get(request_id) or 
                self.completed_requests.get(request_id))
    
    def get_pending_requests(self, 
                           requested_by: str = None,
                           approval_type: ApprovalType = None,
                           priority: ApprovalPriority = None) -> List[ApprovalRequest]:
        """
        Get pending approval requests with optional filters
        
        Args:
            requested_by: Filter by requester
            approval_type: Filter by approval type
            priority: Filter by priority
            
        Returns:
            List of matching pending requests
        """
        requests = list(self.pending_requests.values())
        
        if requested_by:
            requests = [r for r in requests if r.requested_by == requested_by]
        
        if approval_type:
            requests = [r for r in requests if r.approval_type == approval_type]
        
        if priority:
            requests = [r for r in requests if r.priority == priority]
        
        # Sort by priority and creation time
        priority_order = {
            ApprovalPriority.EMERGENCY: 5,
            ApprovalPriority.CRITICAL: 4,
            ApprovalPriority.HIGH: 3,
            ApprovalPriority.MEDIUM: 2,
            ApprovalPriority.LOW: 1
        }
        
        requests.sort(key=lambda r: (priority_order[r.priority], r.requested_at), reverse=True)
        
        return requests
    
    def get_requests_requiring_approval(self, approver: str, approver_roles: Set[str] = None) -> List[ApprovalRequest]:
        """
        Get requests that can be approved by the given user
        
        Args:
            approver: User who could provide approval
            approver_roles: Roles of the potential approver
            
        Returns:
            List of requests the user can approve
        """
        eligible_requests = []
        
        for request in self.pending_requests.values():
            # Skip if already decided by this user
            if any(d.approver == approver for d in request.decisions):
                continue
            
            # Skip if self-approval not allowed and user is requester
            if not request.allow_self_approval and approver == request.requested_by:
                continue
            
            # Check role requirements
            if request.required_roles and approver_roles:
                if not request.required_roles.intersection(approver_roles):
                    continue
            
            # Skip if already fully approved or rejected
            if request.status != ApprovalStatus.PENDING:
                continue
            
            eligible_requests.append(request)
        
        return eligible_requests
    
    async def cancel_approval_request(self, request_id: str, cancelled_by: str, reason: str = "") -> bool:
        """
        Cancel pending approval request
        
        Args:
            request_id: ID of request to cancel
            cancelled_by: User cancelling the request
            reason: Reason for cancellation
            
        Returns:
            True if successfully cancelled
        """
        if request_id not in self.pending_requests:
            return False
        
        request = self.pending_requests[request_id]
        
        # Only requester or admin can cancel
        # (In production, would check admin permissions)
        if cancelled_by != request.requested_by and not cancelled_by.startswith("admin_"):
            raise ValueError("Only requester or admin can cancel approval requests")
        
        request.status = ApprovalStatus.CANCELLED
        request.metadata["cancelled_by"] = cancelled_by
        request.metadata["cancelled_at"] = datetime.now().isoformat()
        request.metadata["cancellation_reason"] = reason
        
        await self._move_to_completed(request_id)
        
        # Log cancellation
        self._log_approval_event(
            event_type="approval_request_cancelled",
            request_id=request_id,
            user=cancelled_by,
            details={
                "reason": reason,
                "original_requester": request.requested_by
            }
        )
        
        logger.info(f"Approval request cancelled: {request_id} by {cancelled_by}")
        
        return True
    
    async def escalate_request(self, request_id: str, escalated_by: str, reason: str = "") -> bool:
        """
        Manually escalate approval request
        
        Args:
            request_id: ID of request to escalate
            escalated_by: User escalating the request
            reason: Reason for escalation
            
        Returns:
            True if successfully escalated
        """
        if request_id not in self.pending_requests:
            return False
        
        request = self.pending_requests[request_id]
        
        if request.escalated:
            return False  # Already escalated
        
        request.escalated = True
        request.escalated_at = datetime.now()
        request.escalation_reason = reason
        
        # Reduce required approvals for escalated requests
        if request.required_approvals > 1:
            request.required_approvals = max(1, request.required_approvals - 1)
        
        # Log escalation
        self._log_approval_event(
            event_type="approval_request_escalated",
            request_id=request_id,
            user=escalated_by,
            details={
                "reason": reason,
                "new_required_approvals": request.required_approvals
            }
        )
        
        # Execute escalation callbacks
        for callback in self.escalation_callbacks:
            try:
                await callback(request, escalated_by, reason)
            except Exception as e:
                logger.error(f"Escalation callback failed: {str(e)}")
        
        logger.info(f"Approval request escalated: {request_id} by {escalated_by}")
        
        return True
    
    def get_approval_statistics(self) -> Dict[str, Any]:
        """Get approval system statistics"""
        total_requests = len(self.pending_requests) + len(self.completed_requests)
        
        # Calculate completion rates
        completed_requests = list(self.completed_requests.values())
        approved_count = len([r for r in completed_requests if r.status == ApprovalStatus.APPROVED])
        rejected_count = len([r for r in completed_requests if r.status == ApprovalStatus.REJECTED])
        expired_count = len([r for r in completed_requests if r.status == ApprovalStatus.EXPIRED])
        cancelled_count = len([r for r in completed_requests if r.status == ApprovalStatus.CANCELLED])
        
        # Calculate average processing times
        processing_times = []
        for request in completed_requests:
            if request.decisions:
                last_decision_time = max(d.timestamp for d in request.decisions)
                processing_time = (last_decision_time - request.requested_at).total_seconds() / 3600
                processing_times.append(processing_time)
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "total_requests": total_requests,
            "pending_requests": len(self.pending_requests),
            "completed_requests": len(self.completed_requests),
            "approval_rate_pct": (approved_count / max(len(completed_requests), 1)) * 100,
            "rejection_rate_pct": (rejected_count / max(len(completed_requests), 1)) * 100,
            "expiry_rate_pct": (expired_count / max(len(completed_requests), 1)) * 100,
            "cancellation_rate_pct": (cancelled_count / max(len(completed_requests), 1)) * 100,
            "avg_processing_time_hours": avg_processing_time,
            "escalated_requests": len([r for r in self.pending_requests.values() if r.escalated])
        }
    
    def add_approval_callback(self, callback: callable):
        """Add callback for approval events"""
        self.approval_callbacks.append(callback)
    
    def add_rejection_callback(self, callback: callable):
        """Add callback for rejection events"""
        self.rejection_callbacks.append(callback)
    
    def add_escalation_callback(self, callback: callable):
        """Add callback for escalation events"""
        self.escalation_callbacks.append(callback)
    
    async def _evaluate_request_status(self, request: ApprovalRequest) -> Dict[str, Any]:
        """Evaluate and update request status after new decision"""
        
        if request.is_rejected:
            request.status = ApprovalStatus.REJECTED
            await self._move_to_completed(request.request_id)
            
            # Execute rejection callbacks
            for callback in self.rejection_callbacks:
                try:
                    await callback(request)
                except Exception as e:
                    logger.error(f"Rejection callback failed: {str(e)}")
            
            return {
                "status": "rejected",
                "final": True,
                "message": "Request has been rejected"
            }
        
        elif request.is_fully_approved:
            request.status = ApprovalStatus.APPROVED
            await self._move_to_completed(request.request_id)
            
            # Execute approval callbacks
            for callback in self.approval_callbacks:
                try:
                    await callback(request)
                except Exception as e:
                    logger.error(f"Approval callback failed: {str(e)}")
            
            return {
                "status": "approved",
                "final": True,
                "message": "Request has been fully approved"
            }
        
        else:
            remaining_approvals = request.required_approvals - request.approval_count
            return {
                "status": "pending",
                "final": False,
                "message": f"Request needs {remaining_approvals} more approval(s)",
                "approvals_received": request.approval_count,
                "approvals_required": request.required_approvals
            }
    
    async def _move_to_completed(self, request_id: str):
        """Move request from pending to completed"""
        if request_id in self.pending_requests:
            request = self.pending_requests.pop(request_id)
            self.completed_requests[request_id] = request
    
    async def _cleanup_expired_requests(self):
        """Background task to clean up expired requests"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                expired_requests = []
                for request_id, request in self.pending_requests.items():
                    if request.is_expired:
                        expired_requests.append(request_id)
                
                for request_id in expired_requests:
                    request = self.pending_requests[request_id]
                    request.status = ApprovalStatus.EXPIRED
                    await self._move_to_completed(request_id)
                    
                    self._log_approval_event(
                        event_type="approval_request_expired",
                        request_id=request_id,
                        user="system",
                        details={"expired_at": datetime.now().isoformat()}
                    )
                    
                    logger.info(f"Approval request expired: {request_id}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
    
    async def _check_escalations(self):
        """Background task to check for automatic escalations"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                for request_id, request in self.pending_requests.items():
                    if request.escalated:
                        continue
                    
                    # Get escalation requirements
                    requirements = self.approval_requirements.get(
                        request.approval_type, 
                        self.DEFAULT_REQUIREMENTS[request.approval_type]
                    )
                    
                    # Check if escalation time has passed
                    escalation_time = request.requested_at + timedelta(hours=requirements.escalation_hours)
                    
                    if datetime.now() > escalation_time:
                        await self.escalate_request(
                            request_id=request_id,
                            escalated_by="system",
                            reason="Automatic escalation due to timeout"
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in escalation task: {str(e)}")
    
    def _log_approval_event(self, event_type: str, request_id: str, user: str, details: Dict[str, Any]):
        """Log approval event to audit store"""
        
        audit_record = {
            "event_type": event_type,
            "user": user,
            "resource": "approval_engine",
            "action": f"{event_type}_{request_id}",
            "details": {
                "request_id": request_id,
                **details
            },
            "timestamp": datetime.now().isoformat()
        }
        
        self.audit_store.append_record(audit_record)
