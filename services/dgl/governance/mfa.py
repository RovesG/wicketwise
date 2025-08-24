# Purpose: Multi-Factor Authentication (MFA) manager for DGL governance
# Author: WicketWise AI, Last Modified: 2024

"""
MFA Manager

Implements multi-factor authentication for DGL governance:
- TOTP (Time-based One-Time Password) support
- SMS-based verification (stub)
- Email-based verification (stub)
- Hardware token support (stub)
- MFA challenge management
"""

import logging
import secrets
import hashlib
import hmac
import base64
import struct
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
import uuid

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .audit import GovernanceAuditStore


logger = logging.getLogger(__name__)


class MFAMethod(Enum):
    """Supported MFA methods"""
    TOTP = "totp"           # Time-based One-Time Password (Google Authenticator, etc.)
    SMS = "sms"             # SMS-based verification
    EMAIL = "email"         # Email-based verification
    HARDWARE_TOKEN = "hardware_token"  # Hardware security keys
    BACKUP_CODES = "backup_codes"      # One-time backup codes


class MFAChallengeStatus(Enum):
    """Status of MFA challenges"""
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class MFAChallenge:
    """MFA challenge for verification"""
    challenge_id: str
    user: str
    method: MFAMethod
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(minutes=5))
    status: MFAChallengeStatus = MFAChallengeStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if challenge has expired"""
        return datetime.now() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """Check if challenge is still valid"""
        return (self.status == MFAChallengeStatus.PENDING and 
                not self.is_expired and 
                self.attempts < self.max_attempts)


@dataclass
class MFAResponse:
    """Response to MFA challenge"""
    challenge_id: str
    user: str
    method: MFAMethod
    token: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False
    error_message: str = ""


@dataclass
class MFADevice:
    """MFA device registration"""
    device_id: str
    user: str
    method: MFAMethod
    display_name: str
    secret_key: Optional[str] = None  # For TOTP
    phone_number: Optional[str] = None  # For SMS
    email_address: Optional[str] = None  # For Email
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    backup_codes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MFAManager:
    """
    Multi-Factor Authentication manager for DGL governance
    
    Provides comprehensive MFA support including TOTP, SMS, email,
    and hardware tokens with challenge management and audit logging.
    """
    
    def __init__(self, audit_store: GovernanceAuditStore):
        """
        Initialize MFA manager
        
        Args:
            audit_store: Audit store for logging MFA events
        """
        self.audit_store = audit_store
        
        # MFA device storage
        self.devices: Dict[str, MFADevice] = {}
        
        # Active challenges
        self.active_challenges: Dict[str, MFAChallenge] = {}
        
        # MFA settings
        self.totp_window = 1  # Allow 1 time step before/after current
        self.totp_step = 30   # 30-second time steps
        self.challenge_ttl_minutes = 5
        self.max_attempts = 3
        
        logger.info("MFA manager initialized")
    
    def register_totp_device(self, 
                           user: str, 
                           display_name: str,
                           secret_key: str = None) -> Dict[str, Any]:
        """
        Register TOTP device for user
        
        Args:
            user: Username
            display_name: Human-readable device name
            secret_key: Optional pre-generated secret key
            
        Returns:
            Device registration details including QR code data
        """
        if not secret_key:
            secret_key = self._generate_totp_secret()
        
        device_id = f"totp_{user}_{uuid.uuid4().hex[:8]}"
        
        device = MFADevice(
            device_id=device_id,
            user=user,
            method=MFAMethod.TOTP,
            display_name=display_name,
            secret_key=secret_key
        )
        
        self.devices[device_id] = device
        
        # Generate QR code data for authenticator apps
        qr_data = self._generate_totp_qr_data(user, secret_key, display_name)
        
        # Log device registration
        self._log_mfa_event(
            event_type="mfa_device_registered",
            user=user,
            details={
                "device_id": device_id,
                "method": MFAMethod.TOTP.value,
                "display_name": display_name
            }
        )
        
        logger.info(f"TOTP device registered for user {user}: {device_id}")
        
        return {
            "device_id": device_id,
            "secret_key": secret_key,
            "qr_code_data": qr_data,
            "backup_codes": self._generate_backup_codes(device_id)
        }
    
    def register_sms_device(self, 
                          user: str, 
                          display_name: str,
                          phone_number: str) -> str:
        """
        Register SMS device for user
        
        Args:
            user: Username
            display_name: Human-readable device name
            phone_number: Phone number for SMS
            
        Returns:
            Device ID
        """
        device_id = f"sms_{user}_{uuid.uuid4().hex[:8]}"
        
        device = MFADevice(
            device_id=device_id,
            user=user,
            method=MFAMethod.SMS,
            display_name=display_name,
            phone_number=phone_number
        )
        
        self.devices[device_id] = device
        
        # Log device registration
        self._log_mfa_event(
            event_type="mfa_device_registered",
            user=user,
            details={
                "device_id": device_id,
                "method": MFAMethod.SMS.value,
                "display_name": display_name,
                "phone_number": phone_number[-4:]  # Log only last 4 digits
            }
        )
        
        logger.info(f"SMS device registered for user {user}: {device_id}")
        
        return device_id
    
    def register_email_device(self, 
                            user: str, 
                            display_name: str,
                            email_address: str) -> str:
        """
        Register email device for user
        
        Args:
            user: Username
            display_name: Human-readable device name
            email_address: Email address for verification
            
        Returns:
            Device ID
        """
        device_id = f"email_{user}_{uuid.uuid4().hex[:8]}"
        
        device = MFADevice(
            device_id=device_id,
            user=user,
            method=MFAMethod.EMAIL,
            display_name=display_name,
            email_address=email_address
        )
        
        self.devices[device_id] = device
        
        # Log device registration
        self._log_mfa_event(
            event_type="mfa_device_registered",
            user=user,
            details={
                "device_id": device_id,
                "method": MFAMethod.EMAIL.value,
                "display_name": display_name,
                "email_domain": email_address.split('@')[1] if '@' in email_address else ""
            }
        )
        
        logger.info(f"Email device registered for user {user}: {device_id}")
        
        return device_id
    
    def get_user_devices(self, user: str) -> List[MFADevice]:
        """
        Get all MFA devices for user
        
        Args:
            user: Username
            
        Returns:
            List of user's MFA devices
        """
        return [device for device in self.devices.values() 
                if device.user == user and device.is_active]
    
    def create_mfa_challenge(self, 
                           user: str, 
                           method: MFAMethod = None,
                           device_id: str = None) -> str:
        """
        Create MFA challenge for user
        
        Args:
            user: Username
            method: Preferred MFA method (optional)
            device_id: Specific device ID (optional)
            
        Returns:
            Challenge ID
            
        Raises:
            ValueError: If user has no MFA devices or method not available
        """
        user_devices = self.get_user_devices(user)
        
        if not user_devices:
            raise ValueError(f"User {user} has no registered MFA devices")
        
        # Select device
        selected_device = None
        
        if device_id:
            selected_device = next((d for d in user_devices if d.device_id == device_id), None)
            if not selected_device:
                raise ValueError(f"Device {device_id} not found for user {user}")
        elif method:
            selected_device = next((d for d in user_devices if d.method == method), None)
            if not selected_device:
                raise ValueError(f"No {method.value} device found for user {user}")
        else:
            # Use first available device (prefer TOTP)
            totp_devices = [d for d in user_devices if d.method == MFAMethod.TOTP]
            selected_device = totp_devices[0] if totp_devices else user_devices[0]
        
        challenge_id = f"mfa_{selected_device.method.value}_{uuid.uuid4().hex[:8]}"
        
        challenge = MFAChallenge(
            challenge_id=challenge_id,
            user=user,
            method=selected_device.method,
            metadata={"device_id": selected_device.device_id}
        )
        
        self.active_challenges[challenge_id] = challenge
        
        # Send challenge based on method
        if selected_device.method == MFAMethod.SMS:
            self._send_sms_challenge(selected_device, challenge_id)
        elif selected_device.method == MFAMethod.EMAIL:
            self._send_email_challenge(selected_device, challenge_id)
        # TOTP doesn't need sending - user generates code from their app
        
        # Log challenge creation
        self._log_mfa_event(
            event_type="mfa_challenge_created",
            user=user,
            details={
                "challenge_id": challenge_id,
                "method": selected_device.method.value,
                "device_id": selected_device.device_id
            }
        )
        
        logger.info(f"MFA challenge created for user {user}: {challenge_id}")
        
        return challenge_id
    
    def verify_mfa_challenge(self, 
                           challenge_id: str, 
                           token: str) -> MFAResponse:
        """
        Verify MFA challenge with provided token
        
        Args:
            challenge_id: Challenge ID to verify
            token: MFA token provided by user
            
        Returns:
            MFA response with verification result
        """
        if challenge_id not in self.active_challenges:
            return MFAResponse(
                challenge_id=challenge_id,
                user="unknown",
                method=MFAMethod.TOTP,
                token=token,
                success=False,
                error_message="Challenge not found"
            )
        
        challenge = self.active_challenges[challenge_id]
        
        # Check if challenge is still valid
        if not challenge.is_valid:
            challenge.status = MFAChallengeStatus.EXPIRED if challenge.is_expired else MFAChallengeStatus.FAILED
            
            return MFAResponse(
                challenge_id=challenge_id,
                user=challenge.user,
                method=challenge.method,
                token=token,
                success=False,
                error_message="Challenge expired or exceeded max attempts"
            )
        
        # Increment attempt count
        challenge.attempts += 1
        
        # Get device
        device_id = challenge.metadata.get("device_id")
        device = self.devices.get(device_id)
        
        if not device:
            challenge.status = MFAChallengeStatus.FAILED
            return MFAResponse(
                challenge_id=challenge_id,
                user=challenge.user,
                method=challenge.method,
                token=token,
                success=False,
                error_message="Device not found"
            )
        
        # Verify token based on method
        verification_success = False
        
        if challenge.method == MFAMethod.TOTP:
            verification_success = self._verify_totp_token(device, token)
        elif challenge.method == MFAMethod.SMS:
            verification_success = self._verify_sms_token(challenge_id, token)
        elif challenge.method == MFAMethod.EMAIL:
            verification_success = self._verify_email_token(challenge_id, token)
        elif challenge.method == MFAMethod.BACKUP_CODES:
            verification_success = self._verify_backup_code(device, token)
        
        # Update challenge status
        if verification_success:
            challenge.status = MFAChallengeStatus.VERIFIED
            device.last_used = datetime.now()
            
            # Remove challenge from active list
            del self.active_challenges[challenge_id]
        else:
            if challenge.attempts >= challenge.max_attempts:
                challenge.status = MFAChallengeStatus.FAILED
        
        # Create response
        response = MFAResponse(
            challenge_id=challenge_id,
            user=challenge.user,
            method=challenge.method,
            token=token,
            success=verification_success,
            error_message="" if verification_success else "Invalid token"
        )
        
        # Log verification attempt
        self._log_mfa_event(
            event_type="mfa_verification_attempt",
            user=challenge.user,
            details={
                "challenge_id": challenge_id,
                "method": challenge.method.value,
                "success": verification_success,
                "attempts": challenge.attempts
            }
        )
        
        logger.info(f"MFA verification attempt: {challenge_id} - {'SUCCESS' if verification_success else 'FAILED'}")
        
        return response
    
    def revoke_device(self, device_id: str, revoked_by: str) -> bool:
        """
        Revoke MFA device
        
        Args:
            device_id: Device ID to revoke
            revoked_by: User revoking the device
            
        Returns:
            True if device was revoked
        """
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        device.is_active = False
        
        # Log device revocation
        self._log_mfa_event(
            event_type="mfa_device_revoked",
            user=revoked_by,
            details={
                "device_id": device_id,
                "device_user": device.user,
                "method": device.method.value
            }
        )
        
        logger.info(f"MFA device revoked: {device_id} by {revoked_by}")
        
        return True
    
    def cleanup_expired_challenges(self):
        """Clean up expired MFA challenges"""
        expired_challenges = []
        
        for challenge_id, challenge in self.active_challenges.items():
            if challenge.is_expired:
                challenge.status = MFAChallengeStatus.EXPIRED
                expired_challenges.append(challenge_id)
        
        for challenge_id in expired_challenges:
            del self.active_challenges[challenge_id]
        
        if expired_challenges:
            logger.info(f"Cleaned up {len(expired_challenges)} expired MFA challenges")
    
    def get_mfa_statistics(self) -> Dict[str, Any]:
        """Get MFA system statistics"""
        total_devices = len(self.devices)
        active_devices = len([d for d in self.devices.values() if d.is_active])
        
        devices_by_method = {}
        for method in MFAMethod:
            count = len([d for d in self.devices.values() if d.method == method and d.is_active])
            devices_by_method[method.value] = count
        
        return {
            "total_devices": total_devices,
            "active_devices": active_devices,
            "devices_by_method": devices_by_method,
            "active_challenges": len(self.active_challenges),
            "users_with_mfa": len(set(d.user for d in self.devices.values() if d.is_active))
        }
    
    def _generate_totp_secret(self) -> str:
        """Generate TOTP secret key"""
        return base64.b32encode(secrets.token_bytes(20)).decode('utf-8')
    
    def _generate_totp_qr_data(self, user: str, secret: str, issuer: str) -> str:
        """Generate TOTP QR code data"""
        return f"otpauth://totp/{issuer}:{user}?secret={secret}&issuer={issuer}"
    
    def _generate_backup_codes(self, device_id: str, count: int = 10) -> List[str]:
        """Generate backup codes for device"""
        backup_codes = []
        
        for _ in range(count):
            code = ''.join([str(secrets.randbelow(10)) for _ in range(8)])
            backup_codes.append(code)
        
        # Store backup codes with device
        if device_id in self.devices:
            self.devices[device_id].backup_codes = backup_codes
        
        return backup_codes
    
    def _verify_totp_token(self, device: MFADevice, token: str) -> bool:
        """Verify TOTP token"""
        if not device.secret_key:
            return False
        
        try:
            # Convert token to integer
            token_int = int(token)
        except ValueError:
            return False
        
        # Get current time step
        current_time = int(time.time()) // self.totp_step
        
        # Check token for current time and nearby time steps
        for time_step in range(current_time - self.totp_window, current_time + self.totp_window + 1):
            expected_token = self._generate_totp_token(device.secret_key, time_step)
            if token_int == expected_token:
                return True
        
        return False
    
    def _generate_totp_token(self, secret: str, time_step: int) -> int:
        """Generate TOTP token for given time step"""
        # Decode base32 secret
        key = base64.b32decode(secret)
        
        # Convert time step to bytes
        time_bytes = struct.pack('>Q', time_step)
        
        # Generate HMAC-SHA1
        hmac_digest = hmac.new(key, time_bytes, hashlib.sha1).digest()
        
        # Dynamic truncation
        offset = hmac_digest[-1] & 0x0F
        truncated = struct.unpack('>I', hmac_digest[offset:offset + 4])[0]
        truncated &= 0x7FFFFFFF
        
        # Generate 6-digit token
        token = truncated % 1000000
        
        return token
    
    def _send_sms_challenge(self, device: MFADevice, challenge_id: str):
        """Send SMS challenge (stub implementation)"""
        # Generate 6-digit code
        code = ''.join([str(secrets.randbelow(10)) for _ in range(6)])
        
        # Store code for verification
        challenge = self.active_challenges[challenge_id]
        challenge.metadata["sms_code"] = code
        
        # In production, would send SMS via SMS service
        logger.info(f"SMS challenge sent to {device.phone_number[-4:]} (code: {code})")
    
    def _send_email_challenge(self, device: MFADevice, challenge_id: str):
        """Send email challenge (stub implementation)"""
        # Generate 6-digit code
        code = ''.join([str(secrets.randbelow(10)) for _ in range(6)])
        
        # Store code for verification
        challenge = self.active_challenges[challenge_id]
        challenge.metadata["email_code"] = code
        
        # In production, would send email via email service
        logger.info(f"Email challenge sent to {device.email_address} (code: {code})")
    
    def _verify_sms_token(self, challenge_id: str, token: str) -> bool:
        """Verify SMS token"""
        challenge = self.active_challenges.get(challenge_id)
        if not challenge:
            return False
        
        expected_code = challenge.metadata.get("sms_code")
        return token == expected_code
    
    def _verify_email_token(self, challenge_id: str, token: str) -> bool:
        """Verify email token"""
        challenge = self.active_challenges.get(challenge_id)
        if not challenge:
            return False
        
        expected_code = challenge.metadata.get("email_code")
        return token == expected_code
    
    def _verify_backup_code(self, device: MFADevice, token: str) -> bool:
        """Verify backup code"""
        if token in device.backup_codes:
            # Remove used backup code
            device.backup_codes.remove(token)
            return True
        
        return False
    
    def _log_mfa_event(self, event_type: str, user: str, details: Dict[str, Any]):
        """Log MFA event to audit store"""
        
        audit_record = {
            "event_type": event_type,
            "user": user,
            "resource": "mfa_manager",
            "action": event_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        
        self.audit_store.append_record(audit_record)


# Utility functions for MFA integration

def generate_mfa_qr_code_url(user: str, secret: str, issuer: str = "WicketWise DGL") -> str:
    """Generate QR code URL for TOTP setup"""
    return f"otpauth://totp/{issuer}:{user}?secret={secret}&issuer={issuer}"


def validate_totp_token_format(token: str) -> bool:
    """Validate TOTP token format"""
    return len(token) == 6 and token.isdigit()


def validate_phone_number_format(phone: str) -> bool:
    """Validate phone number format (basic validation)"""
    # Remove common formatting characters
    cleaned = ''.join(c for c in phone if c.isdigit())
    
    # Check length (10-15 digits is reasonable for international numbers)
    return 10 <= len(cleaned) <= 15


def validate_email_format(email: str) -> bool:
    """Validate email format (basic validation)"""
    return '@' in email and '.' in email.split('@')[1]
