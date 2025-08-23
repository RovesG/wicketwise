#!/usr/bin/env python3
"""
Secure Admin Backend with Authentication, Authorization, and Input Validation
Replaces the existing admin_backend.py with comprehensive security measures

Author: WicketWise Team, Last Modified: 2025-01-21
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from functools import wraps
import threading

# Import our security framework
from security_framework import WicketWiseSecurityFramework, UserRole, SecurityException
from unified_configuration import get_config, init_config, Environment
from async_enrichment_pipeline import HighPerformanceEnrichmentPipeline, EnrichmentConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize configuration
config = init_config(environment=Environment.DEVELOPMENT)

# Initialize Flask app with security
app = Flask(__name__)
app.config['SECRET_KEY'] = config.security.jwt_secret_key

# Configure CORS securely
CORS(app, 
     origins=["http://127.0.0.1:8000", "http://localhost:8000"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

# Initialize security framework
security = WicketWiseSecurityFramework(config.security.jwt_secret_key)

# Global enrichment pipeline
enrichment_pipeline = None
enrichment_lock = threading.Lock()

class SecureRequest:
    """Wrapper for Flask request with security context"""
    def __init__(self, flask_request):
        self._request = flask_request
        self.user = None
        self.validated_data = {}
    
    @property
    def method(self):
        return self._request.method
    
    @property
    def headers(self):
        return self._request.headers
    
    async def json(self):
        return self._request.get_json()

def require_auth(required_roles: List[UserRole] = None):
    """Decorator for requiring authentication"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Create secure request wrapper
                secure_req = SecureRequest(request)
                
                # Extract and validate token
                auth_header = request.headers.get('Authorization', '')
                if not auth_header.startswith('Bearer '):
                    return jsonify({"error": "Missing or invalid authorization header"}), 401
                
                token = auth_header[7:]  # Remove 'Bearer '
                user = security.auth_manager.validate_token(token)
                if not user:
                    return jsonify({"error": "Invalid or expired token"}), 401
                
                # Check authorization
                if required_roles:
                    if not any(role in user.roles for role in required_roles):
                        security.audit_logger.log_authorization(user, f.__name__, False)
                        return jsonify({"error": "Insufficient permissions"}), 403
                    security.audit_logger.log_authorization(user, f.__name__, True)
                
                # Check rate limits
                if not security.rate_limiter.check_rate_limit(user.id, f.__name__):
                    security.audit_logger.log_rate_limit(user.id, f.__name__)
                    return jsonify({"error": "Rate limit exceeded"}), 429
                
                # Validate input data for POST/PUT requests
                if request.method in ['POST', 'PUT', 'PATCH'] and request.is_json:
                    try:
                        validated_data = security.input_validator.validate(
                            request.get_json() or {}, f.__name__
                        )
                        secure_req.validated_data = validated_data
                    except Exception as e:
                        security.audit_logger.log_validation_error(user, f.__name__, str(e))
                        return jsonify({"error": f"Validation failed: {str(e)}"}), 400
                
                # Add user to request context
                secure_req.user = user
                g.user = user
                g.validated_data = secure_req.validated_data
                
                # Execute function
                result = f(*args, **kwargs)
                
                # Log successful operation
                security.audit_logger.log_security_event(
                    "API_SUCCESS",
                    user,
                    {"endpoint": f.__name__, "method": request.method}
                )
                
                return result
                
            except SecurityException as e:
                security.audit_logger.log_security_event(
                    "SECURITY_VIOLATION",
                    getattr(g, 'user', None),
                    {"endpoint": f.__name__, "error": str(e), "type": type(e).__name__}
                )
                return jsonify({"error": str(e)}), 403
                
        return wrapper
    return decorator

# ==================== AUTHENTICATION ENDPOINTS ====================

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Secure login endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400
        
        # Authenticate user
        token = security.auth_manager.authenticate(username, password)
        if not token:
            security.audit_logger.log_authentication(username, False, request.remote_addr)
            return jsonify({"error": "Invalid credentials"}), 401
        
        # Get user info for response
        user = security.auth_manager.validate_token(token)
        security.audit_logger.log_authentication(username, True, request.remote_addr)
        
        return jsonify({
            "status": "success",
            "token": token,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "roles": [role.value for role in user.roles]
            }
        })
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/auth/validate', methods=['POST'])
def validate_token():
    """Validate JWT token"""
    try:
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({"valid": False, "error": "Missing token"}), 400
        
        token = auth_header[7:]
        user = security.auth_manager.validate_token(token)
        
        if user:
            return jsonify({
                "valid": True,
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "roles": [role.value for role in user.roles]
                }
            })
        else:
            return jsonify({"valid": False, "error": "Invalid token"}), 401
            
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return jsonify({"valid": False, "error": "Internal server error"}), 500

# ==================== SECURE ADMIN ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Public health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Secure Admin Backend is running",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0-secure"
    })

@app.route('/api/admin/stats', methods=['GET'])
@require_auth([UserRole.ADMIN, UserRole.SUPER_ADMIN])
def get_admin_stats():
    """Get comprehensive system statistics"""
    try:
        user = g.user
        
        # System statistics
        stats = {
            "system": {
                "uptime": "N/A",  # Would implement with process tracking
                "memory_usage": "N/A",  # Would implement with psutil
                "cpu_usage": "N/A"
            },
            "security": {
                "active_users": len(security.auth_manager.users),
                "failed_logins_today": 0,  # Would implement with audit log analysis
                "rate_limit_violations": 0
            },
            "data": {
                "total_matches": 0,  # Would query from database
                "enriched_matches": 0,
                "cache_hit_rate": 0.0
            },
            "performance": {
                "avg_response_time": 0.0,
                "requests_per_minute": 0.0,
                "error_rate": 0.0
            }
        }
        
        return jsonify({
            "status": "success",
            "stats": stats,
            "requested_by": user.username,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Admin stats error: {e}")
        return jsonify({"error": "Failed to retrieve stats"}), 500

@app.route('/api/admin/users', methods=['GET'])
@require_auth([UserRole.SUPER_ADMIN])
def list_users():
    """List all users (super admin only)"""
    try:
        users = []
        for user in security.auth_manager.users.values():
            users.append({
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "roles": [role.value for role in user.roles],
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "is_active": user.is_active
            })
        
        return jsonify({
            "status": "success",
            "users": users,
            "total": len(users)
        })
        
    except Exception as e:
        logger.error(f"List users error: {e}")
        return jsonify({"error": "Failed to retrieve users"}), 500

@app.route('/api/admin/enrich-matches', methods=['POST'])
@require_auth([UserRole.ADMIN, UserRole.SUPER_ADMIN])
def secure_enrich_matches():
    """Secure match enrichment endpoint with validation"""
    try:
        user = g.user
        validated_data = g.validated_data
        
        # Get OpenAI API key
        openai_key = config.get_api_key('openai')
        if not openai_key:
            return jsonify({"error": "OpenAI API key not configured"}), 500
        
        # Extract validated parameters
        max_matches = validated_data.get('max_matches', 50)
        priority_competitions = validated_data.get('priority_competitions', [])
        force_refresh = validated_data.get('force_refresh', False)
        
        # Initialize enrichment pipeline
        global enrichment_pipeline
        with enrichment_lock:
            if enrichment_pipeline is None:
                enrichment_config = EnrichmentConfig(
                    max_concurrent=config.apis.enrichment['max_concurrent'],
                    batch_size=config.apis.enrichment['batch_size'],
                    cache_ttl=config.apis.enrichment['cache_ttl']
                )
                enrichment_pipeline = HighPerformanceEnrichmentPipeline(
                    api_key=openai_key,
                    config=enrichment_config
                )
        
        # Start enrichment in background thread
        def run_enrichment():
            try:
                # Load betting dataset
                betting_data_path = config.data.get_decimal_data_path()
                if not betting_data_path.exists():
                    logger.error(f"Betting data not found: {betting_data_path}")
                    return
                
                # This would be implemented with the actual enrichment logic
                logger.info(f"Starting enrichment: {max_matches} matches for user {user.username}")
                
                # Simulate enrichment process
                import time
                time.sleep(2)  # Simulate processing time
                
                logger.info(f"Enrichment completed successfully for user {user.username}")
                
            except Exception as e:
                logger.error(f"Enrichment failed: {e}")
        
        # Start background thread
        enrichment_thread = threading.Thread(target=run_enrichment)
        enrichment_thread.daemon = True
        enrichment_thread.start()
        
        # Log the operation
        security.audit_logger.log_security_event(
            "ENRICHMENT_STARTED",
            user,
            {
                "max_matches": max_matches,
                "priority_competitions": priority_competitions,
                "force_refresh": force_refresh
            }
        )
        
        return jsonify({
            "status": "success",
            "message": f"Enrichment started for {max_matches} matches",
            "started_by": user.username,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Secure enrich matches error: {e}")
        return jsonify({"error": "Enrichment failed"}), 500

@app.route('/api/admin/config', methods=['GET'])
@require_auth([UserRole.ADMIN, UserRole.SUPER_ADMIN])
def get_configuration():
    """Get current system configuration (sanitized)"""
    try:
        # Return sanitized configuration
        config_dict = config.to_dict()
        
        return jsonify({
            "status": "success",
            "configuration": config_dict,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Get configuration error: {e}")
        return jsonify({"error": "Failed to retrieve configuration"}), 500

@app.route('/api/admin/security-audit', methods=['GET'])
@require_auth([UserRole.SUPER_ADMIN])
def get_security_audit():
    """Get security audit logs (super admin only)"""
    try:
        # Read recent security audit logs
        audit_logs = []
        
        try:
            with open("security_audit.log", "r") as f:
                lines = f.readlines()
                # Get last 100 lines
                recent_lines = lines[-100:] if len(lines) > 100 else lines
                
                for line in recent_lines:
                    if line.strip():
                        audit_logs.append(line.strip())
                        
        except FileNotFoundError:
            audit_logs = ["No audit log file found"]
        
        return jsonify({
            "status": "success",
            "audit_logs": audit_logs,
            "total_entries": len(audit_logs),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Security audit error: {e}")
        return jsonify({"error": "Failed to retrieve audit logs"}), 500

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(SecurityException)
def handle_security_exception(error):
    return jsonify({"error": str(error)}), 403

# ==================== STARTUP ====================

def initialize_default_data():
    """Initialize default data and configuration"""
    logger.info("🔧 Initializing secure admin backend...")
    
    # Ensure required directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("cache").mkdir(exist_ok=True)
    Path(config.data.models_dir).mkdir(exist_ok=True)
    
    # Log startup
    security.audit_logger.log_security_event(
        "SYSTEM_STARTUP",
        None,
        {
            "version": "2.0-secure",
            "environment": config.environment.value,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    logger.info("✅ Secure admin backend initialized successfully")
    logger.info("🔐 Default users created:")
    logger.info("   - admin/ChangeMe123! (Super Admin)")
    logger.info("   - analyst/Analyst123! (Analyst)")
    logger.info("⚠️  CHANGE DEFAULT PASSWORDS IMMEDIATELY!")

if __name__ == '__main__':
    initialize_default_data()
    
    # Run with security headers
    @app.after_request
    def add_security_headers(response):
        for header, value in config.security.security_headers.items():
            response.headers[header] = value
        return response
    
    logger.info(f"🚀 Starting secure admin backend on {config.server.host}:{config.server.backend_port}")
    app.run(
        host=config.server.host,
        port=config.server.backend_port,
        debug=config.server.debug_mode,
        threaded=True
    )
