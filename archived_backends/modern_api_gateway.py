#!/usr/bin/env python3
"""
Modern API Gateway with FastAPI
High-performance, async, production-ready API gateway replacing Flask

Author: WicketWise Team, Last Modified: 2025-01-21
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, Depends, status, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Import our services and security
from service_container import get_container, ServiceHealth
from security_framework import WicketWiseSecurityFramework, UserRole, SecurityException
from unified_configuration import get_config, Environment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize configuration and security
config = get_config()
security = WicketWiseSecurityFramework(config.security.jwt_secret_key)

# FastAPI app with metadata
app = FastAPI(
    title="WicketWise API Gateway",
    description="High-performance cricket analytics platform API",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Security scheme
security_scheme = HTTPBearer()

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# ==================== PYDANTIC MODELS ====================

class AuthRequest(BaseModel):
    """Authentication request model"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)

class AuthResponse(BaseModel):
    """Authentication response model"""
    status: str
    token: str
    user: Dict[str, Any]

class EnrichmentRequest(BaseModel):
    """Match enrichment request model"""
    max_matches: int = Field(50, ge=1, le=1000)
    priority_competitions: List[str] = Field(default_factory=list)
    force_refresh: bool = False

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    message: str
    timestamp: str
    version: str
    services: Optional[Dict[str, Any]] = None

class PlayerQueryRequest(BaseModel):
    """Player query request model"""
    player_name: str = Field(..., min_length=2, max_length=100)
    format: Optional[str] = Field("All", pattern="^(T20|ODI|Test|All)$")
    venue: Optional[str] = Field(None, max_length=200)

class AlignmentRequest(BaseModel):
    """Dataset alignment request model"""
    dataset1_path: str
    dataset2_path: str
    strategy: str = Field("hybrid", pattern="^(fingerprint|dna_hash|hybrid)$")
    similarity_threshold: float = Field(0.8, ge=0.0, le=1.0)

# ==================== AUTHENTICATION ====================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    """Get current authenticated user"""
    try:
        token = credentials.credentials
        user = security.auth_manager.validate_token(token)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user
        
    except SecurityException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

def require_role(required_roles: List[UserRole]):
    """Dependency to require specific roles"""
    def role_checker(user=Depends(get_current_user)):
        if not any(role in user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return user
    return role_checker

# ==================== WEBSOCKET MANAGER ====================

class ConnectionManager:
    """WebSocket connection manager for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str = None):
        """Accept websocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, user_id: str = None):
        """Remove websocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        if user_id and user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)
    
    async def send_personal_message(self, message: Dict[str, Any], user_id: str):
        """Send message to specific user"""
        if user_id in self.user_connections:
            for connection in self.user_connections[user_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    logger.warning(f"Failed to send message to user {user_id}: {e}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connections"""
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to broadcast message: {e}")

manager = ConnectionManager()

# ==================== PUBLIC ENDPOINTS ====================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Public health check endpoint"""
    container = get_container()
    
    try:
        # Get service health status
        service_health = await container.get_health_status()
        
        # Convert to serializable format
        services = {}
        for name, health in service_health.items():
            services[name] = {
                "status": health.status.value,
                "uptime": health.uptime,
                "error": health.error_message
            }
        
        return HealthResponse(
            status="healthy",
            message="WicketWise API Gateway is running",
            timestamp=datetime.utcnow().isoformat(),
            version="2.0.0",
            services=services
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            timestamp=datetime.utcnow().isoformat(),
            version="2.0.0"
        )

@app.post("/api/auth/login", response_model=AuthResponse)
async def login(request: AuthRequest):
    """Secure login endpoint"""
    try:
        token = security.auth_manager.authenticate(request.username, request.password)
        
        if not token:
            security.audit_logger.log_authentication(request.username, False)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        user = security.auth_manager.validate_token(token)
        security.audit_logger.log_authentication(request.username, True)
        
        return AuthResponse(
            status="success",
            token=token,
            user={
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "roles": [role.value for role in user.roles]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )

@app.post("/api/auth/validate")
async def validate_token(user=Depends(get_current_user)):
    """Validate JWT token"""
    return {
        "valid": True,
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "roles": [role.value for role in user.roles]
        }
    }

# ==================== PROTECTED ENDPOINTS ====================

@app.get("/api/admin/services")
async def get_services_status(user=Depends(require_role([UserRole.ADMIN, UserRole.SUPER_ADMIN]))):
    """Get detailed services status"""
    container = get_container()
    service_health = await container.get_health_status()
    
    detailed_status = {}
    for name, health in service_health.items():
        detailed_status[name] = {
            "name": health.name,
            "status": health.status.value,
            "uptime": health.uptime,
            "last_check": health.last_check.isoformat(),
            "error_message": health.error_message,
            "metrics": health.metrics or {}
        }
    
    return {
        "status": "success",
        "services": detailed_status,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/admin/enrich-matches")
async def enrich_matches(
    request: EnrichmentRequest,
    user=Depends(require_role([UserRole.ADMIN, UserRole.SUPER_ADMIN]))
):
    """Secure match enrichment endpoint"""
    try:
        container = get_container()
        enrichment_service = container.resolve("enrichment_service")
        
        # Validate OpenAI API key
        if not config.get_api_key('openai'):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OpenAI API key not configured"
            )
        
        # Create mock matches for demonstration
        # In production, this would load from the actual dataset
        sample_matches = [
            {
                "home": "Mumbai Indians",
                "away": "Chennai Super Kings",
                "venue": "Wankhede Stadium",
                "date": "2024-04-01",
                "competition": "Indian Premier League"
            }
        ] * min(request.max_matches, 10)  # Limit for demo
        
        # Start enrichment process
        logger.info(f"Starting enrichment for {len(sample_matches)} matches by user {user.username}")
        
        # Send real-time update
        await manager.send_personal_message({
            "type": "enrichment_started",
            "matches_count": len(sample_matches),
            "timestamp": datetime.utcnow().isoformat()
        }, user.id)
        
        # In production, this would be done asynchronously
        # results = await enrichment_service.enrich_matches(sample_matches)
        
        # Log the operation
        security.audit_logger.log_security_event(
            "ENRICHMENT_STARTED",
            user,
            {
                "max_matches": request.max_matches,
                "priority_competitions": request.priority_competitions,
                "force_refresh": request.force_refresh
            }
        )
        
        return {
            "status": "success",
            "message": f"Enrichment started for {len(sample_matches)} matches",
            "started_by": user.username,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Enrichment error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enrichment failed: {str(e)}"
        )

@app.post("/api/kg/build")
async def build_knowledge_graph(
    data_path: str,
    user=Depends(require_role([UserRole.ADMIN, UserRole.SUPER_ADMIN]))
):
    """Build knowledge graph"""
    try:
        container = get_container()
        kg_service = container.resolve("knowledge_graph_service")
        
        # Validate data path
        if not Path(data_path).exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Data file not found"
            )
        
        # Send real-time update
        await manager.send_personal_message({
            "type": "kg_build_started",
            "data_path": data_path,
            "timestamp": datetime.utcnow().isoformat()
        }, user.id)
        
        # Build graph (in production, this would be async)
        logger.info(f"Building KG from {data_path} for user {user.username}")
        
        # Mock response for now
        return {
            "status": "success",
            "message": "Knowledge graph building started",
            "data_path": data_path,
            "started_by": user.username,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"KG build error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"KG build failed: {str(e)}"
        )

@app.post("/api/alignment/align")
async def align_datasets(
    request: AlignmentRequest,
    user=Depends(require_role([UserRole.ANALYST, UserRole.ADMIN, UserRole.SUPER_ADMIN]))
):
    """Align two datasets"""
    try:
        container = get_container()
        alignment_service = container.resolve("match_alignment_service")
        
        # Validate paths
        if not Path(request.dataset1_path).exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset 1 not found"
            )
        
        if not Path(request.dataset2_path).exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset 2 not found"
            )
        
        # Start alignment
        logger.info(f"Aligning datasets for user {user.username}")
        
        # Mock response for now
        return {
            "status": "success",
            "message": "Dataset alignment started",
            "strategy": request.strategy,
            "similarity_threshold": request.similarity_threshold,
            "started_by": user.username,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alignment error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Alignment failed: {str(e)}"
        )

@app.post("/api/query/player")
async def query_player(
    request: PlayerQueryRequest,
    user=Depends(get_current_user)
):
    """Query player information"""
    try:
        # In production, this would query the KG service
        logger.info(f"Player query: {request.player_name} by {user.username}")
        
        # Mock response
        return {
            "status": "success",
            "player": {
                "name": request.player_name,
                "format": request.format,
                "venue": request.venue,
                "stats": {
                    "matches": 150,
                    "runs": 4500,
                    "average": 35.5,
                    "strike_rate": 142.3
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Player query error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Player query failed: {str(e)}"
        )

# ==================== WEBSOCKET ENDPOINTS ====================

@app.websocket("/api/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, user_id)
    
    try:
        # Send welcome message
        await manager.send_personal_message({
            "type": "connection_established",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }, user_id)
        
        # Keep connection alive
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Echo back for now (in production, would handle different message types)
            await manager.send_personal_message({
                "type": "echo",
                "original_message": message,
                "timestamp": datetime.utcnow().isoformat()
            }, user_id)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info(f"WebSocket disconnected for user {user_id}")

# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting WicketWise API Gateway")
    
    # Initialize service container
    container = get_container()
    
    try:
        await container.start_all_services()
        logger.info("✅ All services started successfully")
    except Exception as e:
        logger.error(f"❌ Failed to start services: {e}")
        # In production, might want to exit here

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down WicketWise API Gateway")
    
    # Stop all services
    container = get_container()
    await container.stop_all_services()
    
    logger.info("✅ Shutdown complete")

# ==================== DEVELOPMENT SERVER ====================

if __name__ == "__main__":
    # Development server configuration
    uvicorn.run(
        "modern_api_gateway:app",
        host=config.server.host,
        port=config.server.backend_port,
        reload=config.server.debug_mode,
        log_level="info",
        access_log=True
    )
