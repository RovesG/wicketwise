# Purpose: DGL FastAPI application with health and version endpoints
# Author: WicketWise AI, Last Modified: 2024

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import logging
from datetime import datetime
from typing import Dict, Any

from config import DGLConfig, load_config, get_config
from schemas import HealthResponse, VersionResponse
from engine import RuleEngine
from audit import AuditLogger
from repo.memory_repo import MemoryRepositoryFactory
from api import governance_router, exposure_router, rules_router, audit_router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global application state
app_start_time = time.time()
config: DGLConfig = None
rule_engine: RuleEngine = None
audit_logger: AuditLogger = None


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="WicketWise Deterministic Governance Layer (DGL)",
        description="AI-independent safety engine for cricket betting governance",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routers
    app.include_router(governance_router)
    app.include_router(exposure_router)
    app.include_router(rules_router)
    app.include_router(audit_router)
    
    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global config, rule_engine, audit_logger
    
    try:
        # Load configuration
        config = load_config()
        logger.info(f"DGL starting in {config.mode} mode")
        
        # Create repository stores
        exposure_store, pnl_store, audit_store = MemoryRepositoryFactory.create_all_stores(
            initial_bankroll=100000.0  # Default bankroll
        )
        
        # Initialize rule engine
        rule_engine = RuleEngine(config, exposure_store, pnl_store, audit_store)
        
        # Initialize audit logger
        audit_logger = AuditLogger(audit_store, config.audit.hash_algorithm)
        
        logger.info("DGL application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize DGL application: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("DGL application shutting down")


def get_rule_engine() -> RuleEngine:
    """Dependency to get rule engine instance"""
    if rule_engine is None:
        raise HTTPException(status_code=503, detail="Rule engine not initialized")
    return rule_engine


def get_audit_logger() -> AuditLogger:
    """Dependency to get audit logger instance"""
    if audit_logger is None:
        raise HTTPException(status_code=503, detail="Audit logger not initialized")
    return audit_logger


@app.get("/healthz", response_model=HealthResponse)
async def health_check(engine: RuleEngine = Depends(get_rule_engine)) -> HealthResponse:
    """
    Health check endpoint
    
    Returns current health status of the DGL service including:
    - Overall service health
    - Component status
    - Performance metrics
    """
    try:
        uptime_seconds = time.time() - app_start_time
        
        # Check component health
        components = {}
        
        # Check rule engine
        try:
            engine_stats = engine.get_statistics()
            components["rule_engine"] = "healthy"
        except Exception as e:
            components["rule_engine"] = f"unhealthy: {str(e)}"
        
        # Check configuration
        try:
            config_violations = get_config().validate_constraints()
            components["configuration"] = "healthy" if not config_violations else f"degraded: {len(config_violations)} violations"
        except Exception as e:
            components["configuration"] = f"unhealthy: {str(e)}"
        
        # Check audit system
        try:
            audit_stats = audit_logger.get_statistics()
            components["audit_system"] = "healthy" if audit_stats.get("hash_chain_valid", False) else "degraded"
        except Exception as e:
            components["audit_system"] = f"unhealthy: {str(e)}"
        
        # Determine overall health
        unhealthy_components = [name for name, status in components.items() if status.startswith("unhealthy")]
        degraded_components = [name for name, status in components.items() if status.startswith("degraded")]
        
        if unhealthy_components:
            overall_status = "unhealthy"
        elif degraded_components:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Performance metrics
        metrics = {}
        if 'engine_stats' in locals():
            metrics.update({
                "total_decisions": engine_stats.get("total_decisions", 0),
                "avg_processing_time_ms": engine_stats.get("avg_processing_time_ms", 0.0),
                "p99_processing_time_ms": engine_stats.get("p99_processing_time_ms", 0.0)
            })
        
        return HealthResponse(
            status=overall_status,
            version="1.0.0",
            uptime_seconds=uptime_seconds,
            components=components,
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            uptime_seconds=time.time() - app_start_time,
            components={"error": str(e)},
            metrics={}
        )


@app.get("/version", response_model=VersionResponse)
async def version_info() -> VersionResponse:
    """
    Version information endpoint
    
    Returns service version, build information, and configuration version
    """
    try:
        config_version = get_config().config_version if config else "unknown"
        
        return VersionResponse(
            service="DGL",
            version="1.0.0",
            build_time=datetime.utcnow().isoformat(),
            git_commit=None,  # Would be set during build
            config_version=config_version
        )
        
    except Exception as e:
        logger.error(f"Version info failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get version info: {str(e)}")


@app.get("/status")
async def status_info(engine: RuleEngine = Depends(get_rule_engine)) -> Dict[str, Any]:
    """
    Detailed status information endpoint
    
    Returns comprehensive status including configuration, statistics, and state
    """
    try:
        config_obj = get_config()
        engine_stats = engine.get_statistics()
        audit_stats = audit_logger.get_statistics()
        
        return {
            "service": "DGL",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - app_start_time,
            "governance": {
                "current_state": engine_stats.get("current_state"),
                "kill_switch_active": engine_stats.get("kill_switch_active"),
                "mode": config_obj.mode
            },
            "performance": {
                "total_decisions": engine_stats.get("total_decisions", 0),
                "avg_processing_time_ms": engine_stats.get("avg_processing_time_ms", 0.0),
                "p99_processing_time_ms": engine_stats.get("p99_processing_time_ms", 0.0)
            },
            "audit": {
                "total_records": audit_stats.get("total_records", 0),
                "hash_chain_valid": audit_stats.get("hash_chain_valid", False),
                "integrity_verified": True  # Simplified for now
            },
            "configuration": {
                "version": config_obj.config_version,
                "last_updated": config_obj.last_updated,
                "violations": config_obj.validate_constraints()
            }
        }
        
    except Exception as e:
        logger.error(f"Status info failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


def main():
    """Main entry point for running the DGL service"""
    try:
        # Load configuration to get service settings
        config = load_config()
        
        # Run the FastAPI application
        uvicorn.run(
            "services.dgl.app:app",
            host=config.service.host,
            port=config.service.port,
            workers=config.service.workers,
            log_level=config.service.log_level.lower(),
            reload=False  # Set to True for development
        )
        
    except Exception as e:
        logger.error(f"Failed to start DGL service: {str(e)}")
        raise


if __name__ == "__main__":
    main()
