"""
Pytest configuration and shared fixtures for WicketWise test suite
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, Generator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our modules
import sys
sys.path.append('.')

from unified_configuration import WicketWiseConfig, Environment
from service_container import ServiceContainer
from security_framework import WicketWiseSecurityFramework
from database_layer import MockGraphDatabase, DatabaseService

# ==================== PYTEST CONFIGURATION ====================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "security: Security tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance tests"
    )
    config.addinivalue_line(
        "markers", "api: API tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "external: Tests requiring external services"
    )

def pytest_collection_modifyitems(config, items):
    """Add markers based on test location"""
    for item in items:
        # Add markers based on file path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "api" in str(item.fspath):
            item.add_marker(pytest.mark.api)

# ==================== ASYNC FIXTURES ====================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# ==================== CONFIGURATION FIXTURES ====================

@pytest.fixture(scope="session")
def test_config() -> WicketWiseConfig:
    """Test configuration with safe defaults"""
    config = WicketWiseConfig(environment=Environment.TESTING)
    
    # Override with test-specific values
    config.server.debug_mode = True
    config.data.data_dir = tempfile.mkdtemp()
    config.security.jwt_secret_key = "test-secret-key-not-for-production"
    
    return config

@pytest.fixture(scope="function")
def temp_directory() -> Generator[Path, None, None]:
    """Create temporary directory for test files"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

# ==================== SECURITY FIXTURES ====================

@pytest.fixture(scope="function")
def security_framework(test_config) -> WicketWiseSecurityFramework:
    """Security framework for testing"""
    return WicketWiseSecurityFramework(test_config.security.jwt_secret_key)

@pytest.fixture(scope="function")
def test_user_token(security_framework) -> str:
    """Generate test user token"""
    token = security_framework.auth_manager.authenticate("admin", "ChangeMe123!")
    return token

# ==================== DATABASE FIXTURES ====================

@pytest.fixture(scope="function")
async def mock_database_service() -> DatabaseService:
    """Mock database service for testing"""
    service = DatabaseService()
    
    # Use mock implementations
    service.graph_db = MockGraphDatabase()
    service.vector_db = None  # Skip vector DB for basic tests
    service.timeseries_db = None  # Skip time series DB for basic tests
    
    await service.start()
    yield service
    await service.stop()

# ==================== DATA FIXTURES ====================

@pytest.fixture(scope="function")
def sample_cricket_data() -> pd.DataFrame:
    """Generate sample cricket data for testing"""
    np.random.seed(42)  # For reproducible tests
    
    data = []
    for match_id in range(1, 6):  # 5 matches
        for over in range(1, 21):  # 20 overs
            for ball in range(1, 7):  # 6 balls per over
                data.append({
                    'match_id': match_id,
                    'over': over,
                    'ball': ball,
                    'batter': f'Player_{np.random.randint(1, 12)}',
                    'bowler': f'Bowler_{np.random.randint(1, 8)}',
                    'runs_scored': np.random.choice([0, 1, 2, 3, 4, 6], p=[0.3, 0.3, 0.2, 0.05, 0.1, 0.05]),
                    'is_wicket': np.random.choice([True, False], p=[0.05, 0.95]),
                    'is_boundary': np.random.choice([True, False], p=[0.15, 0.85]),
                    'is_four': np.random.choice([True, False], p=[0.1, 0.9]),
                    'is_six': np.random.choice([True, False], p=[0.05, 0.95]),
                    'venue': f'Stadium_{match_id % 3 + 1}',
                    'date': (datetime.now() - timedelta(days=match_id)).strftime('%Y-%m-%d')
                })
    
    return pd.DataFrame(data)

@pytest.fixture(scope="function")
def sample_enrichment_data() -> Dict[str, Any]:
    """Generate sample enrichment data for testing"""
    return {
        "match": {
            "competition": "Test League",
            "format": "T20",
            "date": "2024-04-15",
            "start_time_local": "19:30",
            "timezone": "Asia/Kolkata"
        },
        "venue": {
            "name": "Test Stadium",
            "city": "Test City",
            "country": "Test Country",
            "latitude": 19.0760,
            "longitude": 72.8777
        },
        "teams": [
            {
                "name": "Team A",
                "short_name": "TMA",
                "is_home": True,
                "players": [
                    {
                        "name": "Player 1",
                        "role": "batter",
                        "batting_style": "RHB",
                        "bowling_style": "RM",
                        "captain": True,
                        "wicket_keeper": False,
                        "playing_xi": True
                    }
                ]
            },
            {
                "name": "Team B",
                "short_name": "TMB",
                "is_home": False,
                "players": [
                    {
                        "name": "Player 2",
                        "role": "bowler",
                        "batting_style": "LHB",
                        "bowling_style": "LF",
                        "captain": False,
                        "wicket_keeper": True,
                        "playing_xi": True
                    }
                ]
            }
        ],
        "weather_hourly": [
            {
                "time_local": "2024-04-15T19:00:00",
                "temperature_c": 28.5,
                "humidity_pct": 65,
                "wind_speed_kph": 12.0,
                "precip_mm": 0.0,
                "weather_code": "clear"
            }
        ],
        "toss": {
            "won_by": "Team A",
            "decision": "bat"
        },
        "confidence_score": 0.95
    }

# ==================== SERVICE FIXTURES ====================

@pytest.fixture(scope="function")
async def service_container() -> ServiceContainer:
    """Service container for testing"""
    container = ServiceContainer()
    
    # Register mock services for testing
    from unittest.mock import AsyncMock
    
    # Mock enrichment service
    enrichment_service = AsyncMock()
    enrichment_service.name = "enrichment_service"
    enrichment_service.start = AsyncMock()
    enrichment_service.stop = AsyncMock()
    enrichment_service.health_check = AsyncMock(return_value={
        "status": "healthy",
        "processed_matches": 0,
        "error_count": 0
    })
    
    container._services["enrichment_service"] = enrichment_service
    
    yield container
    
    # Cleanup
    await container.stop_all_services()

# ==================== API FIXTURES ====================

@pytest.fixture(scope="function")
def mock_api_client():
    """Mock API client for testing"""
    client = MagicMock()
    
    # Mock successful responses
    client.get.return_value.status_code = 200
    client.get.return_value.json.return_value = {"status": "success", "data": {}}
    
    client.post.return_value.status_code = 200
    client.post.return_value.json.return_value = {"status": "success", "data": {}}
    
    return client

# ==================== PERFORMANCE FIXTURES ====================

@pytest.fixture(scope="function")
def performance_timer():
    """Performance timing utility for tests"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()

# ==================== FILE FIXTURES ====================

@pytest.fixture(scope="function")
def sample_csv_file(temp_directory, sample_cricket_data) -> Path:
    """Create sample CSV file for testing"""
    csv_file = temp_directory / "sample_data.csv"
    sample_cricket_data.to_csv(csv_file, index=False)
    return csv_file

@pytest.fixture(scope="function")
def sample_config_file(temp_directory) -> Path:
    """Create sample configuration file for testing"""
    config_file = temp_directory / "test_config.yaml"
    
    config_content = """
server:
  host: "127.0.0.1"
  backend_port: 5001
  frontend_port: 8000

data:
  data_dir: "/tmp/test_data"
  models_dir: "test_models"

security:
  jwt_secret_key: "test-secret-key"
  jwt_expiry_hours: 24

apis:
  openai:
    model: "gpt-4o"
    temperature: 0.1
    max_tokens: 4000
"""
    
    config_file.write_text(config_content)
    return config_file

# ==================== UTILITY FIXTURES ====================

@pytest.fixture(scope="function")
def mock_logger():
    """Mock logger for testing"""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.debug = MagicMock()
    return logger

@pytest.fixture(scope="function")
def capture_logs(caplog):
    """Capture logs during test execution"""
    with caplog.at_level("DEBUG"):
        yield caplog

# ==================== CLEANUP FIXTURES ====================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test"""
    yield
    
    # Clean up any test files
    test_files = [
        "test_wicketwise.db",
        "test_security_audit.log",
        "test_enrichment_cache.json"
    ]
    
    for file in test_files:
        try:
            Path(file).unlink(missing_ok=True)
        except Exception:
            pass

# ==================== PARAMETRIZATION HELPERS ====================

# Common test parameters
TEST_USER_ROLES = ["viewer", "analyst", "admin", "super_admin"]
TEST_API_ENDPOINTS = ["/api/health", "/api/admin/services", "/api/query/player"]
TEST_DATABASE_OPERATIONS = ["create", "read", "update", "delete"]
TEST_ENRICHMENT_STRATEGIES = ["fingerprint", "dna_hash", "hybrid"]

# Pytest parametrize shortcuts
pytest_user_roles = pytest.mark.parametrize("user_role", TEST_USER_ROLES)
pytest_api_endpoints = pytest.mark.parametrize("endpoint", TEST_API_ENDPOINTS)
pytest_db_operations = pytest.mark.parametrize("operation", TEST_DATABASE_OPERATIONS)
pytest_enrichment_strategies = pytest.mark.parametrize("strategy", TEST_ENRICHMENT_STRATEGIES)
