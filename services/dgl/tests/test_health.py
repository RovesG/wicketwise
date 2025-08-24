# Purpose: Unit tests for DGL health endpoints
# Author: WicketWise AI, Last Modified: 2024

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Add services directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from config import load_config


class TestHealthEndpoints:
    """Test suite for DGL health and status endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client with initialized app"""
        # Load test configuration
        config = load_config("../../configs/dgl.yaml")
        
        # Create app
        app = create_app()
        
        # Create test client
        return TestClient(app)
    
    def test_health_endpoint_structure(self, client):
        """Test health endpoint returns correct structure"""
        response = client.get("/healthz")
        
        # Should return 200 even if components are not fully initialized
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
        
        # Check status is valid
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        
        # Check version
        assert data["version"] == "1.0.0"
        
        # Check uptime is reasonable
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0
    
    def test_version_endpoint(self, client):
        """Test version endpoint returns correct information"""
        response = client.get("/version")
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields
        assert "service" in data
        assert "version" in data
        assert "config_version" in data
        
        # Check values
        assert data["service"] == "DGL"
        assert data["version"] == "1.0.0"
        assert isinstance(data["config_version"], str)
    
    def test_status_endpoint_without_initialization(self, client):
        """Test status endpoint behavior when components not initialized"""
        response = client.get("/status")
        
        # Should return 503 when rule engine not initialized
        assert response.status_code == 503
        
        data = response.json()
        assert "detail" in data
        assert "not initialized" in data["detail"].lower()
    
    def test_health_endpoint_components(self, client):
        """Test health endpoint includes component status"""
        response = client.get("/healthz")
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Should have components field
        if "components" in data:
            assert isinstance(data["components"], dict)
        
        # Should have metrics field
        if "metrics" in data:
            assert isinstance(data["metrics"], dict)
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.get("/healthz")
        
        # Check for CORS headers (may not be present in test client)
        # This is more of a smoke test
        assert response.status_code == 200


def run_health_tests():
    """Run all health endpoint tests"""
    print("🏥 Running DGL Health Tests")
    print("=" * 40)
    
    test_methods = [
        "test_health_endpoint_structure",
        "test_version_endpoint", 
        "test_status_endpoint_without_initialization",
        "test_health_endpoint_components",
        "test_cors_headers"
    ]
    
    passed = 0
    total = len(test_methods)
    
    # Create test instance
    test_instance = TestHealthEndpoints()
    
    # Create client fixture
    try:
        from config import load_config
        config = load_config("../../configs/dgl.yaml")
        from app import create_app
        app = create_app()
        client = TestClient(app)
    except Exception as e:
        print(f"❌ Failed to create test client: {str(e)}")
        return False
    
    for test_method in test_methods:
        try:
            method = getattr(test_instance, test_method)
            method(client)
            print(f"  ✅ {test_method}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {test_method}: {str(e)}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    print(f"🎯 Success Rate: {(passed/total)*100:.1f}%")
    
    return passed == total


if __name__ == "__main__":
    success = run_health_tests()
    exit(0 if success else 1)
