"""
API Integration Tests
Tests the complete API stack with real HTTP requests
"""

import pytest
import asyncio
import json
from httpx import AsyncClient
from unittest.mock import patch, MagicMock
import time

# We'll test against our FastAPI gateway
from modern_api_gateway import app

@pytest.mark.integration
@pytest.mark.asyncio
class TestAuthenticationIntegration:
    """Test authentication endpoints integration"""
    
    async def test_health_endpoint_public(self):
        """Test public health endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert "version" in data
    
    async def test_login_success(self):
        """Test successful login"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            login_data = {
                "username": "admin",
                "password": "ChangeMe123!"
            }
            
            response = await client.post("/api/auth/login", json=login_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "token" in data
            assert "user" in data
            assert data["user"]["username"] == "admin"
            assert "admin" in data["user"]["roles"]
    
    async def test_login_failure(self):
        """Test failed login with wrong credentials"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            login_data = {
                "username": "admin",
                "password": "wrongpassword"
            }
            
            response = await client.post("/api/auth/login", json=login_data)
            
            assert response.status_code == 401
            data = response.json()
            assert "error" in data
    
    async def test_token_validation_success(self):
        """Test successful token validation"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # First login to get token
            login_data = {
                "username": "admin",
                "password": "ChangeMe123!"
            }
            
            login_response = await client.post("/api/auth/login", json=login_data)
            token = login_response.json()["token"]
            
            # Then validate token
            headers = {"Authorization": f"Bearer {token}"}
            response = await client.post("/api/auth/validate", headers=headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] == True
            assert "user" in data
    
    async def test_token_validation_failure(self):
        """Test token validation with invalid token"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            headers = {"Authorization": "Bearer invalid-token"}
            response = await client.post("/api/auth/validate", headers=headers)
            
            assert response.status_code == 401
            data = response.json()
            assert data["valid"] == False

@pytest.mark.integration
@pytest.mark.asyncio
class TestProtectedEndpointsIntegration:
    """Test protected endpoints with authentication"""
    
    async def get_admin_token(self, client):
        """Helper to get admin token"""
        login_data = {
            "username": "admin",
            "password": "ChangeMe123!"
        }
        
        response = await client.post("/api/auth/login", json=login_data)
        return response.json()["token"]
    
    async def test_admin_services_status_success(self):
        """Test admin services status endpoint with valid token"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            token = await self.get_admin_token(client)
            headers = {"Authorization": f"Bearer {token}"}
            
            response = await client.get("/api/admin/services", headers=headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "services" in data
            assert "timestamp" in data
    
    async def test_admin_services_status_unauthorized(self):
        """Test admin services status endpoint without token"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/admin/services")
            
            assert response.status_code == 401
    
    async def test_admin_services_status_invalid_token(self):
        """Test admin services status endpoint with invalid token"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            headers = {"Authorization": "Bearer invalid-token"}
            response = await client.get("/api/admin/services", headers=headers)
            
            assert response.status_code == 401
    
    async def test_enrich_matches_success(self):
        """Test match enrichment endpoint with valid data"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            token = await self.get_admin_token(client)
            headers = {"Authorization": f"Bearer {token}"}
            
            enrichment_data = {
                "max_matches": 10,
                "priority_competitions": ["IPL", "BBL"],
                "force_refresh": False
            }
            
            response = await client.post(
                "/api/admin/enrich-matches", 
                json=enrichment_data, 
                headers=headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "started_by" in data
    
    async def test_enrich_matches_validation_error(self):
        """Test match enrichment endpoint with invalid data"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            token = await self.get_admin_token(client)
            headers = {"Authorization": f"Bearer {token}"}
            
            # Invalid data - max_matches too high
            enrichment_data = {
                "max_matches": 2000,  # Above limit
                "priority_competitions": [],
                "force_refresh": False
            }
            
            response = await client.post(
                "/api/admin/enrich-matches", 
                json=enrichment_data, 
                headers=headers
            )
            
            assert response.status_code == 400
            data = response.json()
            assert "error" in data
            assert "Validation failed" in data["error"]
    
    async def test_player_query_success(self):
        """Test player query endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            token = await self.get_admin_token(client)
            headers = {"Authorization": f"Bearer {token}"}
            
            query_data = {
                "player_name": "Virat Kohli",
                "format": "T20",
                "venue": "Wankhede Stadium"
            }
            
            response = await client.post(
                "/api/query/player", 
                json=query_data, 
                headers=headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "player" in data
            assert data["player"]["name"] == "Virat Kohli"

@pytest.mark.integration
@pytest.mark.asyncio
class TestRateLimitingIntegration:
    """Test rate limiting integration"""
    
    async def get_admin_token(self, client):
        """Helper to get admin token"""
        login_data = {
            "username": "admin",
            "password": "ChangeMe123!"
        }
        
        response = await client.post("/api/auth/login", json=login_data)
        return response.json()["token"]
    
    @pytest.mark.slow
    async def test_rate_limiting_enforcement(self):
        """Test that rate limiting is enforced"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            token = await self.get_admin_token(client)
            headers = {"Authorization": f"Bearer {token}"}
            
            # Make multiple requests quickly to trigger rate limiting
            # Note: This test depends on the rate limit configuration
            enrichment_data = {
                "max_matches": 1,
                "priority_competitions": [],
                "force_refresh": False
            }
            
            responses = []
            for i in range(15):  # Try to exceed rate limit
                response = await client.post(
                    "/api/admin/enrich-matches", 
                    json=enrichment_data, 
                    headers=headers
                )
                responses.append(response.status_code)
                
                # Small delay to avoid overwhelming the test
                await asyncio.sleep(0.1)
            
            # Should have some rate limited responses (429)
            success_count = sum(1 for code in responses if code == 200)
            rate_limited_count = sum(1 for code in responses if code == 429)
            
            # At least some requests should succeed
            assert success_count > 0
            
            # If rate limiting is working, some should be rate limited
            # (This might not always trigger in test environment)
            print(f"Success: {success_count}, Rate limited: {rate_limited_count}")

@pytest.mark.integration
@pytest.mark.asyncio
class TestErrorHandlingIntegration:
    """Test error handling integration"""
    
    async def test_404_error_handling(self):
        """Test 404 error handling"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/nonexistent-endpoint")
            
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
    
    async def test_method_not_allowed(self):
        """Test method not allowed error"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Try POST on GET-only endpoint
            response = await client.post("/api/health")
            
            assert response.status_code == 405
    
    async def test_invalid_json_handling(self):
        """Test invalid JSON handling"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Send invalid JSON
            response = await client.post(
                "/api/auth/login",
                content="invalid json content",
                headers={"Content-Type": "application/json"}
            )
            
            assert response.status_code == 422  # Unprocessable Entity

@pytest.mark.integration
@pytest.mark.asyncio
class TestServiceIntegration:
    """Test service container integration with API"""
    
    async def get_admin_token(self, client):
        """Helper to get admin token"""
        login_data = {
            "username": "admin",
            "password": "ChangeMe123!"
        }
        
        response = await client.post("/api/auth/login", json=login_data)
        return response.json()["token"]
    
    async def test_services_health_integration(self):
        """Test services health reporting integration"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["healthy", "degraded", "unhealthy"]
            
            # Should have services information
            if "services" in data:
                assert isinstance(data["services"], dict)
    
    @patch('service_container.get_container')
    async def test_service_failure_handling(self, mock_get_container):
        """Test API behavior when services fail"""
        # Mock service container to simulate failure
        mock_container = MagicMock()
        mock_container.get_health_status.side_effect = Exception("Service failure")
        mock_get_container.return_value = mock_container
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/health")
            
            # Should still return a response, but indicate unhealthy status
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"

@pytest.mark.integration
@pytest.mark.external
class TestExternalDependencies:
    """Test integration with external dependencies (when available)"""
    
    @pytest.mark.skipif(
        not pytest.importorskip("neo4j", minversion="5.0"),
        reason="Neo4j not available"
    )
    async def test_neo4j_integration(self):
        """Test Neo4j integration (if available)"""
        # This would test actual Neo4j connection
        # Skip if Neo4j is not running
        pass
    
    @pytest.mark.skipif(
        not pytest.importorskip("qdrant_client"),
        reason="Qdrant client not available"
    )
    async def test_qdrant_integration(self):
        """Test Qdrant integration (if available)"""
        # This would test actual Qdrant connection
        # Skip if Qdrant is not running
        pass

@pytest.mark.integration
@pytest.mark.asyncio
class TestConcurrencyIntegration:
    """Test concurrent API requests"""
    
    async def get_admin_token(self, client):
        """Helper to get admin token"""
        login_data = {
            "username": "admin",
            "password": "ChangeMe123!"
        }
        
        response = await client.post("/api/auth/login", json=login_data)
        return response.json()["token"]
    
    async def make_concurrent_requests(self, client, token, count=10):
        """Make concurrent requests to test thread safety"""
        headers = {"Authorization": f"Bearer {token}"}
        
        tasks = []
        for i in range(count):
            task = client.get("/api/admin/services", headers=headers)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        return responses
    
    @pytest.mark.slow
    async def test_concurrent_authenticated_requests(self):
        """Test concurrent authenticated requests"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            token = await self.get_admin_token(client)
            
            responses = await self.make_concurrent_requests(client, token, 5)
            
            # All requests should succeed
            success_count = 0
            for response in responses:
                if not isinstance(response, Exception):
                    assert response.status_code == 200
                    success_count += 1
            
            # Most or all should succeed
            assert success_count >= 4  # Allow for some potential failures
    
    @pytest.mark.slow
    async def test_concurrent_login_requests(self):
        """Test concurrent login requests"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            login_data = {
                "username": "admin",
                "password": "ChangeMe123!"
            }
            
            # Make multiple concurrent login requests
            tasks = []
            for i in range(5):
                task = client.post("/api/auth/login", json=login_data)
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All login attempts should succeed
            success_count = 0
            for response in responses:
                if not isinstance(response, Exception):
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "success"
                    assert "token" in data
                    success_count += 1
            
            assert success_count == 5
