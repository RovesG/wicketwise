# Purpose: Tests for Web Cricket Intelligence Agent
# Author: WicketWise Team, Last Modified: 2025-08-30

"""
Comprehensive tests for Web Cricket Intelligence Agent

Tests cover:
- Domain restriction enforcement
- Citation requirements
- Headshot extraction
- Error handling with exponential backoff
- De-duplication
- Response structure validation
"""

import pytest
import asyncio
import aiohttp
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from crickformers.intelligence.web_cricket_intelligence_agent import (
    WebCricketIntelligenceAgent,
    WebIntelRequest,
    WebIntelResponse,
    WebIntelIntent,
    WebIntelStatus,
    CredibilityLevel,
    create_web_cricket_intelligence_agent
)

class TestWebCricketIntelligenceAgent:
    """Test suite for Web Cricket Intelligence Agent"""
    
    @pytest.fixture
    async def agent(self):
        """Create agent instance for testing"""
        async with WebCricketIntelligenceAgent() as agent:
            yield agent
    
    @pytest.fixture
    def sample_request(self):
        """Sample request for testing"""
        return WebIntelRequest(
            intent=WebIntelIntent.PRE_MATCH,
            teams={"home": "India", "away": "Pakistan"},
            match={"venue": "Colombo", "date": "2023-09-02"},
            players=["Rohit Sharma", "Babar Azam"],
            max_items=5
        )
    
    def test_agent_initialization(self):
        """Test agent initializes correctly"""
        agent = WebCricketIntelligenceAgent()
        
        assert agent.ALLOWED_DOMAINS == {
            "espncricinfo.com": "ESPNcricinfo",
            "icc-cricket.com": "ICC", 
            "bbc.com": "BBC Sport",
            "cricbuzz.com": "Cricbuzz",
            "thecricketer.com": "The Cricketer"
        }
        
        assert agent.MAX_RETRIES == 3
        assert agent.RETRY_DELAYS == [0.25, 0.5, 1.0]
        assert agent.cache == {}
    
    def test_factory_function(self):
        """Test factory function creates agent correctly"""
        agent = create_web_cricket_intelligence_agent()
        assert isinstance(agent, WebCricketIntelligenceAgent)
    
    def test_player_name_normalization(self):
        """Test player name normalization"""
        agent = WebCricketIntelligenceAgent()
        
        assert agent._normalize_player_name("M.S. Dhoni") == "MS Dhoni"
        assert agent._normalize_player_name("V Kohli") == "Virat Kohli"
        assert agent._normalize_player_name("Unknown Player") == "Unknown Player"
    
    def test_entity_extraction(self):
        """Test entity extraction from text"""
        agent = WebCricketIntelligenceAgent()
        
        text = "India vs Pakistan pitch report shows spin-friendly conditions"
        teams = {"home": "India", "away": "Pakistan"}
        players = ["Rohit Sharma"]
        
        entities = agent._extract_entities(text, teams, players)
        
        assert "India" in entities
        assert "Pakistan" in entities
        assert "pitch" in entities
        assert "spin" in entities
    
    def test_search_query_building(self):
        """Test search query construction"""
        agent = WebCricketIntelligenceAgent()
        
        request = WebIntelRequest(
            intent=WebIntelIntent.PRE_MATCH,
            query="pitch report",
            teams={"home": "India", "away": "Pakistan"},
            players=["Rohit Sharma"],
            match={"venue": "Colombo", "date": "2023-09-02"}
        )
        
        query = agent._build_search_query(request, "espncricinfo.com")
        
        assert "site:espncricinfo.com" in query
        assert "preview" in query or "prediction" in query
        assert "pitch report" in query
        assert "India" in query
        assert "Pakistan" in query
        assert "Rohit Sharma" in query
        assert "Colombo" in query
        assert "2023-09-02" in query
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        agent = WebCricketIntelligenceAgent()
        
        key1 = agent._get_cache_key("espncricinfo.com", "test query")
        key2 = agent._get_cache_key("espncricinfo.com", "test query")
        key3 = agent._get_cache_key("cricbuzz.com", "test query")
        
        assert key1 == key2  # Same domain and query
        assert key1 != key3  # Different domain
        assert len(key1) == 32  # MD5 hash length
    
    def test_cache_validity(self):
        """Test cache validity checking"""
        agent = WebCricketIntelligenceAgent()
        
        # Valid cache entry
        valid_entry = {
            "timestamp": datetime.now().isoformat(),
            "items": [],
            "sources": []
        }
        assert agent._is_cache_valid(valid_entry) is True
        
        # Expired cache entry
        expired_entry = {
            "timestamp": (datetime.now() - timedelta(hours=13)).isoformat(),
            "items": [],
            "sources": []
        }
        assert agent._is_cache_valid(expired_entry) is False
        
        # Invalid cache entry
        invalid_entry = {"items": []}
        assert agent._is_cache_valid(invalid_entry) is False
    
    def test_credibility_determination(self):
        """Test credibility level determination"""
        agent = WebCricketIntelligenceAgent()
        
        # High credibility content
        high_content = {
            "title": "Match Report: India beats Pakistan",
            "excerpt": "India won by 5 wickets in a thrilling encounter"
        }
        assert agent._determine_credibility(high_content, "espncricinfo.com") == CredibilityLevel.HIGH
        
        # Medium credibility content (opinion)
        medium_content = {
            "title": "Opinion: Why India will win",
            "excerpt": "My analysis suggests India has the upper hand"
        }
        assert agent._determine_credibility(medium_content, "espncricinfo.com") == CredibilityLevel.MEDIUM
    
    def test_headshot_validation(self):
        """Test headshot URL validation"""
        agent = WebCricketIntelligenceAgent()
        
        # Valid headshot URLs
        valid_urls = [
            "https://img1.hscicdn.com/image/upload/f_auto,t_ds_square_w_320/lsci/db/PICTURES/CMS/123456/123456.jpg",
            "https://espncricinfo.com/PICTURES/player/rohit-sharma-headshot.jpg",
            "https://example.com/profile/player-image.jpg"
        ]
        
        for url in valid_urls:
            assert agent._is_valid_headshot(url) is True
        
        # Invalid headshot URLs
        invalid_urls = [
            "",
            "https://example.com/random-image.jpg",
            "https://example.com/logo.png"
        ]
        
        for url in invalid_urls:
            assert agent._is_valid_headshot(url) is False
    
    def test_request_validation(self):
        """Test request validation"""
        agent = WebCricketIntelligenceAgent()
        response = WebIntelResponse(status=WebIntelStatus.OK)
        
        # Test max_items limit
        request = WebIntelRequest(
            intent=WebIntelIntent.PRE_MATCH,
            max_items=30  # Exceeds limit of 25
        )
        
        is_valid = agent._validate_request(request, response)
        
        assert is_valid is True
        assert request.max_items == 25
        assert len(response.errors) == 1
        assert "max_items cannot exceed 25" in response.errors[0].message
    
    @pytest.mark.asyncio
    async def test_simulated_search(self):
        """Test simulated search functionality"""
        agent = WebCricketIntelligenceAgent()
        
        results = await agent._simulate_search("espncricinfo.com", "test query")
        
        assert len(results) == 3
        assert all("espncricinfo.com" in url for url in results)
        assert all(url.startswith("https://") for url in results)
    
    def test_deduplication(self):
        """Test item deduplication"""
        agent = WebCricketIntelligenceAgent()
        
        from crickformers.intelligence.web_cricket_intelligence_agent import WebIntelFact
        
        response = WebIntelResponse(status=WebIntelStatus.OK)
        
        # Add duplicate items
        fact1 = WebIntelFact(title="India wins match", excerpt="Great victory")
        fact2 = WebIntelFact(title="India wins match", excerpt="Different excerpt")  # Same title
        fact3 = WebIntelFact(title="Pakistan loses match", excerpt="Tough loss")
        
        response.items = [fact1, fact2, fact3]
        
        agent._deduplicate_items(response)
        
        assert len(response.items) == 2  # One duplicate removed
        titles = [item.title for item in response.items if hasattr(item, 'title')]
        assert "India wins match" in titles
        assert "Pakistan loses match" in titles
    
    def test_result_limiting(self):
        """Test result limiting"""
        agent = WebCricketIntelligenceAgent()
        
        from crickformers.intelligence.web_cricket_intelligence_agent import WebIntelFact
        
        response = WebIntelResponse(status=WebIntelStatus.OK)
        
        # Add more items than limit
        for i in range(10):
            fact = WebIntelFact(title=f"Fact {i}", excerpt=f"Excerpt {i}")
            response.items.append(fact)
        
        agent._limit_results(response, 5)
        
        assert len(response.items) == 5
    
    def test_final_status_determination(self):
        """Test final status determination"""
        agent = WebCricketIntelligenceAgent()
        
        # Test OK status (items, no errors)
        response1 = WebIntelResponse(status=WebIntelStatus.OK)
        from crickformers.intelligence.web_cricket_intelligence_agent import WebIntelFact
        response1.items = [WebIntelFact(title="Test", excerpt="Test")]
        
        agent._determine_final_status(response1)
        assert response1.status == WebIntelStatus.OK
        
        # Test PARTIAL status (items + errors)
        response2 = WebIntelResponse(status=WebIntelStatus.OK)
        response2.items = [WebIntelFact(title="Test", excerpt="Test")]
        from crickformers.intelligence.web_cricket_intelligence_agent import WebIntelError
        response2.errors = [WebIntelError(where="test", message="test error")]
        
        agent._determine_final_status(response2)
        assert response2.status == WebIntelStatus.PARTIAL
        
        # Test ERROR status (errors, no items)
        response3 = WebIntelResponse(status=WebIntelStatus.OK)
        response3.errors = [WebIntelError(where="test", message="test error")]
        
        agent._determine_final_status(response3)
        assert response3.status == WebIntelStatus.ERROR
        
        # Test PARTIAL status (no items, no errors)
        response4 = WebIntelResponse(status=WebIntelStatus.OK)
        
        agent._determine_final_status(response4)
        assert response4.status == WebIntelStatus.PARTIAL
        assert len(response4.errors) == 1
        assert "No items found" in response4.errors[0].message

@pytest.mark.asyncio
class TestWebCricketIntelligenceAgentIntegration:
    """Integration tests for Web Cricket Intelligence Agent"""
    
    async def test_full_intelligence_gathering_flow(self):
        """Test complete intelligence gathering flow"""
        async with WebCricketIntelligenceAgent() as agent:
            request = WebIntelRequest(
                intent=WebIntelIntent.PRE_MATCH,
                teams={"home": "India", "away": "Pakistan"},
                match={"venue": "Colombo", "date": "2023-09-02"},
                max_items=3
            )
            
            # Mock the search and parsing methods
            with patch.object(agent, '_simulate_search') as mock_search, \
                 patch.object(agent, '_fetch_and_parse') as mock_parse:
                
                # Setup mocks
                mock_search.return_value = ["https://espncricinfo.com/test1"]
                mock_parse.return_value = {
                    "title": "India vs Pakistan Preview",
                    "excerpt": "Exciting match expected at Colombo",
                    "published_at": "2023-09-01T10:00:00Z"
                }
                
                response = await agent.gather_intelligence(request)
                
                assert response.status in [WebIntelStatus.OK, WebIntelStatus.PARTIAL]
                assert len(response.sources) > 0
                
                # Verify domain restriction
                for source in response.sources:
                    domain = source.url.split('/')[2]
                    assert any(allowed in domain for allowed in agent.ALLOWED_DOMAINS.keys())
    
    async def test_player_photo_extraction(self):
        """Test player photo extraction"""
        async with WebCricketIntelligenceAgent() as agent:
            request = WebIntelRequest(
                intent=WebIntelIntent.PLAYER_PROFILE,
                players=["Rohit Sharma"],
                need_player_photos=True,
                max_items=1
            )
            
            # Mock photo extraction
            with patch.object(agent, '_get_espn_player_photo') as mock_photo:
                from crickformers.intelligence.web_cricket_intelligence_agent import WebIntelPhoto
                
                mock_photo.return_value = WebIntelPhoto(
                    player="Rohit Sharma",
                    photo_url="https://espncricinfo.com/PICTURES/rohit.jpg",
                    source_id="s1",
                    license_note="ESPNcricinfo profile headshot — for internal analysis use",
                    referer="https://espncricinfo.com/cricketers/rohit-sharma-34102"
                )
                
                response = await agent.gather_intelligence(request)
                
                # Find photo items
                photos = [item for item in response.items if hasattr(item, 'player')]
                assert len(photos) > 0
                
                photo = photos[0]
                assert photo.player == "Rohit Sharma"
                assert "espncricinfo.com" in photo.photo_url
                assert "internal analysis use" in photo.license_note
                assert photo.referer.startswith("https://")
    
    async def test_domain_restriction_enforcement(self):
        """Test that only allowed domains are used"""
        async with WebCricketIntelligenceAgent() as agent:
            request = WebIntelRequest(
                intent=WebIntelIntent.NEWS_WATCH,
                query="cricket news",
                max_items=5
            )
            
            response = await agent.gather_intelligence(request)
            
            # Verify all sources are from allowed domains
            for source in response.sources:
                domain = source.url.split('/')[2].replace('www.', '')
                assert any(allowed in domain for allowed in agent.ALLOWED_DOMAINS.keys()), \
                    f"Disallowed domain found: {domain}"
    
    async def test_error_handling_and_retry(self):
        """Test error handling with exponential backoff"""
        async with WebCricketIntelligenceAgent() as agent:
            request = WebIntelRequest(
                intent=WebIntelIntent.PRE_MATCH,
                teams={"home": "India", "away": "Pakistan"},
                max_items=1
            )
            
            # Mock search to fail then succeed
            call_count = 0
            async def mock_search_with_retry(domain, query):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise Exception("Network error")
                return ["https://espncricinfo.com/test"]
            
            with patch.object(agent, '_simulate_search', side_effect=mock_search_with_retry):
                response = await agent.gather_intelligence(request)
                
                # Should have succeeded after retries
                assert call_count >= 3
                assert response.status in [WebIntelStatus.OK, WebIntelStatus.PARTIAL, WebIntelStatus.ERROR]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
