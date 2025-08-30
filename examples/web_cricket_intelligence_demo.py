#!/usr/bin/env python3
# Purpose: Demo of Web Cricket Intelligence Agent integration
# Author: WicketWise Team, Last Modified: 2025-08-30

"""
Web Cricket Intelligence Agent Demo

Shows how the new Web Cricket Intelligence Agent integrates with:
- Player Cards API
- Betting Intelligence System
- Existing agent architecture

This demonstrates the "compatible" design following existing patterns.
"""

import asyncio
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from crickformers.intelligence import (
    WebCricketIntelligenceAgent,
    WebIntelRequest,
    WebIntelIntent,
    WEB_CRICKET_INTEL_AVAILABLE
)

async def demo_pre_match_intelligence():
    """Demo: Pre-match intelligence gathering"""
    print("🏏 Demo: Pre-match Intelligence Gathering")
    print("=" * 50)
    
    if not WEB_CRICKET_INTEL_AVAILABLE:
        print("❌ Web Cricket Intelligence Agent not available")
        return
    
    async with WebCricketIntelligenceAgent() as agent:
        request = WebIntelRequest(
            intent=WebIntelIntent.PRE_MATCH,
            teams={"home": "India", "away": "Pakistan"},
            match={"venue": "Colombo", "date": "2023-09-02"},
            players=["Rohit Sharma", "Babar Azam"],
            max_items=3
        )
        
        print(f"📋 Request: {request.intent.value}")
        print(f"🏟️ Match: {request.teams['home']} vs {request.teams['away']} at {request.match['venue']}")
        print(f"👥 Players: {', '.join(request.players)}")
        print()
        
        response = await agent.gather_intelligence(request)
        
        print(f"📊 Response Status: {response.status.value}")
        print(f"📰 Items Found: {len(response.items)}")
        print(f"📚 Sources: {len(response.sources)}")
        print(f"⚠️ Errors: {len(response.errors)}")
        print()
        
        # Show sample items
        for i, item in enumerate(response.items[:2], 1):
            if hasattr(item, 'title'):  # Fact item
                print(f"📰 Fact {i}:")
                print(f"   Title: {item.title}")
                print(f"   Excerpt: {item.excerpt[:100]}...")
                print(f"   Entities: {', '.join(item.entities[:3])}")
                print(f"   Credibility: {item.credibility.value}")
                print(f"   Source: {item.source_id}")
            else:  # Photo item
                print(f"📸 Photo {i}:")
                print(f"   Player: {item.player}")
                print(f"   URL: {item.photo_url}")
                print(f"   License: {item.license_note}")
            print()
        
        # Show sources
        for source in response.sources[:2]:
            print(f"📚 Source {source.id}:")
            print(f"   Publisher: {source.publisher}")
            print(f"   URL: {source.url}")
            print(f"   Retrieved: {source.retrieved_at}")
            print()

async def demo_player_profile_with_photos():
    """Demo: Player profile with headshot extraction"""
    print("👤 Demo: Player Profile with Photos")
    print("=" * 50)
    
    if not WEB_CRICKET_INTEL_AVAILABLE:
        print("❌ Web Cricket Intelligence Agent not available")
        return
    
    async with WebCricketIntelligenceAgent() as agent:
        request = WebIntelRequest(
            intent=WebIntelIntent.PLAYER_PROFILE,
            players=["MS Dhoni", "Virat Kohli"],
            need_player_photos=True,
            max_items=5
        )
        
        print(f"📋 Request: {request.intent.value}")
        print(f"👥 Players: {', '.join(request.players)}")
        print(f"📸 Photos Requested: {request.need_player_photos}")
        print()
        
        response = await agent.gather_intelligence(request)
        
        print(f"📊 Response Status: {response.status.value}")
        print(f"📰 Total Items: {len(response.items)}")
        
        # Separate facts and photos
        facts = [item for item in response.items if hasattr(item, 'title')]
        photos = [item for item in response.items if hasattr(item, 'player')]
        
        print(f"📰 Facts: {len(facts)}")
        print(f"📸 Photos: {len(photos)}")
        print()
        
        # Show photos
        for photo in photos:
            print(f"📸 Player Photo:")
            print(f"   Player: {photo.player}")
            print(f"   URL: {photo.photo_url}")
            print(f"   Referer: {photo.referer}")
            print(f"   License: {photo.license_note}")
            print()

async def demo_integration_with_player_cards():
    """Demo: Integration with Player Cards API"""
    print("🎴 Demo: Integration with Player Cards")
    print("=" * 50)
    
    if not WEB_CRICKET_INTEL_AVAILABLE:
        print("❌ Web Cricket Intelligence Agent not available")
        return
    
    # Simulate how the Player Cards API would use this agent
    player_name = "Rohit Sharma"
    
    async with WebCricketIntelligenceAgent() as web_agent:
        # Get recent form information
        form_request = WebIntelRequest(
            intent=WebIntelIntent.NEWS_WATCH,
            query=f"{player_name} recent form performance",
            players=[player_name],
            max_items=3
        )
        
        form_response = await web_agent.gather_intelligence(form_request)
        
        # Get injury status
        injury_request = WebIntelRequest(
            intent=WebIntelIntent.NEWS_WATCH,
            query=f"{player_name} injury fitness status",
            players=[player_name],
            max_items=2
        )
        
        injury_response = await web_agent.gather_intelligence(injury_request)
        
        # Get player photo
        photo_request = WebIntelRequest(
            intent=WebIntelIntent.PLAYER_PROFILE,
            players=[player_name],
            need_player_photos=True,
            max_items=1
        )
        
        photo_response = await web_agent.gather_intelligence(photo_request)
        
        print(f"🎴 Enhanced Player Card Data for {player_name}:")
        print()
        
        # Form intelligence
        print("📈 Recent Form Intelligence:")
        for item in form_response.items[:2]:
            if hasattr(item, 'title'):
                print(f"   • {item.title}")
                print(f"     {item.excerpt[:80]}...")
        print()
        
        # Injury intelligence
        print("🏥 Injury/Fitness Intelligence:")
        for item in injury_response.items[:2]:
            if hasattr(item, 'title'):
                print(f"   • {item.title}")
                print(f"     {item.excerpt[:80]}...")
        print()
        
        # Photo
        photos = [item for item in photo_response.items if hasattr(item, 'player')]
        if photos:
            photo = photos[0]
            print("📸 Player Photo:")
            print(f"   URL: {photo.photo_url}")
            print(f"   License: {photo.license_note}")
        print()
        
        # Show how this enhances the player card
        print("🎯 Enhanced Player Card Output:")
        print(json.dumps({
            "player_name": player_name,
            "recent_form_intelligence": [
                {
                    "title": item.title,
                    "excerpt": item.excerpt,
                    "source": item.source_id,
                    "credibility": item.credibility.value
                }
                for item in form_response.items[:2] if hasattr(item, 'title')
            ],
            "injury_status_intelligence": [
                {
                    "title": item.title,
                    "excerpt": item.excerpt,
                    "source": item.source_id,
                    "credibility": item.credibility.value
                }
                for item in injury_response.items[:2] if hasattr(item, 'title')
            ],
            "player_photo": {
                "url": photos[0].photo_url if photos else None,
                "license": photos[0].license_note if photos else None,
                "referer": photos[0].referer if photos else None
            } if photos else None,
            "intelligence_sources": [
                {
                    "id": source.id,
                    "publisher": source.publisher,
                    "url": source.url,
                    "retrieved_at": source.retrieved_at
                }
                for source in (form_response.sources + injury_response.sources + photo_response.sources)
            ]
        }, indent=2))

async def demo_domain_restrictions():
    """Demo: Domain restriction enforcement"""
    print("🔒 Demo: Domain Restriction Enforcement")
    print("=" * 50)
    
    if not WEB_CRICKET_INTEL_AVAILABLE:
        print("❌ Web Cricket Intelligence Agent not available")
        return
    
    agent = WebCricketIntelligenceAgent()
    
    print("✅ Allowed Domains:")
    for domain, publisher in agent.ALLOWED_DOMAINS.items():
        print(f"   • {domain} ({publisher})")
    print()
    
    print("🔍 Search Query Examples:")
    
    request = WebIntelRequest(
        intent=WebIntelIntent.PRE_MATCH,
        query="pitch report",
        teams={"home": "India", "away": "Pakistan"},
        match={"venue": "Mumbai"}
    )
    
    for domain in list(agent.ALLOWED_DOMAINS.keys())[:3]:
        query = agent._build_search_query(request, domain)
        print(f"   {domain}: {query}")
    print()
    
    print("📋 Non-negotiable Requirements:")
    print("   ✅ Only allowed domains")
    print("   ✅ Citations required for all facts")
    print("   ✅ ESPNcricinfo headshots with proper licensing")
    print("   ✅ Exponential backoff retry (250ms → 500ms → 1s)")
    print("   ✅ De-duplication across domain mirrors")
    print("   ✅ UTC timestamps in ISO 8601")

async def main():
    """Run all demos"""
    print("🌐 Web Cricket Intelligence Agent Demo")
    print("🏏 WicketWise Compatible Agent Architecture")
    print("=" * 60)
    print()
    
    if not WEB_CRICKET_INTEL_AVAILABLE:
        print("❌ Web Cricket Intelligence Agent not available")
        print("   Missing dependencies: aiohttp, beautifulsoup4")
        return
    
    demos = [
        demo_pre_match_intelligence,
        demo_player_profile_with_photos,
        demo_integration_with_player_cards,
        demo_domain_restrictions
    ]
    
    for i, demo in enumerate(demos, 1):
        try:
            await demo()
            if i < len(demos):
                print("\n" + "─" * 60 + "\n")
        except Exception as e:
            print(f"❌ Demo {i} failed: {e}")
            print()
    
    print("🎉 Demo Complete!")
    print()
    print("🔗 Integration Points:")
    print("   • Player Cards API: Enhanced intelligence data")
    print("   • Betting Agents: Real-time market intelligence")
    print("   • Match Analysis: Pre/in/post match insights")
    print("   • Strategy Systems: Expert analysis and photos")

if __name__ == "__main__":
    asyncio.run(main())
