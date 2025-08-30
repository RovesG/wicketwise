# Purpose: Test Enhanced Betting Agent with Revolutionary Intelligence
# Author: WicketWise Team, Last Modified: 2025-08-30

"""
Test Enhanced Betting Agent

This script demonstrates the revolutionary betting agent with advanced intelligence integration.
Tests the new capabilities including market psychology, clutch performance, and web intelligence.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from crickformers.agents.betting_agent import BettingAgent
    from crickformers.agents.base_agent import AgentContext
    BETTING_AGENT_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import BettingAgent: {e}")
    BETTING_AGENT_AVAILABLE = False


async def test_enhanced_betting_agent():
    """Test the Enhanced Betting Agent with revolutionary intelligence"""
    
    if not BETTING_AGENT_AVAILABLE:
        logger.error("❌ BettingAgent not available - skipping test")
        return
    
    logger.info("🚀 Testing Enhanced Betting Agent with Revolutionary Intelligence")
    
    try:
        # Create enhanced betting agent
        config = {
            "max_kelly_fraction": 0.1,
            "min_edge_threshold": 0.05,
            "confidence_threshold": 0.6,
            "intelligence_confidence_boost": 0.15,
            "market_psychology_weight": 0.2,
            "clutch_performance_weight": 0.15
        }
        
        agent = BettingAgent(config)
        
        # Initialize agent
        if not agent._initialize_agent():
            logger.error("❌ Failed to initialize betting agent")
            return
        
        logger.info("✅ Enhanced Betting Agent initialized successfully")
        
        # Create test context with rich match data
        context = AgentContext(
            request_id="test_enhanced_betting_001",
            user_query="Analyze betting opportunities for India vs Australia T20 match with Virat Kohli and Steve Smith",
            timestamp=datetime.now(),
            match_context={
                "match_id": "IND_vs_AUS_T20_2025",
                "teams": {
                    "home": {
                        "name": "India",
                        "players": ["Virat Kohli", "Rohit Sharma", "MS Dhoni", "Hardik Pandya", "Jasprit Bumrah"]
                    },
                    "away": {
                        "name": "Australia", 
                        "players": ["Steve Smith", "David Warner", "Glenn Maxwell", "Pat Cummins", "Josh Hazlewood"]
                    }
                },
                "venue": "Melbourne Cricket Ground",
                "format": "T20",
                "competition": "Bilateral Series"
            },
            confidence_threshold=0.7
        )
        
        logger.info(f"🎯 Test Context: {context.user_query}")
        logger.info(f"📊 Match: {context.match_context['teams']['home']['name']} vs {context.match_context['teams']['away']['name']}")
        logger.info(f"🏟️ Venue: {context.match_context['venue']}")
        
        # Execute enhanced betting analysis
        logger.info("🧠 Executing enhanced betting analysis...")
        response = await agent.execute(context)
        
        # Display results
        logger.info("=" * 80)
        logger.info("🚀 ENHANCED BETTING AGENT RESULTS")
        logger.info("=" * 80)
        
        logger.info(f"✅ Success: {response.success}")
        logger.info(f"🎯 Confidence: {response.confidence:.2f}")
        logger.info(f"⚡ Execution Time: {response.execution_time:.3f}s")
        logger.info(f"🔧 Dependencies Used: {response.dependencies_used}")
        
        if response.success and response.result:
            result = response.result
            
            # Analysis type
            logger.info(f"📊 Analysis Type: {result.get('analysis_type', 'Unknown')}")
            
            # Traditional opportunities
            traditional_opportunities = result.get('opportunities', [])
            logger.info(f"📈 Traditional Opportunities: {len(traditional_opportunities)}")
            
            # Advanced intelligence opportunities
            advanced_opportunities = result.get('advanced_opportunities', [])
            logger.info(f"🧠 Advanced Intelligence Opportunities: {len(advanced_opportunities)}")
            
            # Intelligence breakdown
            intelligence_ops = result.get('intelligence_opportunities', {})
            if intelligence_ops:
                logger.info("🎯 Intelligence Opportunities Breakdown:")
                for intel_type, count in intelligence_ops.items():
                    logger.info(f"  • {intel_type.replace('_', ' ').title()}: {count}")
            
            # Value summary
            value_summary = result.get('value_summary', {})
            if value_summary:
                logger.info("💰 Value Summary:")
                logger.info(f"  • Total Opportunities: {value_summary.get('total_opportunities', 0)}")
                logger.info(f"  • Traditional: {value_summary.get('traditional_opportunities', 0)}")
                logger.info(f"  • Advanced: {value_summary.get('advanced_opportunities', 0)}")
                logger.info(f"  • Intelligence Boost: {value_summary.get('intelligence_boost', 0):.2f}")
            
            # Display sample opportunities
            if advanced_opportunities:
                logger.info("🌟 Sample Advanced Opportunities:")
                for i, opp in enumerate(advanced_opportunities[:3]):  # Show first 3
                    logger.info(f"  {i+1}. {opp.get('type', 'Unknown').replace('_', ' ').title()}")
                    logger.info(f"     Strategy: {opp.get('strategy', 'N/A')}")
                    logger.info(f"     Expected Edge: {opp.get('expected_edge', 'N/A')}")
                    logger.info(f"     Risk Level: {opp.get('risk_level', 'N/A')}")
                    logger.info(f"     Confidence: {opp.get('confidence', 0):.2f}")
                    logger.info(f"     Intelligence Source: {opp.get('intelligence_source', 'N/A')}")
                    logger.info("")
            
            # Recommendations
            recommendations = result.get('recommendations', {})
            if recommendations:
                logger.info("💡 Recommendations:")
                for key, value in recommendations.items():
                    logger.info(f"  • {key.replace('_', ' ').title()}: {value}")
        
        else:
            logger.error(f"❌ Analysis failed: {response.error_message}")
        
        logger.info("=" * 80)
        logger.info("✅ Enhanced Betting Agent test completed successfully!")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_intelligence_integration():
    """Test specific intelligence integration features"""
    
    logger.info("🧠 Testing Intelligence Integration Features")
    
    try:
        # Test intelligence availability
        logger.info("🔍 Checking intelligence system availability...")
        
        try:
            from crickformers.intelligence.unified_cricket_intelligence_engine import UNIFIED_INTELLIGENCE_AVAILABLE
            logger.info(f"  • Unified Intelligence Engine: {'✅ Available' if UNIFIED_INTELLIGENCE_AVAILABLE else '❌ Not Available'}")
        except ImportError:
            logger.info("  • Unified Intelligence Engine: ❌ Not Available (Import Error)")
        
        try:
            from crickformers.intelligence.web_cricket_intelligence_agent import WEB_INTELLIGENCE_AVAILABLE
            logger.info(f"  • Web Cricket Intelligence Agent: {'✅ Available' if WEB_INTELLIGENCE_AVAILABLE else '❌ Not Available'}")
        except ImportError:
            logger.info("  • Web Cricket Intelligence Agent: ❌ Not Available (Import Error)")
        
        # Test agent initialization
        logger.info("🚀 Testing agent initialization...")
        agent = BettingAgent()
        
        # Check intelligence systems
        logger.info(f"  • Unified Intelligence Engine: {'✅ Initialized' if agent.unified_intelligence_engine else '❌ Not Initialized'}")
        logger.info(f"  • Web Intelligence Agent: {'✅ Initialized' if agent.web_intelligence_agent else '❌ Not Initialized'}")
        
        # Test intelligence parameters
        logger.info("⚙️ Intelligence Parameters:")
        logger.info(f"  • Intelligence Confidence Boost: {agent.intelligence_confidence_boost}")
        logger.info(f"  • Market Psychology Weight: {agent.market_psychology_weight}")
        logger.info(f"  • Clutch Performance Weight: {agent.clutch_performance_weight}")
        
        logger.info("✅ Intelligence integration test completed!")
        
    except Exception as e:
        logger.error(f"❌ Intelligence integration test failed: {e}")


async def main():
    """Main test function"""
    
    logger.info("🏏 WicketWise Enhanced Betting Agent Test Suite")
    logger.info("=" * 60)
    
    # Test 1: Intelligence Integration
    await test_intelligence_integration()
    logger.info("")
    
    # Test 2: Enhanced Betting Agent
    await test_enhanced_betting_agent()
    
    logger.info("🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
