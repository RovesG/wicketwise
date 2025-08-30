# Purpose: Demo Revolutionary Betting Intelligence Integration
# Author: WicketWise Team, Last Modified: 2025-08-30

"""
Demo Revolutionary Betting Intelligence

This script demonstrates the revolutionary betting intelligence capabilities
including market psychology, clutch performance, and partnership intelligence.
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


async def demo_value_opportunities_with_intelligence():
    """Demo value opportunities analysis with revolutionary intelligence"""
    
    logger.info("🚀 DEMO: Revolutionary Betting Intelligence")
    logger.info("=" * 80)
    
    if not BETTING_AGENT_AVAILABLE:
        logger.error("❌ BettingAgent not available")
        return
    
    try:
        # Create enhanced betting agent
        agent = BettingAgent({
            "intelligence_confidence_boost": 0.20,
            "market_psychology_weight": 0.25,
            "clutch_performance_weight": 0.20
        })
        
        # Initialize agent
        agent._initialize_agent()
        
        # Create context for value opportunities analysis
        context = AgentContext(
            request_id="demo_value_opportunities",
            user_query="Find value betting opportunities for Virat Kohli and MS Dhoni performance",
            timestamp=datetime.now(),
            match_context={
                "match_id": "IND_vs_AUS_FINAL_2025",
                "teams": {
                    "home": {
                        "name": "India",
                        "players": ["Virat Kohli", "MS Dhoni", "Rohit Sharma", "Hardik Pandya"]
                    },
                    "away": {
                        "name": "Australia", 
                        "players": ["Steve Smith", "David Warner", "Glenn Maxwell", "Pat Cummins"]
                    }
                },
                "venue": "Melbourne Cricket Ground",
                "format": "T20",
                "competition": "World Cup Final",
                "match_situation": "high_pressure"
            }
        )
        
        logger.info("🎯 Analyzing value opportunities with revolutionary intelligence...")
        logger.info(f"📊 Match: {context.match_context['competition']} - {context.match_context['teams']['home']['name']} vs {context.match_context['teams']['away']['name']}")
        logger.info(f"🏟️ Venue: {context.match_context['venue']}")
        logger.info(f"⚡ Situation: {context.match_context['match_situation']}")
        
        # Execute value opportunities analysis
        response = await agent.execute(context)
        
        if response.success:
            result = response.result
            
            logger.info("=" * 80)
            logger.info("🧠 REVOLUTIONARY INTELLIGENCE RESULTS")
            logger.info("=" * 80)
            
            # Intelligence Summary
            intelligence_summary = result.get('intelligence_summary', {})
            if intelligence_summary:
                logger.info("📊 Intelligence Summary:")
                logger.info(f"  • Players Analyzed: {intelligence_summary.get('players_analyzed', 0)}")
                logger.info(f"  • Market Psychology Profiles: {intelligence_summary.get('market_psychology_profiles', 0)}")
                logger.info(f"  • Clutch Analysis Profiles: {intelligence_summary.get('clutch_analysis_profiles', 0)}")
                logger.info(f"  • Partnership Profiles: {intelligence_summary.get('partnership_profiles', 0)}")
                logger.info(f"  • Web Intelligence Items: {intelligence_summary.get('web_intelligence_items', 0)}")
                logger.info(f"  • Intelligence Confidence: {intelligence_summary.get('intelligence_confidence', 0):.2f}")
                logger.info("")
            
            # Advanced Opportunities
            advanced_opportunities = result.get('advanced_opportunities', [])
            if advanced_opportunities:
                logger.info(f"🌟 Advanced Intelligence Opportunities ({len(advanced_opportunities)}):")
                
                for i, opp in enumerate(advanced_opportunities, 1):
                    logger.info(f"  {i}. {opp.get('type', 'Unknown').replace('_', ' ').title()}")
                    logger.info(f"     🎯 Strategy: {opp.get('strategy', 'N/A')}")
                    logger.info(f"     💰 Expected Edge: {opp.get('expected_edge', 'N/A')}")
                    logger.info(f"     ⚠️ Risk Level: {opp.get('risk_level', 'N/A')}")
                    logger.info(f"     🎲 Confidence: {opp.get('confidence', 0):.2f}")
                    logger.info(f"     🧠 Intelligence Source: {opp.get('intelligence_source', 'N/A')}")
                    
                    # Show details if available
                    details = opp.get('details', {})
                    if details:
                        logger.info(f"     📋 Details:")
                        for key, value in details.items():
                            logger.info(f"        • {key.replace('_', ' ').title()}: {value}")
                    logger.info("")
            
            # Intelligence Opportunities Breakdown
            intelligence_ops = result.get('intelligence_opportunities', {})
            if intelligence_ops:
                logger.info("🎯 Intelligence Opportunities Breakdown:")
                for intel_type, count in intelligence_ops.items():
                    if count > 0:
                        logger.info(f"  • {intel_type.replace('_', ' ').title()}: {count} opportunities")
                logger.info("")
            
            # Value Summary
            value_summary = result.get('value_summary', {})
            if value_summary:
                logger.info("💎 Enhanced Value Summary:")
                logger.info(f"  • Total Opportunities: {value_summary.get('total_opportunities', 0)}")
                logger.info(f"  • Traditional Opportunities: {value_summary.get('traditional_opportunities', 0)}")
                logger.info(f"  • Advanced Intelligence Opportunities: {value_summary.get('advanced_opportunities', 0)}")
                logger.info(f"  • Intelligence Boost: +{value_summary.get('intelligence_boost', 0):.1%}")
                logger.info("")
            
            # Recommendations
            recommendations = result.get('recommendations', {})
            if recommendations:
                logger.info("💡 Enhanced Recommendations:")
                for key, value in recommendations.items():
                    logger.info(f"  • {key.replace('_', ' ').title()}: {value}")
                logger.info("")
            
            logger.info("=" * 80)
            logger.info("✅ REVOLUTIONARY BETTING INTELLIGENCE DEMO COMPLETE!")
            logger.info("=" * 80)
            
            # Summary stats
            total_opportunities = len(result.get('opportunities', [])) + len(advanced_opportunities)
            logger.info(f"🎉 RESULTS SUMMARY:")
            logger.info(f"   • Total Betting Opportunities Found: {total_opportunities}")
            logger.info(f"   • Revolutionary Intelligence Sources: {len([k for k, v in intelligence_ops.items() if v > 0])}")
            logger.info(f"   • Analysis Confidence: {response.confidence:.1%}")
            logger.info(f"   • Intelligence Enhancement: +{intelligence_summary.get('intelligence_confidence', 0):.1%}")
            
        else:
            logger.error(f"❌ Analysis failed: {response.error_message}")
            
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


async def demo_specific_intelligence_types():
    """Demo specific intelligence types"""
    
    logger.info("🧠 DEMO: Specific Intelligence Types")
    logger.info("=" * 60)
    
    try:
        agent = BettingAgent()
        agent._initialize_agent()
        
        # Test data with intelligence profiles
        test_data = {
            "market_psychology": {
                "Virat Kohli": {
                    "market_mover": True,
                    "excitement_rating": 85,
                    "overreaction_frequency": "High",
                    "betting_psychology": {
                        "six_impact": "Significant odds shift",
                        "four_impact": "Moderate shift", 
                        "normalization_time": "2-3 balls"
                    }
                },
                "MS Dhoni": {
                    "market_mover": True,
                    "excitement_rating": 90,
                    "overreaction_frequency": "Moderate",
                    "betting_psychology": {
                        "six_impact": "Major odds shift",
                        "four_impact": "Significant shift",
                        "normalization_time": "3-4 balls"
                    }
                }
            },
            "clutch_analysis": {
                "Virat Kohli": {
                    "clutch_factor": 1.12,  # 12% better under pressure
                    "clutch_rating": "Elite",
                    "pressure_situations": 45
                },
                "MS Dhoni": {
                    "clutch_factor": 1.25,  # 25% better under pressure
                    "clutch_rating": "Legendary",
                    "pressure_situations": 78
                }
            },
            "partnership_intelligence": {
                "Virat Kohli": {
                    "MS Dhoni": {
                        "synergy_rating": 92,
                        "runs_together": 2847,
                        "partnerships": 67,
                        "avg_partnership": 42.5
                    }
                }
            }
        }
        
        # Test Market Psychology Analysis
        logger.info("💰 Testing Market Psychology Analysis...")
        psychology_ops = await agent._market_psychology_value_analysis(test_data)
        logger.info(f"   Found {len(psychology_ops)} market psychology opportunities")
        for opp in psychology_ops:
            logger.info(f"   • {opp['type']}: {opp['strategy']} (Edge: {opp['expected_edge']})")
        
        # Test Clutch Performance Analysis
        logger.info("💪 Testing Clutch Performance Analysis...")
        clutch_ops = await agent._clutch_performance_value_analysis(test_data)
        logger.info(f"   Found {len(clutch_ops)} clutch performance opportunities")
        for opp in clutch_ops:
            logger.info(f"   • {opp['type']}: {opp['strategy']} (Edge: {opp['expected_edge']})")
        
        # Test Partnership Intelligence Analysis
        logger.info("🤝 Testing Partnership Intelligence Analysis...")
        partnership_ops = await agent._partnership_value_analysis(test_data)
        logger.info(f"   Found {len(partnership_ops)} partnership opportunities")
        for opp in partnership_ops:
            logger.info(f"   • {opp['type']}: {opp['strategy']} (Edge: {opp['expected_edge']})")
        
        logger.info("✅ Specific intelligence types demo completed!")
        
    except Exception as e:
        logger.error(f"❌ Specific intelligence demo failed: {e}")


async def main():
    """Main demo function"""
    
    logger.info("🏏 WicketWise Revolutionary Betting Intelligence Demo")
    logger.info("🚀 The Future of Cricket Betting AI")
    logger.info("=" * 80)
    
    # Demo 1: Full Value Opportunities Analysis
    await demo_value_opportunities_with_intelligence()
    logger.info("")
    
    # Demo 2: Specific Intelligence Types
    await demo_specific_intelligence_types()
    
    logger.info("🎉 Revolutionary Betting Intelligence Demo Complete!")
    logger.info("🏆 WicketWise: The Most Advanced Cricket Betting AI Ever Built!")


if __name__ == "__main__":
    asyncio.run(main())
