# Purpose: Enhanced Dashboard API - Backend integration for Cricket Intelligence Engine
# Author: WicketWise Team, Last Modified: 2025-01-19

"""
Flask API endpoints for the enhanced dashboard that provides Cricket Intelligence Engine
functionality including player search, similarity analysis, and persona-specific cards.
"""

import sys
import os
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_enhanced_dashboard import MockIntelligenceEngine
from enhanced_player_cards import PlayerCardManager, EnhancedPlayerCard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global components
intelligence_engine = None
card_manager = None


def initialize_enhanced_dashboard():
    """Initialize the enhanced dashboard components"""
    global intelligence_engine, card_manager
    
    try:
        logger.info("🚀 Initializing Enhanced Dashboard API...")
        
        # Initialize Intelligence Engine
        people_csv = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/people.csv"
        intelligence_engine = MockIntelligenceEngine(people_csv_path=people_csv)
        
        # Initialize Card Manager
        card_manager = PlayerCardManager(intelligence_engine)
        
        logger.info("✅ Enhanced Dashboard API initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Enhanced Dashboard API: {e}")
        raise


@app.route('/api/enhanced/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Enhanced Dashboard API",
        "components": {
            "intelligence_engine": intelligence_engine is not None,
            "card_manager": card_manager is not None
        }
    })


@app.route('/api/enhanced/search-players', methods=['GET'])
def search_players():
    """Search for players with autocomplete"""
    try:
        query = request.args.get('query', '').strip()
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400
        
        if len(query) < 2:
            return jsonify({"results": [], "count": 0})
        
        # Search players
        results = intelligence_engine.search_players(query, limit=limit)
        
        return jsonify({
            "query": query,
            "results": results,
            "count": len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in search_players: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/enhanced/intelligence-query', methods=['POST'])
def process_intelligence_query():
    """Process natural language intelligence queries"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        context = data.get('context', {})
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Process the query (mock implementation)
        response = {
            "query": query,
            "response": f"Intelligence analysis for: '{query}'",
            "insights": [
                "Key insight based on comprehensive data analysis",
                "Situational performance patterns identified",
                "Predictive modeling suggests favorable conditions"
            ],
            "confidence": 0.85,
            "context": context,
            "analysis_type": "comprehensive_intelligence"
        }
        
        # Add specific analysis based on query content
        if "similar" in query.lower():
            # Extract player name (simplified)
            words = query.split()
            potential_player = None
            for word in words:
                if word[0].isupper() and len(word) > 2:
                    potential_player = word
                    break
            
            if potential_player:
                similar_result = intelligence_engine.find_similar_players(potential_player, top_k=5)
                response["similarity_analysis"] = similar_result
        
        elif "predict" in query.lower() or "vs" in query.lower():
            response["prediction_type"] = "matchup_analysis"
            response["prediction_confidence"] = 0.78
        
        elif "powerplay" in query.lower() or "death" in query.lower():
            response["situational_analysis"] = {
                "phase": "powerplay" if "powerplay" in query.lower() else "death_overs",
                "specialist_players": ["Player A", "Player B", "Player C"],
                "average_performance": "Above league average"
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in process_intelligence_query: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/enhanced/player-similarity', methods=['POST'])
def find_similar_players():
    """Find players similar to a given player"""
    try:
        data = request.get_json()
        player_name = data.get('player_name', '').strip()
        top_k = int(data.get('top_k', 5))
        context = data.get('context', {})
        
        if not player_name:
            return jsonify({"error": "Player name is required"}), 400
        
        # Find similar players
        similar_result = intelligence_engine.find_similar_players(player_name, top_k=top_k)
        
        if 'error' in similar_result:
            return jsonify(similar_result), 404
        
        return jsonify(similar_result)
        
    except Exception as e:
        logger.error(f"Error in find_similar_players: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/enhanced/player-intelligence', methods=['POST'])
def analyze_player_intelligence():
    """Analyze player intelligence for specific persona"""
    try:
        data = request.get_json()
        player_name = data.get('player_name', '').strip()
        persona = data.get('persona', 'betting')
        context = data.get('context', {})
        
        if not player_name:
            return jsonify({"error": "Player name is required"}), 400
        
        # Analyze player intelligence
        intelligence_result = intelligence_engine.analyze_player_intelligence(
            player_name, persona, context
        )
        
        if 'error' in intelligence_result:
            return jsonify(intelligence_result), 404
        
        return jsonify(intelligence_result)
        
    except Exception as e:
        logger.error(f"Error in analyze_player_intelligence: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/enhanced/player-cards', methods=['GET'])
def get_player_cards():
    """Get enhanced player cards for dashboard"""
    try:
        persona = request.args.get('persona', 'betting')
        count = int(request.args.get('count', 6))
        context_str = request.args.get('context', '{}')
        
        try:
            context = json.loads(context_str)
        except:
            context = {}
        
        # Generate player cards
        cards = card_manager.generate_dashboard_cards(
            persona=persona,
            count=count, 
            context=context
        )
        
        return jsonify({
            "persona": persona,
            "count": len(cards),
            "cards": cards,
            "context": context
        })
        
    except Exception as e:
        logger.error(f"Error in get_player_cards: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/enhanced/player-card', methods=['POST'])
def get_single_player_card():
    """Get a single enhanced player card"""
    try:
        data = request.get_json()
        player_name = data.get('player_name', '').strip()
        persona = data.get('persona', 'betting')
        context = data.get('context', {})
        include_live_data = data.get('include_live_data', True)
        
        if not player_name:
            return jsonify({"error": "Player name is required"}), 400
        
        # Generate single player card
        card_generator = EnhancedPlayerCard(intelligence_engine)
        card = card_generator.generate_card(
            player_name=player_name,
            persona=persona,
            context=context,
            include_live_data=include_live_data
        )
        
        return jsonify(card)
        
    except Exception as e:
        logger.error(f"Error in get_single_player_card: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/enhanced/update-live-data', methods=['POST'])
def update_live_data():
    """Update live data for a player card"""
    try:
        data = request.get_json()
        player_name = data.get('player_name', '').strip()
        persona = data.get('persona', 'betting')
        
        if not player_name:
            return jsonify({"error": "Player name is required"}), 400
        
        # Update live data
        updated_card = card_manager.update_live_data(player_name, persona)
        
        return jsonify({
            "player_name": player_name,
            "persona": persona,
            "updated_card": updated_card,
            "timestamp": updated_card.get('updated_at')
        })
        
    except Exception as e:
        logger.error(f"Error in update_live_data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/enhanced/matchup-prediction', methods=['POST'])
def predict_matchup():
    """Predict matchup favorability"""
    try:
        data = request.get_json()
        batter = data.get('batter', '').strip()
        bowler_or_type = data.get('bowler_or_type', '').strip()
        match_context = data.get('match_context', {})
        
        if not batter or not bowler_or_type:
            return jsonify({"error": "Both batter and bowler/type are required"}), 400
        
        # Mock matchup prediction
        prediction = {
            "batter": batter,
            "opponent": bowler_or_type,
            "match_context": match_context,
            "predictions": {
                "expected_average": 35.2 if "pace" in bowler_or_type.lower() else 42.8,
                "expected_strike_rate": 125.5 if "pace" in bowler_or_type.lower() else 138.2,
                "matchup_advantage": "batter" if "spin" in bowler_or_type.lower() else "neutral",
                "confidence": 0.73,
                "key_factors": [
                    "Historical performance against this type",
                    "Recent form and conditions",
                    "Venue-specific advantages"
                ]
            },
            "analysis_type": "comprehensive_matchup"
        }
        
        return jsonify(prediction)
        
    except Exception as e:
        logger.error(f"Error in predict_matchup: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/enhanced/dashboard-stats', methods=['GET'])
def get_dashboard_stats():
    """Get overall dashboard statistics"""
    try:
        stats = {
            "total_players": 17016 if intelligence_engine and intelligence_engine.player_index is not None else 11995,
            "knowledge_graph_nodes": 11997 if intelligence_engine and intelligence_engine.gnn_analytics else 0,
            "available_personas": ["betting", "commentary", "coaching", "fantasy"],
            "features_available": {
                "player_search": True,
                "similarity_analysis": True,
                "intelligence_queries": True,
                "persona_cards": True,
                "live_data": True,
                "matchup_predictions": True
            },
            "system_status": {
                "intelligence_engine": "active" if intelligence_engine else "inactive",
                "card_manager": "active" if card_manager else "inactive",
                "knowledge_graph": "loaded" if intelligence_engine and intelligence_engine.gnn_analytics else "not_loaded"
            }
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error in get_dashboard_stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


def main():
    """Run the enhanced dashboard API server"""
    logger.info("🚀 Starting Enhanced Dashboard API Server...")
    
    try:
        # Initialize components
        initialize_enhanced_dashboard()
        
        # Print available endpoints
        logger.info("\n📡 Available API Endpoints:")
        logger.info("  GET  /api/enhanced/health - Health check")
        logger.info("  GET  /api/enhanced/search-players - Search players")
        logger.info("  POST /api/enhanced/intelligence-query - Process intelligence queries")
        logger.info("  POST /api/enhanced/player-similarity - Find similar players")
        logger.info("  POST /api/enhanced/player-intelligence - Analyze player intelligence")
        logger.info("  GET  /api/enhanced/player-cards - Get dashboard player cards")
        logger.info("  POST /api/enhanced/player-card - Get single player card")
        logger.info("  POST /api/enhanced/update-live-data - Update live data")
        logger.info("  POST /api/enhanced/matchup-prediction - Predict matchups")
        logger.info("  GET  /api/enhanced/dashboard-stats - Get dashboard stats")
        
        # Start server
        logger.info(f"\n🌐 Server starting on http://127.0.0.1:5002")
        logger.info("🎯 Ready to serve Enhanced Cricket Intelligence!")
        
        app.run(host='127.0.0.1', port=5002, debug=True)
        
    except Exception as e:
        logger.error(f"❌ Failed to start Enhanced Dashboard API: {e}")
        raise


if __name__ == "__main__":
    main()
