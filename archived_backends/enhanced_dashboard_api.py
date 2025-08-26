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

# Add SIM module to path
sim_path = project_root / 'sim'
if sim_path.exists():
    sys.path.insert(0, str(sim_path))
    try:
        from data_integration import HoldoutDataManager, integrate_holdout_data_with_sim
        from config import create_holdout_replay_config
        from orchestrator import SimOrchestrator
        SIM_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"SIM module not available: {e}")
        SIM_AVAILABLE = False
else:
    SIM_AVAILABLE = False

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


# Simulation API Endpoints
@app.route('/api/simulation/holdout-matches', methods=['GET'])
def get_holdout_matches():
    """Get available holdout matches for simulation"""
    try:
        if not SIM_AVAILABLE:
            return jsonify({
                "status": "error",
                "message": "SIM module not available",
                "match_count": 0
            })
        
        integration_result = integrate_holdout_data_with_sim()
        
        return jsonify({
            "status": integration_result["status"],
            "message": integration_result["message"],
            "match_count": integration_result.get("match_count", 0),
            "matches": integration_result.get("matches", [])[:10],  # Return first 10 as sample
            "integrity_report": integration_result.get("integrity_report", {})
        })
        
    except Exception as e:
        logger.error(f"Error getting holdout matches: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to get holdout matches: {str(e)}",
            "match_count": 0
        })


@app.route('/api/simulation/run', methods=['POST'])
def run_simulation():
    """Run a strategy simulation"""
    try:
        if not SIM_AVAILABLE:
            return jsonify({
                "status": "error",
                "message": "SIM module not available"
            })
        
        data = request.get_json()
        strategy = data.get('strategy', 'edge_kelly_v3')
        match_selection = data.get('match_selection', 'auto')
        use_holdout_data = data.get('use_holdout_data', True)
        
        logger.info(f"🎯 Running simulation: strategy={strategy}, matches={match_selection}")
        
        # Create configuration
        if use_holdout_data:
            config = create_holdout_replay_config(strategy)
        else:
            from config import create_replay_config
            config = create_replay_config(["mock_match_1"], strategy)
        
        # Adjust match selection
        if match_selection == "auto":
            # Already limited to 10 matches in create_holdout_replay_config
            pass
        elif match_selection == "all":
            # Use all available matches (but limit to 20 for performance)
            manager = HoldoutDataManager()
            all_matches = manager.get_holdout_matches()
            config.match_ids = all_matches[:20]
        
        # Run simulation
        orchestrator = SimOrchestrator()
        
        if not orchestrator.initialize(config):
            return jsonify({
                "status": "error",
                "message": "Failed to initialize simulation"
            })
        
        result = orchestrator.run()
        
        if result:
            return jsonify({
                "status": "success",
                "message": "Simulation completed successfully",
                "run_id": result.run_id,
                "kpis": result.kpis.to_dict(),
                "violations": result.violations,
                "runtime_seconds": result.runtime_seconds,
                "balls_processed": result.balls_processed,
                "matches_processed": result.matches_processed
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Simulation failed to complete"
            })
        
    except Exception as e:
        logger.error(f"Error running simulation: {e}")
        return jsonify({
            "status": "error",
            "message": f"Simulation failed: {str(e)}"
        })


# Training Pipeline Endpoints for Admin Panel
@app.route('/api/training-pipeline/stats', methods=['GET'])
def get_training_pipeline_stats():
    """Get training pipeline statistics for admin panel"""
    try:
        # Mock data for now - in a real implementation, this would query actual data sources
        stats = {
            "json_dataset": {
                "rows": 916090,  # From our decimal data
                "size_mb": 125.4,
                "last_updated": "2024-01-15T10:30:00Z"
            },
            "decimal_dataset": {
                "rows": 916090,
                "size_mb": 125.4,
                "last_updated": "2024-01-15T10:30:00Z"
            },
            "entity_harmonizer": {
                "players": {
                    "total": 11995,  # From our KG data
                    "with_batting_stats": 11407,
                    "with_bowling_stats": 8652
                },
                "venues": {
                    "total": 450,
                    "with_coordinates": 380
                }
            },
            "knowledge_graph": {
                "nodes": 32505,
                "edges": 138553,
                "last_built": "2024-01-15T12:00:00Z"
            }
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting training pipeline stats: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to get stats: {str(e)}"
        })


@app.route('/api/training-pipeline/enrichment-statistics', methods=['GET'])
@app.route('/api/enrichment-statistics', methods=['GET'])  # Shorter alias for admin panel
def get_enrichment_statistics():
    """Get enrichment statistics for admin panel"""
    try:
        # Mock enrichment statistics in the format expected by admin panel
        stats = {
            "dataset_stats": {
                "total_enriched": 3983,
                "total_available": 3983,
                "success_rate": 100.0,
                "last_enrichment": "2024-01-15T14:30:00Z"
            },
            "enrichment_types": {
                "player_stats": 3983,
                "venue_data": 3983,
                "weather_data": 2800,
                "pitch_conditions": 3200
            },
            "processing_time": {
                "average_per_match": 2.3,
                "total_hours": 2.5
            }
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting enrichment statistics: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to get enrichment stats: {str(e)}"
        })


@app.route('/api/cricsheet/check-updates', methods=['GET'])
def check_cricsheet_updates():
    """Check for Cricsheet updates"""
    try:
        # Mock update check - in format expected by admin panel
        update_info = {
            "update_available": True,  # Changed from "updates_available"
            "new_files_count": 15,
            "last_check": "2024-01-15T16:00:00Z",
            "latest_match_date": "2024-01-14",
            "current_data_date": "2024-01-10",
            "remote_info": {
                "size_mb": 45.2,
                "last_modified": "2024-01-14T18:30:00Z"
            },
            "new_files": [
                {"name": "t20s_male_2024_01_11.json", "size": "2.3 MB", "date": "2024-01-11"},
                {"name": "t20s_male_2024_01_12.json", "size": "1.8 MB", "date": "2024-01-12"},
                {"name": "t20s_male_2024_01_13.json", "size": "3.1 MB", "date": "2024-01-13"},
                {"name": "t20s_male_2024_01_14.json", "size": "2.7 MB", "date": "2024-01-14"}
            ]
        }
        
        return jsonify(update_info)
        
    except Exception as e:
        logger.error(f"Error checking Cricsheet updates: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to check updates: {str(e)}"
        })


@app.route('/api/cricsheet/update', methods=['POST'])
def update_cricsheet_data():
    """Update Cricsheet data"""
    try:
        # Mock update process - in real implementation, this would download and process new files
        import time
        
        # Simulate processing time
        time.sleep(2)
        
        # Mock successful update
        update_result = {
            "files_downloaded": 4,
            "files_processed": 4,
            "new_matches": 28,
            "processing_time": 2.1,
            "status": "completed",
            "timestamp": "2024-01-15T16:05:00Z"
        }
        
        return jsonify({
            "status": "started",  # Changed from "success" to match admin panel expectation
            "message": "Cricsheet data update started successfully",
            "data": update_result
        })
        
    except Exception as e:
        logger.error(f"Error updating Cricsheet data: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to update data: {str(e)}"
        })


# Configuration Settings Endpoints for Admin Panel
@app.route('/api/aligner-settings', methods=['GET'])
def get_aligner_settings():
    """Get aligner configuration settings"""
    try:
        settings = {
            "entity_matching_threshold": 0.85,
            "fuzzy_match_cutoff": 0.8,
            "auto_merge_duplicates": True,
            "manual_review_required": False,
            "batch_size": 1000,
            "parallel_processing": True,
            "cache_results": True
        }
        return jsonify(settings)
    except Exception as e:
        logger.error(f"Error getting aligner settings: {e}")
        return jsonify({"error": str(e)})


@app.route('/api/alljson-settings', methods=['GET'])
def get_alljson_settings():
    """Get all JSON processing settings"""
    try:
        settings = {
            "data_source": "cricsheet",
            "file_format": "json",
            "compression": "gzip",
            "validation_enabled": True,
            "schema_version": "v2.1",
            "batch_processing": True,
            "max_file_size_mb": 500,
            "timeout_seconds": 300
        }
        return jsonify(settings)
    except Exception as e:
        logger.error(f"Error getting alljson settings: {e}")
        return jsonify({"error": str(e)})


@app.route('/api/training-settings', methods=['GET'])
def get_training_settings():
    """Get training pipeline settings"""
    try:
        settings = {
            "model_type": "crickformer",
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 100,
            "validation_split": 0.2,
            "early_stopping": True,
            "patience": 10,
            "use_gpu": True,
            "mixed_precision": True,
            "checkpoint_frequency": 5
        }
        return jsonify(settings)
    except Exception as e:
        logger.error(f"Error getting training settings: {e}")
        return jsonify({"error": str(e)})


@app.route('/api/operation-status/<operation_id>', methods=['GET'])
def get_operation_status(operation_id):
    """Get status of a running operation"""
    try:
        # Mock operation status - in a real system this would track actual operations
        # For now, simulate operations completing after a few polls
        import time
        current_time = time.time()
        
        # Simulate different operation durations
        operation_durations = {
            'cricsheet_update': 30,  # 30 seconds
            'kg_build': 45,         # 45 seconds  
            'gnn_training': 60,     # 60 seconds
            'match_enrichment': 25, # 25 seconds
            'model_training': 90    # 90 seconds
        }
        
        # For demo purposes, always return completed after first poll
        # In real system, you'd track actual operation state
        status_response = {
            "status": "completed",
            "operation_id": operation_id,
            "progress": 100,
            "message": f"{operation_id.replace('_', ' ').title()} completed successfully",
            "started_at": "2024-01-15T16:00:00Z",
            "completed_at": "2024-01-15T16:02:30Z",
            "duration_seconds": operation_durations.get(operation_id, 30)
        }
        
        return jsonify(status_response)
        
    except Exception as e:
        logger.error(f"Error getting operation status for {operation_id}: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to get operation status: {str(e)}"
        })


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
        
        if SIM_AVAILABLE:
            logger.info("  GET  /api/simulation/holdout-matches - Get holdout matches")
            logger.info("  POST /api/simulation/run - Run strategy simulation")
        
        logger.info("  GET  /api/training-pipeline/stats - Get training pipeline statistics")
        logger.info("  GET  /api/training-pipeline/enrichment-statistics - Get enrichment statistics")
        logger.info("  GET  /api/cricsheet/check-updates - Check for Cricsheet updates")
        logger.info("  POST /api/cricsheet/update - Update Cricsheet data")
        logger.info("  GET  /api/aligner-settings - Get aligner configuration")
        logger.info("  GET  /api/alljson-settings - Get JSON processing settings")
        logger.info("  GET  /api/training-settings - Get training pipeline settings")
        logger.info("  GET  /api/operation-status/<operation_id> - Get operation status")
        
        # Start server
        logger.info(f"\n🌐 Server starting on http://127.0.0.1:5001")
        logger.info("🎯 Ready to serve Enhanced Cricket Intelligence!")
        
        app.run(host='127.0.0.1', port=5001, debug=True)
        
    except Exception as e:
        logger.error(f"❌ Failed to start Enhanced Dashboard API: {e}")
        raise


if __name__ == "__main__":
    main()
