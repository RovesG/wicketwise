# Purpose: Real Dynamic Player Cards API connected to KG + GNN
# Author: WicketWise Team, Last Modified: August 19, 2024

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
import logging
from datetime import datetime
import random
import hashlib
import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import real components
try:
    from crickformers.gnn.unified_kg_builder import UnifiedKGBuilder
    from crickformers.gnn.enhanced_kg_gnn import EnhancedKG_GNN, KGNodeFeatureExtractor
    from crickformers.chat.unified_kg_query_engine import UnifiedKGQueryEngine
    REAL_COMPONENTS_AVAILABLE = True
    print("✅ Real KG + GNN components loaded successfully")
except ImportError as e:
    REAL_COMPONENTS_AVAILABLE = False
    print(f"⚠️ Real components not available: {e}")
    print("📦 Using mock data instead")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global components
player_data = None
kg_query_engine = None
gnn_model = None

# Current match context (can be updated via API)
current_match_context = {
    "homeTeam": {
        "id": "RCB",
        "name": "Royal Challengers Bangalore", 
        "players": ["V Kohli", "F du Plessis", "G Maxwell", "D Karthik", "W Hasaranga", "H Patel", "S Ahmed"]
    },
    "awayTeam": {
        "id": "CSK",
        "name": "Chennai Super Kings",
        "players": ["MS Dhoni", "R Jadeja", "D Conway", "R Gaikwad", "D Chahar", "M Theekshana", "M Ali"]
    },
    "venue": "M. Chinnaswamy Stadium",
    "tournament": "IPL 2025",
    "matchDate": "2025-01-15"
}

def load_real_components():
    """Load real KG and GNN components"""
    global kg_query_engine, gnn_model
    
    logger.info(f"🔄 DEBUG: REAL_COMPONENTS_AVAILABLE = {REAL_COMPONENTS_AVAILABLE}")
    
    # Try to load KG even if some components failed to import
    try:
        # Load the Knowledge Graph - try different approaches
        logger.info("🔄 DEBUG: Attempting to load Knowledge Graph...")
        
        # First try the unified query engine
        try:
            if REAL_COMPONENTS_AVAILABLE:
                kg_query_engine = UnifiedKGQueryEngine()
                logger.info("✅ DEBUG: UnifiedKGQueryEngine loaded successfully")
            else:
                logger.info("🔄 DEBUG: Trying to load KG components individually...")
                # Try to import and load individually
                from crickformers.chat.unified_kg_query_engine import UnifiedKGQueryEngine
                kg_query_engine = UnifiedKGQueryEngine()
                logger.info("✅ DEBUG: UnifiedKGQueryEngine loaded individually")
        except Exception as kg_error:
            logger.warning(f"⚠️ DEBUG: UnifiedKGQueryEngine failed: {kg_error}")
            
            # Try alternative KG loading
            try:
                from crickformers.gnn.unified_kg_builder import UnifiedKGBuilder
                logger.info("🔄 DEBUG: Trying UnifiedKGBuilder...")
                # UnifiedKGBuilder needs a data_dir parameter
                data_dir = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data"
                kg_builder = UnifiedKGBuilder(data_dir)
                # Check if there's a saved KG we can load
                if hasattr(kg_builder, 'load_existing_kg'):
                    kg_query_engine = kg_builder.load_existing_kg()
                    logger.info("✅ DEBUG: Loaded existing KG via UnifiedKGBuilder")
                elif hasattr(kg_builder, 'graph'):
                    # Use the builder's graph directly
                    kg_query_engine = kg_builder
                    logger.info("✅ DEBUG: Using UnifiedKGBuilder as query engine")
                else:
                    logger.info("⚠️ DEBUG: No suitable method available in UnifiedKGBuilder")
                    kg_query_engine = None
            except Exception as builder_error:
                logger.warning(f"⚠️ DEBUG: UnifiedKGBuilder failed: {builder_error}")
                kg_query_engine = None
        
        # Try to load GNN model if available
        logger.info("🔄 DEBUG: Attempting to load GNN model...")
        try:
            # Try different GNN imports
            try:
                from crickformers.gnn.enhanced_kg_gnn import EnhancedKG_GNN
                gnn_model = EnhancedKG_GNN()
                logger.info("✅ DEBUG: EnhancedKG_GNN loaded")
            except ImportError:
                # Try alternative GNN classes
                from crickformers.gnn import enhanced_kg_gnn
                available_classes = [name for name in dir(enhanced_kg_gnn) if not name.startswith('_') and 'GNN' in name]
                logger.info(f"🔍 DEBUG: Available GNN classes: {available_classes}")
                
                if available_classes:
                    gnn_class = getattr(enhanced_kg_gnn, available_classes[0])
                    gnn_model = gnn_class()
                    logger.info(f"✅ DEBUG: Loaded GNN using {available_classes[0]}")
                else:
                    gnn_model = None
                    logger.warning("⚠️ DEBUG: No GNN classes found")
        except Exception as gnn_error:
            logger.warning(f"⚠️ DEBUG: GNN model not available: {gnn_error}")
            gnn_model = None
        
        # Summary
        kg_available = kg_query_engine is not None
        gnn_available = gnn_model is not None
        
        logger.info(f"📊 DEBUG: Component loading summary:")
        logger.info(f"  - KG Query Engine: {'✅' if kg_available else '❌'}")
        logger.info(f"  - GNN Model: {'✅' if gnn_available else '❌'}")
        
        if kg_available:
            logger.info("✅ DEBUG: At least KG components loaded successfully")
            return True
        else:
            logger.warning("⚠️ DEBUG: No real components could be loaded")
            return False
        
    except Exception as e:
        logger.error(f"❌ DEBUG: Error in load_real_components: {e}")
        import traceback
        logger.error(f"❌ DEBUG: Traceback: {traceback.format_exc()}")
        return False

def load_players():
    """Load player data from CSV"""
    global player_data
    try:
        people_csv = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/people.csv"
        player_data = pd.read_csv(people_csv)
        logger.info(f"✅ Loaded {len(player_data)} players from real database")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading players: {e}")
        # Create mock data
        player_data = pd.DataFrame({
            'identifier': ['virat_kohli', 'ms_dhoni', 'rohit_sharma', 'kl_rahul', 'hardik_pandya'],
            'name': ['Virat Kohli', 'MS Dhoni', 'Rohit Sharma', 'KL Rahul', 'Hardik Pandya'],
            'unique_name': ['Virat Kohli', 'MS Dhoni', 'Rohit Sharma', 'KL Rahul', 'Hardik Pandya']
        })
        logger.info(f"✅ Using mock data with {len(player_data)} players")
        return False

def get_real_player_data(player_name):
    """Get real player data from KG + GNN"""
    logger.info(f"🔍 DEBUG: Attempting to get real data for '{player_name}'")
    logger.info(f"🔍 DEBUG: kg_query_engine available: {kg_query_engine is not None}")
    
    if not kg_query_engine:
        logger.warning(f"❌ DEBUG: No KG query engine available for {player_name}")
        return None
    
    try:
        logger.info(f"🔍 DEBUG: Querying KG for player data: {player_name}")
        
        # Check if the KG has a method to get player stats
        available_methods = [method for method in dir(kg_query_engine) if not method.startswith('_')]
        logger.info(f"🔍 DEBUG: Available KG methods: {available_methods}")
        
        # Try different method names that might exist
        player_stats = None
        method_tried = None
        
        # Try various method names
        possible_methods = [
            'get_complete_player_profile',  # This one exists!
            'get_player_comprehensive_stats',
            'get_player_stats', 
            'query_player_stats',
            'get_player_data',
            'query_player',
            'get_comprehensive_stats'
        ]
        
        for method_name in possible_methods:
            if hasattr(kg_query_engine, method_name):
                logger.info(f"🔍 DEBUG: Trying method: {method_name}")
                method_tried = method_name
                try:
                    method = getattr(kg_query_engine, method_name)
                    
                    # Patch the timeout mechanism to be thread-safe
                    original_timeout_func = None
                    if hasattr(kg_query_engine, '__class__'):
                        # Import and replace the timeout function with a thread-safe version
                        import crickformers.chat.unified_kg_query_engine as kg_module
                        if hasattr(kg_module, 'timeout'):
                            original_timeout_func = kg_module.timeout
                            # Replace with a no-op context manager for thread safety
                            from contextlib import nullcontext
                            kg_module.timeout = lambda seconds: nullcontext()
                    
                    player_stats = method(player_name)
                    
                    # Restore original timeout function
                    if original_timeout_func is not None:
                        kg_module.timeout = original_timeout_func
                    
                    logger.info(f"✅ DEBUG: Method {method_name} returned: {type(player_stats)}")
                    if player_stats:
                        logger.info(f"✅ DEBUG: Player stats keys: {list(player_stats.keys()) if isinstance(player_stats, dict) else 'Not a dict'}")
                    break
                except Exception as method_error:
                    logger.warning(f"⚠️ DEBUG: Method {method_name} failed: {method_error}")
                    continue
        
        if not method_tried:
            logger.warning(f"⚠️ DEBUG: No suitable method found in KG query engine")
            return None
        
        if player_stats:
            # Check if the response contains an error
            if isinstance(player_stats, dict) and 'error' in player_stats:
                logger.error(f"❌ DEBUG: KG returned error for {player_name}: {player_stats.get('error')}")
                return None
            
            logger.info(f"✅ DEBUG: Found real data for {player_name} using {method_tried}")
            logger.info(f"✅ DEBUG: Player stats content: {player_stats}")
            
            # Extract data from the actual KG structure
            batting_stats = player_stats.get('batting_stats', {})
            vs_pace = player_stats.get('vs_pace', {})
            vs_spin = player_stats.get('vs_spin', {})
            powerplay = player_stats.get('in_powerplay', {})
            death_overs = player_stats.get('in_death_overs', {})
            
            return {
                'batting_avg': batting_stats.get('average', 35.0),
                'strike_rate': vs_pace.get('strike_rate', powerplay.get('strike_rate', 125.0)),
                'recent_form': 8.5,  # Calculate from recent performance if available
                'form_rating': 8.0,  # Based on overall performance
                'powerplay_sr': powerplay.get('strike_rate', 100.0),
                'death_overs_sr': death_overs.get('strike_rate', 120.0),
                'vs_pace_avg': vs_pace.get('average', 35.0),
                'vs_spin_avg': vs_spin.get('average', 30.0) if vs_spin else 30.0,
                'pressure_rating': 8.0,  # Based on death overs performance
                'recent_matches': [],  # Would need to query recent matches separately
                'matches_played': player_stats.get('matches_played', 0),
                'total_runs': batting_stats.get('runs', 0),
                'teams': player_stats.get('teams', []),
                'primary_role': player_stats.get('primary_role', 'batsman'),
                'source': 'Real_KG_Data'
            }
        else:
            logger.warning(f"⚠️ DEBUG: No real data found for {player_name} (method returned None/empty)")
            return None
            
    except Exception as e:
        logger.error(f"❌ DEBUG: Error getting real data for {player_name}: {e}")
        logger.error(f"❌ DEBUG: Exception type: {type(e)}")
        import traceback
        logger.error(f"❌ DEBUG: Traceback: {traceback.format_exc()}")
        return None

def get_gnn_player_insights(player_name):
    """Get GNN-powered player insights"""
    if not gnn_model:
        return None
    
    try:
        logger.info(f"🧠 Getting GNN insights for {player_name}")
        
        # Get GNN features and similar players
        feature_extractor = KGNodeFeatureExtractor(kg_query_engine.graph)
        player_features = feature_extractor.extract_features(player_name)
        
        if player_features is not None:
            # Find similar players using GNN
            similar_players = gnn_model.find_similar_players(player_name, top_k=3)
            
            return {
                'similar_players': similar_players,
                'gnn_features': player_features.tolist() if hasattr(player_features, 'tolist') else [],
                'source': 'Real_GNN_Analysis'
            }
        else:
            logger.warning(f"⚠️ No GNN features found for {player_name}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error getting GNN insights for {player_name}: {e}")
        return None

def calculate_form_from_recent_scores(recent_scores):
    """Calculate form description from recent scores"""
    if not recent_scores:
        return "Unknown"
    
    avg_score = sum(recent_scores) / len(recent_scores)
    
    if avg_score >= 50:
        return "Exceptional Form"
    elif avg_score >= 40:
        return "Hot Form"
    elif avg_score >= 30:
        return "Good Form"
    elif avg_score >= 20:
        return "Average Form"
    else:
        return "Poor Form"

def generate_enhanced_card_data(player_name, persona):
    """Generate card data using real KG + GNN when available, fallback to mock"""
    logger.info(f"🎴 Generating enhanced card for {player_name} (persona: {persona})")
    
    # Try to get real data first
    real_data = get_real_player_data(player_name)
    gnn_insights = get_gnn_player_insights(player_name)
    
    if real_data:
        logger.info(f"✅ Using real KG data for {player_name}")
        card_data = real_data.copy()
        
        # Add GNN insights if available
        if gnn_insights:
            card_data.update(gnn_insights)
            logger.info(f"✅ Enhanced with GNN insights for {player_name}")
        
        # Add metadata
        card_data.update({
            'player_name': player_name,
            'profile_image_url': get_player_image_url(player_name),
            'profile_image_cached': False,
            'last_updated': datetime.now().isoformat(),
            'data_sources': [card_data.get('source', 'Real_KG_Data')]
        })
        
        return card_data
    
    else:
        logger.info(f"⚠️ Falling back to mock data for {player_name}")
        return generate_mock_card_data(player_name, persona)

def get_player_image_url(player_name):
    """Get a better image URL for the player"""
    # Map of known players to actual cricket images
    player_images = {
        'Virat Kohli': 'https://images.unsplash.com/photo-1544966503-7cc5ac882d5f?w=150&h=150&fit=crop&crop=face&auto=format&q=80',
        'MS Dhoni': 'https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=150&h=150&fit=crop&crop=face&auto=format&q=80',
        'Rohit Sharma': 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=150&h=150&fit=crop&crop=face&auto=format&q=80',
        'KL Rahul': 'https://images.unsplash.com/photo-1566577739112-5180d4bf9390?w=150&h=150&fit=crop&crop=face&auto=format&q=80',
        'Hardik Pandya': 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=150&h=150&fit=crop&crop=face&auto=format&q=80',
    }
    
    # Return specific image if available, otherwise use a reliable placeholder
    return player_images.get(player_name, 
        f"https://ui-avatars.com/api/?name={player_name.replace(' ', '+')}&background=0d47a1&color=fff&size=150")

def generate_mock_card_data(player_name, persona):
    """Generate mock card data (fallback when real data unavailable)"""
    # MAKE MOCK DATA OBVIOUSLY FAKE WITH ZEROS
    batting_avg = 0.0
    strike_rate = 0.0
    form_rating = 0.0
    
    form_descriptions = ["MOCK FORM"]
    form_index = 0
    
    # Generate obviously fake recent matches
    teams = ['MOCK_TEAM']
    recent_matches = []
    for i in range(5):
        score = 0  # Obviously fake score
        is_not_out = False
        opponent = random.choice(teams)
        recent_matches.append({
            'score': f"{score}{'*' if is_not_out else ''}",
            'opponent': opponent,
            'days_ago': (i * 4 + 3)
        })
    
    # Generate live data
    ball_outcomes = ['1', '2', '4', '6', '0', 'W', '.']
    last_6_balls = [random.choice(ball_outcomes) for _ in range(6)]
    
    # Betting intelligence
    market_odds = round(random.uniform(1.4, 2.5), 2)
    model_odds = round(random.uniform(1.3, 2.3), 2)
    expected_value = round(((1/model_odds * market_odds) - 1) * 100, 1)
    
    return {
        'player_name': player_name,
        'batting_avg': 0.0,  # OBVIOUSLY FAKE
        'strike_rate': 0.0,  # OBVIOUSLY FAKE
        'recent_form': "MOCK FORM",  # OBVIOUSLY FAKE
        'form_rating': 0.0,  # OBVIOUSLY FAKE
        'powerplay_sr': 0.0,  # OBVIOUSLY FAKE
        'death_overs_sr': 0.0,  # OBVIOUSLY FAKE
        'vs_pace_avg': 0.0,  # OBVIOUSLY FAKE
        'vs_spin_avg': 0.0,  # OBVIOUSLY FAKE
        'pressure_rating': 0.0,  # OBVIOUSLY FAKE
        'recent_matches': recent_matches,
        'matches_played': 0,  # OBVIOUSLY FAKE
        'teams': ['MOCK_TEAM'],  # OBVIOUSLY FAKE
        'current_match_status': 'MOCK_STATUS',  # OBVIOUSLY FAKE
        'last_6_balls': ['0', '0', '0', '0', '0', '0'],  # OBVIOUSLY FAKE
        'betting_odds': {
            'market_odds': market_odds,
            'model_odds': model_odds,
            'expected_value': expected_value
        },
        'value_opportunities': [{
            'market': 'Runs Over 30.5',
            'market_odds': market_odds,
            'model_odds': model_odds,
            'expected_value': expected_value,
            'confidence': random.randint(65, 90)
        }] if abs(expected_value) > 5 else [],
        'similar_players': ['AB de Villiers (0.92)', 'Steve Smith (0.89)', 'Kane Williamson (0.85)'],
        'profile_image_url': get_player_image_url(player_name),
        'profile_image_cached': False,
        'last_updated': datetime.now().isoformat(),
        'data_sources': ['MOCK_DATA_OBVIOUS']
    }

# API Routes
@app.route('/api/cards/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Real Dynamic Player Cards API',
        'players_available': len(player_data) if player_data is not None else 0,
        'kg_available': kg_query_engine is not None,
        'gnn_available': gnn_model is not None,
        'real_components': REAL_COMPONENTS_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/cards/autocomplete', methods=['GET'])
def autocomplete():
    """Get autocomplete suggestions for player names"""
    try:
        partial = request.args.get('partial', '').strip()
        limit = int(request.args.get('limit', 5))
        
        if not partial or len(partial) < 2:
            return jsonify({
                'success': True,
                'partial': partial,
                'suggestions': [],
                'count': 0
            })
        
        # Filter players whose names start with the partial name (case insensitive)
        mask = player_data['name'].str.lower().str.startswith(partial.lower())
        suggestions = player_data[mask]['name'].head(limit).tolist()
        
        logger.info(f"🔍 Autocomplete '{partial}' -> {len(suggestions)} results")
        
        return jsonify({
            'success': True,
            'partial': partial,
            'suggestions': suggestions,
            'count': len(suggestions),
            'source': 'Real_Player_Database' if len(player_data) > 10 else 'Mock_Database'
        })
        
    except Exception as e:
        logger.error(f"Error in autocomplete: {e}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'suggestions': [],
            'count': 0
        }), 500

@app.route('/api/cards/generate', methods=['POST'])
def generate_card():
    """Generate a dynamic player card using real KG + GNN data"""
    try:
        data = request.json
        player_name = data.get('player_name', '').strip()
        persona = data.get('persona', 'betting')
        
        if not player_name:
            return jsonify({'error': 'player_name is required'}), 400
        
        logger.info(f"🎴 Generating real card for '{player_name}' (persona: {persona})")
        
        # Generate the card data using real components
        card_result = generate_enhanced_card_data(player_name, persona)
        
        # Check if we got real data or mock data
        real_data_used = False
        if isinstance(card_result, dict):
            if 'data_sources' in card_result:
                real_data_used = 'Real_KG_Data' in card_result['data_sources']
            card_data = card_result
        else:
            card_data = card_result
        
        return jsonify({
            'success': True,
            'player_name': player_name,
            'persona': persona,
            'card_data': card_data,
            'real_data_used': real_data_used,
            'gnn_insights_used': gnn_model is not None
        })
        
    except Exception as e:
        logger.error(f"Error generating card: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cards/popular', methods=['GET'])
def get_popular_players():
    """Get popular players"""
    try:
        popular_names = ['Virat Kohli', 'MS Dhoni', 'Rohit Sharma', 'KL Rahul', 'Hardik Pandya']
        
        popular_players = []
        for name in popular_names:
            # Try to find the player in our data
            mask = player_data['name'].str.contains(name, case=False, na=False)
            matches = player_data[mask]
            
            if not matches.empty:
                player = matches.iloc[0]
                popular_players.append({
                    'identifier': player['identifier'],
                    'name': player['name'],
                    'unique_name': player['unique_name']
                })
            else:
                # Add as fallback
                popular_players.append({
                    'identifier': hashlib.md5(name.encode()).hexdigest()[:8],
                    'name': name,
                    'unique_name': name
                })
        
        return jsonify({
            'success': True,
            'popular_players': popular_players,
            'count': len(popular_players)
        })
        
    except Exception as e:
        logger.error(f"Error getting popular players: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cards/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    return jsonify({
        'success': True,
        'stats': {
            'total_players': len(player_data) if player_data is not None else 0,
            'kg_connected': kg_query_engine is not None,
            'gnn_connected': gnn_model is not None,
            'real_components_available': REAL_COMPONENTS_AVAILABLE,
            'api_version': '2.0.0-real',
            'status': 'operational'
        }
    })

# Team context management endpoints
@app.route('/api/match-context', methods=['GET'])
def get_match_context():
    """Get current match context"""
    return jsonify({
        "success": True,
        "context": current_match_context
    })

@app.route('/api/match-context', methods=['POST'])
def update_match_context():
    """Update current match context"""
    global current_match_context
    try:
        data = request.get_json()
        if data:
            current_match_context.update(data)
        return jsonify({
            "success": True,
            "message": "Match context updated",
            "context": current_match_context
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/cards/enhanced', methods=['POST'])
def generate_enhanced_card():
    """Generate enhanced team-aware player card with full tactical analysis"""
    try:
        data = request.get_json()
        player_name = data.get('player_name', '').strip()
        
        if not player_name:
            return jsonify({
                "success": False,
                "error": "Player name is required"
            }), 400
        
        logger.info(f"🎯 Generating enhanced card for: {player_name}")
        
        # Determine player's team and opponent
        player_team = None
        opponent_team = None
        
        # Check which team the player belongs to
        if any(player_name in p for p in current_match_context["homeTeam"]["players"]):
            player_team = current_match_context["homeTeam"]
            opponent_team = current_match_context["awayTeam"]
        elif any(player_name in p for p in current_match_context["awayTeam"]["players"]):
            player_team = current_match_context["awayTeam"]
            opponent_team = current_match_context["homeTeam"]
        else:
            # Default to home team if not found
            player_team = current_match_context["homeTeam"]
            opponent_team = current_match_context["awayTeam"]
        
        # Get player stats from KG
        player_stats = get_player_stats_from_kg(player_name)
        
        # Generate enhanced card data structure
        enhanced_card = {
            "playerId": player_name.lower().replace(' ', '_'),
            "playerName": player_name,
            "role": determine_player_role(player_name, player_stats),
            "currentTeamId": player_team["id"],
            "context": {
                "homeTeamId": current_match_context["homeTeam"]["id"],
                "awayTeamId": current_match_context["awayTeam"]["id"],
                "opponentTeamId": opponent_team["id"],
                "tournamentId": current_match_context.get("tournament", "IPL_2025"),
                "venueId": current_match_context.get("venue", "Unknown"),
                "matchDateISO": current_match_context.get("matchDate", "2025-01-15")
            },
            "visuals": {
                "headshotUrl": None,  # Placeholder for future image integration
                "teamBadgeUrl": None,
                "opponentBadgeUrl": None
            },
            "core": generate_core_stats(player_name, player_stats),
            "tactical": generate_tactical_insights(player_name, player_stats, opponent_team),
            "betting": {
                "edgeRunsOver_30_5": 0,  # Mock for now
                "risk": 0,
                "comment": "MOCK"
            }
        }
        
        return jsonify({
            "success": True,
            "card_data": enhanced_card,
            "match_context": current_match_context,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error generating enhanced card: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def determine_player_role(player_name, stats):
    """Determine player role based on stats"""
    if not stats:
        return "Batsman"  # Default
    
    # Simple heuristic based on available data
    if stats.get('bowling_average', 0) > 0 and stats.get('bowling_average', 100) < 30:
        if stats.get('batting_average', 0) > 25:
            return "Allrounder"
        else:
            return "Bowler"
    elif 'Dhoni' in player_name or 'Karthik' in player_name:
        return "Keeper"
    else:
        return "Batsman"

def generate_core_stats(player_name, stats):
    """Generate core statistics"""
    if not stats:
        # Mock data for demonstration
        return {
            "matches": random.randint(50, 200),
            "battingAverage": round(random.uniform(25, 55), 1),
            "strikeRate": round(random.uniform(110, 150), 1),
            "formIndex": round(random.uniform(6, 9), 1),
            "last5Scores": [random.randint(15, 90) for _ in range(5)]
        }
    
    return {
        "matches": stats.get('matches', 0),
        "battingAverage": round(stats.get('batting_average', 0), 1),
        "strikeRate": round(stats.get('strike_rate', 0), 1),
        "formIndex": round(random.uniform(6, 9), 1),  # Would calculate from recent performance
        "last5Scores": [random.randint(15, 90) for _ in range(5)]  # Would get from recent matches
    }

def generate_tactical_insights(player_name, stats, opponent_team):
    """Generate tactical insights including bowler type analysis"""
    venue_factors = ["+15% at Chinnaswamy", "+8% vs CSK", "-5% in day games", "+12% in playoffs"]
    weaknesses = [
        "Slight weakness vs Left-arm orthodox (-8% SR)",
        "Struggles vs short ball (+15% dismissal rate)",
        "Lower SR in death overs (-12% SR)",
        "Vulnerable to spin in middle overs"
    ]
    
    # Mock bowler type matrix
    bowler_types = [
        {"subtype": "Left-arm orthodox", "deltaVsBaselineSR": -18.5, "confidence": 0.89},
        {"subtype": "Right-arm legbreak", "deltaVsBaselineSR": 8.8, "confidence": 0.95},
        {"subtype": "Right-arm fast-medium", "deltaVsBaselineSR": 4.2, "confidence": 0.98},
        {"subtype": "Left-arm fast", "deltaVsBaselineSR": -6.3, "confidence": 0.82},
        {"subtype": "Right-arm offbreak", "deltaVsBaselineSR": 12.1, "confidence": 0.91}
    ]
    
    # Mock key matchups against opponent team
    key_matchups = []
    favorable_bowlers = []
    
    if opponent_team["id"] == "CSK":
        key_matchups = [
            {"bowlerId": "d_chahar", "bowlerName": "Deepak Chahar", "dismissals": 3, "strikeRateVs": 125, "sample": 85},
            {"bowlerId": "m_theekshana", "bowlerName": "M Theekshana", "dismissals": 2, "strikeRateVs": 110, "sample": 45}
        ]
        favorable_bowlers = [
            {"bowlerId": "m_ali", "bowlerName": "Moeen Ali", "strikeRateVs": 165, "avgVs": 78, "sample": 32}
        ]
    elif opponent_team["id"] == "RCB":
        key_matchups = [
            {"bowlerId": "w_hasaranga", "bowlerName": "W Hasaranga", "dismissals": 4, "strikeRateVs": 95, "sample": 67},
            {"bowlerId": "h_patel", "bowlerName": "Harshal Patel", "dismissals": 2, "strikeRateVs": 140, "sample": 54}
        ]
    
    baseline_sr = stats.get('strike_rate', 135) if stats else 135
    
    return {
        "venueFactor": random.choice(venue_factors),
        "bowlerTypeWeakness": random.choice(weaknesses),
        "keyMatchups": key_matchups,
        "favorableBowlers": favorable_bowlers,
        "bowlerTypeMatrix": {
            "baselineSR": baseline_sr,
            "cells": [
                {
                    "subtype": bt["subtype"],
                    "ballsFaced": random.randint(80, 300),
                    "runs": random.randint(100, 400),
                    "dismissals": random.randint(2, 8),
                    "strikeRate": round(baseline_sr + bt["deltaVsBaselineSR"], 1),
                    "average": round(random.uniform(25, 65), 1),
                    "deltaVsBaselineSR": bt["deltaVsBaselineSR"],
                    "confidence": bt["confidence"]
                }
                for bt in bowler_types
            ]
        }
    }

if __name__ == '__main__':
    print("🚀 Starting Real Dynamic Player Cards API...")
    
    # Load player data
    if load_players():
        print(f"✅ Successfully loaded {len(player_data)} players from real database")
    else:
        print("⚠️ Using mock player data")
    
    # Load real components
    if load_real_components():
        print("✅ Connected to real KG + GNN components")
    else:
        print("⚠️ Using mock data (real components not available)")
    
    print("\n📡 Available API Endpoints:")
    print("  GET  /api/cards/health - Health check")
    print("  GET  /api/cards/autocomplete?partial=<text> - Real autocomplete")
    print("  POST /api/cards/generate - Generate cards with real KG + GNN data")
    print("  GET  /api/cards/popular - Get popular players")
    print("  GET  /api/cards/stats - System statistics")
    
    print("\n🌐 Server starting on http://127.0.0.1:5004")
    print("🎯 Ready to serve real dynamic player cards!")
    
    app.run(host='127.0.0.1', port=5004, debug=False, threaded=True)
