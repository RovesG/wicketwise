# Purpose: Enhanced Flask API for dynamic cricket player cards with KG integration
# Author: Assistant, Last Modified: 2025-01-19

import os
import sys
import json
import random
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add crickformers to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'crickformers'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for KG and GNN
kg_query_engine = None
gnn_model = None

def initialize_systems():
    """Initialize KG and GNN systems"""
    global kg_query_engine, gnn_model
    
    try:
        # Initialize Knowledge Graph Query Engine
        from crickformers.chat.unified_kg_query_engine import UnifiedKGQueryEngine
        
        # Load the built knowledge graph
        kg_path = "models/unified_cricket_kg.pkl"
        if os.path.exists(kg_path):
            kg_query_engine = UnifiedKGQueryEngine(kg_path)
            logger.info("✅ Knowledge Graph Query Engine initialized")
        else:
            logger.warning("⚠️ Knowledge graph not found - using mock data")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize KG: {e}")
    
    try:
        # Initialize GNN (if available)
        # TODO: Add GNN initialization when ready
        logger.info("ℹ️ GNN initialization skipped (not implemented yet)")
    except Exception as e:
        logger.error(f"❌ Failed to initialize GNN: {e}")

def get_player_image_url(player_name, teams=None):
    """Get the best available cricket player image from multiple sources"""
    try:
        from enhanced_cricket_image_service import get_enhanced_player_image
        
        # Use enhanced image service for real cricket photos
        image_result = get_enhanced_player_image(player_name, teams)
        
        logger.info(f"🖼️ Image for {player_name}: {image_result['source']} ({'cached' if image_result['cached'] else 'fresh'})")
        
        return image_result['url']
        
    except ImportError:
        logger.warning("Enhanced image service not available, using fallback")
        # Fallback to team-colored avatar
        clean_name = player_name.replace(' ', '+').replace("'", "").replace('.', '')
        bg_color = '1e40af'  # Default cricket blue
        
        if teams and len(teams) > 0:
            team_colors = {
                'Chennai Super Kings': 'fbbf24', 'Mumbai Indians': '1e40af',
                'Royal Challengers Bangalore': 'dc2626', 'Royal Challengers Bengaluru': 'dc2626',
                'Kolkata Knight Riders': '7c3aed', 'Delhi Capitals': '1e40af',
                'Rajasthan Royals': 'ec4899', 'Sunrisers Hyderabad': 'ea580c',
                'Punjab Kings': 'dc2626', 'Gujarat Titans': '1e40af',
                'Lucknow Super Giants': '06b6d4', 'India': '1e40af',
                'Australia': 'fbbf24', 'England': 'dc2626', 'South Africa': '059669',
                'Pakistan': '059669', 'West Indies': '7c2d12', 'New Zealand': '000000',
                'Sri Lanka': '1e40af', 'Bangladesh': '059669'
            }
            bg_color = team_colors.get(teams[0], '1e40af')
        
        return (f"https://ui-avatars.com/api/"
               f"?name={clean_name}&background={bg_color}&color=fff&size=150"
               f"&font-size=0.4&format=png&rounded=true&bold=true")
    
    except Exception as e:
        logger.error(f"❌ Image service failed for {player_name}: {e}")
        # Ultimate fallback
        return f"https://ui-avatars.com/api/?name={player_name.replace(' ', '+')}&background=1e40af&color=fff&size=150"

def get_openai_player_image(player_name):
    """
    Future: Use OpenAI to search for cricket player images
    This would require OpenAI API key and proper image search integration
    """
    # TODO: Implement OpenAI image search
    # Example prompt: f"Find a professional headshot photo of cricket player {player_name}"
    # This would use OpenAI's image search capabilities or DALL-E for consistent avatars
    pass

def generate_cricket_avatar(player_name, team_colors=None):
    """
    Generate a cricket-themed avatar for the player
    """
    # Clean name for URL
    clean_name = player_name.replace(' ', '+').replace("'", "")
    
    # Cricket-themed colors
    cricket_colors = {
        'blue': '1e40af',      # Cricket blue
        'green': '059669',     # Cricket field green  
        'orange': 'ea580c',    # Cricket ball orange
        'red': 'dc2626',       # Test cricket ball red
        'gold': 'f59e0b',      # Trophy gold
    }
    
    # Use team colors if provided, otherwise default to cricket blue
    bg_color = cricket_colors['blue']
    if team_colors:
        bg_color = team_colors.get('primary', cricket_colors['blue'])
    
    return f"https://ui-avatars.com/api/?name={clean_name}&background={bg_color}&color=fff&size=150&font-size=0.4&format=png&rounded=true"

def get_real_player_data(player_name):
    """Get comprehensive real player data from KG"""
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
            
            # Extract comprehensive data from the KG structure
            batting_stats = player_stats.get('batting_stats', {})
            bowling_stats = player_stats.get('bowling_stats', {})
            vs_pace = player_stats.get('vs_pace', {})
            vs_spin = player_stats.get('vs_spin', {})
            powerplay = player_stats.get('in_powerplay', {})
            death_overs = player_stats.get('in_death_overs', {})
            
            # Determine player role and format stats
            primary_role = player_stats.get('primary_role', 'batsman')
            
            # Build format-specific strike rates
            format_stats = {
                'overall_sr': vs_pace.get('strike_rate', batting_stats.get('strike_rate', 0)),
                'powerplay_sr': powerplay.get('strike_rate', 0),
                'death_overs_sr': death_overs.get('strike_rate', 0),
                'vs_pace_sr': vs_pace.get('strike_rate', 0),
                'vs_spin_sr': vs_spin.get('strike_rate', 0),
                # TODO: Add T20, Test, ODI when available in KG
            }
            
            # Calculate bowling economy if bowling data exists
            bowling_economy = 0
            if bowling_stats.get('balls', 0) > 0:
                bowling_economy = round(bowling_stats.get('runs', 0) / bowling_stats.get('balls', 1) * 6, 2)
            
            return {
                'player_name': player_stats.get('player', player_name),
                'primary_role': primary_role,
                'batting_avg': batting_stats.get('average', 0),
                'strike_rate': format_stats['overall_sr'],
                'format_stats': format_stats,
                'matches_played': player_stats.get('matches_played', 0),
                'teams': player_stats.get('teams', []),
                'powerplay_sr': powerplay.get('strike_rate', 0),
                'death_overs_sr': death_overs.get('strike_rate', 0),
                'vs_pace_avg': vs_pace.get('average', 0),
                'vs_spin_avg': vs_spin.get('average', 0),
                # Bowling stats
                'bowling_stats': {
                    'balls': bowling_stats.get('balls', 0),
                    'runs_conceded': bowling_stats.get('runs', 0),
                    'wickets': bowling_stats.get('wickets', 0),
                    'economy': bowling_economy,
                    'bowling_avg': round(bowling_stats.get('runs', 0) / max(bowling_stats.get('wickets', 1), 1), 2) if bowling_stats.get('wickets', 0) > 0 else 0
                },
                # Additional insights
                'strengths': player_stats.get('strengths', []),
                'style_analysis': player_stats.get('style_analysis', {}),
                'venues_played': player_stats.get('venues_played', []),
                'venue_performance': player_stats.get('venue_performance', {}),
                'profile_image_url': get_player_image_url(player_name, player_stats.get('teams', [])),
                'last_updated': datetime.now().isoformat(),
                'data_sources': ['Real_KG_Data'],
                'source': 'Real_KG_Data'
            }
        else:
            logger.warning(f"⚠️ DEBUG: No player stats returned for {player_name}")
            return None
            
    except Exception as e:
        logger.error(f"❌ DEBUG: Error getting player profile: {e}")
        return None

def generate_mock_card_data(player_name, persona):
    """Generate OBVIOUSLY FAKE mock card data (fallback when real data unavailable)"""
    logger.info(f"⚠️ Using MOCK data for {player_name}")
    
    # Generate obviously fake recent matches
    recent_matches = []
    for i in range(5):
        recent_matches.append({
            'score': '0',  # Obviously fake score
            'opponent': 'MOCK_TEAM',
            'days_ago': (i * 4 + 3)
        })
    
    return {
        'player_name': player_name,
        'primary_role': 'MOCK_ROLE',
        'batting_avg': 0.0,  # OBVIOUSLY FAKE
        'strike_rate': 0.0,  # OBVIOUSLY FAKE
        'format_stats': {
            'overall_sr': 0.0,
            'powerplay_sr': 0.0,
            'death_overs_sr': 0.0,
            'vs_pace_sr': 0.0,
            'vs_spin_sr': 0.0,
        },
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
        'bowling_stats': {
            'balls': 0,
            'runs_conceded': 0,
            'wickets': 0,
            'economy': 0.0,
            'bowling_avg': 0.0
        },
        'current_match_status': 'MOCK_STATUS',  # OBVIOUSLY FAKE
        'last_6_balls': ['0', '0', '0', '0', '0', '0'],  # OBVIOUSLY FAKE
        'profile_image_url': get_player_image_url(player_name, ['MOCK_TEAM']),
        'profile_image_cached': False,
        'last_updated': datetime.now().isoformat(),
        'data_sources': ['MOCK_DATA_OBVIOUS'],
        'source': 'MOCK_DATA_OBVIOUS'
    }

def generate_enhanced_card_data(player_name, persona):
    """Generate enhanced card data with real data first, fallback to mock"""
    logger.info(f"🎴 Generating enhanced card for {player_name} (persona: {persona})")
    
    # Try to get real data first
    real_data = get_real_player_data(player_name)
    
    if real_data:
        logger.info(f"✅ Using real KG data for {player_name}")
        return real_data
    else:
        logger.info(f"⚠️ Falling back to mock data for {player_name}")
        return generate_mock_card_data(player_name, persona)

# API Routes
@app.route('/api/cards/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'kg_available': kg_query_engine is not None,
        'gnn_available': gnn_model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/cards/generate', methods=['POST'])
def generate_card():
    """Generate a dynamic player card"""
    try:
        data = request.get_json()
        player_name = data.get('player_name', '').strip()
        persona = data.get('persona', 'betting').lower()
        
        if not player_name:
            return jsonify({'success': False, 'error': 'Player name is required'}), 400
        
        logger.info(f"🎴 Generating real card for '{player_name}' (persona: {persona})")
        
        # Generate card data
        card_result = generate_enhanced_card_data(player_name, persona)
        
        # Check if we got real data or mock data
        real_data_used = False
        if isinstance(card_result, dict):
            if 'data_sources' in card_result:
                real_data_used = 'Real_KG_Data' in card_result['data_sources']
            elif 'source' in card_result:
                real_data_used = card_result['source'] == 'Real_KG_Data'
            card_data = card_result
        else:
            card_data = card_result # This path is for mock data
        
        return jsonify({
            'success': True,
            'player_name': player_name,
            'persona': persona,
            'card_data': card_data,
            'real_data_used': real_data_used,
            'gnn_insights_used': gnn_model is not None
        })
        
    except Exception as e:
        logger.error(f"❌ Error generating card: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cards/autocomplete', methods=['GET'])
def autocomplete_players():
    """Get player name suggestions"""
    try:
        partial = request.args.get('partial', '').strip()
        limit = int(request.args.get('limit', 5))
        
        if len(partial) < 2:
            return jsonify({'success': True, 'suggestions': [], 'source': 'none'})
        
        # For now, return some common players
        # TODO: Implement real autocomplete from people.csv
        common_players = [
            'Virat Kohli', 'MS Dhoni', 'Rohit Sharma', 'KL Rahul', 'Hardik Pandya',
            'Jasprit Bumrah', 'Ravindra Jadeja', 'Shikhar Dhawan', 'Rishabh Pant',
            'Mohammed Shami', 'Yuzvendra Chahal', 'Bhuvneshwar Kumar'
        ]
        
        # Filter based on partial match
        suggestions = [player for player in common_players 
                      if partial.lower() in player.lower()][:limit]
        
        return jsonify({
            'success': True,
            'suggestions': suggestions,
            'source': 'common_players'
        })
        
    except Exception as e:
        logger.error(f"❌ Error in autocomplete: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cards/popular', methods=['GET'])
def get_popular_players():
    """Get list of popular players"""
    popular_players = [
        'Virat Kohli', 'MS Dhoni', 'Rohit Sharma', 'KL Rahul', 
        'Hardik Pandya', 'Jasprit Bumrah', 'Ravindra Jadeja'
    ]
    
    return jsonify({
        'success': True,
        'players': popular_players
    })

@app.route('/api/betting/intelligence', methods=['POST'])
def get_betting_intelligence():
    """Get comprehensive betting intelligence for a player"""
    try:
        data = request.get_json()
        player_name = data.get('player_name')
        player_stats = data.get('player_stats', {})
        
        if not player_name:
            return jsonify({'error': 'Player name is required'}), 400
        
        logger.info(f"🎯 Generating betting intelligence for {player_name}")
        
        # Import and use betting intelligence system
        try:
            from betting_intelligence_system import get_player_betting_intelligence
            
            intelligence = get_player_betting_intelligence(player_name, player_stats)
            
            # Convert to JSON-serializable format
            intelligence_dict = {
                'player_name': intelligence.player_name,
                'form_rating': intelligence.form_rating,
                'volatility': intelligence.volatility,
                'consistency_score': intelligence.consistency_score,
                'pressure_rating': intelligence.pressure_rating,
                'venue_factor': intelligence.venue_factor,
                'matchup_factor': intelligence.matchup_factor,
                'risk_assessment': intelligence.risk_assessment,
                'recent_trends': intelligence.recent_trends,
                'markets': [
                    {
                        'market_type': m.market_type,
                        'line': m.line,
                        'over_odds': m.over_odds,
                        'under_odds': m.under_odds,
                        'model_probability': m.model_probability,
                        'market_probability': m.market_probability,
                        'expected_value': m.expected_value,
                        'confidence': m.confidence,
                        'volume': m.volume
                    }
                    for m in intelligence.markets
                ],
                'betting_recommendations': intelligence.betting_recommendations
            }
            
            logger.info(f"✅ Betting intelligence generated: {len(intelligence.markets)} markets, {len(intelligence.betting_recommendations)} recommendations")
            
            return jsonify(intelligence_dict)
            
        except ImportError as e:
            logger.error(f"❌ Betting intelligence system not available: {e}")
            return jsonify({'error': 'Betting intelligence system not available'}), 503
            
    except Exception as e:
        logger.error(f"❌ Error generating betting intelligence: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/betting/query', methods=['POST'])
def process_betting_query():
    """Process natural language betting queries"""
    try:
        data = request.get_json()
        player_name = data.get('player_name')
        query = data.get('query')
        player_stats = data.get('player_stats', {})
        
        if not player_name or not query:
            return jsonify({'error': 'Player name and query are required'}), 400
        
        logger.info(f"🤖 Processing betting query for {player_name}: {query}")
        
        # Import and use betting query system
        try:
            from betting_intelligence_system import process_player_betting_query
            
            result = process_player_betting_query(player_name, query, player_stats)
            
            logger.info(f"✅ Betting query processed successfully")
            
            return jsonify(result)
            
        except ImportError as e:
            logger.error(f"❌ Betting query system not available: {e}")
            # Fallback response
            return jsonify({
                'response': f"I'd love to help with betting analysis for {player_name}, but the betting intelligence system is currently unavailable. Please try again later.",
                'confidence': 0.0
            })
            
    except Exception as e:
        logger.error(f"❌ Error processing betting query: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Dynamic Cards API v2.0...")
    
    # Initialize systems
    initialize_systems()
    
    print("🌐 API available at: http://127.0.0.1:5005")
    print("📊 Health check: http://127.0.0.1:5005/api/cards/health")
    print("🎴 Ready to generate dynamic player cards!")
    
    app.run(host='0.0.0.0', port=5005, debug=True)
