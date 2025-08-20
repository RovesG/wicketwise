# Purpose: API endpoints for dynamic player cards
# Author: WicketWise Team, Last Modified: August 19, 2024

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from dynamic_player_cards import create_dynamic_card_system
from dataclasses import asdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize the dynamic card system
people_csv_path = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/people.csv"
openai_api_key = os.getenv('OPENAI_API_KEY')  # Optional

# Initialize card system
card_system = create_dynamic_card_system(
    people_csv_path=people_csv_path,
    openai_api_key=openai_api_key
)

@app.route('/api/cards/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Dynamic Player Cards API',
        'players_available': len(card_system.player_index),
        'openai_enabled': bool(openai_api_key)
    })

@app.route('/api/cards/search', methods=['GET'])
def search_players():
    """Search for players with autocomplete support"""
    try:
        query = request.args.get('q', '')
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({'error': 'Query parameter "q" is required'}), 400
        
        results = card_system.search_players(query, limit)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in player search: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cards/autocomplete', methods=['GET'])
def autocomplete():
    """Get autocomplete suggestions for player names"""
    try:
        partial = request.args.get('partial', '')
        limit = int(request.args.get('limit', 5))
        
        if not partial:
            return jsonify({'error': 'Parameter "partial" is required'}), 400
        
        suggestions = card_system.get_autocomplete_suggestions(partial, limit)
        
        return jsonify({
            'success': True,
            'partial': partial,
            'suggestions': suggestions,
            'count': len(suggestions)
        })
        
    except Exception as e:
        logger.error(f"Error in autocomplete: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cards/generate', methods=['POST'])
def generate_card():
    """Generate a dynamic player card"""
    try:
        data = request.json
        player_name = data.get('player_name')
        persona = data.get('persona', 'betting')
        
        if not player_name:
            return jsonify({'error': 'player_name is required'}), 400
        
        # Generate the card
        card_data = card_system.generate_player_card(player_name, persona)
        
        return jsonify({
            'success': True,
            'player_name': player_name,
            'persona': persona,
            'card_data': asdict(card_data)
        })
        
    except Exception as e:
        logger.error(f"Error generating card: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cards/batch-generate', methods=['POST'])
def batch_generate_cards():
    """Generate multiple player cards at once"""
    try:
        data = request.json
        player_names = data.get('player_names', [])
        persona = data.get('persona', 'betting')
        
        if not player_names:
            return jsonify({'error': 'player_names array is required'}), 400
        
        cards = {}
        errors = {}
        
        for player_name in player_names:
            try:
                card_data = card_system.generate_player_card(player_name, persona)
                cards[player_name] = asdict(card_data)
            except Exception as e:
                errors[player_name] = str(e)
                logger.error(f"Error generating card for {player_name}: {e}")
        
        return jsonify({
            'success': True,
            'persona': persona,
            'cards': cards,
            'errors': errors,
            'generated_count': len(cards),
            'error_count': len(errors)
        })
        
    except Exception as e:
        logger.error(f"Error in batch generation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cards/popular', methods=['GET'])
def get_popular_players():
    """Get a list of popular players for quick access"""
    try:
        # Define popular players (could be data-driven in the future)
        popular_players = [
            'Virat Kohli', 'MS Dhoni', 'Rohit Sharma', 'KL Rahul', 
            'Hardik Pandya', 'Suryakumar Yadav', 'Rishabh Pant',
            'Jasprit Bumrah', 'AB de Villiers', 'David Warner'
        ]
        
        # Filter to only include players that exist in our index
        available_popular = []
        for player in popular_players:
            search_results = card_system.search_players(player, limit=1)
            if search_results:
                available_popular.append(search_results[0])
        
        return jsonify({
            'success': True,
            'popular_players': available_popular,
            'count': len(available_popular)
        })
        
    except Exception as e:
        logger.error(f"Error getting popular players: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cards/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics"""
    try:
        cache_dir = card_system.cache_dir
        cached_cards = len(list(cache_dir.glob("*_card.json")))
        cached_images = len(list(cache_dir.glob("*_image.json")))
        
        return jsonify({
            'success': True,
            'stats': {
                'total_players': len(card_system.player_index),
                'cached_cards': cached_cards,
                'cached_images': cached_images,
                'openai_enabled': bool(card_system.openai_api_key),
                'kg_enabled': bool(card_system.kg_query_engine)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Dynamic Player Cards API Server...")
    logger.info(f"📊 Loaded {len(card_system.player_index)} players")
    logger.info(f"🔑 OpenAI enabled: {bool(openai_api_key)}")
    
    print("\n📡 Available API Endpoints:")
    print("  GET  /api/cards/health - Health check")
    print("  GET  /api/cards/search?q=<query> - Search players")
    print("  GET  /api/cards/autocomplete?partial=<text> - Autocomplete")
    print("  POST /api/cards/generate - Generate single card")
    print("  POST /api/cards/batch-generate - Generate multiple cards")
    print("  GET  /api/cards/popular - Get popular players")
    print("  GET  /api/cards/stats - System statistics")
    print("\n🌐 Server starting on http://127.0.0.1:5003")
    print("🎯 Ready to serve dynamic player cards!")
    
    app.run(host='127.0.0.1', port=5003, debug=True)
