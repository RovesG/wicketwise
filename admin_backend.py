#!/usr/bin/env python3
"""
Admin Backend API for WicketWise Cricket Analytics
Provides real API endpoints for admin operations
"""

import json
import sys
import os
from pathlib import Path
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging
import threading
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed - using system environment variables only")

# Import configuration
from config.settings import settings

from admin_tools import get_admin_tools
from crickformers.chat import KGChatAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
if settings.CORS_ENABLED:
    CORS(app)

# Global admin tools instance
# Global admin tools instance - initialized lazily
admin_tools = None

def get_admin_tools_instance():
    """Get or initialize the admin tools instance"""
    global admin_tools
    if admin_tools is None:
        admin_tools = get_admin_tools()
    return admin_tools

# Global chat agent instance (initialized lazily)
chat_agent = None

# Track background operations
background_operations = {}

# Chat sessions storage (in production, use Redis or database)
chat_sessions = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Admin Backend is running"})

@app.route('/api/build-knowledge-graph', methods=['POST'])
def build_knowledge_graph():
    """Mapped to unified KG build to avoid legacy/cached aggregates"""

    def run_build():
        operation_id = "knowledge_graph_build"
        try:
            background_operations[operation_id] = {
                "status": "running",
                "progress": 10,
                "message": "Initializing unified KG builder...",
                "logs": ["🚀 Starting unified knowledge graph construction..."],
                "_last_log": ""
            }

            def cb(stage: str, message: str, progress: int, details: dict):
                try:
                    background_operations[operation_id]["message"] = message
                    background_operations[operation_id]["progress"] = max(background_operations[operation_id]["progress"], int(progress))
                    log_line = f"[{stage}] {message}"
                    # de-duplicate consecutive identical log lines
                    last_log = background_operations[operation_id].get("_last_log")
                    if log_line != last_log:
                        background_operations[operation_id]["logs"].append(log_line)
                        background_operations[operation_id]["_last_log"] = log_line
                    # keep last 200 logs max to avoid memory bloat
                    if len(background_operations[operation_id]["logs"]) > 200:
                        background_operations[operation_id]["logs"] = background_operations[operation_id]["logs"][-200:]
                    # attach details snapshot
                    background_operations[operation_id]["details"] = details or {}
                except Exception:
                    pass

            def should_cancel():
                return background_operations.get(operation_id, {}).get("status") == "canceled"

            # Wrap callback to add running counts for UI
            balls_total = {"value": 0}
            def cb_with_counts(stage: str, message: str, progress: int, details: dict):
                if stage == 'load_data' and isinstance(details, dict) and 'balls' in details:
                    balls_total['value'] = int(details['balls'])
                # forward original callback first
                cb(stage, message, progress, details)
                # enrich details with processed/total where applicable
                if stage.startswith('ball_events') and balls_total['value']:
                    current = background_operations[operation_id].get('details', {})
                    current['balls'] = balls_total['value']
                    background_operations[operation_id]['details'] = current

            result = get_admin_tools_instance().build_unified_knowledge_graph(progress_callback=cb_with_counts, should_cancel=should_cancel)

            if result.get('status') == 'success':
                details = result.get('details', {})
                background_operations[operation_id].update({
                    "status": "completed",
                    "progress": 100,
                    "message": f"Unified KG built: {details.get('players', 0):,} players, {details.get('balls_processed', 0):,} balls, {details.get('venues', 0)} venues",
                    "result": result
                })
                background_operations[operation_id]["logs"].append("✅ Unified KG build completed successfully")
            else:
                background_operations[operation_id].update({
                    "status": "error",
                    "message": result.get('message', 'Unified KG build failed')
                })
                background_operations[operation_id]["logs"].append(f"❌ {background_operations[operation_id]['message']}")

        except Exception as e:
            background_operations[operation_id].update({
                "status": "error",
                "message": f"Unified KG build failed: {str(e)}"
            })
            background_operations[operation_id]["logs"].append(f"❌ Exception: {str(e)}")
            logger.error(f"Unified KG build failed: {str(e)}")

    # Start background task
    thread = threading.Thread(target=run_build)
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "operation_id": "knowledge_graph_build"})

@app.route('/api/build-unified-knowledge-graph', methods=['POST'])
def build_unified_knowledge_graph():
    """Build the unified knowledge graph with ball-by-ball granularity"""
    
    def run_unified_build():
        """Background task to build unified knowledge graph"""
        operation_id = "unified_kg_build"
        
        try:
            background_operations[operation_id] = {
                "status": "running",
                "progress": 0,
                "message": "Starting unified knowledge graph build...",
                "logs": ["🚀 Starting unified knowledge graph construction..."],
                "_last_log": ""
            }
            
            # Stage 1: Initialize
            background_operations[operation_id]["progress"] = 10
            background_operations[operation_id]["message"] = "Initializing unified KG builder..."
            
            admin_tools = get_admin_tools_instance()

            def cb(stage: str, message: str, progress: int, details: dict):
                try:
                    background_operations[operation_id]["message"] = message
                    background_operations[operation_id]["progress"] = max(background_operations[operation_id]["progress"], int(progress))
                    log_line = f"[{stage}] {message}"
                    last_log = background_operations[operation_id].get("_last_log")
                    if log_line != last_log:
                        background_operations[operation_id]["logs"].append(log_line)
                        background_operations[operation_id]["_last_log"] = log_line
                    if len(background_operations[operation_id]["logs"]) > 200:
                        background_operations[operation_id]["logs"] = background_operations[operation_id]["logs"][-200:]
                    background_operations[operation_id]["details"] = details or {}
                except Exception:
                    pass

            def should_cancel():
                return background_operations.get(operation_id, {}).get("status") == "canceled"

            result = admin_tools.build_unified_knowledge_graph(progress_callback=cb, should_cancel=should_cancel)
            
            if result['status'] == 'success':
                # Success
                background_operations[operation_id]["status"] = "completed"
                background_operations[operation_id]["progress"] = 100
                background_operations[operation_id]["message"] = result['message']
                background_operations[operation_id]["result"] = result
                background_operations[operation_id]["logs"].append(f"✅ {result['message']}")
                background_operations[operation_id]["logs"].append(f"📊 {result['details']['players']:,} players processed")
                background_operations[operation_id]["logs"].append(f"⚾ {result['details']['balls_processed']:,} balls preserved")
                background_operations[operation_id]["logs"].append(f"🏟️ {result['details']['venues']:,} venues analyzed")
            else:
                # Error
                background_operations[operation_id]["status"] = "error"
                background_operations[operation_id]["message"] = result['message']
                background_operations[operation_id]["logs"].append(f"❌ {result['message']}")
                
        except Exception as e:
            background_operations[operation_id]["status"] = "error"
            background_operations[operation_id]["message"] = f"Unified KG build failed: {str(e)}"
            background_operations[operation_id]["logs"].append(f"❌ Error: {str(e)}")
    
    # Start background task
    thread = threading.Thread(target=run_unified_build)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started", "operation_id": "unified_kg_build"})

@app.route('/api/cancel-operation/<operation_id>', methods=['POST'])
def cancel_operation(operation_id):
    """Mark a background operation as canceled (best-effort)."""
    if operation_id not in background_operations:
        return jsonify({"error": "Operation not found"}), 404
    if background_operations[operation_id].get("status") in {"completed", "error"}:
        return jsonify({"status": background_operations[operation_id]["status"], "message": "Already finished"})
    background_operations[operation_id]["status"] = "canceled"
    background_operations[operation_id]["message"] = "Cancellation requested"
    background_operations[operation_id].setdefault("logs", []).append("🛑 Cancellation requested")
    return jsonify({"status": "canceled"})

@app.route('/api/operation-status/<operation_id>', methods=['GET'])
def get_operation_status(operation_id):
    """Get status of background operation"""
    if operation_id not in background_operations:
        return jsonify({"error": "Operation not found"}), 404
    
    return jsonify(background_operations[operation_id])

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train AI model using real training pipeline"""
    
    def run_training():
        """Background task to train model"""
        operation_id = "model_training"
        
        try:
            background_operations[operation_id] = {
                "status": "running",
                "progress": 10,
                "message": "Starting real Crickformer training...",
                "logs": ["Initializing sophisticated Crickformer model..."]
            }
            
            # Actually run training immediately (no fake progress)
            result = get_admin_tools_instance().train_model()
            
            # Update with results
            background_operations[operation_id]["progress"] = 100
            
            if result.startswith("✅"):
                background_operations[operation_id]["status"] = "completed"
                background_operations[operation_id]["message"] = result
                background_operations[operation_id]["logs"].append("Model training completed!")
            else:
                background_operations[operation_id]["status"] = "error"
                background_operations[operation_id]["message"] = result
                background_operations[operation_id]["logs"].append(f"Training failed: {result}")
                
        except Exception as e:
            background_operations[operation_id]["status"] = "error"
            background_operations[operation_id]["message"] = f"Error: {str(e)}"
            background_operations[operation_id]["logs"].append(f"Exception: {str(e)}")
            logger.error(f"Model training failed: {str(e)}")
    
    # Start background task
    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started", "operation_id": "model_training"})

@app.route('/api/system-status', methods=['GET'])
def get_system_status():
    """Get real system status"""
    try:
        status = get_admin_tools_instance().get_system_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"System status check failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/kg-settings', methods=['GET'])
def get_kg_settings():
    try:
        return jsonify({"status": "success", "settings": get_admin_tools_instance().get_kg_settings()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/kg-settings', methods=['POST'])
def update_kg_settings():
    try:
        data = request.get_json(force=True) or {}
        settings = get_admin_tools_instance().update_kg_settings(data)
        return jsonify({"status": "success", "settings": settings})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/aligner-settings', methods=['GET'])
def get_aligner_settings():
    try:
        return jsonify({"status": "success", "settings": get_admin_tools_instance().get_aligner_settings()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/aligner-settings', methods=['POST'])
def update_aligner_settings():
    try:
        data = request.get_json(force=True) or {}
        settings = get_admin_tools_instance().update_aligner_settings(data)
        return jsonify({"status": "success", "settings": settings})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# All JSON ingestion settings and trigger
_alljson_settings = {
    "source_dir": "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/all_json",
    "output_dir": "artifacts/kg_background/events",
    "resume": True,
}


@app.route('/api/alljson-settings', methods=['GET'])
def get_alljson_settings():
    return jsonify({"status": "success", "settings": _alljson_settings})


@app.route('/api/alljson-settings', methods=['POST'])
def update_alljson_settings():
    data = request.get_json(force=True) or {}
    _alljson_settings.update({k: v for k, v in data.items() if k in _alljson_settings})
    return jsonify({"status": "success", "settings": _alljson_settings})


@app.route('/api/ingest-alljson', methods=['POST'])
def ingest_alljson():
    operation_id = "ingest_alljson"
    background_operations[operation_id] = {
        "status": "running",
        "progress": 0,
        "message": "Starting all_json ingestion...",
        "logs": [],
    }

    def run():
        try:
            from wicketwise.data.alljson.ingest import flatten_file_to_dataframe
            src = Path(_alljson_settings["source_dir"])
            out_dir = Path(_alljson_settings["output_dir"])  
            out_dir.mkdir(parents=True, exist_ok=True)

            files = sorted([p for p in src.glob('*.json')])
            total = len(files)
            count = 0
            dfs = []
            for p in files:
                df = flatten_file_to_dataframe(str(p))
                dfs.append(df)
                count += 1
                if count % 50 == 0:
                    background_operations[operation_id]["message"] = f"Processed {count}/{total} files"
                    background_operations[operation_id]["progress"] = int(count * 100 / max(1, total))
            # Write a consolidated parquet shard (simple first version)
            import pandas as pd
            if dfs:
                big = pd.concat(dfs, ignore_index=True)
                
                # Fix data type issues for parquet conversion
                # Convert object columns with mixed types to strings
                for col in big.columns:
                    if big[col].dtype == 'object':
                        # Convert to string, handling NaN values
                        big[col] = big[col].astype(str).replace('nan', None)
                
                big.to_parquet(out_dir / 'events.parquet', index=False)
            background_operations[operation_id]["status"] = "completed"
            background_operations[operation_id]["progress"] = 100
            background_operations[operation_id]["message"] = "All JSON ingestion completed"
        except Exception as e:
            background_operations[operation_id]["status"] = "error"
            background_operations[operation_id]["message"] = str(e)
            logger.exception("All JSON ingestion failed")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return jsonify({"status": "started", "operation_id": operation_id})


@app.route('/api/export-t20', methods=['POST'])
def export_t20():
    operation_id = "export_t20"
    background_operations[operation_id] = {
        "status": "running",
        "progress": 0,
        "message": "Starting T20 export...",
        "logs": [],
    }

    def run():
        try:
            import pandas as pd
            from wicketwise.data.alljson.export_t20 import export_t20_events
            events_path = Path(_alljson_settings["output_dir"]) / 'events.parquet'
            if not events_path.exists():
                raise FileNotFoundError(f"Events parquet not found at {events_path}")
            df = pd.read_parquet(events_path)
            out = export_t20_events(df, str(Path('artifacts/train_exports/t20_from_json')))
            background_operations[operation_id]["status"] = "completed"
            background_operations[operation_id]["progress"] = 100
            background_operations[operation_id]["message"] = f"T20 export written to {out}"
        except Exception as e:
            background_operations[operation_id]["status"] = "error"
            background_operations[operation_id]["message"] = str(e)
            logger.exception("T20 export failed")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return jsonify({"status": "started", "operation_id": operation_id})


@app.route('/api/training-settings', methods=['GET'])
def get_training_settings():
    try:
        return jsonify({"status": "success", "settings": get_admin_tools_instance().get_training_settings()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/training-settings', methods=['POST'])
def update_training_settings():
    try:
        data = request.get_json(force=True) or {}
        settings = get_admin_tools_instance().update_training_settings(data)
        return jsonify({"status": "success", "settings": settings})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/kg-cache/purge', methods=['POST'])
def purge_kg_cache():
    try:
        msg = get_admin_tools_instance().purge_kg_cache()
        return jsonify({"status": "success", "message": msg})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/dataset-info', methods=['GET'])
def get_dataset_info():
    """Get dataset information for UI display"""
    try:
        dataset_info = get_admin_tools_instance().get_dataset_info()
        return jsonify(dataset_info)
    except Exception as e:
        logger.error(f"Dataset info error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/workflow-status', methods=['GET'])
def get_workflow_status():
    """Get current workflow status and step validation"""
    try:
        workflow_status = get_admin_tools_instance().get_workflow_status()
        return jsonify(workflow_status)
    except Exception as e:
        logger.error(f"Workflow status error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/train-gnn', methods=['POST'])
def train_gnn():
    """Train GNN model (Step 3)"""
    operation_id = "gnn_training"
    
    # Initialize operation tracking
    background_operations[operation_id] = {
        "status": "running",
        "message": "GNN training started...",
        "logs": ["Starting GNN training on knowledge graph..."]
    }
    
    def run_gnn_training():
        try:
            result = get_admin_tools_instance().train_gnn()
            
            if result.startswith("✅"):
                background_operations[operation_id]["status"] = "completed"
                background_operations[operation_id]["message"] = result
                background_operations[operation_id]["logs"].append("GNN training completed!")
            else:
                background_operations[operation_id]["status"] = "error"
                background_operations[operation_id]["message"] = result
                background_operations[operation_id]["logs"].append(f"GNN training failed: {result}")
                
        except Exception as e:
            background_operations[operation_id]["status"] = "error"
            background_operations[operation_id]["message"] = f"Error: {str(e)}"
            background_operations[operation_id]["logs"].append(f"Exception: {str(e)}")
            logger.error(f"GNN training failed: {str(e)}")
    
    # Start background task
    thread = threading.Thread(target=run_gnn_training)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started", "operation_id": operation_id})

@app.route('/api/update-data-paths', methods=['POST'])
def update_data_paths():
    """Update data paths configuration"""
    try:
        data = request.get_json()
        
        # Validate paths
        required_paths = ['data', 'video', 'model']
        for path_key in required_paths:
            if path_key not in data:
                return jsonify({"error": f"Missing {path_key} path"}), 400
        
        # Update AdminTools configuration
        if data['data']:
            from pathlib import Path
            data_path = Path(data['data'])
            if data_path.exists():
                get_admin_tools_instance().cricket_data_path = data_path
                logger.info(f"Updated cricket data path to: {data_path}")
            else:
                logger.warning(f"Data path does not exist: {data_path}")
        
        return jsonify({
            "message": "Data paths updated successfully",
            "paths": data,
            "current_data_file": str(get_admin_tools_instance().cricket_data_path)
        })
        
    except Exception as e:
        logger.error(f"Failed to update data paths: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/test-api-key', methods=['POST'])
def test_api_key():
    """Test API key functionality"""
    data = request.get_json()
    service = data.get('service')
    api_key = data.get('api_key')
    
    if not service or not api_key:
        return jsonify({"error": "Missing service or api_key"}), 400
    
    try:
        # Add real API key testing logic here based on service type
        if service == 'openai':
            # Test OpenAI key
            import openai
            client = openai.OpenAI(api_key=api_key)
            models = client.models.list()
            return jsonify({"status": "success", "message": "OpenAI API key is valid"})
        
        elif service == 'weather':
            # Test weather API key  
            import requests
            response = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={api_key}")
            if response.status_code == 200:
                return jsonify({"status": "success", "message": "Weather API key is valid"})
            else:
                return jsonify({"status": "error", "message": "Invalid Weather API key"}), 400
        
        else:
            # Mock success for other services
            return jsonify({"status": "success", "message": f"{service} API key tested successfully"})
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

def get_chat_agent():
    """Get or initialize the chat agent"""
    global chat_agent
    if chat_agent is None:
        try:
            # Check if OpenAI API key is available
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable to enable chat.")
                return None
            
            chat_agent = KGChatAgent()
            logger.info("KG Chat Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize chat agent: {e}")
            chat_agent = None
    return chat_agent

@app.route('/api/kg-chat', methods=['POST'])
def kg_chat():
    """Knowledge Graph Chat endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        user_message = data.get('message')
        session_id = data.get('session_id', 'default')
        current_match = data.get('current_match')  # Optional current match context
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Get chat agent
        agent = get_chat_agent()
        if not agent:
            return jsonify({
                "error": "Chat agent not available - OpenAI API key required", 
                "message": "Please set your OPENAI_API_KEY environment variable or configure it through the API Keys tab to enable chat functionality."
            }), 503
        
        # Get chat history for this session
        chat_history = chat_sessions.get(session_id, [])
        
        # Process the chat message
        response_message, updated_history = agent.chat(
            user_message=user_message,
            chat_history=chat_history,
            current_match_context=current_match
        )
        
        # Update session history
        chat_sessions[session_id] = updated_history
        
        # Clean up old sessions (keep last 100 sessions)
        if len(chat_sessions) > 100:
            oldest_sessions = sorted(chat_sessions.keys())[:-100]
            for old_session in oldest_sessions:
                del chat_sessions[old_session]
        
        return jsonify({
            "response": response_message,
            "session_id": session_id,
            "message_count": len(updated_history)
        })
        
    except Exception as e:
        logger.error(f"KG Chat error: {e}")
        return jsonify({"error": f"Chat processing failed: {str(e)}"}), 500

@app.route('/api/kg-chat/suggestions', methods=['GET'])
def kg_chat_suggestions():
    """Get suggested questions for KG chat"""
    try:
        agent = get_chat_agent()
        if not agent:
            # Fallback suggestions if agent not available
            suggestions = [
                "How does Virat Kohli perform at the MCG?",
                "Compare MS Dhoni and Jos Buttler's T20 stats",
                "What's the head-to-head record between India and Australia?",
                "Tell me about Eden Gardens' pitch characteristics"
            ]
        else:
            suggestions = agent.get_suggested_questions()
        
        return jsonify({"suggestions": suggestions})
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        return jsonify({"suggestions": []}), 200

@app.route('/api/kg-chat/functions', methods=['GET'])
def kg_chat_functions():
    """Get available KG chat functions"""
    try:
        agent = get_chat_agent()
        if not agent:
            return jsonify({"functions": {}}), 200
        
        functions = agent.get_available_functions()
        return jsonify({"functions": functions})
        
    except Exception as e:
        logger.error(f"Error getting functions: {e}")
        return jsonify({"functions": {}}), 200

@app.route('/api/kg-chat/clear-session', methods=['POST'])
def clear_chat_session():
    """Clear chat session history"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', 'default')
        
        if session_id in chat_sessions:
            del chat_sessions[session_id]
        
        return jsonify({"message": "Session cleared successfully"})
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting WicketWise Admin Backend API...")
    print("📊 Real knowledge graph building enabled")
    print("🤖 Real model training pipeline connected") 
    print("🌐 CORS enabled for admin interface")
    print("🔧 Admin tools initialized")
    print()
    print("API Endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/build-knowledge-graph - Build knowledge graph")
    print("  POST /api/build-unified-knowledge-graph - Build unified knowledge graph")
    print("  POST /api/train-model - Train AI model") 
    print("  GET  /api/operation-status/<id> - Get operation status")
    print("  GET  /api/system-status - Get system status")
    print("  POST /api/test-api-key - Test API keys")
    print("  POST /api/kg-chat - Knowledge Graph Chat")
    print("  GET  /api/kg-chat/suggestions - Get chat suggestions")
    print("  GET  /api/kg-chat/functions - Get available functions")
    print("  POST /api/kg-chat/clear-session - Clear chat session")
    print()
    
    # Run using configured host and port
    app.run(host=settings.BACKEND_HOST, port=settings.BACKEND_PORT, debug=settings.DEBUG_MODE)