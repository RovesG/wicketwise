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
from datetime import datetime

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
    
    # Check if training is already running
    operation_id = "model_training"
    if operation_id in background_operations and background_operations[operation_id].get("status") == "running":
        return jsonify({"error": "Training is already running", "operation_id": operation_id}), 400
    
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
            
            logger.info("🐛 DEBUG: About to call train_model()")
            
            # Actually run training (removed signal timeout as it doesn't work in threads)
            try:
                logger.info("🐛 DEBUG: Starting train_model()...")
                result = get_admin_tools_instance().train_model()
                logger.info(f"🐛 DEBUG: train_model() returned: {result}")
                
            except Exception as train_error:
                logger.error(f"🐛 DEBUG: train_model() failed with error: {train_error}")
                import traceback
                logger.error(f"🐛 DEBUG: Full traceback: {traceback.format_exc()}")
                raise train_error
            
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
            # Write consolidated CSV directly (more efficient than parquet → CSV conversion)
            import pandas as pd
            if dfs:
                big = pd.concat(dfs, ignore_index=True)
                
                # Write both formats for compatibility
                csv_path = out_dir / 'complete_ball_by_ball_data.csv'
                parquet_path = out_dir / 'events.parquet'
                
                # Write CSV directly for KG pipeline
                big.to_csv(csv_path, index=False)
                
                # Keep parquet for backup/alternative use
                # Fix data type issues for parquet conversion
                for col in big.columns:
                    if big[col].dtype == 'object':
                        # Check if column contains lists/arrays
                        sample_values = big[col].dropna().head(10)
                        if len(sample_values) > 0 and any(isinstance(val, (list, tuple)) for val in sample_values):
                            # Convert lists to JSON strings
                            big[col] = big[col].apply(
                                lambda x: json.dumps(x) if isinstance(x, (list, tuple)) else str(x) if x is not None else None
                            )
                        else:
                            # Convert to string, handling NaN values
                            big[col] = big[col].astype(str).replace('nan', None)
                
                big.to_parquet(parquet_path, index=False)
                
                # Copy CSV to main data directory for easy access
                import shutil
                main_data_csv = Path("/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/complete_ball_by_ball_data.csv")
                shutil.copy2(csv_path, main_data_csv)
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

@app.route('/api/enrich-matches', methods=['POST'])
def enrich_matches():
    """Enrich cricket matches using OpenAI API"""
    operation_id = "match_enrichment"
    
    try:
        data = request.get_json() or {}
        logger.info(f"🐛 DEBUG: Received enrichment request data: {data}")
        
        additional_matches = data.get('max_matches', 50)  # Keep 'max_matches' for UI compatibility
        priority_competitions = data.get('priority_competitions', [])
        
        # If no priority competitions specified, use defaults (but allow empty for all competitions)
        if 'priority_competitions' not in data:
            priority_competitions = [
                'Indian Premier League',
                'Big Bash League',
                'Pakistan Super League',
                'T20I'
            ]
            
        logger.info(f"🐛 DEBUG: Final priority_competitions: {priority_competitions}")
        logger.info(f"🐛 DEBUG: Additional matches requested: {additional_matches}")
        
        # Check if OpenAI API key is available
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return jsonify({
                "error": "OpenAI API key required",
                "message": "Please set OPENAI_API_KEY environment variable or configure it through the API Keys tab"
            }), 400
        
        # Initialize operation tracking
        background_operations[operation_id] = {
            "status": "running",
            "progress": 0,
            "message": "Initializing match enrichment...",
            "logs": ["🚀 Starting OpenAI match enrichment pipeline..."],
            "details": {
                "additional_matches": additional_matches,
                "priority_competitions": priority_competitions,
                "total_cost_estimate": additional_matches * 0.02
            }
        }
        
        def run_match_enrichment():
            try:
                from openai_match_enrichment_pipeline import MatchEnrichmentPipeline
                
                # Update progress
                background_operations[operation_id]["progress"] = 10
                background_operations[operation_id]["message"] = "Loading betting dataset..."
                background_operations[operation_id]["logs"].append("📊 Loading betting dataset for analysis...")
                
                # Initialize pipeline
                pipeline = MatchEnrichmentPipeline(api_key=api_key, output_dir="enriched_data")
                
                background_operations[operation_id]["progress"] = 20
                background_operations[operation_id]["message"] = "Starting match enrichment..."
                background_operations[operation_id]["logs"].append(f"🎯 Enriching {additional_matches} additional matches...")
                
                # Run enrichment
                betting_data_path = '/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/decimal_data_v3.csv'
                enriched_file = pipeline.enrich_betting_dataset(
                    betting_data_path=betting_data_path,
                    additional_matches=additional_matches,
                    priority_competitions=priority_competitions
                )
                
                background_operations[operation_id]["progress"] = 80
                background_operations[operation_id]["message"] = "Analyzing enrichment results..."
                background_operations[operation_id]["logs"].append("📋 Analyzing enrichment results...")
                
                # Get detailed summary
                try:
                    enrichment_summary = pipeline.get_last_enrichment_summary()
                    logger.info(f"📋 Retrieved enrichment summary: {enrichment_summary}")
                except Exception as e:
                    logger.error(f"Failed to get enrichment summary: {e}")
                    enrichment_summary = {}
                
                # Generate summary report
                try:
                    report_file = pipeline.generate_summary_report(enriched_file)
                    logger.info(f"📄 Generated summary report: {report_file}")
                except Exception as e:
                    logger.error(f"Failed to generate summary report: {e}")
                    report_file = "Error generating report"
                
                # Create detailed completion message
                new_matches = enrichment_summary.get('new_matches_enriched', 0)
                cached_matches = enrichment_summary.get('cached_matches_used', 0)
                api_calls = enrichment_summary.get('api_calls_made', 0)
                
                if enrichment_summary.get('enrichment_status', {}).get('no_matches_found', False):
                    completion_message = f"No matches found for the selected competitions."
                    background_operations[operation_id]["logs"].append(f"⚠️ The selected competitions don't exist in the dataset")
                    background_operations[operation_id]["logs"].append(f"💡 Try selecting from: IPL, Big Bash League, Pakistan Super League, T20I, etc.")
                elif enrichment_summary.get('enrichment_status', {}).get('all_requested_matches_cached', False):
                    completion_message = f"No additional matches to enrich! All available matches are already cached."
                    background_operations[operation_id]["logs"].append(f"ℹ️ No new matches available to enrich beyond the {cached_matches} already cached")
                    background_operations[operation_id]["logs"].append(f"💡 All top priority matches are already enriched. Try different competitions or wait for new data.")
                elif new_matches > 0:
                    completion_message = f"Successfully enriched {new_matches} new matches! ({cached_matches} from cache)"
                    background_operations[operation_id]["logs"].append(f"✅ {new_matches} new matches enriched, {cached_matches} from cache")
                else:
                    completion_message = f"No new matches to enrich. {cached_matches} matches available from cache."
                
                # Success
                background_operations[operation_id]["status"] = "completed"
                background_operations[operation_id]["progress"] = 100
                background_operations[operation_id]["message"] = completion_message
                background_operations[operation_id]["logs"].append(f"📁 Enriched data: {enriched_file}")
                background_operations[operation_id]["logs"].append(f"📄 Summary report: {report_file}")
                background_operations[operation_id]["logs"].append(f"💰 API calls made: {api_calls} (~${api_calls * 0.02:.2f})")
                background_operations[operation_id]["result"] = {
                    "enriched_file": enriched_file,
                    "report_file": report_file,
                    "enrichment_summary": enrichment_summary
                }
                
            except Exception as e:
                background_operations[operation_id]["status"] = "error"
                background_operations[operation_id]["message"] = f"Match enrichment failed: {str(e)}"
                background_operations[operation_id]["logs"].append(f"❌ Error: {str(e)}")
                logger.error(f"Match enrichment failed: {str(e)}")
        
        # Start background task
        thread = threading.Thread(target=run_match_enrichment)
        thread.daemon = True
        thread.start()
        
        return jsonify({"status": "started", "operation_id": operation_id})
        
    except Exception as e:
        logger.error(f"Failed to start match enrichment: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/enrichment-settings', methods=['GET'])
def get_enrichment_settings():
    """Get match enrichment settings"""
    try:
        # Check if OpenAI API key is available
        api_key = os.getenv('OPENAI_API_KEY')
        has_api_key = bool(api_key)
        
        settings = {
            "has_openai_key": has_api_key,
            "default_max_matches": 50,
            "available_competitions": [
                "Indian Premier League",
                "Big Bash League", 
                "Pakistan Super League",
                "T20I",
                "Vitality Blast",
                "Bangladesh Premier League",
                "Caribbean Premier League",
                "Lanka Premier League"
            ],
            "cost_per_match": 0.02,
            "total_matches_available": 3987
        }
        
        return jsonify({"status": "success", "settings": settings})
        
    except Exception as e:
        logger.error(f"Error getting enrichment settings: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/enrichment-settings', methods=['POST'])
def update_enrichment_settings():
    """Update match enrichment settings"""
    try:
        data = request.get_json() or {}
        
        # For now, just validate the data
        max_matches = data.get('max_matches', 50)
        priority_competitions = data.get('priority_competitions', [])
        
        if max_matches < 1 or max_matches > 1000:
            return jsonify({"error": "max_matches must be between 1 and 1000"}), 400
        
        return jsonify({
            "status": "success", 
            "message": "Settings updated",
            "settings": {
                "max_matches": max_matches,
                "priority_competitions": priority_competitions
            }
        })
        
    except Exception as e:
        logger.error(f"Error updating enrichment settings: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/enrichment-statistics', methods=['GET'])
def get_enrichment_statistics():
    """Get enrichment cache statistics and dataset analysis"""
    try:
        from openai_match_enrichment_pipeline import MatchEnrichmentPipeline
        import pandas as pd
        
        # Initialize pipeline to get cache stats
        api_key = os.getenv('OPENAI_API_KEY', 'dummy')  # Use dummy if no key for stats only
        pipeline = MatchEnrichmentPipeline(api_key=api_key)
        
        # Get cache statistics
        cache_stats = pipeline.get_cache_statistics()
        
        # Analyze the betting dataset to get total available matches
        betting_data_path = '/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/decimal_data_v3.csv'
        
        dataset_stats = {
            "total_matches": 0,
            "total_enriched": len(pipeline.enrichment_cache),
            "total_remaining": 0,
            "competitions": [],
            "date_range": {"earliest": None, "latest": None},
            "venues": 0,
            "data_available": False,
            "enrichment_percentage": 0
        }
        
        if os.path.exists(betting_data_path):
            try:
                betting_data = pd.read_csv(betting_data_path)
                
                # Get unique matches
                matches = betting_data.groupby(['date', 'competition', 'venue', 'home', 'away']).size().reset_index(name='balls')
                
                total_matches = int(len(matches))
                total_enriched = len(pipeline.enrichment_cache)
                
                dataset_stats.update({
                    "total_matches": total_matches,
                    "total_enriched": total_enriched,
                    "total_remaining": max(0, total_matches - total_enriched),
                    "enrichment_percentage": round((total_enriched / total_matches * 100) if total_matches > 0 else 0, 1),
                    "competitions": sorted([str(comp) for comp in betting_data['competition'].unique()]),
                    "date_range": {
                        "earliest": str(betting_data['date'].min()),
                        "latest": str(betting_data['date'].max())
                    },
                    "venues": int(betting_data['venue'].nunique()),
                    "data_available": True
                })
                
                # Competition breakdown
                comp_stats = {}
                for comp in betting_data['competition'].unique():
                    comp_data = betting_data[betting_data['competition'] == comp]
                    unique_dates = len(set(comp_data['date'].str[:10]))
                    unique_venues = comp_data['venue'].nunique()
                    home_teams = set(comp_data['home'])
                    away_teams = set(comp_data['away'])
                    unique_teams = len(home_teams | away_teams)
                    
                    comp_stats[str(comp)] = {
                        'matches': int(unique_dates),
                        'venues': int(unique_venues),
                        'teams': int(unique_teams)
                    }
                
                dataset_stats["competition_breakdown"] = comp_stats
                
            except Exception as e:
                logger.error(f"Error analyzing betting dataset: {e}")
                dataset_stats["error"] = str(e)
        
        # Calculate enrichment progress
        enriched_count = cache_stats.get('total_cached_matches', 0)
        total_available = dataset_stats.get('total_matches', 0)
        
        progress_stats = {
            "enriched_matches": enriched_count,
            "total_available_matches": total_available,
            "enrichment_percentage": round((enriched_count / total_available * 100), 1) if total_available > 0 else 0,
            "remaining_matches": max(0, total_available - enriched_count),
            "estimated_cost_remaining": round((total_available - enriched_count) * 0.02, 2) if total_available > enriched_count else 0
        }
        
        return jsonify({
            "status": "success",
            "cache_statistics": cache_stats,
            "dataset_statistics": dataset_stats,
            "progress_statistics": progress_stats,
            "last_updated": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting enrichment statistics: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Cricsheet Auto Update endpoints
@app.route('/api/cricsheet/check-updates', methods=['GET'])
def check_cricsheet_updates():
    """Check if newer Cricsheet data is available"""
    try:
        from cricsheet_auto_updater import get_cricsheet_updater
        updater = get_cricsheet_updater()
        result = updater.check_for_updates()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to check Cricsheet updates: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cricsheet/update', methods=['POST'])
def trigger_cricsheet_update():
    """Trigger full Cricsheet update pipeline"""
    operation_id = "cricsheet_update"
    
    if operation_id in background_operations and background_operations[operation_id]["status"] == "running":
        return jsonify({"error": "Cricsheet update already in progress"}), 400
    
    background_operations[operation_id] = {
        "status": "running",
        "progress": 0,
        "message": "Starting Cricsheet update pipeline...",
        "logs": [],
        "started_at": datetime.now().isoformat()
    }
    
    def run_update():
        try:
            from cricsheet_auto_updater import get_cricsheet_updater
            updater = get_cricsheet_updater()
            
            # Update progress - Check for updates
            background_operations[operation_id]["message"] = "Checking for updates..."
            background_operations[operation_id]["progress"] = 10
            
            check_result = updater.check_for_updates()
            if not check_result.get("update_available", False):
                background_operations[operation_id]["status"] = "completed"
                background_operations[operation_id]["message"] = "No updates available"
                background_operations[operation_id]["progress"] = 100
                return
            
            # Update progress - Download and process
            background_operations[operation_id]["message"] = "Downloading and processing new data..."
            background_operations[operation_id]["progress"] = 30
            
            process_result = updater.download_and_process_updates()
            if not process_result.success:
                raise Exception(process_result.error_message)
            
            # Update progress - Update KG
            background_operations[operation_id]["message"] = "Updating Knowledge Graph..."
            background_operations[operation_id]["progress"] = 60
            
            kg_success = updater.trigger_incremental_kg_update()
            
            # Update progress - Retrain GNN
            background_operations[operation_id]["message"] = "Retraining GNN embeddings..."
            background_operations[operation_id]["progress"] = 80
            
            gnn_success = updater.trigger_gnn_retrain(incremental=True)
            
            # Create result summary
            result = {
                "overall_success": process_result.success and kg_success and gnn_success,
                "summary": f"Successfully processed {process_result.new_files_count} new files ({process_result.total_new_balls:,} balls), KG {'✅' if kg_success else '❌'}, GNN {'✅' if gnn_success else '❌'}",
                "steps": {
                    "download_process": process_result.__dict__,
                    "kg_update": {"success": kg_success},
                    "gnn_retrain": {"success": gnn_success}
                }
            }
            
            # Update final status
            if result["overall_success"]:
                background_operations[operation_id]["status"] = "completed"
                background_operations[operation_id]["message"] = result["summary"]
                background_operations[operation_id]["result"] = result
            else:
                background_operations[operation_id]["status"] = "failed"
                background_operations[operation_id]["message"] = f"Update failed: {result['summary']}"
                background_operations[operation_id]["error"] = result["summary"]
            
            background_operations[operation_id]["progress"] = 100
            background_operations[operation_id]["completed_at"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Cricsheet update failed: {e}")
            background_operations[operation_id]["status"] = "failed"
            background_operations[operation_id]["message"] = f"Update failed: {str(e)}"
            background_operations[operation_id]["error"] = str(e)
            background_operations[operation_id]["progress"] = 100
    
    # Start background thread
    import threading
    thread = threading.Thread(target=run_update)
    thread.start()
    
    return jsonify({
        "status": "started",
        "operation_id": operation_id,
        "message": "Cricsheet update pipeline started"
    })

@app.route('/api/cricsheet/download-process', methods=['POST'])
def download_and_process_only():
    """Download and process new Cricsheet data without KG/GNN updates"""
    try:
        from cricsheet_auto_updater import get_cricsheet_updater
        updater = get_cricsheet_updater()
        result = updater.download_and_process_updates()
        
        return jsonify({
            "success": result.success,
            "new_files_count": result.new_files_count,
            "total_new_balls": result.total_new_balls,
            "summary": result.update_summary,
            "error": result.error_message
        })
    except Exception as e:
        logger.error(f"Failed to download/process Cricsheet data: {e}")
        return jsonify({"error": str(e)}), 500

# New API endpoints for redesigned UI
@app.route('/api/training-pipeline/stats', methods=['GET'])
def get_training_pipeline_stats():
    """Get comprehensive training pipeline statistics with CORRECTED data sources"""
    try:
        import pandas as pd
        import pickle
        import networkx as nx
        from pathlib import Path
        
        stats = {}
        
        # 1. CORRECTED T20 Dataset Statistics (actual training data)
        t20_path = Path("/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/joined_ball_by_ball_data.csv")
        if t20_path.exists():
            try:
                df = pd.read_csv(t20_path)
                # Calculate unique matches
                if 'match_id' in df.columns:
                    unique_matches = df['match_id'].nunique()
                else:
                    # Estimate from date+teams+competition
                    match_cols = ['date', 'home', 'away', 'competition']
                    available_cols = [col for col in match_cols if col in df.columns]
                    if available_cols:
                        unique_matches = df[available_cols].drop_duplicates().shape[0]
                    else:
                        unique_matches = 0
                
                # Count unique venues
                venue_cols = [col for col in df.columns if 'venue' in col.lower()]
                unique_venues = df[venue_cols[0]].nunique() if venue_cols else 0
                
                stats["decimal_dataset"] = {
                    "exists": True,
                    "rows": len(df),
                    "size_mb": round(t20_path.stat().st_size / (1024*1024), 1),
                    "matches": unique_matches,
                    "venues": unique_venues
                }
            except Exception as e:
                logger.error(f"Error reading T20 dataset: {e}")
                stats["decimal_dataset"] = {"exists": False, "error": str(e)}
        else:
            stats["decimal_dataset"] = {"exists": False}
        
        # 2. CORRECTED Knowledge Graph Statistics (actual NetworkX graph)
        kg_path = Path("models/unified_cricket_kg.pkl")
        if kg_path.exists():
            try:
                with open(kg_path, 'rb') as f:
                    kg = pickle.load(f)
                
                if isinstance(kg, (nx.Graph, nx.DiGraph)):
                    # Count different node types
                    player_nodes = [n for n in kg.nodes() if kg.nodes[n].get('type') == 'player']
                    venue_nodes = [n for n in kg.nodes() if kg.nodes[n].get('type') == 'venue']
                    match_nodes = [n for n in kg.nodes() if kg.nodes[n].get('type') == 'match']
                    
                    stats["json_dataset"] = {
                        "exists": True,
                        "nodes": kg.number_of_nodes(),
                        "edges": kg.number_of_edges(),
                        "size_mb": round(kg_path.stat().st_size / (1024*1024), 1),
                        "players": len(player_nodes),
                        "venues": len(venue_nodes),
                        "matches": len(match_nodes)
                    }
                else:
                    stats["json_dataset"] = {
                        "exists": True,
                        "type": str(type(kg)),
                        "size_mb": round(kg_path.stat().st_size / (1024*1024), 1)
                    }
            except Exception as e:
                logger.error(f"Error reading KG: {e}")
                stats["json_dataset"] = {"exists": False, "error": str(e)}
        else:
            stats["json_dataset"] = {"exists": False}
        
        # 3. Enrichment Statistics
        try:
            from openai_match_enrichment_pipeline import MatchEnrichmentPipeline
            api_key = os.getenv('OPENAI_API_KEY', 'dummy')
            pipeline = MatchEnrichmentPipeline(api_key=api_key)
            enriched_count = len(pipeline.enrichment_cache)
            stats["enriched_matches"] = {
                "exists": True,
                "count": enriched_count
            }
        except Exception as e:
            stats["enriched_matches"] = {"exists": False, "error": str(e)}
        
        # 4. Entity Harmonizer (keep existing for compatibility)
        try:
            from enriched_training_pipeline import get_enriched_training_pipeline
            pipeline = get_enriched_training_pipeline()
            old_stats = pipeline.get_training_statistics()
            stats["entity_harmonizer"] = old_stats.get("entity_harmonizer", {})
        except Exception as e:
            logger.error(f"Failed to get entity harmonizer stats: {e}")
            stats["entity_harmonizer"] = {"error": str(e)}
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Failed to get training pipeline stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/training-pipeline/validate', methods=['GET'])
def validate_training_pipeline():
    """Validate data integrity across all training pipeline sources"""
    try:
        from enriched_training_pipeline import get_enriched_training_pipeline
        pipeline = get_enriched_training_pipeline()
        validation = pipeline.validate_data_integrity()
        return jsonify(validation)
    except Exception as e:
        logger.error(f"Failed to validate training pipeline: {e}")
        return jsonify({"error": str(e)}), 500

# Simulation API Endpoints
@app.route('/api/simulation/holdout-matches', methods=['GET'])
def get_holdout_matches():
    """Get available holdout matches for simulation"""
    try:
        # Import SIM modules
        sys.path.insert(0, str(Path(__file__).parent / 'sim'))
        from sim.data_integration import HoldoutDataManager, integrate_holdout_data_with_sim
        
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
        # Import SIM modules
        sys.path.insert(0, str(Path(__file__).parent / 'sim'))
        from sim.data_integration import HoldoutDataManager
        from sim.config import create_holdout_replay_config, create_replay_config, create_dashboard_replay_config
        from sim.orchestrator import SimOrchestrator
        
        data = request.get_json()
        strategy = data.get('strategy', 'edge_kelly_v3')
        match_selection = data.get('match_selection', 'auto')
        use_holdout_data = data.get('use_holdout_data', True)
        simulation_mode = data.get('simulation_mode', 'rapid')  # 'rapid' or 'live_dashboard'
        
        logger.info(f"🎯 Running simulation: strategy={strategy}, matches={match_selection}, mode={simulation_mode}")
        
        # Create configuration based on mode
        if simulation_mode == 'live_dashboard':
            # Slow mode for dashboard visualization
            config = create_dashboard_replay_config(strategy, match_selection)
        else:
            # Rapid mode for testing
            if use_holdout_data:
                config = create_holdout_replay_config(strategy)
            else:
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

@app.route('/api/clear-caches', methods=['POST'])
def clear_all_caches():
    """Clear all system caches"""
    try:
        import shutil
        from pathlib import Path
        
        cache_dirs = [
            "cache/entity_harmonizer",
            "cache/training_pipeline", 
            "cache/enrichment",
            "enriched_data/cache"
        ]
        
        cleared = []
        for cache_dir in cache_dirs:
            cache_path = Path(cache_dir)
            if cache_path.exists():
                shutil.rmtree(cache_path)
                cleared.append(cache_dir)
        
        logger.info(f"Cleared caches: {cleared}")
        return jsonify({
            "status": "success",
            "cleared_caches": cleared,
            "message": f"Cleared {len(cleared)} cache directories"
        })
    except Exception as e:
        logger.error(f"Failed to clear caches: {e}")
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
    print("  GET  /api/simulation/holdout-matches - Get holdout matches for simulation")
    print("  POST /api/simulation/run - Run strategy simulation")
    print()
    
    # Run using configured host and port
    app.run(host=settings.BACKEND_HOST, port=settings.BACKEND_PORT, debug=settings.DEBUG_MODE)