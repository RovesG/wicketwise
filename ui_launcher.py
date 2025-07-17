# Purpose: Streamlit UI launcher for WicketWise cricket analysis tools
# Author: WicketWise Team, Last Modified: 2024-01-15
# ⚠️ DEV UI ONLY - NOT PRODUCTION HARDENED ⚠️

import streamlit as st
import os
import subprocess
import json
import yaml
from datetime import datetime
from pathlib import Path
import sys
import threading
import queue
import time
from typing import Dict, Any, Optional
import pandas as pd # Added for displaying dataframe results

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from config_loader import ConfigLoader
except ImportError:
    # Fallback if config_loader doesn't exist yet
    class ConfigLoader:
        @staticmethod
        def get_available_configs():
            return ["default_config.yaml", "test_config.yaml"]
        
        @staticmethod
        def get_model_checkpoints():
            return ["model_checkpoint_latest.pt", "model_checkpoint_best.pt"]

try:
    from env_manager import get_env_manager
    ENV_MANAGER_AVAILABLE = True
except ImportError:
    ENV_MANAGER_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="WicketWise Dev UI",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'process_logs' not in st.session_state:
    st.session_state.process_logs = {}
if 'process_status' not in st.session_state:
    st.session_state.process_status = {}
if 'api_keys_saved' not in st.session_state:
    st.session_state.api_keys_saved = False

def save_api_keys_to_session(betfair_key: str, openai_key: str) -> None:
    """Save API keys to session state securely using env_manager"""
    if ENV_MANAGER_AVAILABLE:
        try:
            env_manager = get_env_manager()
            env_manager.set_api_key('betfair', betfair_key, persist=False)
            env_manager.set_api_key('openai', openai_key, persist=False)
            st.session_state.api_keys_saved = True
            return
        except Exception as e:
            st.error(f"Failed to save keys with env_manager: {e}")
    
    # Fallback to session state
    st.session_state.betfair_api_key = betfair_key
    st.session_state.openai_api_key = openai_key
    st.session_state.api_keys_saved = True

def get_api_keys_for_subprocess() -> Dict[str, str]:
    """Get API keys for subprocess environment"""
    env_vars = {}
    
    if ENV_MANAGER_AVAILABLE:
        try:
            env_manager = get_env_manager()
            try:
                env_vars['BETFAIR_API_KEY'] = env_manager.get_api_key('betfair')
            except KeyError:
                pass
            try:
                env_vars['OPENAI_API_KEY'] = env_manager.get_api_key('openai')
            except KeyError:
                pass
            return env_vars
        except Exception:
            pass
    
    # Fallback to session state
    if hasattr(st.session_state, 'betfair_api_key'):
        env_vars['BETFAIR_API_KEY'] = st.session_state.betfair_api_key
    if hasattr(st.session_state, 'openai_api_key'):
        env_vars['OPENAI_API_KEY'] = st.session_state.openai_api_key
    
    return env_vars

def create_temp_env_file(betfair_key: str, openai_key: str) -> str:
    """Create temporary .env file for subprocess usage"""
    env_content = f"""# Temporary API keys for WicketWise UI session
BETFAIR_API_KEY={betfair_key}
OPENAI_API_KEY={openai_key}
"""
    temp_env_path = ".env.temp"
    with open(temp_env_path, "w") as f:
        f.write(env_content)
    return temp_env_path

def run_subprocess_safely(command: list, process_name: str) -> Dict[str, Any]:
    """Run subprocess with logging and error handling"""
    try:
        # Create environment with API keys
        env = os.environ.copy()
        api_keys = get_api_keys_for_subprocess()
        env.update(api_keys)
        
        # Run process
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=300  # 5 minute timeout
        )
        
        # Store results
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'command': ' '.join(command),
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
        
        # Update session state
        if process_name not in st.session_state.process_logs:
            st.session_state.process_logs[process_name] = []
        st.session_state.process_logs[process_name].append(log_entry)
        st.session_state.process_status[process_name] = 'completed' if log_entry['success'] else 'failed'
        
        return log_entry
        
    except subprocess.TimeoutExpired:
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'command': ' '.join(command),
            'error': 'Process timed out after 5 minutes',
            'success': False
        }
        st.session_state.process_logs[process_name].append(error_log)
        st.session_state.process_status[process_name] = 'timeout'
        return error_log
        
    except Exception as e:
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'command': ' '.join(command),
            'error': str(e),
            'success': False
        }
        st.session_state.process_logs[process_name].append(error_log)
        st.session_state.process_status[process_name] = 'error'
        return error_log

def display_process_logs(process_name: str) -> None:
    """Display logs for a specific process"""
    if process_name in st.session_state.process_logs:
        logs = st.session_state.process_logs[process_name]
        if logs:
            latest_log = logs[-1]
            
            # Status indicator
            if latest_log.get('success', False):
                st.success(f"✅ {process_name} completed successfully")
            else:
                st.error(f"❌ {process_name} failed")
            
            # Expandable log details
            with st.expander(f"View {process_name} logs"):
                st.text(f"Command: {latest_log.get('command', 'N/A')}")
                st.text(f"Timestamp: {latest_log.get('timestamp', 'N/A')}")
                
                if 'stdout' in latest_log and latest_log['stdout']:
                    st.text("STDOUT:")
                    st.code(latest_log['stdout'])
                
                if 'stderr' in latest_log and latest_log['stderr']:
                    st.text("STDERR:")
                    st.code(latest_log['stderr'])
                
                if 'error' in latest_log:
                    st.text("ERROR:")
                    st.code(latest_log['error'])

def check_api_keys_available() -> bool:
    """Check if API keys are available"""
    if ENV_MANAGER_AVAILABLE:
        try:
            env_manager = get_env_manager()
            availability = env_manager.list_available_keys()
            return availability.get('betfair', False) and availability.get('openai', False)
        except Exception:
            pass
    
    # Fallback check
    return (hasattr(st.session_state, 'betfair_api_key') and 
            hasattr(st.session_state, 'openai_api_key') and
            st.session_state.api_keys_saved)

# Main UI Layout
st.title("🏏 WicketWise Development UI")
st.markdown("---")

# Warning banner
st.warning("⚠️ **DEVELOPMENT UI ONLY** - Not production hardened. Use for testing and development purposes only.")

# Environment manager status
if ENV_MANAGER_AVAILABLE:
    st.info("✅ **Enhanced Security**: Using env_manager for secure API key handling")
else:
    st.warning("⚠️ **Basic Security**: env_manager not available, using session state only")

# Sidebar for API Keys
with st.sidebar:
    st.header("🔑 API Configuration")
    
    # API Key inputs
    betfair_key = st.text_input(
        "Betfair API Key",
        type="password",
        help="Enter your Betfair API key for betting data access"
    )
    
    openai_key = st.text_input(
        "OpenAI API Key", 
        type="password",
        help="Enter your OpenAI API key for AI agent functionality"
    )
    
    # Save keys button
    if st.button("💾 Save API Keys"):
        if betfair_key and openai_key:
            save_api_keys_to_session(betfair_key, openai_key)
            st.success("API keys saved securely!")
        else:
            st.error("Please enter both API keys")
    
    # Status indicator
    if check_api_keys_available():
        st.success("✅ API keys configured")
        
        # Show key validation if env_manager is available
        if ENV_MANAGER_AVAILABLE:
            try:
                env_manager = get_env_manager()
                validation_results = env_manager.validate_all_keys()
                
                with st.expander("🔍 Key Validation"):
                    for service, result in validation_results.items():
                        if result['available']:
                            format_status = "✅" if result['valid_format'] else "⚠️"
                            st.text(f"{format_status} {service}: {result['preview']}")
            except Exception:
                pass
    else:
        st.info("⏳ API keys not configured")
    
    st.markdown("---")
    
    # Configuration options
    st.header("⚙️ Configuration")
    
    config_loader = ConfigLoader()
    
    selected_config = st.selectbox(
        "Training Config",
        options=config_loader.get_available_configs(),
        help="Select configuration file for training"
    )
    
    selected_checkpoint = st.selectbox(
        "Model Checkpoint",
        options=config_loader.get_model_checkpoints(),
        help="Select model checkpoint for evaluation"
    )
    
    # Sample data selection
    sample_files = ["samples/test_match.csv", "samples/evaluation_data.csv"]
    selected_sample = st.selectbox(
        "Sample Data",
        options=sample_files,
        help="Select sample data file for testing"
    )
    
    # Environment management section
    if ENV_MANAGER_AVAILABLE:
        st.markdown("---")
        st.header("🔧 Environment Management")
        
        if st.button("💾 Save Keys to .env"):
            if check_api_keys_available():
                try:
                    env_manager = get_env_manager()
                    env_manager.write_env(backup=True)
                    st.success("Keys saved to .env file!")
                except Exception as e:
                    st.error(f"Failed to save to .env: {e}")
            else:
                st.error("No API keys to save")
        
        if st.button("📁 Load from .env"):
            try:
                env_manager = get_env_manager()
                loaded_keys = env_manager.load_env()
                if loaded_keys:
                    st.success(f"Loaded {len(loaded_keys)} keys from .env")
                    st.session_state.api_keys_saved = True
                else:
                    st.info("No keys found in .env file")
            except Exception as e:
                st.error(f"Failed to load from .env: {e}")

# Main panel
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🚀 Main Functions")
    
    # Data workflow section
    st.subheader("📊 Data Workflow")
    
    # Match alignment
    use_hybrid = st.checkbox("Use Hybrid LLM-Enhanced Alignment", value=True, 
                           help="Use LLM for intelligent column mapping and similarity configuration")
    
    if st.button("🔗 Run Match Alignment", key="align_btn"):
        if not check_api_keys_available():
            st.error("Please configure API keys first")
        else:
            st.session_state.process_status['alignment'] = 'running'
            st.info("🔄 Starting match alignment...")
            
            if use_hybrid:
                # Use hybrid approach
                command = ["python", "-c", f"""
import sys
sys.path.append('.')
from hybrid_match_aligner import hybrid_align_matches

# API key will be automatically detected by hybrid_align_matches
data_path = '/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data'

matches = hybrid_align_matches(
    f'{{data_path}}/nvplay_data_v3.csv',
    f'{{data_path}}/decimal_data_v3.csv',
    None,  # Let hybrid_align_matches auto-detect API key
    'hybrid_aligned_matches.csv'
)

print(f'Found {{len(matches)}} matches using hybrid approach')
"""]
            else:
                # Use traditional approach
                command = ["python", "match_aligner.py", 
                          "nvplay_data_v3.csv", "decimal_data_v3.csv",
                          "--output", "aligned_matches.csv"]
            
            result = run_subprocess_safely(command, "alignment")
            
            if result['success']:
                st.success("Match alignment completed successfully!")
                # Show results
                try:
                    if use_hybrid and Path("hybrid_aligned_matches.csv").exists():
                        matches_df = pd.read_csv("hybrid_aligned_matches.csv")
                        st.info(f"✅ Found {len(matches_df)} matches using hybrid approach")
                        with st.expander("View Matches"):
                            st.dataframe(matches_df.head(10))
                    elif Path("aligned_matches.csv").exists():
                        matches_df = pd.read_csv("aligned_matches.csv")
                        st.info(f"✅ Found {len(matches_df)} matches using traditional approach")
                        with st.expander("View Matches"):
                            st.dataframe(matches_df.head(10))
                except Exception as e:
                    st.warning(f"Could not display results: {e}")
            else:
                st.error("Match alignment failed - check logs below")
            
            st.rerun()
    
    # Match splitting
    if st.button("✂️ Run Match Splitting", key="split_btn"):
        # Check which alignment file exists
        hybrid_file = Path("hybrid_aligned_matches.csv")
        traditional_file = Path("aligned_matches.csv")
        
        if hybrid_file.exists():
            input_file = "hybrid_aligned_matches.csv"
            st.info("🎯 Using hybrid aligned matches for splitting")
        elif traditional_file.exists():
            input_file = "aligned_matches.csv"
            st.info("📊 Using traditional aligned matches for splitting")
        else:
            st.error("❌ No aligned matches found. Please run match alignment first.")
            st.stop()
        
        st.session_state.process_status['splitting'] = 'running'
        st.info("🔄 Starting match splitting...")
        
        # Run splitting
        command = ["python", "-c", f"""
from wicketwise.match_splitter import split_matches
train_matches, val_matches, test_matches = split_matches(
    '{input_file}', 
    '.', 
    0.8, 0.1, 0.1, 
    42, 
    'nvplay_match_id'
)
print(f'Split completed: Train={{len(train_matches)}}, Val={{len(val_matches)}}, Test={{len(test_matches)}}')
"""]
        
        result = run_subprocess_safely(command, "splitting")
        
        if result['success']:
            st.success("Match splitting completed successfully!")
            st.info("✅ Created: train_matches.csv, val_matches.csv, test_matches.csv")
            
            # Show split statistics
            try:
                train_df = pd.read_csv("train_matches.csv")
                val_df = pd.read_csv("val_matches.csv")
                test_df = pd.read_csv("test_matches.csv")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Matches", len(train_df))
                with col2:
                    st.metric("Validation Matches", len(val_df))
                with col3:
                    st.metric("Test Matches", len(test_df))
                    
            except Exception as e:
                st.warning(f"Could not display split statistics: {e}")
        else:
            st.error("Match splitting failed - check logs below")
        
        st.rerun()
    
    # Training section
    st.subheader("📈 Model Training")
    
    # Training options
    use_match_splits = st.checkbox("Use Match-Level Splits", value=True, 
                                  help="Use proper match-level train/val splits (recommended)")
    
    if st.button("▶️ Run Training Pipeline", key="train_btn"):
        if not check_api_keys_available():
            st.error("Please configure API keys first")
        else:
            st.session_state.process_status['training'] = 'running'
            st.info("🔄 Starting training pipeline...")
            
            # Build training command
            command = ["python", "crickformers/train.py", "--config", f"config/{selected_config}"]
            
            if use_match_splits:
                # Check if split files exist
                if Path("train_matches.csv").exists() and Path("val_matches.csv").exists():
                    command.extend(["--train-matches", "train_matches.csv"])
                    command.extend(["--val-matches", "val_matches.csv"])
                    st.info("🎯 Using match-level splits for training")
                else:
                    st.warning("⚠️ Split files not found. Run match splitting first or uncheck match-level splits.")
                    st.session_state.process_status['training'] = 'failed'
                    st.rerun()
                    st.stop()
            else:
                st.warning("⚠️ Using random sample-level splits (may cause data leakage)")
            
            # Run training
            result = run_subprocess_safely(command, "training")
            
            if result['success']:
                st.success("Training completed successfully!")
            else:
                st.error("Training failed - check logs below")
            
            st.rerun()
    
    # Evaluation section
    st.subheader("📊 Model Evaluation")
    if st.button("▶️ Run Model Evaluation", key="eval_btn"):
        if not check_api_keys_available():
            st.error("Please configure API keys first")
        else:
            st.session_state.process_status['evaluation'] = 'running'
            st.info("🔄 Starting model evaluation...")
            
            # Run evaluation
            command = ["python", "eval.py", "--checkpoint", selected_checkpoint, "--data", selected_sample]
            result = run_subprocess_safely(command, "evaluation")
            
            if result['success']:
                st.success("Evaluation completed successfully!")
            else:
                st.error("Evaluation failed - check logs below")
            
            st.rerun()
    
    # Post-match report section
    st.subheader("📋 Post-Match Analysis")
    if st.button("▶️ Generate Post-Match Report", key="report_btn"):
        if not check_api_keys_available():
            st.error("Please configure API keys first")
        else:
            st.session_state.process_status['report'] = 'running'
            st.info("🔄 Generating post-match report...")
            
            # Run post-match report
            command = ["python", "crickformers/post_match_report.py", "--data", selected_sample]
            result = run_subprocess_safely(command, "report")
            
            if result['success']:
                st.success("Report generated successfully!")
            else:
                st.error("Report generation failed - check logs below")
            
            st.rerun()
    
    # Live inference section
    st.subheader("🔴 Live Inference")
    match_file = st.file_uploader(
        "Upload Match File",
        type=['csv'],
        help="Upload a CSV file for live inference"
    )
    
    if st.button("▶️ Run Live Inference", key="inference_btn"):
        if not check_api_keys_available():
            st.error("Please configure API keys first")
        elif not match_file:
            st.error("Please upload a match file first")
        else:
            st.session_state.process_status['inference'] = 'running'
            st.info("🔄 Running live inference...")
            
            # Save uploaded file temporarily
            temp_file = f"temp_match_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(temp_file, "wb") as f:
                f.write(match_file.getvalue())
            
            # Run inference
            command = ["python", "crickformers/live_inference.py", "--data", temp_file, "--checkpoint", selected_checkpoint]
            result = run_subprocess_safely(command, "inference")
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            if result['success']:
                st.success("Live inference completed successfully!")
            else:
                st.error("Live inference failed - check logs below")
            
            st.rerun()

with col2:
    st.header("📊 Process Status")
    
    # Display current process statuses
    for process_name, status in st.session_state.process_status.items():
        if status == 'running':
            st.info(f"🔄 {process_name.title()}: Running...")
        elif status == 'completed':
            st.success(f"✅ {process_name.title()}: Completed")
        elif status == 'failed':
            st.error(f"❌ {process_name.title()}: Failed")
        elif status == 'timeout':
            st.warning(f"⏰ {process_name.title()}: Timeout")
        elif status == 'error':
            st.error(f"🚨 {process_name.title()}: Error")
    
    # Clear logs button
    if st.button("🗑️ Clear All Logs"):
        st.session_state.process_logs = {}
        st.session_state.process_status = {}
        st.success("Logs cleared!")
        st.rerun()

# Process logs section
st.markdown("---")
st.header("📝 Process Logs")

# Display logs for each process
for process_name in ['alignment', 'splitting', 'training', 'evaluation', 'report', 'inference']:
    if process_name in st.session_state.process_logs:
        display_process_logs(process_name)

# Footer
st.markdown("---")
footer_text = "**WicketWise Development UI** | Built with Streamlit"
if ENV_MANAGER_AVAILABLE:
    footer_text += " | 🔒 Secure API Management"
footer_text += " | ⚠️ Development Use Only"
st.markdown(footer_text) 