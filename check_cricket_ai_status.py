#!/usr/bin/env python3
"""
WicketWise Cricket AI Systems Status Checker
===========================================

Comprehensive checker for all WicketWise cricket AI components:
- Original Cricket Intelligence (betting models, KG, GNN)
- DGL Governance Layer
- Admin panels and training interfaces

Author: WicketWise AI
"""

import requests
import subprocess
import sys
from pathlib import Path
import json
import time

def check_port(port):
    """Check if a port is in use"""
    try:
        result = subprocess.run(['lsof', '-i', f':{port}'], 
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def check_url(url, timeout=5):
    """Check if a URL is accessible"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200, response.status_code
    except requests.exceptions.RequestException as e:
        return False, str(e)

def main():
    print("🏏 WicketWise Cricket AI Systems Status")
    print("=" * 60)
    
    # Cricket AI Systems
    systems = {
        "🏏 Main Cricket Dashboard": {
            "port": 8000, 
            "url": "http://localhost:8000/wicketwise_dashboard.html",
            "description": "Primary cricket intelligence interface with KG, GNN, betting models"
        },
        "🤖 Agent Dashboard Backend": {
            "port": 5001, 
            "url": "http://localhost:5001/api/health",
            "description": "Multi-agent orchestration system"
        },
        "🧠 Enhanced Dashboard API": {
            "port": 5002, 
            "url": "http://localhost:5002/api/health",
            "description": "Cricket Intelligence Engine API"
        },
        "🛡️ DGL Governance Backend": {
            "port": 8001, 
            "url": "http://localhost:8001/healthz",
            "description": "Risk management and governance"
        },
        "🎛️ DGL Streamlit UI": {
            "port": 8501, 
            "url": "http://localhost:8501",
            "description": "Governance dashboard"
        }
    }
    
    print("\n📡 System Status:")
    running_systems = []
    
    for name, config in systems.items():
        port_active = check_port(config["port"])
        if port_active:
            is_accessible, status = check_url(config["url"])
            status_icon = "✅" if is_accessible else "⚠️"
            print(f"  {status_icon} {name}")
            print(f"      Port: {config['port']} | Status: {status}")
            print(f"      {config['description']}")
            if is_accessible:
                running_systems.append((name, config))
        else:
            print(f"  ❌ {name}")
            print(f"      Port: {config['port']} - Not running")
            print(f"      {config['description']}")
        print()
    
    # Check key files for cricket AI work
    print("🎯 Cricket AI Components:")
    ai_files = {
        "Knowledge Graph Builder": "optimized_kg_builder.py",
        "GNN Training": "explain_gnn.py", 
        "Crickformer Training": "crickformers/train.py",
        "Betting Intelligence": "betting_intelligence_system.py",
        "Enhanced Player Cards": "enhanced_player_cards.py",
        "Cricket Intelligence Engine": "cricket_intelligence_engine.py"
    }
    
    for component, file_path in ai_files.items():
        if Path(file_path).exists():
            print(f"  ✅ {component}: {file_path}")
        else:
            print(f"  ❌ {component}: {file_path} - Missing")
    
    # Check data directories
    print("\n📊 Data & Models:")
    data_paths = {
        "Knowledge Graph Cache": "kg_cache/",
        "Player Cache": "player_cache/",
        "Models Directory": "models/",
        "Enriched Data": "enriched_data/",
        "Test Cache": "test_cache/"
    }
    
    for component, path in data_paths.items():
        if Path(path).exists():
            # Count files in directory
            try:
                file_count = len(list(Path(path).rglob('*')))
                print(f"  ✅ {component}: {file_count} files")
            except:
                print(f"  ✅ {component}: Available")
        else:
            print(f"  ❌ {component}: Missing")
    
    # Recommendations based on what's running
    print("\n💡 Quick Access - Running Systems:")
    
    if any("Main Cricket Dashboard" in name for name, _ in running_systems):
        print("  🏏 CRICKET INTELLIGENCE DASHBOARD:")
        print("     http://localhost:8000/wicketwise_dashboard.html")
        print("     → Your original work: KG queries, GNN insights, betting models")
        print()
    
    if any("Enhanced Dashboard" in name for name, _ in running_systems):
        print("  🧠 ENHANCED CRICKET ENGINE:")
        print("     http://localhost:8000/wicketwise_dashboard.html (Intelligence Engine tab)")
        print("     → Advanced player analysis with persona switching")
        print()
    
    if any("Agent Dashboard" in name for name, _ in running_systems):
        print("  🤖 MULTI-AGENT SYSTEM:")
        print("     http://localhost:5001")
        print("     → Agent orchestration and coordination")
        print()
    
    if any("DGL" in name for name, _ in running_systems):
        print("  🛡️ RISK MANAGEMENT:")
        print("     http://localhost:8501 (Streamlit UI)")
        print("     http://localhost:8001/docs (API)")
        print("     → Betting governance and risk controls")
        print()
    
    # Start missing systems
    print("🚀 Start Missing Systems:")
    
    if not check_port(8000):
        print("  1. Start Static Server (for Cricket Dashboard):")
        print("     python -m http.server 8000")
        print()
    
    if not check_port(5002):
        print("  2. Start Enhanced Dashboard API:")
        print("     python enhanced_dashboard_api.py")
        print()
    
    if not check_port(5001):
        print("  3. Start Agent Dashboard:")
        print("     python agent_dashboard_backend.py")
        print()
    
    # Admin panels
    print("⚙️ Admin & Training Interfaces:")
    admin_urls = []
    
    if check_port(8000):
        admin_urls.append("📋 Admin Panel: http://localhost:8000/wicketwise_admin_simple.html")
    
    if admin_urls:
        for url in admin_urls:
            print(f"  {url}")
        print("     → Build KG, Train GNN, Train Models, Run Evaluations")
    else:
        print("  ❌ Admin panels require static server on port 8000")
    
    print("\n" + "=" * 60)
    print("🎯 SUMMARY:")
    print(f"  Running Systems: {len(running_systems)}/5")
    
    if any("Main Cricket Dashboard" in name for name, _ in running_systems):
        print("  ✅ Your original cricket AI work is accessible!")
    else:
        print("  ⚠️  Main cricket dashboard needs static server on port 8000")

if __name__ == "__main__":
    main()
