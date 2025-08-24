#!/usr/bin/env python3
"""
WicketWise Frontend Status Checker
=================================

Comprehensive checker for all WicketWise frontend components
and their backend connections.

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
    print("🔍 WicketWise Frontend Status Check")
    print("=" * 50)
    
    # Check backend services
    backends = {
        "DGL Backend": {"port": 8001, "url": "http://localhost:8001/healthz"},
        "Flask Backend": {"port": 5001, "url": "http://localhost:5001/api/health"},
        "Static Server": {"port": 8000, "url": "http://localhost:8000"}
    }
    
    print("\n📡 Backend Services:")
    for name, config in backends.items():
        port_active = check_port(config["port"])
        if port_active:
            is_accessible, status = check_url(config["url"])
            status_icon = "✅" if is_accessible else "⚠️"
            print(f"  {status_icon} {name}: Port {config['port']} - {status}")
        else:
            print(f"  ❌ {name}: Port {config['port']} - Not running")
    
    # Check frontend files
    print("\n🎨 Frontend Files:")
    frontend_files = [
        "wicketwise_dashboard.html",
        "wicketwise_admin_simple.html", 
        "wicketwise_agent_dashboard.html",
        "agent_dashboard.js",
        "wicketwise_styles.css"
    ]
    
    for file in frontend_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - Missing")
    
    # Check DGL integration
    print("\n🛡️ DGL Integration:")
    dgl_running = check_port(8001)
    if dgl_running:
        print("  ✅ DGL Service: Running on port 8001")
        
        # Test DGL endpoints
        endpoints = [
            "/healthz",
            "/governance/health", 
            "/rules/health",
            "/audit/health"
        ]
        
        for endpoint in endpoints:
            is_accessible, status = check_url(f"http://localhost:8001{endpoint}")
            icon = "✅" if is_accessible else "⚠️"
            print(f"    {icon} {endpoint}: {status}")
    else:
        print("  ❌ DGL Service: Not running")
        print("    💡 Start with: cd services/dgl && ./start_simple.sh")
    
    # Check Streamlit DGL UI
    print("\n🎛️ DGL UI (Streamlit):")
    streamlit_running = check_port(8501)
    if streamlit_running:
        print("  ✅ Streamlit UI: Running on port 8501")
    else:
        print("  ❌ Streamlit UI: Not running")
        print("    💡 Start with: cd services/dgl && streamlit run ui/streamlit_app.py")
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    if not dgl_running:
        print("  1. Start DGL Backend:")
        print("     cd services/dgl && ./start_simple.sh")
    
    if not check_port(8000):
        print("  2. Start Static Server for HTML Dashboard:")
        print("     python -m http.server 8000")
    
    if not check_port(5001):
        print("  3. Start Flask Backend for Agent Dashboard:")
        print("     python agent_dashboard_backend.py")
    
    if not streamlit_running and dgl_running:
        print("  4. Start DGL Streamlit UI:")
        print("     cd services/dgl && streamlit run ui/streamlit_app.py")
    
    # Quick access URLs
    print("\n🌐 Quick Access URLs:")
    if check_port(8000):
        print("  📊 Main Dashboard: http://localhost:8000/wicketwise_dashboard.html")
        print("  ⚙️ Admin Panel: http://localhost:8000/wicketwise_admin_simple.html")
    
    if dgl_running:
        print("  🛡️ DGL API: http://localhost:8001/docs")
        print("  🔍 DGL Health: http://localhost:8001/healthz")
    
    if streamlit_running:
        print("  🎛️ DGL UI: http://localhost:8501")
    
    if check_port(5001):
        print("  🤖 Agent Dashboard: http://localhost:5001")

if __name__ == "__main__":
    main()
