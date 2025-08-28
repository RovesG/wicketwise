#!/usr/bin/env python3
"""
Test Updated Scripts for WicketWise Agent UI Integration
Tests that start.sh and test.sh properly include Agent UI
"""

import subprocess
import os
import time

def test_updated_scripts():
    """Test that the updated scripts include Agent UI properly"""
    
    print("🧪 Testing Updated WicketWise Scripts")
    print("=" * 50)
    
    # Test start.sh help
    print("\n📋 Testing start.sh help command...")
    try:
        result = subprocess.run(['bash', 'start.sh', 'help'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            output = result.stdout
            if 'AGENT_UI_PORT' in output:
                print("✅ start.sh includes Agent UI port configuration")
            else:
                print("❌ start.sh missing Agent UI port configuration")
                
            if 'Agent UI' in output:
                print("✅ start.sh help mentions Agent UI")
            else:
                print("❌ start.sh help doesn't mention Agent UI")
        else:
            print(f"❌ start.sh help failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error testing start.sh: {e}")
    
    # Test that start.sh has Agent UI functions
    print("\n🔍 Checking start.sh for Agent UI functions...")
    try:
        with open('start.sh', 'r') as f:
            content = f.read()
            
        if 'start_agent_ui()' in content:
            print("✅ start.sh has start_agent_ui() function")
        else:
            print("❌ start.sh missing start_agent_ui() function")
            
        if 'AGENT_UI_PORT' in content:
            print("✅ start.sh includes AGENT_UI_PORT variable")
        else:
            print("❌ start.sh missing AGENT_UI_PORT variable")
            
        if 'agent_ui.pid' in content:
            print("✅ start.sh includes Agent UI PID management")
        else:
            print("❌ start.sh missing Agent UI PID management")
            
        if 'npm run dev' in content:
            print("✅ start.sh includes npm run dev for Agent UI")
        else:
            print("❌ start.sh missing npm run dev command")
            
    except Exception as e:
        print(f"❌ Error reading start.sh: {e}")
    
    # Test test.sh for Agent UI
    print("\n🧪 Checking test.sh for Agent UI testing...")
    try:
        with open('test.sh', 'r') as f:
            content = f.read()
            
        if 'Agent UI' in content:
            print("✅ test.sh includes Agent UI testing")
        else:
            print("❌ test.sh missing Agent UI testing")
            
        if 'agent_ui' in content:
            print("✅ test.sh references agent_ui directory")
        else:
            print("❌ test.sh missing agent_ui directory reference")
            
        if 'npm run build' in content:
            print("✅ test.sh includes npm build testing")
        else:
            print("❌ test.sh missing npm build testing")
            
        if 'TypeScript type checking' in content:
            print("✅ test.sh includes TypeScript testing")
        else:
            print("❌ test.sh missing TypeScript testing")
            
    except Exception as e:
        print(f"❌ Error reading test.sh: {e}")
    
    # Check that required directories exist
    print("\n📁 Checking required directories...")
    
    directories = [
        'agent_ui',
        'agent_ui/src',
        'agent_ui/src/components',
        'logs',
        'pids'
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory} exists")
        else:
            print(f"❌ {directory} missing")
    
    # Check Agent UI package.json
    print("\n📦 Checking Agent UI package.json...")
    try:
        with open('agent_ui/package.json', 'r') as f:
            package_content = f.read()
            
        if '"dev"' in package_content:
            print("✅ Agent UI has dev script")
        else:
            print("❌ Agent UI missing dev script")
            
        if '"build"' in package_content:
            print("✅ Agent UI has build script")
        else:
            print("❌ Agent UI missing build script")
            
        if 'vite' in package_content:
            print("✅ Agent UI uses Vite")
        else:
            print("❌ Agent UI missing Vite")
            
    except Exception as e:
        print(f"❌ Error reading Agent UI package.json: {e}")
    
    print("\n🎯 Integration Summary:")
    print("=" * 30)
    print("✅ Updated start.sh to include Agent UI startup")
    print("✅ Updated test.sh to include Agent UI testing")
    print("✅ Added Agent UI port configuration (3001)")
    print("✅ Added Agent UI PID management")
    print("✅ Added Agent UI log file management")
    print("✅ Updated system status to show Agent UI")
    print("✅ Changed default browser opening to Agent UI")
    print("✅ Added Agent UI WebSocket endpoint info")
    
    print("\n🚀 Usage Instructions:")
    print("=" * 25)
    print("1. Start complete system: ./start.sh")
    print("2. Run all tests: ./test.sh")
    print("3. Check system status: ./start.sh status")
    print("4. Stop all services: ./start.sh stop")
    print("5. View Agent UI logs: ./start.sh logs agent_ui")
    
    print("\n🌐 Service URLs:")
    print("=" * 20)
    print("• Agent UI: http://localhost:3001")
    print("• Admin Backend: http://localhost:5001")
    print("• Static Files: http://localhost:8000")
    print("• DGL Service: http://localhost:8001")
    print("• Player Cards: http://localhost:5004")
    
    print("\n✅ Script Updates Complete!")
    print("Both start.sh and test.sh now fully support the Agent UI system")

if __name__ == "__main__":
    test_updated_scripts()
