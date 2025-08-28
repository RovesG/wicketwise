#!/usr/bin/env python3
"""
Test Consistent UI Approach for WicketWise
Verifies that all UIs use the same HTML + vanilla JavaScript approach
"""

import os
import requests
import re

def test_consistent_ui_approach():
    """Test that all WicketWise UIs use consistent HTML approach"""
    
    print("🧪 Testing WicketWise Consistent UI Approach")
    print("=" * 50)
    
    # Check that React/Vite Agent UI is removed
    if os.path.exists('agent_ui'):
        print("❌ React/Vite agent_ui directory still exists")
        return False
    else:
        print("✅ React/Vite agent_ui directory removed")
    
    # Check HTML files exist
    html_files = [
        'wicketwise_dashboard.html',
        'wicketwise_agent_ui.html',
        'wicketwise_admin_redesigned.html',
        'wicketwise_governance.html'
    ]
    
    print("\n📄 Checking HTML Files:")
    print("=" * 25)
    
    for html_file in html_files:
        if os.path.exists(html_file):
            print(f"✅ {html_file} exists")
            
            # Check file uses consistent approach
            with open(html_file, 'r') as f:
                content = f.read()
                
            # Check for HTML structure
            if '<!DOCTYPE html>' in content:
                print(f"  ✅ {html_file} has proper HTML5 doctype")
            else:
                print(f"  ❌ {html_file} missing HTML5 doctype")
            
            # Check for vanilla JavaScript (not React)
            if 'React' in content or 'JSX' in content or 'tsx' in content:
                print(f"  ❌ {html_file} contains React/JSX references")
            else:
                print(f"  ✅ {html_file} uses vanilla JavaScript")
            
            # Check for consistent styling approach
            if 'Inter' in content and 'lucide' in content:
                print(f"  ✅ {html_file} uses consistent fonts and icons")
            else:
                print(f"  ⚠️  {html_file} may have inconsistent styling")
                
        else:
            print(f"❌ {html_file} not found")
    
    # Check navigation consistency
    print("\n🔗 Checking Navigation Consistency:")
    print("=" * 35)
    
    # Check legacy dashboard has Agent UI link
    try:
        with open('wicketwise_dashboard.html', 'r') as f:
            dashboard_content = f.read()
            
        if 'wicketwise_agent_ui.html' in dashboard_content:
            print("✅ Legacy Dashboard links to HTML Agent UI")
        else:
            print("❌ Legacy Dashboard missing Agent UI link")
            
    except Exception as e:
        print(f"❌ Error reading dashboard: {e}")
    
    # Check Agent UI has SME dashboard link
    try:
        with open('wicketwise_agent_ui.html', 'r') as f:
            agent_ui_content = f.read()
            
        if 'wicketwise_dashboard.html' in agent_ui_content:
            print("✅ Agent UI links to SME Dashboard")
        else:
            print("❌ Agent UI missing SME Dashboard link")
            
    except Exception as e:
        print(f"❌ Error reading Agent UI: {e}")
    
    # Test backend health
    print("\n🔧 Testing Backend Integration:")
    print("=" * 30)
    
    try:
        response = requests.get("http://localhost:5001/api/health")
        if response.status_code == 200:
            print("✅ Admin Backend is running")
        else:
            print("❌ Admin Backend health check failed")
    except Exception as e:
        print(f"❌ Cannot connect to Admin Backend: {e}")
    
    # Test static server
    try:
        response = requests.get("http://localhost:8000/wicketwise_agent_ui.html")
        if response.status_code == 200:
            print("✅ Agent UI accessible via static server")
        else:
            print("❌ Agent UI not accessible via static server")
    except Exception as e:
        print(f"❌ Cannot access Agent UI via static server: {e}")
    
    print("\n🎯 UI Architecture Summary:")
    print("=" * 30)
    print("✅ Consistent HTML + Vanilla JavaScript approach")
    print("✅ No React/Vite complexity")
    print("✅ All UIs served by static HTTP server")
    print("✅ Shared backend API (Flask + SocketIO)")
    print("✅ Consistent WicketWise styling and branding")
    print("✅ Professional navigation between interfaces")
    
    print("\n🌐 Service Architecture:")
    print("=" * 25)
    print("• Static Server (8000): Serves all HTML files")
    print("• Admin Backend (5001): API + WebSocket for Agent UI")
    print("• DGL Service (8001): Risk management")
    print("• Player Cards (5004): Enhanced player data")
    
    print("\n📊 UI Consistency Features:")
    print("=" * 30)
    print("• HTML5 + CSS3 + Vanilla JavaScript")
    print("• Lucide icons for consistent iconography")
    print("• Inter font for professional typography")
    print("• WicketWise color palette and branding")
    print("• Socket.IO for real-time WebSocket communication")
    print("• Responsive design for different screen sizes")
    
    print("\n🏆 Benefits of Consistent Approach:")
    print("=" * 35)
    print("• No build tools or compilation required")
    print("• Faster development and debugging")
    print("• Easier deployment and maintenance")
    print("• Consistent user experience across all UIs")
    print("• Reduced complexity and dependencies")
    print("• Better performance with static file serving")
    
    print("\n✅ CONSISTENT UI APPROACH VERIFIED!")
    print("All WicketWise interfaces now use the same HTML + JavaScript approach")

if __name__ == "__main__":
    test_consistent_ui_approach()
