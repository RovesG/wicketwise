#!/usr/bin/env python3
"""
Test script to demonstrate real knowledge graph building functionality
"""

import sys
import logging
from pathlib import Path
from admin_tools import AdminTools

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the real knowledge graph building functionality"""
    
    print("🚀 WicketWise Knowledge Graph Builder Test")
    print("=" * 50)
    print()
    
    try:
        # Initialize admin tools
        print("📊 Initializing AdminTools...")
        admin_tools = AdminTools()
        
        # Check system status first
        print("🔍 Checking system status...")
        try:
            status = admin_tools.get_system_status()
            print(f"✅ System Status: {status}")
        except Exception as e:
            print(f"⚠️  System status check failed: {e}")
        
        print()
        print("🧠 Starting REAL Knowledge Graph Building...")
        print("-" * 40)
        
        # Build knowledge graph using real implementation
        result = admin_tools.build_knowledge_graph()
        
        print()
        print("📋 Results:")
        print(f"   {result}")
        
        # Check if the graph file was created
        graph_path = Path("models/cricket_knowledge_graph.pkl")
        if graph_path.exists():
            print(f"✅ Knowledge graph saved to: {graph_path}")
            print(f"📁 File size: {graph_path.stat().st_size:,} bytes")
            
            # Try to load and examine the graph
            try:
                import pickle
                with open(graph_path, 'rb') as f:
                    graph = pickle.load(f)
                    
                print(f"🔗 Graph Details:")
                print(f"   - Nodes: {graph.number_of_nodes():,}")
                print(f"   - Edges: {graph.number_of_edges():,}")
                
                # Show node types
                if hasattr(graph, 'nodes'):
                    node_types = {}
                    for node, data in graph.nodes(data=True):
                        node_type = data.get('node_type', 'unknown')
                        node_types[node_type] = node_types.get(node_type, 0) + 1
                    
                    print(f"   - Node Types:")
                    for node_type, count in node_types.items():
                        print(f"     • {node_type}: {count}")
                        
            except Exception as e:
                print(f"⚠️  Could not examine graph details: {e}")
                
        else:
            print("❌ Knowledge graph file not found")
            
        print()
        print("✅ Knowledge Graph Test Complete!")
        
    except Exception as e:
        print(f"❌ Knowledge graph building failed: {e}")
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()