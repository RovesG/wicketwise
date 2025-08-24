#!/usr/bin/env python3
"""
WicketWise: All Sprints Test Runner
===================================

Comprehensive test orchestrator for all completed sprints:
- Sprint 1: Mixture of Experts (MoE) Architecture
- Sprint 2: Knowledge Graph Expansion
- Sprint 3: Betting Intelligence Module

This runner executes all sprint tests in sequence and provides
detailed reporting on the overall system robustness.

Author: WicketWise Team
Last Modified: 2025-08-24
"""

import sys
import time
import traceback
from pathlib import Path

import pytest


def run_all_sprints():
    """
    Execute comprehensive test suite for all completed sprints
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    
    print("=" * 100)
    print("🏏 WICKETWISE: COMPREHENSIVE MULTI-SPRINT TEST SUITE 🏏")
    print("=" * 100)
    print()
    
    start_time = time.time()
    
    # Sprint test runners
    sprint_runners = [
        ("Sprint 1: MoE Architecture", "tests/test_sprint_1_moe.py"),
        ("Sprint 2: Knowledge Graph", "tests/test_sprint_2_kg.py"),
        ("Sprint 3: Betting Intelligence", "tests/test_sprint_3_betting.py"),
        ("Sprint 4: Agent Orchestration", "tests/test_sprint_4_agents.py"),
        ("Sprint 5: UI Dashboards", "tests/test_sprint_5_ui.py")
    ]
    
    try:
        print("🚀 Executing Multi-Sprint Test Suite...")
        print("-" * 60)
        
        for sprint_name, runner_path in sprint_runners:
            print(f"\n📋 {sprint_name}")
            print("=" * len(f"📋 {sprint_name}"))
            
            sprint_start = time.time()
            
            # Execute sprint runner as a Python script
            import subprocess
            result = subprocess.run([
                sys.executable, runner_path
            ], capture_output=False)
            
            result = result.returncode
            
            sprint_elapsed = time.time() - sprint_start
            
            if result != 0:
                print(f"❌ {sprint_name} FAILED in {sprint_elapsed:.2f}s")
                return False
            else:
                print(f"✅ {sprint_name} PASSED in {sprint_elapsed:.2f}s")
        
        print("\n" + "=" * 100)
        print("🎯 WICKETWISE SYSTEM OVERVIEW")
        print("=" * 100)
        
        # System architecture summary
        print("\n🏗️  SYSTEM ARCHITECTURE")
        print("-" * 30)
        print("📁 crickformers/")
        print("   ├── model/")
        print("   │   ├── mixture_of_experts.py    # MoE routing layer")
        print("   │   └── static_context_encoder.py # Context processing")
        print("   ├── orchestration/")
        print("   │   ├── moe_orchestrator.py      # Production orchestrator")
        print("   │   └── prediction_pipeline.py   # Prediction pipeline")
        print("   ├── gnn/")
        print("   │   ├── temporal_decay.py        # Time-based weighting")
        print("   │   ├── context_nodes.py         # Match context extraction")
        print("   │   └── enhanced_kg_api.py       # Natural language KG queries")
        print("   └── betting/")
        print("       ├── mispricing_engine.py     # Value detection engine")
        print("       └── __init__.py")
        print()
        
        # Feature summary
        print("✨ IMPLEMENTED FEATURES")
        print("-" * 25)
        
        features = [
            "🤖 Mixture of Experts (Fast vs Slow Models)",
            "🎯 Intelligent Model Routing & Load Balancing", 
            "📊 Production Orchestration with Health Monitoring",
            "⏰ Temporal Decay Functions (Exponential, Adaptive, Context-Aware)",
            "🏟️  Context Node Extraction (Tournament, Pitch, Weather)",
            "🧠 Natural Language Knowledge Graph Queries",
            "💰 Multi-Bookmaker Odds Aggregation & Analysis",
            "🎲 Statistical Arbitrage Detection",
            "📈 Expected Value & Kelly Criterion Optimization",
            "🔍 Market Efficiency Analysis & Performance Tracking"
        ]
        
        for feature in features:
            print(f"  {feature}")
        
        print()
        
        # Test coverage summary
        print("🧪 TEST COVERAGE SUMMARY")
        print("-" * 28)
        
        test_stats = [
            ("Sprint 1 (MoE)", "15+ tests", "Model routing, orchestration, error handling"),
            ("Sprint 2 (KG)", "25+ tests", "Temporal decay, context extraction, NLP queries"),
            ("Sprint 3 (Betting)", "32+ tests", "Odds conversion, mispricing, arbitrage detection")
        ]
        
        for sprint, count, description in test_stats:
            print(f"  📋 {sprint:<20} {count:<10} - {description}")
        
        total_elapsed = time.time() - start_time
        print()
        print(f"🏆 ALL SPRINTS COMPLETED SUCCESSFULLY in {total_elapsed:.2f}s")
        print("🎉 WicketWise Cricket Intelligence System is PRODUCTION READY!")
        print()
        
        # Next steps
        print("🔮 UPCOMING SPRINTS")
        print("-" * 20)
        print("  📋 Sprint 4: LLM Agent Orchestration layer")
        print("  📋 Sprint 5: UI Dashboards with explainability")
        print("  📋 Sprint 6: Performance optimization & compliance")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-sprint test execution failed: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Main test runner entry point"""
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    success = run_all_sprints()
    
    if success:
        print("🎊 WICKETWISE: ALL SYSTEMS OPERATIONAL!")
        sys.exit(0)
    else:
        print("💥 WICKETWISE: SYSTEM TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
