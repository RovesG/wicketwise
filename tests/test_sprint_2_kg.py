# Purpose: Sprint 2 comprehensive test runner for Knowledge Graph expansion
# Author: WicketWise Team, Last Modified: 2025-08-23

"""
Comprehensive test runner for Sprint 2: Knowledge Graph Expansion

This module runs all tests related to the KG expansion implementation:
- Temporal decay functions and engine
- Context node system with extractors
- Enhanced KG API with natural language processing
- Integration tests for the complete system

Validates that all components work together seamlessly.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestSprint2KGExpansion:
    """Comprehensive tests for Sprint 2 KG expansion implementation"""
    
    def test_temporal_decay_system(self):
        """Test temporal decay functions and engine"""
        exit_code = pytest.main([
            "tests/gnn/test_temporal_decay.py",
            "-v",
            "--tb=short"
        ])
        assert exit_code == 0, "Temporal decay system tests failed"
    
    def test_context_node_system(self):
        """Test context node extractors and manager"""
        exit_code = pytest.main([
            "tests/gnn/test_context_nodes.py", 
            "-v",
            "--tb=short"
        ])
        assert exit_code == 0, "Context node system tests failed"
    
    def test_enhanced_kg_api(self):
        """Test enhanced KG API with NLP and caching"""
        exit_code = pytest.main([
            "tests/gnn/test_enhanced_kg_api.py",
            "-v", 
            "--tb=short"
        ])
        assert exit_code == 0, "Enhanced KG API tests failed"
    
    def test_kg_integration_compatibility(self):
        """Test that KG expansion doesn't break existing functionality"""
        # Test existing GNN components still work
        existing_gnn_tests = [
            "tests/gnn/test_embedding_fetcher.py",
            "tests/gnn/test_biomechanical_features.py",
            "tests/gnn/test_aggregator.py"
        ]
        
        for test_file in existing_gnn_tests:
            exit_code = pytest.main([test_file, "-v", "--tb=short"])
            assert exit_code == 0, f"Existing GNN test {test_file} failed after KG expansion"
    
    def test_sprint_2_performance_benchmarks(self):
        """Test performance benchmarks for Sprint 2 components"""
        import time
        from crickformers.gnn.temporal_decay import TemporalDecayEngine, PerformanceEvent, DecayType
        from crickformers.gnn.context_nodes import ContextNodeManager
        from crickformers.gnn.enhanced_kg_api import EnhancedKGQueryEngine
        from datetime import datetime, timedelta
        
        # Benchmark temporal decay with large dataset
        engine = TemporalDecayEngine()
        
        # Create 1000 performance events
        events = []
        base_date = datetime(2024, 1, 1)
        for i in range(1000):
            event = PerformanceEvent(
                event_id=f"event_{i}",
                player_id=f"player_{i % 100}",  # 100 unique players
                date=base_date + timedelta(days=i),
                performance_metrics={"runs": 50 + (i % 50), "strike_rate": 120 + (i % 30)},
                context={"format": "t20", "venue": "home"}
            )
            events.append(event)
        
        # Benchmark temporal weight calculation
        start_time = time.time()
        weighted_events = engine.calculate_temporal_weights(events, decay_type=DecayType.EXPONENTIAL)
        temporal_time = time.time() - start_time
        
        assert len(weighted_events) == 1000
        assert temporal_time < 1.0, f"Temporal decay took {temporal_time:.3f}s, should be < 1s"
        
        # Benchmark context node extraction
        context_manager = ContextNodeManager()
        
        match_data = {
            "tournament": "IPL 2024",
            "match_type": "Final",
            "total_runs": 180,
            "total_overs": 20,
            "total_wickets": 10,
            "weather_description": "Clear with dew"
        }
        
        start_time = time.time()
        for _ in range(100):  # Extract context nodes 100 times
            context_nodes = context_manager.extract_all_context_nodes(match_data)
        context_time = time.time() - start_time
        
        assert len(context_nodes) > 0
        assert context_time < 0.5, f"Context extraction took {context_time:.3f}s, should be < 0.5s"
        
        print(f"✅ Performance benchmarks passed:")
        print(f"   📊 Temporal decay (1000 events): {temporal_time:.3f}s")
        print(f"   🏗️  Context extraction (100x): {context_time:.3f}s")
    
    def test_sprint_2_memory_usage(self):
        """Test memory usage of Sprint 2 components"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large temporal decay engine
        from crickformers.gnn.temporal_decay import TemporalDecayEngine, PerformanceEvent
        from datetime import datetime, timedelta
        
        engine = TemporalDecayEngine()
        
        # Create 5000 events
        events = []
        base_date = datetime(2024, 1, 1)
        for i in range(5000):
            event = PerformanceEvent(
                event_id=f"event_{i}",
                player_id=f"player_{i % 200}",
                date=base_date + timedelta(days=i),
                performance_metrics={"runs": 50, "strike_rate": 120},
                context={"format": "t20"}
            )
            events.append(event)
        
        # Process events
        weighted_events = engine.calculate_temporal_weights(events)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use excessive memory
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f}MB, should be < 100MB"
        
        print(f"✅ Memory usage test passed: {memory_increase:.1f}MB increase")


if __name__ == "__main__":
    # Run all Sprint 2 tests
    test_runner = TestSprint2KGExpansion()
    
    print("🚀 Running Sprint 2: Knowledge Graph Expansion Tests")
    print("=" * 60)
    
    try:
        test_runner.test_temporal_decay_system()
        print("✅ Temporal decay system: PASSED")
        
        test_runner.test_context_node_system()
        print("✅ Context node system: PASSED")
        
        test_runner.test_enhanced_kg_api()
        print("✅ Enhanced KG API: PASSED")
        
        test_runner.test_kg_integration_compatibility()
        print("✅ Integration compatibility: PASSED")
        
        test_runner.test_sprint_2_performance_benchmarks()
        print("✅ Performance benchmarks: PASSED")
        
        test_runner.test_sprint_2_memory_usage()
        print("✅ Memory usage: PASSED")
        
        print("=" * 60)
        print("🎉 ALL SPRINT 2 TESTS PASSED!")
        print("📊 Total components tested: 3 (temporal decay, context nodes, enhanced API)")
        print("🧪 Total test cases: 45 + 35 + 37 = 117 tests")
        
    except AssertionError as e:
        print(f"❌ Sprint 2 tests failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error in Sprint 2 tests: {e}")
        sys.exit(1)
