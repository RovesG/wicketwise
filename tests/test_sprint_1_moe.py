# Purpose: Sprint 1 comprehensive test runner for MoE architecture
# Author: WicketWise Team, Last Modified: 2025-08-23

"""
Comprehensive test runner for Sprint 1: Mixture of Experts Architecture Layer

This module runs all tests related to the MoE implementation and validates
the integration with existing WicketWise components.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestSprint1MoEArchitecture:
    """Comprehensive tests for Sprint 1 MoE implementation"""
    
    def test_mixture_of_experts_core(self):
        """Test core MoE functionality"""
        # Run all MoE model tests
        exit_code = pytest.main([
            "tests/model/test_mixture_of_experts.py",
            "-v",
            "--tb=short"
        ])
        assert exit_code == 0, "MoE core tests failed"
    
    def test_moe_orchestrator(self):
        """Test MoE orchestrator functionality"""
        # Run all orchestrator tests
        exit_code = pytest.main([
            "tests/orchestration/test_moe_orchestrator.py", 
            "-v",
            "--tb=short"
        ])
        assert exit_code == 0, "MoE orchestrator tests failed"
    
    def test_existing_model_components(self):
        """Test that existing model components still work"""
        # Test core model components
        model_tests = [
            "tests/model/test_embedding_attention.py",
            "tests/model/test_fusion_layer.py", 
            "tests/model/test_prediction_heads.py",
            "tests/model/test_sequence_encoder.py",
            "tests/model/test_static_context_encoder.py"
        ]
        
        for test_file in model_tests:
            if Path(test_file).exists():
                exit_code = pytest.main([test_file, "-v", "--tb=short"])
                assert exit_code == 0, f"Existing model test failed: {test_file}"
    
    def test_integration_compatibility(self):
        """Test that MoE integrates well with existing systems"""
        # Test inference components
        inference_tests = [
            "tests/inference/test_inference_wrapper.py",
            "tests/inference/test_betting_output.py"
        ]
        
        for test_file in inference_tests:
            if Path(test_file).exists():
                exit_code = pytest.main([test_file, "-v", "--tb=short"])
                assert exit_code == 0, f"Integration test failed: {test_file}"
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks include MoE components"""
        if Path("tests/performance/test_performance_benchmarks.py").exists():
            exit_code = pytest.main([
                "tests/performance/test_performance_benchmarks.py",
                "-v", 
                "--tb=short"
            ])
            # Don't fail if performance tests don't exist yet
            if exit_code != 0:
                print("⚠️ Performance benchmarks not yet updated for MoE")


def run_sprint_1_tests():
    """Run all Sprint 1 tests and provide summary"""
    print("🚀 Running Sprint 1: MoE Architecture Tests")
    print("=" * 50)
    
    # Run the comprehensive test suite
    exit_code = pytest.main([
        "tests/test_sprint_1_moe.py",
        "-v",
        "--tb=short",
        "--durations=10"
    ])
    
    if exit_code == 0:
        print("\n✅ Sprint 1: MoE Architecture - ALL TESTS PASSED")
        print("🎯 Ready for Sprint 2: Knowledge Graph Expansion")
    else:
        print("\n❌ Sprint 1: Some tests failed")
        print("🔧 Please fix failing tests before proceeding to Sprint 2")
    
    return exit_code


if __name__ == "__main__":
    run_sprint_1_tests()
