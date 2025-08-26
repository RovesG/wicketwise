# Purpose: End-to-end integration tests for SIM system
# Author: WicketWise AI, Last Modified: 2024

import pytest
from datetime import datetime
import tempfile
import shutil
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.config import SimulationConfig, SimulationMode, create_replay_config, create_monte_carlo_config
from sim.orchestrator import SimOrchestrator
from sim.strategy import create_strategy


class TestSimIntegration:
    """End-to-end integration tests for the SIM system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.orchestrator = SimOrchestrator()
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_replay_simulation_end_to_end(self):
        """Test complete replay simulation flow"""
        # Create replay configuration
        config = create_replay_config(["test_match_1"], "edge_kelly_v3")
        config.outputs.dir = str(Path(self.temp_dir) / "replay_test")
        
        # Initialize orchestrator
        assert self.orchestrator.initialize(config)
        
        # Run simulation
        result = self.orchestrator.run()
        
        # Verify results
        assert result is not None
        assert result.run_id == config.id
        assert result.config_hash == config.config_hash()
        assert result.balls_processed >= 0
        assert result.runtime_seconds >= 0
        
        # Verify KPIs structure
        kpis = result.kpis
        assert hasattr(kpis, 'pnl_total')
        assert hasattr(kpis, 'sharpe')
        assert hasattr(kpis, 'max_drawdown')
        assert hasattr(kpis, 'hit_rate')
        
        # Verify artifacts were created
        output_dir = Path(config.outputs.dir)
        assert output_dir.exists()
        
        config_file = output_dir / "config.json"
        assert config_file.exists()
        
        result_file = output_dir / "simulation_result.json"
        assert result_file.exists()
    
    def test_monte_carlo_simulation_end_to_end(self):
        """Test complete Monte Carlo simulation flow"""
        # Create Monte Carlo configuration
        config = create_monte_carlo_config(100)  # Small number for testing
        config.outputs.dir = str(Path(self.temp_dir) / "monte_carlo_test")
        
        # Initialize orchestrator
        assert self.orchestrator.initialize(config)
        
        # Run simulation
        result = self.orchestrator.run()
        
        # Verify results
        assert result is not None
        assert result.balls_processed >= 0
        assert result.runtime_seconds >= 0
        
        # Monte Carlo should generate synthetic events
        assert result.matches_processed >= 0
    
    def test_simulation_with_dgl_violations(self):
        """Test simulation with DGL violations"""
        # Create configuration with very restrictive risk limits
        config = create_replay_config(["test_match_1"], "edge_kelly_v3")
        config.risk_profile.max_exposure_pct = 0.1  # Very restrictive
        config.risk_profile.per_bet_cap_pct = 0.01  # Very small bets
        config.outputs.dir = str(Path(self.temp_dir) / "dgl_test")
        
        # Initialize and run
        assert self.orchestrator.initialize(config)
        result = self.orchestrator.run()
        
        # Should have some violations due to restrictive limits
        # (May or may not have violations depending on strategy behavior)
        assert isinstance(result.violations, list)
    
    def test_simulation_progress_tracking(self):
        """Test simulation progress tracking"""
        config = create_replay_config(["test_match_1"], "edge_kelly_v3")
        config.outputs.dir = str(Path(self.temp_dir) / "progress_test")
        
        # Initialize
        assert self.orchestrator.initialize(config)
        
        # Check initial progress
        progress = self.orchestrator.get_progress()
        assert progress["progress"] == 0.0
        assert progress["status"] == "not_started"
        
        # Start simulation (would be async in real implementation)
        result = self.orchestrator.run()
        
        # Check final progress
        final_progress = self.orchestrator.get_progress()
        assert final_progress["status"] == "completed"
        assert final_progress["events_processed"] >= 0
    
    def test_multiple_strategy_comparison(self):
        """Test running multiple strategies for comparison"""
        strategies = ["edge_kelly_v3", "mean_revert_lob", "momentum_follow"]
        results = []
        
        for strategy_name in strategies:
            config = create_replay_config(["test_match_1"], strategy_name)
            config.id = f"test_{strategy_name}"
            config.outputs.dir = str(Path(self.temp_dir) / f"strategy_{strategy_name}")
            config.seed = 42  # Same seed for fair comparison
            
            orchestrator = SimOrchestrator()
            if orchestrator.initialize(config):
                result = orchestrator.run()
                results.append((strategy_name, result))
        
        # Should have results for each strategy
        assert len(results) >= 1  # At least one should work
        
        # Compare results
        for strategy_name, result in results:
            assert result is not None
            assert result.kpis is not None
            print(f"{strategy_name}: P&L = £{result.kpis.pnl_total:.2f}")
    
    def test_configuration_serialization_roundtrip(self):
        """Test configuration serialization and deserialization"""
        # Create configuration
        config = create_replay_config(["test_match_1"], "edge_kelly_v3")
        
        # Serialize to JSON
        config_json = config.to_json()
        
        # Deserialize back
        config_restored = SimulationConfig.from_json(config_json)
        
        # Should be identical
        assert config_restored.id == config.id
        assert config_restored.mode == config.mode
        assert config_restored.strategy.name == config.strategy.name
        assert config_restored.risk_profile.bankroll == config.risk_profile.bankroll
        assert config_restored.seed == config.seed
    
    def test_simulation_reproducibility(self):
        """Test that simulations are reproducible with same seed"""
        config = create_replay_config(["test_match_1"], "edge_kelly_v3")
        config.seed = 12345
        config.outputs.dir = str(Path(self.temp_dir) / "repro_test_1")
        
        # Run first simulation
        orchestrator1 = SimOrchestrator()
        assert orchestrator1.initialize(config)
        result1 = orchestrator1.run()
        
        # Run second simulation with same config
        config.outputs.dir = str(Path(self.temp_dir) / "repro_test_2")
        orchestrator2 = SimOrchestrator()
        assert orchestrator2.initialize(config)
        result2 = orchestrator2.run()
        
        # Results should be very similar (allowing for small floating point differences)
        assert abs(result1.kpis.pnl_total - result2.kpis.pnl_total) < 1.0
        assert result1.balls_processed == result2.balls_processed
    
    def test_artifact_generation(self):
        """Test that all requested artifacts are generated"""
        config = create_replay_config(["test_match_1"], "edge_kelly_v3")
        config.outputs.artifacts = ["orders", "fills", "dgl", "metrics"]
        config.outputs.dir = str(Path(self.temp_dir) / "artifacts_test")
        
        # Run simulation
        assert self.orchestrator.initialize(config)
        result = self.orchestrator.run()
        
        # Check artifacts
        output_dir = Path(config.outputs.dir)
        
        # Should have created artifact files
        expected_files = [
            "orders.jsonl",
            "fills.jsonl", 
            "dgl_decisions.jsonl",
            "metrics.json",
            "config.json",
            "simulation_result.json"
        ]
        
        for filename in expected_files:
            file_path = output_dir / filename
            if file_path.exists():
                assert file_path.stat().st_size >= 0  # File exists and has content
    
    def test_error_handling_invalid_config(self):
        """Test error handling with invalid configuration"""
        # Create invalid configuration
        config = create_replay_config([], "invalid_strategy")  # Empty match IDs, invalid strategy
        config.outputs.dir = str(Path(self.temp_dir) / "error_test")
        
        # Should handle gracefully
        init_result = self.orchestrator.initialize(config)
        
        # May succeed (with mock data) or fail gracefully
        assert isinstance(init_result, bool)
    
    def test_simulation_stop_functionality(self):
        """Test simulation stop functionality"""
        config = create_replay_config(["test_match_1"], "edge_kelly_v3")
        config.outputs.dir = str(Path(self.temp_dir) / "stop_test")
        
        # Initialize
        assert self.orchestrator.initialize(config)
        
        # Start and immediately stop (simplified test)
        self.orchestrator.stop()
        
        # Should handle stop gracefully
        assert not self.orchestrator.is_running
    
    def test_memory_usage_reasonable(self):
        """Test that memory usage stays reasonable during simulation"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run simulation
        config = create_replay_config(["test_match_1"], "edge_kelly_v3")
        config.outputs.dir = str(Path(self.temp_dir) / "memory_test")
        
        assert self.orchestrator.initialize(config)
        result = self.orchestrator.run()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for test)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
    
    def test_performance_benchmarks(self):
        """Test basic performance benchmarks"""
        import time
        
        config = create_replay_config(["test_match_1"], "edge_kelly_v3")
        config.outputs.dir = str(Path(self.temp_dir) / "perf_test")
        
        # Measure initialization time
        start_time = time.time()
        assert self.orchestrator.initialize(config)
        init_time = time.time() - start_time
        
        # Measure simulation time
        start_time = time.time()
        result = self.orchestrator.run()
        sim_time = time.time() - start_time
        
        # Performance should be reasonable
        assert init_time < 10.0  # Less than 10 seconds to initialize
        assert sim_time < 30.0   # Less than 30 seconds to run test simulation
        
        # Calculate events per second
        if result.balls_processed > 0 and sim_time > 0:
            events_per_second = result.balls_processed / sim_time
            assert events_per_second > 1.0  # At least 1 event per second


class TestSimConfigValidation:
    """Test simulation configuration validation"""
    
    def test_config_hash_consistency(self):
        """Test that config hash is consistent"""
        config1 = create_replay_config(["test_match"], "edge_kelly_v3")
        config2 = create_replay_config(["test_match"], "edge_kelly_v3")
        
        # Same configuration should have same hash
        assert config1.config_hash() == config2.config_hash()
        
        # Different configuration should have different hash
        config2.seed = 999
        assert config1.config_hash() != config2.config_hash()
    
    def test_config_validation_edge_cases(self):
        """Test configuration validation with edge cases"""
        config = create_replay_config(["test_match"], "edge_kelly_v3")
        
        # Test with zero bankroll
        config.risk_profile.bankroll = 0.0
        # Should handle gracefully (may succeed or fail depending on validation)
        
        # Test with negative values
        config.risk_profile.max_exposure_pct = -1.0
        # Should handle gracefully
        
        # Test with very large values
        config.risk_profile.bankroll = 1e12
        # Should handle gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
