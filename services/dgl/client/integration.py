# Purpose: DGL integration patterns and utilities
# Author: WicketWise AI, Last Modified: 2024

"""
DGL Integration

Provides integration patterns and utilities for:
- End-to-end workflow testing
- Integration validation
- Performance benchmarking
- Error handling patterns
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import BetProposal, GovernanceDecision, DecisionType, RuleId
from client.dgl_client import DGLClient, DGLClientConfig, DGLServiceUnavailableError
from client.orchestrator_mock import MockOrchestrator, OrchestratorMode, MarketCondition


logger = logging.getLogger(__name__)


class IntegrationTestType(Enum):
    """Types of integration tests"""
    BASIC_CONNECTIVITY = "basic_connectivity"
    PROPOSAL_FLOW = "proposal_flow"
    BATCH_PROCESSING = "batch_processing"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE_LOAD = "performance_load"
    STRESS_TEST = "stress_test"
    END_TO_END = "end_to_end"


@dataclass
class IntegrationTestConfig:
    """Configuration for integration tests"""
    dgl_base_url: str = "http://localhost:8001"
    test_duration_minutes: int = 5
    proposals_per_minute: int = 6
    concurrent_requests: int = 5
    total_stress_requests: int = 50
    enable_detailed_logging: bool = True
    save_results_to_file: bool = False
    results_file_path: str = "integration_test_results.json"


@dataclass
class IntegrationTestResult:
    """Result of an integration test"""
    test_type: IntegrationTestType
    success: bool
    duration_seconds: float
    details: Dict[str, Any]
    errors: List[str]
    timestamp: datetime


class DGLIntegration:
    """
    DGL integration testing and validation framework
    
    Provides comprehensive testing of DGL integration patterns
    including connectivity, performance, and error handling.
    """
    
    def __init__(self, config: Optional[IntegrationTestConfig] = None):
        """
        Initialize DGL integration framework
        
        Args:
            config: Integration test configuration
        """
        self.config = config or IntegrationTestConfig()
        self.test_results: List[IntegrationTestResult] = []
        
        # Configure logging
        if self.config.enable_detailed_logging:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info(f"DGL integration framework initialized for {self.config.dgl_base_url}")
    
    async def run_basic_connectivity_test(self) -> IntegrationTestResult:
        """
        Test basic connectivity to DGL service
        
        Returns:
            Test result with connectivity status
        """
        logger.info("Running basic connectivity test")
        start_time = datetime.now()
        errors = []
        
        try:
            client_config = DGLClientConfig(base_url=self.config.dgl_base_url)
            
            async with DGLClient(client_config) as client:
                # Test health check
                health = await client.health_check()
                
                # Test ping
                ping_result = await client.ping()
                
                # Test basic endpoints
                exposure = await client.get_current_exposure()
                rules_config = await client.get_rules_configuration()
                
                details = {
                    "health_status": health.get("status"),
                    "ping_successful": ping_result,
                    "exposure_bankroll": exposure.bankroll,
                    "rules_loaded": len(rules_config.keys()) if rules_config else 0,
                    "endpoints_tested": ["health", "ping", "exposure", "rules"]
                }
                
                success = (
                    health.get("status") in ["healthy", "degraded"] and
                    ping_result and
                    exposure.bankroll > 0
                )
                
        except Exception as e:
            errors.append(f"Connectivity test failed: {str(e)}")
            details = {"error": str(e)}
            success = False
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = IntegrationTestResult(
            test_type=IntegrationTestType.BASIC_CONNECTIVITY,
            success=success,
            duration_seconds=duration,
            details=details,
            errors=errors,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        logger.info(f"Basic connectivity test: {'PASSED' if success else 'FAILED'} ({duration:.2f}s)")
        
        return result
    
    async def run_proposal_flow_test(self) -> IntegrationTestResult:
        """
        Test end-to-end proposal evaluation flow
        
        Returns:
            Test result with proposal flow validation
        """
        logger.info("Running proposal flow test")
        start_time = datetime.now()
        errors = []
        
        try:
            client_config = DGLClientConfig(base_url=self.config.dgl_base_url)
            
            async with DGLClient(client_config) as client:
                orchestrator = MockOrchestrator(client, OrchestratorMode.BALANCED)
                
                # Test different proposal scenarios
                test_scenarios = [
                    {"name": "conservative", "stake": 200, "odds": 2.0},
                    {"name": "aggressive", "stake": 1000, "odds": 5.0},
                    {"name": "high_stake", "stake": 2000, "odds": 1.8},
                    {"name": "high_odds", "stake": 500, "odds": 8.0}
                ]
                
                scenario_results = []
                
                for scenario in test_scenarios:
                    try:
                        proposal = await orchestrator.generate_proposal(
                            custom_params=scenario
                        )
                        
                        # Test validation
                        validation = await client.validate_proposal(proposal)
                        
                        # Test evaluation
                        decision = await client.evaluate_proposal(proposal)
                        
                        scenario_results.append({
                            "scenario": scenario["name"],
                            "proposal_valid": validation.get("is_valid", False),
                            "decision": decision.decision.value,
                            "processing_time_ms": decision.processing_time_ms,
                            "rules_triggered": len(decision.rule_ids_triggered),
                            "success": True
                        })
                        
                    except Exception as e:
                        scenario_results.append({
                            "scenario": scenario["name"],
                            "success": False,
                            "error": str(e)
                        })
                        errors.append(f"Scenario {scenario['name']} failed: {str(e)}")
                
                # Calculate success metrics
                successful_scenarios = sum(1 for r in scenario_results if r.get("success", False))
                success_rate = (successful_scenarios / len(test_scenarios)) * 100
                
                details = {
                    "scenarios_tested": len(test_scenarios),
                    "successful_scenarios": successful_scenarios,
                    "success_rate_pct": success_rate,
                    "scenario_results": scenario_results,
                    "avg_processing_time_ms": sum(
                        r.get("processing_time_ms", 0) for r in scenario_results if r.get("success")
                    ) / max(successful_scenarios, 1)
                }
                
                success = success_rate >= 75  # 75% success rate required
                
        except Exception as e:
            errors.append(f"Proposal flow test failed: {str(e)}")
            details = {"error": str(e)}
            success = False
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = IntegrationTestResult(
            test_type=IntegrationTestType.PROPOSAL_FLOW,
            success=success,
            duration_seconds=duration,
            details=details,
            errors=errors,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        logger.info(f"Proposal flow test: {'PASSED' if success else 'FAILED'} ({duration:.2f}s)")
        
        return result
    
    async def run_batch_processing_test(self) -> IntegrationTestResult:
        """
        Test batch proposal processing
        
        Returns:
            Test result with batch processing validation
        """
        logger.info("Running batch processing test")
        start_time = datetime.now()
        errors = []
        
        try:
            client_config = DGLClientConfig(base_url=self.config.dgl_base_url)
            
            async with DGLClient(client_config) as client:
                orchestrator = MockOrchestrator(client, OrchestratorMode.BALANCED)
                
                # Generate batch of proposals
                batch_sizes = [5, 10, 20]
                batch_results = []
                
                for batch_size in batch_sizes:
                    try:
                        proposals = []
                        for _ in range(batch_size):
                            proposal = await orchestrator.generate_proposal()
                            proposals.append(proposal)
                        
                        # Submit batch
                        batch_start = datetime.now()
                        batch_response = await client.evaluate_batch_proposals(proposals)
                        batch_duration = (datetime.now() - batch_start).total_seconds() * 1000
                        
                        # Analyze batch results
                        decisions = batch_response.get("decisions", [])
                        summary = batch_response.get("summary", {})
                        
                        batch_results.append({
                            "batch_size": batch_size,
                            "decisions_returned": len(decisions),
                            "processing_time_ms": batch_duration,
                            "avg_time_per_proposal_ms": batch_duration / batch_size,
                            "decision_breakdown": summary.get("decision_breakdown", {}),
                            "success": len(decisions) == batch_size
                        })
                        
                    except Exception as e:
                        batch_results.append({
                            "batch_size": batch_size,
                            "success": False,
                            "error": str(e)
                        })
                        errors.append(f"Batch size {batch_size} failed: {str(e)}")
                
                # Calculate metrics
                successful_batches = sum(1 for r in batch_results if r.get("success", False))
                
                details = {
                    "batch_sizes_tested": batch_sizes,
                    "successful_batches": successful_batches,
                    "batch_results": batch_results,
                    "total_proposals_processed": sum(
                        r.get("decisions_returned", 0) for r in batch_results
                    )
                }
                
                success = successful_batches == len(batch_sizes)
                
        except Exception as e:
            errors.append(f"Batch processing test failed: {str(e)}")
            details = {"error": str(e)}
            success = False
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = IntegrationTestResult(
            test_type=IntegrationTestType.BATCH_PROCESSING,
            success=success,
            duration_seconds=duration,
            details=details,
            errors=errors,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        logger.info(f"Batch processing test: {'PASSED' if success else 'FAILED'} ({duration:.2f}s)")
        
        return result
    
    async def run_error_handling_test(self) -> IntegrationTestResult:
        """
        Test error handling and resilience
        
        Returns:
            Test result with error handling validation
        """
        logger.info("Running error handling test")
        start_time = datetime.now()
        errors = []
        
        try:
            client_config = DGLClientConfig(
                base_url=self.config.dgl_base_url,
                max_retries=2,
                timeout_seconds=5
            )
            
            async with DGLClient(client_config) as client:
                error_scenarios = []
                
                # Test 1: Invalid proposal data
                try:
                    invalid_proposal = BetProposal(
                        market_id="test",
                        match_id="test",
                        side="BACK",
                        selection="Test",
                        odds=-1.0,  # Invalid odds
                        stake=-100.0,  # Invalid stake
                        model_confidence=0.8,
                        fair_odds=2.0,
                        expected_edge_pct=5.0
                    )
                    await client.evaluate_proposal(invalid_proposal)
                    error_scenarios.append({"test": "invalid_proposal", "handled": False})
                except Exception as e:
                    error_scenarios.append({
                        "test": "invalid_proposal", 
                        "handled": True,
                        "error_type": type(e).__name__
                    })
                
                # Test 2: Timeout handling (simulate with very short timeout)
                try:
                    short_timeout_client = DGLClient(DGLClientConfig(
                        base_url=self.config.dgl_base_url,
                        timeout_seconds=0.001  # Very short timeout
                    ))
                    
                    orchestrator = MockOrchestrator(short_timeout_client, OrchestratorMode.BALANCED)
                    proposal = await orchestrator.generate_proposal()
                    await short_timeout_client.evaluate_proposal(proposal)
                    
                    error_scenarios.append({"test": "timeout", "handled": False})
                    await short_timeout_client.close()
                    
                except Exception as e:
                    error_scenarios.append({
                        "test": "timeout",
                        "handled": True,
                        "error_type": type(e).__name__
                    })
                
                # Test 3: Invalid endpoint
                try:
                    response = await client._make_request("GET", "/invalid/endpoint")
                    error_scenarios.append({"test": "invalid_endpoint", "handled": False})
                except Exception as e:
                    error_scenarios.append({
                        "test": "invalid_endpoint",
                        "handled": True,
                        "error_type": type(e).__name__
                    })
                
                # Calculate error handling success
                handled_errors = sum(1 for s in error_scenarios if s.get("handled", False))
                error_handling_rate = (handled_errors / len(error_scenarios)) * 100
                
                details = {
                    "error_scenarios_tested": len(error_scenarios),
                    "errors_properly_handled": handled_errors,
                    "error_handling_rate_pct": error_handling_rate,
                    "scenario_results": error_scenarios
                }
                
                success = error_handling_rate >= 80  # 80% error handling required
                
        except Exception as e:
            errors.append(f"Error handling test failed: {str(e)}")
            details = {"error": str(e)}
            success = False
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = IntegrationTestResult(
            test_type=IntegrationTestType.ERROR_HANDLING,
            success=success,
            duration_seconds=duration,
            details=details,
            errors=errors,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        logger.info(f"Error handling test: {'PASSED' if success else 'FAILED'} ({duration:.2f}s)")
        
        return result
    
    async def run_performance_load_test(self) -> IntegrationTestResult:
        """
        Test performance under load
        
        Returns:
            Test result with performance metrics
        """
        logger.info("Running performance load test")
        start_time = datetime.now()
        errors = []
        
        try:
            client_config = DGLClientConfig(base_url=self.config.dgl_base_url)
            
            async with DGLClient(client_config) as client:
                orchestrator = MockOrchestrator(client, OrchestratorMode.BALANCED)
                
                # Run simulation
                simulation_results = await orchestrator.run_simulation(
                    duration_minutes=self.config.test_duration_minutes,
                    proposals_per_minute=self.config.proposals_per_minute
                )
                
                # Analyze performance
                stats = orchestrator.get_statistics()
                
                # Performance thresholds
                min_approval_rate = 50  # At least 50% approval rate
                max_avg_response_time = 100  # Max 100ms average response time
                min_throughput = self.config.proposals_per_minute * 0.8  # 80% of target throughput
                
                actual_throughput = simulation_results.get("actual_proposals_per_minute", 0)
                
                performance_checks = {
                    "approval_rate_ok": stats["approval_rate_pct"] >= min_approval_rate,
                    "response_time_ok": stats["avg_response_time_ms"] <= max_avg_response_time,
                    "throughput_ok": actual_throughput >= min_throughput,
                    "error_rate_low": stats["errors"] / max(stats["total_proposals"], 1) < 0.05
                }
                
                details = {
                    "simulation_results": simulation_results,
                    "performance_stats": stats,
                    "performance_checks": performance_checks,
                    "thresholds": {
                        "min_approval_rate_pct": min_approval_rate,
                        "max_avg_response_time_ms": max_avg_response_time,
                        "min_throughput_proposals_per_min": min_throughput
                    }
                }
                
                success = all(performance_checks.values())
                
        except Exception as e:
            errors.append(f"Performance load test failed: {str(e)}")
            details = {"error": str(e)}
            success = False
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = IntegrationTestResult(
            test_type=IntegrationTestType.PERFORMANCE_LOAD,
            success=success,
            duration_seconds=duration,
            details=details,
            errors=errors,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        logger.info(f"Performance load test: {'PASSED' if success else 'FAILED'} ({duration:.2f}s)")
        
        return result
    
    async def run_stress_test(self) -> IntegrationTestResult:
        """
        Test system under stress conditions
        
        Returns:
            Test result with stress test metrics
        """
        logger.info("Running stress test")
        start_time = datetime.now()
        errors = []
        
        try:
            client_config = DGLClientConfig(base_url=self.config.dgl_base_url)
            
            async with DGLClient(client_config) as client:
                orchestrator = MockOrchestrator(client, OrchestratorMode.STRESS_TEST)
                
                # Run stress test
                stress_results = await orchestrator.stress_test(
                    concurrent_requests=self.config.concurrent_requests,
                    total_requests=self.config.total_stress_requests
                )
                
                # Analyze stress test results
                success_rate = stress_results.get("success_rate_pct", 0)
                throughput = stress_results.get("throughput_requests_per_second", 0)
                
                # Stress test thresholds
                min_success_rate = 90  # 90% success rate under stress
                min_throughput = 5     # At least 5 requests per second
                
                stress_checks = {
                    "success_rate_ok": success_rate >= min_success_rate,
                    "throughput_ok": throughput >= min_throughput,
                    "no_critical_errors": stress_results.get("failed_requests", 0) < self.config.total_stress_requests * 0.1
                }
                
                details = {
                    "stress_test_results": stress_results,
                    "stress_checks": stress_checks,
                    "thresholds": {
                        "min_success_rate_pct": min_success_rate,
                        "min_throughput_req_per_sec": min_throughput
                    }
                }
                
                success = all(stress_checks.values())
                
        except Exception as e:
            errors.append(f"Stress test failed: {str(e)}")
            details = {"error": str(e)}
            success = False
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = IntegrationTestResult(
            test_type=IntegrationTestType.STRESS_TEST,
            success=success,
            duration_seconds=duration,
            details=details,
            errors=errors,
            timestamp=datetime.now()
        )
        
        self.test_results.append(result)
        logger.info(f"Stress test: {'PASSED' if success else 'FAILED'} ({duration:.2f}s)")
        
        return result
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """
        Run comprehensive integration test suite
        
        Returns:
            Complete test suite results
        """
        logger.info("Starting comprehensive DGL integration test suite")
        suite_start_time = datetime.now()
        
        # Define test sequence
        test_sequence = [
            ("Basic Connectivity", self.run_basic_connectivity_test),
            ("Proposal Flow", self.run_proposal_flow_test),
            ("Batch Processing", self.run_batch_processing_test),
            ("Error Handling", self.run_error_handling_test),
            ("Performance Load", self.run_performance_load_test),
            ("Stress Test", self.run_stress_test)
        ]
        
        suite_results = {
            "start_time": suite_start_time.isoformat(),
            "test_config": {
                "dgl_base_url": self.config.dgl_base_url,
                "test_duration_minutes": self.config.test_duration_minutes,
                "proposals_per_minute": self.config.proposals_per_minute,
                "concurrent_requests": self.config.concurrent_requests
            },
            "test_results": [],
            "summary": {}
        }
        
        # Run each test
        for test_name, test_func in test_sequence:
            logger.info(f"Running {test_name} test...")
            
            try:
                result = await test_func()
                suite_results["test_results"].append({
                    "name": test_name,
                    "type": result.test_type.value,
                    "success": result.success,
                    "duration_seconds": result.duration_seconds,
                    "errors": result.errors,
                    "timestamp": result.timestamp.isoformat()
                })
                
                logger.info(f"{test_name}: {'✅ PASSED' if result.success else '❌ FAILED'}")
                
            except Exception as e:
                logger.error(f"{test_name} test failed with exception: {str(e)}")
                suite_results["test_results"].append({
                    "name": test_name,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Calculate suite summary
        total_tests = len(test_sequence)
        passed_tests = sum(1 for r in suite_results["test_results"] if r.get("success", False))
        success_rate = (passed_tests / total_tests) * 100
        total_duration = (datetime.now() - suite_start_time).total_seconds()
        
        suite_results.update({
            "end_time": datetime.now().isoformat(),
            "total_duration_seconds": total_duration,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate_pct": success_rate,
                "overall_success": success_rate >= 80  # 80% pass rate required
            }
        })
        
        # Save results to file if configured
        if self.config.save_results_to_file:
            try:
                with open(self.config.results_file_path, 'w') as f:
                    json.dump(suite_results, f, indent=2, default=str)
                logger.info(f"Test results saved to {self.config.results_file_path}")
            except Exception as e:
                logger.error(f"Failed to save results: {str(e)}")
        
        logger.info(f"Integration test suite completed: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)")
        
        return suite_results
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all test results"""
        if not self.test_results:
            return {"message": "No tests have been run yet"}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        
        return {
            "total_tests_run": total_tests,
            "tests_passed": passed_tests,
            "tests_failed": total_tests - passed_tests,
            "success_rate_pct": (passed_tests / total_tests) * 100,
            "test_types_covered": list(set(r.test_type.value for r in self.test_results)),
            "total_test_duration_seconds": sum(r.duration_seconds for r in self.test_results),
            "last_test_timestamp": max(r.timestamp for r in self.test_results).isoformat()
        }


# Convenience function for quick integration testing

async def run_quick_integration_test(
    dgl_base_url: str = "http://localhost:8001",
    test_duration_minutes: int = 2
) -> Dict[str, Any]:
    """
    Run a quick integration test with minimal configuration
    
    Args:
        dgl_base_url: DGL service URL
        test_duration_minutes: Test duration
        
    Returns:
        Integration test results
    """
    config = IntegrationTestConfig(
        dgl_base_url=dgl_base_url,
        test_duration_minutes=test_duration_minutes,
        proposals_per_minute=10,
        concurrent_requests=3,
        total_stress_requests=20
    )
    
    integration = DGLIntegration(config)
    return await integration.run_comprehensive_test_suite()
