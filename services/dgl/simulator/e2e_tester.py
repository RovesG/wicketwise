# Purpose: End-to-end testing framework for DGL
# Author: WicketWise AI, Last Modified: 2024

"""
End-to-End Tester

Comprehensive end-to-end testing framework for DGL:
- Full workflow validation
- Integration testing across all components
- Production readiness assessment
- Performance benchmarking
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import BetProposal, GovernanceDecision, DecisionType, RuleId
from client.dgl_client import DGLClient, DGLClientConfig
from client.orchestrator_mock import MockOrchestrator, OrchestratorMode
from simulator.shadow_simulator import ShadowSimulator, ShadowMode, ProductionMockEngine
from simulator.scenario_generator import ScenarioGenerator, ScenarioType, ScenarioConfig


logger = logging.getLogger(__name__)


class E2ETestType(Enum):
    """Types of end-to-end tests"""
    BASIC_WORKFLOW = "basic_workflow"
    COMPREHENSIVE_SCENARIOS = "comprehensive_scenarios"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    SHADOW_VALIDATION = "shadow_validation"
    STRESS_ENDURANCE = "stress_endurance"
    PRODUCTION_SIMULATION = "production_simulation"


@dataclass
class E2ETestConfig:
    """Configuration for end-to-end tests"""
    test_type: E2ETestType
    dgl_base_url: str = "http://localhost:8001"
    duration_minutes: int = 10
    proposals_per_minute: int = 6
    include_shadow_testing: bool = True
    include_performance_metrics: bool = True
    save_detailed_results: bool = False
    results_directory: str = "e2e_test_results"


@dataclass
class E2ETestResult:
    """Result of end-to-end test"""
    test_id: str
    test_type: E2ETestType
    config: E2ETestConfig
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    total_proposals: int
    successful_proposals: int
    failed_proposals: int
    governance_decisions: List[GovernanceDecision]
    performance_metrics: Dict[str, Any]
    shadow_results: Optional[Dict[str, Any]] = None
    scenario_results: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class EndToEndTester:
    """
    Comprehensive end-to-end testing framework for DGL
    
    Provides complete workflow testing including proposal generation,
    governance evaluation, shadow testing, and performance validation.
    """
    
    def __init__(self, config: E2ETestConfig):
        """
        Initialize end-to-end tester
        
        Args:
            config: E2E test configuration
        """
        self.config = config
        self.test_results: List[E2ETestResult] = []
        
        # Initialize components
        self.dgl_client_config = DGLClientConfig(base_url=config.dgl_base_url)
        self.scenario_generator = ScenarioGenerator(seed=42)  # Reproducible tests
        
        # Callbacks for custom monitoring
        self.progress_callback: Optional[Callable] = None
        self.decision_callback: Optional[Callable] = None
        
        logger.info(f"E2E tester initialized for {config.dgl_base_url}")
    
    async def run_test(self, test_id: Optional[str] = None) -> E2ETestResult:
        """
        Run end-to-end test based on configuration
        
        Args:
            test_id: Optional test identifier
            
        Returns:
            E2E test results
        """
        test_id = test_id or f"e2e_{self.config.test_type.value}_{int(time.time())}"
        
        logger.info(f"Starting E2E test {test_id} - {self.config.test_type.value}")
        
        if self.config.test_type == E2ETestType.BASIC_WORKFLOW:
            return await self._run_basic_workflow_test(test_id)
        elif self.config.test_type == E2ETestType.COMPREHENSIVE_SCENARIOS:
            return await self._run_comprehensive_scenarios_test(test_id)
        elif self.config.test_type == E2ETestType.PERFORMANCE_BENCHMARK:
            return await self._run_performance_benchmark_test(test_id)
        elif self.config.test_type == E2ETestType.SHADOW_VALIDATION:
            return await self._run_shadow_validation_test(test_id)
        elif self.config.test_type == E2ETestType.STRESS_ENDURANCE:
            return await self._run_stress_endurance_test(test_id)
        elif self.config.test_type == E2ETestType.PRODUCTION_SIMULATION:
            return await self._run_production_simulation_test(test_id)
        else:
            raise ValueError(f"Unknown test type: {self.config.test_type}")
    
    async def _run_basic_workflow_test(self, test_id: str) -> E2ETestResult:
        """Run basic workflow validation test"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            async with DGLClient(self.dgl_client_config) as dgl_client:
                # Test 1: Service connectivity
                try:
                    health = await dgl_client.health_check()
                    if health.get("status") not in ["healthy", "degraded"]:
                        warnings.append(f"Service health status: {health.get('status')}")
                except Exception as e:
                    errors.append(f"Health check failed: {str(e)}")
                
                # Test 2: Basic proposal evaluation
                orchestrator = MockOrchestrator(dgl_client, OrchestratorMode.BALANCED)
                
                test_proposals = []
                governance_decisions = []
                
                # Generate and evaluate test proposals
                for i in range(10):  # Small set for basic test
                    try:
                        proposal = await orchestrator.generate_proposal()
                        test_proposals.append(proposal)
                        
                        decision = await dgl_client.evaluate_proposal(proposal)
                        governance_decisions.append(decision)
                        
                        if self.decision_callback:
                            await self.decision_callback(proposal, decision)
                        
                    except Exception as e:
                        errors.append(f"Proposal {i} evaluation failed: {str(e)}")
                
                # Test 3: Batch evaluation
                try:
                    if len(test_proposals) >= 5:
                        batch_result = await dgl_client.evaluate_batch_proposals(test_proposals[:5])
                        if len(batch_result.get("decisions", [])) != 5:
                            warnings.append("Batch evaluation returned unexpected number of decisions")
                except Exception as e:
                    errors.append(f"Batch evaluation failed: {str(e)}")
                
                # Test 4: Exposure monitoring
                try:
                    exposure = await dgl_client.get_current_exposure()
                    if exposure.bankroll <= 0:
                        warnings.append("Invalid bankroll in exposure data")
                except Exception as e:
                    errors.append(f"Exposure monitoring failed: {str(e)}")
                
                # Test 5: Rules configuration
                try:
                    rules_config = await dgl_client.get_rules_configuration()
                    if not rules_config:
                        warnings.append("Empty rules configuration")
                except Exception as e:
                    errors.append(f"Rules configuration failed: {str(e)}")
                
                # Calculate metrics
                successful_proposals = len(governance_decisions)
                failed_proposals = len(test_proposals) - successful_proposals
                
                performance_metrics = {
                    "avg_response_time_ms": sum(d.processing_time_ms for d in governance_decisions) / max(len(governance_decisions), 1),
                    "approval_rate_pct": sum(1 for d in governance_decisions if d.decision == DecisionType.APPROVE) / max(len(governance_decisions), 1) * 100,
                    "error_rate_pct": (len(errors) / max(len(test_proposals), 1)) * 100
                }
                
        except Exception as e:
            errors.append(f"Basic workflow test failed: {str(e)}")
            test_proposals = []
            governance_decisions = []
            performance_metrics = {}
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Create summary
        summary = {
            "workflow_status": "PASSED" if len(errors) == 0 else "FAILED",
            "components_tested": ["connectivity", "proposal_evaluation", "batch_processing", "exposure_monitoring", "rules_config"],
            "critical_errors": len(errors),
            "warnings": len(warnings),
            "recommendation": "Basic workflow operational" if len(errors) == 0 else "Issues require investigation"
        }
        
        result = E2ETestResult(
            test_id=test_id,
            test_type=E2ETestType.BASIC_WORKFLOW,
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            total_proposals=len(test_proposals),
            successful_proposals=len(governance_decisions),
            failed_proposals=len(test_proposals) - len(governance_decisions),
            governance_decisions=governance_decisions,
            performance_metrics=performance_metrics,
            errors=errors,
            warnings=warnings,
            summary=summary
        )
        
        self.test_results.append(result)
        return result
    
    async def _run_comprehensive_scenarios_test(self, test_id: str) -> E2ETestResult:
        """Run comprehensive scenario testing"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            async with DGLClient(self.dgl_client_config) as dgl_client:
                orchestrator = MockOrchestrator(dgl_client, OrchestratorMode.BALANCED)
                
                # Test multiple scenario types
                scenario_types = [
                    ScenarioType.NORMAL_OPERATIONS,
                    ScenarioType.EDGE_CASES,
                    ScenarioType.STRESS_CONDITIONS,
                    ScenarioType.MARKET_VOLATILITY,
                    ScenarioType.RISK_LIMITS
                ]
                
                all_proposals = []
                all_decisions = []
                scenario_results = {}
                
                for scenario_type in scenario_types:
                    try:
                        # Generate scenario proposals
                        config = ScenarioConfig(
                            scenario_type=scenario_type,
                            num_proposals=20  # 20 per scenario
                        )
                        
                        proposals = await self.scenario_generator.generate_scenario(config)
                        
                        # Evaluate proposals
                        scenario_decisions = []
                        scenario_errors = []
                        
                        for proposal in proposals:
                            try:
                                decision = await dgl_client.evaluate_proposal(proposal)
                                scenario_decisions.append(decision)
                                all_decisions.append(decision)
                                
                            except Exception as e:
                                scenario_errors.append(str(e))
                        
                        all_proposals.extend(proposals)
                        
                        # Analyze scenario results
                        scenario_results[scenario_type.value] = {
                            "proposals_generated": len(proposals),
                            "successful_evaluations": len(scenario_decisions),
                            "errors": len(scenario_errors),
                            "approval_rate_pct": sum(1 for d in scenario_decisions if d.decision == DecisionType.APPROVE) / max(len(scenario_decisions), 1) * 100,
                            "avg_processing_time_ms": sum(d.processing_time_ms for d in scenario_decisions) / max(len(scenario_decisions), 1),
                            "error_details": scenario_errors[:5]  # First 5 errors
                        }
                        
                        logger.info(f"Completed {scenario_type.value}: {len(scenario_decisions)}/{len(proposals)} successful")
                        
                    except Exception as e:
                        errors.append(f"Scenario {scenario_type.value} failed: {str(e)}")
                        scenario_results[scenario_type.value] = {"error": str(e)}
                
                # Calculate overall metrics
                performance_metrics = {
                    "total_scenarios_tested": len(scenario_types),
                    "successful_scenarios": len([r for r in scenario_results.values() if "error" not in r]),
                    "overall_approval_rate_pct": sum(1 for d in all_decisions if d.decision == DecisionType.APPROVE) / max(len(all_decisions), 1) * 100,
                    "overall_avg_processing_time_ms": sum(d.processing_time_ms for d in all_decisions) / max(len(all_decisions), 1),
                    "scenario_breakdown": scenario_results
                }
                
        except Exception as e:
            errors.append(f"Comprehensive scenarios test failed: {str(e)}")
            all_proposals = []
            all_decisions = []
            scenario_results = {}
            performance_metrics = {}
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Create summary
        successful_scenarios = len([r for r in scenario_results.values() if "error" not in r])
        total_scenarios = len(scenario_results)
        
        summary = {
            "scenarios_status": "PASSED" if successful_scenarios == total_scenarios else "PARTIAL",
            "scenarios_passed": successful_scenarios,
            "scenarios_total": total_scenarios,
            "coverage_pct": (successful_scenarios / max(total_scenarios, 1)) * 100,
            "recommendation": f"Scenario coverage: {successful_scenarios}/{total_scenarios}"
        }
        
        result = E2ETestResult(
            test_id=test_id,
            test_type=E2ETestType.COMPREHENSIVE_SCENARIOS,
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            total_proposals=len(all_proposals),
            successful_proposals=len(all_decisions),
            failed_proposals=len(all_proposals) - len(all_decisions),
            governance_decisions=all_decisions,
            performance_metrics=performance_metrics,
            scenario_results=scenario_results,
            errors=errors,
            warnings=warnings,
            summary=summary
        )
        
        self.test_results.append(result)
        return result
    
    async def _run_performance_benchmark_test(self, test_id: str) -> E2ETestResult:
        """Run performance benchmark test"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            async with DGLClient(self.dgl_client_config) as dgl_client:
                orchestrator = MockOrchestrator(dgl_client, OrchestratorMode.BALANCED)
                
                # Performance test parameters
                test_duration = self.config.duration_minutes
                target_rate = self.config.proposals_per_minute
                
                # Run continuous performance test
                simulation_result = await orchestrator.run_simulation(
                    duration_minutes=test_duration,
                    proposals_per_minute=target_rate,
                    progress_callback=self.progress_callback
                )
                
                # Extract results
                all_decisions = simulation_result.get("decisions", [])
                orchestrator_stats = orchestrator.get_statistics()
                
                # Performance analysis
                actual_rate = simulation_result.get("actual_proposals_per_minute", 0)
                rate_achievement = (actual_rate / target_rate) * 100 if target_rate > 0 else 0
                
                # Response time analysis
                processing_times = [d.get("processing_time_ms", 0) for d in all_decisions]
                if processing_times:
                    avg_time = sum(processing_times) / len(processing_times)
                    p95_time = sorted(processing_times)[int(len(processing_times) * 0.95)] if len(processing_times) > 20 else max(processing_times)
                    p99_time = sorted(processing_times)[int(len(processing_times) * 0.99)] if len(processing_times) > 100 else max(processing_times)
                else:
                    avg_time = p95_time = p99_time = 0
                
                # Performance thresholds
                performance_checks = {
                    "rate_achievement_ok": rate_achievement >= 90,  # 90% of target rate
                    "avg_response_time_ok": avg_time <= 50,  # 50ms average
                    "p95_response_time_ok": p95_time <= 100,  # 100ms P95
                    "p99_response_time_ok": p99_time <= 200,  # 200ms P99
                    "error_rate_ok": orchestrator_stats.get("errors", 0) / max(orchestrator_stats.get("total_proposals", 1), 1) < 0.01  # <1% error rate
                }
                
                performance_metrics = {
                    "target_rate_per_minute": target_rate,
                    "actual_rate_per_minute": actual_rate,
                    "rate_achievement_pct": rate_achievement,
                    "avg_response_time_ms": avg_time,
                    "p95_response_time_ms": p95_time,
                    "p99_response_time_ms": p99_time,
                    "total_requests": orchestrator_stats.get("total_proposals", 0),
                    "error_count": orchestrator_stats.get("errors", 0),
                    "error_rate_pct": (orchestrator_stats.get("errors", 0) / max(orchestrator_stats.get("total_proposals", 1), 1)) * 100,
                    "performance_checks": performance_checks,
                    "orchestrator_stats": orchestrator_stats
                }
                
                # Check for performance issues
                if not performance_checks["rate_achievement_ok"]:
                    warnings.append(f"Rate achievement below target: {rate_achievement:.1f}%")
                
                if not performance_checks["avg_response_time_ok"]:
                    warnings.append(f"Average response time high: {avg_time:.1f}ms")
                
                if not performance_checks["error_rate_ok"]:
                    warnings.append(f"Error rate high: {performance_metrics['error_rate_pct']:.2f}%")
                
        except Exception as e:
            errors.append(f"Performance benchmark test failed: {str(e)}")
            performance_metrics = {}
            all_decisions = []
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Create summary
        performance_score = sum(performance_checks.values()) / len(performance_checks) * 100 if 'performance_checks' in locals() else 0
        
        summary = {
            "performance_status": "EXCELLENT" if performance_score >= 90 else "GOOD" if performance_score >= 80 else "NEEDS_IMPROVEMENT",
            "performance_score_pct": performance_score,
            "throughput_achieved": rate_achievement if 'rate_achievement' in locals() else 0,
            "recommendation": f"Performance score: {performance_score:.1f}%"
        }
        
        result = E2ETestResult(
            test_id=test_id,
            test_type=E2ETestType.PERFORMANCE_BENCHMARK,
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            total_proposals=len(all_decisions) if 'all_decisions' in locals() else 0,
            successful_proposals=len(all_decisions) if 'all_decisions' in locals() else 0,
            failed_proposals=0,
            governance_decisions=[],  # Not storing individual decisions for performance test
            performance_metrics=performance_metrics,
            errors=errors,
            warnings=warnings,
            summary=summary
        )
        
        self.test_results.append(result)
        return result
    
    async def _run_shadow_validation_test(self, test_id: str) -> E2ETestResult:
        """Run shadow mode validation test"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            async with DGLClient(self.dgl_client_config) as dgl_client:
                # Create production mock and shadow simulator
                production_mock = ProductionMockEngine(
                    approval_bias=0.75,  # 75% approval rate
                    conservative_factor=1.1,  # Slightly conservative
                    processing_delay_ms=12.0
                )
                
                shadow_simulator = ShadowSimulator(dgl_client, production_mock)
                
                # Run shadow comparison test
                shadow_result = await shadow_simulator.run_continuous_shadow_test(
                    mode=ShadowMode.COMPARE_DECISIONS,
                    duration_minutes=self.config.duration_minutes,
                    proposals_per_minute=self.config.proposals_per_minute,
                    orchestrator_mode=OrchestratorMode.BALANCED
                )
                
                # Extract shadow test results
                shadow_decisions = shadow_result.shadow_decisions
                production_decisions = shadow_result.production_decisions
                comparisons = shadow_result.comparisons
                
                # Analyze shadow performance
                agreement_rate = shadow_result.metrics.get("agreement_rate_pct", 0)
                shadow_faster = sum(1 for c in comparisons if c.processing_time_delta_ms < 0) / max(len(comparisons), 1) * 100
                
                shadow_results = {
                    "agreement_rate_pct": agreement_rate,
                    "total_comparisons": len(comparisons),
                    "shadow_faster_pct": shadow_faster,
                    "avg_confidence_delta": shadow_result.metrics.get("avg_confidence_delta", 0),
                    "shadow_more_conservative_count": shadow_result.metrics.get("shadow_more_conservative_count", 0),
                    "shadow_more_aggressive_count": shadow_result.metrics.get("shadow_more_aggressive_count", 0),
                    "overall_status": shadow_result.summary.get("overall_status"),
                    "recommendation": shadow_result.summary.get("recommendation")
                }
                
                # Performance comparison
                performance_metrics = {
                    "shadow_avg_time_ms": shadow_result.metrics.get("shadow_metrics", {}).get("avg_processing_time_ms", 0),
                    "production_avg_time_ms": shadow_result.metrics.get("production_metrics", {}).get("avg_processing_time_ms", 0),
                    "shadow_approval_rate_pct": shadow_result.metrics.get("shadow_metrics", {}).get("approval_rate_pct", 0),
                    "production_approval_rate_pct": shadow_result.metrics.get("production_metrics", {}).get("approval_rate_pct", 0)
                }
                
                # Check shadow validation criteria
                if agreement_rate < 80:
                    warnings.append(f"Low agreement rate with production: {agreement_rate:.1f}%")
                
                if shadow_faster < 50:
                    warnings.append(f"Shadow not consistently faster: {shadow_faster:.1f}%")
                
        except Exception as e:
            errors.append(f"Shadow validation test failed: {str(e)}")
            shadow_decisions = []
            shadow_results = {}
            performance_metrics = {}
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Create summary
        validation_score = agreement_rate if 'agreement_rate' in locals() else 0
        
        summary = {
            "shadow_status": "READY" if validation_score >= 85 else "NEEDS_REVIEW" if validation_score >= 70 else "NOT_READY",
            "validation_score_pct": validation_score,
            "production_readiness": validation_score >= 85,
            "recommendation": f"Shadow validation score: {validation_score:.1f}%"
        }
        
        result = E2ETestResult(
            test_id=test_id,
            test_type=E2ETestType.SHADOW_VALIDATION,
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            total_proposals=len(shadow_decisions) if 'shadow_decisions' in locals() else 0,
            successful_proposals=len(shadow_decisions) if 'shadow_decisions' in locals() else 0,
            failed_proposals=0,
            governance_decisions=shadow_decisions if 'shadow_decisions' in locals() else [],
            performance_metrics=performance_metrics,
            shadow_results=shadow_results if 'shadow_results' in locals() else None,
            errors=errors,
            warnings=warnings,
            summary=summary
        )
        
        self.test_results.append(result)
        return result
    
    async def _run_stress_endurance_test(self, test_id: str) -> E2ETestResult:
        """Run stress and endurance test"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            async with DGLClient(self.dgl_client_config) as dgl_client:
                orchestrator = MockOrchestrator(dgl_client, OrchestratorMode.STRESS_TEST)
                
                # Extended stress test
                extended_duration = max(self.config.duration_minutes, 15)  # Minimum 15 minutes
                high_rate = self.config.proposals_per_minute * 2  # Double the normal rate
                
                # Run stress simulation
                stress_result = await orchestrator.run_simulation(
                    duration_minutes=extended_duration,
                    proposals_per_minute=high_rate,
                    progress_callback=self.progress_callback
                )
                
                # Run concurrent stress test
                concurrent_result = await orchestrator.stress_test(
                    concurrent_requests=20,  # High concurrency
                    total_requests=200
                )
                
                # Analyze stress test results
                stress_stats = orchestrator.get_statistics()
                
                performance_metrics = {
                    "stress_simulation": {
                        "duration_minutes": extended_duration,
                        "target_rate": high_rate,
                        "actual_rate": stress_result.get("actual_proposals_per_minute", 0),
                        "total_proposals": stress_stats.get("total_proposals", 0),
                        "error_rate_pct": (stress_stats.get("errors", 0) / max(stress_stats.get("total_proposals", 1), 1)) * 100
                    },
                    "concurrent_test": {
                        "concurrent_requests": 20,
                        "success_rate_pct": concurrent_result.get("success_rate_pct", 0),
                        "throughput_req_per_sec": concurrent_result.get("throughput_requests_per_second", 0),
                        "total_requests": concurrent_result.get("total_requests", 0)
                    }
                }
                
                # Check stress criteria
                stress_checks = {
                    "simulation_error_rate_ok": performance_metrics["stress_simulation"]["error_rate_pct"] < 5.0,
                    "concurrent_success_rate_ok": performance_metrics["concurrent_test"]["success_rate_pct"] >= 90,
                    "throughput_adequate": performance_metrics["concurrent_test"]["throughput_req_per_sec"] >= 5.0
                }
                
                performance_metrics["stress_checks"] = stress_checks
                
                # Check for stress issues
                if not stress_checks["simulation_error_rate_ok"]:
                    warnings.append(f"High error rate under stress: {performance_metrics['stress_simulation']['error_rate_pct']:.1f}%")
                
                if not stress_checks["concurrent_success_rate_ok"]:
                    warnings.append(f"Low concurrent success rate: {performance_metrics['concurrent_test']['success_rate_pct']:.1f}%")
                
        except Exception as e:
            errors.append(f"Stress endurance test failed: {str(e)}")
            performance_metrics = {}
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Create summary
        stress_score = sum(stress_checks.values()) / len(stress_checks) * 100 if 'stress_checks' in locals() else 0
        
        summary = {
            "stress_status": "EXCELLENT" if stress_score >= 90 else "GOOD" if stress_score >= 75 else "NEEDS_IMPROVEMENT",
            "stress_score_pct": stress_score,
            "endurance_validated": duration >= 900,  # 15+ minutes
            "recommendation": f"Stress test score: {stress_score:.1f}%"
        }
        
        result = E2ETestResult(
            test_id=test_id,
            test_type=E2ETestType.STRESS_ENDURANCE,
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            total_proposals=performance_metrics.get("stress_simulation", {}).get("total_proposals", 0) if 'performance_metrics' in locals() else 0,
            successful_proposals=0,  # Calculated differently for stress test
            failed_proposals=0,
            governance_decisions=[],
            performance_metrics=performance_metrics,
            errors=errors,
            warnings=warnings,
            summary=summary
        )
        
        self.test_results.append(result)
        return result
    
    async def _run_production_simulation_test(self, test_id: str) -> E2ETestResult:
        """Run production simulation test"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            async with DGLClient(self.dgl_client_config) as dgl_client:
                # Create realistic production conditions
                orchestrator = MockOrchestrator(dgl_client, OrchestratorMode.BALANCED)
                
                # Generate realistic mixed scenarios
                mixed_proposals = []
                
                # 60% normal operations
                normal_config = ScenarioConfig(ScenarioType.NORMAL_OPERATIONS, num_proposals=60)
                normal_proposals = await self.scenario_generator.generate_scenario(normal_config)
                mixed_proposals.extend(normal_proposals)
                
                # 20% edge cases
                edge_config = ScenarioConfig(ScenarioType.EDGE_CASES, num_proposals=20)
                edge_proposals = await self.scenario_generator.generate_scenario(edge_config)
                mixed_proposals.extend(edge_proposals)
                
                # 10% market volatility
                volatility_config = ScenarioConfig(ScenarioType.MARKET_VOLATILITY, num_proposals=10)
                volatility_proposals = await self.scenario_generator.generate_scenario(volatility_config)
                mixed_proposals.extend(volatility_proposals)
                
                # 10% risk limits
                risk_config = ScenarioConfig(ScenarioType.RISK_LIMITS, num_proposals=10)
                risk_proposals = await self.scenario_generator.generate_scenario(risk_config)
                mixed_proposals.extend(risk_proposals)
                
                # Shuffle to simulate realistic order
                import random
                random.shuffle(mixed_proposals)
                
                # Process proposals with realistic timing
                all_decisions = []
                processing_times = []
                
                for i, proposal in enumerate(mixed_proposals):
                    try:
                        proposal_start = datetime.now()
                        decision = await dgl_client.evaluate_proposal(proposal)
                        proposal_end = datetime.now()
                        
                        all_decisions.append(decision)
                        processing_times.append((proposal_end - proposal_start).total_seconds() * 1000)
                        
                        # Simulate realistic inter-proposal delays
                        if i < len(mixed_proposals) - 1:
                            await asyncio.sleep(random.uniform(0.1, 2.0))  # 0.1-2 second delays
                        
                        if self.progress_callback and i % 10 == 0:
                            await self.progress_callback(i / len(mixed_proposals), i, {"decisions": len(all_decisions)})
                        
                    except Exception as e:
                        errors.append(f"Proposal {i} failed: {str(e)}")
                
                # Analyze production simulation results
                if all_decisions:
                    approval_rate = sum(1 for d in all_decisions if d.decision == DecisionType.APPROVE) / len(all_decisions) * 100
                    rejection_rate = sum(1 for d in all_decisions if d.decision == DecisionType.REJECT) / len(all_decisions) * 100
                    amendment_rate = sum(1 for d in all_decisions if d.decision == DecisionType.AMEND) / len(all_decisions) * 100
                    
                    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
                    
                    # Rule trigger analysis
                    rule_triggers = {}
                    for decision in all_decisions:
                        for rule_id in decision.rule_ids_triggered:
                            rule_triggers[rule_id.value] = rule_triggers.get(rule_id.value, 0) + 1
                    
                    performance_metrics = {
                        "total_proposals_processed": len(mixed_proposals),
                        "successful_evaluations": len(all_decisions),
                        "error_count": len(errors),
                        "success_rate_pct": (len(all_decisions) / len(mixed_proposals)) * 100,
                        "approval_rate_pct": approval_rate,
                        "rejection_rate_pct": rejection_rate,
                        "amendment_rate_pct": amendment_rate,
                        "avg_processing_time_ms": avg_processing_time,
                        "rule_trigger_frequency": rule_triggers,
                        "scenario_breakdown": {
                            "normal_operations": len(normal_proposals),
                            "edge_cases": len(edge_proposals),
                            "market_volatility": len(volatility_proposals),
                            "risk_limits": len(risk_proposals)
                        }
                    }
                else:
                    performance_metrics = {"error": "No successful evaluations"}
                
        except Exception as e:
            errors.append(f"Production simulation test failed: {str(e)}")
            mixed_proposals = []
            all_decisions = []
            performance_metrics = {}
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Create summary
        success_rate = performance_metrics.get("success_rate_pct", 0) if 'performance_metrics' in locals() else 0
        
        summary = {
            "simulation_status": "PRODUCTION_READY" if success_rate >= 95 else "NEEDS_TUNING" if success_rate >= 90 else "NOT_READY",
            "success_rate_pct": success_rate,
            "production_readiness": success_rate >= 95 and len(errors) == 0,
            "recommendation": f"Production simulation success rate: {success_rate:.1f}%"
        }
        
        result = E2ETestResult(
            test_id=test_id,
            test_type=E2ETestType.PRODUCTION_SIMULATION,
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            total_proposals=len(mixed_proposals) if 'mixed_proposals' in locals() else 0,
            successful_proposals=len(all_decisions) if 'all_decisions' in locals() else 0,
            failed_proposals=len(mixed_proposals) - len(all_decisions) if 'mixed_proposals' in locals() and 'all_decisions' in locals() else 0,
            governance_decisions=all_decisions if 'all_decisions' in locals() else [],
            performance_metrics=performance_metrics,
            errors=errors,
            warnings=warnings,
            summary=summary
        )
        
        self.test_results.append(result)
        return result
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all E2E test results"""
        if not self.test_results:
            return {"message": "No E2E tests have been run"}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if len(r.errors) == 0)
        
        return {
            "total_tests_run": total_tests,
            "tests_passed": passed_tests,
            "tests_failed": total_tests - passed_tests,
            "success_rate_pct": (passed_tests / total_tests) * 100,
            "test_types_covered": list(set(r.test_type.value for r in self.test_results)),
            "total_proposals_tested": sum(r.total_proposals for r in self.test_results),
            "total_test_duration_seconds": sum(r.duration_seconds for r in self.test_results),
            "last_test_timestamp": max(r.end_time for r in self.test_results).isoformat()
        }
    
    def export_results(self, file_path: str):
        """Export all test results to JSON file"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_tests": len(self.test_results),
            "test_summary": self.get_test_summary(),
            "detailed_results": []
        }
        
        for result in self.test_results:
            export_data["detailed_results"].append({
                "test_id": result.test_id,
                "test_type": result.test_type.value,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "duration_seconds": result.duration_seconds,
                "total_proposals": result.total_proposals,
                "successful_proposals": result.successful_proposals,
                "failed_proposals": result.failed_proposals,
                "performance_metrics": result.performance_metrics,
                "shadow_results": result.shadow_results,
                "scenario_results": result.scenario_results,
                "errors": result.errors,
                "warnings": result.warnings,
                "summary": result.summary
            })
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(self.test_results)} E2E test results to {file_path}")


# Convenience functions for quick E2E testing

async def run_quick_e2e_validation(dgl_base_url: str = "http://localhost:8001") -> E2ETestResult:
    """
    Run quick E2E validation test
    
    Args:
        dgl_base_url: DGL service URL
        
    Returns:
        E2E test results
    """
    config = E2ETestConfig(
        test_type=E2ETestType.BASIC_WORKFLOW,
        dgl_base_url=dgl_base_url,
        duration_minutes=2
    )
    
    tester = EndToEndTester(config)
    return await tester.run_test()


async def run_comprehensive_e2e_suite(
    dgl_base_url: str = "http://localhost:8001",
    duration_minutes: int = 10
) -> List[E2ETestResult]:
    """
    Run comprehensive E2E test suite
    
    Args:
        dgl_base_url: DGL service URL
        duration_minutes: Test duration for each test type
        
    Returns:
        List of E2E test results
    """
    test_types = [
        E2ETestType.BASIC_WORKFLOW,
        E2ETestType.COMPREHENSIVE_SCENARIOS,
        E2ETestType.PERFORMANCE_BENCHMARK,
        E2ETestType.SHADOW_VALIDATION
    ]
    
    results = []
    
    for test_type in test_types:
        config = E2ETestConfig(
            test_type=test_type,
            dgl_base_url=dgl_base_url,
            duration_minutes=duration_minutes
        )
        
        tester = EndToEndTester(config)
        result = await tester.run_test()
        results.append(result)
        
        logger.info(f"Completed {test_type.value}: {result.summary.get('recommendation', 'No summary')}")
    
    return results
