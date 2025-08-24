# Purpose: Performance benchmarking suite for DGL system
# Author: WicketWise AI, Last Modified: 2024

"""
Benchmark Suite

Comprehensive performance benchmarking for DGL:
- Component-level benchmarks
- End-to-end performance tests
- Regression testing
- Performance baselines
- Comparative analysis
"""

import logging
import asyncio
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import json
import uuid
import random

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import BetProposal
from engine import RuleEngine
from client.dgl_client import DGLClient
from observability.metrics_collector import MetricsCollector


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark"""
    benchmark_name: str
    component: str
    start_time: datetime
    end_time: datetime
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    cpu_usage_pct: float
    success_rate_pct: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "benchmark_name": self.benchmark_name,
            "component": self.component,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "iterations": self.iterations,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "p95_time_ms": self.p95_time_ms,
            "p99_time_ms": self.p99_time_ms,
            "throughput_ops_per_sec": self.throughput_ops_per_sec,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_pct": self.cpu_usage_pct,
            "success_rate_pct": self.success_rate_pct,
            "errors_count": len(self.errors),
            "metadata": self.metadata
        }


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    component: str
    benchmark_name: str
    baseline_date: datetime
    avg_time_ms: float
    p95_time_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    version: str = "unknown"
    environment: str = "test"
    
    def compare_with_result(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Compare baseline with benchmark result"""
        
        return {
            "avg_time_change_pct": ((result.avg_time_ms - self.avg_time_ms) / self.avg_time_ms) * 100,
            "p95_time_change_pct": ((result.p95_time_ms - self.p95_time_ms) / self.p95_time_ms) * 100,
            "throughput_change_pct": ((result.throughput_ops_per_sec - self.throughput_ops_per_sec) / self.throughput_ops_per_sec) * 100,
            "memory_change_pct": ((result.memory_usage_mb - self.memory_usage_mb) / self.memory_usage_mb) * 100,
            "performance_regression": result.avg_time_ms > self.avg_time_ms * 1.1,  # 10% slower
            "performance_improvement": result.avg_time_ms < self.avg_time_ms * 0.9   # 10% faster
        }


class BenchmarkSuite:
    """
    Comprehensive performance benchmarking suite
    
    Provides systematic performance testing across all DGL components
    with baseline comparison and regression detection.
    """
    
    def __init__(self, dgl_base_url: str = "http://localhost:8000"):
        """
        Initialize benchmark suite
        
        Args:
            dgl_base_url: Base URL for DGL service
        """
        self.dgl_base_url = dgl_base_url
        
        # Import DGLClientConfig here to avoid circular imports
        from client.dgl_client import DGLClientConfig
        config = DGLClientConfig(base_url=dgl_base_url)
        self.dgl_client = DGLClient(config=config)
        
        # Benchmark results storage
        self.results: List[BenchmarkResult] = []
        self.baselines: Dict[str, PerformanceBaseline] = {}
        
        # Test data
        self.test_proposals = self._generate_test_proposals()
        
        logger.info("Benchmark suite initialized")
    
    async def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all performance benchmarks"""
        
        logger.info("Starting comprehensive benchmark suite")
        
        benchmarks = [
            ("rule_engine_evaluation", self._benchmark_rule_engine),
            ("decision_processing", self._benchmark_decision_processing),
            ("api_endpoint_performance", self._benchmark_api_endpoints),
            ("concurrent_requests", self._benchmark_concurrent_requests),
            ("memory_usage", self._benchmark_memory_usage),
            ("database_operations", self._benchmark_database_operations),
            ("metrics_collection", self._benchmark_metrics_collection),
            ("governance_operations", self._benchmark_governance_operations)
        ]
        
        results = []
        
        for benchmark_name, benchmark_func in benchmarks:
            try:
                logger.info(f"Running benchmark: {benchmark_name}")
                result = await benchmark_func()
                results.append(result)
                self.results.append(result)
                
                logger.info(f"Benchmark completed: {benchmark_name}")
                logger.info(f"  Avg time: {result.avg_time_ms:.2f}ms")
                logger.info(f"  Throughput: {result.throughput_ops_per_sec:.1f} ops/sec")
                
            except Exception as e:
                logger.error(f"Benchmark failed: {benchmark_name} - {str(e)}")
                
                # Create error result
                error_result = BenchmarkResult(
                    benchmark_name=benchmark_name,
                    component="unknown",
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    iterations=0,
                    total_time_ms=0,
                    avg_time_ms=0,
                    min_time_ms=0,
                    max_time_ms=0,
                    p95_time_ms=0,
                    p99_time_ms=0,
                    throughput_ops_per_sec=0,
                    memory_usage_mb=0,
                    cpu_usage_pct=0,
                    success_rate_pct=0,
                    errors=[str(e)]
                )
                
                results.append(error_result)
                self.results.append(error_result)
        
        logger.info(f"Benchmark suite completed: {len(results)} benchmarks")
        
        return results
    
    async def _benchmark_rule_engine(self) -> BenchmarkResult:
        """Benchmark rule engine evaluation performance"""
        
        from config import load_config
        from repo.memory_repo import MemoryExposureStore, MemoryPnLStore, MemoryAuditStore
        
        # Setup rule engine
        config = load_config()
        exposure_store = MemoryExposureStore()
        pnl_store = MemoryPnLStore()
        audit_store = MemoryAuditStore()
        
        rule_engine = RuleEngine(config, exposure_store, pnl_store, audit_store)
        
        # Benchmark parameters
        iterations = 1000
        execution_times = []
        errors = []
        
        start_time = datetime.now()
        memory_before = self._get_memory_usage()
        
        for i in range(iterations):
            try:
                proposal = self.test_proposals[i % len(self.test_proposals)]
                
                exec_start = time.time()
                decision = rule_engine.evaluate_proposal(proposal)
                exec_time = (time.time() - exec_start) * 1000
                
                execution_times.append(exec_time)
                
            except Exception as e:
                errors.append(str(e))
        
        end_time = datetime.now()
        memory_after = self._get_memory_usage()
        
        # Calculate statistics
        total_time_ms = sum(execution_times)
        avg_time_ms = statistics.mean(execution_times) if execution_times else 0
        min_time_ms = min(execution_times) if execution_times else 0
        max_time_ms = max(execution_times) if execution_times else 0
        p95_time_ms = self._percentile(execution_times, 95)
        p99_time_ms = self._percentile(execution_times, 99)
        
        duration_seconds = (end_time - start_time).total_seconds()
        throughput = len(execution_times) / duration_seconds if duration_seconds > 0 else 0
        
        return BenchmarkResult(
            benchmark_name="rule_engine_evaluation",
            component="rule_engine",
            start_time=start_time,
            end_time=end_time,
            iterations=iterations,
            total_time_ms=total_time_ms,
            avg_time_ms=avg_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            p95_time_ms=p95_time_ms,
            p99_time_ms=p99_time_ms,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=memory_after - memory_before,
            cpu_usage_pct=0,  # Would measure with psutil in production
            success_rate_pct=(len(execution_times) / iterations) * 100,
            errors=errors
        )
    
    async def _benchmark_decision_processing(self) -> BenchmarkResult:
        """Benchmark end-to-end decision processing"""
        
        iterations = 500
        execution_times = []
        errors = []
        
        start_time = datetime.now()
        memory_before = self._get_memory_usage()
        
        for i in range(iterations):
            try:
                proposal = self.test_proposals[i % len(self.test_proposals)]
                
                exec_start = time.time()
                response = await self.dgl_client.evaluate_proposal(proposal)
                exec_time = (time.time() - exec_start) * 1000
                
                execution_times.append(exec_time)
                
            except Exception as e:
                errors.append(str(e))
        
        end_time = datetime.now()
        memory_after = self._get_memory_usage()
        
        # Calculate statistics
        total_time_ms = sum(execution_times)
        avg_time_ms = statistics.mean(execution_times) if execution_times else 0
        min_time_ms = min(execution_times) if execution_times else 0
        max_time_ms = max(execution_times) if execution_times else 0
        p95_time_ms = self._percentile(execution_times, 95)
        p99_time_ms = self._percentile(execution_times, 99)
        
        duration_seconds = (end_time - start_time).total_seconds()
        throughput = len(execution_times) / duration_seconds if duration_seconds > 0 else 0
        
        return BenchmarkResult(
            benchmark_name="decision_processing",
            component="dgl_service",
            start_time=start_time,
            end_time=end_time,
            iterations=iterations,
            total_time_ms=total_time_ms,
            avg_time_ms=avg_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            p95_time_ms=p95_time_ms,
            p99_time_ms=p99_time_ms,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=memory_after - memory_before,
            cpu_usage_pct=0,
            success_rate_pct=(len(execution_times) / iterations) * 100,
            errors=errors
        )
    
    async def _benchmark_api_endpoints(self) -> BenchmarkResult:
        """Benchmark API endpoint performance"""
        
        import aiohttp
        
        endpoints = [
            "/healthz",
            "/governance/status",
            "/exposure/current",
            "/rules/status"
        ]
        
        iterations_per_endpoint = 100
        total_iterations = len(endpoints) * iterations_per_endpoint
        execution_times = []
        errors = []
        
        start_time = datetime.now()
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                url = f"{self.dgl_base_url}{endpoint}"
                
                for _ in range(iterations_per_endpoint):
                    try:
                        exec_start = time.time()
                        async with session.get(url) as response:
                            await response.text()
                        exec_time = (time.time() - exec_start) * 1000
                        
                        execution_times.append(exec_time)
                        
                    except Exception as e:
                        errors.append(f"{endpoint}: {str(e)}")
        
        end_time = datetime.now()
        
        # Calculate statistics
        total_time_ms = sum(execution_times)
        avg_time_ms = statistics.mean(execution_times) if execution_times else 0
        min_time_ms = min(execution_times) if execution_times else 0
        max_time_ms = max(execution_times) if execution_times else 0
        p95_time_ms = self._percentile(execution_times, 95)
        p99_time_ms = self._percentile(execution_times, 99)
        
        duration_seconds = (end_time - start_time).total_seconds()
        throughput = len(execution_times) / duration_seconds if duration_seconds > 0 else 0
        
        return BenchmarkResult(
            benchmark_name="api_endpoint_performance",
            component="api_endpoints",
            start_time=start_time,
            end_time=end_time,
            iterations=total_iterations,
            total_time_ms=total_time_ms,
            avg_time_ms=avg_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            p95_time_ms=p95_time_ms,
            p99_time_ms=p99_time_ms,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=0,
            cpu_usage_pct=0,
            success_rate_pct=(len(execution_times) / total_iterations) * 100,
            errors=errors
        )
    
    async def _benchmark_concurrent_requests(self) -> BenchmarkResult:
        """Benchmark concurrent request handling"""
        
        concurrent_users = 20
        requests_per_user = 25
        total_iterations = concurrent_users * requests_per_user
        
        execution_times = []
        errors = []
        
        start_time = datetime.now()
        
        async def user_requests(user_id: int):
            user_times = []
            user_errors = []
            
            for i in range(requests_per_user):
                try:
                    proposal = self.test_proposals[(user_id * requests_per_user + i) % len(self.test_proposals)]
                    
                    exec_start = time.time()
                    response = await self.dgl_client.evaluate_proposal(proposal)
                    exec_time = (time.time() - exec_start) * 1000
                    
                    user_times.append(exec_time)
                    
                except Exception as e:
                    user_errors.append(f"User {user_id}: {str(e)}")
            
            return user_times, user_errors
        
        # Run concurrent users
        tasks = [user_requests(user_id) for user_id in range(concurrent_users)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for result in results:
            if isinstance(result, tuple):
                times, errs = result
                execution_times.extend(times)
                errors.extend(errs)
            else:
                errors.append(f"Task error: {str(result)}")
        
        end_time = datetime.now()
        
        # Calculate statistics
        total_time_ms = sum(execution_times)
        avg_time_ms = statistics.mean(execution_times) if execution_times else 0
        min_time_ms = min(execution_times) if execution_times else 0
        max_time_ms = max(execution_times) if execution_times else 0
        p95_time_ms = self._percentile(execution_times, 95)
        p99_time_ms = self._percentile(execution_times, 99)
        
        duration_seconds = (end_time - start_time).total_seconds()
        throughput = len(execution_times) / duration_seconds if duration_seconds > 0 else 0
        
        return BenchmarkResult(
            benchmark_name="concurrent_requests",
            component="dgl_service",
            start_time=start_time,
            end_time=end_time,
            iterations=total_iterations,
            total_time_ms=total_time_ms,
            avg_time_ms=avg_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            p95_time_ms=p95_time_ms,
            p99_time_ms=p99_time_ms,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=0,
            cpu_usage_pct=0,
            success_rate_pct=(len(execution_times) / total_iterations) * 100,
            errors=errors,
            metadata={"concurrent_users": concurrent_users, "requests_per_user": requests_per_user}
        )
    
    async def _benchmark_memory_usage(self) -> BenchmarkResult:
        """Benchmark memory usage patterns"""
        
        import gc
        
        iterations = 1000
        memory_measurements = []
        
        start_time = datetime.now()
        initial_memory = self._get_memory_usage()
        
        for i in range(iterations):
            # Create and process proposals
            proposal = self.test_proposals[i % len(self.test_proposals)]
            
            try:
                response = await self.dgl_client.evaluate_proposal(proposal)
            except Exception:
                pass  # Ignore errors for memory test
            
            # Measure memory every 100 iterations
            if i % 100 == 0:
                memory_measurements.append(self._get_memory_usage())
        
        # Force garbage collection
        gc.collect()
        final_memory = self._get_memory_usage()
        
        end_time = datetime.now()
        
        # Calculate memory statistics
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_measurements) if memory_measurements else final_memory
        avg_memory = statistics.mean(memory_measurements) if memory_measurements else final_memory
        
        duration_seconds = (end_time - start_time).total_seconds()
        throughput = iterations / duration_seconds if duration_seconds > 0 else 0
        
        return BenchmarkResult(
            benchmark_name="memory_usage",
            component="dgl_service",
            start_time=start_time,
            end_time=end_time,
            iterations=iterations,
            total_time_ms=duration_seconds * 1000,
            avg_time_ms=(duration_seconds * 1000) / iterations,
            min_time_ms=0,
            max_time_ms=0,
            p95_time_ms=0,
            p99_time_ms=0,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=memory_growth,
            cpu_usage_pct=0,
            success_rate_pct=100,
            metadata={
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "max_memory_mb": max_memory,
                "avg_memory_mb": avg_memory,
                "memory_growth_mb": memory_growth
            }
        )
    
    async def _benchmark_database_operations(self) -> BenchmarkResult:
        """Benchmark database/storage operations"""
        
        from repo.memory_repo import MemoryExposureStore, MemoryAuditStore
        
        # Setup stores
        exposure_store = MemoryExposureStore()
        audit_store = MemoryAuditStore()
        
        iterations = 2000
        execution_times = []
        errors = []
        
        start_time = datetime.now()
        
        for i in range(iterations):
            try:
                exec_start = time.time()
                
                # Simulate database operations
                if i % 3 == 0:
                    # Exposure operations
                    exposure_store.update_exposure(
                        match_id=f"match_{i % 10}",
                        market_id=f"market_{i % 5}",
                        correlation_group=f"group_{i % 3}",
                        exposure_delta=float(i % 100)
                    )
                    exposure_store.get_current_exposure()
                
                elif i % 3 == 1:
                    # Audit operations
                    audit_record = {
                        "event_type": "benchmark_test",
                        "user": "benchmark_user",
                        "resource": "test_resource",
                        "action": f"test_action_{i}",
                        "timestamp": datetime.now().isoformat()
                    }
                    audit_store.append_record(audit_record)
                    audit_store.get_recent_records(10)
                
                else:
                    # Mixed operations
                    exposure_store.get_match_exposure(f"match_{i % 10}")
                    audit_store.get_recent_records(5)
                
                exec_time = (time.time() - exec_start) * 1000
                execution_times.append(exec_time)
                
            except Exception as e:
                errors.append(str(e))
        
        end_time = datetime.now()
        
        # Calculate statistics
        total_time_ms = sum(execution_times)
        avg_time_ms = statistics.mean(execution_times) if execution_times else 0
        min_time_ms = min(execution_times) if execution_times else 0
        max_time_ms = max(execution_times) if execution_times else 0
        p95_time_ms = self._percentile(execution_times, 95)
        p99_time_ms = self._percentile(execution_times, 99)
        
        duration_seconds = (end_time - start_time).total_seconds()
        throughput = len(execution_times) / duration_seconds if duration_seconds > 0 else 0
        
        return BenchmarkResult(
            benchmark_name="database_operations",
            component="data_stores",
            start_time=start_time,
            end_time=end_time,
            iterations=iterations,
            total_time_ms=total_time_ms,
            avg_time_ms=avg_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            p95_time_ms=p95_time_ms,
            p99_time_ms=p99_time_ms,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=0,
            cpu_usage_pct=0,
            success_rate_pct=(len(execution_times) / iterations) * 100,
            errors=errors
        )
    
    async def _benchmark_metrics_collection(self) -> BenchmarkResult:
        """Benchmark metrics collection performance"""
        
        from observability.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        
        iterations = 5000
        execution_times = []
        
        start_time = datetime.now()
        
        for i in range(iterations):
            exec_start = time.time()
            
            # Various metrics operations
            collector.increment_counter(f"benchmark.counter.{i % 10}", 1.0, {"test": "benchmark"})
            collector.set_gauge(f"benchmark.gauge.{i % 5}", float(i % 100), {"test": "benchmark"})
            collector.record_histogram(f"benchmark.histogram.{i % 3}", float(i % 1000))
            collector.record_timer(f"benchmark.timer.{i % 7}", float(i % 500))
            
            exec_time = (time.time() - exec_start) * 1000
            execution_times.append(exec_time)
        
        end_time = datetime.now()
        
        # Calculate statistics
        total_time_ms = sum(execution_times)
        avg_time_ms = statistics.mean(execution_times)
        min_time_ms = min(execution_times)
        max_time_ms = max(execution_times)
        p95_time_ms = self._percentile(execution_times, 95)
        p99_time_ms = self._percentile(execution_times, 99)
        
        duration_seconds = (end_time - start_time).total_seconds()
        throughput = iterations / duration_seconds
        
        return BenchmarkResult(
            benchmark_name="metrics_collection",
            component="metrics_collector",
            start_time=start_time,
            end_time=end_time,
            iterations=iterations,
            total_time_ms=total_time_ms,
            avg_time_ms=avg_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            p95_time_ms=p95_time_ms,
            p99_time_ms=p99_time_ms,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=0,
            cpu_usage_pct=0,
            success_rate_pct=100,
            metadata={"metrics_types": 4, "unique_metric_names": 25}
        )
    
    async def _benchmark_governance_operations(self) -> BenchmarkResult:
        """Benchmark governance system operations"""
        
        from governance.state_machine import GovernanceStateMachine
        from governance.rbac import RBACManager
        from governance.audit import GovernanceAuditStore
        
        # Setup governance components
        audit_store = GovernanceAuditStore()
        state_machine = GovernanceStateMachine(audit_store)
        rbac_manager = RBACManager(audit_store)
        
        iterations = 1000
        execution_times = []
        errors = []
        
        start_time = datetime.now()
        
        for i in range(iterations):
            try:
                exec_start = time.time()
                
                # Various governance operations
                if i % 4 == 0:
                    # State machine operations
                    valid_transitions = state_machine.get_valid_transitions()
                    system_status = state_machine.get_system_status()
                
                elif i % 4 == 1:
                    # RBAC operations
                    user_permissions = rbac_manager.get_user_permissions("test_user")
                    has_permission = rbac_manager.has_permission("test_user", list(rbac_manager.roles["viewer"].permissions)[0])
                
                elif i % 4 == 2:
                    # Audit operations
                    audit_record = {
                        "event_type": "benchmark_governance",
                        "user": f"benchmark_user_{i}",
                        "resource": "governance_benchmark",
                        "action": f"benchmark_action_{i}",
                        "timestamp": datetime.now().isoformat()
                    }
                    audit_store.append_record(audit_record)
                
                else:
                    # Mixed operations
                    audit_store.get_recent_records(10)
                    rbac_manager.list_roles()
                
                exec_time = (time.time() - exec_start) * 1000
                execution_times.append(exec_time)
                
            except Exception as e:
                errors.append(str(e))
        
        end_time = datetime.now()
        
        # Calculate statistics
        total_time_ms = sum(execution_times)
        avg_time_ms = statistics.mean(execution_times) if execution_times else 0
        min_time_ms = min(execution_times) if execution_times else 0
        max_time_ms = max(execution_times) if execution_times else 0
        p95_time_ms = self._percentile(execution_times, 95)
        p99_time_ms = self._percentile(execution_times, 99)
        
        duration_seconds = (end_time - start_time).total_seconds()
        throughput = len(execution_times) / duration_seconds if duration_seconds > 0 else 0
        
        return BenchmarkResult(
            benchmark_name="governance_operations",
            component="governance_system",
            start_time=start_time,
            end_time=end_time,
            iterations=iterations,
            total_time_ms=total_time_ms,
            avg_time_ms=avg_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            p95_time_ms=p95_time_ms,
            p99_time_ms=p99_time_ms,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=0,
            cpu_usage_pct=0,
            success_rate_pct=(len(execution_times) / iterations) * 100,
            errors=errors
        )
    
    def _generate_test_proposals(self) -> List[BetProposal]:
        """Generate test bet proposals for benchmarking"""
        
        proposals = []
        
        from schemas import BetSide
        
        for i in range(100):
            proposal = BetProposal(
                proposal_id=f"benchmark_{uuid.uuid4().hex[:8]}",
                match_id=f"BENCH_MATCH_{i % 10}",
                market_id=f"benchmark_market_{i % 5}",
                side=BetSide.BACK,  # Default to BACK
                selection=f"Selection_{i % 3}",
                stake=round(random.uniform(10.0, 1000.0), 2),
                odds=round(random.uniform(1.5, 5.0), 2),
                model_confidence=random.uniform(0.7, 0.95),
                fair_odds=round(random.uniform(1.4, 5.2), 2),
                expected_edge_pct=random.uniform(1.0, 15.0),  # 1-15% edge
                correlation_group=f"benchmark_group_{i % 3}",
                metadata={"benchmark": True, "iteration": i}
            )
            proposals.append(proposal)
        
        return proposals
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        
        lower_index = int(index)
        upper_index = lower_index + 1
        
        if upper_index >= len(sorted_values):
            return sorted_values[-1]
        
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def set_baseline(self, result: BenchmarkResult, version: str = "current", environment: str = "test"):
        """Set performance baseline from benchmark result"""
        
        baseline = PerformanceBaseline(
            component=result.component,
            benchmark_name=result.benchmark_name,
            baseline_date=result.start_time,
            avg_time_ms=result.avg_time_ms,
            p95_time_ms=result.p95_time_ms,
            throughput_ops_per_sec=result.throughput_ops_per_sec,
            memory_usage_mb=result.memory_usage_mb,
            version=version,
            environment=environment
        )
        
        baseline_key = f"{result.component}_{result.benchmark_name}"
        self.baselines[baseline_key] = baseline
        
        logger.info(f"Baseline set for {baseline_key}")
    
    def compare_with_baseline(self, result: BenchmarkResult) -> Optional[Dict[str, Any]]:
        """Compare benchmark result with baseline"""
        
        baseline_key = f"{result.component}_{result.benchmark_name}"
        baseline = self.baselines.get(baseline_key)
        
        if not baseline:
            return None
        
        return baseline.compare_with_result(result)
    
    def export_results(self, filename: str):
        """Export benchmark results to JSON"""
        
        export_data = {
            "benchmark_suite_results": [result.to_dict() for result in self.results],
            "baselines": {
                key: {
                    "component": baseline.component,
                    "benchmark_name": baseline.benchmark_name,
                    "baseline_date": baseline.baseline_date.isoformat(),
                    "avg_time_ms": baseline.avg_time_ms,
                    "p95_time_ms": baseline.p95_time_ms,
                    "throughput_ops_per_sec": baseline.throughput_ops_per_sec,
                    "memory_usage_mb": baseline.memory_usage_mb,
                    "version": baseline.version,
                    "environment": baseline.environment
                }
                for key, baseline in self.baselines.items()
            },
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Benchmark results exported to {filename}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results"""
        
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Calculate overall statistics
        all_avg_times = [r.avg_time_ms for r in self.results if r.avg_time_ms > 0]
        all_throughputs = [r.throughput_ops_per_sec for r in self.results if r.throughput_ops_per_sec > 0]
        
        return {
            "total_benchmarks": len(self.results),
            "successful_benchmarks": len([r for r in self.results if r.success_rate_pct > 90]),
            "avg_response_time_ms": statistics.mean(all_avg_times) if all_avg_times else 0,
            "total_throughput_ops_per_sec": sum(all_throughputs),
            "components_tested": list(set(r.component for r in self.results)),
            "benchmark_names": [r.benchmark_name for r in self.results],
            "total_iterations": sum(r.iterations for r in self.results),
            "total_errors": sum(len(r.errors) for r in self.results)
        }


# Utility functions for benchmarking

def create_benchmark_suite(dgl_base_url: str = "http://localhost:8000") -> BenchmarkSuite:
    """Create and configure benchmark suite"""
    return BenchmarkSuite(dgl_base_url)


async def run_quick_benchmark(dgl_base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Run a quick benchmark for basic validation"""
    
    suite = create_benchmark_suite(dgl_base_url)
    
    # Run a subset of benchmarks
    result = await suite._benchmark_rule_engine()
    
    return {
        "benchmark": result.benchmark_name,
        "avg_time_ms": result.avg_time_ms,
        "throughput_ops_per_sec": result.throughput_ops_per_sec,
        "success_rate_pct": result.success_rate_pct,
        "iterations": result.iterations
    }
