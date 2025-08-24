# Purpose: Load generation for DGL performance testing
# Author: WicketWise AI, Last Modified: 2024

"""
Load Generator

Generates realistic load patterns for DGL testing:
- Configurable load scenarios
- Realistic bet proposal generation
- Variable load patterns (constant, ramp, spike, etc.)
- Concurrent request simulation
- Performance metrics collection during load
"""

import logging
import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
import uuid
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import BetProposal, DecisionType
from client.dgl_client import DGLClient


logger = logging.getLogger(__name__)


class LoadPattern(Enum):
    """Load generation patterns"""
    CONSTANT = "constant"         # Steady load
    RAMP_UP = "ramp_up"          # Gradually increasing load
    RAMP_DOWN = "ramp_down"      # Gradually decreasing load
    SPIKE = "spike"              # Sudden load spikes
    WAVE = "wave"                # Sinusoidal load pattern
    STEP = "step"                # Step increases in load
    RANDOM = "random"            # Random load variations


@dataclass
class LoadScenario:
    """Load testing scenario configuration"""
    name: str
    description: str
    pattern: LoadPattern
    duration_seconds: int
    base_rps: float  # Base requests per second
    peak_rps: float  # Peak requests per second
    concurrent_users: int = 10
    ramp_duration_seconds: int = 60
    think_time_ms: Tuple[int, int] = (100, 500)  # Min, max think time
    error_threshold_pct: float = 5.0
    response_time_p95_ms: float = 100.0
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadTestMetrics:
    """Metrics collected during load testing"""
    scenario_name: str
    start_time: datetime
    end_time: datetime
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    throughput_rps: List[float] = field(default_factory=list)
    
    @property
    def success_rate_pct(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def error_rate_pct(self) -> float:
        """Calculate error rate percentage"""
        return 100.0 - self.success_rate_pct
    
    @property
    def avg_response_time_ms(self) -> float:
        """Calculate average response time"""
        if not self.response_times_ms:
            return 0.0
        return sum(self.response_times_ms) / len(self.response_times_ms)
    
    @property
    def p95_response_time_ms(self) -> float:
        """Calculate 95th percentile response time"""
        if not self.response_times_ms:
            return 0.0
        
        sorted_times = sorted(self.response_times_ms)
        index = int(0.95 * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    @property
    def avg_throughput_rps(self) -> float:
        """Calculate average throughput"""
        if not self.throughput_rps:
            return 0.0
        return sum(self.throughput_rps) / len(self.throughput_rps)


class LoadGenerator:
    """
    Comprehensive load generator for DGL performance testing
    
    Generates realistic load patterns and collects detailed
    performance metrics for analysis and optimization.
    """
    
    def __init__(self, dgl_base_url: str = "http://localhost:8000"):
        """
        Initialize load generator
        
        Args:
            dgl_base_url: Base URL for DGL service
        """
        self.dgl_base_url = dgl_base_url
        
        # Import DGLClientConfig here to avoid circular imports
        from client.dgl_client import DGLClientConfig
        config = DGLClientConfig(base_url=dgl_base_url)
        self.dgl_client = DGLClient(config=config)
        
        # Load testing state
        self.active_scenarios: Dict[str, LoadTestMetrics] = {}
        self.completed_scenarios: List[LoadTestMetrics] = []
        
        # Request templates for realistic load
        self.bet_templates = self._create_bet_templates()
        
        logger.info(f"Load generator initialized for {dgl_base_url}")
    
    async def run_load_scenario(self, scenario: LoadScenario) -> LoadTestMetrics:
        """
        Execute a load testing scenario
        
        Args:
            scenario: Load scenario configuration
            
        Returns:
            Load test metrics
        """
        logger.info(f"Starting load scenario: {scenario.name}")
        
        # Initialize metrics
        metrics = LoadTestMetrics(
            scenario_name=scenario.name,
            start_time=datetime.now(),
            end_time=datetime.now()  # Will be updated at end
        )
        
        self.active_scenarios[scenario.name] = metrics
        
        try:
            # Create semaphore for concurrent users
            semaphore = asyncio.Semaphore(scenario.concurrent_users)
            
            # Generate load pattern
            load_schedule = self._generate_load_schedule(scenario)
            
            # Execute load test
            tasks = []
            
            for timestamp, rps in load_schedule:
                # Calculate requests for this time slice
                requests_this_slice = int(rps)
                
                # Create tasks for this time slice
                for _ in range(requests_this_slice):
                    task = asyncio.create_task(
                        self._execute_request(scenario, metrics, semaphore)
                    )
                    tasks.append(task)
                
                # Wait for time slice duration (1 second)
                await asyncio.sleep(1.0)
                
                # Update throughput metrics
                current_rps = len([t for t in tasks if t.done()]) / len(load_schedule)
                metrics.throughput_rps.append(current_rps)
            
            # Wait for all remaining tasks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Finalize metrics
            metrics.end_time = datetime.now()
            
            # Move to completed scenarios
            self.completed_scenarios.append(metrics)
            del self.active_scenarios[scenario.name]
            
            logger.info(f"Load scenario completed: {scenario.name}")
            logger.info(f"  Total requests: {metrics.total_requests}")
            logger.info(f"  Success rate: {metrics.success_rate_pct:.1f}%")
            logger.info(f"  Avg response time: {metrics.avg_response_time_ms:.1f}ms")
            logger.info(f"  P95 response time: {metrics.p95_response_time_ms:.1f}ms")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in load scenario {scenario.name}: {str(e)}")
            metrics.errors.append(f"Scenario error: {str(e)}")
            metrics.end_time = datetime.now()
            
            # Move to completed scenarios even on error
            if scenario.name in self.active_scenarios:
                self.completed_scenarios.append(metrics)
                del self.active_scenarios[scenario.name]
            
            return metrics
    
    async def run_multiple_scenarios(self, scenarios: List[LoadScenario]) -> List[LoadTestMetrics]:
        """
        Run multiple load scenarios sequentially
        
        Args:
            scenarios: List of load scenarios to execute
            
        Returns:
            List of load test metrics
        """
        results = []
        
        for scenario in scenarios:
            result = await self.run_load_scenario(scenario)
            results.append(result)
            
            # Brief pause between scenarios
            await asyncio.sleep(5)
        
        return results
    
    async def run_concurrent_scenarios(self, scenarios: List[LoadScenario]) -> List[LoadTestMetrics]:
        """
        Run multiple load scenarios concurrently
        
        Args:
            scenarios: List of load scenarios to execute concurrently
            
        Returns:
            List of load test metrics
        """
        tasks = [self.run_load_scenario(scenario) for scenario in scenarios]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        return [r for r in results if isinstance(r, LoadTestMetrics)]
    
    def create_realistic_scenarios(self) -> List[LoadScenario]:
        """Create realistic load testing scenarios"""
        
        scenarios = [
            # Normal business load
            LoadScenario(
                name="normal_business_load",
                description="Typical business hours load pattern",
                pattern=LoadPattern.CONSTANT,
                duration_seconds=300,  # 5 minutes
                base_rps=10.0,
                peak_rps=10.0,
                concurrent_users=5,
                response_time_p95_ms=50.0
            ),
            
            # Peak traffic simulation
            LoadScenario(
                name="peak_traffic",
                description="Peak traffic during major cricket events",
                pattern=LoadPattern.RAMP_UP,
                duration_seconds=600,  # 10 minutes
                base_rps=5.0,
                peak_rps=50.0,
                concurrent_users=20,
                ramp_duration_seconds=120,
                response_time_p95_ms=100.0
            ),
            
            # Spike load test
            LoadScenario(
                name="traffic_spike",
                description="Sudden traffic spike simulation",
                pattern=LoadPattern.SPIKE,
                duration_seconds=180,  # 3 minutes
                base_rps=10.0,
                peak_rps=100.0,
                concurrent_users=50,
                response_time_p95_ms=200.0,
                error_threshold_pct=10.0
            ),
            
            # Wave pattern (realistic daily pattern)
            LoadScenario(
                name="daily_wave_pattern",
                description="Daily traffic wave pattern",
                pattern=LoadPattern.WAVE,
                duration_seconds=900,  # 15 minutes
                base_rps=5.0,
                peak_rps=30.0,
                concurrent_users=15,
                response_time_p95_ms=75.0
            ),
            
            # Stress test scenario
            LoadScenario(
                name="stress_test",
                description="High load stress testing",
                pattern=LoadPattern.STEP,
                duration_seconds=300,  # 5 minutes
                base_rps=20.0,
                peak_rps=200.0,
                concurrent_users=100,
                response_time_p95_ms=500.0,
                error_threshold_pct=15.0
            )
        ]
        
        return scenarios
    
    def _generate_load_schedule(self, scenario: LoadScenario) -> List[Tuple[float, float]]:
        """Generate load schedule based on pattern"""
        
        schedule = []
        duration = scenario.duration_seconds
        base_rps = scenario.base_rps
        peak_rps = scenario.peak_rps
        
        for second in range(duration):
            progress = second / duration
            
            if scenario.pattern == LoadPattern.CONSTANT:
                rps = base_rps
                
            elif scenario.pattern == LoadPattern.RAMP_UP:
                rps = base_rps + (peak_rps - base_rps) * progress
                
            elif scenario.pattern == LoadPattern.RAMP_DOWN:
                rps = peak_rps - (peak_rps - base_rps) * progress
                
            elif scenario.pattern == LoadPattern.SPIKE:
                # Spike at 50% through the test
                if 0.4 <= progress <= 0.6:
                    rps = peak_rps
                else:
                    rps = base_rps
                    
            elif scenario.pattern == LoadPattern.WAVE:
                # Sinusoidal pattern
                import math
                rps = base_rps + (peak_rps - base_rps) * (math.sin(progress * 2 * math.pi) + 1) / 2
                
            elif scenario.pattern == LoadPattern.STEP:
                # Step increases every 25% of duration
                step = int(progress * 4)
                rps = base_rps + (peak_rps - base_rps) * (step / 4)
                
            elif scenario.pattern == LoadPattern.RANDOM:
                # Random variation around base
                variation = random.uniform(0.5, 1.5)
                rps = base_rps * variation
                rps = min(rps, peak_rps)
                
            else:
                rps = base_rps
            
            schedule.append((second, max(0, rps)))
        
        return schedule
    
    async def _execute_request(self, scenario: LoadScenario, 
                             metrics: LoadTestMetrics, 
                             semaphore: asyncio.Semaphore):
        """Execute a single request with metrics collection"""
        
        async with semaphore:
            start_time = time.time()
            
            try:
                # Add think time
                think_time = random.uniform(
                    scenario.think_time_ms[0] / 1000,
                    scenario.think_time_ms[1] / 1000
                )
                await asyncio.sleep(think_time)
                
                # Generate realistic bet proposal
                bet_proposal = self._generate_bet_proposal()
                
                # Make request to DGL
                response = await self.dgl_client.evaluate_proposal(bet_proposal)
                
                # Record successful request
                response_time = (time.time() - start_time) * 1000
                metrics.response_times_ms.append(response_time)
                metrics.successful_requests += 1
                metrics.total_requests += 1
                
            except Exception as e:
                # Record failed request
                response_time = (time.time() - start_time) * 1000
                metrics.response_times_ms.append(response_time)
                metrics.failed_requests += 1
                metrics.total_requests += 1
                metrics.errors.append(str(e))
                
                logger.debug(f"Request failed: {str(e)}")
    
    def _generate_bet_proposal(self) -> BetProposal:
        """Generate realistic bet proposal for load testing"""
        
        # Select random template
        template = random.choice(self.bet_templates)
        
        # Add some randomization
        stake = round(random.uniform(10.0, 1000.0), 2)
        odds = round(random.uniform(1.5, 5.0), 2)
        
        from schemas import BetSide
        
        return BetProposal(
            proposal_id=f"load_test_{uuid.uuid4().hex[:8]}",
            match_id=template["match_id"],
            market_id=template["market_id"],
            side=BetSide.BACK,  # Default to BACK
            selection=template["selection"],
            stake=stake,
            odds=odds,
            model_confidence=random.uniform(0.7, 0.95),
            fair_odds=odds * random.uniform(0.95, 1.05),
            expected_edge_pct=random.uniform(1.0, 15.0),  # 1-15% edge
            correlation_group=template.get("correlation_group"),
            metadata={
                "load_test": True,
                "scenario": "load_generation",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _create_bet_templates(self) -> List[Dict[str, Any]]:
        """Create realistic bet proposal templates"""
        
        templates = [
            {
                "match_id": "IND_vs_AUS_2024_T20_01",
                "market_id": "match_winner",
                "selection": "India",
                "correlation_group": "match_outcome"
            },
            {
                "match_id": "IND_vs_AUS_2024_T20_01", 
                "market_id": "total_runs",
                "selection": "Over 160.5",
                "correlation_group": "match_totals"
            },
            {
                "match_id": "ENG_vs_PAK_2024_ODI_05",
                "market_id": "top_batsman",
                "selection": "Babar Azam",
                "correlation_group": "individual_performance"
            },
            {
                "match_id": "ENG_vs_PAK_2024_ODI_05",
                "market_id": "method_of_dismissal",
                "selection": "Caught",
                "correlation_group": "dismissal_method"
            },
            {
                "match_id": "SA_vs_NZ_2024_TEST_02",
                "market_id": "session_runs",
                "selection": "40-49 runs",
                "correlation_group": "session_performance"
            }
        ]
        
        return templates
    
    def get_scenario_summary(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """Get summary of completed scenario"""
        
        scenario_metrics = next(
            (s for s in self.completed_scenarios if s.scenario_name == scenario_name),
            None
        )
        
        if not scenario_metrics:
            return None
        
        duration = (scenario_metrics.end_time - scenario_metrics.start_time).total_seconds()
        
        return {
            "scenario_name": scenario_metrics.scenario_name,
            "duration_seconds": duration,
            "total_requests": scenario_metrics.total_requests,
            "success_rate_pct": scenario_metrics.success_rate_pct,
            "error_rate_pct": scenario_metrics.error_rate_pct,
            "avg_response_time_ms": scenario_metrics.avg_response_time_ms,
            "p95_response_time_ms": scenario_metrics.p95_response_time_ms,
            "avg_throughput_rps": scenario_metrics.avg_throughput_rps,
            "errors_count": len(scenario_metrics.errors),
            "start_time": scenario_metrics.start_time.isoformat(),
            "end_time": scenario_metrics.end_time.isoformat()
        }
    
    def get_all_scenarios_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all completed scenarios"""
        
        return [
            self.get_scenario_summary(scenario.scenario_name)
            for scenario in self.completed_scenarios
        ]
    
    def export_results_csv(self, filename: str):
        """Export load test results to CSV"""
        
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'scenario_name', 'duration_seconds', 'total_requests',
                'success_rate_pct', 'error_rate_pct', 'avg_response_time_ms',
                'p95_response_time_ms', 'avg_throughput_rps', 'errors_count'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for scenario in self.completed_scenarios:
                summary = self.get_scenario_summary(scenario.scenario_name)
                if summary:
                    writer.writerow({k: v for k, v in summary.items() if k in fieldnames})
        
        logger.info(f"Load test results exported to {filename}")


# Utility functions for load testing

def create_load_generator(dgl_base_url: str = "http://localhost:8000") -> LoadGenerator:
    """Create and configure load generator"""
    return LoadGenerator(dgl_base_url)


async def run_quick_load_test(dgl_base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Run a quick load test for basic validation"""
    
    generator = create_load_generator(dgl_base_url)
    
    # Create a simple scenario
    quick_scenario = LoadScenario(
        name="quick_validation",
        description="Quick load test for validation",
        pattern=LoadPattern.CONSTANT,
        duration_seconds=30,
        base_rps=5.0,
        peak_rps=5.0,
        concurrent_users=3
    )
    
    # Run the test
    result = await generator.run_load_scenario(quick_scenario)
    
    return {
        "scenario": quick_scenario.name,
        "success": result.success_rate_pct > 90,
        "total_requests": result.total_requests,
        "success_rate_pct": result.success_rate_pct,
        "avg_response_time_ms": result.avg_response_time_ms
    }
