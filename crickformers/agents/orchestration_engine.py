# Purpose: Agent Orchestration Engine - Multi-Agent Coordination
# Author: WicketWise Team, Last Modified: 2025-08-24

"""
Agent Orchestration Engine
==========================

Coordinates multiple specialized agents to handle complex cricket analysis
queries. Provides intelligent routing, parallel execution, and result
aggregation capabilities.

Key Components:
- OrchestrationEngine: Main coordinator
- AgentCoordinator: Agent lifecycle management
- ExecutionPlan: Query decomposition and planning
- ResultAggregator: Multi-agent result synthesis
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from .base_agent import BaseAgent, AgentCapability, AgentContext, AgentResponse, AgentPriority

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Agent execution strategies"""
    SEQUENTIAL = "sequential"      # Execute agents one by one
    PARALLEL = "parallel"         # Execute all agents simultaneously
    PIPELINE = "pipeline"         # Chain agent outputs as inputs
    CONDITIONAL = "conditional"   # Execute based on conditions
    HYBRID = "hybrid"            # Mix of strategies based on query


@dataclass
class AgentTask:
    """Individual agent task within an execution plan"""
    agent_id: str
    capability: AgentCapability
    priority: AgentPriority
    context: AgentContext
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class ExecutionPlan:
    """Execution plan for handling a complex query"""
    plan_id: str
    original_query: str
    strategy: ExecutionStrategy
    tasks: List[AgentTask]
    estimated_duration: float
    confidence_threshold: float
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_critical_tasks(self) -> List[AgentTask]:
        """Get tasks with critical priority"""
        return [task for task in self.tasks if task.priority == AgentPriority.CRITICAL]
    
    def get_parallel_tasks(self) -> List[AgentTask]:
        """Get tasks that can be executed in parallel"""
        return [task for task in self.tasks if not task.dependencies]


@dataclass
class OrchestrationResult:
    """Result from orchestrated agent execution"""
    plan_id: str
    success: bool
    overall_confidence: float
    execution_time: float
    agent_responses: List[AgentResponse]
    aggregated_result: Optional[Dict[str, Any]] = None
    failed_tasks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentCoordinator:
    """Manages agent lifecycle and capabilities"""
    
    def __init__(self):
        self.registered_agents: Dict[str, BaseAgent] = {}
        self.capability_map: Dict[AgentCapability, List[str]] = {}
        self.agent_health: Dict[str, Dict[str, Any]] = {}
        
    def register_agent(self, agent: BaseAgent) -> bool:
        """Register an agent with the coordinator"""
        try:
            # Initialize agent
            if not agent.initialize():
                logger.error(f"Failed to initialize agent {agent.agent_id}")
                return False
            
            # Register agent
            self.registered_agents[agent.agent_id] = agent
            
            # Update capability mapping
            for capability in agent.capabilities:
                if capability not in self.capability_map:
                    self.capability_map[capability] = []
                self.capability_map[capability].append(agent.agent_id)
            
            # Initialize health tracking
            self.agent_health[agent.agent_id] = {
                "status": "healthy",
                "last_check": datetime.now(),
                "consecutive_failures": 0
            }
            
            logger.info(f"Successfully registered agent {agent.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent.agent_id}: {str(e)}")
            return False
    
    def get_agents_for_capability(self, capability: AgentCapability) -> List[BaseAgent]:
        """Get all healthy agents that support a capability"""
        agent_ids = self.capability_map.get(capability, [])
        healthy_agents = []
        
        for agent_id in agent_ids:
            if (agent_id in self.registered_agents and 
                self.agent_health[agent_id]["status"] == "healthy"):
                healthy_agents.append(self.registered_agents[agent_id])
        
        return healthy_agents
    
    def select_best_agent(self, capability: AgentCapability, context: AgentContext) -> Optional[BaseAgent]:
        """Select the best agent for a capability based on performance and context"""
        candidates = self.get_agents_for_capability(capability)
        
        if not candidates:
            return None
        
        # Filter by context compatibility
        compatible_agents = [
            agent for agent in candidates 
            if agent.can_handle(capability, context)
        ]
        
        if not compatible_agents:
            return None
        
        # Select based on performance metrics
        best_agent = max(compatible_agents, key=lambda a: (
            a.success_count / max(a.execution_count, 1),  # Success rate
            a.average_confidence,                          # Average confidence
            -a.total_execution_time / max(a.execution_count, 1)  # Speed (negative for min)
        ))
        
        return best_agent
    
    def update_agent_health(self, agent_id: str, success: bool):
        """Update agent health status based on execution result"""
        if agent_id not in self.agent_health:
            return
        
        health = self.agent_health[agent_id]
        health["last_check"] = datetime.now()
        
        if success:
            health["consecutive_failures"] = 0
            health["status"] = "healthy"
        else:
            health["consecutive_failures"] += 1
            
            # Mark as unhealthy after 3 consecutive failures
            if health["consecutive_failures"] >= 3:
                health["status"] = "unhealthy"
                logger.warning(f"Agent {agent_id} marked as unhealthy")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        total_agents = len(self.registered_agents)
        healthy_agents = sum(1 for h in self.agent_health.values() if h["status"] == "healthy")
        
        return {
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "health_percentage": (healthy_agents / max(total_agents, 1)) * 100,
            "agent_details": dict(self.agent_health)
        }


class OrchestrationEngine:
    """
    Main orchestration engine for coordinating multiple agents
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.coordinator = AgentCoordinator()
        
        # Execution tracking
        self.execution_history: List[OrchestrationResult] = []
        self.active_executions: Dict[str, ExecutionPlan] = {}
        
        # Configuration
        self.max_parallel_agents = self.config.get("max_parallel_agents", 5)
        self.default_timeout = self.config.get("default_timeout", 60.0)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
        logger.info("Initialized OrchestrationEngine")
    
    def register_agent(self, agent: BaseAgent) -> bool:
        """Register an agent with the orchestration engine"""
        return self.coordinator.register_agent(agent)
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> OrchestrationResult:
        """
        Process a complex query using multiple coordinated agents
        
        Args:
            query: Natural language query
            context: Optional context parameters
            
        Returns:
            OrchestrationResult with aggregated responses
        """
        start_time = time.time()
        plan_id = str(uuid.uuid4())
        
        try:
            # Create execution context
            agent_context = AgentContext(
                request_id=plan_id,
                user_query=query,
                timestamp=datetime.now(),
                confidence_threshold=self.confidence_threshold,
                max_execution_time=self.default_timeout,
                metadata=context or {}
            )
            
            # Generate execution plan
            execution_plan = self._create_execution_plan(query, agent_context)
            if not execution_plan:
                return self._create_error_result(
                    plan_id, "Failed to create execution plan", start_time
                )
            
            self.active_executions[plan_id] = execution_plan
            
            # Execute plan
            result = await self._execute_plan(execution_plan)
            
            # Clean up
            self.active_executions.pop(plan_id, None)
            self.execution_history.append(result)
            
            # Limit history size
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return self._create_error_result(
                plan_id, f"Processing error: {str(e)}", start_time
            )
    
    def _create_execution_plan(self, query: str, context: AgentContext) -> Optional[ExecutionPlan]:
        """Create execution plan based on query analysis"""
        try:
            # Analyze query to determine required capabilities
            required_capabilities = self._analyze_query_capabilities(query)
            
            if not required_capabilities:
                logger.warning(f"No capabilities identified for query: {query}")
                return None
            
            # Create tasks for each capability
            tasks = []
            for capability, priority in required_capabilities:
                agent = self.coordinator.select_best_agent(capability, context)
                if agent:
                    task = AgentTask(
                        agent_id=agent.agent_id,
                        capability=capability,
                        priority=priority,
                        context=context,
                        timeout=self.default_timeout
                    )
                    tasks.append(task)
                else:
                    logger.warning(f"No agent available for capability: {capability}")
            
            if not tasks:
                return None
            
            # Determine execution strategy
            strategy = self._determine_execution_strategy(tasks)
            
            # Estimate duration
            estimated_duration = self._estimate_execution_duration(tasks, strategy)
            
            return ExecutionPlan(
                plan_id=str(uuid.uuid4()),
                original_query=query,
                strategy=strategy,
                tasks=tasks,
                estimated_duration=estimated_duration,
                confidence_threshold=self.confidence_threshold
            )
            
        except Exception as e:
            logger.error(f"Failed to create execution plan: {str(e)}")
            return None
    
    def _analyze_query_capabilities(self, query: str) -> List[Tuple[AgentCapability, AgentPriority]]:
        """Analyze query to determine required agent capabilities"""
        query_lower = query.lower()
        capabilities = []
        
        # Performance analysis keywords
        performance_keywords = ["performance", "stats", "statistics", "runs", "wickets", "average", "strike rate"]
        if any(keyword in query_lower for keyword in performance_keywords):
            capabilities.append((AgentCapability.PERFORMANCE_ANALYSIS, AgentPriority.HIGH))
        
        # Tactical analysis keywords
        tactical_keywords = ["strategy", "tactics", "field", "bowling", "batting", "plan"]
        if any(keyword in query_lower for keyword in tactical_keywords):
            capabilities.append((AgentCapability.TACTICAL_ANALYSIS, AgentPriority.MEDIUM))
        
        # Prediction keywords
        prediction_keywords = ["predict", "forecast", "outcome", "winner", "result", "probability"]
        if any(keyword in query_lower for keyword in prediction_keywords):
            capabilities.append((AgentCapability.MATCH_PREDICTION, AgentPriority.HIGH))
        
        # Betting analysis keywords
        betting_keywords = ["bet", "odds", "value", "arbitrage", "bookmaker", "stake"]
        if any(keyword in query_lower for keyword in betting_keywords):
            capabilities.append((AgentCapability.BETTING_ANALYSIS, AgentPriority.MEDIUM))
        
        # Comparison keywords
        comparison_keywords = ["compare", "versus", "vs", "better", "best", "head to head"]
        if any(keyword in query_lower for keyword in comparison_keywords):
            capabilities.append((AgentCapability.COMPARISON_ANALYSIS, AgentPriority.MEDIUM))
        
        # Default to performance analysis if no specific keywords found
        if not capabilities:
            capabilities.append((AgentCapability.PERFORMANCE_ANALYSIS, AgentPriority.MEDIUM))
        
        return capabilities
    
    def _determine_execution_strategy(self, tasks: List[AgentTask]) -> ExecutionStrategy:
        """Determine optimal execution strategy for tasks"""
        # Simple heuristic: parallel if no dependencies, sequential otherwise
        has_dependencies = any(task.dependencies for task in tasks)
        has_critical_tasks = any(task.priority == AgentPriority.CRITICAL for task in tasks)
        
        if has_dependencies:
            return ExecutionStrategy.PIPELINE
        elif has_critical_tasks and len(tasks) <= self.max_parallel_agents:
            return ExecutionStrategy.PARALLEL
        else:
            return ExecutionStrategy.SEQUENTIAL
    
    def _estimate_execution_duration(self, tasks: List[AgentTask], strategy: ExecutionStrategy) -> float:
        """Estimate total execution duration"""
        if strategy == ExecutionStrategy.PARALLEL:
            return max(task.timeout for task in tasks)
        else:
            return sum(task.timeout for task in tasks)
    
    async def _execute_plan(self, plan: ExecutionPlan) -> OrchestrationResult:
        """Execute the orchestration plan"""
        start_time = time.time()
        agent_responses = []
        failed_tasks = []
        
        try:
            if plan.strategy == ExecutionStrategy.PARALLEL:
                responses = await self._execute_parallel(plan.tasks)
            elif plan.strategy == ExecutionStrategy.SEQUENTIAL:
                responses = await self._execute_sequential(plan.tasks)
            else:
                responses = await self._execute_sequential(plan.tasks)  # Fallback
            
            agent_responses.extend(responses)
            
            # Identify failed tasks
            failed_tasks = [
                resp.agent_id for resp in agent_responses if not resp.success
            ]
            
            # Update agent health
            for response in agent_responses:
                self.coordinator.update_agent_health(response.agent_id, response.success)
            
            # Aggregate results
            aggregated_result = self._aggregate_results(agent_responses)
            overall_confidence = self._calculate_overall_confidence(agent_responses)
            
            success = len(failed_tasks) == 0 or self._has_sufficient_success(agent_responses, plan)
            
            return OrchestrationResult(
                plan_id=plan.plan_id,
                success=success,
                overall_confidence=overall_confidence,
                execution_time=time.time() - start_time,
                agent_responses=agent_responses,
                aggregated_result=aggregated_result,
                failed_tasks=failed_tasks,
                metadata={
                    "strategy": plan.strategy.value,
                    "task_count": len(plan.tasks),
                    "estimated_duration": plan.estimated_duration
                }
            )
            
        except Exception as e:
            logger.error(f"Plan execution failed: {str(e)}")
            return self._create_error_result(
                plan.plan_id, f"Execution error: {str(e)}", start_time
            )
    
    async def _execute_parallel(self, tasks: List[AgentTask]) -> List[AgentResponse]:
        """Execute tasks in parallel"""
        coroutines = []
        
        for task in tasks:
            agent = self.coordinator.registered_agents.get(task.agent_id)
            if agent:
                coroutine = agent._execute_with_monitoring(task.context)
                coroutines.append(coroutine)
        
        if coroutines:
            responses = await asyncio.gather(*coroutines, return_exceptions=True)
            
            # Handle exceptions
            valid_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Task {i} failed with exception: {response}")
                    # Create error response
                    task = tasks[i]
                    error_response = AgentResponse(
                        agent_id=task.agent_id,
                        capability=task.capability,
                        success=False,
                        confidence=0.0,
                        execution_time=0.0,
                        error_message=str(response)
                    )
                    valid_responses.append(error_response)
                else:
                    valid_responses.append(response)
            
            return valid_responses
        
        return []
    
    async def _execute_sequential(self, tasks: List[AgentTask]) -> List[AgentResponse]:
        """Execute tasks sequentially"""
        responses = []
        
        for task in tasks:
            agent = self.coordinator.registered_agents.get(task.agent_id)
            if agent:
                try:
                    response = await agent._execute_with_monitoring(task.context)
                    responses.append(response)
                    
                    # Stop on critical task failure
                    if task.priority == AgentPriority.CRITICAL and not response.success:
                        logger.error(f"Critical task failed: {task.agent_id}")
                        break
                        
                except Exception as e:
                    logger.error(f"Task execution failed: {str(e)}")
                    error_response = AgentResponse(
                        agent_id=task.agent_id,
                        capability=task.capability,
                        success=False,
                        confidence=0.0,
                        execution_time=0.0,
                        error_message=str(e)
                    )
                    responses.append(error_response)
        
        return responses
    
    def _aggregate_results(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """Aggregate results from multiple agent responses"""
        successful_responses = [r for r in responses if r.success]
        
        if not successful_responses:
            return {"error": "No successful agent responses"}
        
        aggregated = {
            "agent_count": len(responses),
            "successful_agents": len(successful_responses),
            "capabilities_used": list(set(r.capability.value for r in successful_responses)),
            "results": {}
        }
        
        # Aggregate results by capability
        for response in successful_responses:
            capability = response.capability.value
            if capability not in aggregated["results"]:
                aggregated["results"][capability] = []
            
            aggregated["results"][capability].append({
                "agent_id": response.agent_id,
                "confidence": response.confidence,
                "result": response.result,
                "execution_time": response.execution_time
            })
        
        return aggregated
    
    def _calculate_overall_confidence(self, responses: List[AgentResponse]) -> float:
        """Calculate overall confidence from agent responses"""
        successful_responses = [r for r in responses if r.success]
        
        if not successful_responses:
            return 0.0
        
        # Weighted average by execution success
        total_confidence = sum(r.confidence for r in successful_responses)
        return total_confidence / len(successful_responses)
    
    def _has_sufficient_success(self, responses: List[AgentResponse], plan: ExecutionPlan) -> bool:
        """Check if we have sufficient successful responses"""
        critical_tasks = plan.get_critical_tasks()
        
        # All critical tasks must succeed
        for task in critical_tasks:
            task_responses = [r for r in responses if r.agent_id == task.agent_id]
            if not task_responses or not any(r.success for r in task_responses):
                return False
        
        # At least 50% of non-critical tasks should succeed
        non_critical_responses = [
            r for r in responses 
            if not any(t.agent_id == r.agent_id and t.priority == AgentPriority.CRITICAL 
                      for t in plan.tasks)
        ]
        
        if non_critical_responses:
            success_rate = sum(1 for r in non_critical_responses if r.success) / len(non_critical_responses)
            return success_rate >= 0.5
        
        return True
    
    def _create_error_result(self, plan_id: str, error_message: str, start_time: float) -> OrchestrationResult:
        """Create error result"""
        return OrchestrationResult(
            plan_id=plan_id,
            success=False,
            overall_confidence=0.0,
            execution_time=time.time() - start_time,
            agent_responses=[],
            failed_tasks=["orchestration_engine"],
            metadata={"error": error_message}
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        health = self.coordinator.get_system_health()
        
        return {
            "orchestration_engine": {
                "active_executions": len(self.active_executions),
                "execution_history_size": len(self.execution_history),
                "max_parallel_agents": self.max_parallel_agents,
                "default_timeout": self.default_timeout
            },
            "agent_system": health,
            "recent_performance": self._get_recent_performance_stats()
        }
    
    def _get_recent_performance_stats(self) -> Dict[str, Any]:
        """Get recent performance statistics"""
        if not self.execution_history:
            return {"no_data": True}
        
        recent = self.execution_history[-10:]  # Last 10 executions
        
        return {
            "total_executions": len(recent),
            "success_rate": sum(1 for r in recent if r.success) / len(recent),
            "average_confidence": sum(r.overall_confidence for r in recent) / len(recent),
            "average_execution_time": sum(r.execution_time for r in recent) / len(recent),
            "most_used_capabilities": self._get_capability_usage(recent)
        }
    
    def _get_capability_usage(self, results: List[OrchestrationResult]) -> Dict[str, int]:
        """Get capability usage statistics"""
        usage = {}
        
        for result in results:
            for response in result.agent_responses:
                if response.success:
                    cap = response.capability.value
                    usage[cap] = usage.get(cap, 0) + 1
        
        return dict(sorted(usage.items(), key=lambda x: x[1], reverse=True))
