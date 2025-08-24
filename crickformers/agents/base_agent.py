# Purpose: Base Agent Protocol and Abstract Classes
# Author: WicketWise Team, Last Modified: 2025-08-24

"""
Base Agent Architecture
=======================

Defines the foundational protocols and abstract classes for the LLM agent
orchestration system. All specialized agents inherit from these base classes
to ensure consistent interfaces and behavior.

Key Components:
- AgentCapability: Enum of agent capabilities
- AgentResponse: Standardized response format
- BaseAgent: Abstract base class for all agents
- AgentContext: Execution context and state management
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """Agent capability types"""
    PERFORMANCE_ANALYSIS = "performance_analysis"
    TACTICAL_ANALYSIS = "tactical_analysis" 
    MATCH_PREDICTION = "match_prediction"
    BETTING_ANALYSIS = "betting_analysis"
    DATA_RETRIEVAL = "data_retrieval"
    TREND_ANALYSIS = "trend_analysis"
    COMPARISON_ANALYSIS = "comparison_analysis"
    CONTEXTUAL_REASONING = "contextual_reasoning"
    MULTI_FORMAT_ANALYSIS = "multi_format_analysis"
    REAL_TIME_PROCESSING = "real_time_processing"


class AgentPriority(Enum):
    """Agent execution priority levels"""
    CRITICAL = 1    # Must complete successfully
    HIGH = 2        # Important but can fail gracefully  
    MEDIUM = 3      # Standard priority
    LOW = 4         # Background/optional tasks


@dataclass
class AgentContext:
    """Execution context for agent operations"""
    request_id: str
    user_query: str
    timestamp: datetime
    match_context: Optional[Dict[str, Any]] = None
    player_context: Optional[Dict[str, Any]] = None
    team_context: Optional[Dict[str, Any]] = None
    format_context: Optional[str] = None
    temporal_context: Optional[Dict[str, Any]] = None
    confidence_threshold: float = 0.7
    max_execution_time: float = 30.0  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class AgentResponse:
    """Standardized agent response format"""
    agent_id: str
    capability: AgentCapability
    success: bool
    confidence: float
    execution_time: float
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    dependencies_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "agent_id": self.agent_id,
            "capability": self.capability.value,
            "success": self.success,
            "confidence": self.confidence,
            "execution_time": self.execution_time,
            "result": self.result,
            "error_message": self.error_message,
            "intermediate_steps": self.intermediate_steps,
            "dependencies_used": self.dependencies_used,
            "metadata": self.metadata
        }


class BaseAgent(ABC):
    """
    Abstract base class for all LLM agents in the orchestration system
    
    Provides common functionality for:
    - Agent identification and capabilities
    - Execution context management
    - Performance monitoring
    - Error handling and recovery
    - Dependency management
    """
    
    def __init__(self, agent_id: str, capabilities: List[AgentCapability], config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.config = config or {}
        
        # Performance tracking
        self.execution_count = 0
        self.success_count = 0
        self.total_execution_time = 0.0
        self.average_confidence = 0.0
        
        # Dependencies
        self.required_dependencies: List[str] = []
        self.optional_dependencies: List[str] = []
        
        # State management
        self.is_initialized = False
        self.last_execution_time = None
        
        # Configuration
        self.max_execution_time = self.config.get("max_execution_time", 30.0)
        self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.5)
        self.retry_attempts = self.config.get("retry_attempts", 2)
        
        logger.info(f"Initialized agent {self.agent_id} with capabilities: {[c.value for c in capabilities]}")
    
    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResponse:
        """
        Execute the agent's primary functionality
        
        Args:
            context: Execution context with query and parameters
            
        Returns:
            AgentResponse with results and metadata
        """
        pass
    
    @abstractmethod
    def can_handle(self, capability: AgentCapability, context: AgentContext) -> bool:
        """
        Check if agent can handle the given capability and context
        
        Args:
            capability: Required capability
            context: Execution context
            
        Returns:
            True if agent can handle the request
        """
        pass
    
    def initialize(self) -> bool:
        """
        Initialize agent dependencies and resources
        
        Returns:
            True if initialization successful
        """
        try:
            # Check required dependencies
            for dep in self.required_dependencies:
                if not self._check_dependency(dep):
                    logger.error(f"Agent {self.agent_id}: Required dependency {dep} not available")
                    return False
            
            # Perform agent-specific initialization
            if not self._initialize_agent():
                return False
            
            self.is_initialized = True
            logger.info(f"Agent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} initialization failed: {str(e)}")
            return False
    
    def _initialize_agent(self) -> bool:
        """Agent-specific initialization logic - override in subclasses"""
        return True
    
    def _check_dependency(self, dependency: str) -> bool:
        """Check if a dependency is available - override in subclasses"""
        return True
    
    async def _execute_with_monitoring(self, context: AgentContext) -> AgentResponse:
        """
        Execute agent with performance monitoring and error handling
        """
        start_time = time.time()
        self.execution_count += 1
        self.last_execution_time = datetime.now()
        
        try:
            # Validate context
            if not self._validate_context(context):
                return self._create_error_response(
                    context, "Invalid execution context", start_time
                )
            
            # Check if initialized
            if not self.is_initialized:
                if not self.initialize():
                    return self._create_error_response(
                        context, "Agent initialization failed", start_time
                    )
            
            # Execute with timeout
            response = await self.execute(context)
            
            # Update performance metrics
            execution_time = time.time() - start_time
            if response.success:
                self.success_count += 1
                self._update_confidence_tracking(response.confidence)
            
            self.total_execution_time += execution_time
            response.execution_time = execution_time
            
            return response
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} execution failed: {str(e)}")
            return self._create_error_response(
                context, f"Execution error: {str(e)}", start_time
            )
    
    def _validate_context(self, context: AgentContext) -> bool:
        """Validate execution context"""
        if not context.user_query:
            return False
        if context.max_execution_time <= 0:
            return False
        return True
    
    def _create_error_response(self, context: AgentContext, error_message: str, start_time: float) -> AgentResponse:
        """Create standardized error response"""
        return AgentResponse(
            agent_id=self.agent_id,
            capability=self.capabilities[0] if self.capabilities else AgentCapability.DATA_RETRIEVAL,
            success=False,
            confidence=0.0,
            execution_time=time.time() - start_time,
            error_message=error_message,
            metadata={
                "execution_count": self.execution_count,
                "context_request_id": context.request_id
            }
        )
    
    def _update_confidence_tracking(self, confidence: float):
        """Update rolling average confidence"""
        if self.success_count == 1:
            self.average_confidence = confidence
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_confidence = alpha * confidence + (1 - alpha) * self.average_confidence
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        success_rate = self.success_count / max(self.execution_count, 1)
        avg_execution_time = self.total_execution_time / max(self.execution_count, 1)
        
        return {
            "agent_id": self.agent_id,
            "capabilities": [c.value for c in self.capabilities],
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "average_confidence": self.average_confidence,
            "last_execution": self.last_execution_time.isoformat() if self.last_execution_time else None,
            "is_initialized": self.is_initialized
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking metrics"""
        self.execution_count = 0
        self.success_count = 0
        self.total_execution_time = 0.0
        self.average_confidence = 0.0
        self.last_execution_time = None
        
        logger.info(f"Reset performance stats for agent {self.agent_id}")
    
    def __str__(self) -> str:
        return f"Agent({self.agent_id}, capabilities={len(self.capabilities)}, initialized={self.is_initialized})"
    
    def __repr__(self) -> str:
        return f"BaseAgent(agent_id='{self.agent_id}', capabilities={self.capabilities})"
